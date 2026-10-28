"""The structured triage graph.

    extract ──▶ assess ──▶ reply
       │           └──▶ record ──▶ reply   (on finish)
       ├──▶ fail ──▶ reply                 (extraction failed)
       └──▶ reply                          (model unreachable: the fail-safe)

Each node does one thing, and the routing lives in the graph, so the failure path and the
finished record show up as nodes in LangGraph Studio and in traces. Dependencies are passed
in, so tests run the whole graph with a scripted extractor.
``make_engine``, ``make_extractor`` and ``make_graph`` are the composition root: the only
place the real clinical content and model are wired together.
"""

import logging
from collections.abc import Callable
from datetime import datetime
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer, Command

from pet_triage.clinical.complaints import COMPLAINTS
from pet_triage.clinical.pathways import PATHWAYS
from pet_triage.config import Settings
from pet_triage.domain.observation import Answers, Observation
from pet_triage.engine import FailSafe, Finish, NextAction, Triage, TriageEngine
from pet_triage.extractor import (
    PROMPT_VERSION,
    ExtractionError,
    Extractor,
    LangChainExtractor,
    ModelUnavailableError,
)
from pet_triage.intake import PET_ID, SUPPORTED_SPECIES, pet_observation
from pet_triage.messages import render
from pet_triage.practice import get_client_patients
from pet_triage.record import Assessment, Audit, ModelParams, TriageRecord

logger = logging.getLogger(__name__)


class TriageState(TypedDict):
    """Graph state, kept per thread by the checkpointer.

    Attributes:
        messages: The chat transcript, as Agent Chat UI expects.
        triage: The triage in progress; absent until the first message is assessed.
        answers: What ``extract`` read from the latest message.
        action: What ``assess`` decided to do next.
        record: The finished record for the vet, as JSON; set once the triage finishes.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    triage: NotRequired[Triage]
    answers: NotRequired[Answers]
    action: NotRequired[NextAction]
    record: NotRequired[dict[str, Any]]


class TriageNodes:
    """The graph nodes, sharing their injected dependencies.

    Args:
        engine: Runs the conversation.
        extractor: Reads owner messages.
        settings: Supplies the urgent-care number and the model name.
    """

    def __init__(self, engine: TriageEngine, extractor: Extractor, settings: Settings) -> None:
        """Store the dependencies the nodes share."""
        self._engine = engine
        self._extractor = extractor
        self._settings = settings

    def extract(self, state: TriageState) -> Command[Literal["assess", "fail", "reply"]]:
        """Read the latest owner message into ``Answers``, then assess them.

        The messages before it go to the extractor too, which shows it the last few.

        If the model can't be reached, the owner gets the fail-safe at once: asking them to
        rephrase wouldn't help. The triage is left as it was, so a later message carries on.
        """
        *history, latest = state["messages"]
        try:
            answers = self._extractor.extract(latest.text, _triage(state).last_asked, history)
        except ExtractionError as error:
            logger.warning("Extraction failed: %s", error)
            return Command(goto="fail")
        except ModelUnavailableError as error:
            logger.error("Model unavailable: %s", error)
            return Command(update={"action": FailSafe()}, goto="reply")
        return Command(update={"answers": answers}, goto="assess")

    def assess(self, state: TriageState) -> Command[Literal["record", "reply"]]:
        """Run one engine step, recording the triage if it has finished."""
        triage, action = self._engine.step(_triage(state), state["answers"])
        return Command(
            update={"triage": triage, "action": action},
            goto="record" if isinstance(action, Finish) else "reply",
        )

    def fail(self, state: TriageState) -> dict[str, object]:
        """Count a failed extraction; the engine decides whether to ask again or give up."""
        triage, action = self._engine.fail(_triage(state))
        return {"triage": triage, "action": action}

    def record(self, state: TriageState, config: RunnableConfig) -> dict[str, object]:
        """Add the finished record for the vet to the conversation's state.

        In production this node would post the record to the practice management system.
        For the demo the record is the conversation's final output: it shows in the last
        turn's MLflow trace and in the thread's state in LangGraph.
        """
        finish = state["action"]
        if not isinstance(finish, Finish):
            raise TypeError(f"record is only routed to on Finish, got {finish!r}")
        record = self._record(state, finish.assessment, config["configurable"]["thread_id"])
        return {"record": record.model_dump(mode="json")}

    def reply(self, state: TriageState) -> dict[str, object]:
        """Render the action as the assistant's message."""
        action = state["action"]
        first = not any(isinstance(message, AIMessage) for message in state["messages"])
        patient = self._engine.patient(_triage(state).answers)
        pet_name = None if patient is None else patient.name
        text = render(
            action, self._settings.urgent_care_phone_number, first=first, pet_name=pet_name
        )
        return {"messages": [AIMessage(text)]}

    def _record(self, state: TriageState, assessment: Assessment, thread_id: str) -> TriageRecord:
        """The finished record for the vet. Patient answers go in ``patient`` only.

        Raises:
            ValueError: If no patient was chosen, which a finished triage always has.
        """
        answers = state["triage"].answers
        patient = self._engine.patient(answers)
        if patient is None:
            raise ValueError("A finished triage has no patient.")
        return TriageRecord(
            patient=patient,
            record_id=thread_id,
            assessment=assessment,
            observations={key: value for key, value in answers.items() if key != PET_ID},
            owner_messages=tuple(
                msg.text for msg in state["messages"] if isinstance(msg, HumanMessage)
            ),
            audit=Audit(
                model=self._settings.model,
                params=ModelParams(
                    thinking=self._settings.thinking,
                    temperature=self._settings.temperature,
                    context_length=self._settings.context_length,
                ),
                prompt_version=PROMPT_VERSION,
                thread_id=thread_id,
            ),
        )


def _triage(state: TriageState) -> Triage:
    """The triage in progress, or a fresh one on the first message."""
    return state.get("triage", Triage())


def build_graph(
    nodes: TriageNodes, checkpointer: Checkpointer = None
) -> CompiledStateGraph[TriageState]:
    """Wire the nodes into a compiled graph. ``extract`` and ``assess`` route themselves.

    Args:
        nodes: The nodes, with their dependencies already injected.
        checkpointer: Keeps each thread's state between messages. ``langgraph dev``
            supplies its own, so the app passes none; tests pass an in-memory one.

    Returns:
        The compiled graph.
    """
    graph = StateGraph(TriageState)
    graph.add_node("extract", nodes.extract)
    graph.add_node("assess", nodes.assess)
    graph.add_node("fail", nodes.fail)
    graph.add_node("record", nodes.record)
    graph.add_node("reply", nodes.reply)
    graph.add_edge(START, "extract")
    graph.add_edge("fail", "reply")
    graph.add_edge("record", "reply")
    graph.add_edge("reply", END)
    return graph.compile(checkpointer=checkpointer)


def make_engine(settings: Settings) -> TriageEngine:
    """The engine over the real clinical content and the signed-in client's animals.

    In production the client's animals would be looked up when their session starts; the
    demo's client is fixed in the settings, so they are looked up once, at start-up.

    Raises:
        ValueError: If the client has no animals registered, so there is nothing to triage.
    """
    patients = get_client_patients(settings.client_id)
    if not patients:
        raise ValueError(f"Client {settings.client_id!r} has no animals registered.")
    return TriageEngine(
        PATHWAYS,
        pet_observation(patients),
        {patient.name: patient for patient in patients},
        SUPPORTED_SPECIES,
        COMPLAINTS,
    )


def make_model(settings: Settings) -> BaseChatModel:
    """The configured Ollama model, for the extractor and the evaluation's baseline.

    Fails fast if Ollama can't be reached or the model isn't pulled.
    """
    return init_chat_model(
        f"ollama:{settings.model}",
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
        num_ctx=settings.context_length,
        reasoning=settings.thinking,
        validate_model_on_init=True,
        client_kwargs={"timeout": settings.timeout_seconds},
    )


def make_extractor(
    settings: Settings,
    observations: tuple[Observation, ...],
    now: Callable[[], datetime] = datetime.now,
) -> LangChainExtractor:
    """The extractor on the configured Ollama model, warm and checked, for the app and evals.

    Fails fast if the model can't be loaded or extraction doesn't work, so nothing starts in
    a state where it can't read an owner's message.
    """
    extractor = LangChainExtractor(make_model(settings), observations, now)
    # Warm the model up, so the first real message isn't slowed by it
    extractor.extract("Hello", None)
    return extractor


def make_graph() -> CompiledStateGraph[TriageState]:
    """Build the graph with real dependencies, once, from ``pet_triage.app``."""
    settings = Settings()
    engine = make_engine(settings)
    extractor = make_extractor(settings, engine.observations)
    return build_graph(TriageNodes(engine, extractor, settings))
