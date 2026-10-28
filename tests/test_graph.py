"""A whole call through the real graph and clinical content, with a scripted extractor."""

from collections.abc import Iterator, Sequence

from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from pet_triage.clinical.gastrointestinal.observations import (
    BELLY_SWOLLEN_QUICKLY,
    VOMIT_COUNT_24H,
)
from pet_triage.clinical.pathways import VOMITING
from pet_triage.config import Settings
from pet_triage.domain.observation import Answers
from pet_triage.domain.outcome import Outcome
from pet_triage.extractor import ExtractionError, Extractor, ModelUnavailableError
from pet_triage.graph import TriageNodes, TriageState, build_graph, make_engine
from pet_triage.intake import pet_observation
from pet_triage.messages import ANNOUNCEMENTS, FAIL_SAFE, OPENER, REPHRASE, YOUR_PET
from pet_triage.practice import get_client_patients
from pet_triage.record import ModelParams, TriageRecord

PHONE = "01632 960000"
PET = pet_observation(get_client_patients(Settings().client_id))
CALL: RunnableConfig = {"configurable": {"thread_id": "call-1"}}


class ScriptedExtractor:
    """Returns scripted answers in order; ``None`` stands for a failed extraction."""

    def __init__(self, *replies: Answers | None) -> None:
        """Queue the replies."""
        self._replies: Iterator[Answers | None] = iter(replies)
        self.histories: list[list[str]] = []

    def extract(
        self, message: str, last_asked: str | None, history: Sequence[AnyMessage] = ()
    ) -> Answers:
        """The next scripted reply, or ``ExtractionError`` for ``None``. Keeps each history."""
        self.histories.append([earlier.text for earlier in history])
        answers = next(self._replies)
        if answers is None:
            raise ExtractionError("scripted failure")
        return answers


class UnreachableExtractor:
    """An extractor whose model can never be reached."""

    def extract(
        self, message: str, last_asked: str | None, history: Sequence[AnyMessage] = ()
    ) -> Answers:
        """Always ``ModelUnavailableError``."""
        raise ModelUnavailableError("scripted outage")


def graph(*replies: Answers | None) -> CompiledStateGraph[TriageState]:
    """The real graph and clinical content, reading owner messages from a script."""
    return graph_with(ScriptedExtractor(*replies))


def graph_with(extractor: Extractor) -> CompiledStateGraph[TriageState]:
    """The real graph and clinical content, reading owner messages with ``extractor``."""
    settings = Settings(urgent_care_phone_number=PHONE)
    nodes = TriageNodes(make_engine(settings), extractor, settings)
    return build_graph(nodes, checkpointer=InMemorySaver())


def say(app: CompiledStateGraph[TriageState], text: str) -> str:
    """Send one owner message on a single thread and return the bot's reply."""
    result = app.invoke(TriageState(messages=[HumanMessage(text)]), CALL)
    return str(result["messages"][-1].text)


def bella(text: str) -> str:
    """The text as the owner reads it once they have said the call is about Bella."""
    return text.replace(YOUR_PET, "Bella")


def test_a_whole_call_ends_with_advice_and_a_record() -> None:
    app = graph(
        {},
        {"pet": "Bella"},
        {
            "complaints": frozenset({"vomiting"}),
            "ingested_items": frozenset({"sock"}),
            "hours_since_ingestion": 30.0,
        },
        {"vomit_count_24h": 2.0},
        {"black_tarry_stool": False},
        {"belly_swollen_quickly": False},
    )

    assert say(app, "Hi") == f"{OPENER.format(phone=PHONE)}\n\n{PET.question}"
    assert say(app, "Bella") == "What's going on with Bella today?"
    announcement = ANNOUNCEMENTS[Outcome.VERY_URGENT]
    assert say(app, "He's sick, he ate a sock yesterday") == bella(
        f"{VOMITING.acknowledgement}\n\n{announcement}\n\n{VOMIT_COUNT_24H.question}"
    )
    assert say(app, "Twice") == "Is Bella's poo black and tarry?"
    assert say(app, "No") == bella(BELLY_SWOLLEN_QUICKLY.question)
    assert say(app, "No").endswith("Come in now.")

    record = TriageRecord.model_validate(app.get_state(CALL).values["record"])
    assert record.assessment.outcome is Outcome.VERY_URGENT
    assert record.assessment.outcome_driven_by == ("foreign_body_over_24h",)
    assert record.patient.name == "Bella"
    assert "pet" not in record.observations
    assert len(record.owner_messages) == 6
    assert record.audit.params == ModelParams(thinking=True, temperature=0.0, context_length=8192)


def test_failed_extractions_ask_to_rephrase_then_give_up() -> None:
    app = graph(None, None, None)
    assert say(app, "???").endswith(REPHRASE)
    assert say(app, "???") == REPHRASE
    assert say(app, "???") == FAIL_SAFE.format(phone=PHONE)


def test_an_unreachable_model_gets_the_fail_safe_at_once() -> None:
    app = graph_with(UnreachableExtractor())
    reply = say(app, "my dog is vomiting")
    assert reply == f"{OPENER.format(phone=PHONE)}\n\n{FAIL_SAFE.format(phone=PHONE)}"


def test_the_extractor_is_given_the_messages_before_the_latest() -> None:
    extractor = ScriptedExtractor({}, {"pet": "Bella"})
    app = graph_with(extractor)
    opener = say(app, "Hi")
    say(app, "Bella")
    assert extractor.histories == [[], ["Hi", opener]]
