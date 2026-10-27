"""Turns an owner's message into typed answers. The only place the model reads owner text.

The extractor never decides what an answer means: thresholds and item lists are applied
afterwards by the domain's conditions. It sees only what it needs for one message (the
message, the question asked and the last few messages before it), never the engine's
state.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, make_dataclass
from datetime import datetime, time, timedelta
from operator import itemgetter
from typing import Any, Literal, Protocol

import httpx
from langchain.agents.middleware.pii import detect_credit_card, detect_email
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import TypeAdapter, ValidationError

from pet_triage.domain.observation import OTHER, Answers, Kind, Observation

PROMPT_VERSION = "extract@2"
"""Recorded on every trace and record, so results can be tied to the prompt."""

#Earlier messages shown with each new one: the last two exchanges
HISTORY_MESSAGES = 4

PROMPT = f"""You read one message from a dog owner on a vet triage line and record only \
what they actually said.

Rules:
- Fill in a field only if the latest message answers it. Leave every other field empty.
- Earlier messages, if shown, are only there so you can tell what the latest message \
refers to, like "the one I mentioned" or "again". Never record something said only earlier.
- Never guess. An owner who says they don't know has not answered.
- Where a field lists allowed values, use one of them, or "{OTHER}" for anything not listed.
- For when something happened, give the hours ago if the owner said so; otherwise give \
days_ago (0 today, 1 yesterday) and the 24-hour time, and never work out the hours yourself.
- Reply with only a JSON object holding the fields you filled in.

Fields:
{{fields}}"""


@dataclass(frozen=True, slots=True)
class DayAndTime:
    """When something happened, as the owner said it: "yesterday at 8pm" is 1 and ``20:00``.

    The model reads these; ``hours_between`` works out the hours, so no arithmetic is left
    to the model.
    """

    days_ago: int
    time: str


class ExtractionError(Exception):
    """The model gave no answer in the schema, or one that didn't validate."""


class ModelUnavailableError(Exception):
    """The model couldn't be reached, even after retrying."""


class Extractor(Protocol):
    """Reads one owner message. The real one calls a model; tests use a scripted fake."""

    def extract(
        self, message: str, last_asked: str | None, history: Sequence[AnyMessage] = ()
    ) -> Answers:
        """Extract answers from one owner message.

        Args:
            message: What the owner typed.
            last_asked: ID of the question the owner is replying to, if any.
            history: The conversation before this message, oldest first. Only the last
                ``HISTORY_MESSAGES`` are shown, to resolve what the message refers to.

        Returns:
            The answers found. Anything not mentioned is left out, meaning unknown.

        Raises:
            ExtractionError: If the model gives no answer in the schema, or one that can't be
                validated.
            ModelUnavailableError: If the model can't be reached.
        """
        ...


class LangChainExtractor:
    """Extracts with one LangChain chain: prompt, then model, then validation.

    It is a chain, not an agent: one model call per message, with no tools and no loop.
    Ollama is given the schema from ``_build_schema`` to write to (``format``), and the
    reply is validated against the same schema, so the model can't invent a field, item or
    choice. Invalid output isn't retried here: it raises ``ExtractionError``, and the
    engine's own policy (ask the owner again, then the fail-safe) takes over, so there is
    one retry policy for what the model says.

    Reaching the model is a different failure: a refused connection is retried twice, with
    backoff (``with_retry``), then ``ModelUnavailableError`` is raised. Timeouts aren't
    retried, so a hung model can't keep the owner waiting several timeouts long.

    The last few messages before this one are shown as context, so "the one I mentioned"
    can be resolved, but the prompt only lets the model record what the latest message
    says. A "no" read from history still only counts for the question just asked, as the
    engine drops any other.

    Emails and card numbers are redacted before the model sees the message, with
    LangChain's PII detectors, so they stay out of prompts and traces even if a hosted
    model replaces the local one. That doesn't touch the chat history the graph keeps.
    Phone numbers aren't redacted: a pattern loose enough to catch them also caught dates,
    which are clinical facts here.

    Args:
        model: The chat model: local Ollama in the app, a scripted fake in tests.
        observations: Every observation to look for, from the engine.
        now: The clock a day and time are measured against.
    """

    def __init__(
        self,
        model: BaseChatModel,
        observations: tuple[Observation, ...],
        now: Callable[[], datetime] = datetime.now,
    ) -> None:
        """Build the chain once; every message reuses it."""
        self._now = now
        self._questions = {observation.id: observation.question for observation in observations}
        self._schema: TypeAdapter[Any] = TypeAdapter(_build_schema(observations))
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(PROMPT.format(fields=describe_fields(observations))),
                ("human", "{message}"),
            ]
        )
        reader = model.bind(format=self._schema.json_schema()).with_retry(
            retry_if_exception_type=(httpx.ConnectError,), stop_after_attempt=3
        )
        self._chain = prompt | reader

    def extract(
        self, message: str, last_asked: str | None, history: Sequence[AnyMessage] = ()
    ) -> Answers:
        """Implement ``Extractor.extract`` with one chain call."""
        context = describe_history(history)
        if last_asked:
            context += f"The owner was asked: {self._questions[last_asked]}\n"
        try:
            reply = self._chain.invoke({"message": f"{context}The owner wrote: {_redact(message)}"})
        except httpx.ConnectError as error:
            raise ModelUnavailableError(str(error)) from error
        try:
            answered = vars(self._schema.validate_json(reply.text))
            return {
                key: self._to_value(value) for key, value in answered.items() if value is not None
            }
        except (ValidationError, ValueError) as error:
            raise ExtractionError(str(error)) from error

    def _to_value(self, value: Any) -> Any:  # noqa: ANN401
        """A schema answer as a ``Value``: a day and time becomes hours ago, the rest as is."""
        if isinstance(value, DayAndTime):
            return hours_between(self._now(), value.days_ago, value.time)
        return value


def describe_history(history: Sequence[AnyMessage]) -> str:
    """The last ``HISTORY_MESSAGES`` messages for the prompt, redacted, or "" if none.

    The evaluation's baseline is given the same lines.
    """
    lines = [
        f"{'Owner' if isinstance(message, HumanMessage) else 'Triage line'}: "
        f"{_redact(message.text)}"
        for message in history[-HISTORY_MESSAGES:]
    ]
    if not lines:
        return ""
    return "Earlier in the conversation:\n" + "\n".join(lines) + "\n"


def _redact(message: str) -> str:
    """The message with emails and card numbers replaced, as ``PIIMiddleware`` would.

    The detectors are LangChain's; replacing from the end keeps earlier positions valid.
    """
    matches = [*detect_email(message), *detect_credit_card(message)]
    for match in sorted(matches, key=itemgetter("start"), reverse=True):
        marker = f"[REDACTED_{match['type'].upper()}]"
        message = f"{message[: match['start']]}{marker}{message[match['end'] :]}"
    return message


def hours_between(now: datetime, days_ago: int, clock_time: str) -> float:
    """Hours from a moment ``days_ago`` days back at ``clock_time`` until ``now``.

    Args:
        now: The current time.
        days_ago: 0 for today, 1 for yesterday, and so on.
        clock_time: The time of day, 24-hour ``HH:MM``.

    Returns:
        The hours elapsed, to one decimal place.

    Raises:
        ValueError: If the time isn't ``HH:MM``, or the moment is in the future.
    """
    day = now.date() - timedelta(days=days_ago)
    moment = datetime.combine(day, time.fromisoformat(clock_time))
    hours = (now - moment) / timedelta(hours=1)
    if hours < 0:
        raise ValueError(f"{clock_time} on that day hasn't happened yet; check days_ago.")
    return round(hours, 1)


def _build_schema(observations: tuple[Observation, ...]) -> type:
    """Build the structured-output schema from the observations.

    Each observation becomes an optional field typed by its kind, and an observation with
    options only accepts those values, so the model can't invent an item or choice. The
    model's reply is validated against it. The schema holds every observation,
    which is small at demo scale; with many groups it would be limited to the groups the
    owner's concern touches.

    Args:
        observations: Every observation to look for.

    Returns:
        A dataclass with one optional field per observation.
    """
    fields = [(obs.id, _answer_type(obs) | None, field(default=None)) for obs in observations]
    return make_dataclass("OwnerAnswers", fields, frozen=True)


def _answer_type(observation: Observation) -> Any:  # noqa: ANN401
    """The type an observation's answer must have: the shape of ``Value`` by kind.

    Returns a type form such as ``Literal["dog", "cat"]``. Options are only known at
    runtime, and subscript syntax can't take them there, so ``Literal`` and ``frozenset``
    are subscripted through the calls the syntax makes. Typing has no static type for type
    forms yet (PEP 747), hence ``Any``.
    """
    match observation.kind:
        case Kind.YES_NO:
            return bool
        case Kind.NUMBER:
            return float
        case Kind.HOURS_AGO:
            return float | DayAndTime
        case Kind.CHOICE:
            return _one_of(observation.options)
        case Kind.ITEMS:
            return frozenset.__class_getitem__(_one_of(observation.options))


def _one_of(options: frozenset[str]) -> Any:  # noqa: ANN401
    """``Literal`` over the options, sorted so the schema is the same on every run.

    A set's iteration order changes between processes, so it can't be used as it is.
    """
    return Literal.__getitem__(tuple(sorted(options)))


def describe_fields(observations: tuple[Observation, ...]) -> str:
    """One line per field for the prompt: its name, its question and any allowed values.

    The schema already restricts the values, but the model only sees them as constraints
    on what it may write; listing them in the prompt is what lets it map "chocolate" onto
    ``chocolate``. The evaluation's baseline is given the same lines.
    """
    return "\n".join(_describe_one(observation) for observation in observations)


def _describe_one(observation: Observation) -> str:
    """The prompt line for one field."""
    line = f"- {observation.id}: {observation.question}"
    if not observation.options:
        return line
    return f"{line} Allowed values: {', '.join(sorted(observation.options))}."
