"""The extractor's agent, driven by a scripted chat model instead of Ollama."""

import json
from datetime import datetime

import httpx
import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult

from fakes import ScriptedChatModel
from pet_triage.domain.observation import Kind, Observation
from pet_triage.extractor import (
    ExtractionError,
    LangChainExtractor,
    ModelUnavailableError,
    _redact,
    hours_between,
)

SPECIES = Observation(
    "species", "Is your pet a dog?", Kind.CHOICE, options=frozenset({"dog", "cat"})
)
COUGHING = Observation("coughing", "Is your pet coughing?", Kind.YES_NO)
COUGH_DAYS = Observation("cough_days", "For how many days?", Kind.NUMBER)
SWALLOWED = Observation(
    "swallowed", "What did they swallow?", Kind.ITEMS, frozenset({"bone", "other"})
)
COUGH_STARTED = Observation("cough_started", "When did it start?", Kind.HOURS_AGO)
OBSERVATIONS = (SPECIES, COUGHING, COUGH_DAYS, SWALLOWED, COUGH_STARTED)

NOW = datetime(2026, 10, 23, 15, 0)


def extractor(reply: str) -> LangChainExtractor:
    """An extractor whose model gives one scripted reply."""
    model = ScriptedChatModel(messages=iter([AIMessage(reply)]))
    return LangChainExtractor(model, OBSERVATIONS, now=lambda: NOW)


class UnreachableChatModel(ScriptedChatModel):
    """A chat model whose server refuses every connection, counting the attempts."""

    calls: int = 0

    def _generate(self, *args: object, **kwargs: object) -> ChatResult:
        """Fail as Ollama's client does when nothing is listening."""
        self.calls += 1
        raise httpx.ConnectError("Connection refused")


def test_answers_are_typed_and_unanswered_fields_left_out() -> None:
    reply = {"coughing": True, "cough_days": 3, "swallowed": ["bone"], "species": None}
    result = extractor(json.dumps(reply)).extract("Coughing for 3 days", None)
    assert result == {"coughing": True, "cough_days": 3.0, "swallowed": frozenset({"bone"})}


def test_an_answer_outside_the_options_fails_extraction() -> None:
    with pytest.raises(ExtractionError):
        extractor(json.dumps({"species": "cow"})).extract("She's a cow", None)


def test_a_reply_that_is_not_json_fails_extraction() -> None:
    with pytest.raises(ExtractionError):
        extractor("The dog is coughing.").extract("Coughing for 3 days", COUGH_DAYS.id)


def test_an_unreachable_model_is_retried_then_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("time.sleep", lambda _: None)  # skip the backoff
    model = UnreachableChatModel(messages=iter([]))
    with pytest.raises(ModelUnavailableError):
        LangChainExtractor(model, OBSERVATIONS, now=lambda: NOW).extract("Hello", None)
    assert model.calls == 3  # the first try and two retries


@pytest.mark.parametrize(
    ("said", "expected"),
    [(3, 3.0), ({"days_ago": 1, "time": "20:00"}, 19.0)],
    ids=["hours ago", "day and time"],
)
def test_times_become_hours_ago(said: object, expected: float) -> None:
    reply = json.dumps({"cough_started": said})
    assert extractor(reply).extract("...", COUGH_STARTED.id) == {"cough_started": expected}


def test_a_time_still_to_come_fails_extraction() -> None:
    reply = json.dumps({"cough_started": {"days_ago": 0, "time": "20:00"}})
    with pytest.raises(ExtractionError):
        extractor(reply).extract("Tonight at 8", COUGH_STARTED.id)


@pytest.mark.parametrize(
    ("days_ago", "clock_time", "expected"),
    [(0, "09:30", 5.5), (1, "20:00", 19.0), (2, "15:00", 48.0), (0, "15:00", 0.0)],
    ids=["this morning", "last night", "two days ago", "just now"],
)
def test_hours_between(days_ago: int, clock_time: str, expected: float) -> None:
    assert hours_between(NOW, days_ago, clock_time) == expected


@pytest.mark.parametrize(
    ("days_ago", "clock_time"), [(0, "20:00"), (0, "8pm")], ids=["later today", "not HH:MM"]
)
def test_impossible_times_are_rejected(days_ago: int, clock_time: str) -> None:
    with pytest.raises(ValueError, match=r".+"):
        hours_between(NOW, days_ago, clock_time)


def test_emails_and_card_numbers_are_redacted_before_the_model_sees_them() -> None:
    message = "email me at sam@example.com, card 4111 1111 1111 1111, she was sick at 9"
    assert _redact(message) == (
        "email me at [REDACTED_EMAIL], card [REDACTED_CREDIT_CARD], she was sick at 9"
    )
