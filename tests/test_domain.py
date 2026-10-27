"""The domain's pure logic: outcome order, conditions and symptom status."""

import pytest

from pet_triage.domain.condition import AnyOf, AtLeast, Status, Yes
from pet_triage.domain.observation import Answers, Kind, Observation
from pet_triage.domain.outcome import Outcome
from pet_triage.domain.symptom import Symptom

LIMPING = Observation("limping", "Is your pet limping?", Kind.YES_NO)
DAYS_LIMPING = Observation("days_limping", "For how many days?", Kind.NUMBER)
SWALLOWED = Observation("swallowed", "What did your pet swallow?", Kind.ITEMS)

LAME = Symptom("lame", Outcome.STANDARD, Yes(LIMPING))
LAME_FOR_A_WEEK = Symptom(
    "lame_for_a_week", Outcome.URGENT, AtLeast(DAYS_LIMPING, 7), requires=LAME
)
SWALLOWED_STONE = Symptom("swallowed_stone", Outcome.URGENT, AnyOf(SWALLOWED, frozenset({"stone"})))


def test_outcomes_are_ordered_from_least_to_most_urgent() -> None:
    assert sorted(Outcome, reverse=True) == list(reversed(Outcome))
    assert max(Outcome.URGENT, Outcome.IMMEDIATE, Outcome.STANDARD) is Outcome.IMMEDIATE


@pytest.mark.parametrize(
    ("answers", "expected"),
    [
        ({"swallowed": frozenset({"stone", "grass"})}, Status.PRESENT),
        ({"swallowed": frozenset({"stone", "other"})}, Status.PRESENT),
        ({"swallowed": frozenset({"grass", "other"})}, Status.UNKNOWN),
        ({"swallowed": frozenset({"grass"})}, Status.ABSENT),
        ({"swallowed": frozenset()}, Status.ABSENT),
        ({}, Status.UNKNOWN),
    ],
    ids=["match", "match and unlisted", "unlisted", "no match", "nothing", "unanswered"],
)
def test_any_of(answers: Answers, expected: Status) -> None:
    assert SWALLOWED_STONE.status(answers) is expected


@pytest.mark.parametrize(
    ("answers", "expected"),
    [
        ({"limping": True, "days_limping": 8.0}, Status.PRESENT),
        ({"limping": True, "days_limping": 7.0}, Status.PRESENT),
        ({"limping": True, "days_limping": 6.0}, Status.ABSENT),
        ({"limping": True}, Status.UNKNOWN),
        ({"limping": False, "days_limping": 8.0}, Status.ABSENT),
        ({"days_limping": 8.0}, Status.UNKNOWN),
    ],
    ids=["above", "at", "below", "unanswered", "prerequisite absent", "prerequisite unknown"],
)
def test_at_least_with_prerequisite(answers: Answers, expected: Status) -> None:
    assert LAME_FOR_A_WEEK.status(answers) is expected


@pytest.mark.parametrize(
    ("symptom", "answers"),
    [
        (LAME, {"limping": "yes"}),
        (LAME_FOR_A_WEEK, {"limping": True, "days_limping": "eight"}),
        (SWALLOWED_STONE, {"swallowed": "stone"}),
    ],
    ids=["yes/no", "number", "items"],
)
def test_wrongly_typed_answers_fail_loudly(symptom: Symptom, answers: Answers) -> None:
    with pytest.raises(TypeError):
        symptom.status(answers)
