"""Clinical judgement over a small fake pathway, independent of the real clinical content."""

import pytest

from pet_triage.assessment import assess, next_question
from pet_triage.domain.condition import AnyOf, AtLeast, Status, Yes
from pet_triage.domain.observation import Answers, Kind, Observation
from pet_triage.domain.outcome import Outcome
from pet_triage.domain.pathway import Pathway
from pet_triage.domain.symptom import Symptom
from pet_triage.record import Assessment

COUGHING = Observation("coughing", "Is your pet coughing?", Kind.YES_NO)
SNEEZING = Observation("sneezing", "Is your pet sneezing?", Kind.YES_NO)
COUGH_DAYS = Observation("cough_days", "For how many days?", Kind.NUMBER)
SWALLOWED = Observation("swallowed", "Has your pet swallowed anything?", Kind.ITEMS)
HOURS_SINCE = Observation("hours_since", "How long ago?", Kind.NUMBER)
BLUE_GUMS = Observation("blue_gums", "Are their gums blue?", Kind.YES_NO)

COUGH = Symptom("cough", Outcome.STANDARD, Yes(COUGHING))
SNEEZE = Symptom("sneeze", Outcome.STANDARD, Yes(SNEEZING))
LONG_COUGH = Symptom("long_cough", Outcome.URGENT, AtLeast(COUGH_DAYS, 7.0))
SWALLOWED_BONE = Symptom("swallowed_bone", Outcome.URGENT, AnyOf(SWALLOWED, frozenset({"bone"})))
BONE_STUCK = Symptom(
    "bone_stuck", Outcome.VERY_URGENT, AtLeast(HOURS_SINCE, 24.0), requires=SWALLOWED_BONE
)
CYANOSIS = Symptom("cyanosis", Outcome.IMMEDIATE, Yes(BLUE_GUMS))

COUGHING_PATHWAY = Pathway(
    "coughing", COUGH, (CYANOSIS, BONE_STUCK, SWALLOWED_BONE, LONG_COUGH), "Sorry about the cough."
)
"""Follow-ups deliberately out of urgency order, to show the questions are sorted."""
SNEEZING_PATHWAY = Pathway("sneezing", SNEEZE, (CYANOSIS,), "Sorry about the sneezing.")
"""Shares cyanosis with coughing, to show a shared symptom is only asked about once."""
PATHWAYS = (COUGHING_PATHWAY, SNEEZING_PATHWAY)


def test_no_assessment_before_a_pathway_is_active() -> None:
    assert assess(PATHWAYS, {"coughing": False}) is None


def test_assessment_takes_the_most_urgent_present_symptom_and_reports_the_rest() -> None:
    answers: Answers = {"coughing": True, "cough_days": 8.0, "swallowed": frozenset({"bone"})}
    assert assess(PATHWAYS, answers) == Assessment(
        outcome=Outcome.URGENT,
        symptoms={
            "cough": Status.PRESENT,
            "cyanosis": Status.UNKNOWN,
            "bone_stuck": Status.UNKNOWN,
            "swallowed_bone": Status.PRESENT,
            "long_cough": Status.PRESENT,
        },
        outcome_driven_by=("swallowed_bone", "long_cough"),
    )


@pytest.mark.parametrize(
    ("answers", "expected"),
    [
        ({"coughing": True}, SWALLOWED),
        ({"coughing": True, "swallowed": frozenset({"grass"})}, COUGH_DAYS),
        ({"coughing": True, "swallowed": frozenset({"bone"}), "cough_days": 2.0}, HOURS_SINCE),
        ({"coughing": True, "swallowed": frozenset({"grass"}), "cough_days": 2.0}, BLUE_GUMS),
        ({"coughing": True, "blue_gums": False}, SWALLOWED),
        (
            {
                "coughing": True,
                "swallowed": frozenset({"grass"}),
                "cough_days": 2.0,
                "blue_gums": False,
            },
            None,
        ),
        ({"coughing": False}, None),
    ],
    ids=[
        "least urgent first",
        "then the next",
        "prerequisite present",
        "prerequisite absent, skipped",
        "volunteered answer skipped",
        "all answered",
        "no pathway active",
    ],
)
def test_next_question(answers: Answers, expected: Observation | None) -> None:
    assert next_question(PATHWAYS, answers, asked=frozenset()) == expected


def test_a_question_already_asked_is_not_asked_again() -> None:
    answers: Answers = {"coughing": True}
    assert next_question(PATHWAYS, answers, asked=frozenset({"swallowed"})) == COUGH_DAYS
