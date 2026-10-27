"""What the owner reads when the assistant can't help."""

import pytest

from pet_triage.clinical.complaints import COMPLAINTS
from pet_triage.clinical.pathways import PATHWAYS
from pet_triage.domain.observation import Kind, Observation
from pet_triage.domain.outcome import Outcome
from pet_triage.engine import (
    Ask,
    FailSafe,
    Finish,
    NextAction,
    OutOfScope,
    Rephrase,
    Unsupported,
)
from pet_triage.messages import ANNOUNCEMENTS, WAITING_OUTCOMES, render
from pet_triage.record import Assessment

PHONE = "01632 960000"


def finish(outcome: Outcome) -> Finish:
    """A finished triage at ``outcome``."""
    return Finish(Assessment(outcome=outcome, symptoms={}, outcome_driven_by=()))


@pytest.mark.parametrize(
    "action",
    [*(OutOfScope(reason) for reason in Unsupported), FailSafe()],
    ids=["not on account", "species", "concern", "fail-safe"],
)
def test_messages_that_cannot_help_give_the_phone_number(action: NextAction) -> None:
    assert PHONE in render(action, PHONE)


def test_a_rephrase_asks_the_question_again() -> None:
    question = Observation("coughing", "Is your pet coughing?", Kind.YES_NO)
    assert (
        render(Rephrase(question), PHONE) == "Sorry, I didn't quite follow. Is your pet coughing?"
    )


@pytest.mark.parametrize("outcome", list(Outcome), ids=str)
def test_only_an_owner_left_to_wait_is_told_when_to_call(outcome: Outcome) -> None:
    text = render(finish(outcome), PHONE)
    assert (PHONE in text) is (outcome in WAITING_OUTCOMES)


QUESTIONS = (COMPLAINTS, *(o for pathway in PATHWAYS for o in pathway.observations))
ONCE_THE_PET_IS_KNOWN = [
    *(Ask(question) for question in QUESTIONS),
    *(Rephrase(question) for question in QUESTIONS),
    *(Ask(COMPLAINTS, outcome) for outcome in ANNOUNCEMENTS),
    *(finish(outcome) for outcome in Outcome),
]


@pytest.mark.parametrize("action", ONCE_THE_PET_IS_KNOWN, ids=lambda action: type(action).__name__)
def test_once_the_pet_is_known_it_is_called_by_name(action: NextAction) -> None:
    assert "your pet" not in render(action, PHONE, pet_name="Bella").lower()


def test_another_species_is_told_only_which_species_are_supported() -> None:
    text = render(OutOfScope(Unsupported.SPECIES), PHONE)
    assert "these species: dog." in text
    assert "vomiting" not in text
