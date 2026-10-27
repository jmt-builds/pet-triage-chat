"""Intake: the question asking which of the client's pets the call is about."""

from pet_triage.domain.observation import OTHER
from pet_triage.domain.patient import Patient
from pet_triage.intake import pet_observation

REX = Patient("p1", "c1", "Rex", "dog", "Border Collie", "male", True, 3.0)
TOM = Patient("p2", "c1", "Tom", "cat", "Siamese", "male", True, 5.0)


def test_the_question_names_each_pet_and_offers_other() -> None:
    pet = pet_observation([REX, TOM])
    assert (
        pet.question
        == "Which pet are you asking about today: Rex (dog, Border Collie) or Tom (cat, Siamese)?"
    )
    assert pet.options == {"Rex", "Tom", OTHER}
