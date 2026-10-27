"""The engine's conversation, over small fake pathways."""

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from pet_triage.domain.condition import AtLeast, Status, Yes
from pet_triage.domain.observation import OTHER, Answers, Kind, Observation
from pet_triage.domain.outcome import Outcome
from pet_triage.domain.pathway import Pathway
from pet_triage.domain.patient import Patient
from pet_triage.domain.symptom import Symptom
from pet_triage.engine import (
    MAX_EXTRACTION_FAILURES,
    Ask,
    FailSafe,
    Finish,
    NextAction,
    OutOfScope,
    Rephrase,
    Stage,
    Triage,
    TriageEngine,
    Unsupported,
)
from pet_triage.record import Assessment

PET = Observation("pet", "Rex or Tom?", Kind.CHOICE, options=frozenset({"Rex", "Tom", OTHER}))
REX = Patient("p1", "c1", "Rex", "dog", "Border Collie", "male", True, 3.0)
TOM = Patient("p2", "c1", "Tom", "cat", "Siamese", "male", True, 5.0)
COUGHING = Observation("coughing", "Is your pet coughing?", Kind.YES_NO)
COUGH_DAYS = Observation("cough_days", "For how many days?", Kind.NUMBER)
SNEEZING = Observation("sneezing", "Is your pet sneezing?", Kind.YES_NO)
BLUE_GUMS = Observation("blue_gums", "Are their gums blue?", Kind.YES_NO)
COMPLAINTS = Observation(
    "complaints", "What's wrong?", Kind.ITEMS, frozenset({"coughing", "sneezing", OTHER})
)

COUGH = Symptom("cough", Outcome.STANDARD, Yes(COUGHING))
LONG_COUGH = Symptom("long_cough", Outcome.URGENT, AtLeast(COUGH_DAYS, 7.0))
SNEEZE = Symptom("sneeze", Outcome.STANDARD, Yes(SNEEZING))
CYANOSIS = Symptom("cyanosis", Outcome.IMMEDIATE, Yes(BLUE_GUMS))

COUGHING_PATHWAY = Pathway("coughing", COUGH, (CYANOSIS, LONG_COUGH), "Sorry about the cough.")
SNEEZING_PATHWAY = Pathway("sneezing", SNEEZE, (CYANOSIS,), "Sorry about the sneezing.")

PATHWAYS = (COUGHING_PATHWAY, SNEEZING_PATHWAY)
ENGINE = TriageEngine(PATHWAYS, PET, {"Rex": REX, "Tom": TOM}, frozenset({"dog"}), COMPLAINTS)

A_DOG: Answers = {"pet": "Rex"}
"""The call is about the client's dog."""

NEW_CALL = Triage()
CONCERN_ASKED = Triage(asked=frozenset({COMPLAINTS.id}))
"""A call whose concern has been asked, for tests about the clinical questions."""


def converse(*messages: Answers, start: Triage = NEW_CALL) -> tuple[Triage, NextAction]:
    """Run a call through the engine, one owner message at a time."""
    triage, action = ENGINE.step(start, messages[0])
    for answers in messages[1:]:
        triage, action = ENGINE.step(triage, answers)
    return triage, action


def test_observations_are_pet_then_concern_then_every_pathway_without_duplicates() -> None:
    assert ENGINE.observations == (
        PET,
        COMPLAINTS,
        COUGHING,
        BLUE_GUMS,
        COUGH_DAYS,
        SNEEZING,
    )


def test_the_first_message_asks_which_pet() -> None:
    triage, action = converse({})
    assert action == Ask(PET)
    assert triage.stage is Stage.PATIENT
    assert triage.last_asked == PET.id


def test_which_pet_is_asked_until_answered() -> None:
    _, action = converse({}, {})
    assert action == Ask(PET)


def test_a_pet_of_an_unsupported_species_is_out_of_scope() -> None:
    triage, action = converse({}, {"pet": "Tom"})
    assert action == OutOfScope(Unsupported.SPECIES)
    assert triage.stage is Stage.DONE


def test_a_pet_not_on_the_account_is_out_of_scope() -> None:
    _, action = converse({}, {"pet": OTHER})
    assert action == OutOfScope(Unsupported.PET)


def test_a_clients_only_pet_is_chosen_without_asking() -> None:
    engine = TriageEngine(PATHWAYS, PET, {"Rex": REX}, frozenset({"dog"}), COMPLAINTS)
    triage, action = engine.step(Triage(), {})
    assert action == Ask(COMPLAINTS)
    assert engine.patient(triage.answers) == REX


def test_the_patient_is_the_chosen_pets_record() -> None:
    assert ENGINE.patient({"pet": "Rex"}) == REX


def test_there_is_no_patient_until_one_on_the_account_is_chosen() -> None:
    assert ENGINE.patient({}) is None
    assert ENGINE.patient({"pet": OTHER}) is None


def test_the_concern_is_asked_once_the_pet_is_known() -> None:
    triage, action = converse({}, A_DOG)
    assert action == Ask(COMPLAINTS)
    assert triage.stage is Stage.CONCERN
    assert triage.last_asked == COMPLAINTS.id


def test_an_unclear_concern_is_asked_about_once_more() -> None:
    triage, action = converse({}, A_DOG, {})
    assert action == Rephrase(COMPLAINTS)
    assert triage.last_asked == COMPLAINTS.id
    assert triage.stage is Stage.CLARIFY


def test_a_concern_no_pathway_covers_is_out_of_scope() -> None:
    triage, action = converse({}, A_DOG, {}, {})
    assert action == OutOfScope(Unsupported.CONCERN)
    assert triage.stage is Stage.DONE


def test_a_complaint_no_pathway_covers_is_out_of_scope_even_beside_one_it_does() -> None:
    triage, action = converse(
        {**A_DOG, "coughing": True, "complaints": frozenset({"coughing", OTHER})}
    )
    assert action == OutOfScope(Unsupported.CONCERN)
    assert triage.stage is Stage.DONE


def test_complaints_every_pathway_covers_carry_on() -> None:
    _, action = converse(
        {**A_DOG, "coughing": True, "complaints": frozenset({"coughing"})}, start=CONCERN_ASKED
    )
    assert action == Ask(COUGH_DAYS, acknowledge=(COUGHING_PATHWAY,))


def test_a_pathway_is_acknowledged_on_its_first_question_only() -> None:
    triage, action = converse({**A_DOG, "coughing": True}, start=CONCERN_ASKED)
    assert action == Ask(COUGH_DAYS, acknowledge=(COUGHING_PATHWAY,))

    _, action = ENGINE.step(triage, {"cough_days": 2.0})
    assert action == Ask(BLUE_GUMS)


def test_a_pathway_that_becomes_active_later_is_acknowledged_then() -> None:
    _, action = converse(
        {**A_DOG, "coughing": True}, {"cough_days": 2.0, "sneezing": True}, start=CONCERN_ASKED
    )
    assert action == Ask(BLUE_GUMS, acknowledge=(SNEEZING_PATHWAY,))


def test_a_volunteered_concern_is_still_asked() -> None:
    _, action = converse({**A_DOG, "coughing": True})
    assert action == Ask(COMPLAINTS)


def test_a_rise_to_urgent_is_announced_once() -> None:
    triage, action = converse({"coughing": True, "cough_days": 8.0})
    assert action == Ask(PET, announce=Outcome.URGENT)

    _, action = ENGINE.step(triage, {"pet": "Rex"})
    assert action == Ask(COMPLAINTS)


def test_a_later_answer_replaces_an_earlier_one() -> None:
    triage, action = converse(
        {**A_DOG, "coughing": True, "cough_days": 2.0},
        {"blue_gums": False, "cough_days": 9.0},
        start=CONCERN_ASKED,
    )
    assert triage.answers["cough_days"] == 9.0
    assert isinstance(action, Finish)
    assert action.assessment.outcome is Outcome.URGENT


def test_finish_once_every_question_is_answered() -> None:
    triage, action = converse(
        {**A_DOG, "coughing": True, "cough_days": 2.0}, {"blue_gums": False}, start=CONCERN_ASKED
    )
    assert action == Finish(
        Assessment(
            outcome=Outcome.STANDARD,
            symptoms={
                "cough": Status.PRESENT,
                "cyanosis": Status.ABSENT,
                "long_cough": Status.ABSENT,
            },
            outcome_driven_by=("cough",),
        )
    )
    assert triage.stage is Stage.DONE


def test_failed_extractions_ask_to_rephrase_then_give_up() -> None:
    triage, _ = converse({})
    for _ in range(MAX_EXTRACTION_FAILURES):
        triage, action = ENGINE.fail(triage)
        assert action == Rephrase(PET)  # the question it couldn't read the answer to
        assert triage.last_asked == PET.id

    triage, action = ENGINE.fail(triage)
    assert action == FailSafe()


def test_a_successful_step_resets_the_failure_count() -> None:
    triage, _ = converse({})
    triage, _ = ENGINE.fail(triage)
    triage, _ = ENGINE.step(triage, A_DOG)
    assert triage.extraction_failures == 0


def test_a_clinical_question_the_owner_cannot_answer_stays_unknown() -> None:
    _, action = converse({**A_DOG, "coughing": True}, {}, {"blue_gums": False}, start=CONCERN_ASKED)
    assert isinstance(action, Finish)
    assert action.assessment.symptoms["long_cough"] is Status.UNKNOWN


def test_a_triage_survives_the_checkpoint_unchanged() -> None:
    # LangGraph saves the triage between messages; what it loads back must be equal.
    triage = Triage(
        answers={"coughing": True, "cough_days": 9.0, "swallowed": frozenset({"bone"})},
        stage=Stage.CLINICAL,
        last_asked="cough_days",
        asked=frozenset({"coughing", "cough_days"}),
        announced=Outcome.URGENT,
        extraction_failures=1,
    )
    serializer = JsonPlusSerializer()
    assert serializer.loads_typed(serializer.dumps_typed(triage)) == triage


def test_a_no_to_a_question_not_asked_is_ignored() -> None:
    triage, action = converse({**A_DOG, "coughing": True, "blue_gums": False}, start=CONCERN_ASKED)
    assert "blue_gums" not in triage.answers
    assert action == Ask(COUGH_DAYS, acknowledge=(COUGHING_PATHWAY,))


def test_a_yes_to_a_question_not_asked_is_kept() -> None:
    triage, action = converse({**A_DOG, "coughing": True, "blue_gums": True}, start=CONCERN_ASKED)
    assert triage.answers["blue_gums"] is True
    assert action == Ask(COUGH_DAYS, Outcome.IMMEDIATE, (COUGHING_PATHWAY,))
