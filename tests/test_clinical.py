"""The clinical content must match the source figure and handle uncertain answers safely."""

from pet_triage.clinical.complaints import COMPLAINTS
from pet_triage.clinical.pathways import PATHWAYS, VOMITING
from pet_triage.domain.condition import Status
from pet_triage.domain.observation import OTHER, Answers
from pet_triage.domain.outcome import Outcome

FIGURE_GASTROINTESTINAL = {
    "vomiting": Outcome.STANDARD,
    "persistent_vomiting": Outcome.URGENT,
    "melena": Outcome.URGENT,
    "possible_foreign_body": Outcome.URGENT,
    "foreign_body_over_24h": Outcome.VERY_URGENT,
    "toxin_ingestion": Outcome.VERY_URGENT,
    "abdominal_distention": Outcome.IMMEDIATE,
}


def test_vomiting_pathway_matches_figure() -> None:
    assert {s.id: s.outcome for s in VOMITING.symptoms} == FIGURE_GASTROINTESTINAL


def test_an_unlisted_item_is_logged_as_unknown_without_escalating() -> None:
    answers: Answers = {"complaints": frozenset({"vomiting"}), "ingested_items": frozenset({OTHER})}
    statuses = {symptom.id: symptom.status(answers) for symptom in VOMITING.symptoms}
    assert statuses["toxin_ingestion"] is Status.UNKNOWN
    assert statuses["possible_foreign_body"] is Status.UNKNOWN
    assert statuses["foreign_body_over_24h"] is Status.UNKNOWN
    present = [s.outcome for s in VOMITING.symptoms if statuses[s.id] is Status.PRESENT]
    assert max(present) is Outcome.STANDARD


def test_every_pathway_has_a_complaint_and_every_complaint_a_pathway() -> None:
    assert COMPLAINTS.options == {pathway.id for pathway in PATHWAYS} | {OTHER}
    for pathway in PATHWAYS:
        assert pathway.presenting.when.observation is COMPLAINTS
