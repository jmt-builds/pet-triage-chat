"""Clinical judgement: what the answers so far mean, and what to ask next.

Pure functions of the supported pathways and the answers. They hold no state and know
nothing about the conversation, which ``engine`` runs on top of them.
"""

from itertools import chain
from operator import attrgetter

from pet_triage.domain.condition import Status
from pet_triage.domain.observation import Answers, Observation
from pet_triage.domain.pathway import Pathway
from pet_triage.domain.symptom import Symptom
from pet_triage.record import Assessment


def active_pathways(pathways: tuple[Pathway, ...], answers: Answers) -> tuple[Pathway, ...]:
    """The pathways whose presenting symptom is present. More than one can be active."""
    return tuple(
        pathway for pathway in pathways if pathway.presenting.status(answers) is Status.PRESENT
    )


def assess(pathways: tuple[Pathway, ...], answers: Answers) -> Assessment | None:
    """Judge the answers so far, or return ``None`` if no pathway is active yet.

    The outcome is the most urgent present symptom's. Every active symptom's status is
    reported, unknowns included, so the vet sees what is still open.
    """
    symptoms = _active_symptoms(pathways, answers)
    if not symptoms:
        return None

    statuses = {symptom.id: symptom.status(answers) for symptom in symptoms}
    present = [symptom for symptom in symptoms if statuses[symptom.id] is Status.PRESENT]
    # Never empty: an active pathway's presenting symptom is present by definition.
    outcome = max(symptom.outcome for symptom in present)
    return Assessment(
        outcome=outcome,
        symptoms=statuses,
        outcome_driven_by=tuple(symptom.id for symptom in present if symptom.outcome is outcome),
    )


def next_question(
    pathways: tuple[Pathway, ...], answers: Answers, asked: frozenset[str]
) -> Observation | None:
    """The next clinical question to ask, or ``None`` once there is nothing left to ask.

    Questions go from least to most urgent, so the owner isn't frightened early; this sort
    is the one place to change that. Anything already answered (including what the owner
    volunteered) or already asked is skipped, as is anything whose prerequisite isn't
    present. Asking each question once means an owner who can't answer isn't asked again;
    the symptom stays unknown for the vet.

    Args:
        pathways: Every supported pathway.
        answers: The answers so far.
        asked: IDs of the observations already asked.
    """
    for symptom in sorted(_active_symptoms(pathways, answers), key=attrgetter("outcome")):
        observation_id = symptom.when.observation.id
        if observation_id in answers or observation_id in asked:
            continue
        if not symptom.prerequisite_met(answers):
            continue
        return symptom.when.observation
    return None


def _active_symptoms(pathways: tuple[Pathway, ...], answers: Answers) -> tuple[Symptom, ...]:
    """Every symptom of the active pathways, in order, without duplicates."""
    active = active_pathways(pathways, answers)
    return tuple(dict.fromkeys(chain.from_iterable(pathway.symptoms for pathway in active)))
