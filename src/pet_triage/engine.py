"""The triage engine: a pure step from one message's answers to the next action.

The engine runs the conversation as a state machine: which of the client's animals the
call is about, the owner's concern (always asked, and once more if it's unclear), clinical
questions, then done. What the answers mean comes from ``assessment``. It knows nothing
about the model or how answers were extracted, and it is given its clinical content rather
than importing it. The graph calls ``step`` or ``fail`` and renders the ``NextAction``
returned.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from itertools import chain

from pet_triage.assessment import active_pathways, assess, next_question
from pet_triage.domain.observation import OTHER, Answers, Observation, expect
from pet_triage.domain.outcome import Outcome
from pet_triage.domain.pathway import Pathway
from pet_triage.domain.patient import Patient
from pet_triage.record import Assessment

MAX_EXTRACTION_FAILURES = 2
"""Failed extractions in a row that get a request to rephrase; one more gets the fail-safe."""


class Stage(Enum):
    """Where the conversation is.

    Stages follow the answers forward. A concern no pathway covers is asked about once
    more (``CLARIFY``) before the call is out of scope, as a person on the phone would.
    A message after ``DONE`` is judged like any other, so new information is never
    ignored.
    """

    PATIENT = "patient"
    CONCERN = "concern"
    CLARIFY = "clarify"
    CLINICAL = "clinical"
    DONE = "done"


@dataclass(frozen=True, slots=True)
class Triage:
    """A triage in progress, filled in turn by turn and kept by the graph's checkpointer.

    It holds only the answers and the conversation's progress. Symptom statuses and the
    outcome are always calculated from the answers, never stored beside them, so they can't
    disagree. When the questions run out, the engine produces an ``Assessment`` and the
    graph turns the finished triage into a ``TriageRecord``.

    Attributes:
        answers: Every accepted answer so far, keyed by observation ID.
        stage: Where the conversation is.
        last_asked: ID of the question the owner is replying to, so the extractor has its
            context.
        asked: IDs of every observation asked so far. Each is asked once: if the owner can't
            answer, it stays unanswered and the conversation moves on.
        announced: The highest outcome already told to the owner.
        acknowledged: IDs of the pathways already acknowledged to the owner.
        extraction_failures: Failed extractions in a row.
    """

    answers: Answers = field(default_factory=dict)
    stage: Stage = Stage.PATIENT
    last_asked: str | None = None
    asked: frozenset[str] = frozenset()
    announced: Outcome | None = None
    acknowledged: frozenset[str] = frozenset()
    extraction_failures: int = 0


@dataclass(frozen=True, slots=True)
class Ask:
    """Ask the owner one observation's question.

    Attributes:
        observation: What to ask.
        announce: The outcome to announce first, set only when it has just risen to
            urgent or above.
        acknowledge: The pathways to acknowledge first: those active and not acknowledged
            yet, set only on a pathway's question, so each is acknowledged once.
    """

    observation: Observation
    announce: Outcome | None = None
    acknowledge: tuple[Pathway, ...] = ()


@dataclass(frozen=True, slots=True)
class Rephrase:
    """The owner's message couldn't be read or was unclear: ask again.

    Attributes:
        observation: The question to ask again, or ``None`` if nothing had been asked yet.
    """

    observation: Observation | None = None


class Unsupported(Enum):
    """Why a call is out of scope, so the owner is told the reason that applies to them."""

    PET = "pet"
    """An animal that isn't on the client's account."""
    SPECIES = "species"
    """One of the client's animals, of a species the demo doesn't triage."""
    CONCERN = "concern"
    """A complaint no pathway covers."""


@dataclass(frozen=True, slots=True)
class OutOfScope:
    """Explain that the call isn't supported.

    Attributes:
        reason: Why not.
    """

    reason: Unsupported = Unsupported.CONCERN


@dataclass(frozen=True, slots=True)
class Finish:
    """Every question is answered: give the closing advice and write the record."""

    assessment: Assessment


@dataclass(frozen=True, slots=True)
class FailSafe:
    """Extraction kept failing: tell the owner to contact their vet."""


NextAction = Ask | Rephrase | OutOfScope | Finish | FailSafe


class TriageEngine:
    """Runs a call, one owner message at a time.

    Created once at start-up from question definitions, not answers: the graph passes the
    real ones, tests fake ones.

    Args:
        pathways: Every supported pathway.
        pet: The question asking which of the client's animals the call is about. Its
            answer is a pet's name, or ``OTHER``.
        patients: The client's animals, keyed by name. With only one, it is chosen
            without asking.
        supported_species: Species to triage; an animal of any other is out of scope.
        concern: The observation the owner's concern is read into: the complaints it
            names, one option per pathway. ``OTHER`` among them is out of scope.
    """

    def __init__(
        self,
        pathways: tuple[Pathway, ...],
        pet: Observation,
        patients: Mapping[str, Patient],
        supported_species: frozenset[str],
        concern: Observation,
    ) -> None:
        """Create an engine over the given question definitions."""
        self._pathways = pathways
        self._pet = pet
        self._patients = patients
        self._only_pet = {pet.id: next(iter(patients))} if len(patients) == 1 else {}
        self._supported_species = supported_species
        self._concern = concern
        self._by_id = {observation.id: observation for observation in self.observations}

    @property
    def observations(self) -> tuple[Observation, ...]:
        """Every observation, which the extractor looks for on every turn.

        The pet question, the concern and every pathway's observations, without duplicates.
        Looking for all of them means nothing the owner volunteers is dropped, whatever is
        being asked, and lets a later message correct an earlier answer.
        """
        pathway_observations = (pathway.observations for pathway in self._pathways)
        return tuple(dict.fromkeys(chain((self._pet, self._concern), *pathway_observations)))

    def in_scope(self, answers: Answers) -> bool:
        """Whether the demo can triage this call, as far as the answers so far show."""
        return self._unsupported(answers) is None

    def patient(self, answers: Answers) -> Patient | None:
        """The client's animal the call is about, or ``None`` until one on the account is chosen."""
        name = {**self._only_pet, **answers}.get(self._pet.id)
        if name is None:
            return None
        return self._patients.get(expect(name, str, self._pet))

    def step(self, triage: Triage, answers: Answers) -> tuple[Triage, NextAction]:
        """Add one message's answers and decide what to do next.

        A later answer to the same question replaces the earlier one, so the owner can
        correct themselves. A "no" counts only for the question just asked (see
        ``_accepted``).

        Args:
            triage: The call so far.
            answers: What the owner's latest message said, keyed by observation ID.

        Returns:
            The updated triage and the action for the reply node to render.
        """
        answers = {**self._only_pet, **triage.answers, **_accepted(answers, triage.last_asked)}
        triage = replace(triage, answers=answers, extraction_failures=0)
        assessment = assess(self._pathways, triage.answers)
        announce = _newly_urgent(assessment, triage.announced)
        stage, action = self._decide(triage, assessment, announce)
        asking = action.observation if isinstance(action, Ask | Rephrase) else None
        question = None if asking is None else asking.id
        updated = replace(
            triage,
            stage=stage,
            last_asked=question,
            asked=triage.asked if question is None else triage.asked | {question},
            announced=announce or triage.announced,
            acknowledged=triage.acknowledged | {pathway.id for pathway in _acknowledging(action)},
        )
        return updated, action

    def fail(self, triage: Triage) -> tuple[Triage, NextAction]:
        """Record a failed extraction: ask the owner to rephrase, or give up after too many.

        Args:
            triage: The call so far.

        Returns:
            The updated triage and the action for the reply node to render.
        """
        failures = triage.extraction_failures + 1
        if failures > MAX_EXTRACTION_FAILURES:
            return replace(triage, extraction_failures=failures), FailSafe()
        question = None if triage.last_asked is None else self._by_id[triage.last_asked]
        return replace(triage, extraction_failures=failures), Rephrase(question)

    def _decide(
        self,
        triage: Triage,
        assessment: Assessment | None,
        announce: Outcome | None,
    ) -> tuple[Stage, NextAction]:
        """The transition: the next stage and what to say, in order of priority.

        ``triage`` already holds this message's answers but still the previous stage. A
        stage with nothing left to ask passes straight on to the next, so answers the owner
        volunteered early never hold the conversation up.
        """
        answers = triage.answers
        unsupported = self._unsupported(answers)
        if unsupported is not None:
            return Stage.DONE, OutOfScope(unsupported)

        # Asked until answered: nothing can be triaged without knowing which animal
        if self._pet.id not in answers:
            return Stage.PATIENT, Ask(self._pet, announce)

        # Asked even if the owner volunteered it, so it is read knowing the question
        if self._concern.id not in triage.asked:
            return Stage.CONCERN, Ask(self._concern, announce)

        if assessment is None:
            return self._without_pathway(triage.stage)

        clinical_question = next_question(self._pathways, answers, triage.asked)
        if clinical_question is not None:
            return Stage.CLINICAL, Ask(clinical_question, announce, self._unacknowledged(triage))

        return Stage.DONE, Finish(assessment)

    def _unsupported(self, answers: Answers) -> Unsupported | None:
        """Why the call is out of scope, or ``None`` while it is in scope.

        A complaint no pathway covers puts the call out of scope even beside one that is
        covered: the demo never triages part of a problem.
        """
        name = answers.get(self._pet.id)
        if name is not None:
            patient = self._patients.get(expect(name, str, self._pet))
            if patient is None:
                return Unsupported.PET
            if patient.species not in self._supported_species:
                return Unsupported.SPECIES
        complaints = answers.get(self._concern.id)
        if complaints is not None and OTHER in expect(complaints, frozenset, self._concern):
            return Unsupported.CONCERN
        return None

    def _unacknowledged(self, triage: Triage) -> tuple[Pathway, ...]:
        """The active pathways the owner hasn't been told about yet, in ``PATHWAYS`` order."""
        active = active_pathways(self._pathways, triage.answers)
        return tuple(pathway for pathway in active if pathway.id not in triage.acknowledged)

    def _without_pathway(self, stage: Stage) -> tuple[Stage, NextAction]:
        """Once the concern is asked but no pathway is active: ask it again, then stop."""
        if stage is Stage.CONCERN:
            return Stage.CLARIFY, Rephrase(self._concern)
        return Stage.DONE, OutOfScope()


def _accepted(answers: Answers, last_asked: str | None) -> Answers:
    """The answers to keep: anything stated, but a "no" only for the question just asked.

    A model can turn "not mentioned" into "no", and a "no" lowers urgency and stops the
    question being asked. So, as with outcomes, the model may escalate but never
    de-escalate: an unasked "no" is dropped and the question is asked in its turn. The
    cost is that an owner who volunteers a "no" is asked again.
    """
    return {key: value for key, value in answers.items() if value is not False or key == last_asked}


def _acknowledging(action: NextAction) -> tuple[Pathway, ...]:
    """The pathways this action acknowledges to the owner."""
    return action.acknowledge if isinstance(action, Ask) else ()


def _newly_urgent(assessment: Assessment | None, announced: Outcome | None) -> Outcome | None:
    """The outcome to announce: set only when it is urgent or above and higher than before."""
    if assessment is None or assessment.outcome < Outcome.URGENT:
        return None
    if announced is not None and assessment.outcome <= announced:
        return None
    return assessment.outcome
