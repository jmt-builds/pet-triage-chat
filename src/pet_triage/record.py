"""The triage record: the artifact handed to the vet.

Artifacts are the only pydantic models in the codebase. Everything else is a dataclass;
artifacts leave the system, so they are validated when built and serialised to JSON.
The graph's ``record`` node adds the finished record to the conversation's state.
"""

from pydantic import BaseModel, ConfigDict

from pet_triage.domain.condition import Status
from pet_triage.domain.observation import Value
from pet_triage.domain.outcome import Outcome
from pet_triage.domain.patient import Patient


class Audit(BaseModel):
    """How a record was produced, so any outcome can be traced back.

    Attributes:
        model: The LLM tag used.
        thinking: Whether the model thought before answering.
        prompt_version: The extraction prompt's version.
        thread_id: The conversation's LangGraph thread. MLflow uses it as the session ID, so
            it finds every trace of the conversation (one per turn).
    """

    model_config = ConfigDict(frozen=True)

    model: str
    thinking: bool
    prompt_version: str
    thread_id: str


class Assessment(BaseModel):
    """A triage judgement: the outcome, the symptom statuses, and which symptoms set it.

    Attributes:
        outcome: The triage level.
        symptoms: Each assessed symptom's status, keyed by symptom ID.
        outcome_driven_by: IDs of the present symptoms that set the outcome.
    """

    model_config = ConfigDict(frozen=True)

    outcome: Outcome
    symptoms: dict[str, Status]
    outcome_driven_by: tuple[str, ...]


class TriageRecord(BaseModel):
    """Everything the vet needs when the patient arrives.

    Attributes:
        patient: The animal being triaged. First, so the vet reads it first.
        record_id: Unique identifier for this record.
        assessment: The final outcome, the symptom statuses, and which symptoms set it.
        observations: Every clinical observation's answer; missing means unknown. Patient
            answers are in ``patient`` only.
        owner_messages: What the owner typed, in order.
        audit: How the record was produced.
    """

    model_config = ConfigDict(frozen=True)

    patient: Patient
    record_id: str
    assessment: Assessment
    observations: dict[str, Value]
    owner_messages: tuple[str, ...]
    audit: Audit
