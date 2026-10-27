"""Pathways: presenting complaints and the symptoms they activate."""

from dataclasses import dataclass

from pet_triage.domain.observation import Observation
from pet_triage.domain.symptom import Symptom


@dataclass(frozen=True, slots=True)
class Pathway:
    """A presenting complaint and the follow-up symptoms it activates.

    Its baseline outcome is the presenting symptom's, so there is no separate default.

    Attributes:
        id: Stable identifier, cited in the triage record.
        presenting: The symptom that activates the pathway, taken from the owner's concern.
        follow_ups: The symptoms assessed once the pathway is active.
        acknowledgement: What the owner is told when the pathway becomes active, before its
            first question: one sentence, written with "your pet" so the pet's name can be
            put in (see ``pet_triage.messages.render``).
    """

    id: str
    presenting: Symptom
    follow_ups: tuple[Symptom, ...]
    acknowledgement: str

    @property
    def symptoms(self) -> tuple[Symptom, ...]:
        """The presenting symptom followed by the follow-ups."""
        return (self.presenting, *self.follow_ups)

    @property
    def observations(self) -> tuple[Observation, ...]:
        """Every observation the pathway's symptoms read, in order, without duplicates."""
        return tuple(dict.fromkeys(symptom.when.observation for symptom in self.symptoms))
