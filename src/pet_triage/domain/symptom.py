"""Symptoms: clinical criteria, each with the outcome it triages to."""

from __future__ import annotations

from dataclasses import dataclass

from pet_triage.domain.condition import Condition, Status
from pet_triage.domain.observation import Answers
from pet_triage.domain.outcome import Outcome


@dataclass(frozen=True, slots=True)
class Symptom:
    """A clinical criterion from the source figure, with its outcome.

    Its body-system group is the ``pet_triage.clinical`` package it is defined in.

    Attributes:
        id: Stable identifier, cited in the triage record.
        outcome: The urgency the figure assigns.
        when: The condition that decides whether it is present.
        requires: A symptom that must be present before this one can be.
    """

    id: str
    outcome: Outcome
    when: Condition
    requires: Symptom | None = None

    def status(self, answers: Answers) -> Status:
        """Whether the symptom is present, given the answers so far.

        A symptom whose prerequisite isn't present takes the prerequisite's status: with no
        foreign body there can't be one for over 24 hours, and while the foreign body is
        unknown so is its duration.
        """
        if self.requires is not None:
            prerequisite = self.requires.status(answers)
            if prerequisite is not Status.PRESENT:
                return prerequisite

        answer = answers.get(self.when.observation.id)
        if answer is None:
            return Status.UNKNOWN
        return self.when.judge(answer)

    def prerequisite_met(self, answers: Answers) -> bool:
        """Whether the prerequisite, if there is one, is present, so this symptom can apply."""
        return self.requires is None or self.requires.status(answers) is Status.PRESENT
