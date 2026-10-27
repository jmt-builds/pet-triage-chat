"""Conditions: tests on one observation's answer that decide whether a symptom is present.

Thresholds and item lists are applied here, in code, never by the model. A condition only
judges an answer that exists; ``Symptom.status`` deals with unanswered observations.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from pet_triage.domain.observation import OTHER, Observation, Value, expect


class Status(StrEnum):
    """Whether a symptom is present, given the answers so far."""

    PRESENT = "present"
    ABSENT = "absent"
    UNKNOWN = "unknown"


def _present_if(present: bool) -> Status:
    """``PRESENT`` if ``present`` is true, otherwise ``ABSENT``."""
    return Status.PRESENT if present else Status.ABSENT


class Condition(Protocol):
    """A test on one observation's answer."""

    @property
    def observation(self) -> Observation:
        """The observation this condition reads."""
        ...

    def judge(self, answer: Value) -> Status:
        """Whether the answer makes the symptom present, absent or still unknown."""
        ...


@dataclass(frozen=True, slots=True)
class Yes:
    """Present when a yes/no observation is answered yes."""

    observation: Observation

    def judge(self, answer: Value) -> Status:
        """Present if the owner answered yes, otherwise absent."""
        return _present_if(expect(answer, bool, self.observation))


@dataclass(frozen=True, slots=True)
class AtLeast:
    """Present when a number observation meets a threshold."""

    observation: Observation
    threshold: float

    def judge(self, answer: Value) -> Status:
        """Present if the number is at or above the threshold, otherwise absent."""
        return _present_if(expect(answer, float, self.observation) >= self.threshold)


@dataclass(frozen=True, slots=True)
class AnyOf:
    """Present when any item in an items observation is one of ``items``.

    The model has already mapped the owner's words onto the observation's options ("a
    Galaxy bar" became ``chocolate``) when the extractor read the message, so this only
    compares names.
    """

    observation: Observation
    items: frozenset[str]

    def judge(self, answer: Value) -> Status:
        """Present if any item matches.

        Otherwise unknown if an item isn't in the list (``OTHER``), since we can't tell
        whether it matches, and absent if every item is listed and none match. An unlisted
        item never escalates on its own; the other questions decide how serious things are.
        """
        answered = expect(answer, frozenset, self.observation)
        if answered & self.items:
            return Status.PRESENT
        if OTHER in answered:
            return Status.UNKNOWN
        return Status.ABSENT
