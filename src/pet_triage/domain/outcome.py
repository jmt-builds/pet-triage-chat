"""The Veterinary Triage List (VTL), the scale every outcome is measured on."""

from enum import Enum
from functools import total_ordering


@total_ordering
class Outcome(Enum):
    """A VTL level. Members are declared from least to most urgent, and that order is the rank.

    The source figure and the VTL label levels by colour; the code uses the urgency:

    ============  ======  =====================
    Member        Colour  Maximum wait in clinic
    ============  ======  =====================
    NON_URGENT    blue    240 minutes
    STANDARD      green   120 minutes
    URGENT        yellow  60 minutes
    VERY_URGENT   orange  15 minutes
    IMMEDIATE     red     0 minutes
    ============  ======  =====================

    Values are plain strings, so records and the model's structured output read
    ``"urgent"``, not a number. It is an ``Enum`` rather than a ``StrEnum`` because string
    comparison would order levels alphabetically. What the owner is told for each level
    belongs to ``pet_triage.messages``.
    """

    NON_URGENT = "non_urgent"
    STANDARD = "standard"
    URGENT = "urgent"
    VERY_URGENT = "very_urgent"
    IMMEDIATE = "immediate"

    @property
    def rank(self) -> int:
        """Position on the scale, from 0 for the least urgent."""
        return list(Outcome).index(self)

    def __lt__(self, other: object) -> bool:
        """Order outcomes by rank, so ``max()`` returns the most urgent."""
        if not isinstance(other, Outcome):
            return NotImplemented
        return self.rank < other.rank
