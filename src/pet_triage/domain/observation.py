"""Observations: what we ask the owner, and the answers they give."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

Value = bool | float | str | frozenset[str]
"""An owner's answer to an observation. A missing answer means unknown.

A yes/no answer is a ``bool``, a number or hours ago a ``float``, a choice a ``str``, and an
items answer the set of item names given.
"""

Answers = Mapping[str, Value]
"""Answers keyed by observation ID."""

OTHER = "other"
"""The option given for anything not in an observation's list: an item we don't list, or a
complaint no pathway covers. The model may well know what it is; we just have no option
for it. What that means is decided in code, per observation."""


class Kind(StrEnum):
    """The type of answer an observation expects."""

    YES_NO = "yes_no"
    NUMBER = "number"
    HOURS_AGO = "hours_ago"
    """When something happened, as hours ago. The owner may give a day and a time instead."""
    CHOICE = "choice"
    ITEMS = "items"


@dataclass(frozen=True, slots=True)
class Observation:
    """Something we ask the owner.

    Attributes:
        id: Stable identifier, used as the key in ``Answers``.
        question: The exact text the owner reads.
        kind: The type of answer expected.
        options: Allowed answers for a ``CHOICE`` or ``ITEMS`` observation; empty otherwise.
            The model must map the owner's words onto these.
    """

    id: str
    question: str
    kind: Kind
    options: frozenset[str] = frozenset()


def expect[T](answer: Value, expected: type[T], observation: Observation) -> T:
    """Return the answer as the type its observation expects, or raise ``TypeError``.

    A wrongly typed answer is a bug in the extractor, so it fails loudly instead of being
    misread.
    """
    if not isinstance(answer, expected):
        raise TypeError(f"{observation.id!r} expects {expected.__name__}, got {answer!r}")
    return answer
