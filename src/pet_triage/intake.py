"""Intake: which of the client's animals the call is about, and which species are triaged.

This is application behaviour, not clinical content, so it lives outside the governed
``clinical`` package.
"""

from collections.abc import Sequence

from pet_triage.domain.observation import OTHER, Kind, Observation
from pet_triage.domain.patient import Patient

SUPPORTED_SPECIES = frozenset({"dog"})
"""Species the demo triages; a call about any other animal is out of scope."""

PET_ID = "pet"
"""The observation ID of the pet question, whose answer is the chosen pet's name."""


def pet_observation(patients: Sequence[Patient]) -> Observation:
    """The question asking which of the client's pets the call is about.

    Its options are the pets' names, and ``other`` for an animal not on the account. The
    question names each pet's species and breed, so both the owner and the model can tell
    them apart ("the lab").

    Args:
        patients: The client's registered animals, at least one.
    """
    described = [f"{p.name} ({p.species}, {p.breed})" for p in patients]
    *rest, last = described
    listed = f"{', '.join(rest)} or {last}" if rest else last
    return Observation(
        PET_ID,
        f"Which pet are you asking about today: {listed}?",
        Kind.CHOICE,
        options=frozenset({*(p.name for p in patients), OTHER}),
    )
