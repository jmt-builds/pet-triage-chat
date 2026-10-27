"""Questions we ask the owner that feed the gastrointestinal symptoms.

Wording is our draft and needs a vet nurse's sign-off. If another group comes to need one
of these observations, move it up a level rather than duplicating it.
"""

from pet_triage.clinical.gastrointestinal.ingested_items import KNOWN_ITEMS
from pet_triage.domain.observation import OTHER, Kind, Observation

VOMIT_COUNT_24H = Observation(
    "vomit_count_24h",
    "How many times has your pet been sick in the last 24 hours?",
    Kind.NUMBER,
)
BLACK_TARRY_STOOL = Observation(
    "black_tarry_stool", "Is your pet's poo black and tarry?", Kind.YES_NO
)
INGESTED = Observation(
    "ingested_items",
    "Has your pet eaten anything unusual, like an object or something that might be poisonous?",
    Kind.ITEMS,
    options=frozenset({*KNOWN_ITEMS, OTHER}),
)
"""The model maps the owner's words onto the known items ("a Galaxy bar" is chocolate);
anything not listed, like "a small piece of wood", is ``other``."""
HOURS_SINCE_INGESTION = Observation(
    "hours_since_ingestion",
    "When did your pet swallow the object? A day and rough time is enough, like "
    "'yesterday about 8pm'.",
    Kind.HOURS_AGO,
)
BELLY_SWOLLEN_QUICKLY = Observation(
    "belly_swollen_quickly",
    "Has your pet's belly swollen up quickly, or does it look tight and bloated?",
    Kind.YES_NO,
)
