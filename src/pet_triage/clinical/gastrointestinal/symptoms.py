"""Gastrointestinal symptoms and their thresholds, with the outcomes the figure assigns.

Transcribed from the gastrointestinal group of figure 1 in Thomovsky E, Ilie L. "Basic
triage in dogs and cats: Part I." Can Vet J. 2024 (PMC10783570). Symptom IDs follow the
figure's wording.
"""

from pet_triage.clinical.complaints import COMPLAINTS
from pet_triage.clinical.gastrointestinal.ingested_items import FOREIGN_BODIES, TOXINS
from pet_triage.clinical.gastrointestinal.observations import (
    BELLY_SWOLLEN_QUICKLY,
    BLACK_TARRY_STOOL,
    HOURS_SINCE_INGESTION,
    INGESTED,
    VOMIT_COUNT_24H,
)
from pet_triage.domain.condition import AnyOf, AtLeast, Yes
from pet_triage.domain.outcome import Outcome
from pet_triage.domain.symptom import Symptom

PERSISTENT_VOMITING_THRESHOLD = 3
"""Vomits in 24 hours that count as persistent."""

FOREIGN_BODY_HOURS = 24
"""From the figure: a possible foreign body becomes very urgent after 24 hours."""

VOMITING = Symptom("vomiting", Outcome.STANDARD, AnyOf(COMPLAINTS, frozenset({"vomiting"})))
PERSISTENT_VOMITING = Symptom(
    "persistent_vomiting",
    Outcome.URGENT,
    AtLeast(VOMIT_COUNT_24H, PERSISTENT_VOMITING_THRESHOLD),
)
MELENA = Symptom("melena", Outcome.URGENT, Yes(BLACK_TARRY_STOOL))
POSSIBLE_FOREIGN_BODY = Symptom(
    "possible_foreign_body",
    Outcome.URGENT,
    AnyOf(INGESTED, FOREIGN_BODIES),
)
FOREIGN_BODY_OVER_24H = Symptom(
    "foreign_body_over_24h",
    Outcome.VERY_URGENT,
    AtLeast(HOURS_SINCE_INGESTION, FOREIGN_BODY_HOURS),
    requires=POSSIBLE_FOREIGN_BODY,
)
TOXIN_INGESTION = Symptom(
    "toxin_ingestion",
    Outcome.VERY_URGENT,
    AnyOf(INGESTED, TOXINS),
)
ABDOMINAL_DISTENTION = Symptom(
    "abdominal_distention",
    Outcome.IMMEDIATE,
    Yes(BELLY_SWOLLEN_QUICKLY),
)
