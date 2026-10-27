"""Supported pathways and the registry the rest of the system reads.

Adding a pathway means defining it here (from symptoms in the group packages), adding it to
``PATHWAYS``; owners are told about it automatically (``messages.CONCERNS_COVERED``).
No code changes.
"""

from pet_triage.clinical.gastrointestinal import symptoms as gi
from pet_triage.domain.pathway import Pathway

VOMITING = Pathway(
    "vomiting",
    presenting=gi.VOMITING,
    follow_ups=(
        gi.PERSISTENT_VOMITING,
        gi.MELENA,
        gi.POSSIBLE_FOREIGN_BODY,
        gi.FOREIGN_BODY_OVER_24H,
        gi.TOXIN_INGESTION,
        gi.ABDOMINAL_DISTENTION,
    ),
    acknowledgement="I'm sorry to hear your pet has been sick.",
)

PATHWAYS = (VOMITING,)
"""Every supported pathway."""
