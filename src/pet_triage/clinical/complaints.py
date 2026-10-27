"""The owner's concern, read as the complaints it names: how a call finds its pathways.

Each option is one pathway's ID, so the model maps "she's been chucking up" onto
``vomiting``. Anything else the owner describes, like a limp or a collapse, is ``other``,
and the call is out of scope even if a pathway also applies: the demo never triages part
of a problem. Adding a pathway means adding its ID here.
"""

from pet_triage.domain.observation import OTHER, Kind, Observation

COMPLAINTS = Observation(
    "complaints",
    "What's going on with your pet today?",
    Kind.ITEMS,
    options=frozenset({"vomiting", OTHER}),
)
"""Asked once the patient questions are answered, even if the owner already said, so the
reply is read knowing the question."""
