"""The practice's records: a stand-in for its database.

The demo reads ``practice_db.json``, but through the same call a data-access layer would
make, so a real database replaces only the body of ``get_client_patients``. Rex belongs to
another client and must never appear in this client's call.
"""

import json
from pathlib import Path

from pet_triage.domain.patient import Patient

DATABASE = Path(__file__).with_name("practice_db.json")
"""The demo's records. In production, a connection to the practice management system."""


def get_client_patients(client_id: str) -> list[Patient]:
    """The patients registered to one client, and only to that client.

    The equivalent of ``SELECT * FROM patients WHERE client_id = :client_id``. Filtering by
    the signed-in client is the privacy guardrail: the assistant never sees, and so can't
    reveal or triage, another client's animals. It is called from code, not offered to the
    model as a tool, so the model has no say in whose records are read.

    Args:
        client_id: The signed-in client.
    """
    rows = json.loads(DATABASE.read_text())["patients"]
    return [Patient(**row) for row in rows if row["client_id"] == client_id]
