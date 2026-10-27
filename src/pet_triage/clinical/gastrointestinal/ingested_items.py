"""What a dog might have eaten, grouped by what it means for triage.

Illustrative until reviewed by a vet nurse. The model only maps the owner's words onto
these names (the ``INGESTED`` observation's options); which set a name is in decides the
outcome, in code.

Only items that change the triage are listed. Safe things a dog might eat are too many to
list, and listing them would change nothing: anything unlisted is ``other``, which records
the toxin and foreign-body symptoms as unknown without escalating, and the vet reads what
it was in the owner's messages on the record.
"""

TOXINS = frozenset({"chocolate", "grapes", "raisins", "xylitol"})
FOREIGN_BODIES = frozenset({"sock", "toy", "bone"})

# Every item the model can name; anything else is ``other``.
KNOWN_ITEMS = TOXINS | FOREIGN_BODIES
