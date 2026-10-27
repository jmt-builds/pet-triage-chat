"""The triage domain model: types and pure logic, with no clinical data and no LangChain.

The model has three layers:

* An ``Observation`` is something we ask the owner.
* A ``Symptom`` is a clinical criterion with an outcome. It is defined by a ``Condition`` on
  one observation.
* A ``Pathway`` is a presenting complaint: a presenting symptom plus the follow-up
  symptoms it activates.

The clinical content built from these types lives in ``pet_triage.clinical``.
"""
