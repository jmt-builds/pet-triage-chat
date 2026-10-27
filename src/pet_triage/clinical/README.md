# Clinical content

Everything in this package is clinically governed: any change needs sign-off from a
qualified professional. It holds data only, built from the types in
`pet_triage.domain`; the engine applies it, and nothing here decides anything on its own.

The source is figure 1 of Thomovsky E, Ilie L. "Basic triage in dogs and cats: Part I."
*Can Vet J.* 2024 ([PMC10783570](https://pmc.ncbi.nlm.nih.gov/articles/PMC10783570/)). The
figure sorts presenting signs into eight body-system groups and assigns each sign a
Veterinary Triage List level (`Outcome`).

## Body-system groups

| Group | Covers | Status |
|---|---|---|
| `gastrointestinal` | Vomiting, ingestion of toxins or foreign bodies, gut bleeding, abdominal swelling | Implemented: the demo's vomiting pathway |
| `respiratory` | Breathing difficulty, noisy breathing, coughing | Pending |
| `circulatory` | Shock, bleeding, heart problems | Pending |
| `neurologic` | Seizures, collapse, changes in consciousness or behaviour | Pending |
| `obstetrical` | Pregnancy, labour and birth | Pending |
| `generalized` | Whole-body signs: temperature, blood sugar, pain, dehydration | Pending |
| `trauma` | Injuries, wounds, fractures, envenomation | Pending |
| `urogenital` | Urinary and reproductive problems | Pending |

## Layout

```
clinical/
  complaints.py          the complaints a concern can name: one per pathway, or other
  pathways.py            which symptoms each presenting complaint activates
  <group>/
    observations.py      the questions the group's symptoms read
    symptoms.py          the figure's criteria for the group, with thresholds and outcomes
```

Anything only one group uses stays in that group's package, such as
`gastrointestinal/ingested_items.py`. Content moves up a level only when a second group
needs it.

## Adding a group

1. **Questions.** In `<group>/observations.py`, define an `Observation` for each thing the
   owner must be asked, in words an owner can answer on the phone.
2. **Symptoms.** In `<group>/symptoms.py`, define a `Symptom` for each of the figure's
   criteria in the group: its ID in the figure's wording, its `Outcome`, and the
   `Condition` that decides it. Put thresholds in named constants with a docstring.
3. **Pathway.** A group adds no scope by itself. In `pathways.py`, define a `Pathway` with
   the presenting symptom and its follow-ups (they may come from any group), and add it to
   `PATHWAYS`. Add the pathway's ID to the options of `COMPLAINTS` in `complaints.py`, and
   make its presenting symptom `AnyOf(COMPLAINTS, {<its ID>})`. Any complaint without a
   pathway is `other`, which puts the call out of scope. Owners are shown the pathway IDs
   as the concerns the demo covers, so name it in plain words. Give it an
   `acknowledgement`: one sentence the owner reads before its first question, written with
   "your pet" (e.g. "I'm sorry to hear your pet has been sick.").
4. **Test.** Add a test that the pathway's symptoms and outcomes match the figure, like
   `test_vomiting_pathway_matches_figure`.
5. **Sign-off.** A vet professional reviews the questions, thresholds and outcomes.

No engine, graph or extractor code changes.
