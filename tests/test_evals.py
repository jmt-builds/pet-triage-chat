"""The extraction evaluation's cases and scorers, checked without a model."""

from typing import Any

import pytest
import yaml
from langchain_core.messages import AIMessage, HumanMessage

from evals.extraction import (
    CASES,
    RUNS,
    invented,
    load_cases,
    messages,
    missing,
    overwritten,
    typed,
    under_triage,
    wrong,
)


def test_every_case_loads_with_real_fields_and_unique_ids() -> None:
    cases = yaml.safe_load(CASES.read_text())["cases"]
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids))
    for case in cases:
        typed(case.get("before", {}))  # raises KeyError on a field that isn't an observation
        messages(case.get("history", []))  # raises ValueError on an unknown speaker
    assert len(load_cases()) == len(cases) * RUNS


def turn(before: dict[str, Any], got: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    """Scorer arguments for one turn."""
    return {
        "inputs": {"before": before},
        "outputs": {"answers": got, "urgency": 0},
        "expectations": {"answers": expected, "urgency": 0},
    }


def test_each_field_error_is_counted_once() -> None:
    args = turn(
        before={"vomit_count_24h": 3},
        got={"vomit_count_24h": 1, "sex": "male", "belly_swollen_quickly": False},
        expected={"vomit_count_24h": 3, "sex": "female", "vomiting": True},
    )
    records = {"outputs": args["outputs"], "expectations": args["expectations"]}
    assert overwritten(**args) == 1  # the count was right and this turn changed it
    assert wrong(**args) == 1  # sex
    assert missing(**records) == 1  # vomiting
    assert invented(**records) == 1  # belly


def test_lower_urgency_than_expected_is_under_triage() -> None:
    assert under_triage(
        outputs={"answers": {}, "urgency": 1}, expectations={"answers": {}, "urgency": 3}
    )


def test_history_entries_become_owner_and_line_messages() -> None:
    history = messages([{"owner": "she's sick"}, {"line": "How many times?"}])
    assert [type(message) for message in history] == [HumanMessage, AIMessage]
    assert [message.text for message in history] == ["she's sick", "How many times?"]


def test_a_history_entry_with_an_unknown_speaker_fails_loudly() -> None:
    with pytest.raises(ValueError, match="Unknown speaker"):
        messages([{"vet": "hello"}])
