"""The evaluation's baseline, driven by a scripted chat model instead of Ollama."""

import json
from datetime import datetime

import pytest
from langchain_core.messages import AIMessage

from evals.baseline import BaselineJudge, Judgement, describe_criteria
from fakes import ScriptedChatModel
from pet_triage.clinical.pathways import PATHWAYS
from pet_triage.config import Settings
from pet_triage.domain.outcome import Outcome
from pet_triage.graph import make_engine


def test_the_prompt_states_every_criterion_the_engine_applies() -> None:
    criteria = describe_criteria(PATHWAYS)
    for symptom in (symptom for pathway in PATHWAYS for symptom in pathway.symptoms):
        assert f"- {symptom.id}: " in criteria
    assert "- persistent_vomiting: vomit_count_24h is at least 3 -> urgent" in criteria
    assert (
        "- foreign_body_over_24h: hours_since_ingestion is at least 24 hours ago, "
        "and possible_foreign_body is met -> very_urgent"
    ) in criteria


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ({"outcome": "urgent", "out_of_scope": False}, Judgement(Outcome.URGENT, False)),
        ({"outcome": None, "out_of_scope": False}, Judgement(None, False)),
        ({"outcome": "standard", "out_of_scope": True}, Judgement(Outcome.STANDARD, True)),
    ],
    ids=["met", "none met", "out of scope"],
)
def test_the_judgement_is_the_models(reply: dict[str, object], expected: Judgement) -> None:
    model = ScriptedChatModel(messages=iter([AIMessage(json.dumps(reply))]))
    judge = BaselineJudge(
        model,
        PATHWAYS,
        make_engine(Settings()).observations,
        now=lambda: datetime(2026, 10, 23, 15, 0),
    )
    assert judge.judge({"pet": "Bella"}, "vomit_count_24h", "four times") == expected
