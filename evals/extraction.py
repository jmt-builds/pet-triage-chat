"""Extraction evaluation: one owner turn at a time, against the real model.

Each case in ``extraction.yaml`` is a point in a conversation: the record so far, any
earlier messages, the question just asked, and the owner's reply. The real extractor reads
the reply and the real engine merges it into the record, exactly as in production, where
the extractor sees the latest message, the question and the last few messages before it.
The record that comes out is scored against the one expected.

Every case runs ``RUNS`` times, because the model's output can vary. Run it inside the
agent container, so it uses the same model, Ollama and MLflow as the app::

    docker compose run --rm agent python -m evals.extraction

``--design baseline`` runs the same cases through the baseline instead (see ``baseline``),
where the model judges the urgency itself. It produces no record, so only the triage
scorers apply.
"""

import argparse
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import mlflow
import yaml
from langchain.agents.structured_output import StructuredOutputError
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from mlflow.genai.scorers import Scorer, scorer

from evals import baseline
from pet_triage.assessment import assess
from pet_triage.clinical.pathways import PATHWAYS
from pet_triage.config import Settings
from pet_triage.domain.observation import Answers, Kind, Observation, Value
from pet_triage.domain.outcome import Outcome
from pet_triage.engine import Triage
from pet_triage.extractor import PROMPT_VERSION, ExtractionError, LangChainExtractor
from pet_triage.graph import make_engine, make_extractor, make_model

CASES = Path(__file__).with_name("extraction.yaml")
RUNS = 3
NOW = datetime(2026, 10, 23, 15, 0)
"""The clock the extractor reads, so expected hours ("last night at 8") don't drift."""

ENGINE = make_engine(Settings())
OBSERVATIONS = {observation.id: observation for observation in ENGINE.observations}


def load_cases(path: Path = CASES) -> list[dict[str, Any]]:
    """The cases as MLflow evaluation rows, each repeated ``RUNS`` times."""
    cases = yaml.safe_load(path.read_text())["cases"]
    return [
        {
            "inputs": {
                "before": case.get("before", {}),
                "history": case.get("history", []),
                "asked": case.get("asked"),
                "message": case["message"],
            },
            "expectations": {
                "answers": plain(typed(case["expected"])),
                "urgency": urgency(typed(case["expected"])),
            },
            "tags": {"case": case["id"], "run": str(run)},
        }
        for case in cases
        for run in range(1, RUNS + 1)
    ]


def typed(answers: dict[str, Any]) -> Answers:
    """YAML answers as the engine's types: items become a set, numbers a float.

    Raises:
        KeyError: If a field isn't an observation, so a typo in a case fails loudly.
    """
    return {key: _typed(OBSERVATIONS[key], value) for key, value in answers.items()}


def _typed(observation: Observation, value: Any) -> Value:  # noqa: ANN401
    """One YAML answer as the type its observation expects."""
    if observation.kind is Kind.ITEMS:
        return frozenset(value)
    if observation.kind in (Kind.NUMBER, Kind.HOURS_AGO):
        return float(value)
    return cast(Value, value)


def messages(history: list[dict[str, str]]) -> list[AnyMessage]:
    """A case's history as chat messages: each entry is ``{owner: ...}`` or ``{line: ...}``.

    Raises:
        ValueError: If an entry isn't one of those, so a typo in a case fails loudly.
    """
    speakers = {"owner": HumanMessage, "line": AIMessage}
    result: list[AnyMessage] = []
    for entry in history:
        ((speaker, text),) = entry.items()
        if speaker not in speakers:
            raise ValueError(f"Unknown speaker {speaker!r}; use owner or line.")
        result.append(speakers[speaker](text))
    return result


def plain(answers: Answers) -> dict[str, Any]:
    """Answers as JSON-friendly values for MLflow: item sets become sorted lists."""
    return {
        key: sorted(value) if isinstance(value, frozenset) else value
        for key, value in answers.items()
    }


OUT_OF_SCOPE = len(Outcome)
"""The rank of a call the engine ends as out of scope. It sits above every outcome: the owner
is sent to their vet or the urgent number, and missing that is under-triage."""


def urgency(answers: Answers) -> int:
    """The engine's verdict on a record as a rank.

    ``OUT_OF_SCOPE`` when the engine ends the call as out of scope, -1 when no pathway
    applies yet, and otherwise the outcome's rank.
    """
    if not ENGINE.in_scope(answers):
        return OUT_OF_SCOPE
    assessment = assess(PATHWAYS, answers)
    return -1 if assessment is None else assessment.outcome.rank


def play_turn_with(extractor: LangChainExtractor) -> Callable[..., dict[str, Any]]:
    """The function MLflow evaluates: one owner turn through the real extractor and engine.

    A failed extraction leaves the record as it was, as in the app, where the owner is asked
    to say it again, and is counted by ``failed``.
    """

    @mlflow.trace
    def play_turn(
        before: dict[str, Any],
        history: list[dict[str, str]],
        asked: str | None,
        message: str,
    ) -> dict[str, Any]:
        triage = Triage(answers=typed(before), last_asked=asked)
        try:
            extracted = extractor.extract(message, asked, messages(history))
        except ExtractionError:
            answers = plain(triage.answers)
            return {"failed": True, "answers": answers, "urgency": urgency(triage.answers)}
        after, _ = ENGINE.step(triage, extracted)
        return {
            "extracted": plain(extracted),
            "answers": plain(after.answers),
            "urgency": urgency(after.answers),
        }

    return play_turn


def judge_turn_with(judge: baseline.BaselineJudge) -> Callable[..., dict[str, Any]]:
    """The function MLflow evaluates for the baseline: the same turn, judged by the model."""

    @mlflow.trace
    def judge_turn(
        before: dict[str, Any],
        history: list[dict[str, str]],
        asked: str | None,
        message: str,
    ) -> dict[str, Any]:
        try:
            judgement = judge.judge(before, asked, message, messages(history))
        except StructuredOutputError:
            return {"failed": True, "urgency": -1}
        if judgement.out_of_scope:
            return {"urgency": OUT_OF_SCOPE}
        return {"urgency": -1 if judgement.outcome is None else judgement.outcome.rank}

    return judge_turn


# Scorers. Each field error counts in exactly one of missing, wrong, overwritten and
# invented, and each run is exactly one of correct_triage, under_triage and over_triage.


@scorer
def complete_record_match(outputs: dict[str, Any], expectations: dict[str, Any]) -> bool:
    """Every field of the record is exactly as expected."""
    return bool(outputs["answers"] == expectations["answers"])


@scorer
def correct_triage(outputs: dict[str, Any], expectations: dict[str, Any]) -> bool:
    """The outcome is right, whatever the fields."""
    return bool(outputs["urgency"] == expectations["urgency"])


@scorer
def under_triage(outputs: dict[str, Any], expectations: dict[str, Any]) -> bool:
    """The outcome is less urgent than it should be: the dangerous error."""
    return bool(outputs["urgency"] < expectations["urgency"])


@scorer
def over_triage(outputs: dict[str, Any], expectations: dict[str, Any]) -> bool:
    """The outcome is more urgent than it should be: a frightened owner, a wasted visit."""
    return bool(outputs["urgency"] > expectations["urgency"])


@scorer
def missing(outputs: dict[str, Any], expectations: dict[str, Any]) -> int:
    """Fields the owner gave that the record doesn't have."""
    return len(expectations["answers"].keys() - outputs["answers"].keys())


@scorer
def wrong(inputs: dict[str, Any], outputs: dict[str, Any], expectations: dict[str, Any]) -> int:
    """Fields recorded with the wrong value, other than earlier answers overwritten."""
    return len(_wrong_fields(outputs, expectations) - _kept_fields(inputs, expectations))


@scorer
def overwritten(
    inputs: dict[str, Any], outputs: dict[str, Any], expectations: dict[str, Any]
) -> int:
    """Earlier answers that were right and that this turn changed."""
    return len(_wrong_fields(outputs, expectations) & _kept_fields(inputs, expectations))


@scorer
def invented(outputs: dict[str, Any], expectations: dict[str, Any]) -> int:
    """Fields in the record that the owner never gave."""
    return len(outputs["answers"].keys() - expectations["answers"].keys())


@scorer
def failed(outputs: dict[str, Any]) -> bool:
    """The model's reply couldn't be read at all."""
    return bool(outputs.get("failed", False))


TRIAGE_SCORERS = [correct_triage, under_triage, over_triage, failed]
"""The scorers both designs share."""

SCORERS = [complete_record_match, *TRIAGE_SCORERS, missing, wrong, overwritten, invented]


def _wrong_fields(outputs: dict[str, Any], expectations: dict[str, Any]) -> set[str]:
    """Fields in both records whose values differ."""
    got, expected = outputs["answers"], expectations["answers"]
    return {key for key in got.keys() & expected.keys() if got[key] != expected[key]}


def _kept_fields(inputs: dict[str, Any], expectations: dict[str, Any]) -> set[str]:
    """Fields already right before the turn that should stay as they were."""
    before, expected = plain(typed(inputs["before"])), expectations["answers"]
    return {key for key in before.keys() & expected.keys() if before[key] == expected[key]}


@dataclass(frozen=True, slots=True)
class Design:
    """A design under evaluation.

    Attributes:
        predict_fn: Plays one case's turn and returns what the scorers read.
        scorers: The scorers that apply to its output.
        prompt_version: Its prompt's version, recorded with the results.
    """

    predict_fn: Callable[..., dict[str, Any]]
    scorers: list[Scorer]
    prompt_version: str


def _build(design: str, settings: Settings) -> Design:
    """The named design on the configured model."""
    if design == "baseline":
        judge = baseline.BaselineJudge(
            make_model(settings), PATHWAYS, ENGINE.observations, now=lambda: NOW
        )
        return Design(judge_turn_with(judge), TRIAGE_SCORERS, baseline.PROMPT_VERSION)
    extractor = make_extractor(settings, ENGINE.observations, now=lambda: NOW)
    return Design(play_turn_with(extractor), SCORERS, PROMPT_VERSION)


def main() -> None:
    """Run every case ``RUNS`` times through one design and log the results as one MLflow run."""
    parser = argparse.ArgumentParser(description="Run the extraction evaluation.")
    parser.add_argument("--design", choices=("structured", "baseline"), default="structured")
    name = parser.parse_args().design
    # One case at a time: Ollama answers one request at a time anyway, and MLflow's default
    # of parallel rows hangs on the shared extractor.
    os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_WORKERS", "1")
    settings = Settings()
    design = _build(name, settings)
    mlflow.set_experiment("pet-triage-eval")
    mlflow.langchain.autolog()
    thinking = " thinking" if settings.thinking else ""
    run_name = f"{name} {settings.model}{thinking} {design.prompt_version}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "design": name,
                "model": settings.model,
                "thinking": settings.thinking,
                "temperature": settings.temperature,
                "context_length": settings.context_length,
                "prompt_version": design.prompt_version,
            }
        )
        result = mlflow.genai.evaluate(
            data=load_cases(), predict_fn=design.predict_fn, scorers=design.scorers
        )
    runs = len(load_cases())
    print(f"{run_name}: {runs} runs")
    for scorer_name in (score.name for score in design.scorers):
        print(f"  {scorer_name:22} {round(result.metrics[f'{scorer_name}/mean'] * runs):4}")


if __name__ == "__main__":
    main()
