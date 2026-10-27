"""The baseline the structured design is measured against: the model judges urgency itself.

It gets the same model and settings, the same inputs per case (what is already known, the
question just asked and the owner's reply), the same field descriptions as the extractor,
the same clinical criteria, written into its prompt from ``PATHWAYS``, and the same scope:
what owners are told the demo covers. The only difference is who applies the criteria:
here the model, in the structured design code. It exists only for the evaluation and is
never served.
"""

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime

from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from pet_triage.domain.condition import AnyOf, AtLeast, Condition, Yes
from pet_triage.domain.observation import Kind, Observation
from pet_triage.domain.outcome import Outcome
from pet_triage.domain.pathway import Pathway
from pet_triage.domain.symptom import Symptom
from pet_triage.extractor import describe_fields
from pet_triage.messages import CONCERNS_COVERED, SPECIES_COVERED

# Recorded on every evaluation run, so results can be tied to the prompt
PROMPT_VERSION = "baseline@1"
PROMPT = """You read messages from a dog owner on a vet triage line and decide how urgently \
the dog needs to be seen, using only the criteria below.

Rules:
- Each criterion sets an urgency. The outcome is the most urgent criterion met, on this \
scale from least to most urgent: {levels}.
- A pathway's criteria apply only once its first criterion is met. If no criterion is met, \
the outcome is null.
- The line only triages the client's own pets (the pet field's allowed values), of these \
species: {species}; and these concerns: {concerns}. If the pet isn't one of them or isn't \
covered, or the owner describes any problem the pathways don't cover, set out_of_scope to \
true, even if a pathway also applies.
- Count only what the owner actually said. Never guess.

Criteria:
{criteria}

Facts are named by these fields:
{fields}"""


@dataclass(frozen=True, slots=True)
class Judgement:
    """The baseline's answer.

    Attributes:
        outcome: The outcome, or ``None`` when no criterion is met.
        out_of_scope: Whether the call is one the line doesn't triage.
    """

    outcome: Outcome | None
    out_of_scope: bool


class BaselineJudge:
    """Judges urgency with one ``create_agent`` call per owner message.

    Args:
        model: The chat model: the same Ollama model as the extractor, or a fake in tests.
        pathways: Every supported pathway, whose criteria go into the prompt.
        observations: Every observation, for the field descriptions and the questions.
        now: The clock, given to the model so it can work out how long ago things happened.
    """

    def __init__(
        self,
        model: BaseChatModel,
        pathways: tuple[Pathway, ...],
        observations: tuple[Observation, ...],
        now: Callable[[], datetime] = datetime.now,
    ) -> None:
        """Build the agent once; every message reuses it."""
        self._now = now
        self._questions = {observation.id: observation.question for observation in observations}
        prompt = PROMPT.format(
            levels=", ".join(outcome.value for outcome in Outcome),
            species=SPECIES_COVERED,
            concerns=CONCERNS_COVERED,
            criteria=describe_criteria(pathways),
            fields=describe_fields(observations),
        )
        self._agent = create_agent(
            model, system_prompt=prompt, response_format=ProviderStrategy(Judgement)
        )

    def judge(self, before: Mapping[str, object], asked: str | None, message: str) -> Judgement:
        """The model's judgement of the case after the owner's reply.

        Args:
            before: What is already known, keyed by field.
            asked: ID of the question the owner is replying to, if any.
            message: What the owner wrote.

        Returns:
            The outcome, and whether the call is out of scope.
        """
        lines = [
            f"It is now {self._now():%H:%M on %A %d %B %Y}.",
            f"Already known: {json.dumps(before)}",
        ]
        if asked:
            lines.append(f"The owner was asked: {self._questions[asked]}")
        lines.append(f"The owner wrote: {message}")
        result = self._agent.invoke({"messages": [HumanMessage("\n".join(lines))]})
        judgement: Judgement = result["structured_response"]
        return judgement


def describe_criteria(pathways: tuple[Pathway, ...]) -> str:
    """Every pathway's criteria for the prompt, the presenting symptom first."""
    return "\n\n".join(
        "\n".join((f"Pathway {pathway.id}:", *map(_describe_symptom, pathway.symptoms)))
        for pathway in pathways
    )


def _describe_symptom(symptom: Symptom) -> str:
    """One criterion: its condition, any prerequisite, and the urgency it sets."""
    condition = _describe_condition(symptom.when)
    if symptom.requires is not None:
        condition += f", and {symptom.requires.id} is met"
    return f"- {symptom.id}: {condition} -> {symptom.outcome.value}"


def _describe_condition(condition: Condition) -> str:
    """A condition in words, naming the field it reads.

    Raises:
        TypeError: For a condition type with no description yet.
    """
    match condition:
        case Yes(observation):
            return f"{observation.id} is true"
        case AtLeast(observation, threshold):
            unit = " hours ago" if observation.kind is Kind.HOURS_AGO else ""
            return f"{observation.id} is at least {threshold:g}{unit}"
        case AnyOf(observation, items):
            return f"{observation.id} includes any of {', '.join(sorted(items))}"
    raise TypeError(f"No description for {condition!r}")
