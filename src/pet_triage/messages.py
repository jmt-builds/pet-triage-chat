"""Everything the owner reads that isn't an observation's question.

Every message is fixed text, so no generated text ever reaches the owner. Wording is a
draft for a vet nurse to review.
"""

from collections.abc import Mapping

from pet_triage.clinical.pathways import PATHWAYS
from pet_triage.domain.outcome import Outcome
from pet_triage.engine import (
    Ask,
    FailSafe,
    Finish,
    NextAction,
    OutOfScope,
    Rephrase,
    Unsupported,
)
from pet_triage.intake import SUPPORTED_SPECIES

SPECIES_COVERED = ", ".join(sorted(SUPPORTED_SPECIES))
CONCERNS_COVERED = ", ".join(pathway.id for pathway in PATHWAYS)
"""What the demo covers, as owners are told it. Built from the supported species and the
pathways, so adding either updates every message that mentions scope."""

YOUR_PET = "your pet"
"""How the owner's pet is written in fixed text; ``render`` swaps in the name once known."""

CONTACT_VET = "Please contact your vet, or call {phone} if it's urgent."
"""The fallback whenever the assistant can't help."""

OPENER = (
    "Hello. This is a demo triage assistant. It currently supports:\n"
    f"- **Species:** {SPECIES_COVERED}\n"
    f"- **Concerns:** {CONCERNS_COVERED}\n\n"
    "If your pet is struggling to breathe, has collapsed or isn't responding, is having a "
    "seizure or is bleeding heavily, call {phone} now."
)
NOT_FOLLOWED = "Sorry, I didn't quite follow."
REPHRASE = f"{NOT_FOLLOWED} Could you tell me a bit more?"
NOT_ON_ACCOUNT = (
    "I can only help with the pets on your account. If this is an emergency, call {phone} "
    "now. Otherwise, please contact your vet."
)
OUT_OF_SCOPE = (
    "This demo only supports these {kind}: {covered}. If this is an emergency, call {phone} "
    "now. Otherwise, please contact your vet."
)
FAIL_SAFE = "Sorry, I'm having trouble understanding. " + CONTACT_VET
CLOSING = "Thank you. I've passed everything to one of our vets. {advice}."

ADVICE: Mapping[Outcome, str] = {
    Outcome.NON_URGENT: (
        "Keep an eye on your pet at home, and call us if it doesn't settle in a day or two"
    ),
    Outcome.STANDARD: "Please call us to book an appointment in the next day or two",
    Outcome.URGENT: "Be seen today",
    Outcome.VERY_URGENT: "Come in now",
    Outcome.IMMEDIATE: "Come in now and call ahead",
}
"""What the owner is told to do for each outcome, used in the closing.

The VTL's levels are waits in the clinic, not when to come in, so the timings here are our
own placeholders until a vet nurse sets them."""

WORSENING = (
    "If your pet gets worse or seems weak, in pain or very quiet, or anything else worries "
    "you, call {phone}."
)
"""Added to the closing when the owner is left to wait, so they know when to call."""

WAITING_OUTCOMES = frozenset({Outcome.NON_URGENT, Outcome.STANDARD, Outcome.URGENT})
"""Outcomes where the pet isn't coming in now, so the closing adds ``WORSENING``."""

QUESTIONS_AHEAD = "I'll ask a few quick questions so the vet has what they need."
"""Said after a pathway's acknowledgement (``Pathway.acknowledgement``), so the owner knows
questions are coming."""

ANNOUNCEMENTS: Mapping[Outcome, str] = {
    Outcome.URGENT: (
        "Based on what you've told me, your pet should be seen by a vet today. Please call "
        "the practice to arrange it. I'll ask a few more questions so the vet has everything "
        "they need."
    ),
    Outcome.VERY_URGENT: "This is more urgent than I first thought. Please bring your pet in now.",
    Outcome.IMMEDIATE: "Please bring your pet in now and call {phone} on the way.",
}
"""Said when the outcome rises to urgent or above. Lower outcomes only get the closing."""


def render(
    action: NextAction, phone: str, *, first: bool = False, pet_name: str | None = None
) -> str:
    """Turn the engine's next action into the message the owner reads.

    Args:
        action: What the engine decided to do next.
        phone: The urgent-care number, for messages that give it.
        first: Whether this is the bot's first reply, which starts with the opener. Agent
            Chat UI only runs the graph once the owner has written, so the opener answers
            their first message.
        pet_name: The pet the call is about, once the owner has said which. Every "your
            pet" in the body becomes the name, so questions and advice are written with
            "your pet" and never a pronoun. The model still reads "your pet".

    Returns:
        The full message text.
    """
    body = _body(action, phone)
    if pet_name is not None:
        body = body.replace(YOUR_PET, pet_name)
    if not first:
        return body
    return "\n\n".join((OPENER.format(phone=phone), body))


def _body(action: NextAction, phone: str) -> str:
    """The text for one action."""
    match action:
        case Ask():
            return _ask(action, phone)
        case Rephrase(observation):
            return REPHRASE if observation is None else f"{NOT_FOLLOWED} {observation.question}"
        case OutOfScope(reason):
            return _out_of_scope(reason, phone)
        case Finish(assessment):
            return _closing(assessment.outcome, phone)
        case FailSafe():
            return FAIL_SAFE.format(phone=phone)


def _closing(outcome: Outcome, phone: str) -> str:
    """The closing advice, with what to watch for if the owner is left to wait."""
    closing = CLOSING.format(advice=ADVICE[outcome])
    if outcome not in WAITING_OUTCOMES:
        return closing
    return f"{closing} {WORSENING.format(phone=phone)}"


def _out_of_scope(reason: Unsupported, phone: str) -> str:
    """Why the assistant can't help, in the owner's terms."""
    match reason:
        case Unsupported.PET:
            return NOT_ON_ACCOUNT.format(phone=phone)
        case Unsupported.SPECIES:
            return OUT_OF_SCOPE.format(kind="species", covered=SPECIES_COVERED, phone=phone)
        case Unsupported.CONCERN:
            return OUT_OF_SCOPE.format(kind="concerns", covered=CONCERNS_COVERED, phone=phone)


def _ask(ask: Ask, phone: str) -> str:
    """The question, after any acknowledgement and the announcement if the outcome has risen.

    An urgent announcement already says more questions are coming, so ``QUESTIONS_AHEAD``
    is only added without one.
    """
    acknowledgement = [pathway.acknowledgement for pathway in ask.acknowledge]
    if acknowledgement and ask.announce is None:
        acknowledgement.append(QUESTIONS_AHEAD)
    paragraphs = [" ".join(acknowledgement)] if acknowledgement else []
    if ask.announce is not None:
        paragraphs.append(ANNOUNCEMENTS[ask.announce].format(phone=phone))
    return "\n\n".join((*paragraphs, ask.observation.question))
