# Pet triage chat

A small, safety-first pet triage assistant. 

It asks the owner the questions as a human would, tells them how urgently the dog needs to be seen, and writes a
record for the vet. Built with LangGraph and LangChain on a local model (Ollama), traced and
evaluated with MLflow.

NOTE: Currently this only supports dogs on the gastrointestinal pathway. 

## Prerequisites

- Linux with an Nvidia GPU and >8 GB of free VRAM.
- [Docker](https://docs.docker.com/engine/install/) with Compose v2, and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- About 20 GB of free disk space for the images and the model.

## Quickstart

```bash
git clone https://github.com/jmt-builds/pet-triage-chat.git
cd pet-triage-chat
docker compose up -d --wait
```

The first start takes a few minutes while it builds two images and downloads `gemma4:e4b`
(6.6 GB). `--wait` returns once the model is loaded. Then open:

- **Chat:** http://localhost:3000
- **MLflow** (traces and evaluations): http://localhost:5000

Say hello, choose Bella (the demo client's Labrador) and describe the problem, for example:

> she's been sick 4 times since this morning and ate a sock last night

Choose Milo, the client's cat, to see how the assistant declines what it can't triage. The
finished record for the vet is the output of the conversation's last trace in MLflow
(experiment `pet-triage`).

Stop it with `docker compose down`; the model stays downloaded.

**On a remote machine**, both ports listen on localhost only, so tunnel to them from your
own computer and open the same addresses:

```bash
ssh -N -L 3000:localhost:3000 -L 5000:localhost:5000 <user>@<host>
```

The browser must reach the chat on port 3000. If a port is taken on the server, copy
`.env.example` to `.env` and change the server side there.

**Try another model** by setting `MODEL`, `THINKING` and `CONTEXT_LENGTH` in `.env` (see
`.env.example`). `MODEL` takes an Ollama tag or a GGUF repo on Hugging Face
(`hf.co/<user>/<repo>:<quantisation>`). Then `docker compose up -d --wait` pulls it and
restarts what changed; nothing is rebuilt.

**Run the evaluation** (45 owner messages, three runs each, logged to the MLflow experiment
`pet-triage-eval`):

```bash
docker compose run --rm agent python -m evals.extraction
docker compose run --rm agent python -m evals.extraction --design baseline
```

## Design

```
owner message ──▶ extract (LLM) ──▶ assess (rules) ──▶ reply (fixed text)
```

**The model reads; code decides.** 

A LangChain chain turns each owner message into typed
answers, constrained to a schema built from the questions. Code does everything else: the
question order, the urgency and every word the owner reads. It's a workflow, not an agent,
on purpose: in triage, being confidently wrong has consequences, so the decisions stay
deterministic and unit-tested. A "no" only counts for the question just asked, so the model
can't lower the urgency by inventing one; it can still miss something, which is what the
evaluation measures.

**Clinical content stands in for vet-approved rules.** 

Everything in
[pet_triage/clinical](src/pet_triage/clinical/README.md) is transcribed from a published
triage figure and treated as if a vet had signed it off. The item lists and thresholds are
placeholders, as are the timings in the closing advice ("in the next day or two"): the
figure's levels are waits in the clinic, not when to come in. Adding a body-system pathway
is a data change only.


### Results

The evaluation scores one owner message at a time. "Correct" means the outcome the clinical
rules give for what the owner said, not a clinical judgement. The baseline is one model
call that gets the same facts, field descriptions, criteria and scope, and judges the
urgency itself; only who applies the rules differs. With `gemma4:e4b`, thinking on,
prompts `extract@1` and `baseline@1`, 135 runs each:

| Design | Correct | Under-triage | Over-triage | Failed |
|---|---|---|---|---|
| Structured (model reads, code decides) | 122 | 13 | 0 | 5 |
| Baseline (model decides) | 120 | 5 | 10 | 0 |

The structured design is **not** safer per message: the baseline under-triages less, 5
runs to 13. Repeat runs of a case are nearly identical, so the real sample is 45 cases. The
structured design's 13 under-triages come from 6 of them: three where vomiting hides a
complaint the demo doesn't cover (a collapse, twice, and a limp), so the call is triaged
instead of sent out of scope; "the one I mentioned"; a Dairy Milk bar not read as
chocolate; and "vomiting blood" once. The baseline's 5 come from 3 of the same cases.
What code buys is predictability: 9 of the baseline's 10 over-triages name an urgency
when nothing applies yet ("hello"). Repeat runs vary by a few counts even at temperature 0,
so small differences are noise. "Failed" means the model's reply couldn't be read; the
owner is asked again. Thinking matters: on an earlier set of cases, turning it off took the
structured design from 8 under-triages to 27, for a median of 0.4 s per message instead
of 1.1 s.

### Known trade-offs

- **No chat history.** The model sees only the latest message and the question asked, so
  every answer traces to one message, but "the one I mentioned" can't be resolved.
- **The latest answer wins.** A later message can overwrite a good answer, including the
  complaint, and the per-message evaluation can't see it. Corrections aren't always
  picked up.
- **Each question is asked once.** An owner who replies with a question of their own loses
  that answer, and their question goes unanswered because every reply is fixed text.
- **Vague times aren't recorded.** "The day before yesterday in the evening" has no clock
  time, so an over-24-hour foreign body stays one level too low.
- **Served by `langgraph dev`**, LangGraph's in-memory development server, so the demo runs
  with one command: conversations are lost when the agent restarts and there's no auth.
  Production would serve the graph with a persistent checkpointer (`build_graph` already
  takes one).
- **Not designed:** vague amounts, several animals in one call, or an owner withdrawing the
  original complaint.

### What to test next

- **Whole conversations.** A simulated owner given each scenario's facts, scored on the
  outcome at the end of the call. This would catch overwritten answers, which the
  per-message evaluation misses.
- **Who chooses the questions.** An agent that holds the conversation itself, with the same
  criteria, against this workflow, in that simulation.
- **Over-triage cases.** Hypotheticals ("could she have eaten grapes?"), another animal
  ("my other dog ate a sock") and near-miss items ("chocolate-flavoured treats").
- **Short chat history.** Whether the last few turns fix "the one I mentioned" without
  costing traceability.
- **Other models**, on the same cases.

## Develop

With [uv](https://docs.astral.sh/uv/); the tests use scripted models, so they need neither a
GPU nor Ollama:

```bash
uv sync
uv run pytest
uv run ruff check . && uv run mypy src tests evals
```
