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

**Try another model** by setting `MODEL`, `THINKING`, `TEMPERATURE` and `CONTEXT_LENGTH`
in `.env` (see `.env.example`). `MODEL` takes an Ollama tag or a GGUF repo on Hugging Face
(`hf.co/<user>/<repo>:<quantisation>`). Then `docker compose up -d --wait` pulls it and
restarts what changed; nothing is rebuilt.

**Run the evaluation** (46 owner messages, three runs each, logged to the MLflow experiment
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

Each owner message is scored on its own against the outcome the clinical rules give. The
baseline is one model call with the same inputs that judges the urgency itself. With
`gemma4:e4b`, thinking on, 45 cases run three times each:

| Design | Correct | Under-triage | Over-triage | Failed |
|---|---|---|---|---|
| Structured (model reads, code decides) | 125 | 10 | 0 | 8 |
| Baseline (model decides) | 123 | 2 | 10 | 0 |

- The structured design is **not** safer per message. Its under-triages come from
  extraction misses: a collapse or limp hidden behind vomiting, a chocolate brand not
  recognised, and a time without a clock time.
- What code buys is predictability: 9 of the baseline's 10 over-triages name an urgency
  when nothing applies yet ("hello").
- Whole runs vary by a few counts even at temperature 0, so small differences are noise.
- "Failed" means the reply couldn't be read and the owner is asked again. `poo_no` isn't
  scored yet: YAML reads its message, `no`, as `false`.

### Known trade-offs

- **Read-only history.** The model sees the last four messages but records only what the
  latest one says, so every answer traces to one message. "She's just thrown up again"
  doesn't add one to the count.
- **The latest answer wins.** A later message can overwrite a good answer, including the
  complaint, and the per-message evaluation can't see it. Corrections aren't always
  picked up.
- **Each question is asked once.** An owner who replies with a question of their own loses
  that answer, and their question goes unanswered because every reply is fixed text.
- **Vague times aren't recorded.** "The day before yesterday in the evening" has no clock
  time, so an over-24-hour foreign body stays one level too low. One unreadable time fails
  the whole reply, losing the rest of the message.
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
- **Other models**, on the same cases.

## Develop

With [uv](https://docs.astral.sh/uv/); the tests use scripted models, so they need neither a
GPU nor Ollama:

```bash
uv sync
uv run pytest
uv run ruff check . && uv run mypy src tests evals
```
