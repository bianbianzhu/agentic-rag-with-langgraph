# Agentic RAG Reference System with LangGraph

An engineering-first course and locally runnable Reference System for building
an authorization-aware, multi-turn Agentic RAG application with LangGraph.

The project keeps deterministic components explicit—Corpus Sync, embeddings,
indexing, authorized retrieval, reranking, Evidence assembly, and citation
validation—and uses agents only for contextual decisions such as question
rewriting, research planning, answer generation, and verification.

Development v1 is intentionally local. It includes PostgreSQL/pgvector,
LangGraph Agent Server and Studio inspection, deterministic L1–L3 tests, and a
real OpenAI/LangSmith L4 release workflow. It does **not** include deployment,
production authentication, a public chat API, or a custom application UI.

## System at a glance

```text
Source Documents
      |
      v
Corpus Sync -> PostgreSQL + pgvector -> authorized dense + lexical retrieval
                                           |
User message -> contextualize -> plan -----+
                                           v
                                      rerank + Evidence Set
                                           |
                                           v
                                  generate -> cite -> verify
                                           |
                              revalidate current Access Scope
                                           |
                                           v
                                  Cited Answer / safe terminal
```

The Principal comes from trusted Runtime Context. User and document text are
Untrusted Content: they may express intent or provide Evidence, but they cannot
change identity, Access Grants, SQL predicates, budgets, or citation rules.

For the authorization model, open
[`docs/architecture/access-scope-authorization.html`](docs/architecture/access-scope-authorization.html)
in a browser. For the implementation sequence, start with
[`IMPLEMENTATION.md`](IMPLEMENTATION.md).

## Prerequisites

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- Docker with Compose
- Optional for L4 only: an OpenAI API key and a LangSmith API key

## Quick start

Install the development and evaluation dependencies:

```bash
uv sync --group dev --group eval
```

Create the local environment file:

```bash
cp .env.example .env
```

For credential-free local development, leave both API keys empty and set:

```dotenv
LANGSMITH_TRACING=false
```

Start PostgreSQL with pgvector:

```bash
docker compose up -d postgres
docker compose ps
```

Run the complete local verification surface:

```bash
uv run pytest -q
uv run pyright
uv lock --check
git diff --check
```

The database migrations are forward-only and are applied by the test and live
evaluation composition code. The local Compose service exposes PostgreSQL at
`127.0.0.1:55432` and uses passwordless host-local trust authentication for
development only.

## Run tests by layer

Use the smallest layer that matches the change, then run all accumulated local
checks before considering the work complete.

| Layer | Purpose | Command |
| --- | --- | --- |
| L1 | Contracts and deterministic units | `uv run pytest tests/unit -q` |
| L2 | Real PostgreSQL authorization, sync, and retrieval | `uv run pytest tests/data_integration -q` |
| L3 | Compiled subgraphs, multi-turn behavior, and scenarios | `uv run pytest tests/graph_integration -q` |
| Smoke | Local Agent Server lifecycle and Thread reuse | `uv run pytest tests/smoke/test_agent_server.py -q` |
| All local | L1–L3 plus smoke | `uv run pytest -q` |
| Types | Python type checking | `uv run pyright` |

L2, L3, and the smoke test require the Compose database. L1 does not require
PostgreSQL, OpenAI, or LangSmith.

## Inspect the graph in LangGraph Studio

Start the local Agent Server without opening a browser automatically:

```bash
LANGSMITH_TRACING=false uv run langgraph dev --no-browser
```

The command serves the `engineering_assistant` graph from `langgraph.json` at
`http://127.0.0.1:2024` and prints a Studio URL. In Studio, inspect:

- the top-level Turn graph and its Research and Answer subgraphs;
- checkpointed `thread` versus disposable `current_turn` State;
- the trusted `principal_id` Context field;
- route counters, terminal outcomes, and authorization restart behavior; and
- the same Thread across multiple Turns.

A minimal development input is:

```json
{
  "thread": {},
  "current_turn": {
    "turn_id": "turn-1",
    "user_message": "Hello"
  }
}
```

Supply Context separately:

```json
{
  "principal_id": "alice"
}
```

Important: development v1 does not construct database pools or model clients
from JSON Context inside Agent Server. The Studio surface is therefore for
graph, State, checkpoint, and lifecycle inspection. The fixed L4 target below
is the repository's real-model end-to-end composition. An arbitrary-message
chat runtime is a later application boundary, not a hidden README step.

## Run the static graph prototype

Students can replay the synthetic scenarios without PostgreSQL, model access,
or API keys:

```bash
python3 -m http.server 8000 --directory docs/prototype
```

Open <http://127.0.0.1:8000/>. See the
[prototype instructions](docs/prototype/README.md) for its scope.

## Run the live LangSmith evaluation workflow

L4 uses the fixed golden dataset and real OpenAI models:

- `openai:gpt-5.4-mini-2026-03-17` for contextual decisions and answers;
- `openai:gpt-5.4-nano-2026-03-17` for listwise reranking; and
- `openai:text-embedding-3-small` for embeddings.

Put secrets only in the ignored `.env` file:

```dotenv
OPENAI_API_KEY=...
LANGSMITH_API_KEY=...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=agentic-rag-reference-system-dev
```

Never commit `.env`. The client-side anonymizer removes configured API keys and
database credentials before trace upload, but secrets still do not belong in
State, fixtures, logs, prompts, or reports.

Validate the local dataset contract without credentials:

```bash
uv run python -m evals.run preview
```

Publish an untagged LangSmith dataset candidate and save the returned exact
`dataset_version` timestamp:

```bash
uv run --env-file .env python -m evals.run publish
```

Run the initial release candidate against that exact version:

```bash
uv run --env-file .env python -m evals.run experiment \
  --dataset-version <ISO-8601 timestamp> \
  --initial-release
```

This command first requires L1–L3, Pyright, lock verification, and the Agent
Server smoke test to pass. It then runs the nine release examples and the
poisoned-source diagnostic three times each, uploads traces and evaluator
feedback to LangSmith, and writes ignored review artifacts under
`evals/artifacts/`.

After inspecting every output, Evidence Item, citation, evaluator explanation,
and trace, accept the candidate explicitly:

```bash
uv run --env-file .env python -m evals.run accept \
  --candidate-report evals/artifacts/candidate-report.json \
  --human-approved
```

Acceptance moves `release-v1` only when all automated gates pass and human
approval is explicit. Set the accepted release experiment as the LangSmith
baseline in the UI after acceptance. Later candidates use the accepted report:

```bash
uv run --env-file .env python -m evals.run experiment \
  --dataset-version <current release-v1 timestamp> \
  --baseline-report evals/artifacts/accepted-release-report.json
```

See
[`docs/course/11-evaluation-and-release-evidence.md`](docs/course/11-evaluation-and-release-evidence.md)
for evaluator applicability, non-compensating gates, latency/token/cost limits,
and the poisoned-source diagnostic.

## Reference scenarios

The canonical fixtures under `fixtures/reference-system/` cover:

- authorized factual retrieval and citations;
- greetings and clarification without retrieval;
- one bounded research refinement;
- authorization revocation and one safe restart;
- document prompt injection containment;
- a deliberately false but authorized source;
- answer repair and hard execution budgets;
- retrieval failure without an unsafe fallback;
- Principal/Thread mismatch;
- multi-turn contextualization; and
- relevant but unauthorized content.

These are trusted scenario commands around untrusted user and document content.
The live target rejects commands that do not exactly match the local golden
fixture.

## Repository map

| Path | Responsibility |
| --- | --- |
| `src/agentic_rag/agents/` | Contextualization, research, generation, and verification decisions |
| `src/agentic_rag/corpus/` | Source identity, chunking, embeddings, indexing, and Corpus Sync |
| `src/agentic_rag/retrieval/` | Authorized hybrid retrieval, reranking, and Evidence assembly |
| `src/agentic_rag/graph/` | Research, Answer, and top-level Turn graphs |
| `src/agentic_rag/authorization.py` | Authorization Snapshot and Access Scope contracts |
| `src/agentic_rag/conversation.py` | Conversation Thread, Turn Record, and bounded memory |
| `src/agentic_rag/citations.py` | Citation hydration, validation, and rendering |
| `migrations/` | PostgreSQL schema, authorization predicate, and read-only retrieval role |
| `fixtures/reference-system/` | Versioned corpus, trusted grants, attacks, and expected scenarios |
| `evals/` | LangSmith dataset, live target, evaluators, reports, and release gates |
| `tests/` | L1 unit, L2 data, L3 graph, and Agent Server smoke tests |
| `docs/course/` | Eleven-chapter English course documentation |
| `docs/prototype/` | Static deterministic graph replay for students |

## Course path

1. [System contract and development loop](docs/course/01-system-contract-and-development-loop.md)
2. [Corpus as a transaction](docs/course/02-corpus-as-a-transaction.md)
3. [Authorization before relevance](docs/course/03-authorization-before-relevance.md)
4. [Authorized hybrid retrieval](docs/course/04-authorized-hybrid-retrieval.md)
5. [Reranking and the Evidence Set](docs/course/05-reranking-and-the-evidence-set.md)
6. [Answers that can be checked](docs/course/06-answers-that-can-be-checked.md)
7. [Bounded agentic research](docs/course/07-bounded-agentic-research.md)
8. [Multi-turn memory without authority drift](docs/course/08-multi-turn-memory-without-authority-drift.md)
9. [Authorization changes mid-conversation](docs/course/09-authorization-changes-mid-conversation.md)
10. [Adversarial and failure scenarios](docs/course/10-adversarial-and-failure-scenarios.md)
11. [Evaluation and release evidence](docs/course/11-evaluation-and-release-evidence.md)

## Troubleshooting

**PostgreSQL connection refused**

```bash
docker compose up -d postgres
docker compose ps
```

Wait for the `postgres` service to become healthy. To reset this disposable
development database, run `docker compose down` and start it again.

**Agent Server starts but cannot produce a live knowledge answer**

This is expected when using only JSON State and `principal_id` Context. Studio
does not receive database pools, model objects, an embedder, or a reranker. Use
the L1–L3 scenarios for deterministic behavior and the L4 workflow for the
real-model end-to-end composition.

**L4 reports missing credentials**

Run the command with `uv run --env-file .env ...` and confirm the two API keys
exist in `.env`. Do not print their values while debugging.

**Port 55432 or 2024 is already in use**

Stop the conflicting local service. Tests choose an unused Agent Server port,
but the documented interactive server uses 2024 and Compose uses 55432.

## Scope and status

This repository is a development Reference System and course, not a production
service. Production authentication, RLS defense in depth, deployment, scaling,
online monitoring, retention operations, and a user-facing application remain
out of scope for v1. The authoritative implementation status is always recorded
in [`IMPLEMENTATION.md`](IMPLEMENTATION.md).
