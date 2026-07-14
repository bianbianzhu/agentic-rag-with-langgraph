# Chapter 07: Bounded Agentic Research

## Outcome

The Reference System now has a private compiled Research subgraph that can plan
one authorized search, assess its Evidence, refine the query once, and retrieve
once more. Every loop edge is selected by application code and constrained by a
trusted Turn Execution Budget. Retrieval failure, insufficient Evidence,
deadline exhaustion, clarification, refusal, and direct non-knowledge responses
are explicit terminals.

The subgraph does not create a second retrieval abstraction. Its component node
calls the existing authorization-constrained `retrieve(...)`, then the existing
Evidence Set assembler and context-expansion query. Deterministic L3 tests replace
only planning, assessment, refinement, embeddings, and reranking variability;
PostgreSQL, pgvector, full-text search, and authorization remain real.

## The bounded route

```text
START
  |
  v
plan ── direct / clarification / refusal ─────────────> END
  |
  v
retrieve #1 ── failed ────────────────────────────────> END
  |
  v
assess ── sufficient ───────────────────────> Evidence ready
  |
  v
refine query exactly once
  |
  v
retrieve #2 ── failed ────────────────────────────────> END
  |
  v
assess ── sufficient -> Evidence ready
       └─ insufficient -> incomplete
```

The frozen `ResearchAgentConfig` binds the pinned 5.4 mini identity, model
parameter contract, and prompt/schema versions into one deterministic
fingerprint. The model emits only three small structured decisions:

- `ResearchPlan`: retrieve with a query and a subset of the two fixed Knowledge
  Sources, or select a structured direct/clarification/refusal terminal;
- `EvidenceAssessment`: one Boolean sufficiency decision without reasoning; and
- `QueryRefinement`: one replacement query over unchanged Knowledge Sources.

The plan schema contains no Principal, grants, SQL, paths, URLs, tool names,
result limits, or budget fields. Terminal prose is application-owned rather than
model-authored. A model cannot turn empty Evidence into sufficient Evidence:
that contextual invariant is checked after structured-output validation.

## Real authorized retrieval

For each planned query, application code creates the `RetrievalRequest`. It owns
the request ID, dense and lexical candidate limits, final result limit, current
Authorization Snapshot, embedding dependency, global reranker, and fingerprinted
Retrieval Config. The selected Knowledge Sources are parsed through the fixed
enum before the real retrieval function is called.

Each Retrieval Result stays in disposable Research state. The graph assembles all
completed rounds under the existing item/token budgets and performs authorized
post-selection context expansion. A required Retrieval Result failure or
incomplete Evidence Set ends research as `failed`; it never falls back to stale
or partial Evidence.

## Trusted counters and deadline

`TurnExecutionBudget` lives in frozen Runtime Context, not checkpoint State or
model output. Its version and every limit form a deterministic fingerprint.
Chapter 07 enforces these current limits:

```text
model calls             <= 10
Retrieval Requests      <= 2
research refinements    <= 1
elapsed Turn time       <= 90 seconds
```

`ResearchCounters` records consumed operations in graph state for routing and
inspection. Code checks all counters before incrementing them. Refinement
atomically reserves both its one research iteration and its model call, so a
partial counter update cannot create an extra edge.

The monotonic clock and absolute deadline are trusted Runtime dependencies. The
graph checks the deadline before and after every synchronous model or retrieval
call. It cannot cancel a blocking dependency by itself, but a call that returns
after the deadline can only produce the safe `deadline_exceeded` terminal; its
late result cannot become Evidence-ready or user-visible success.

## Terminal contracts

- `evidence_ready` contains a complete Evidence Set for the Answer subgraph.
- `direct` is a code-owned greeting and performs no retrieval.
- `clarification` asks a generic code-owned clarifying question and performs no
  retrieval.
- `refused` covers requests outside the configured knowledge task or a trusted
  context mismatch.
- `incomplete` covers insufficient Evidence or exhausted counters/deadline.
- `failed` covers planning, retrieval, assessment, refinement, or Evidence
  assembly failure with a sanitized reason and safe response.

Raw Evidence remains disposable and no research terminal commits Thread Memory.
Chapter 08 will wire persistent multi-turn context and terminal Turn Records;
Chapter 09 will own authorization-change restart of the complete Current Turn
Work. The verified Answer subgraph remains the only path from Evidence to factual
user output.

## Run the verification

```bash
docker compose up -d postgres
uv run pytest tests/unit/test_research.py -q
uv run pytest tests/graph_integration/test_research_graph.py -q
uv run pytest -q
uv run pyright src tests
```

L1 rejects authority or budget fields in model plans, inconsistent actions,
duplicate Knowledge Sources, free-text assessment rationale, invalid counters,
and unbounded routes. L3 runs the compiled graph with real PostgreSQL through
authorized Evidence success, one successful refinement, repeated empty results,
invalid empty-Evidence sufficiency, reranker failure, pre-call and post-call
deadline exhaustion, and clarification without retrieval. No OpenAI or
LangSmith credential is required for deterministic verification.
