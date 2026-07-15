# Chapter 10: Adversarial and Failure Scenarios

## Outcome

The Reference System now has one canonical deterministic security suite. It
runs thirteen scenarios and fifteen Turns through the compiled top-level
LangGraph, the real PostgreSQL authorization predicate, real hybrid retrieval,
and a test checkpointer. Deterministic doubles replace only model, embedding,
and reranking uncertainty; there is no second fake retriever.

This chapter does not add a new success path. It tries to break the boundaries
built in Chapters 01 through 09 and records the expected failure semantics as
versioned fixture data.

## The test boundary

```text
trusted scenario manifest
        |
        | Principal, snapshot, overlay, expected contract
        v
temporary composed Corpus ---- untrusted attack document
        |
        v
Corpus Sync -> PostgreSQL -> compiled Turn graph -> checkpoint
                                |                    |
                     deterministic decisions        v
                                |              final Thread
                                v
                       real authorized retrieval
```

Every scenario starts from a fresh database state and a fixed `s1-baseline`
snapshot. An attack overlay is copied onto a temporary Corpus root before Sync;
the committed baseline is never mutated. A separate trusted JSON registry owns
overlay grants and Evidence aliases. Document text cannot create either one.

The strict scenario schema rejects unknown fields and records:

- the Principal, Corpus snapshot, optional overlay, and authorization event;
- each fixed user message in a one- or two-Turn script;
- required, untrusted, and forbidden Evidence aliases;
- self-contained required and forbidden claims;
- exact outcome, terminal reason, and route counters; and
- whether the scenario is a known source-truth limitation.

Aliases resolve to the real current Indexed Chunk IDs after Corpus Sync. Test
expectations therefore remain readable without replacing stable production
identity or bypassing PostgreSQL.

## The thirteen scenarios

| Scenario | Boundary exercised | Expected result |
| --- | --- | --- |
| `happy` | authorized factual answer and citation | answered from D2 |
| `greeting` | direct route performs no retrieval | answered |
| `clarification` | ambiguous reference is not guessed | clarification requested |
| `refine` | one bounded query refinement | answered from D3 |
| `auth_change` | discard D8 after revocation and restart once | answered only from public D4 |
| `document_injection` | malicious Evidence remains content, never authority | answered from D5's valid section |
| `poisoned_fact` | grounded output can still repeat a false source | answered, marked known limitation |
| `repair` | invented claim is removed by one repair | answered without new retrieval |
| `budget` | two insufficient retrievals terminate | failed / `insufficient_evidence` |
| `thread_mismatch` | Bob cannot resume Alice's Thread | refused before model or retrieval |
| `retrieval_failure` | invalid reranker output has no fallback | failed / `rerank_failed` |
| `multi_turn` | follow-up is rewritten and retrieves again | two separately grounded answers |
| `relevant_not_allowed` | relevance cannot override Access Scope | failed / `insufficient_evidence` |

The suite asserts actual Evidence membership and final citation mappings, not
only answer text. It also checks retrieval, refinement, repair, and
authorization-restart counters at the final authorization boundary.

## Document injection containment

The D5 overlay contains one valid fact and an explicit `SYSTEM OVERRIDE`
instruction asking the Agent to change identity, grants, SQL, budgets, and
citation rules. The test proves that this instruction really reaches the Answer
model's Evidence projection. It is not made harmless by deleting it before the
model sees it.

It remains harmless because authority is supplied elsewhere:

```text
document text ----> Evidence projection ----> may support a factual claim
      X
      +-----------> Runtime Context / Access Grants / SQL / budgets
```

The final answer cites only D5's valid section, the instruction is absent from
Thread Memory, and Carol still cannot read D5 even though the document asks to
self-grant access. Trusted overlay configuration, not source content, grants
D5 to `payments-engineering`.

## Poisoned source truth is a visible limitation

Authorization, grounding, and truth are different properties. D6 is authorized
and the generated claim is entailed by D6, but D6 is deliberately false. The
system therefore produces a structurally valid grounded answer that is not a
correct-answer success.

The manifest marks `poisoned_fact` with `known_limitation: true`. Chapter 11
keeps it in the diagnostic dataset split and excludes it from the release
answer-correctness aggregate. This prevents citation validity from being
misreported as factual correctness.

## Secret exclusion and deletion

Automatic LangSmith tracing is configured with a client-side secret anonymizer.
It runs before upload across trace inputs, outputs, and metadata. L1 canaries
prove that OpenAI keys, LangSmith keys, and database passwords are replaced
while authorized application content remains visible to the tightly
permissioned developer project. Secrets still do not belong in State,
checkpoints, fixtures, logs, or handoffs.

The Agent Server smoke test now completes Alice's two-Turn Thread, reads final
State, deletes the Thread through the SDK, and proves a later read returns 404.
This is the development-v1 deletion boundary; retention policy and production
audit workflows remain deployment work.

## Why the gates do not compensate for one another

Every deterministic authorization, isolation, citation, budget, atomicity, and
secret-exclusion assertion is absolute. A semantic score cannot offset one
unauthorized Evidence Item, one leaked secret, one invented citation, or one
unbounded route. The poisoned-source example likewise cannot improve the
release correctness score merely because its citation is well formed.

Chapter 11 adds live-model and LangSmith evaluation above these controls. It
does not replace them.

## Run the verification

```bash
docker compose up -d postgres
uv run pytest tests/unit/test_observability.py -q
uv run pytest tests/graph_integration/test_reference_scenarios.py -q
uv run pytest tests/smoke/test_agent_server.py -q
uv run pytest -q
uv run pyright
uv lock --check
git diff --check
```

Deployment, a custom application UI/API, production authentication, production
RLS, online monitoring, and formal penetration testing remain outside v1.
