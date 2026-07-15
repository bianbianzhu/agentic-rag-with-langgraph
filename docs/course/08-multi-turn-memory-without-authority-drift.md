# Chapter 08: Multi-turn Memory without Authority Drift

## Outcome

The Reference System can now continue a Conversation Thread without treating
old answers as current authority. The first run binds the Thread to the trusted
Runtime Context Principal. Later runs reuse the same `thread_id`, supply a new
unique `turn_id`, reauthorize every historical citation dependency, rewrite a
follow-up into an inspectable Standalone Question, and atomically append one
terminal Turn Record.

This chapter connects the multi-turn boundary only. The top-level graph still
uses the Chapter 01 deterministic terminal message while the already verified
Research and Answer subgraphs remain private. Later chapters compose those
pieces and add whole-Turn authorization-change restart; no old Evidence Set or
draft answer is reused here.

## Three different kinds of state

```text
trusted Runtime Context       persistent Thread Memory
principal_id                  principal binding
database + model handles      complete terminal Turn Records
deadline + hard budgets       structured Conversation Summary
             \               /
              v             v
                 Current Turn Work
                 user input + active turn_id
                 temporary rewrite/research/answer state
                             |
                    terminal commit only
                             v
                 Current Turn Work = cleared
```

The distinction is a security boundary, not just organization:

- Runtime Context is trusted and is never copied into semantic memory.
- Thread Memory contains original user messages, Standalone Questions,
  terminal assistant messages, terminal reasons, and citation identities. Its
  strict schemas reject raw Evidence, drafts, prompts, and tool traces.
- Current Turn Work is disposable. The latest committed state sets it to
  `None`; checkpoint history may still exist for local Studio debugging, but a
  later model does not receive it as active memory.

## Bind identity once, evaluate authority every Turn

The graph checkpoints a `prepare_turn` node before doing model work. It binds an
empty Thread to `RuntimeContext.principal_id` and stores an input fingerprint
for the active `turn_id` and user message.

```text
new thread + alice -> bind alice -> continue
alice thread + bob -> thread_principal_mismatch
active turn-7 + turn-8 -> thread_busy
active turn-7 + changed input -> turn_resume_incompatible
completed turn-7 repeated -> return existing terminal record
```

Principal binding never freezes Access Scope. Before historical assistant text,
Standalone Questions, or citation details enter the Contextual Rewriter or
Summarizer, application code checks every cited Source Document with the same
PostgreSQL authorization predicate used by retrieval.

If a dependency is no longer authorized, the projection retains the original
user message but removes the old answer, Standalone Question, Source Locator,
and citation identity. The model sees only the marker
`historical_evidence_no_longer_authorized`. A citation explains why an earlier
answer was produced; it is never a present-day grant.

## Contextual rewriting is a narrow decision

The Contextual Rewriter receives only the current user message and the bounded,
currently authorized Thread projection. Its strict output contains:

```text
standalone_question
depends_on_history
referenced_turn_ids
clarification_needed
ambiguity_reason?
```

It cannot return a Principal, Access Scope, retrieval parameters, an answer, or
free-text reasoning. Referenced Turn IDs must exist in the supplied projection.
If authorized history cannot uniquely resolve “did it happen there too?”, the
terminal outcome is `clarification_requested`; the graph does not guess and
retrieve against an invented interpretation.

The pinned `ContextualizationConfig` records the 5.4 mini model identity plus
rewrite and summary prompt/schema versions. Its fingerprint makes changes to
that contract inspectable.

## Bounded memory and safe compaction

Active semantic context has a 2,000-token budget using the versioned
characters-divided-by-four approximation. The budget includes the current user
message, authorized Conversation Summary, and a contiguous suffix of newest
complete Turn Records. A Turn Record is never split, and the projection never
skips an oversized newest record to expose an older one.

The graph reserves at most 500 of those tokens for the structured summary. It
selects the smallest oldest Turn prefix whose replacement leaves that reserve
plus the newest complete records within 2,000 tokens. If even one historical
Turn is too large, that Turn itself is eligible for compaction.

When retained memory is too large, the graph reauthorizes the oldest eligible
complete records and asks the Summarizer to update a structured summary. A
summary item has one explicit role:

- topic;
- user constraint;
- evidence-derived claim;
- discussed outcome; or
- unresolved question.

Every item carries source Turn IDs. Evidence-derived claims must also copy real
citation dependencies from those authorized source Turns. Validation rejects
invented Turn IDs, invented citation dependencies, the wrong model identity,
the wrong schema, incomplete coverage, and an output that still exceeds the
active-memory budget.

The summary update gets at most two attempts. Compaction and rewriting consume
the same code-owned model-call limit and absolute Turn deadline. If both summary
attempts fail, the graph commits a safe `failed` record with
`context_compaction_failed` and keeps every old Turn; it never discards history
to make the test pass.

## Atomic terminal commit and resume

```text
START
  |
  v
prepare_turn  -- bind input and checkpoint code-owned work
  |
  v
compact_context (0..2 checkpointed attempts)
  |
  v
contextualize_turn (checkpoint model-call count + result)
  |
  +-- ambiguous ----------------> clarification_requested
  +-- unsafe failure -----------> failed
  +-- valid --------------------> answered
                                     |
                                     v
                         append one Turn Record
                         clear active_turn_id
                         clear Current Turn Work
```

Only `answered`, `refused`, `clarification_requested`, and `failed` are legal
terminal outcomes. Non-answer records require a structured terminal reason;
answered records cannot carry one. The commit appends the record and clears the
active input fingerprint in one immutable state transition.

The L3 resume tests interrupt after `prepare_turn` and after each compaction
attempt. They prove that no partial Turn Record exists, failed-attempt counters
and model-call consumption survive resume, successful summary work is not
repeated, and exactly one terminal record is committed. A changed input,
forged code-owned Current Turn Work, or incompatible Thread schema fails instead
of silently replaying under a different contract.

Conversation Thread deletion remains a whole-Thread Agent Server lifecycle
operation. The graph deliberately offers no individual-Turn deletion or custom
web/API surface; checkpoint deletion does not modify the corpus, grants, or
independent LangSmith traces.

## Run the verification

```bash
docker compose up -d postgres
uv run pytest tests/unit/test_conversation.py -q
uv run pytest tests/graph_integration/test_conversation_graph.py -q
uv run pytest tests/smoke/test_agent_server.py -q
uv run pytest -q
uv run pyright
uv lock --check
```

L1 rejects raw Evidence and drafts in Thread Memory, inconsistent rewrite
references, forged summary dependencies, unpinned summary models, invalid
terminal records, oversized user input, incompatible schemas, non-prefix
compaction, concurrent Turns, and cross-Principal reuse. L3 uses real
PostgreSQL authorization plus a deterministic structured-output model to prove
two-Turn rewriting, grant-revoked historical projection, authorization before
summarization, bounded compaction retry/failure, shared model-call budgets,
idempotent completed Turn IDs, and checkpoint resume. The Agent Server smoke
test reuses one actual `thread_id` across two runs. No OpenAI or LangSmith
credential is required for these deterministic gates.
