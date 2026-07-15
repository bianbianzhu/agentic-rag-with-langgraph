# Chapter 09: Authorization Changes Mid-conversation

## Outcome

The Reference System now composes the verified Contextualization, Research, and
Answer paths into one top-level Turn. Every derived result belongs to one
Authorization Snapshot. Immediately before terminal commit, application code
captures the current effective Access Scope again. Unchanged scope commits;
changed scope destroys the current result and restarts once; a second change
fails closed.

This is whole-Turn consistency. The check is not limited to cited documents.
If any Source Document enters or leaves the Principal's effective scope while a
Turn is running, the revision changes and all work derived under the old
revision becomes disposable.

## The completed Turn topology

```text
           prepare
                |
                v
       capture snapshot S1
                |
                v
   compact + contextualize
                |
                v
      Research subgraph ---- direct/refusal/incomplete --+
                |                                        |
          Evidence Set                                   |
                v                                        |
       Answer subgraph                                   |
                |                                        |
        buffered CitedAnswer                             |
                +-------------------+--------------------+
                                    v
                      capture current snapshot S2
                                    |
              +---------------------+--------------------+
              |                     |                    |
          S1 == S2             changed once        changed twice
              |                     |                    |
              v                     v                    v
        commit Turn Record    discard all work      safe failed record
                              preserve counters      no answer/citations
                              contextualize again
```

The Research and Answer subgraphs remain private implementation details. Their
structured outputs are adapted into `CurrentTurnWork`; they do not write Thread
Memory themselves. Only the top-level final-authorization node may commit a
post-retrieval result.

## What an Authorization Snapshot means

The snapshot contains only the trusted Principal ID and a content-free hash of
the ordered Source Document identities currently authorized for that Principal.
It contains no document text, ACL explanation, group list, or secret.

```text
S1 = hash(principal_id + authorized Source Document identities)
S2 = hash(the same data captured immediately before commit)
```

The revision changes for direct grants, Group-derived grants, organization
public grants, and membership changes. User text, retrieved document text, and
Agent output cannot select the Principal or supply either snapshot.

Specific controls still run inside the Turn:

- retrieval applies the shared SQL authorization predicate before ranking;
- context expansion and citation hydration apply it again;
- historical citation dependencies are reauthorized before a model sees them;
- the Answer subgraph rehydrates citation mappings before releasing its buffer;
- the top-level gate compares the whole effective scope before Thread commit.

The layers answer different questions. Citation hydration asks whether these
exact cited chunks are still valid. The final gate asks whether every decision
in this Turn was made under one stable Access Scope.

## Restart means discard, not continue

Compaction is checkpointed for resume, but it is not allowed to become
irreversible before final authorization. Current Turn Work retains the
pre-compaction semantic Thread. On the first revision change, the graph restores
that Thread and `restart_after_authorization_change()` removes:

- the Standalone Question derived from historical context;
- the old Authorization Snapshot;
- all Retrieval Results and the final Evidence Set;
- citation mappings, draft-derived `CitedAnswer`, and assistant message;
- outcome and terminal reason.

The graph returns through prepare, captures a new snapshot, and repeats any
required compaction and contextualization. Historical memory is therefore
projected under current authority again. It never merely re-runs the final
check, retains an old query rewrite, or keeps a summary produced under the old
scope.

Consumed resources are different from derived work. Model calls, Retrieval
Requests, research iterations, answer repairs, compaction attempts, and the one
authorization restart remain consumed. The absolute deadline also remains the
same. A retry therefore cannot turn a bounded Turn into an unbounded loop.

## Why the second change fails

One restart covers a normal grant or Group-membership update racing with a
request. Repeated change means the graph cannot establish one defensible scope
within its fixed budget. It commits only an application-owned safe message with
`authorization_changed_twice`.

The failed record contains the original user message and terminal metadata. It
contains no Evidence Set, model draft, discarded response, citation dependency,
unauthorized identity, result count, or ACL detail.

Cross-Principal reuse is a different violation. `prepare_turn` rejects a
trusted Runtime Context Principal that differs from the Thread binding before
historical context, retrieval, or model work. Snapshot comparison also rejects
different Principal IDs instead of treating them as a restartable scope change.

## What the tests prove

The deterministic L1 tests prove the three transition decisions and validate
that restart clears derived fields while preserving every consumed counter.
The real PostgreSQL L3 tests then exercise the compiled graph:

1. an unrelated grant changes after the first verified draft; the graph runs
   Research and Answer twice, persists only the second answer, and records real
   citation dependencies;
2. scope changes again during the restarted attempt; the graph returns
   `authorization_changed_twice`, with neither draft canary nor citations in
   Thread Memory;
3. Bob loses the `payments-engineering` Group during answer verification; the
   private-document answer is discarded, the Turn replans under the reduced
   scope, and only a safe direct result commits;
4. a scope change during a clarification rewrite forces a second rewrite and
   final revalidation instead of bypassing the gate;
5. a summary containing an old-scope canary is rolled back, rebuilt from a
   reauthorized historical projection, and absent from final Thread Memory;
6. if scope changes again after that rebuild, the second compacted summary is
   also rolled back before the safe failed record commits;
7. Principal mismatch fails before the model boundary.

These tests use the real PostgreSQL authorization predicate, pgvector-backed
retrieval, compiled Research and Answer subgraphs, and deterministic model,
embedding, and reranking doubles. No fake retriever bypasses the security seam.

## Run the verification

```bash
docker compose up -d postgres
uv run pytest tests/unit/test_authorization.py tests/unit/test_graph.py -q
uv run pytest tests/graph_integration/test_conversation_graph.py -q
uv run pytest tests/unit tests/data_integration tests/graph_integration -q
uv run pytest -q
uv run pyright
uv lock --check
git diff --check
```

Chapter 10 will broaden these invariants into the complete deterministic
adversarial and failure suite. Deployment, authentication, production RLS, and
custom UI/API work remain outside development-only v1.
