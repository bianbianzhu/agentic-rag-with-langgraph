# Chapter 06: Answers That Can Be Checked

## Outcome

The Reference System now keeps model prose behind a release boundary. It assigns
answer-local citation keys in code, asks the model for a structured draft,
validates citation integrity deterministically, verifies semantic support in a
separate model decision, permits one bounded repair, and releases only a typed
`CitedAnswer`.

An invalid, unsupported, stale, or unauthorized draft never becomes the user
message. After the one repair is exhausted, the graph returns a deterministic
`incomplete` answer with no citations and no draft text.

## The release pipeline

```text
complete Evidence Set
        |
        v
final PostgreSQL authorization + citation hydration
        |
        v
generate buffered CitationDraft
        |
        v
deterministic citation validation -- invalid --> one repair --┐
        |                                                     |
        v                                                     |
semantic support verification ------ unsupported ------------┘
        |
        v
render CitedAnswer --> user projection
```

The compiled Answer subgraph owns this route. Generation and verification use
LangChain's `BaseChatModel.with_structured_output(...)`; the graph, state,
authorization query, validators, repair budget, and rendering remain ordinary
Python and LangGraph code.

## Code owns citation identity

The model never creates document paths, titles, URLs, chunk IDs, or citation
metadata. After final Evidence selection, code assigns local keys in stable
order:

```text
Evidence Item 1 -> E1
Evidence Item 2 -> E2
...
Evidence Item 8 -> E8
```

Before those mappings are used, PostgreSQL re-reads every selected chunk in a
read-only repeatable-read transaction. The query applies the same
`source_document_is_authorized(...)` predicate as retrieval and checks document,
Knowledge Source, source revision, processing revision, and published Corpus
Revision identity. Path, title, and source locator come from this database read
rather than model output. The graph repeats the same hydration immediately
before release and requires the mapping to remain identical, closing the
revocation window while generation and verification run.

The trusted Runtime Context Principal must equal the Principal in the
Authorization Snapshot. Revocation, stale identity, missing chunks, revision
drift, a database failure, or a Principal mismatch fails before generation and
produces a safe incomplete answer.

## Structured drafts and deterministic validation

A factual `CitationDraft` is a bounded list of claims. Every claim contains text
and one or more supplied local citation keys. Refusal and insufficient-Evidence
drafts contain no claims and require a structured reason such as
`insufficient_evidence`.

The deterministic validator rejects:

- a key outside the final Evidence Set;
- a cited key without a code-owned mapping;
- conflicting duplicate mappings;
- metadata that does not exactly match final Evidence; and
- a factual claim without a citation.

This stage checks integrity, not meaning. A syntactically valid `[E1]` can still
point to text that does not support the claim, so a separate verifier receives
only the claims and their cited Evidence text. It returns `supported` plus
bounded unsupported claim indexes, without free-text reasoning.

## One repair, same Evidence

Citation failure or semantic failure may route to one repair. Repair receives
only structured error codes or unsupported claim indexes. It uses the same
question, the same hydrated mappings, and the same Evidence Set; it cannot issue
a new Retrieval Request or expand authority.

```text
first draft invalid/unsupported -> repair_count = 1 -> validate again
second failure                -> incomplete terminal
verifier exception            -> incomplete terminal
```

The draft and intermediate decisions may exist in internal graph state for
debugging, but the user projection is only `CitedAnswer`. Rendering rechecks that
citation validation passed and semantic support is true. It emits only mappings
actually cited by the rendered claims.

## Run the verification

```bash
docker compose up -d postgres
uv run pytest tests/unit/test_citations.py -q
uv run pytest tests/data_integration/test_citations_postgres.py -q
uv run pytest tests/graph_integration/test_answer_graph.py -q
uv run pytest -q
uv run pyright src tests
```

L1 covers invalid citation keys, missing and conflicting mappings, uncited
claims, typed non-factual answers, and the render release guard. L2 proves that
citation hydration uses live PostgreSQL authority and authoritative metadata.
L3 invokes the real compiled Answer subgraph with real authorized retrieval and
PostgreSQL, while replacing only the chat model with deterministic structured
responses. It covers verified release, invalid-citation repair, semantic repair,
repair exhaustion, mid-answer grant revocation, and Runtime Context Principal
mismatch.

Bounded research routing, multi-turn history, authorization restart across
Turns, deployment, and production operations remain outside this chapter.
