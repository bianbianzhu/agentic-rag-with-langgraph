# Chapter 05: Reranking and the Evidence Set

## Outcome

Authorized dense and lexical candidates now pass through one global structured
listwise reranker. The Reference System validates the model output, applies a
versioned relevance threshold and hard Evidence budgets, expands only authorized
same-revision context, and assembles multiple Retrieval Results into one
deterministic Evidence Set.

No failure falls back to the RRF order. Invalid output, an unknown or duplicate
chunk ID, a missing candidate, a non-descending score sequence, or a reranker
exception produces a sanitized `rerank_failed` Retrieval Result with no Evidence
Items.

## One global reranker

RRF remains local to each Knowledge Source. Its bounded outputs are combined
before the reranker is called:

```text
Engineering dense + lexical ──> per-source RRF ──┐
                                                 ├─> one global listwise rerank
Runbooks dense + lexical ─────> per-source RRF ──┘
                                                        |
                                                        v
                                           threshold + item/token budgets
```

The callable receives only authorized candidates. Each candidate contains its
stable `chunk_id`, content, title, and source path; it contains no Principal,
Access Grant, Group, or unrestricted tool argument. The structured output must
return the complete candidate ID permutation with a relevance score from zero to
one in descending order.

`build_reranker(...)` binds LangChain's chat-model interface directly to this
small callable. Runtime composition can initialize the resolved model once:

```python
from langchain.chat_models import init_chat_model

from agentic_rag.retrieval.reranking import build_reranker

model = init_chat_model("openai:gpt-5.4-nano-2026-03-17")
reranker = build_reranker(model)
```

Candidate content is serialized as untrusted data and the system instruction
explicitly forbids following document instructions. Deterministic L1/L2 tests
replace only the callable; they still exercise the real retrieval code and real
PostgreSQL authorization predicate.

## Versioned selection and reproducibility

`RetrievalConfig` now fingerprints all ranking and Evidence-shaping inputs:

```text
embedding identity + dense/lexical semantics
branch budgets + per-source fusion cutoff + RRF constant
global rerank cutoff + reranker model/prompt identity
relevance threshold + result limit
context window + Evidence token limit + token-counting semantics
```

The request result limit is capped at eight final Evidence Items. The default
Evidence projection budget is 6,000 approximate tokens using the explicit
`chars-div-4-v1` rule. A successful search whose authorized candidates all fall
below the threshold is `no_evidence`, not a failure; its authorized stage counts
remain available for diagnostics.

Each returned Evidence Item now records global rerank rank and score in addition
to dense, lexical, and fused provenance. Retrieval timings include reranking;
the final Evidence Set separately records context-expansion timing. Raw model
exceptions and unauthorized counts never enter either artifact.

## Context is a second authorization query

Retrieval Results contain matched chunks without neighboring context.
`assemble_evidence_set(results, config, ...)` derives its token limit from the
same fingerprinted Retrieval Config and first performs cross-request selection;
only then does `expand_evidence_set_context(...)` hydrate the selected items.
PostgreSQL performs a new read-only query that requires all of the following:

```text
same Source Document
+ same source and processing revisions
+ same published Corpus Revision observed by retrieval
+ shared source_document_is_authorized(...) predicate
+ configured ordinal window
+ remaining total token budget
```

If authorization or the Corpus Revision changes between retrieval and context
expansion, the Evidence Set becomes incomplete, records the context-stage timing
and structured failure, and exposes no Evidence. Neighbor chunk IDs are recorded
separately in `context_chunk_ids`; they do not borrow the matched chunk's identity
or citation authority.

The Evidence Set records its own fingerprint for item limit, context window,
token limit, and token-counting semantics, plus every input Retrieval Result
fingerprint. Context expansion rejects a different configuration instead of
silently shaping Evidence under an unrelated reproducibility record.

## Evidence Set assembly

One answer attempt may have multiple Retrieval Requests. Their successful
results are assembled without comparing scores across requests:

```text
request A: A1, A2, A3 ─┐
request B: B1, B2     ─┼─> A1, B1, A2, B2, A3
request C: no_evidence ┘    (deduplicate by chunk_id)
```

The assembler takes the first eligible item from every successful request, then
round-robins later items. Duplicate chunk IDs appear once, while every distinct
Retrieval Provenance record is retained. Item and token exhaustion are explicit.
`no_evidence` outcomes remain visible, and any required failed request makes the
Evidence Set incomplete so later sufficiency routing cannot mistake partial work
for success.

## Run the verification

```bash
docker compose up -d postgres
uv run pytest tests/unit/test_evidence_set.py -q
uv run pytest tests/unit/test_retrieval.py -q
uv run pytest tests/data_integration/test_retrieval_postgres.py -q
uv run pytest -q
uv run pyright src tests
```

L1 proves cross-request round-robin selection, selection-before-context,
provenance merging, completeness, and both budgets. L2 proves fail-closed invalid
output and exceptions, one global cross-source call, threshold behavior,
pre-cutoff counts, authorization-safe context, Corpus Revision drift failure, and
the canonical frozen-fixture Recall@5/MRR/stable-order/forbidden-Evidence gates.
No OpenAI or LangSmith credential is required for deterministic verification.

Structured answer generation, citation keys, answer verification and repair,
agentic refinement, multi-turn memory, deployment, and production operations
remain outside this chapter.
