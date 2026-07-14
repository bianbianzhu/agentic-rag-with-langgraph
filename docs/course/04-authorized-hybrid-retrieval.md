# Chapter 04: Authorized Hybrid Retrieval

## Outcome

The Reference System can now execute one bounded Retrieval Request across the
fixed Engineering Docs and Operational Runbooks Knowledge Sources. It embeds the
query once, runs exact pgvector dense retrieval and PostgreSQL full-text lexical
retrieval, fuses their authorized candidates with reciprocal rank fusion, and
returns a structured Retrieval Result.

Authorization is part of both SQL branches. Neither branch fetches broad results
for Python to filter later. A highly relevant but unauthorized Indexed Chunk is
therefore absent from candidate counts, Evidence Items, provenance, and errors.

## The authority-free request

`RetrievalRequest` contains only retrieval intent:

```text
request_id
query
selected Knowledge Sources
dense candidate limit
lexical candidate limit
result limit
```

It has no Principal, Access Grant, SQL, path, or arbitrary filter field. Extra
fields are rejected, Knowledge Sources must be distinct members of the two-value
enum, branch limits are capped at 50, and the returned-item limit is capped at
20. The trusted caller supplies the separately captured Authorization Snapshot.

```text
Retrieval Request ───────────────┐
                                v
Authorization Snapshot ──> retrieve(...)
                                |
Query Embedder ──────────────────┘
```

Retrieval accepts LangChain's `Embeddings` interface directly and calls its
`embed_query(...)` operation. Deterministic tests subclass that interface; a
later runtime composition can supply the configured OpenAI integration without
an application-owned adapter. The committed `langchain[openai]` dependency owns
both the unified LangChain surface and its provider integration.

## Authorization before both rankings

Each Knowledge Source is read in one repeatable-read, read-only PostgreSQL
transaction:

```text
published Corpus Revision
          |
          +--> exact dense SQL
          |      WHERE source_document_is_authorized(...)
          |      ORDER BY embedding <=> bound_query_vector, chunk_id
          |
          +--> lexical SQL
                 WHERE source_document_is_authorized(...)
                   AND search_vector @@ bound_tsquery
                 ORDER BY lexical_score DESC, chunk_id
```

The Principal identifier, query, query vector, Knowledge Source, and limits are
bound parameters. The shared PostgreSQL authorization function from Chapter 03
resolves organization-public, direct Principal, and trusted Group grants. The
function returns false for an unknown Principal or an ungranted Source Document.

The dense branch performs exact cosine-distance ordering. No HNSW or IVFFlat
index is present in v1. The lexical branch uses the stored English `tsvector` and
`websearch_to_tsquery`. A zero or non-finite query embedding fails before SQL so
pgvector cannot produce an unusable score.

## Reciprocal rank fusion

Raw dense and lexical scores remain diagnostics; they are not normalized or
compared. Within each Knowledge Source, RRF adds rank contributions:

```text
fused_score = Σ 1 / (rrf_k + branch_rank)
```

With `rrf_k = 60`, an Indexed Chunk at dense rank 2 and lexical rank 1 receives
`1/62 + 1/61`. A dense-only rank-1 candidate still receives `1/61`, so a single
branch match stays eligible. Fused score descending and final `chunk_id`
ascending form a deterministic order. Candidates from requested Knowledge
Sources are then combined and bounded by the request's result limit. The global
reranker remains Chapter 05 work.

## Structured result and Evidence

The Retrieval Result distinguishes three outcomes:

- `completed`: at least one authorized Evidence Item was returned;
- `no_evidence`: retrieval succeeded but the authorized candidate set was
  empty; and
- `failed`: embedding, dense, lexical, or corpus availability failed.

One Knowledge Source failure cannot appear as complete multi-source success.
Every source retains its own status, Corpus Revision, authorized candidate
counts, timings, failed stage, and sanitized error code.

Every result also records `retrieval-v1` and a content-free retrieval config
fingerprint. The fingerprint covers the embedding model identity, exact-cosine
and English-websearch query semantics, dense/lexical/result budgets, and RRF
constant. The same configuration reproduces the same fingerprint; changing a
budget or model identity changes it. Chapter 05 will extend the same owned
configuration with reranking, relevance, and context-expansion inputs.

An Evidence Item is one authorized Indexed Chunk. It carries revision-scoped
identity, content, title, source path, a structured Markdown section locator,
and branch/fusion provenance. It never carries Access Grants, Group membership,
raw authorization data, prompt formatting, or an internal exception. Context
expansion and rerank fields remain empty until Chapter 05.

## Separate database capabilities

`DATABASE_URL` remains the Corpus Sync writer connection. Migration 0004 creates
the local `agentic_rag_retrieval` login, grants only the tables and authorization
function needed for retrieval, and makes transactions read-only by default.
Local development uses:

```text
RETRIEVAL_DATABASE_URL=
  postgresql://agentic_rag_retrieval@127.0.0.1:55432/agentic_rag
```

The Compose service uses host-local trust authentication for development, so
the example contains no password. Production authentication and secret-manager
design remain outside v1.

## Run the verification

```bash
docker compose up -d postgres
uv run pytest tests/unit/test_retrieval.py -q
uv run pytest tests/data_integration/test_retrieval_postgres.py -q
uv run pytest -q
uv run pyright
```

The L1 suite rejects authority-bearing fields, duplicate Knowledge Sources,
unbounded budgets, and Access Grant leakage. The L2 suite uses real PostgreSQL
and proves that a perfect dense plus lexical unauthorized canary never escapes,
the retrieval role cannot write, empty Access Scope returns zero visible
candidates, failures are sanitized, RRF is stable, and partial source failure is
explicit. No OpenAI or LangSmith credential is used.

Global reranking, relevance thresholds, neighboring context, Evidence Set
assembly, answer generation, graph routing, live embeddings, authentication,
production RLS, approximate vector indexes, and deployment remain outside this
chapter.
