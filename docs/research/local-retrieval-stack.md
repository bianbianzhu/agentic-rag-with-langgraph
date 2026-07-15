# Compare local retrieval and persistence stacks

Date: 2026-07-13

## Decision

Use **PostgreSQL with pgvector and PostgreSQL full-text search** for the v1 Reference System. Implement hybrid retrieval as two explicit, Access Scope-constrained SQL branches—exact dense k-nearest-neighbor search and lexical `tsvector` search—then fuse their ranks and pass structured candidates to the reranker.

Use the database through a small retrieval Component rather than treating `PGVectorStore` as the whole retrieval design. `langchain-postgres` is a useful maintained vector integration, but the v1 contract also needs lexical retrieval, shared authorization predicates, rank fusion, and richer reranking inputs.

Qdrant is the preferred revisit candidate when measured filtered-ANN latency or recall makes PostgreSQL the bottleneck. LanceDB remains a strong embedded option for experiments, but its authorization boundary and current LangChain integration are weaker fits for this course.

## Scope and criteria

The comparison is for the already-agreed v1: local development, two Knowledge Sources, incremental Corpus Sync, runtime Principal and Access Scope, multi-turn LangGraph execution, deterministic tests, and no deployment work. It does not rank products by popularity or assume future scale.

Scores are 1 (poor) to 5 (strong). Weights reflect correctness and course value: Access Scope 25%, retrieval foundation 20%, Corpus Sync 15%, LangChain boundary 10%, deterministic testing 10%, and local operations plus persistence consolidation 20%.

| Candidate | Access Scope (25%) | Dense + lexical + reranking inputs (20%) | Corpus Sync (15%) | LangChain boundary (10%) | Deterministic tests (10%) | Local operations and persistence (20%) | Weighted total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PostgreSQL + pgvector + FTS | 5 | 4 | 5 | 4 | 5 | 5 | **4.70** |
| Qdrant | 5 | 5 | 4 | 5 | 5 | 2 | **4.25** |
| LanceDB OSS | 4 | 5 | 4 | 2 | 5 | 4 | **4.10** |

## Evidence

### PostgreSQL + pgvector + full-text search

- PostgreSQL provides `tsvector`/`tsquery`, relevance functions including `ts_rank` and `ts_rank_cd`, and GIN as its preferred text-search index. This supplies deterministic lexical retrieval without a second model or search service ([text-search functions](https://www.postgresql.org/docs/current/functions-textsearch.html), [text-search indexes](https://www.postgresql.org/docs/current/textsearch-indexes.html)).
- pgvector performs exact nearest-neighbor search by default and documents perfect recall for that mode. HNSW and IVFFlat are available later, but approximate indexes trade recall for speed. Critically, pgvector documents that a `WHERE` filter is applied after the approximate index scan and may yield too few matches; iterative scans, partial indexes, or partitioning mitigate but do not erase that design concern ([pgvector indexing and filtering](https://github.com/pgvector/pgvector#indexing)).
- Access Scope can be a bound predicate in both retrieval branches, so unauthorized rows never leave the storage query. PostgreSQL can additionally enforce row-level policies; when enabled, policy expressions determine which rows are visible, with default deny when no policy applies ([row security policies](https://www.postgresql.org/docs/current/ddl-rowsecurity.html)). For v1, an explicit predicate is simpler than mapping every Principal to a database role; RLS is a later defense-in-depth option, not a substitute for tests.
- Stable IDs plus `INSERT ... ON CONFLICT DO UPDATE` support idempotent additions and updates, while ordinary deletes remove documents absent from the current Knowledge Source. A transaction makes the additions, updates, and deletions visible all at once or not at all ([`INSERT ... ON CONFLICT`](https://www.postgresql.org/docs/current/sql-insert.html), [transactions](https://www.postgresql.org/docs/current/tutorial-transactions.html)). This is the strongest fit for Corpus Sync reconciliation.
- LangChain maintains `PGVectorStore` in the dedicated `langchain-postgres` package, including custom schemas, metadata columns, filters, async methods, scores, and deletion by ID ([LangChain PGVectorStore integration](https://docs.langchain.com/oss/python/integrations/vectorstores/pgvectorstore)). The hybrid Component should nevertheless use explicit SQL because that integration does not define the required PostgreSQL FTS plus fusion contract.
- Tests can omit ANN indexes, use fixed fake embeddings, and add a stable ID tie-breaker after distance/rank. That gives exact expected result sets instead of testing approximate-index accidents. Integration tests still require an ephemeral PostgreSQL process or container.
- PostgreSQL can also back LangGraph checkpoints through the first-party `PostgresSaver`/`AsyncPostgresSaver`; LangChain describes it as the production-oriented checkpointer. Agent Server manages checkpointing itself, so v1 should not manually attach a saver when running through `langgraph dev`, but selecting PostgreSQL avoids introducing a different persistence technology for direct or later standalone execution ([LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence)).

The main cost is code: PostgreSQL has no built-in dense/lexical fusion query. The Component must run both branches, retain their individual ranks, fuse them—RRF is a suitable untuned baseline—and expose one structured candidate type. This explicitness is useful course material and remains small.

### Qdrant

- Qdrant natively stores dense and sparse named vectors and its Query API fuses multiple prefetches with RRF or distribution-based score fusion. Its official guidance calls RRF the safe default when there is no evaluation set or trustworthy cross-retriever score scale ([hybrid queries](https://qdrant.tech/documentation/search/hybrid-queries/)).
- Payload indexes support boolean metadata filters, and Qdrant extends HNSW so indexed filters are applied during graph traversal rather than as a simple pre- or post-filter. This is the strongest filtered-ANN design in the comparison ([filtering](https://qdrant.tech/documentation/search/filtering/), [filterable HNSW](https://qdrant.tech/documentation/overview/#payload-indexes)). An Access Scope payload index must be created before ingestion.
- Stable point IDs make writes idempotent; re-uploading an ID overwrites the point. Points can be deleted by ID or filter, so additions, updates, and removals are all expressible for Corpus Sync ([point idempotence](https://qdrant.tech/documentation/manage-data/points/#idempotence), [delete points](https://api.qdrant.tech/api-reference/points/delete-points)). PostgreSQL still has the clearer all-or-nothing reconciliation transaction across document metadata and chunks.
- `langchain-qdrant` directly supports dense, sparse, and hybrid modes, metadata payloads, scores, and local in-memory/on-disk clients. Its documented hybrid example adds `fastembed` and a sparse encoder such as `Qdrant/bm25`, which is another dependency and model artifact to pin ([LangChain Qdrant integration](https://docs.langchain.com/oss/python/integrations/vectorstores/qdrant)).
- Qdrant local mode is designed for testing and small stores, and exact search can bypass HNSW for stable ground-truth tests ([local mode](https://docs.langchain.com/oss/python/integrations/vectorstores/qdrant#local-mode), [exact search](https://qdrant.tech/documentation/search/search/#search-api)).

Qdrant loses on v1 operational simplicity: it adds a retrieval server while multi-turn checkpoint persistence still needs Agent Server-managed storage or another database. Its quality advantage should be demonstrated by the evaluation dataset before accepting that extra boundary.

### LanceDB OSS

- LanceDB OSS is an embedded local library. It natively combines vector and full-text retrieval, defaults hybrid fusion to RRF, returns row identity and relevance fields, and allows a custom or cross-encoder reranker ([quickstart](https://docs.lancedb.com/quickstart), [hybrid search](https://docs.lancedb.com/search/hybrid-search)).
- Metadata `where` clauses prefilter by default, including both halves of a hybrid query, which can enforce Access Scope before scoring. Scalar indexes accelerate common filter columns ([metadata filtering](https://docs.lancedb.com/search/filtering), [hybrid prefiltering](https://docs.lancedb.com/search/hybrid-search#prefilter-vs-postfilter)). Unlike PostgreSQL row security, this remains an application-supplied predicate; omission is not default-deny.
- `merge_insert` supports update-on-match, insert-on-miss, and delete-when-missing-from-source, which closely matches Corpus Sync. However, OSS index maintenance is manual: `optimize()` incrementally updates vector, FTS, and scalar indexes and performs cleanup; unindexed rows remain searchable through slower flat scans ([table updates](https://docs.lancedb.com/tables/update), [reindexing](https://docs.lancedb.com/indexing/reindexing)).
- No-index or `bypass_vector_index()` search is exact with 100% recall, making temporary-directory fixtures deterministic ([exact vector search](https://docs.lancedb.com/search/vector-search#exact-vs-approximate-distances)).
- The current Python LangChain wrapper is in `langchain-community`, and its source repository is archived. It does not provide the same first-class native-hybrid path as `langchain-qdrant`; using LanceDB well would mean calling its SDK behind the project Component ([LangChain community LanceDB source](https://github.com/langchain-ai/langchain-community/blob/main/libs/community/langchain_community/vectorstores/lancedb.py)).

LanceDB is operationally light for a single process, but cross-process freshness is another explicit concern: local tables do not automatically check other writers by default unless `read_consistency_interval` or `checkout_latest` is used ([LanceDB consistency](https://docs.lancedb.com/tables/consistency)). That is awkward for a Corpus Sync process and a separately running Agent Server.

## Recommended v1 contract

1. Store source metadata, stable Document/Chunk IDs, content hashes, Access Scope metadata, chunk text, a `tsvector`, and dense embeddings in PostgreSQL. The exact schema for Access Scope grants remains a separate design decision.
2. Run dense and lexical retrieval independently with the **same bound Access Scope predicate**. Do not retrieve broadly and filter in Python.
3. Use exact pgvector search in v1 unless evaluation proves it misses the latency target. Add a deterministic secondary sort by Chunk ID. Add a GIN index for the lexical vector and ordinary indexes needed by the authorization predicate.
4. Fuse oversampled branch results with RRF, deduplicate by Chunk ID, and return structured reranking inputs: Document ID, Chunk ID, content, Knowledge Source, citation metadata, dense rank/score, lexical rank/score, and fused score.
5. Execute each Knowledge Source's Corpus Sync reconciliation in one transaction: skip unchanged hashes, upsert changed/new records, and delete records absent from the observed source snapshot.
6. Test the retrieval Component against an ephemeral PostgreSQL database with fake deterministic embeddings. Include explicit negative tests proving a Principal cannot retrieve chunks outside its Access Scope from either branch or the fused result.

## Revisit triggers

Re-evaluate Qdrant—not merely add a PostgreSQL ANN index—when evaluation shows that exact dense search misses the local latency target at representative corpus size, or when Access Scope-filtered ANN recall is unacceptable. Re-evaluate LanceDB if eliminating every local service becomes more important than database-enforced authorization and cross-process Corpus Sync behavior.

Before any switch, run the same frozen retrieval dataset against both candidates and compare Recall@k/MRR, Access Scope leakage (must remain zero), Corpus Sync correctness, p95 latency, and local setup time.

