"""Authorization-constrained dense, lexical, and fusion execution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from time import perf_counter
from typing import Sequence, cast

from langchain_core.embeddings import Embeddings
from psycopg import Connection, Error as PsycopgError
from psycopg_pool import ConnectionPool

from agentic_rag.authorization import AuthorizationSnapshot
from agentic_rag.corpus.models import (
    CorpusRevision,
    KnowledgeSource,
    SourceLocator,
)
from agentic_rag.retrieval.models import (
    CandidateCounts,
    EvidenceItem,
    KnowledgeSourceRetrievalOutcome,
    RetrievalConfig,
    RetrievalErrorCode,
    RetrievalProvenance,
    RetrievalRequest,
    RetrievalResult,
    RetrievalStage,
    RetrievalStatus,
    RetrievalTimings,
)


@dataclass
class _Candidate:
    chunk_id: str
    document_id: str
    knowledge_source: KnowledgeSource
    source_revision: str
    processing_revision: str
    corpus_revision: str
    heading_path: tuple[str, ...]
    content: str
    source_path: str
    title: str
    dense_rank: int | None = None
    dense_score: float | None = None
    lexical_rank: int | None = None
    lexical_score: float | None = None
    fused_rank: int = 0
    fused_score: float = 0.0


@dataclass(frozen=True)
class _SourceSearch:
    knowledge_source: KnowledgeSource
    corpus_revision: CorpusRevision | None
    status: RetrievalStatus
    candidates: tuple[_Candidate, ...]
    dense_count: int
    lexical_count: int
    dense_ms: float
    lexical_ms: float
    fusion_ms: float
    total_ms: float
    failed_stage: RetrievalStage | None = None
    error_code: RetrievalErrorCode | None = None


def retrieve(
    pool: ConnectionPool,
    snapshot: AuthorizationSnapshot | None,
    request: RetrievalRequest,
    config: RetrievalConfig,
    embedder: Embeddings,
) -> RetrievalResult:
    """Execute one bounded Retrieval Request under trusted authority."""

    started = perf_counter()
    if snapshot is None:
        return _failed_result(
            request,
            None,
            config,
            started,
            RetrievalStage.AUTHORIZATION,
            RetrievalErrorCode.AUTHORIZATION_CONTEXT_MISSING,
        )

    embedding_started = perf_counter()
    try:
        query_embedding = embedder.embed_query(request.query)
    except Exception:
        return _failed_result(
            request,
            snapshot,
            config,
            started,
            RetrievalStage.EMBEDDING,
            RetrievalErrorCode.EMBEDDING_FAILED,
            embedding_ms=_elapsed_ms(embedding_started),
        )
    embedding_ms = _elapsed_ms(embedding_started)
    if not _is_valid_embedding(query_embedding):
        return _failed_result(
            request,
            snapshot,
            config,
            started,
            RetrievalStage.EMBEDDING,
            RetrievalErrorCode.EMBEDDING_FAILED,
            embedding_ms=embedding_ms,
        )

    vector_literal = _vector_literal(query_embedding)
    searches = tuple(
        _retrieve_knowledge_source(
            pool,
            snapshot,
            request,
            config,
            knowledge_source,
            vector_literal,
        )
        for knowledge_source in request.knowledge_sources
    )
    candidates = sorted(
        (
            candidate
            for search in searches
            if search.status is not RetrievalStatus.FAILED
            for candidate in search.candidates
        ),
        key=lambda candidate: (
            -candidate.fused_score,
            candidate.knowledge_source.value,
            candidate.chunk_id,
        ),
    )
    selected = tuple(candidates[: request.result_limit])
    selected_counts = {
        source: sum(
            candidate.knowledge_source is source for candidate in selected
        )
        for source in request.knowledge_sources
    }
    failed_search = next(
        (
            search
            for search in searches
            if search.status is RetrievalStatus.FAILED
        ),
        None,
    )
    status = (
        RetrievalStatus.FAILED
        if failed_search is not None
        else RetrievalStatus.COMPLETED
        if selected
        else RetrievalStatus.NO_EVIDENCE
    )
    return RetrievalResult(
        retrieval_contract_version=config.contract_version,
        retrieval_config_fingerprint=_retrieval_config_fingerprint(
            request, config
        ),
        request_id=request.request_id,
        executed_query=request.query,
        requested_knowledge_sources=request.knowledge_sources,
        corpus_revisions=tuple(
            search.corpus_revision
            for search in searches
            if search.corpus_revision is not None
        ),
        access_scope_fingerprint=snapshot.revision,
        candidate_counts=CandidateCounts(
            dense=sum(search.dense_count for search in searches),
            lexical=sum(search.lexical_count for search in searches),
            deduplicated=sum(len(search.candidates) for search in searches),
            reranked=None,
            returned=len(selected),
        ),
        timings=RetrievalTimings(
            embedding_ms=embedding_ms,
            dense_ms=sum(search.dense_ms for search in searches),
            lexical_ms=sum(search.lexical_ms for search in searches),
            fusion_ms=sum(search.fusion_ms for search in searches),
            total_ms=_elapsed_ms(started),
        ),
        evidence_items=tuple(
            _evidence_item(request.request_id, candidate)
            for candidate in selected
        ),
        source_outcomes=tuple(
            _source_outcome(search, selected_counts[search.knowledge_source])
            for search in searches
        ),
        status=status,
        failed_stage=(failed_search.failed_stage if failed_search else None),
        error_code=(failed_search.error_code if failed_search else None),
    )


def _retrieve_knowledge_source(
    pool: ConnectionPool,
    snapshot: AuthorizationSnapshot,
    request: RetrievalRequest,
    config: RetrievalConfig,
    knowledge_source: KnowledgeSource,
    vector_literal: str,
) -> _SourceSearch:
    started = perf_counter()
    dense_ms = lexical_ms = fusion_ms = 0.0
    dense_count = lexical_count = 0
    stage = RetrievalStage.CORPUS
    corpus_revision: CorpusRevision | None = None
    try:
        with pool.connection() as connection:
            with connection.transaction():
                connection.execute(
                    "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"
                )
                corpus_revision_row = connection.execute(
                    """
                    SELECT corpus_revision FROM knowledge_sources
                    WHERE knowledge_source = %s AND corpus_revision IS NOT NULL
                    """,
                    (knowledge_source.value,),
                ).fetchone()
                if corpus_revision_row is None:
                    return _failed_source_search(
                        knowledge_source,
                        started,
                        RetrievalStage.CORPUS,
                        RetrievalErrorCode.CORPUS_UNAVAILABLE,
                    )
                corpus_revision = CorpusRevision(
                    knowledge_source=knowledge_source,
                    revision=str(corpus_revision_row[0]),
                )

                stage = RetrievalStage.DENSE
                dense_started = perf_counter()
                dense_candidates = _dense_candidates(
                    connection,
                    snapshot,
                    request,
                    knowledge_source,
                    vector_literal,
                )
                dense_ms = _elapsed_ms(dense_started)
                dense_count = len(dense_candidates)

                stage = RetrievalStage.LEXICAL
                lexical_started = perf_counter()
                lexical_candidates = _lexical_candidates(
                    connection, snapshot, request, knowledge_source
                )
                lexical_ms = _elapsed_ms(lexical_started)
                lexical_count = len(lexical_candidates)
    except PsycopgError:
        return _failed_source_search(
            knowledge_source,
            started,
            stage,
            RetrievalErrorCode.DENSE_FAILED
            if stage is RetrievalStage.DENSE
            else RetrievalErrorCode.LEXICAL_FAILED
            if stage is RetrievalStage.LEXICAL
            else RetrievalErrorCode.CORPUS_UNAVAILABLE,
            corpus_revision=corpus_revision,
            dense_count=dense_count,
            lexical_count=lexical_count,
            dense_ms=dense_ms,
            lexical_ms=lexical_ms,
        )

    fusion_started = perf_counter()
    candidates = _fuse_candidates(
        dense_candidates, lexical_candidates, config.rrf_k
    )
    fusion_ms = _elapsed_ms(fusion_started)
    return _SourceSearch(
        knowledge_source=knowledge_source,
        corpus_revision=corpus_revision,
        status=(
            RetrievalStatus.COMPLETED
            if candidates
            else RetrievalStatus.NO_EVIDENCE
        ),
        candidates=candidates,
        dense_count=dense_count,
        lexical_count=lexical_count,
        dense_ms=dense_ms,
        lexical_ms=lexical_ms,
        fusion_ms=fusion_ms,
        total_ms=_elapsed_ms(started),
    )


def _dense_candidates(
    connection: Connection,
    snapshot: AuthorizationSnapshot,
    request: RetrievalRequest,
    knowledge_source: KnowledgeSource,
    vector_literal: str,
) -> tuple[_Candidate, ...]:
    rows = connection.execute(
        """
        SELECT indexed_chunk.chunk_id,
               indexed_chunk.document_id,
               indexed_chunk.source_revision,
               indexed_chunk.processing_revision,
               knowledge_source.corpus_revision,
               indexed_chunk.heading_path,
               indexed_chunk.content,
               source_document.source_path,
               source_document.title,
               1 - (indexed_chunk.embedding <=> %(embedding)s::vector)
                   AS dense_score
        FROM indexed_chunks AS indexed_chunk
        JOIN source_documents AS source_document
          ON source_document.knowledge_source = indexed_chunk.knowledge_source
         AND source_document.document_id = indexed_chunk.document_id
        JOIN knowledge_sources AS knowledge_source
          ON knowledge_source.knowledge_source = indexed_chunk.knowledge_source
        WHERE indexed_chunk.knowledge_source = %(knowledge_source)s
          AND source_document_is_authorized(
              %(principal_id)s,
              indexed_chunk.knowledge_source,
              indexed_chunk.document_id
          )
        ORDER BY indexed_chunk.embedding <=> %(embedding)s::vector,
                 indexed_chunk.chunk_id
        LIMIT %(candidate_limit)s
        """,
        {
            "embedding": vector_literal,
            "knowledge_source": knowledge_source.value,
            "principal_id": snapshot.principal_id,
            "candidate_limit": request.dense_candidate_limit,
        },
    ).fetchall()
    return tuple(
        _candidate_from_row(
            row,
            knowledge_source,
            dense_rank=rank,
            dense_score=float(row[9]),
        )
        for rank, row in enumerate(rows, start=1)
    )


def _lexical_candidates(
    connection: Connection,
    snapshot: AuthorizationSnapshot,
    request: RetrievalRequest,
    knowledge_source: KnowledgeSource,
) -> tuple[_Candidate, ...]:
    rows = connection.execute(
        """
        WITH lexical_query AS (
            SELECT websearch_to_tsquery('english', %(query)s) AS value
        )
        SELECT indexed_chunk.chunk_id,
               indexed_chunk.document_id,
               indexed_chunk.source_revision,
               indexed_chunk.processing_revision,
               knowledge_source.corpus_revision,
               indexed_chunk.heading_path,
               indexed_chunk.content,
               source_document.source_path,
               source_document.title,
               ts_rank_cd(indexed_chunk.search_vector, lexical_query.value)
                   AS lexical_score
        FROM indexed_chunks AS indexed_chunk
        JOIN source_documents AS source_document
          ON source_document.knowledge_source = indexed_chunk.knowledge_source
         AND source_document.document_id = indexed_chunk.document_id
        JOIN knowledge_sources AS knowledge_source
          ON knowledge_source.knowledge_source = indexed_chunk.knowledge_source
        CROSS JOIN lexical_query
        WHERE indexed_chunk.knowledge_source = %(knowledge_source)s
          AND indexed_chunk.search_vector @@ lexical_query.value
          AND source_document_is_authorized(
              %(principal_id)s,
              indexed_chunk.knowledge_source,
              indexed_chunk.document_id
          )
        ORDER BY lexical_score DESC, indexed_chunk.chunk_id
        LIMIT %(candidate_limit)s
        """,
        {
            "query": request.query,
            "knowledge_source": knowledge_source.value,
            "principal_id": snapshot.principal_id,
            "candidate_limit": request.lexical_candidate_limit,
        },
    ).fetchall()
    return tuple(
        _candidate_from_row(
            row,
            knowledge_source,
            lexical_rank=rank,
            lexical_score=float(row[9]),
        )
        for rank, row in enumerate(rows, start=1)
    )


def _candidate_from_row(
    row: Sequence[object],
    knowledge_source: KnowledgeSource,
    *,
    dense_rank: int | None = None,
    dense_score: float | None = None,
    lexical_rank: int | None = None,
    lexical_score: float | None = None,
) -> _Candidate:
    return _Candidate(
        chunk_id=str(row[0]),
        document_id=str(row[1]),
        knowledge_source=knowledge_source,
        source_revision=str(row[2]),
        processing_revision=str(row[3]),
        corpus_revision=str(row[4]),
        heading_path=tuple(
            str(part) for part in cast(Sequence[object], row[5])
        ),
        content=str(row[6]),
        source_path=str(row[7]),
        title=str(row[8]),
        dense_rank=dense_rank,
        dense_score=dense_score,
        lexical_rank=lexical_rank,
        lexical_score=lexical_score,
    )


def _fuse_candidates(
    dense: Sequence[_Candidate], lexical: Sequence[_Candidate], rrf_k: int
) -> tuple[_Candidate, ...]:
    candidates: dict[str, _Candidate] = {
        candidate.chunk_id: candidate for candidate in dense
    }
    for lexical_candidate in lexical:
        candidate = candidates.get(lexical_candidate.chunk_id)
        if candidate is None:
            candidates[lexical_candidate.chunk_id] = lexical_candidate
        else:
            candidate.lexical_rank = lexical_candidate.lexical_rank
            candidate.lexical_score = lexical_candidate.lexical_score
    for candidate in candidates.values():
        candidate.fused_score = sum(
            1 / (rrf_k + rank)
            for rank in (candidate.dense_rank, candidate.lexical_rank)
            if rank is not None
        )
    ranked = sorted(
        candidates.values(),
        key=lambda candidate: (-candidate.fused_score, candidate.chunk_id),
    )
    for rank, candidate in enumerate(ranked, start=1):
        candidate.fused_rank = rank
    return tuple(ranked)


def _evidence_item(
    request_id: str, candidate: _Candidate
) -> EvidenceItem:
    return EvidenceItem(
        chunk_id=candidate.chunk_id,
        document_id=candidate.document_id,
        knowledge_source=candidate.knowledge_source,
        source_revision=candidate.source_revision,
        processing_revision=candidate.processing_revision,
        corpus_revision=candidate.corpus_revision,
        chunk_text=candidate.content,
        context_text=None,
        context_chunk_ids=(),
        source_path=candidate.source_path,
        title=candidate.title,
        source_locator=SourceLocator(section_path=candidate.heading_path),
        provenance=RetrievalProvenance(
            retrieval_request_id=request_id,
            dense_rank=candidate.dense_rank,
            dense_score=candidate.dense_score,
            lexical_rank=candidate.lexical_rank,
            lexical_score=candidate.lexical_score,
            fused_rank=candidate.fused_rank,
            fused_score=candidate.fused_score,
        ),
    )


def _source_outcome(
    search: _SourceSearch, returned_count: int
) -> KnowledgeSourceRetrievalOutcome:
    return KnowledgeSourceRetrievalOutcome(
        knowledge_source=search.knowledge_source,
        corpus_revision=search.corpus_revision,
        status=search.status,
        candidate_counts=CandidateCounts(
            dense=search.dense_count,
            lexical=search.lexical_count,
            deduplicated=len(search.candidates),
            reranked=None,
            returned=returned_count,
        ),
        timings=RetrievalTimings(
            embedding_ms=0,
            dense_ms=search.dense_ms,
            lexical_ms=search.lexical_ms,
            fusion_ms=search.fusion_ms,
            total_ms=search.total_ms,
        ),
        failed_stage=search.failed_stage,
        error_code=search.error_code,
    )


def _failed_source_search(
    knowledge_source: KnowledgeSource,
    started: float,
    failed_stage: RetrievalStage,
    error_code: RetrievalErrorCode,
    *,
    corpus_revision: CorpusRevision | None = None,
    dense_count: int = 0,
    lexical_count: int = 0,
    dense_ms: float = 0,
    lexical_ms: float = 0,
) -> _SourceSearch:
    return _SourceSearch(
        knowledge_source=knowledge_source,
        corpus_revision=corpus_revision,
        status=RetrievalStatus.FAILED,
        candidates=(),
        dense_count=dense_count,
        lexical_count=lexical_count,
        dense_ms=dense_ms,
        lexical_ms=lexical_ms,
        fusion_ms=0,
        total_ms=_elapsed_ms(started),
        failed_stage=failed_stage,
        error_code=error_code,
    )


def _failed_result(
    request: RetrievalRequest,
    snapshot: AuthorizationSnapshot | None,
    config: RetrievalConfig,
    started: float,
    failed_stage: RetrievalStage,
    error_code: RetrievalErrorCode,
    *,
    embedding_ms: float = 0,
) -> RetrievalResult:
    return RetrievalResult(
        retrieval_contract_version=config.contract_version,
        retrieval_config_fingerprint=_retrieval_config_fingerprint(
            request, config
        ),
        request_id=request.request_id,
        executed_query=request.query,
        requested_knowledge_sources=request.knowledge_sources,
        corpus_revisions=(),
        access_scope_fingerprint=(snapshot.revision if snapshot else None),
        candidate_counts=CandidateCounts(
            dense=0,
            lexical=0,
            deduplicated=0,
            reranked=None,
            returned=0,
        ),
        timings=RetrievalTimings(
            embedding_ms=embedding_ms,
            dense_ms=0,
            lexical_ms=0,
            fusion_ms=0,
            total_ms=_elapsed_ms(started),
        ),
        evidence_items=(),
        source_outcomes=(),
        status=RetrievalStatus.FAILED,
        failed_stage=failed_stage,
        error_code=error_code,
    )


def _is_valid_embedding(embedding: object) -> bool:
    if not isinstance(embedding, list) or not embedding:
        return False
    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(value)
        for value in embedding
    ):
        return False
    return any(value != 0 for value in embedding)


def _vector_literal(embedding: Sequence[float]) -> str:
    return "[" + ",".join(format(value, ".17g") for value in embedding) + "]"


def _retrieval_config_fingerprint(
    request: RetrievalRequest, config: RetrievalConfig
) -> str:
    fingerprint_input = {
        "config": config.model_dump(mode="json"),
        "dense_candidate_limit": request.dense_candidate_limit,
        "lexical_candidate_limit": request.lexical_candidate_limit,
        "result_limit": request.result_limit,
    }
    serialized = json.dumps(
        fingerprint_input, sort_keys=True, separators=(",", ":")
    )
    return f"retrieval_{sha256(serialized.encode()).hexdigest()}"


def _elapsed_ms(started: float) -> float:
    return (perf_counter() - started) * 1_000
