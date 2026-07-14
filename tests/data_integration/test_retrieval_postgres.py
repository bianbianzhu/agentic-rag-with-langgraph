"""L2 verification for authorization-constrained hybrid retrieval."""

import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings
from psycopg.errors import ReadOnlySqlTransaction
from psycopg_pool import ConnectionPool

from agentic_rag.authorization import capture_authorization_snapshot
from agentic_rag.corpus import KnowledgeSource
from agentic_rag.database import apply_migrations, open_database_pool
from agentic_rag.retrieval import (
    RerankItem,
    RerankCandidate,
    RerankOutput,
    RetrievalConfig,
    RetrievalErrorCode,
    RetrievalRequest,
    RetrievalStage,
    RetrievalStatus,
    assemble_evidence_set,
    retrieve,
)
from agentic_rag.retrieval.evidence import expand_evidence_set_context
from reference_fixture import (
    FIXTURE_ROOT,
    FixtureEmbedder,
    load_reference_snapshot,
)


REPOSITORY_ROOT = Path(__file__).parents[2]
DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://agentic_rag@127.0.0.1:55432/agentic_rag",
)
RETRIEVAL_DATABASE_URL = os.environ.get(
    "TEST_RETRIEVAL_DATABASE_URL",
    "postgresql://agentic_rag_retrieval@127.0.0.1:55432/agentic_rag",
)


class FixedQueryEmbedder(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]


class FailingQueryEmbedder(FixedQueryEmbedder):
    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("INTERNAL_EMBEDDING_CANARY")


class ZeroQueryEmbedder(FixedQueryEmbedder):
    def embed_query(self, text: str) -> list[float]:
        return [0.0, 0.0, 0.0, 0.0]


def fixed_reranker(
    _query: str, candidates: tuple[RerankCandidate, ...]
) -> RerankOutput:
    return RerankOutput(
        items=tuple(
            RerankItem(
                chunk_id=candidate.chunk_id,
                score=1 - index / (2 * len(candidates)),
            )
            for index, candidate in enumerate(candidates)
        )
    )


@pytest.fixture(scope="module")
def database_pool() -> Iterator[ConnectionPool]:
    pool = open_database_pool(DATABASE_URL)
    apply_migrations(pool, REPOSITORY_ROOT / "migrations")
    yield pool
    pool.close()


@pytest.fixture(scope="module")
def retrieval_pool(
    database_pool: ConnectionPool,
) -> Iterator[ConnectionPool]:
    pool = open_database_pool(RETRIEVAL_DATABASE_URL)
    yield pool
    pool.close()


@pytest.fixture(autouse=True)
def empty_retrieval_state(database_pool: ConnectionPool) -> None:
    with database_pool.connection() as connection:
        connection.execute(
            """
            TRUNCATE access_grants, principal_group_memberships, groups,
                principals, sync_reports, indexed_chunks, source_documents,
                knowledge_sources CASCADE
            """
        )


def test_dense_and_lexical_branches_never_return_unauthorized_canary(
    database_pool: ConnectionPool,
    retrieval_pool: ConnectionPool,
) -> None:
    _insert_authorized_and_unauthorized_candidates(database_pool)
    snapshot = capture_authorization_snapshot(retrieval_pool, "bob")
    request = RetrievalRequest(
        request_id="request-unsafe-canary",
        query="rollback recovery",
        knowledge_sources=(KnowledgeSource.ENGINEERING_DOCS,),
        dense_candidate_limit=8,
        lexical_candidate_limit=8,
        result_limit=4,
    )

    result = retrieve(
        retrieval_pool,
        snapshot,
        request,
        _config(),
        FixedQueryEmbedder(),
        fixed_reranker,
    )

    assert result.status is RetrievalStatus.COMPLETED
    assert result.candidate_counts.model_dump() == {
        "dense": 1,
        "lexical": 1,
        "deduplicated": 1,
        "reranked": 1,
        "returned": 1,
    }
    assert [item.chunk_id for item in result.evidence_items] == [
        "chunk-authorized"
    ]
    assert result.evidence_items[0].provenance[0].model_dump() == {
        "retrieval_request_id": "request-unsafe-canary",
        "dense_rank": 1,
        "dense_score": pytest.approx(0.9701425),
        "lexical_rank": 1,
        "lexical_score": pytest.approx(0.1),
        "fused_rank": 1,
        "fused_score": pytest.approx(2 / 61),
        "rerank_rank": 1,
        "rerank_score": 1.0,
    }
    assert result.source_outcomes[0].corpus_revision == result.corpus_revisions[0]
    assert "UNAUTHORIZED_CANARY" not in result.model_dump_json()


def test_unknown_reranker_candidate_fails_closed(
    database_pool: ConnectionPool,
    retrieval_pool: ConnectionPool,
) -> None:
    _insert_authorized_and_unauthorized_candidates(database_pool)
    snapshot = capture_authorization_snapshot(retrieval_pool, "bob")

    result = retrieve(
        retrieval_pool,
        snapshot,
        _request("request-invalid-rerank"),
        _config(),
        FixedQueryEmbedder(),
        lambda _query, _candidates: RerankOutput(
            items=(RerankItem(chunk_id="unknown-chunk", score=1.0),)
        ),
    )

    assert result.status is RetrievalStatus.FAILED
    assert result.failed_stage is RetrievalStage.RERANK
    assert result.error_code is RetrievalErrorCode.RERANK_FAILED
    assert result.evidence_items == ()
    assert result.timings.rerank_ms > 0
    assert "UNAUTHORIZED_CANARY" not in result.model_dump_json()


def test_reranker_exception_fails_without_rrf_fallback(
    database_pool: ConnectionPool,
    retrieval_pool: ConnectionPool,
) -> None:
    _insert_rrf_candidates(database_pool)
    snapshot = capture_authorization_snapshot(retrieval_pool, "alice")

    def failing_reranker(
        _query: str, _candidates: tuple[RerankCandidate, ...]
    ) -> RerankOutput:
        raise RuntimeError("INTERNAL_RERANK_CANARY")

    result = retrieve(
        retrieval_pool,
        snapshot,
        _request("request-rerank-exception"),
        _config(),
        FixedQueryEmbedder(),
        failing_reranker,
    )

    assert result.status is RetrievalStatus.FAILED
    assert result.failed_stage is RetrievalStage.RERANK
    assert result.error_code is RetrievalErrorCode.RERANK_FAILED
    assert result.evidence_items == ()
    assert result.timings.rerank_ms > 0
    assert "INTERNAL_RERANK_CANARY" not in result.model_dump_json()


@pytest.mark.parametrize("invalid_output", ["duplicate", "missing", "ascending"])
def test_structurally_invalid_reranker_output_fails_closed(
    database_pool: ConnectionPool,
    retrieval_pool: ConnectionPool,
    invalid_output: str,
) -> None:
    _insert_rrf_candidates(database_pool)
    snapshot = capture_authorization_snapshot(retrieval_pool, "alice")

    def invalid_reranker(
        _query: str, candidates: tuple[RerankCandidate, ...]
    ) -> RerankOutput:
        if invalid_output == "duplicate":
            return RerankOutput(
                items=(
                    RerankItem(chunk_id=candidates[0].chunk_id, score=1),
                    RerankItem(chunk_id=candidates[0].chunk_id, score=0.9),
                    RerankItem(chunk_id=candidates[2].chunk_id, score=0.8),
                )
            )
        if invalid_output == "missing":
            return RerankOutput(
                items=(
                    RerankItem(chunk_id=candidates[0].chunk_id, score=1),
                    RerankItem(chunk_id=candidates[1].chunk_id, score=0.9),
                )
            )
        return RerankOutput(
            items=tuple(
                RerankItem(chunk_id=candidate.chunk_id, score=index / 10)
                for index, candidate in enumerate(candidates, start=1)
            )
        )

    result = retrieve(
        retrieval_pool,
        snapshot,
        _request(f"request-invalid-{invalid_output}"),
        _config(),
        FixedQueryEmbedder(),
        invalid_reranker,
    )

    assert result.status is RetrievalStatus.FAILED
    assert result.failed_stage is RetrievalStage.RERANK
    assert result.error_code is RetrievalErrorCode.RERANK_FAILED
    assert result.evidence_items == ()


def test_retrieval_database_role_cannot_write(
    retrieval_pool: ConnectionPool,
) -> None:
    with pytest.raises(ReadOnlySqlTransaction):
        with retrieval_pool.connection() as connection:
            connection.execute(
                "INSERT INTO principals (principal_id) VALUES ('mallory')"
            )


def test_empty_access_scope_returns_no_evidence_without_inaccessible_counts(
    database_pool: ConnectionPool,
    retrieval_pool: ConnectionPool,
) -> None:
    _insert_authorized_and_unauthorized_candidates(database_pool)
    snapshot = capture_authorization_snapshot(retrieval_pool, "carol")

    result = retrieve(
        retrieval_pool,
        snapshot,
        _request("request-no-evidence"),
        _config(),
        FixedQueryEmbedder(),
        fixed_reranker,
    )

    assert result.status is RetrievalStatus.NO_EVIDENCE
    assert result.candidate_counts.model_dump() == {
        "dense": 0,
        "lexical": 0,
        "deduplicated": 0,
        "reranked": 0,
        "returned": 0,
    }
    assert result.evidence_items == ()
    assert "chunk-unauthorized" not in result.model_dump_json()


def test_embedding_failure_is_sanitized_as_a_failed_result(
    database_pool: ConnectionPool,
    retrieval_pool: ConnectionPool,
) -> None:
    _insert_authorized_and_unauthorized_candidates(database_pool)
    snapshot = capture_authorization_snapshot(retrieval_pool, "bob")

    result = retrieve(
        retrieval_pool,
        snapshot,
        _request("request-embedding-failure"),
        _config(),
        FailingQueryEmbedder(),
        fixed_reranker,
    )

    assert result.status is RetrievalStatus.FAILED
    assert result.failed_stage is RetrievalStage.EMBEDDING
    assert result.error_code is RetrievalErrorCode.EMBEDDING_FAILED
    assert result.evidence_items == ()
    assert "INTERNAL_EMBEDDING_CANARY" not in result.model_dump_json()


def test_zero_query_embedding_fails_before_dense_retrieval(
    database_pool: ConnectionPool,
    retrieval_pool: ConnectionPool,
) -> None:
    _insert_authorized_and_unauthorized_candidates(database_pool)
    snapshot = capture_authorization_snapshot(retrieval_pool, "bob")

    result = retrieve(
        retrieval_pool,
        snapshot,
        _request("request-zero-embedding"),
        _config(),
        ZeroQueryEmbedder(),
        fixed_reranker,
    )

    assert result.status is RetrievalStatus.FAILED
    assert result.failed_stage is RetrievalStage.EMBEDDING
    assert result.error_code is RetrievalErrorCode.EMBEDDING_FAILED
    assert result.candidate_counts.dense == 0


def test_rrf_keeps_single_branch_candidates_and_is_stable(
    database_pool: ConnectionPool,
    retrieval_pool: ConnectionPool,
) -> None:
    _insert_rrf_candidates(database_pool)
    snapshot = capture_authorization_snapshot(retrieval_pool, "alice")
    request = RetrievalRequest(
        request_id="request-rrf",
        query="rollback recovery",
        knowledge_sources=(KnowledgeSource.ENGINEERING_DOCS,),
        dense_candidate_limit=2,
        lexical_candidate_limit=2,
        result_limit=3,
    )

    first = retrieve(
        retrieval_pool,
        snapshot,
        request,
        _config(),
        FixedQueryEmbedder(),
        fixed_reranker,
    )
    second = retrieve(
        retrieval_pool,
        snapshot,
        request,
        _config(),
        FixedQueryEmbedder(),
        fixed_reranker,
    )

    assert [item.chunk_id for item in first.evidence_items] == [
        "chunk-both",
        "chunk-dense",
        "chunk-lexical",
    ]
    assert first.candidate_counts.deduplicated == 3
    assert [
        item.provenance[0].model_dump() for item in first.evidence_items
    ] == [item.provenance[0].model_dump() for item in second.evidence_items]
    assert first.evidence_items[0].provenance[0].dense_rank == 2
    assert first.evidence_items[0].provenance[0].lexical_rank == 1
    assert first.evidence_items[1].provenance[0].lexical_rank is None
    assert first.evidence_items[2].provenance[0].dense_rank is None
    assert (
        first.retrieval_config_fingerprint
        == second.retrieval_config_fingerprint
    )

    changed_config = retrieve(
        retrieval_pool,
        snapshot,
        request,
        RetrievalConfig(embedding_model="deterministic-test-v2"),
        FixedQueryEmbedder(),
        fixed_reranker,
    )
    changed_budget = retrieve(
        retrieval_pool,
        snapshot,
        request.model_copy(update={"dense_candidate_limit": 1}),
        _config(),
        FixedQueryEmbedder(),
        fixed_reranker,
    )
    changed_reranking = retrieve(
        retrieval_pool,
        snapshot,
        request,
        _config().model_copy(update={"relevance_threshold": 0.6}),
        FixedQueryEmbedder(),
        fixed_reranker,
    )
    assert len(
        {
            first.retrieval_config_fingerprint,
            changed_config.retrieval_config_fingerprint,
            changed_budget.retrieval_config_fingerprint,
            changed_reranking.retrieval_config_fingerprint,
        }
    ) == 4


def test_one_global_reranker_controls_order_threshold_and_provenance(
    database_pool: ConnectionPool,
    retrieval_pool: ConnectionPool,
) -> None:
    _insert_rrf_candidates(database_pool)
    snapshot = capture_authorization_snapshot(retrieval_pool, "alice")
    calls: list[tuple[str, ...]] = []

    def reverse_reranker(
        _query: str, candidates: tuple[RerankCandidate, ...]
    ) -> RerankOutput:
        calls.append(tuple(candidate.chunk_id for candidate in candidates))
        return RerankOutput(
            items=(
                RerankItem(chunk_id=candidates[2].chunk_id, score=1.0),
                RerankItem(chunk_id=candidates[1].chunk_id, score=0.8),
                RerankItem(chunk_id=candidates[0].chunk_id, score=0.2),
            )
        )

    result = retrieve(
        retrieval_pool,
        snapshot,
        _request("request-global-rerank").model_copy(
            update={"dense_candidate_limit": 2, "lexical_candidate_limit": 2}
        ),
        _config(),
        FixedQueryEmbedder(),
        reverse_reranker,
    )

    assert calls == [("chunk-both", "chunk-dense", "chunk-lexical")]
    assert [item.chunk_id for item in result.evidence_items] == [
        "chunk-lexical",
        "chunk-dense",
    ]
    assert [item.provenance[0].rerank_rank for item in result.evidence_items] == [
        1,
        2,
    ]
    assert result.candidate_counts.reranked == 3


def test_reranker_is_global_across_selected_knowledge_sources(
    database_pool: ConnectionPool,
    retrieval_pool: ConnectionPool,
) -> None:
    _insert_cross_source_candidates(database_pool)
    snapshot = capture_authorization_snapshot(retrieval_pool, "alice")
    calls: list[tuple[str, ...]] = []

    def reverse_reranker(
        _query: str, candidates: tuple[RerankCandidate, ...]
    ) -> RerankOutput:
        calls.append(tuple(candidate.chunk_id for candidate in candidates))
        return RerankOutput(
            items=tuple(
                RerankItem(chunk_id=candidate.chunk_id, score=1 - index / 4)
                for index, candidate in enumerate(reversed(candidates))
            )
        )

    result = retrieve(
        retrieval_pool,
        snapshot,
        RetrievalRequest(
            request_id="request-cross-source",
            query="rollback recovery",
            knowledge_sources=(
                KnowledgeSource.ENGINEERING_DOCS,
                KnowledgeSource.OPERATIONAL_RUNBOOKS,
            ),
            dense_candidate_limit=1,
            lexical_candidate_limit=1,
            result_limit=2,
        ),
        _config(),
        FixedQueryEmbedder(),
        reverse_reranker,
    )

    assert calls == [("chunk-engineering", "chunk-runbook")]
    assert [item.chunk_id for item in result.evidence_items] == [
        "chunk-runbook",
        "chunk-engineering",
    ]


def test_below_threshold_candidates_are_a_successful_no_evidence_result(
    database_pool: ConnectionPool,
    retrieval_pool: ConnectionPool,
) -> None:
    _insert_rrf_candidates(database_pool)
    snapshot = capture_authorization_snapshot(retrieval_pool, "alice")

    def irrelevant_reranker(
        _query: str, candidates: tuple[RerankCandidate, ...]
    ) -> RerankOutput:
        return RerankOutput(
            items=tuple(
                RerankItem(chunk_id=candidate.chunk_id, score=0.4)
                for candidate in candidates
            )
        )

    result = retrieve(
        retrieval_pool,
        snapshot,
        _request("request-below-threshold"),
        _config(),
        FixedQueryEmbedder(),
        irrelevant_reranker,
    )

    assert result.status is RetrievalStatus.NO_EVIDENCE
    assert result.candidate_counts.deduplicated == 3
    assert result.candidate_counts.reranked == 3
    assert result.candidate_counts.returned == 0
    assert result.evidence_items == ()


def test_deduplicated_count_is_measured_before_fusion_cutoff(
    database_pool: ConnectionPool,
    retrieval_pool: ConnectionPool,
) -> None:
    _insert_rrf_candidates(database_pool)
    snapshot = capture_authorization_snapshot(retrieval_pool, "alice")

    result = retrieve(
        retrieval_pool,
        snapshot,
        _request("request-fusion-cutoff").model_copy(
            update={"dense_candidate_limit": 2, "lexical_candidate_limit": 2}
        ),
        _config().model_copy(update={"fusion_candidate_limit": 1}),
        FixedQueryEmbedder(),
        fixed_reranker,
    )

    assert result.candidate_counts.deduplicated == 3
    assert result.candidate_counts.reranked == 1
    assert result.source_outcomes[0].candidate_counts.deduplicated == 3


def test_frozen_fixture_reranking_meets_recall_mrr_and_safety_gates(
    database_pool: ConnectionPool,
    retrieval_pool: ConnectionPool,
) -> None:
    labels = json.loads(
        (FIXTURE_ROOT / "trusted/retrieval-labels.json").read_text(
            encoding="utf-8"
        )
    )
    case = labels["cases"]["rollback-incompatibility"]
    expected = case["expected_evidence"]
    load_reference_snapshot(database_pool, case["snapshot"])
    snapshot = capture_authorization_snapshot(
        retrieval_pool, case["principal_id"]
    )
    with database_pool.connection() as connection:
        expected_row = connection.execute(
            """
            SELECT indexed_chunk.chunk_id
            FROM indexed_chunks AS indexed_chunk
            JOIN source_documents AS source_document
              ON source_document.knowledge_source = indexed_chunk.knowledge_source
             AND source_document.document_id = indexed_chunk.document_id
            WHERE indexed_chunk.knowledge_source = %(knowledge_source)s
              AND source_document.source_path = %(source_path)s
              AND indexed_chunk.heading_path = %(section_path)s
            """,
            {
                "knowledge_source": expected["knowledge_source"],
                "source_path": expected["source_path"],
                "section_path": expected["section_path"],
            },
        ).fetchone()
        assert expected_row is not None
        expected_chunk_id = str(expected_row[0])
        forbidden_sources = set(case["forbidden_knowledge_sources"])
        forbidden_chunk_ids = {
            str(row[0])
            for row in connection.execute(
                "SELECT chunk_id, knowledge_source FROM indexed_chunks"
            ).fetchall()
            if str(row[1]) in forbidden_sources
        }

    def labeled_reranker(
        _query: str, candidates: tuple[RerankCandidate, ...]
    ) -> RerankOutput:
        ordered = sorted(
            candidates,
            key=lambda candidate: (
                candidate.chunk_id != expected_chunk_id,
                candidate.chunk_id,
            ),
        )
        return RerankOutput(
            items=tuple(
                RerankItem(
                    chunk_id=candidate.chunk_id,
                    score=1 - index / (2 * len(ordered)),
                )
                for index, candidate in enumerate(ordered)
            )
        )

    request = RetrievalRequest(
        request_id="request-frozen-gate",
        query=case["query"],
        knowledge_sources=tuple(
            KnowledgeSource(value) for value in case["knowledge_sources"]
        ),
        dense_candidate_limit=50,
        lexical_candidate_limit=50,
        result_limit=case["result_limit"],
    )
    first = retrieve(
        retrieval_pool,
        snapshot,
        request,
        _config(),
        FixtureEmbedder(),
        labeled_reranker,
    )
    second = retrieve(
        retrieval_pool,
        snapshot,
        request,
        _config(),
        FixtureEmbedder(),
        labeled_reranker,
    )
    first_ids = [item.chunk_id for item in first.evidence_items]
    second_ids = [item.chunk_id for item in second.evidence_items]

    assert expected_chunk_id in first_ids[:5], expected["alias"]
    assert first_ids.index(expected_chunk_id) == 0, expected["alias"]
    assert first_ids == second_ids
    assert len(first_ids) <= request.result_limit
    assert forbidden_chunk_ids.isdisjoint(first_ids)


def test_context_expansion_rechecks_authority_and_total_token_budget(
    database_pool: ConnectionPool,
    retrieval_pool: ConnectionPool,
) -> None:
    _insert_context_candidates(database_pool)
    snapshot = capture_authorization_snapshot(retrieval_pool, "alice")
    request = RetrievalRequest(
        request_id="request-context",
        query="rollback recovery",
        knowledge_sources=(KnowledgeSource.ENGINEERING_DOCS,),
        dense_candidate_limit=1,
        lexical_candidate_limit=1,
        result_limit=1,
    )

    context_config = _config().model_copy(update={"evidence_token_limit": 11})
    retrieval_result = retrieve(
        retrieval_pool,
        snapshot,
        request,
        context_config,
        FixedQueryEmbedder(),
        fixed_reranker,
    )
    matched = assemble_evidence_set((retrieval_result,), context_config)
    assert matched.evidence_items[0].context_text is None
    with pytest.raises(ValueError, match="context configuration disagree"):
        expand_evidence_set_context(
            retrieval_pool,
            snapshot,
            matched,
            context_config.model_copy(update={"context_window": 2}),
        )
    result = expand_evidence_set_context(
        retrieval_pool,
        snapshot,
        matched,
        context_config,
    )

    assert result.complete is True
    assert result.context_expanded is True
    assert result.evidence_items[0].chunk_id == "chunk-context-match"
    assert result.evidence_items[0].context_chunk_ids == (
        "chunk-context-before",
    )
    assert result.evidence_items[0].context_text == "setup details"
    assert "UNAUTHORIZED_CONTEXT_CANARY" not in result.model_dump_json()


def test_context_expansion_rejects_corpus_revision_drift(
    database_pool: ConnectionPool,
    retrieval_pool: ConnectionPool,
) -> None:
    _insert_context_candidates(database_pool)
    snapshot = capture_authorization_snapshot(retrieval_pool, "alice")

    def drift_corpus_revision(
        query: str, candidates: tuple[RerankCandidate, ...]
    ) -> RerankOutput:
        with database_pool.connection() as connection:
            connection.execute(
                """
                UPDATE knowledge_sources SET corpus_revision = 'corpus-test-v2'
                WHERE knowledge_source = 'engineering-docs'
                """
            )
        return fixed_reranker(query, candidates)

    retrieval_result = retrieve(
        retrieval_pool,
        snapshot,
        RetrievalRequest(
            request_id="request-context-drift",
            query="rollback recovery",
            knowledge_sources=(KnowledgeSource.ENGINEERING_DOCS,),
            dense_candidate_limit=1,
            lexical_candidate_limit=1,
            result_limit=1,
        ),
        _config(),
        FixedQueryEmbedder(),
        drift_corpus_revision,
    )
    matched = assemble_evidence_set((retrieval_result,), _config())
    result = expand_evidence_set_context(
        retrieval_pool, snapshot, matched, _config()
    )

    assert result.complete is False
    assert result.failed_stage is RetrievalStage.CONTEXT
    assert result.error_code is RetrievalErrorCode.CORPUS_UNAVAILABLE
    assert result.evidence_items == ()
    assert result.context_ms > 0
    assert retrieval_result.candidate_counts.reranked == 1


def test_missing_knowledge_source_cannot_look_like_complete_success(
    database_pool: ConnectionPool,
    retrieval_pool: ConnectionPool,
) -> None:
    _insert_rrf_candidates(database_pool)
    snapshot = capture_authorization_snapshot(retrieval_pool, "alice")
    request = RetrievalRequest(
        request_id="request-partial-source-failure",
        query="rollback recovery",
        knowledge_sources=(
            KnowledgeSource.ENGINEERING_DOCS,
            KnowledgeSource.OPERATIONAL_RUNBOOKS,
        ),
        dense_candidate_limit=2,
        lexical_candidate_limit=2,
        result_limit=3,
    )

    result = retrieve(
        retrieval_pool,
        snapshot,
        request,
        _config(),
        FixedQueryEmbedder(),
        fixed_reranker,
    )

    assert result.status is RetrievalStatus.FAILED
    assert result.error_code is RetrievalErrorCode.CORPUS_UNAVAILABLE
    assert [outcome.status for outcome in result.source_outcomes] == [
        RetrievalStatus.COMPLETED,
        RetrievalStatus.FAILED,
    ]


def test_dense_failure_preserves_the_queried_corpus_revision(
    database_pool: ConnectionPool,
    retrieval_pool: ConnectionPool,
) -> None:
    _insert_dimension_mismatch_candidate(database_pool)
    snapshot = capture_authorization_snapshot(retrieval_pool, "alice")

    result = retrieve(
        retrieval_pool,
        snapshot,
        _request("request-dense-failure"),
        _config(),
        FixedQueryEmbedder(),
        fixed_reranker,
    )

    assert result.status is RetrievalStatus.FAILED
    assert result.failed_stage is RetrievalStage.DENSE
    assert result.error_code is RetrievalErrorCode.DENSE_FAILED
    assert result.corpus_revisions == (
        result.source_outcomes[0].corpus_revision,
    )


def _insert_authorized_and_unauthorized_candidates(
    pool: ConnectionPool,
) -> None:
    with pool.connection() as connection:
        connection.execute(
            """
            INSERT INTO knowledge_sources (knowledge_source, corpus_revision)
            VALUES ('engineering-docs', 'corpus-test-v1')
            """
        )
        connection.execute(
            """
            INSERT INTO principals (principal_id) VALUES ('bob'), ('carol');
            INSERT INTO groups (group_id)
            VALUES ('payments-engineering'), ('payments-on-call');
            INSERT INTO principal_group_memberships (principal_id, group_id)
            VALUES ('bob', 'payments-engineering');
            """
        )
        connection.execute(
            """
            INSERT INTO source_documents (
                knowledge_source, document_id, source_path, source_revision,
                processing_revision, title, raw_content
            ) VALUES
                ('engineering-docs', 'document-authorized',
                 'payments/authorized.md', 'source-authorized',
                 'processing-test-v1', 'Authorized recovery',
                 'Authorized content.'),
                ('engineering-docs', 'document-unauthorized',
                 'payments/unauthorized.md', 'source-unauthorized',
                 'processing-test-v1', 'Restricted recovery',
                 'Restricted content.');
            """
        )
        connection.execute(
            """
            INSERT INTO indexed_chunks (
                chunk_id, knowledge_source, document_id, source_revision,
                processing_revision, ordinal, heading_path, content, embedding
            ) VALUES
                ('chunk-authorized', 'engineering-docs',
                 'document-authorized', 'source-authorized',
                 'processing-test-v1', 0, ARRAY['Recovery'],
                 'rollback recovery approved steps', '[0.8,0.2,0,0]'::vector),
                ('chunk-unauthorized', 'engineering-docs',
                 'document-unauthorized', 'source-unauthorized',
                 'processing-test-v1', 0, ARRAY['Recovery'],
                 'rollback recovery UNAUTHORIZED_CANARY', '[1,0,0,0]'::vector);
            """
        )
        connection.execute(
            """
            INSERT INTO access_grants (
                knowledge_source, document_id, grant_type, group_id
            ) VALUES
                ('engineering-docs', 'document-authorized', 'group',
                 'payments-engineering'),
                ('engineering-docs', 'document-unauthorized', 'group',
                 'payments-on-call');
            """
        )


def _insert_rrf_candidates(pool: ConnectionPool) -> None:
    with pool.connection() as connection:
        connection.execute(
            """
            INSERT INTO knowledge_sources (knowledge_source, corpus_revision)
            VALUES ('engineering-docs', 'corpus-test-v1');
            INSERT INTO principals (principal_id) VALUES ('alice');
            """
        )
        connection.execute(
            """
            INSERT INTO source_documents (
                knowledge_source, document_id, source_path, source_revision,
                processing_revision, title, raw_content
            ) VALUES
                ('engineering-docs', 'document-both', 'both.md', 'source-both',
                 'processing-test-v1', 'Both branches', 'Both.'),
                ('engineering-docs', 'document-dense', 'dense.md',
                 'source-dense', 'processing-test-v1', 'Dense only', 'Dense.'),
                ('engineering-docs', 'document-lexical', 'lexical.md',
                 'source-lexical', 'processing-test-v1', 'Lexical only',
                 'Lexical.');
            INSERT INTO indexed_chunks (
                chunk_id, knowledge_source, document_id, source_revision,
                processing_revision, ordinal, heading_path, content, embedding
            ) VALUES
                ('chunk-both', 'engineering-docs', 'document-both',
                 'source-both', 'processing-test-v1', 0, ARRAY['Recovery'],
                 'rollback recovery procedure', '[0.9,0.1,0,0]'::vector),
                ('chunk-dense', 'engineering-docs', 'document-dense',
                 'source-dense', 'processing-test-v1', 0, ARRAY['Semantics'],
                 'vector semantics only', '[1,0,0,0]'::vector),
                ('chunk-lexical', 'engineering-docs', 'document-lexical',
                 'source-lexical', 'processing-test-v1', 0, ARRAY['Glossary'],
                 'rollback recovery glossary', '[0,1,0,0]'::vector);
            INSERT INTO access_grants (
                knowledge_source, document_id, grant_type, principal_id
            ) VALUES
                ('engineering-docs', 'document-both', 'principal', 'alice'),
                ('engineering-docs', 'document-dense', 'principal', 'alice'),
                ('engineering-docs', 'document-lexical', 'principal', 'alice');
            """
        )


def _insert_dimension_mismatch_candidate(pool: ConnectionPool) -> None:
    with pool.connection() as connection:
        connection.execute(
            """
            INSERT INTO knowledge_sources (knowledge_source, corpus_revision)
            VALUES ('engineering-docs', 'corpus-test-v1');
            INSERT INTO principals (principal_id) VALUES ('alice');
            INSERT INTO source_documents (
                knowledge_source, document_id, source_path, source_revision,
                processing_revision, title, raw_content
            ) VALUES (
                'engineering-docs', 'document-mismatch', 'mismatch.md',
                'source-mismatch', 'processing-test-v1', 'Mismatch', 'Mismatch.'
            );
            INSERT INTO indexed_chunks (
                chunk_id, knowledge_source, document_id, source_revision,
                processing_revision, ordinal, heading_path, content, embedding
            ) VALUES (
                'chunk-mismatch', 'engineering-docs', 'document-mismatch',
                'source-mismatch', 'processing-test-v1', 0, ARRAY['Mismatch'],
                'rollback recovery mismatch', '[1,0,0]'::vector
            );
            INSERT INTO access_grants (
                knowledge_source, document_id, grant_type, principal_id
            ) VALUES (
                'engineering-docs', 'document-mismatch', 'principal', 'alice'
            );
            """
        )


def _insert_context_candidates(pool: ConnectionPool) -> None:
    with pool.connection() as connection:
        connection.execute(
            """
            INSERT INTO knowledge_sources (knowledge_source, corpus_revision)
            VALUES ('engineering-docs', 'corpus-test-v1');
            INSERT INTO principals (principal_id) VALUES ('alice');
            INSERT INTO source_documents (
                knowledge_source, document_id, source_path, source_revision,
                processing_revision, title, raw_content
            ) VALUES
                ('engineering-docs', 'document-context', 'context.md',
                 'source-context', 'processing-test-v1', 'Context', 'Context.'),
                ('engineering-docs', 'document-forbidden', 'forbidden.md',
                 'source-forbidden', 'processing-test-v1', 'Forbidden',
                 'Forbidden.');
            INSERT INTO indexed_chunks (
                chunk_id, knowledge_source, document_id, source_revision,
                processing_revision, ordinal, heading_path, content, embedding
            ) VALUES
                ('chunk-context-before', 'engineering-docs',
                 'document-context', 'source-context', 'processing-test-v1', 0,
                 ARRAY['Before'], 'setup details', '[0,1,0,0]'::vector),
                ('chunk-context-match', 'engineering-docs',
                 'document-context', 'source-context', 'processing-test-v1', 1,
                 ARRAY['Match'], 'rollback recovery root cause',
                 '[1,0,0,0]'::vector),
                ('chunk-context-after', 'engineering-docs',
                 'document-context', 'source-context', 'processing-test-v1', 2,
                 ARRAY['After'], 'followup actions', '[0,1,0,0]'::vector),
                ('chunk-forbidden', 'engineering-docs', 'document-forbidden',
                 'source-forbidden', 'processing-test-v1', 1,
                 ARRAY['Forbidden'],
                 'rollback recovery UNAUTHORIZED_CONTEXT_CANARY',
                 '[1,0,0,0]'::vector);
            INSERT INTO access_grants (
                knowledge_source, document_id, grant_type, principal_id
            ) VALUES (
                'engineering-docs', 'document-context', 'principal', 'alice'
            );
            """
        )


def _insert_cross_source_candidates(pool: ConnectionPool) -> None:
    with pool.connection() as connection:
        connection.execute(
            """
            INSERT INTO knowledge_sources (knowledge_source, corpus_revision)
            VALUES
                ('engineering-docs', 'corpus-engineering-v1'),
                ('operational-runbooks', 'corpus-runbooks-v1');
            INSERT INTO principals (principal_id) VALUES ('alice');
            INSERT INTO source_documents (
                knowledge_source, document_id, source_path, source_revision,
                processing_revision, title, raw_content
            ) VALUES
                ('engineering-docs', 'document-engineering', 'engineering.md',
                 'source-engineering', 'processing-test-v1', 'Engineering',
                 'Engineering.'),
                ('operational-runbooks', 'document-runbook', 'runbook.md',
                 'source-runbook', 'processing-test-v1', 'Runbook', 'Runbook.');
            INSERT INTO indexed_chunks (
                chunk_id, knowledge_source, document_id, source_revision,
                processing_revision, ordinal, heading_path, content, embedding
            ) VALUES
                ('chunk-engineering', 'engineering-docs',
                 'document-engineering', 'source-engineering',
                 'processing-test-v1', 0, ARRAY['Recovery'],
                 'rollback recovery engineering', '[1,0,0,0]'::vector),
                ('chunk-runbook', 'operational-runbooks', 'document-runbook',
                 'source-runbook', 'processing-test-v1', 0,
                 ARRAY['Recovery'], 'rollback recovery runbook',
                 '[1,0,0,0]'::vector);
            INSERT INTO access_grants (
                knowledge_source, document_id, grant_type, principal_id
            ) VALUES
                ('engineering-docs', 'document-engineering', 'principal',
                 'alice'),
                ('operational-runbooks', 'document-runbook', 'principal',
                 'alice');
            """
        )


def _request(request_id: str) -> RetrievalRequest:
    return RetrievalRequest(
        request_id=request_id,
        query="rollback recovery",
        knowledge_sources=(KnowledgeSource.ENGINEERING_DOCS,),
        dense_candidate_limit=8,
        lexical_candidate_limit=8,
        result_limit=4,
    )


def _config() -> RetrievalConfig:
    return RetrievalConfig(embedding_model="deterministic-test-v1")
