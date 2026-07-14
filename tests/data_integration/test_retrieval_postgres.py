"""L2 verification for authorization-constrained hybrid retrieval."""

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
    RetrievalConfig,
    RetrievalErrorCode,
    RetrievalRequest,
    RetrievalStage,
    RetrievalStatus,
    retrieve,
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
    )

    assert result.status is RetrievalStatus.COMPLETED
    assert result.candidate_counts.model_dump() == {
        "dense": 1,
        "lexical": 1,
        "deduplicated": 1,
        "reranked": None,
        "returned": 1,
    }
    assert [item.chunk_id for item in result.evidence_items] == [
        "chunk-authorized"
    ]
    assert result.evidence_items[0].provenance.model_dump() == {
        "retrieval_request_id": "request-unsafe-canary",
        "dense_rank": 1,
        "dense_score": pytest.approx(0.9701425),
        "lexical_rank": 1,
        "lexical_score": pytest.approx(0.1),
        "fused_rank": 1,
        "fused_score": pytest.approx(2 / 61),
        "rerank_rank": None,
        "rerank_score": None,
    }
    assert result.source_outcomes[0].corpus_revision == result.corpus_revisions[0]
    assert "UNAUTHORIZED_CANARY" not in result.model_dump_json()


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
    )

    assert result.status is RetrievalStatus.NO_EVIDENCE
    assert result.candidate_counts.model_dump() == {
        "dense": 0,
        "lexical": 0,
        "deduplicated": 0,
        "reranked": None,
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
    )
    second = retrieve(
        retrieval_pool,
        snapshot,
        request,
        _config(),
        FixedQueryEmbedder(),
    )

    assert [item.chunk_id for item in first.evidence_items] == [
        "chunk-both",
        "chunk-dense",
        "chunk-lexical",
    ]
    assert first.candidate_counts.deduplicated == 3
    assert [
        item.provenance.model_dump() for item in first.evidence_items
    ] == [item.provenance.model_dump() for item in second.evidence_items]
    assert first.evidence_items[0].provenance.dense_rank == 2
    assert first.evidence_items[0].provenance.lexical_rank == 1
    assert first.evidence_items[1].provenance.lexical_rank is None
    assert first.evidence_items[2].provenance.dense_rank is None
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
    )
    changed_budget = retrieve(
        retrieval_pool,
        snapshot,
        request.model_copy(update={"dense_candidate_limit": 1}),
        _config(),
        FixedQueryEmbedder(),
    )
    assert len(
        {
            first.retrieval_config_fingerprint,
            changed_config.retrieval_config_fingerprint,
            changed_budget.retrieval_config_fingerprint,
        }
    ) == 3


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
