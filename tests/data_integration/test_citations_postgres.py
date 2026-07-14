"""L2 authorization checks for citation hydration."""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from psycopg_pool import ConnectionPool

from agentic_rag.authorization import capture_authorization_snapshot
from agentic_rag.citations import (
    CitationHydrationError,
    hydrate_citation_mappings,
)
from agentic_rag.corpus import KnowledgeSource
from agentic_rag.corpus.models import SourceLocator
from agentic_rag.database import apply_migrations, open_database_pool
from agentic_rag.retrieval import EvidenceItem, RetrievalProvenance
from tests.support.reference_fixture import load_reference_snapshot


REPOSITORY_ROOT = Path(__file__).parents[2]
DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://agentic_rag@127.0.0.1:55432/agentic_rag",
)
RETRIEVAL_DATABASE_URL = os.environ.get(
    "TEST_RETRIEVAL_DATABASE_URL",
    "postgresql://agentic_rag_retrieval@127.0.0.1:55432/agentic_rag",
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
def empty_citation_state(database_pool: ConnectionPool) -> None:
    with database_pool.connection() as connection:
        connection.execute(
            """
            TRUNCATE access_grants, principal_group_memberships, groups,
                principals, sync_reports, indexed_chunks, source_documents,
                knowledge_sources CASCADE
            """
        )


def test_revoked_evidence_cannot_be_hydrated_into_a_citation(
    database_pool: ConnectionPool,
    retrieval_pool: ConnectionPool,
) -> None:
    load_reference_snapshot(database_pool)
    snapshot = capture_authorization_snapshot(retrieval_pool, "alice")
    evidence = _evidence_for_path(
        database_pool,
        KnowledgeSource.OPERATIONAL_RUNBOOKS,
        "finance/settlement-impact.md",
        "Confidential impact",
    )
    with database_pool.connection() as connection:
        connection.execute(
            """
            DELETE FROM access_grants
            WHERE grant_type = 'principal' AND principal_id = 'alice'
              AND document_id = %s
            """,
            (evidence.document_id,),
        )

    with pytest.raises(
        CitationHydrationError, match="citation Evidence is unavailable"
    ) as caught:
        hydrate_citation_mappings(retrieval_pool, snapshot, (evidence,))

    assert "AUD 2.4 million" not in str(caught.value)


def test_authorized_evidence_hydrates_authoritative_citation_metadata(
    database_pool: ConnectionPool,
    retrieval_pool: ConnectionPool,
) -> None:
    load_reference_snapshot(database_pool)
    snapshot = capture_authorization_snapshot(retrieval_pool, "bob")
    evidence = _evidence_for_path(
        database_pool,
        KnowledgeSource.ENGINEERING_DOCS,
        "payments/schema-migration.md",
        "Rollback incompatibility",
    )

    mappings = hydrate_citation_mappings(
        retrieval_pool, snapshot, (evidence,)
    )

    assert mappings[0].key == "E1"
    assert mappings[0].chunk_id == evidence.chunk_id
    assert mappings[0].source_path == "payments/schema-migration.md"
    assert mappings[0].source_locator.section_path == (
        "Rollback incompatibility",
    )


def _evidence_for_path(
    pool: ConnectionPool,
    knowledge_source: KnowledgeSource,
    source_path: str,
    heading: str,
) -> EvidenceItem:
    with pool.connection() as connection:
        row = connection.execute(
            """
            SELECT indexed_chunk.chunk_id, indexed_chunk.document_id,
                   indexed_chunk.source_revision,
                   indexed_chunk.processing_revision,
                   knowledge_source.corpus_revision,
                   indexed_chunk.content, source_document.title
            FROM indexed_chunks AS indexed_chunk
            JOIN source_documents AS source_document
              ON source_document.knowledge_source = indexed_chunk.knowledge_source
             AND source_document.document_id = indexed_chunk.document_id
            JOIN knowledge_sources AS knowledge_source
              ON knowledge_source.knowledge_source = indexed_chunk.knowledge_source
            WHERE indexed_chunk.knowledge_source = %s
              AND source_document.source_path = %s
              AND indexed_chunk.heading_path = %s
            """,
            (knowledge_source.value, source_path, [heading]),
        ).fetchone()
    assert row is not None
    return EvidenceItem(
        chunk_id=str(row[0]),
        document_id=str(row[1]),
        knowledge_source=knowledge_source,
        source_revision=str(row[2]),
        processing_revision=str(row[3]),
        corpus_revision=str(row[4]),
        chunk_text=str(row[5]),
        source_path=source_path,
        title=str(row[6]),
        source_locator=SourceLocator(section_path=(heading,)),
        provenance=(
            RetrievalProvenance(
                retrieval_request_id="request-citation",
                fused_rank=1,
                fused_score=0.03,
                rerank_rank=1,
                rerank_score=1,
            ),
        ),
    )
