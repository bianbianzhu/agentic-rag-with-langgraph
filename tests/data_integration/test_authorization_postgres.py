"""L2 verification for PostgreSQL-backed Access Scope."""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from psycopg_pool import ConnectionPool

from agentic_rag.authorization import (
    UnknownPrincipalError,
    authorization_snapshot_is_current,
    capture_authorization_snapshot,
    source_document_is_authorized,
)
from agentic_rag.corpus import KnowledgeSource
from agentic_rag.corpus.identity import source_document_id
from agentic_rag.database import apply_migrations, open_database_pool
from tests.support.reference_fixture import (
    load_reference_snapshot,
    provision_reference_access_scope,
    sync_reference_snapshot,
)


REPOSITORY_ROOT = Path(__file__).parents[2]
DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://agentic_rag@127.0.0.1:55432/agentic_rag",
)


@pytest.fixture(scope="module")
def database_pool() -> Iterator[ConnectionPool]:
    pool = open_database_pool(DATABASE_URL)
    apply_migrations(pool, REPOSITORY_ROOT / "migrations")
    yield pool
    pool.close()


@pytest.fixture(autouse=True)
def empty_authorization_state(database_pool: ConnectionPool) -> None:
    with database_pool.connection() as connection:
        connection.execute(
            """
            TRUNCATE access_grants, principal_group_memberships, groups,
                principals, sync_reports, indexed_chunks, source_documents,
                knowledge_sources CASCADE
            """
        )


def test_injected_principal_cannot_capture_authorization_snapshot(
    database_pool: ConnectionPool,
) -> None:
    with database_pool.connection() as connection:
        connection.execute(
            "INSERT INTO principals (principal_id) VALUES ('alice')"
        )
    with pytest.raises(UnknownPrincipalError, match="unknown Principal"):
        capture_authorization_snapshot(database_pool, "alice' OR true --")


def test_missing_access_grant_denies_source_document_by_default(
    database_pool: ConnectionPool,
) -> None:
    with database_pool.connection() as connection:
        connection.execute(
            "INSERT INTO principals (principal_id) VALUES ('alice')"
        )
        connection.execute(
            "INSERT INTO knowledge_sources (knowledge_source) VALUES ('engineering-docs')"
        )
        connection.execute(
            """
            INSERT INTO source_documents (
                knowledge_source, document_id, source_path, source_revision,
                processing_revision, title, raw_content
            ) VALUES (
                'engineering-docs', 'doc_denied', 'payments/denied.md',
                'src_test', 'proc_test', 'Denied', 'untrusted content'
            )
            """
        )
    snapshot = capture_authorization_snapshot(database_pool, "alice")

    allowed = source_document_is_authorized(
        database_pool,
        snapshot,
        knowledge_source="engineering-docs",
        document_id="doc_denied",
    )

    assert allowed is False


def test_fixed_principal_matrix_applies_public_group_and_direct_grants(
    database_pool: ConnectionPool,
) -> None:
    load_reference_snapshot(database_pool)
    source_documents = {
        "D1": (
            KnowledgeSource.ENGINEERING_DOCS,
            "payments/platform-overview.md",
        ),
        "D2": (
            KnowledgeSource.ENGINEERING_DOCS,
            "payments/schema-migration.md",
        ),
        "D3": (
            KnowledgeSource.ENGINEERING_DOCS,
            "payments/rollback-worker-compatibility.md",
        ),
        "D4": (
            KnowledgeSource.ENGINEERING_DOCS,
            "announcements/rollback-incident.md",
        ),
        "D7": (
            KnowledgeSource.OPERATIONAL_RUNBOOKS,
            "payments/rollback-recovery.md",
        ),
        "D8": (
            KnowledgeSource.OPERATIONAL_RUNBOOKS,
            "finance/settlement-impact.md",
        ),
    }
    expected_access_scope = {
        "alice": {"D1", "D2", "D3", "D4", "D7", "D8"},
        "bob": {"D1", "D2", "D3", "D4"},
        "carol": {"D4", "D8"},
    }

    actual_access_scope: dict[str, set[str]] = {}
    for principal_id in expected_access_scope:
        snapshot = capture_authorization_snapshot(database_pool, principal_id)
        actual_access_scope[principal_id] = {
            alias
            for alias, (knowledge_source, source_path) in source_documents.items()
            if source_document_is_authorized(
                database_pool,
                snapshot,
                knowledge_source=knowledge_source.value,
                document_id=source_document_id(knowledge_source, source_path),
            )
        }

    assert actual_access_scope == expected_access_scope


def test_authorization_snapshot_detects_effective_access_scope_change(
    database_pool: ConnectionPool,
) -> None:
    sync_reference_snapshot(database_pool, "s1-baseline")
    provision_reference_access_scope(database_pool, "auth-change-before")
    snapshot = capture_authorization_snapshot(database_pool, "alice")
    before_change = authorization_snapshot_is_current(database_pool, snapshot)

    with database_pool.connection() as connection:
        connection.execute(
            """
            DELETE FROM access_grants
            WHERE grant_type = 'principal' AND principal_id = 'alice'
            """
        )

    after_change = authorization_snapshot_is_current(database_pool, snapshot)
    replacement = capture_authorization_snapshot(database_pool, "alice")

    assert (before_change, after_change, replacement.revision != snapshot.revision) == (
        True,
        False,
        True,
    )
