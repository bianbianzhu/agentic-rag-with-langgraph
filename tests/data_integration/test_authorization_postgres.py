"""L2 verification for PostgreSQL-backed Access Scope."""

import json
import os
from collections.abc import Iterator
from hashlib import sha256
from pathlib import Path

import pytest
from psycopg_pool import ConnectionPool

from agentic_rag.authorization import (
    AccessGrant,
    Principal,
    UnknownPrincipalError,
    authorization_snapshot_is_current,
    capture_authorization_snapshot,
    source_document_is_authorized,
)
from agentic_rag.corpus import (
    KnowledgeSource,
    ProcessingConfig,
    sync_knowledge_source,
)
from agentic_rag.corpus.identity import source_document_id
from agentic_rag.database import apply_migrations, open_database_pool


REPOSITORY_ROOT = Path(__file__).parents[2]
DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://agentic_rag@127.0.0.1:55432/agentic_rag",
)
TRUSTED_FIXTURES_ROOT = REPOSITORY_ROOT / "fixtures/reference-system/trusted"
CORPUS_ROOT = REPOSITORY_ROOT / "fixtures/reference-system/corpus/snapshots"
PROCESSING_CONFIG = ProcessingConfig(
    parser_version="markdown-v1",
    chunker_version="heading-v1",
    embedding_model="deterministic-test-v1",
)


class DeterministicEmbedder:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            [byte / 255 for byte in sha256(text.encode()).digest()[:4]]
            for text in texts
        ]


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
    _sync_baseline(database_pool)
    _provision_access_scope(database_pool, "s1-baseline")
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
    _sync_baseline(database_pool)
    _provision_access_scope(database_pool, "auth-change-before")
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


def _sync_baseline(pool: ConnectionPool) -> None:
    grant_fixture = _grant_fixture()
    for knowledge_source in KnowledgeSource:
        assignments = grant_fixture["sets"]["s1-baseline"][
            knowledge_source.value
        ]
        policies = grant_fixture["policies"]
        access_scope_revisions = {
            source_path: sha256(
                json.dumps(
                    policies[policy_name],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            for source_path, policy_name in assignments.items()
        }
        sync_knowledge_source(
            pool,
            knowledge_source,
            CORPUS_ROOT / "s1-baseline" / knowledge_source.value,
            access_scope_revisions,
            PROCESSING_CONFIG,
            DeterministicEmbedder(),
        )


def _provision_access_scope(pool: ConnectionPool, grant_set: str) -> None:
    principals_fixture = json.loads(
        (TRUSTED_FIXTURES_ROOT / "principals.json").read_text(encoding="utf-8")
    )
    principals = [
        Principal.model_validate(item)
        for item in principals_fixture["principals"]
    ]
    grant_fixture = _grant_fixture()
    policies = grant_fixture["policies"]
    with pool.connection() as connection:
        for principal in principals:
            connection.execute(
                "INSERT INTO principals (principal_id) VALUES (%s)",
                (principal.principal_id,),
            )
        for group_id in sorted(
            {group_id for principal in principals for group_id in principal.group_ids}
        ):
            connection.execute(
                "INSERT INTO groups (group_id) VALUES (%s)", (group_id,)
            )
        for principal in principals:
            for group_id in principal.group_ids:
                connection.execute(
                    """
                    INSERT INTO principal_group_memberships (principal_id, group_id)
                    VALUES (%s, %s)
                    """,
                    (principal.principal_id, group_id),
                )
        for knowledge_source, assignments in grant_fixture["sets"][
            grant_set
        ].items():
            for source_path, policy_name in assignments.items():
                source_document_row = connection.execute(
                    """
                    SELECT document_id FROM source_documents
                    WHERE knowledge_source = %s AND source_path = %s
                    """,
                    (knowledge_source, source_path),
                ).fetchone()
                assert source_document_row is not None
                for relation in policies[policy_name]:
                    access_grant = AccessGrant(
                        knowledge_source=knowledge_source,
                        document_id=str(source_document_row[0]),
                        **relation,
                    )
                    connection.execute(
                        """
                        INSERT INTO access_grants (
                            knowledge_source, document_id, grant_type,
                            principal_id, group_id
                        ) VALUES (%s, %s, %s, %s, %s)
                        """,
                        (
                            access_grant.knowledge_source,
                            access_grant.document_id,
                            access_grant.grant_type.value,
                            access_grant.principal_id,
                            access_grant.group_id,
                        ),
                    )


def _grant_fixture():
    return json.loads(
        (TRUSTED_FIXTURES_ROOT / "access-grant-sets.json").read_text(
            encoding="utf-8"
        )
    )
