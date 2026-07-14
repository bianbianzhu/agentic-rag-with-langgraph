"""L2 verification for transactional Corpus Sync."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterator
from hashlib import sha256
from pathlib import Path

import pytest
from psycopg_pool import ConnectionPool

from agentic_rag.corpus import (
    KnowledgeSource,
    ProcessingConfig,
    SyncStatus,
    sync_knowledge_source,
)
from agentic_rag.corpus.identity import source_document_id
from agentic_rag.database import apply_migrations, open_database_pool


REPOSITORY_ROOT = Path(__file__).parents[2]
CORPUS_ROOT = REPOSITORY_ROOT / "fixtures/reference-system/corpus/snapshots"
OVERLAY_ROOT = REPOSITORY_ROOT / "fixtures/reference-system/corpus/overlays"
DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://agentic_rag@127.0.0.1:55432/agentic_rag",
)
CONFIG = ProcessingConfig(
    parser_version="markdown-v1",
    chunker_version="heading-v1",
    embedding_model="deterministic-test-v1",
)
ACCESS_GRANT_SETS_PATH = (
    REPOSITORY_ROOT / "fixtures/reference-system/trusted/access-grant-sets.json"
)


class DeterministicEmbedder:
    """Stable local vectors; no live model belongs in L2."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            [byte / 255 for byte in sha256(text.encode()).digest()[:4]]
            for text in texts
        ]


class CountingEmbedder(DeterministicEmbedder):
    def __init__(self) -> None:
        self.calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return super().embed_documents(texts)


class FailingEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        raise ConnectionError("temporary test outage")


@pytest.fixture(scope="module")
def database_pool() -> Iterator[ConnectionPool]:
    pool = open_database_pool(DATABASE_URL)
    apply_migrations(pool, REPOSITORY_ROOT / "migrations")
    yield pool
    pool.close()


@pytest.fixture(autouse=True)
def empty_corpus(database_pool: ConnectionPool) -> None:
    with database_pool.connection() as connection:
        connection.execute(
            """
            TRUNCATE sync_reports, indexed_chunks, source_documents,
                knowledge_sources CASCADE
            """
        )


def _sync_snapshot(
    pool: ConnectionPool,
    snapshot: str,
    source: KnowledgeSource,
):
    return sync_knowledge_source(
        pool=pool,
        knowledge_source=source,
        source_root=CORPUS_ROOT / snapshot / source.value,
        access_scope_revisions=_load_access_scope_revisions(snapshot, source),
        config=CONFIG,
        embedder=DeterministicEmbedder(),
    )


def _load_access_scope_revisions(
    grant_set: str, source: KnowledgeSource
) -> dict[str, str]:
    fixture = json.loads(ACCESS_GRANT_SETS_PATH.read_text(encoding="utf-8"))
    policies = fixture["policies"]
    assignments = fixture["sets"][grant_set][source.value]
    return {
        source_path: sha256(
            json.dumps(
                policies[policy_name], sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        for source_path, policy_name in assignments.items()
    }


def test_invalid_utf8_fails_atomically_and_keeps_a_durable_report(
    database_pool: ConnectionPool, tmp_path: Path
) -> None:
    _sync_snapshot(
        database_pool, "s1-baseline", KnowledgeSource.ENGINEERING_DOCS
    )
    with database_pool.connection() as connection:
        before_revision = connection.execute(
            """
            SELECT corpus_revision FROM knowledge_sources
            WHERE knowledge_source = %s
            """,
            (KnowledgeSource.ENGINEERING_DOCS.value,),
        ).fetchone()
        before_source_documents = connection.execute(
            "SELECT document_id, source_revision FROM source_documents ORDER BY 1"
        ).fetchall()
        before_indexed_chunks = connection.execute(
            "SELECT chunk_id, content FROM indexed_chunks ORDER BY 1"
        ).fetchall()

    invalid_root = tmp_path / KnowledgeSource.ENGINEERING_DOCS.value
    shutil.copytree(
        CORPUS_ROOT / "s1-baseline" / KnowledgeSource.ENGINEERING_DOCS.value,
        invalid_root,
    )
    corrupt_path = invalid_root / "payments/corrupt-procedure.md"
    shutil.copyfile(
        OVERLAY_ROOT
        / "invalid/engineering-docs/payments/corrupt-procedure.md",
        corrupt_path,
    )
    access = {
        **_load_access_scope_revisions(
            "s1-baseline", KnowledgeSource.ENGINEERING_DOCS
        ),
        "payments/corrupt-procedure.md": "payments-engineering-v1",
    }

    report = sync_knowledge_source(
        pool=database_pool,
        knowledge_source=KnowledgeSource.ENGINEERING_DOCS,
        source_root=invalid_root,
        access_scope_revisions=access,
        config=CONFIG,
        embedder=DeterministicEmbedder(),
    )

    assert report.status is SyncStatus.FAILED
    assert report.failed_source_path == "payments/corrupt-procedure.md"
    assert report.error_summary == "Source Document is not valid UTF-8"
    assert report.corpus_revision is None
    assert "content" not in report.model_dump()
    assert "embedding" not in report.model_dump()
    with database_pool.connection() as connection:
        assert connection.execute(
            """
            SELECT corpus_revision FROM knowledge_sources
            WHERE knowledge_source = %s
            """,
            (KnowledgeSource.ENGINEERING_DOCS.value,),
        ).fetchone() == before_revision
        assert connection.execute(
            "SELECT document_id, source_revision FROM source_documents ORDER BY 1"
        ).fetchall() == before_source_documents
        assert connection.execute(
            "SELECT chunk_id, content FROM indexed_chunks ORDER BY 1"
        ).fetchall() == before_indexed_chunks
        stored_report = connection.execute(
            """
            SELECT status, failed_source_path, corpus_revision
            FROM sync_reports WHERE run_id = %s
            """,
            (report.run_id,),
        ).fetchone()
    assert stored_report == (
        "failed",
        "payments/corrupt-procedure.md",
        None,
    )


def test_s0_then_s1_publish_exact_incremental_changes(
    database_pool: ConnectionPool,
) -> None:
    s0_reports = [
        _sync_snapshot(database_pool, "s0-initial", source)
        for source in KnowledgeSource
    ]

    assert all(report.status is SyncStatus.SUCCEEDED for report in s0_reports)
    assert sum(report.added_count for report in s0_reports) == 6
    assert sum(report.updated_count for report in s0_reports) == 0
    assert sum(report.deleted_count for report in s0_reports) == 0
    assert sum(report.unchanged_count for report in s0_reports) == 0
    assert sum(report.ignored_count for report in s0_reports) == 0
    with database_pool.connection() as connection:
        source_document_count = connection.execute(
            "SELECT count(*) FROM source_documents"
        ).fetchone()
        indexed_chunk_count = connection.execute(
            "SELECT count(*) FROM indexed_chunks"
        ).fetchone()
        d4_before = connection.execute(
            """
            SELECT document_id, source_revision FROM source_documents
            WHERE knowledge_source = %s AND source_path = %s
            """,
            (
                KnowledgeSource.ENGINEERING_DOCS.value,
                "announcements/rollback-incident.md",
            ),
        ).fetchone()
    assert source_document_count == (6,)
    assert indexed_chunk_count is not None and indexed_chunk_count[0] >= 6

    s1_reports = [
        _sync_snapshot(database_pool, "s1-baseline", source)
        for source in KnowledgeSource
    ]

    assert all(report.status is SyncStatus.SUCCEEDED for report in s1_reports)
    assert sum(report.added_count for report in s1_reports) == 1
    assert sum(report.updated_count for report in s1_reports) == 2
    assert sum(report.deleted_count for report in s1_reports) == 1
    assert sum(report.unchanged_count for report in s1_reports) == 3
    assert sum(report.ignored_count for report in s1_reports) == 1
    assert all(report.corpus_revision for report in s1_reports)

    precursor_id = source_document_id(
        KnowledgeSource.ENGINEERING_DOCS,
        "payments/legacy-rollback-worker.md",
    )
    replacement_id = source_document_id(
        KnowledgeSource.ENGINEERING_DOCS,
        "payments/rollback-worker-compatibility.md",
    )
    with database_pool.connection() as connection:
        source_documents = connection.execute(
            "SELECT document_id FROM source_documents"
        ).fetchall()
        d4_after = connection.execute(
            """
            SELECT document_id, source_revision FROM source_documents
            WHERE knowledge_source = %s AND source_path = %s
            """,
            (
                KnowledgeSource.ENGINEERING_DOCS.value,
                "announcements/rollback-incident.md",
            ),
        ).fetchone()
        stored_successes = connection.execute(
            "SELECT count(*) FROM sync_reports WHERE status = 'succeeded'"
        ).fetchone()
    source_document_ids = {str(row[0]) for row in source_documents}
    assert precursor_id not in source_document_ids
    assert replacement_id in source_document_ids
    assert d4_before is not None and d4_after is not None
    assert d4_after[0] == d4_before[0]
    assert d4_after[1] != d4_before[1]
    assert stored_successes == (4,)


def test_unchanged_source_documents_skip_embedding_and_processing_change_rebuilds(
    database_pool: ConnectionPool,
) -> None:
    source = KnowledgeSource.ENGINEERING_DOCS
    root = CORPUS_ROOT / "s0-initial" / source.value
    embedder = CountingEmbedder()
    first = sync_knowledge_source(
        database_pool,
        source,
        root,
        _load_access_scope_revisions("s0-initial", source),
        CONFIG,
        embedder,
    )
    assert first.added_count == 4
    assert embedder.calls == 4

    embedder.calls = 0
    unchanged = sync_knowledge_source(
        database_pool,
        source,
        root,
        _load_access_scope_revisions("s0-initial", source),
        CONFIG,
        embedder,
    )
    assert unchanged.unchanged_count == 4
    assert embedder.calls == 0

    changed_config = CONFIG.model_copy(
        update={"chunker_version": "heading-v2"}
    )
    rebuilt = sync_knowledge_source(
        database_pool,
        source,
        root,
        _load_access_scope_revisions("s0-initial", source),
        changed_config,
        embedder,
    )
    assert rebuilt.updated_count == 4
    assert embedder.calls == 4


def test_transient_embedding_failure_is_bounded_and_publishes_nothing(
    database_pool: ConnectionPool, tmp_path: Path
) -> None:
    root = tmp_path / KnowledgeSource.ENGINEERING_DOCS.value
    source_document_path = root / "payments/new.md"
    source_document_path.parent.mkdir(parents=True)
    source_document_path.write_text(
        "# New\n\n## Detail\n\nSafe content.\n", encoding="utf-8"
    )
    embedder = FailingEmbedder()

    report = sync_knowledge_source(
        database_pool,
        KnowledgeSource.ENGINEERING_DOCS,
        root,
        {"payments/new.md": "payments-engineering-v1"},
        CONFIG,
        embedder,
    )

    assert report.status is SyncStatus.FAILED
    assert report.error_summary == "embedding failed after 3 attempts"
    assert embedder.calls == 3
    with database_pool.connection() as connection:
        published = connection.execute(
            "SELECT count(*) FROM source_documents"
        ).fetchone()
    assert published == (0,)


def test_escaping_symlink_fails_the_complete_snapshot(
    database_pool: ConnectionPool, tmp_path: Path
) -> None:
    root = tmp_path / KnowledgeSource.ENGINEERING_DOCS.value
    root.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("# Outside\n\n## Data\n\nDo not read.\n", encoding="utf-8")
    (root / "escape.md").symlink_to(outside)

    report = sync_knowledge_source(
        database_pool,
        KnowledgeSource.ENGINEERING_DOCS,
        root,
        {"escape.md": "organization-public-v1"},
        CONFIG,
        DeterministicEmbedder(),
    )

    assert report.status is SyncStatus.FAILED
    assert report.failed_source_path == "escape.md"
    assert report.error_summary == "source path escapes the configured root"


def test_partial_directory_cannot_infer_source_document_deletions(
    database_pool: ConnectionPool,
) -> None:
    _sync_snapshot(
        database_pool, "s0-initial", KnowledgeSource.ENGINEERING_DOCS
    )

    report = sync_knowledge_source(
        database_pool,
        KnowledgeSource.ENGINEERING_DOCS,
        CORPUS_ROOT / "s0-initial/engineering-docs/payments",
        {},
        CONFIG,
        DeterministicEmbedder(),
    )

    assert report.status is SyncStatus.FAILED
    assert report.error_summary == "source root does not match the Knowledge Source"
    with database_pool.connection() as connection:
        source_document_count = connection.execute(
            """
            SELECT count(*) FROM source_documents
            WHERE knowledge_source = %s
            """,
            (KnowledgeSource.ENGINEERING_DOCS.value,),
        ).fetchone()
    assert source_document_count == (4,)


def test_publication_database_error_rolls_back_and_records_failure(
    database_pool: ConnectionPool,
) -> None:
    with database_pool.connection() as connection:
        connection.execute(
            """
            CREATE OR REPLACE FUNCTION reject_test_publication()
            RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN
                RAISE EXCEPTION 'forced publication failure';
            END;
            $$
            """
        )
        connection.execute(
            """
            CREATE TRIGGER reject_test_publication
            BEFORE INSERT ON source_documents
            FOR EACH ROW EXECUTE FUNCTION reject_test_publication()
            """
        )
    try:
        report = _sync_snapshot(
            database_pool, "s0-initial", KnowledgeSource.ENGINEERING_DOCS
        )
    finally:
        with database_pool.connection() as connection:
            connection.execute(
                "DROP TRIGGER reject_test_publication ON source_documents"
            )
            connection.execute("DROP FUNCTION reject_test_publication()")

    assert report.status is SyncStatus.FAILED
    assert report.error_summary == "corpus database operation failed"
    with database_pool.connection() as connection:
        source_document_count = connection.execute(
            "SELECT count(*) FROM source_documents"
        ).fetchone()
        stored_report = connection.execute(
            "SELECT status FROM sync_reports WHERE run_id = %s",
            (report.run_id,),
        ).fetchone()
    assert source_document_count == (0,)
    assert stored_report == ("failed",)
