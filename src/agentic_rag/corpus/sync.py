"""Synchronous, transactional publication of one complete corpus snapshot."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Protocol, Sequence
from uuid import UUID, uuid4

from psycopg import Connection, Error as PsycopgError
from psycopg_pool import ConnectionPool

from agentic_rag.corpus.identity import (
    indexed_chunk_id,
    processing_revision,
    source_document_id,
    source_revision,
)
from agentic_rag.corpus.models import (
    KnowledgeSource,
    ProcessingConfig,
    SyncReport,
    SyncStatus,
)


class IndexedChunkEmbedder(Protocol):
    """The narrow embedding seam required by Corpus Sync."""

    # Keep the LangChain Embeddings method name for live/test substitutability.
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class CorpusSyncError(Exception):
    """A deterministic error that safely fails the complete sync attempt."""

    def __init__(self, summary: str, source_path: str | None = None) -> None:
        super().__init__(summary)
        self.summary = summary
        self.source_path = source_path


@dataclass(frozen=True)
class _SourceDocumentCandidate:
    document_id: str
    source_path: str
    raw_source: bytes
    source_revision: str


@dataclass(frozen=True)
class _IndexedChunk:
    chunk_id: str
    ordinal: int
    heading_path: tuple[str, ...]
    content: str
    embedding: tuple[float, ...]


@dataclass(frozen=True)
class _ProcessedSourceDocument:
    candidate: _SourceDocumentCandidate
    title: str
    raw_content: str
    indexed_chunks: tuple[_IndexedChunk, ...]


@dataclass(frozen=True)
class _Counts:
    added: int = 0
    updated: int = 0
    deleted: int = 0
    unchanged: int = 0


def sync_knowledge_source(
    pool: ConnectionPool,
    knowledge_source: KnowledgeSource,
    source_root: Path,
    access_scope_revisions: Mapping[str, str],
    config: ProcessingConfig,
    embedder: IndexedChunkEmbedder,
) -> SyncReport:
    """Validate a complete snapshot and publish it in one transaction."""

    run_id = uuid4()
    started_at = datetime.now(UTC)
    processing_revision_id = processing_revision(config)
    observed_snapshot_size = 0
    ignored_count = 0
    counts = _Counts()

    try:
        candidates, ignored_count = _discover_snapshot(
            knowledge_source, source_root, access_scope_revisions
        )
        observed_snapshot_size = len(candidates)
        existing = _load_existing(pool, knowledge_source)
        counts = _count_changes(candidates, existing, processing_revision_id)
        processed = _process_changed(
            candidates, existing, processing_revision_id, embedder
        )
        report = _publish(
            pool=pool,
            knowledge_source=knowledge_source,
            candidates=candidates,
            expected_existing=existing,
            processed=processed,
            processing_revision_id=processing_revision_id,
            run_id=run_id,
            started_at=started_at,
            ignored_count=ignored_count,
            counts=counts,
        )
    except PsycopgError:
        error = CorpusSyncError("corpus database operation failed")
        report = _failed_report(
            run_id,
            knowledge_source,
            started_at,
            processing_revision_id,
            observed_snapshot_size,
            ignored_count,
            counts,
            error,
        )
        _insert_report(pool, report)
        return report
    except CorpusSyncError as error:
        report = _failed_report(
            run_id,
            knowledge_source,
            started_at,
            processing_revision_id,
            observed_snapshot_size,
            ignored_count,
            counts,
            error,
        )
        _insert_report(pool, report)
        return report

    return report


def _failed_report(
    run_id: UUID,
    knowledge_source: KnowledgeSource,
    started_at: datetime,
    processing_revision_id: str,
    observed_snapshot_size: int,
    ignored_count: int,
    counts: _Counts,
    error: CorpusSyncError,
) -> SyncReport:
    return SyncReport(
        run_id=run_id,
        knowledge_source=knowledge_source,
        status=SyncStatus.FAILED,
        started_at=started_at,
        finished_at=datetime.now(UTC),
        processing_revision=processing_revision_id,
        observed_snapshot_size=observed_snapshot_size,
        added_count=counts.added,
        updated_count=counts.updated,
        deleted_count=counts.deleted,
        unchanged_count=counts.unchanged,
        ignored_count=ignored_count,
        failed_source_path=error.source_path,
        error_summary=error.summary,
    )


def _discover_snapshot(
    knowledge_source: KnowledgeSource,
    source_root: Path,
    access_scope_revisions: Mapping[str, str],
) -> tuple[list[_SourceDocumentCandidate], int]:
    if source_root.name != knowledge_source.value:
        raise CorpusSyncError(
            "source root does not match the Knowledge Source"
        )
    if not source_root.is_dir():
        raise CorpusSyncError("source root is not a directory")

    resolved_root = source_root.resolve()
    candidates: list[_SourceDocumentCandidate] = []
    ignored_count = 0
    try:
        discovered_paths = sorted(source_root.rglob("*"))
    except OSError as error:
        raise CorpusSyncError("source snapshot could not be discovered") from error
    for path in discovered_paths:
        relative_path = path.relative_to(source_root).as_posix()
        if path.is_symlink() and not path.resolve().is_relative_to(resolved_root):
            raise CorpusSyncError(
                "source path escapes the configured root", relative_path
            )
        if not path.is_file():
            continue
        resolved_path = path.resolve()
        if not resolved_path.is_relative_to(resolved_root):
            raise CorpusSyncError(
                "source path escapes the configured root", relative_path
            )
        if resolved_path.suffix.lower() != ".md":
            ignored_count += 1
            continue
        access_revision = access_scope_revisions.get(relative_path)
        if not access_revision:
            raise CorpusSyncError(
                "trusted access scope revision is missing", relative_path
            )
        try:
            raw_source = resolved_path.read_bytes()
        except OSError as error:
            raise CorpusSyncError(
                "Source Document could not be read", relative_path
            ) from error
        candidates.append(
            _SourceDocumentCandidate(
                document_id=source_document_id(knowledge_source, relative_path),
                source_path=relative_path,
                raw_source=raw_source,
                source_revision=source_revision(raw_source, access_revision),
            )
        )
    return candidates, ignored_count


def _load_existing(
    pool: ConnectionPool, knowledge_source: KnowledgeSource
) -> dict[str, tuple[str, str]]:
    with pool.connection() as connection:
        return _load_existing_connection(connection, knowledge_source)


def _load_existing_connection(
    connection: Connection, knowledge_source: KnowledgeSource
) -> dict[str, tuple[str, str]]:
    rows = connection.execute(
        """
        SELECT document_id, source_revision, processing_revision
        FROM source_documents WHERE knowledge_source = %s
        """,
        (knowledge_source.value,),
    ).fetchall()
    return {str(row[0]): (str(row[1]), str(row[2])) for row in rows}


def _count_changes(
    candidates: Sequence[_SourceDocumentCandidate],
    existing: dict[str, tuple[str, str]],
    processing_revision_id: str,
) -> _Counts:
    proposed_ids = {candidate.document_id for candidate in candidates}
    added = updated = unchanged = 0
    for candidate in candidates:
        current = existing.get(candidate.document_id)
        proposed = (candidate.source_revision, processing_revision_id)
        if current is None:
            added += 1
        elif current == proposed:
            unchanged += 1
        else:
            updated += 1
    return _Counts(
        added=added,
        updated=updated,
        deleted=len(existing.keys() - proposed_ids),
        unchanged=unchanged,
    )


def _process_changed(
    candidates: Sequence[_SourceDocumentCandidate],
    existing: dict[str, tuple[str, str]],
    processing_revision_id: str,
    embedder: IndexedChunkEmbedder,
) -> dict[str, _ProcessedSourceDocument]:
    processed: dict[str, _ProcessedSourceDocument] = {}
    for candidate in candidates:
        if existing.get(candidate.document_id) == (
            candidate.source_revision,
            processing_revision_id,
        ):
            continue
        try:
            raw_content = candidate.raw_source.decode("utf-8")
        except UnicodeDecodeError as error:
            raise CorpusSyncError(
                "Source Document is not valid UTF-8", candidate.source_path
            ) from error
        title, sections = _parse_markdown(raw_content, candidate.source_path)
        embeddings = _embed_with_retries(
            embedder, [content for _, content in sections], candidate.source_path
        )
        if not isinstance(embeddings, list) or len(embeddings) != len(sections):
            raise CorpusSyncError(
                "embedding result count does not match Indexed Chunks",
                candidate.source_path,
            )
        indexed_chunks: list[_IndexedChunk] = []
        for ordinal, ((heading_path, content), embedding) in enumerate(
            zip(sections, embeddings, strict=True)
        ):
            if (
                not isinstance(embedding, list)
                or not embedding
                or not all(
                    isinstance(value, (int, float)) and isfinite(value)
                    for value in embedding
                )
            ):
                raise CorpusSyncError(
                    "embedding contains no usable vector", candidate.source_path
                )
            indexed_chunks.append(
                _IndexedChunk(
                    chunk_id=indexed_chunk_id(
                        candidate.document_id,
                        candidate.source_revision,
                        processing_revision_id,
                        ordinal,
                    ),
                    ordinal=ordinal,
                    heading_path=heading_path,
                    content=content,
                    embedding=tuple(float(value) for value in embedding),
                )
            )
        processed[candidate.document_id] = _ProcessedSourceDocument(
            candidate=candidate,
            title=title,
            raw_content=raw_content,
            indexed_chunks=tuple(indexed_chunks),
        )
    return processed


def _parse_markdown(
    raw_content: str, source_path: str
) -> tuple[str, list[tuple[tuple[str, ...], str]]]:
    title: str | None = None
    heading_stack: list[str] = []
    current_heading: tuple[str, ...] = ()
    current_lines: list[str] = []
    sections: list[tuple[tuple[str, ...], str]] = []

    def finish_section() -> None:
        content = "\n".join(current_lines).strip()
        if content:
            sections.append((current_heading, content))

    for line in raw_content.splitlines():
        stripped = line.lstrip()
        level = len(stripped) - len(stripped.lstrip("#"))
        if level and level <= 6 and stripped[level : level + 1] == " ":
            heading = stripped[level + 1 :].strip()
            if not heading:
                continue
            if level == 1 and title is None:
                title = heading
                continue
            finish_section()
            current_lines = []
            heading_stack[level - 1 :] = []
            while len(heading_stack) < level - 1:
                heading_stack.append("")
            heading_stack.append(heading)
            current_heading = tuple(part for part in heading_stack if part)
            continue
        current_lines.append(line)
    finish_section()

    if title is None:
        raise CorpusSyncError("Source Document has no H1 title", source_path)
    if not sections:
        raise CorpusSyncError("Source Document has no indexable content", source_path)
    return title, sections


def _embed_with_retries(
    embedder: IndexedChunkEmbedder, texts: list[str], source_path: str
) -> list[list[float]]:
    for attempt in range(3):
        try:
            return embedder.embed_documents(texts)
        except (ConnectionError, TimeoutError) as error:
            if attempt == 2:
                raise CorpusSyncError(
                    "embedding failed after 3 attempts", source_path
                ) from error
        except Exception as error:
            raise CorpusSyncError("embedding failed", source_path) from error
    raise AssertionError("unreachable")


def _publish(
    *,
    pool: ConnectionPool,
    knowledge_source: KnowledgeSource,
    candidates: Sequence[_SourceDocumentCandidate],
    expected_existing: dict[str, tuple[str, str]],
    processed: dict[str, _ProcessedSourceDocument],
    processing_revision_id: str,
    run_id: UUID,
    started_at: datetime,
    ignored_count: int,
    counts: _Counts,
) -> SyncReport:
    proposed_ids = {candidate.document_id for candidate in candidates}
    corpus_revision_id = _corpus_revision(candidates, processing_revision_id)
    with pool.connection() as connection:
        with connection.transaction():
            connection.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (knowledge_source.value,),
            )
            current = _load_existing_connection(connection, knowledge_source)
            if current != expected_existing:
                raise CorpusSyncError("corpus changed while the snapshot was processing")
            connection.execute(
                """
                INSERT INTO knowledge_sources (knowledge_source)
                VALUES (%s)
                ON CONFLICT (knowledge_source) DO NOTHING
                """,
                (knowledge_source.value,),
            )
            deleted_ids = current.keys() - proposed_ids
            if deleted_ids:
                connection.execute(
                    """
                    DELETE FROM source_documents
                    WHERE knowledge_source = %s AND document_id = ANY(%s)
                    """,
                    (knowledge_source.value, list(deleted_ids)),
                )
            for processed_source_document in processed.values():
                candidate = processed_source_document.candidate
                connection.execute(
                    """
                    INSERT INTO source_documents (
                        knowledge_source, document_id, source_path,
                        source_revision, processing_revision, title, raw_content
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (knowledge_source, document_id) DO UPDATE SET
                        source_path = EXCLUDED.source_path,
                        source_revision = EXCLUDED.source_revision,
                        processing_revision = EXCLUDED.processing_revision,
                        title = EXCLUDED.title,
                        raw_content = EXCLUDED.raw_content
                    """,
                    (
                        knowledge_source.value,
                        candidate.document_id,
                        candidate.source_path,
                        candidate.source_revision,
                        processing_revision_id,
                        processed_source_document.title,
                        processed_source_document.raw_content,
                    ),
                )
                connection.execute(
                    """
                    DELETE FROM indexed_chunks
                    WHERE knowledge_source = %s AND document_id = %s
                    """,
                    (knowledge_source.value, candidate.document_id),
                )
                for indexed_chunk in processed_source_document.indexed_chunks:
                    connection.execute(
                        """
                        INSERT INTO indexed_chunks (
                            chunk_id, knowledge_source, document_id,
                            source_revision, processing_revision, ordinal,
                            heading_path, content, embedding
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                        """,
                        (
                            indexed_chunk.chunk_id,
                            knowledge_source.value,
                            candidate.document_id,
                            candidate.source_revision,
                            processing_revision_id,
                            indexed_chunk.ordinal,
                            list(indexed_chunk.heading_path),
                            indexed_chunk.content,
                            _vector_literal(indexed_chunk.embedding),
                        ),
                    )
            connection.execute(
                """
                UPDATE knowledge_sources SET corpus_revision = %s
                WHERE knowledge_source = %s
                """,
                (corpus_revision_id, knowledge_source.value),
            )
            finished_at = datetime.now(UTC)
            report = SyncReport(
                run_id=run_id,
                knowledge_source=knowledge_source,
                status=SyncStatus.SUCCEEDED,
                started_at=started_at,
                finished_at=finished_at,
                processing_revision=processing_revision_id,
                observed_snapshot_size=len(candidates),
                added_count=counts.added,
                updated_count=counts.updated,
                deleted_count=counts.deleted,
                unchanged_count=counts.unchanged,
                ignored_count=ignored_count,
                corpus_revision=corpus_revision_id,
            )
            _insert_report_connection(connection, report)
    return report


def _corpus_revision(
    candidates: Sequence[_SourceDocumentCandidate], processing_revision_id: str
) -> str:
    revision_input = "\n".join(
        f"{candidate.document_id}\0{candidate.source_revision}\0{processing_revision_id}"
        for candidate in sorted(candidates, key=lambda item: item.document_id)
    )
    return f"corpus_{sha256(revision_input.encode()).hexdigest()}"


def _vector_literal(embedding: Sequence[float]) -> str:
    return "[" + ",".join(format(value, ".17g") for value in embedding) + "]"


_REPORT_INSERT_SQL = """
    INSERT INTO sync_reports (
        run_id, knowledge_source, status, started_at, finished_at,
        processing_revision, observed_snapshot_size, added_count,
        updated_count, deleted_count, unchanged_count, ignored_count,
        failed_source_path, error_summary, corpus_revision
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


def _insert_report(pool: ConnectionPool, report: SyncReport) -> None:
    with pool.connection() as connection:
        _insert_report_connection(connection, report)


def _insert_report_connection(
    connection: Connection, report: SyncReport
) -> None:
    connection.execute(
        _REPORT_INSERT_SQL,
        (
            report.run_id,
            report.knowledge_source.value,
            report.status.value,
            report.started_at,
            report.finished_at,
            report.processing_revision,
            report.observed_snapshot_size,
            report.added_count,
            report.updated_count,
            report.deleted_count,
            report.unchanged_count,
            report.ignored_count,
            report.failed_source_path,
            report.error_summary,
            report.corpus_revision,
        ),
    )
