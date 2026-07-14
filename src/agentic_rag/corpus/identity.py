"""Deterministic corpus identity functions."""

from hashlib import sha256
from pathlib import PurePosixPath

from agentic_rag.corpus.models import KnowledgeSource, ProcessingConfig


def normalize_source_path(source_path: str) -> str:
    """Return one canonical source-relative POSIX path."""

    path = PurePosixPath(source_path)
    if (
        not source_path
        or source_path.startswith("/")
        or "\\" in source_path
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("source_path must be a normalized source-relative path")
    return path.as_posix()


def source_document_id(
    knowledge_source: KnowledgeSource, source_path: str
) -> str:
    """Derive Source Document identity from source and normalized path."""

    normalized_path = normalize_source_path(source_path)
    digest = sha256(
        f"{knowledge_source.value}\0{normalized_path}".encode()
    ).hexdigest()
    return f"doc_{digest}"


def source_revision(raw_source: bytes, access_scope_revision: str) -> str:
    """Derive a Source Revision from bytes and trusted Access Scope metadata."""

    if not access_scope_revision:
        raise ValueError("access_scope_revision must not be empty")
    digest = sha256(
        raw_source + b"\0" + access_scope_revision.encode("utf-8")
    ).hexdigest()
    return f"src_{digest}"


def processing_revision(config: ProcessingConfig) -> str:
    """Derive a Processing Revision from versioned processing semantics."""

    digest = sha256(config.model_dump_json().encode("utf-8")).hexdigest()
    return f"proc_{digest}"


def indexed_chunk_id(
    document_id: str,
    source_revision_id: str,
    processing_revision_id: str,
    ordinal: int,
) -> str:
    """Derive one revision-scoped Indexed Chunk identifier."""

    if ordinal < 0:
        raise ValueError("ordinal must not be negative")
    identity = (
        f"{document_id}\0{source_revision_id}\0"
        f"{processing_revision_id}\0{ordinal}"
    )
    return f"chunk_{sha256(identity.encode()).hexdigest()}"
