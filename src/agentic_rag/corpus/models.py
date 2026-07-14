"""Corpus Sync contracts owned by the corpus module."""

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class KnowledgeSource(StrEnum):
    """The fixed v1 Knowledge Sources."""

    ENGINEERING_DOCS = "engineering-docs"
    OPERATIONAL_RUNBOOKS = "operational-runbooks"


class ProcessingConfig(BaseModel):
    """Versioned inputs that define one Processing Revision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    parser_version: str = Field(min_length=1)
    chunker_version: str = Field(min_length=1)
    embedding_model: str = Field(min_length=1)


class SyncStatus(StrEnum):
    """Terminal status of one Corpus Sync attempt."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


class SyncReport(BaseModel):
    """Content-free audit record returned for every completed attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: UUID
    knowledge_source: KnowledgeSource
    status: SyncStatus
    started_at: datetime
    finished_at: datetime
    processing_revision: str
    observed_snapshot_size: int = Field(ge=0)
    added_count: int = Field(ge=0)
    updated_count: int = Field(ge=0)
    deleted_count: int = Field(ge=0)
    unchanged_count: int = Field(ge=0)
    ignored_count: int = Field(ge=0)
    failed_source_path: str | None = None
    error_summary: str | None = None
    corpus_revision: str | None = None
