"""Contracts owned by authorized hybrid retrieval."""

from collections.abc import Callable
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from agentic_rag.corpus.models import (
    CorpusRevision,
    KnowledgeSource,
    SourceLocator,
)


NonEmptyText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]


class RetrievalRequest(BaseModel):
    """Bounded retrieval intent without authority-bearing fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: NonEmptyText
    query: NonEmptyText
    knowledge_sources: tuple[KnowledgeSource, ...] = Field(
        min_length=1, max_length=len(KnowledgeSource)
    )
    dense_candidate_limit: int = Field(ge=1, le=50)
    lexical_candidate_limit: int = Field(ge=1, le=50)
    result_limit: int = Field(ge=1, le=8)

    @model_validator(mode="after")
    def require_distinct_knowledge_sources(self) -> Self:
        if len(set(self.knowledge_sources)) != len(self.knowledge_sources):
            raise ValueError("Knowledge Sources must be distinct")
        return self


class RetrievalProvenance(BaseModel):
    """Branch and fusion ranks for one authorized Indexed Chunk."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, allow_inf_nan=False
    )

    retrieval_request_id: NonEmptyText
    dense_rank: int | None = Field(default=None, ge=1)
    dense_score: float | None = None
    lexical_rank: int | None = Field(default=None, ge=1)
    lexical_score: float | None = None
    fused_rank: int = Field(ge=1)
    fused_score: float = Field(gt=0)
    rerank_rank: int | None = Field(default=None, ge=1)
    rerank_score: float | None = None


class EvidenceItem(BaseModel):
    """One authorized Indexed Chunk with citation and retrieval data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: NonEmptyText
    document_id: NonEmptyText
    knowledge_source: KnowledgeSource
    source_revision: NonEmptyText
    processing_revision: NonEmptyText
    corpus_revision: NonEmptyText
    chunk_text: NonEmptyText
    context_text: str | None = None
    context_chunk_ids: tuple[NonEmptyText, ...] = ()
    source_path: NonEmptyText
    title: NonEmptyText
    source_locator: SourceLocator
    provenance: tuple[RetrievalProvenance, ...] = Field(min_length=1)


class RerankCandidate(BaseModel):
    """Bounded authorized candidate exposed to the listwise reranker."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: NonEmptyText
    content: NonEmptyText
    title: NonEmptyText
    source_path: NonEmptyText


class RerankItem(BaseModel):
    """One candidate in the reranker's ordered structured output."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, allow_inf_nan=False
    )

    chunk_id: NonEmptyText
    score: float = Field(ge=0, le=1)


class RerankOutput(BaseModel):
    """Complete ordered reranking of the supplied candidates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    items: tuple[RerankItem, ...] = Field(min_length=1, max_length=50)


Reranker = Callable[[str, tuple[RerankCandidate, ...]], RerankOutput]


class RetrievalStatus(StrEnum):
    """Structured outcome of one Retrieval Request."""

    COMPLETED = "completed"
    NO_EVIDENCE = "no_evidence"
    FAILED = "failed"


class RetrievalStage(StrEnum):
    """Retrieval stage that may fail without exposing an exception."""

    AUTHORIZATION = "authorization"
    EMBEDDING = "embedding"
    DENSE = "dense"
    LEXICAL = "lexical"
    RERANK = "rerank"
    CONTEXT = "context"
    CORPUS = "corpus"


class RetrievalErrorCode(StrEnum):
    """Structured errors currently emitted by retrieval."""

    AUTHORIZATION_CONTEXT_MISSING = "authorization_context_missing"
    EMBEDDING_FAILED = "embedding_failed"
    DENSE_FAILED = "dense_failed"
    LEXICAL_FAILED = "lexical_failed"
    RERANK_FAILED = "rerank_failed"
    CORPUS_UNAVAILABLE = "corpus_unavailable"


class RetrievalConfig(BaseModel):
    """Versioned non-secret configuration for retrieval and reranking."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_version: Literal["retrieval-v1"] = "retrieval-v1"
    embedding_model: NonEmptyText
    dense_query_semantics: Literal["exact-cosine-v1"] = "exact-cosine-v1"
    lexical_query_semantics: Literal["english-websearch-v1"] = (
        "english-websearch-v1"
    )
    rrf_k: int = Field(default=60, ge=1, le=1_000)
    fusion_candidate_limit: int = Field(default=50, ge=1, le=50)
    rerank_candidate_limit: int = Field(default=20, ge=1, le=50)
    reranker_model: Literal["openai:gpt-5.4-nano-2026-03-17"] = (
        "openai:gpt-5.4-nano-2026-03-17"
    )
    reranker_prompt_version: Literal["listwise-v2"] = "listwise-v2"
    relevance_threshold: float = Field(default=0.5, ge=0, le=1)
    context_window: int = Field(default=1, ge=0, le=3)
    evidence_token_limit: int = Field(default=6_000, ge=1, le=6_000)
    token_counting_semantics: Literal["chars-div-4-v1"] = (
        "chars-div-4-v1"
    )


class CandidateCounts(BaseModel):
    """Authorized candidate counts only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dense: int = Field(ge=0)
    lexical: int = Field(ge=0)
    deduplicated: int = Field(ge=0)
    reranked: int = Field(ge=0)
    returned: int = Field(ge=0)


class RetrievalTimings(BaseModel):
    """Non-negative per-stage elapsed milliseconds."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, allow_inf_nan=False
    )

    embedding_ms: float = Field(ge=0)
    dense_ms: float = Field(ge=0)
    lexical_ms: float = Field(ge=0)
    fusion_ms: float = Field(ge=0)
    rerank_ms: float = Field(ge=0)
    total_ms: float = Field(ge=0)


class KnowledgeSourceRetrievalOutcome(BaseModel):
    """Independent outcome retained for one requested Knowledge Source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    knowledge_source: KnowledgeSource
    corpus_revision: CorpusRevision | None = None
    status: RetrievalStatus
    candidate_counts: CandidateCounts
    timings: RetrievalTimings
    failed_stage: RetrievalStage | None = None
    error_code: RetrievalErrorCode | None = None

    @model_validator(mode="after")
    def require_consistent_status(self) -> Self:
        has_complete_failure = (
            self.failed_stage is not None and self.error_code is not None
        )
        has_partial_failure = (self.failed_stage is None) != (
            self.error_code is None
        )
        if has_partial_failure or (
            (self.status is RetrievalStatus.FAILED) != has_complete_failure
        ):
            raise ValueError("Retrieval status and failure fields disagree")
        if (
            self.status is RetrievalStatus.NO_EVIDENCE
            and self.candidate_counts.deduplicated != 0
        ):
            raise ValueError("no_evidence cannot contain candidates")
        if (
            self.status is RetrievalStatus.COMPLETED
            and self.candidate_counts.deduplicated == 0
        ):
            raise ValueError("completed retrieval requires candidates")
        return self


class RetrievalResult(BaseModel):
    """Sanitized structured result of one bounded Retrieval Request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    retrieval_contract_version: Literal["retrieval-v1"] = "retrieval-v1"
    retrieval_config_fingerprint: str = Field(
        pattern=r"^retrieval_[0-9a-f]{64}$"
    )
    request_id: NonEmptyText
    executed_query: NonEmptyText
    requested_knowledge_sources: tuple[KnowledgeSource, ...]
    corpus_revisions: tuple[CorpusRevision, ...]
    access_scope_fingerprint: NonEmptyText | None
    candidate_counts: CandidateCounts
    timings: RetrievalTimings
    evidence_items: tuple[EvidenceItem, ...]
    source_outcomes: tuple[KnowledgeSourceRetrievalOutcome, ...]
    status: RetrievalStatus
    failed_stage: RetrievalStage | None = None
    error_code: RetrievalErrorCode | None = None

    @model_validator(mode="after")
    def require_consistent_status(self) -> Self:
        has_complete_failure = (
            self.failed_stage is not None and self.error_code is not None
        )
        has_partial_failure = (self.failed_stage is None) != (
            self.error_code is None
        )
        if has_partial_failure or (
            (self.status is RetrievalStatus.FAILED) != has_complete_failure
        ):
            raise ValueError("Retrieval status and failure fields disagree")
        if self.candidate_counts.returned != len(self.evidence_items):
            raise ValueError("returned count must match Evidence Items")
        if self.status is RetrievalStatus.FAILED and self.evidence_items:
            raise ValueError("failed retrieval cannot expose Evidence Items")
        if (
            self.status is RetrievalStatus.COMPLETED
        ) != bool(self.evidence_items):
            if self.status is not RetrievalStatus.FAILED:
                raise ValueError("completed status must match Evidence Items")
        return self


class EvidenceSetRequestOutcome(BaseModel):
    """Status retained for one required Retrieval Request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: NonEmptyText
    retrieval_config_fingerprint: str = Field(
        pattern=r"^retrieval_[0-9a-f]{64}$"
    )
    status: RetrievalStatus
    failed_stage: RetrievalStage | None = None
    error_code: RetrievalErrorCode | None = None

    @model_validator(mode="after")
    def require_consistent_status(self) -> Self:
        has_failure = self.failed_stage is not None and self.error_code is not None
        if (self.failed_stage is None) != (self.error_code is None) or (
            (self.status is RetrievalStatus.FAILED) != has_failure
        ):
            raise ValueError("Evidence Set request status is inconsistent")
        return self


class EvidenceSetBudget(BaseModel):
    """Hard per-answer Evidence Set limits."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    item_limit: int = Field(default=8, ge=1, le=8)
    token_limit: int = Field(default=6_000, ge=1, le=6_000)


class EvidenceSet(BaseModel):
    """Bounded authorized evidence assembled across required requests."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_items: tuple[EvidenceItem, ...] = Field(max_length=8)
    request_outcomes: tuple[EvidenceSetRequestOutcome, ...]
    budget: EvidenceSetBudget
    evidence_config_fingerprint: str = Field(
        pattern=r"^evidence_[0-9a-f]{64}$"
    )
    complete: bool
    token_count: int = Field(ge=0, le=6_000)
    item_budget_exhausted: bool
    token_budget_exhausted: bool
    context_expanded: bool
    context_ms: float = Field(ge=0)
    failed_stage: RetrievalStage | None = None
    error_code: RetrievalErrorCode | None = None

    @model_validator(mode="after")
    def require_consistent_completeness(self) -> Self:
        has_failure = self.failed_stage is not None and self.error_code is not None
        if (self.failed_stage is None) != (self.error_code is None):
            raise ValueError("Evidence Set failure fields disagree")
        expected = not has_failure and all(
            outcome.status is not RetrievalStatus.FAILED
            for outcome in self.request_outcomes
        )
        if self.complete != expected:
            raise ValueError("Evidence Set completeness disagrees with outcomes")
        if has_failure and self.evidence_items:
            raise ValueError("failed Evidence Set cannot expose Evidence Items")
        if self.token_count > self.budget.token_limit:
            raise ValueError("Evidence Set exceeds its token budget")
        if len(self.evidence_items) > self.budget.item_limit:
            raise ValueError("Evidence Set exceeds its item budget")
        return self
