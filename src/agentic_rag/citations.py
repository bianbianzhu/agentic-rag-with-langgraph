"""Deterministic citation contracts, validation, and rendering."""

from enum import StrEnum
from typing import Annotated, Self, Sequence

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)
from psycopg import Error as PsycopgError
from psycopg_pool import ConnectionPool

from agentic_rag.authorization import AuthorizationSnapshot
from agentic_rag.corpus.models import SourceLocator
from agentic_rag.retrieval import EvidenceItem


NonEmptyText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]
EvidenceKey = Annotated[str, StringConstraints(pattern=r"^E[1-8]$")]


class DraftDisposition(StrEnum):
    """Model-authored answer shape before deterministic validation."""

    FACTUAL = "factual"
    REFUSAL = "refusal"
    INSUFFICIENT = "insufficient"


class RefusalReason(StrEnum):
    """Structured reasons allowed for answers without factual claims."""

    NO_EVIDENCE = "no_evidence"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    POLICY_REFUSAL = "policy_refusal"


class DraftClaim(BaseModel):
    """One factual claim and its model-selected local Evidence keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: NonEmptyText
    citation_keys: tuple[EvidenceKey, ...] = Field(max_length=8)


class CitationDraft(BaseModel):
    """Structured model output kept outside the user projection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    disposition: DraftDisposition
    claims: tuple[DraftClaim, ...] = Field(default=(), max_length=20)
    response_text: NonEmptyText | None = None
    refusal_reason: RefusalReason | None = None

    @model_validator(mode="after")
    def require_consistent_disposition(self) -> Self:
        if self.disposition is DraftDisposition.FACTUAL:
            if not self.claims or self.response_text or self.refusal_reason:
                raise ValueError("factual draft must contain only claims")
        elif self.claims or not self.response_text or not self.refusal_reason:
            raise ValueError("non-factual draft requires text and reason")
        return self


class CitationMapping(BaseModel):
    """Code-owned mapping from one local key to authorized Evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: EvidenceKey
    chunk_id: NonEmptyText
    document_id: NonEmptyText
    source_revision: NonEmptyText
    source_path: NonEmptyText
    title: NonEmptyText
    source_locator: SourceLocator


class CitationValidationError(StrEnum):
    """Deterministic citation-integrity failures."""

    UNKNOWN_KEY = "unknown_key"
    MISSING_MAPPING = "missing_mapping"
    CONFLICTING_MAPPING = "conflicting_mapping"
    MAPPING_MISMATCH = "mapping_mismatch"
    UNCITED_FACTUAL_CLAIM = "uncited_factual_claim"


class CitationValidation(BaseModel):
    """Chain-of-thought-free validation result used for routing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    valid: bool
    errors: tuple[CitationValidationError, ...]

    @model_validator(mode="after")
    def require_consistent_validity(self) -> Self:
        if self.valid == bool(self.errors):
            raise ValueError("citation validity and errors disagree")
        return self


class CitedAnswerStatus(StrEnum):
    """Validated user-visible answer disposition."""

    FACTUAL = "factual"
    REFUSAL = "refusal"
    INSUFFICIENT = "insufficient"
    INCOMPLETE = "incomplete"


class CitedAnswer(BaseModel):
    """Only answer contract allowed into the user projection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: CitedAnswerStatus
    assistant_message: NonEmptyText
    citations: tuple[CitationMapping, ...]
    refusal_reason: RefusalReason | None = None

    @model_validator(mode="after")
    def require_consistent_projection(self) -> Self:
        if self.status is CitedAnswerStatus.FACTUAL:
            if not self.citations or self.refusal_reason is not None:
                raise ValueError("factual answer requires citations only")
        elif self.citations or self.refusal_reason is None:
            raise ValueError("non-factual answer requires a refusal reason")
        return self


class CitationHydrationError(RuntimeError):
    """Authorized citation metadata could not be hydrated safely."""


def citation_mappings(
    evidence_items: Sequence[EvidenceItem],
) -> tuple[CitationMapping, ...]:
    """Assign answer-local keys in final Evidence order."""

    return tuple(
        CitationMapping(
            key=f"E{index}",
            chunk_id=item.chunk_id,
            document_id=item.document_id,
            source_revision=item.source_revision,
            source_path=item.source_path,
            title=item.title,
            source_locator=item.source_locator,
        )
        for index, item in enumerate(evidence_items, start=1)
    )


def hydrate_citation_mappings(
    pool: ConnectionPool,
    snapshot: AuthorizationSnapshot,
    evidence_items: Sequence[EvidenceItem],
) -> tuple[CitationMapping, ...]:
    """Recheck authority and hydrate mappings from PostgreSQL."""

    if not evidence_items:
        return ()
    chunk_ids = [item.chunk_id for item in evidence_items]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise CitationHydrationError("citation Evidence is unavailable")
    try:
        with pool.connection() as connection:
            with connection.transaction():
                connection.execute(
                    "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"
                )
                rows = connection.execute(
                    """
                    SELECT indexed_chunk.chunk_id,
                           indexed_chunk.document_id,
                           indexed_chunk.knowledge_source,
                           indexed_chunk.source_revision,
                           indexed_chunk.processing_revision,
                           knowledge_source.corpus_revision,
                           source_document.source_path,
                           source_document.title,
                           indexed_chunk.heading_path
                    FROM indexed_chunks AS indexed_chunk
                    JOIN source_documents AS source_document
                      ON source_document.knowledge_source =
                         indexed_chunk.knowledge_source
                     AND source_document.document_id = indexed_chunk.document_id
                    JOIN knowledge_sources AS knowledge_source
                      ON knowledge_source.knowledge_source =
                         indexed_chunk.knowledge_source
                    WHERE indexed_chunk.chunk_id = ANY(%(chunk_ids)s)
                      AND source_document_is_authorized(
                          %(principal_id)s,
                          indexed_chunk.knowledge_source,
                          indexed_chunk.document_id
                      )
                    """,
                    {
                        "chunk_ids": chunk_ids,
                        "principal_id": snapshot.principal_id,
                    },
                ).fetchall()
    except PsycopgError:
        raise CitationHydrationError(
            "citation Evidence is unavailable"
        ) from None

    by_chunk_id = {str(row[0]): row for row in rows}
    mappings: list[CitationMapping] = []
    for index, item in enumerate(evidence_items, start=1):
        row = by_chunk_id.get(item.chunk_id)
        if row is None or (
            str(row[1]),
            str(row[2]),
            str(row[3]),
            str(row[4]),
            str(row[5]),
        ) != (
            item.document_id,
            item.knowledge_source.value,
            item.source_revision,
            item.processing_revision,
            item.corpus_revision,
        ):
            raise CitationHydrationError("citation Evidence is unavailable")
        mappings.append(
            CitationMapping(
                key=f"E{index}",
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                source_revision=item.source_revision,
                source_path=str(row[6]),
                title=str(row[7]),
                source_locator=SourceLocator(
                    section_path=tuple(str(part) for part in row[8])
                ),
            )
        )
    return tuple(mappings)


def validate_citation_draft(
    draft: CitationDraft,
    evidence_items: Sequence[EvidenceItem],
    mappings: Sequence[CitationMapping],
) -> CitationValidation:
    """Reject any citation not exactly backed by final Evidence."""

    expected = {
        mapping.key: mapping for mapping in citation_mappings(evidence_items)
    }
    provided: dict[str, list[CitationMapping]] = {}
    for mapping in mappings:
        provided.setdefault(mapping.key, []).append(mapping)

    found: set[CitationValidationError] = set()
    for key, values in provided.items():
        if len(set(values)) > 1:
            found.add(CitationValidationError.CONFLICTING_MAPPING)
        expected_mapping = expected.get(key)
        if expected_mapping is None:
            found.add(CitationValidationError.UNKNOWN_KEY)
        elif any(value != expected_mapping for value in values):
            found.add(CitationValidationError.MAPPING_MISMATCH)

    for claim in draft.claims:
        if not claim.citation_keys:
            found.add(CitationValidationError.UNCITED_FACTUAL_CLAIM)
        for key in claim.citation_keys:
            if key not in expected:
                found.add(CitationValidationError.UNKNOWN_KEY)
            elif key not in provided:
                found.add(CitationValidationError.MISSING_MAPPING)

    errors = tuple(error for error in CitationValidationError if error in found)
    return CitationValidation(valid=not errors, errors=errors)


def render_cited_answer(
    draft: CitationDraft,
    mappings: Sequence[CitationMapping],
    validation: CitationValidation,
    *,
    semantically_supported: bool,
) -> CitedAnswer:
    """Render only a citation-valid, semantically verified draft."""

    if not validation.valid or not semantically_supported:
        raise ValueError("answer draft must be valid and verified")
    if draft.disposition is not DraftDisposition.FACTUAL:
        if draft.response_text is None or draft.refusal_reason is None:
            raise ValueError("non-factual draft is incomplete")
        return CitedAnswer(
            status=CitedAnswerStatus(draft.disposition.value),
            assistant_message=draft.response_text,
            citations=(),
            refusal_reason=draft.refusal_reason,
        )

    mapping_by_key = {mapping.key: mapping for mapping in mappings}
    used_keys: list[str] = []
    rendered_claims: list[str] = []
    for claim in draft.claims:
        for key in claim.citation_keys:
            if key not in used_keys:
                used_keys.append(key)
        rendered_claims.append(
            f"{claim.text} "
            + " ".join(f"[{key}]" for key in claim.citation_keys)
        )
    return CitedAnswer(
        status=CitedAnswerStatus.FACTUAL,
        assistant_message="\n\n".join(rendered_claims),
        citations=tuple(mapping_by_key[key] for key in used_keys),
    )


def incomplete_answer() -> CitedAnswer:
    """Return a deterministic safe terminal without exposing a draft."""

    return CitedAnswer(
        status=CitedAnswerStatus.INCOMPLETE,
        assistant_message=(
            "I could not produce a supported answer from the available evidence."
        ),
        citations=(),
        refusal_reason=RefusalReason.INSUFFICIENT_EVIDENCE,
    )
