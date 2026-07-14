"""L1 deterministic citation-contract verification."""

import pytest
from pydantic import ValidationError

from agentic_rag.citations import (
    CitationDraft,
    CitedAnswerStatus,
    CitationValidationError,
    DraftClaim,
    DraftDisposition,
    RefusalReason,
    citation_mappings,
    render_cited_answer,
    validate_citation_draft,
)
from agentic_rag.corpus import KnowledgeSource
from agentic_rag.corpus.models import SourceLocator
from agentic_rag.retrieval import EvidenceItem, RetrievalProvenance


def test_unknown_citation_key_is_rejected() -> None:
    evidence = (_evidence("chunk-1"),)
    draft = _factual_draft("E2")

    result = validate_citation_draft(
        draft, evidence, citation_mappings(evidence)
    )

    assert result.valid is False
    assert result.errors == (CitationValidationError.UNKNOWN_KEY,)


def test_cited_key_without_mapping_is_rejected() -> None:
    evidence = (_evidence("chunk-1"),)

    result = validate_citation_draft(_factual_draft("E1"), evidence, ())

    assert CitationValidationError.MISSING_MAPPING in result.errors


def test_conflicting_duplicate_mapping_is_rejected() -> None:
    evidence = (_evidence("chunk-1"),)
    expected = citation_mappings(evidence)[0]
    conflicting = expected.model_copy(update={"source_path": "invented.md"})

    result = validate_citation_draft(
        _factual_draft("E1"), evidence, (expected, conflicting)
    )

    assert CitationValidationError.CONFLICTING_MAPPING in result.errors


def test_factual_claim_without_citation_is_rejected() -> None:
    evidence = (_evidence("chunk-1"),)
    draft = CitationDraft(
        disposition=DraftDisposition.FACTUAL,
        claims=(DraftClaim(text="The rollback failed.", citation_keys=()),),
    )

    result = validate_citation_draft(
        draft, evidence, citation_mappings(evidence)
    )

    assert result.errors == (CitationValidationError.UNCITED_FACTUAL_CLAIM,)


def test_refusal_requires_a_structured_reason() -> None:
    with pytest.raises(ValidationError):
        CitationDraft(
            disposition=DraftDisposition.INSUFFICIENT,
            response_text="I do not have enough evidence.",
            refusal_reason=None,
        )

    valid = CitationDraft(
        disposition=DraftDisposition.INSUFFICIENT,
        response_text="I do not have enough evidence.",
        refusal_reason=RefusalReason.INSUFFICIENT_EVIDENCE,
    )
    assert valid.claims == ()


def test_only_valid_verified_factual_draft_renders_to_user_projection() -> None:
    evidence = (_evidence("chunk-1"),)
    mappings = citation_mappings(evidence)
    draft = _factual_draft("E1")
    validation = validate_citation_draft(draft, evidence, mappings)

    answer = render_cited_answer(
        draft, mappings, validation, semantically_supported=True
    )

    assert answer.status is CitedAnswerStatus.FACTUAL
    assert answer.assistant_message == (
        "The rollback failed after schema drift. [E1]"
    )
    assert answer.citations == mappings


def test_unverified_draft_cannot_render() -> None:
    evidence = (_evidence("chunk-1"),)
    mappings = citation_mappings(evidence)
    draft = _factual_draft("E1")
    validation = validate_citation_draft(draft, evidence, mappings)

    with pytest.raises(ValueError, match="verified"):
        render_cited_answer(
            draft, mappings, validation, semantically_supported=False
        )


def test_valid_insufficient_answer_renders_without_citations() -> None:
    draft = CitationDraft(
        disposition=DraftDisposition.INSUFFICIENT,
        response_text="I do not have enough evidence to answer safely.",
        refusal_reason=RefusalReason.INSUFFICIENT_EVIDENCE,
    )
    validation = validate_citation_draft(draft, (), ())

    answer = render_cited_answer(
        draft, (), validation, semantically_supported=True
    )

    assert answer.status is CitedAnswerStatus.INSUFFICIENT
    assert answer.citations == ()
    assert answer.refusal_reason is RefusalReason.INSUFFICIENT_EVIDENCE


def _factual_draft(key: str) -> CitationDraft:
    return CitationDraft(
        disposition=DraftDisposition.FACTUAL,
        claims=(
            DraftClaim(
                text="The rollback failed after schema drift.",
                citation_keys=(key,),
            ),
        ),
    )


def _evidence(chunk_id: str) -> EvidenceItem:
    return EvidenceItem(
        chunk_id=chunk_id,
        document_id=f"document-{chunk_id}",
        knowledge_source=KnowledgeSource.ENGINEERING_DOCS,
        source_revision="source-v1",
        processing_revision="processing-v1",
        corpus_revision="corpus-v1",
        chunk_text="Schema drift caused the rollback failure.",
        source_path="payments/schema-migration.md",
        title="Payments Schema Migration",
        source_locator=SourceLocator(
            section_path=("Rollback incompatibility",)
        ),
        provenance=(
            RetrievalProvenance(
                retrieval_request_id="request-1",
                fused_rank=1,
                fused_score=0.03,
                rerank_rank=1,
                rerank_score=1,
            ),
        ),
    )
