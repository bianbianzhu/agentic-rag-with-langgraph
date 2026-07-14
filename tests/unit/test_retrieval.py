"""L1 contracts for bounded Retrieval Requests."""

import pytest
from pydantic import ValidationError

from agentic_rag.corpus import KnowledgeSource
from agentic_rag.retrieval import (
    EvidenceItem,
    KnowledgeSourceRetrievalOutcome,
    RetrievalRequest,
    RetrievalResult,
)


def test_retrieval_request_rejects_principal_override() -> None:
    with pytest.raises(ValidationError):
        RetrievalRequest.model_validate(
            {
                "request_id": "request-1",
                "query": "Why did the rollback fail?",
                "knowledge_sources": (KnowledgeSource.ENGINEERING_DOCS,),
                "dense_candidate_limit": 8,
                "lexical_candidate_limit": 8,
                "result_limit": 4,
                "principal_id": "bob",
            }
        )


def test_evidence_item_rejects_access_grant_details() -> None:
    with pytest.raises(ValidationError):
        EvidenceItem.model_validate(
            {
                "chunk_id": "chunk-1",
                "document_id": "document-1",
                "knowledge_source": "engineering-docs",
                "source_revision": "source-1",
                "processing_revision": "processing-1",
                "corpus_revision": "corpus-1",
                "chunk_text": "Authorized evidence.",
                "context_text": None,
                "context_chunk_ids": [],
                "source_path": "payments/rollback.md",
                "title": "Rollback",
                "source_locator": {
                    "kind": "markdown-section",
                    "section_path": ["Recovery"],
                },
                "provenance": {
                    "retrieval_request_id": "request-1",
                    "dense_rank": 1,
                    "dense_score": 0.9,
                    "lexical_rank": 1,
                    "lexical_score": 0.8,
                    "fused_rank": 1,
                    "fused_score": 0.03,
                },
                "access_grants": ["payments-on-call"],
            }
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dense_candidate_limit", 51),
        ("lexical_candidate_limit", 51),
        ("result_limit", 21),
    ],
)
def test_retrieval_request_rejects_unbounded_budgets(
    field: str, value: int
) -> None:
    request = {
        "request_id": "request-1",
        "query": "Why did the rollback fail?",
        "knowledge_sources": ["engineering-docs"],
        "dense_candidate_limit": 8,
        "lexical_candidate_limit": 8,
        "result_limit": 4,
        field: value,
    }

    with pytest.raises(ValidationError):
        RetrievalRequest.model_validate(request)


def test_retrieval_request_rejects_duplicate_knowledge_sources() -> None:
    with pytest.raises(ValidationError):
        RetrievalRequest(
            request_id="request-1",
            query="Why did the rollback fail?",
            knowledge_sources=(
                KnowledgeSource.ENGINEERING_DOCS,
                KnowledgeSource.ENGINEERING_DOCS,
            ),
            dense_candidate_limit=8,
            lexical_candidate_limit=8,
            result_limit=4,
        )


def test_source_outcome_rejects_failure_fields_on_completed_status() -> None:
    with pytest.raises(ValidationError):
        KnowledgeSourceRetrievalOutcome.model_validate(
            {
                "knowledge_source": "engineering-docs",
                "corpus_revision": {
                    "knowledge_source": "engineering-docs",
                    "revision": "corpus-test-v1",
                },
                "status": "completed",
                "candidate_counts": {
                    "dense": 1,
                    "lexical": 1,
                    "deduplicated": 1,
                    "reranked": None,
                    "returned": 1,
                },
                "timings": {
                    "embedding_ms": 0,
                    "dense_ms": 1,
                    "lexical_ms": 1,
                    "fusion_ms": 1,
                    "total_ms": 3,
                },
                "failed_stage": "dense",
                "error_code": "dense_failed",
            }
        )


def test_failed_retrieval_result_requires_stage_and_error_code() -> None:
    with pytest.raises(ValidationError):
        RetrievalResult.model_validate(
            {
                "retrieval_contract_version": "retrieval-v1",
                "retrieval_config_fingerprint": "retrieval_" + "0" * 64,
                "request_id": "request-1",
                "executed_query": "rollback recovery",
                "requested_knowledge_sources": ["engineering-docs"],
                "corpus_revisions": [],
                "access_scope_fingerprint": None,
                "candidate_counts": {
                    "dense": 0,
                    "lexical": 0,
                    "deduplicated": 0,
                    "reranked": None,
                    "returned": 0,
                },
                "timings": {
                    "embedding_ms": 0,
                    "dense_ms": 0,
                    "lexical_ms": 0,
                    "fusion_ms": 0,
                    "total_ms": 0,
                },
                "evidence_items": [],
                "source_outcomes": [],
                "status": "failed",
                "failed_stage": None,
                "error_code": None,
            }
        )
