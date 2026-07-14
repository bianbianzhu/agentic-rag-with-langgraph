"""L1 contracts for bounded cross-request Evidence Set assembly."""

import pytest
from pydantic import ValidationError

from agentic_rag.corpus import KnowledgeSource
from agentic_rag.corpus.models import SourceLocator
from agentic_rag.retrieval import (
    CandidateCounts,
    EvidenceItem,
    EvidenceSet,
    RetrievalConfig,
    RetrievalErrorCode,
    RetrievalProvenance,
    RetrievalResult,
    RetrievalStage,
    RetrievalStatus,
    RetrievalTimings,
    assemble_evidence_set,
)


def test_evidence_set_round_robins_without_cross_request_score_comparison() -> None:
    first = _result(
        "request-1",
        (
            _item("chunk-a", "request-1", rank=1, score=0.6),
            _item("chunk-b", "request-1", rank=2, score=0.5),
        ),
    )
    second = _result(
        "request-2",
        (
            _item("chunk-c", "request-2", rank=1, score=0.99),
            _item("chunk-a", "request-2", rank=2, score=0.98),
        ),
    )

    evidence_set = assemble_evidence_set((first, second), _config())

    assert [item.chunk_id for item in evidence_set.evidence_items] == [
        "chunk-a",
        "chunk-c",
        "chunk-b",
    ]
    assert [
        provenance.retrieval_request_id
        for provenance in evidence_set.evidence_items[0].provenance
    ] == ["request-1", "request-2"]
    assert evidence_set.complete is True
    assert evidence_set.request_outcomes[0].retrieval_config_fingerprint == (
        first.retrieval_config_fingerprint
    )

    payload = evidence_set.model_dump()
    payload["budget"]["item_limit"] = 1
    with pytest.raises(ValidationError):
        EvidenceSet.model_validate(payload)


def test_evidence_set_preserves_no_evidence_and_required_failure() -> None:
    no_evidence = _result("request-empty", (), RetrievalStatus.NO_EVIDENCE)
    failed = _result(
        "request-failed",
        (),
        RetrievalStatus.FAILED,
        failed_stage=RetrievalStage.RERANK,
        error_code=RetrievalErrorCode.RERANK_FAILED,
    )

    evidence_set = assemble_evidence_set((no_evidence, failed), _config())

    assert [outcome.status for outcome in evidence_set.request_outcomes] == [
        RetrievalStatus.NO_EVIDENCE,
        RetrievalStatus.FAILED,
    ]
    assert evidence_set.complete is False
    assert evidence_set.evidence_items == ()

    payload = evidence_set.model_dump()
    payload["complete"] = True
    with pytest.raises(ValidationError):
        EvidenceSet.model_validate(payload)


def test_failed_retrieval_result_cannot_expose_partial_evidence() -> None:
    with pytest.raises(ValidationError):
        _result(
            "request-failed-with-evidence",
            (_item("chunk-a", "request-failed-with-evidence", rank=1, score=1),),
            RetrievalStatus.FAILED,
            failed_stage=RetrievalStage.RERANK,
            error_code=RetrievalErrorCode.RERANK_FAILED,
        )


def test_evidence_set_enforces_item_and_token_budgets() -> None:
    result = _result(
        "request-budget",
        (
            _item("chunk-short", "request-budget", rank=1, score=1.0),
            _item(
                "chunk-too-long",
                "request-budget",
                rank=2,
                score=0.9,
                text="x" * 40,
            ),
            _item("chunk-third", "request-budget", rank=3, score=0.8),
        ),
    )

    evidence_set = assemble_evidence_set(
        (result,), _config(token_limit=5), item_limit=1
    )

    assert [item.chunk_id for item in evidence_set.evidence_items] == [
        "chunk-short"
    ]
    assert evidence_set.item_budget_exhausted is True
    assert evidence_set.token_budget_exhausted is True
    assert evidence_set.token_count <= 5

    payload = evidence_set.model_dump()
    payload["token_count"] = 6
    with pytest.raises(ValidationError):
        EvidenceSet.model_validate(payload)


def test_evidence_set_selects_matched_chunks_before_optional_context() -> None:
    item_with_unselected_context = _item(
        "chunk-match", "request-context", rank=1, score=1
    ).model_copy(
        update={
            "context_text": "optional context " * 100,
            "context_chunk_ids": ("chunk-neighbor",),
        }
    )

    evidence_set = assemble_evidence_set(
        (_result("request-context", (item_with_unselected_context,)),),
        _config(token_limit=4),
    )

    assert [item.chunk_id for item in evidence_set.evidence_items] == [
        "chunk-match"
    ]
    assert evidence_set.evidence_items[0].context_text is None
    assert evidence_set.token_count == 4


def _item(
    chunk_id: str,
    request_id: str,
    *,
    rank: int,
    score: float,
    text: str = "short evidence",
) -> EvidenceItem:
    return EvidenceItem(
        chunk_id=chunk_id,
        document_id=f"document-{chunk_id}",
        knowledge_source=KnowledgeSource.ENGINEERING_DOCS,
        source_revision="source-v1",
        processing_revision="processing-v1",
        corpus_revision="corpus-v1",
        chunk_text=text,
        source_path=f"{chunk_id}.md",
        title=chunk_id,
        source_locator=SourceLocator(section_path=("Evidence",)),
        provenance=(
            RetrievalProvenance(
                retrieval_request_id=request_id,
                fused_rank=rank,
                fused_score=0.01,
                rerank_rank=rank,
                rerank_score=score,
            ),
        ),
    )


def _config(token_limit: int = 6_000) -> RetrievalConfig:
    return RetrievalConfig(
        embedding_model="deterministic-test-v1",
        evidence_token_limit=token_limit,
    )


def _result(
    request_id: str,
    items: tuple[EvidenceItem, ...],
    status: RetrievalStatus = RetrievalStatus.COMPLETED,
    *,
    failed_stage: RetrievalStage | None = None,
    error_code: RetrievalErrorCode | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        retrieval_config_fingerprint="retrieval_" + "0" * 64,
        request_id=request_id,
        executed_query=request_id,
        requested_knowledge_sources=(KnowledgeSource.ENGINEERING_DOCS,),
        corpus_revisions=(),
        access_scope_fingerprint="scope-v1",
        candidate_counts=CandidateCounts(
            dense=len(items),
            lexical=len(items),
            deduplicated=len(items),
            reranked=len(items),
            returned=len(items),
        ),
        timings=RetrievalTimings(
            embedding_ms=0,
            dense_ms=0,
            lexical_ms=0,
            fusion_ms=0,
            rerank_ms=0,
            total_ms=0,
        ),
        evidence_items=items,
        source_outcomes=(),
        status=status,
        failed_stage=failed_stage,
        error_code=error_code,
    )
