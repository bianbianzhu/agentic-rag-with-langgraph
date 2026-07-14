"""Authorized retrieval and Evidence Set public boundary."""

from agentic_rag.retrieval.evidence import (
    assemble_evidence_set,
)

from agentic_rag.retrieval.models import (
    CandidateCounts,
    EvidenceItem,
    EvidenceSet,
    EvidenceSetBudget,
    EvidenceSetRequestOutcome,
    KnowledgeSourceRetrievalOutcome,
    RerankCandidate,
    Reranker,
    RerankItem,
    RerankOutput,
    RetrievalConfig,
    RetrievalErrorCode,
    RetrievalProvenance,
    RetrievalRequest,
    RetrievalResult,
    RetrievalStage,
    RetrievalStatus,
    RetrievalTimings,
)
from agentic_rag.retrieval.search import retrieve

__all__ = [
    "CandidateCounts",
    "EvidenceItem",
    "EvidenceSet",
    "EvidenceSetBudget",
    "EvidenceSetRequestOutcome",
    "KnowledgeSourceRetrievalOutcome",
    "RerankCandidate",
    "Reranker",
    "RerankItem",
    "RerankOutput",
    "RetrievalConfig",
    "RetrievalErrorCode",
    "RetrievalProvenance",
    "RetrievalRequest",
    "RetrievalResult",
    "RetrievalStage",
    "RetrievalStatus",
    "RetrievalTimings",
    "assemble_evidence_set",
    "retrieve",
]
