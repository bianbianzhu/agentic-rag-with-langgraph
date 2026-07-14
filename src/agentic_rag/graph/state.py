"""LangGraph state owned by the Reference System graph."""

from typing import Literal, NotRequired, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from agentic_rag.conversation import ThreadState
from agentic_rag.authorization import AuthorizationSnapshot
from agentic_rag.agents.answering import VerificationDecision
from agentic_rag.citations import (
    CitationDraft,
    CitationMapping,
    CitationValidation,
    CitedAnswer,
)
from agentic_rag.retrieval import EvidenceSet


class CurrentTurnWork(BaseModel):
    """Disposable work for the active Turn."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    user_message: str = Field(min_length=1)
    assistant_message: str | None = None
    status: Literal["pending", "answered"] = "pending"


class GraphState(TypedDict):
    """Checkpointed mutable state; trusted dependencies stay in RuntimeContext."""

    thread: ThreadState | dict[str, object]
    current_turn: CurrentTurnWork | dict[str, object] | None


class AnswerGraphState(TypedDict):
    """Disposable buffered state for the bounded Answer subgraph."""

    question: str
    evidence_set: EvidenceSet | dict[str, object]
    authorization_snapshot: AuthorizationSnapshot | dict[str, object]
    citation_mappings: NotRequired[
        tuple[CitationMapping, ...] | list[dict[str, object]]
    ]
    draft: NotRequired[CitationDraft | dict[str, object] | None]
    validation: NotRequired[CitationValidation | dict[str, object] | None]
    verification: NotRequired[
        VerificationDecision | dict[str, object] | None
    ]
    repair_count: int
    cited_answer: NotRequired[CitedAnswer | dict[str, object] | None]
    failure_reason: NotRequired[str | None]
    status: Literal["pending", "answered", "incomplete"]
