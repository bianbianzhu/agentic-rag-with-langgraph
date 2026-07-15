"""LangGraph state owned by the Reference System graph."""

import json
from hashlib import sha256
from typing import Literal, NotRequired, Self, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentic_rag.agents.answering import VerificationDecision
from agentic_rag.agents.research import (
    EvidenceAssessment,
    QueryRefinement,
    ResearchPlan,
)
from agentic_rag.authorization import AuthorizationSnapshot
from agentic_rag.citations import (
    CitationDraft,
    CitationMapping,
    CitationValidation,
    CitedAnswer,
)
from agentic_rag.conversation import ThreadState, TurnOutcome
from agentic_rag.retrieval import EvidenceSet, RetrievalResult


class CurrentTurnWork(BaseModel):
    """Disposable work for the active Turn."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["current-turn-work-v1"] = "current-turn-work-v1"
    turn_id: str = Field(min_length=1)
    user_message: str = Field(min_length=1, max_length=4_000)
    stage: Literal["new", "prepared", "compacting", "terminal"] = "new"
    model_calls: int = Field(default=0, ge=0, le=10)
    compaction_attempts: int = Field(default=0, ge=0, le=2)
    standalone_question: str | None = None
    assistant_message: str | None = None
    outcome: TurnOutcome | None = None
    terminal_reason: str | None = None

    @model_validator(mode="after")
    def require_complete_terminal_work(self) -> Self:
        terminal_values = (
            self.standalone_question,
            self.assistant_message,
            self.outcome,
        )
        if self.stage == "terminal" and any(
            value is None for value in terminal_values
        ):
            raise ValueError("terminal Current Turn Work is incomplete")
        if self.stage != "terminal" and any(
            value is not None
            for value in terminal_values + (self.terminal_reason,)
        ):
            raise ValueError("non-terminal Current Turn Work has a result")
        if self.stage == "terminal" and self.outcome is not None:
            if (self.outcome is TurnOutcome.ANSWERED) == (
                self.terminal_reason is not None
            ):
                raise ValueError("terminal reason and Turn outcome disagree")
        return self

    def resume_fingerprint(self) -> str:
        """Fingerprint every code-owned field persisted at a safe boundary."""

        encoded = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return f"turn_work_{sha256(encoded).hexdigest()}"


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


class ResearchCounters(BaseModel):
    """Code-owned consumed research operations for one Turn."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_calls: int = Field(default=0, ge=0, le=10)
    retrieval_requests: int = Field(default=0, ge=0, le=2)
    research_iterations: int = Field(default=0, ge=0, le=1)


class ResearchGraphState(TypedDict):
    """Disposable state for the bounded Research subgraph."""

    question: str
    authorization_snapshot: AuthorizationSnapshot | dict[str, object]
    counters: ResearchCounters | dict[str, object]
    retrieval_results: list[RetrievalResult | dict[str, object]]
    plan: NotRequired[ResearchPlan | dict[str, object]]
    assessment: NotRequired[EvidenceAssessment | dict[str, object]]
    refinement: NotRequired[QueryRefinement | dict[str, object]]
    evidence_set: NotRequired[EvidenceSet | dict[str, object]]
    active_query: NotRequired[str]
    knowledge_sources: NotRequired[list[str]]
    response_text: NotRequired[str]
    terminal_reason: NotRequired[str]
    failure_reason: NotRequired[str]
    next_node: NotRequired[Literal["retrieve", "assess", "refine", "end"]]
    status: Literal[
        "pending",
        "evidence_ready",
        "direct",
        "clarification",
        "refused",
        "incomplete",
        "failed",
    ]
