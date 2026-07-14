"""Trusted dependencies supplied to one graph run."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from time import monotonic
from typing import Literal

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, ConfigDict, Field, computed_field

from agentic_rag.agents.research import ResearchAgentConfig
from agentic_rag.retrieval import Reranker, RetrievalConfig


class TurnExecutionBudget(BaseModel):
    """Code-owned hard limits used by the current Turn graph."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal["turn-budget-v1"] = "turn-budget-v1"
    model_call_limit: int = Field(default=10, ge=1, le=10)
    retrieval_request_limit: int = Field(default=2, ge=1, le=2)
    research_iteration_limit: int = Field(default=1, ge=0, le=1)
    deadline_seconds: float = Field(default=90, gt=0, le=90)

    @computed_field
    @property
    def fingerprint(self) -> str:
        payload = self.model_dump(exclude={"fingerprint"})
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode()
        return f"turn_budget_{sha256(encoded).hexdigest()}"


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    """Immutable trusted dependencies for one graph run."""

    principal_id: str
    database_pool: ConnectionPool | None = None
    answer_model: BaseChatModel | None = None
    research_model: BaseChatModel | None = None
    embedder: Embeddings | None = None
    reranker: Reranker | None = None
    retrieval_config: RetrievalConfig | None = None
    research_config: ResearchAgentConfig = ResearchAgentConfig()
    execution_budget: TurnExecutionBudget = TurnExecutionBudget()
    clock: Callable[[], float] = monotonic
    deadline_at: float | None = None

    def __post_init__(self) -> None:
        if not self.principal_id.strip():
            raise ValueError("principal_id must not be empty")
        now = self.clock()
        if self.deadline_at is None:
            object.__setattr__(
                self,
                "deadline_at",
                now + self.execution_budget.deadline_seconds,
            )
        elif (
            not isfinite(self.deadline_at)
            or self.deadline_at
            > now + self.execution_budget.deadline_seconds
        ):
            raise ValueError("deadline_at exceeds the Turn budget")
