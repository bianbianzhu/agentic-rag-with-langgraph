"""LangGraph state owned by the Reference System graph."""

from typing import Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from agentic_rag.conversation import ThreadState


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
