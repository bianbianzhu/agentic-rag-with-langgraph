"""Persistent Conversation Thread contracts."""

from pydantic import BaseModel, ConfigDict, Field


class ThreadState(BaseModel):
    """Minimal persistent state for the development-loop tracer bullet."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    completed_turns: int = Field(default=0, ge=0)
