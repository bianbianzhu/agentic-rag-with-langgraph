"""Trusted dependencies supplied to one graph run."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    """Immutable trusted identity for the current run."""

    principal_id: str

    def __post_init__(self) -> None:
        if not self.principal_id.strip():
            raise ValueError("principal_id must not be empty")
