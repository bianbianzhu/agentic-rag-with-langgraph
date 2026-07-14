"""Trusted dependencies supplied to one graph run."""

from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from psycopg_pool import ConnectionPool


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    """Immutable trusted dependencies for one graph run."""

    principal_id: str
    database_pool: ConnectionPool | None = None
    answer_model: BaseChatModel | None = None

    def __post_init__(self) -> None:
        if not self.principal_id.strip():
            raise ValueError("principal_id must not be empty")
