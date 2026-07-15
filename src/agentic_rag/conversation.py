"""Persistent Conversation Thread contracts and transitions."""

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)
from psycopg_pool import ConnectionPool

from agentic_rag.authorization import (
    AuthorizationSnapshot,
    source_document_is_authorized,
)
from agentic_rag.corpus import KnowledgeSource
from agentic_rag.corpus.models import SourceLocator


NonEmptyText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]


class TurnOutcome(StrEnum):
    """Terminal outcomes allowed into semantic Thread Memory."""

    ANSWERED = "answered"
    REFUSED = "refused"
    CLARIFICATION_REQUESTED = "clarification_requested"
    FAILED = "failed"


class SummaryItemKind(StrEnum):
    """Inspectable semantic roles retained by Conversation Summary."""

    TOPIC = "topic"
    USER_CONSTRAINT = "user_constraint"
    EVIDENCE_CLAIM = "evidence_claim"
    OUTCOME = "outcome"
    UNRESOLVED_QUESTION = "unresolved_question"


class CitationDependency(BaseModel):
    """Historical identity to reauthorize before projecting an answer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    knowledge_source: KnowledgeSource
    document_id: NonEmptyText
    source_revision: NonEmptyText
    source_locator: SourceLocator


class TurnRecord(BaseModel):
    """One complete semantic Turn without raw Evidence or draft state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    turn_id: NonEmptyText
    user_message: NonEmptyText
    standalone_question: NonEmptyText
    assistant_message: NonEmptyText
    outcome: TurnOutcome
    terminal_reason: NonEmptyText | None = None
    citation_dependencies: tuple[CitationDependency, ...] = Field(
        default=(), max_length=8
    )

    @model_validator(mode="after")
    def require_terminal_reason_for_non_answers(self) -> Self:
        if (self.outcome is TurnOutcome.ANSWERED) == (
            self.terminal_reason is not None
        ):
            raise ValueError("terminal reason and Turn outcome disagree")
        return self


class SummaryItem(BaseModel):
    """One bounded summary fact with removable citation dependencies."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: SummaryItemKind
    text: NonEmptyText
    source_turn_ids: tuple[NonEmptyText, ...] = Field(min_length=1)
    citation_dependencies: tuple[CitationDependency, ...] = Field(
        default=(), max_length=8
    )

    @model_validator(mode="after")
    def require_evidence_claim_dependencies(self) -> Self:
        if (
            self.kind is SummaryItemKind.EVIDENCE_CLAIM
            and not self.citation_dependencies
        ):
            raise ValueError(
                "evidence-derived summary items need citation dependencies"
            )
        return self


class ConversationSummary(BaseModel):
    """Structured bounded summary of compacted complete Turns."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["conversation-summary-schema-v1"] = (
        "conversation-summary-schema-v1"
    )
    model_version: NonEmptyText
    items: tuple[SummaryItem, ...] = Field(default=(), max_length=40)
    covered_through_turn_id: NonEmptyText


class ThreadState(BaseModel):
    """Bounded semantic state retained across Turns."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["thread-state-v1"] = "thread-state-v1"
    principal_id: NonEmptyText | None = None
    turn_records: tuple[TurnRecord, ...] = ()
    summary: ConversationSummary | None = None
    active_turn_id: NonEmptyText | None = None
    active_turn_fingerprint: NonEmptyText | None = None

    @model_validator(mode="after")
    def require_distinct_turn_ids(self) -> Self:
        turn_ids = [record.turn_id for record in self.turn_records]
        if len(turn_ids) != len(set(turn_ids)):
            raise ValueError("Turn IDs must be distinct")
        if self.principal_id is None and (
            self.turn_records
            or self.summary
            or self.active_turn_id
            or self.active_turn_fingerprint
        ):
            raise ValueError("unbound Thread cannot contain state")
        if (
            self.active_turn_id is None
            and self.active_turn_fingerprint is not None
        ):
            raise ValueError("active Turn identity and fingerprint disagree")
        return self


class ThreadPrincipalMismatchError(ValueError):
    """Trusted Principal cannot reuse another Principal's Thread."""


class ThreadBusyError(ValueError):
    """Another Turn is active on the Conversation Thread."""


class TurnResumeIncompatibleError(ValueError):
    """A resumed Turn does not match its checkpointed input."""


class AuthorizedTurnContext(BaseModel):
    """One historical Turn projected under current authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    turn_id: NonEmptyText
    user_message: NonEmptyText
    standalone_question: NonEmptyText | None = None
    assistant_message: NonEmptyText | None = None
    citation_dependencies: tuple[CitationDependency, ...] = ()
    marker: Literal["historical_evidence_no_longer_authorized"] | None = None


class AuthorizedThreadContext(BaseModel):
    """Bounded currently authorized input for the Contextual Rewriter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    recent_turns: tuple[AuthorizedTurnContext, ...]
    summary_items: tuple[SummaryItem, ...]


def project_authorized_thread_context(
    pool: ConnectionPool,
    snapshot: AuthorizationSnapshot,
    thread: ThreadState,
    *,
    current_user_message: str,
    token_limit: int = 2_000,
) -> AuthorizedThreadContext:
    """Reauthorize and bound historical context before any model sees it."""

    if thread.principal_id != snapshot.principal_id:
        raise ThreadPrincipalMismatchError("thread_principal_mismatch")
    if not 1 <= token_limit <= 2_000:
        raise ValueError("Thread context token limit is invalid")

    if not current_user_message.strip():
        raise ValueError("current user message must not be empty")
    consumed = _approximate_tokens(current_user_message)
    if consumed > token_limit:
        raise ValueError("current user message exceeds Thread context budget")

    summary_items: tuple[SummaryItem, ...] = ()
    authorized_items = project_authorized_summary_items(
        pool, snapshot, thread.summary
    )
    if authorized_items:
        summary_tokens = _approximate_tokens(
            "".join(item.model_dump_json() for item in authorized_items)
        )
        if consumed + summary_tokens > token_limit:
            raise ValueError("Conversation Summary exceeds context budget")
        summary_items = authorized_items
        consumed += summary_tokens

    selected: list[AuthorizedTurnContext] = []
    for record in reversed(thread.turn_records):
        projected = project_authorized_turn_record(
            pool, snapshot, record
        )
        tokens = _approximate_tokens(projected.model_dump_json())
        if consumed + tokens > token_limit:
            break
        selected.append(projected)
        consumed += tokens
    return AuthorizedThreadContext(
        recent_turns=tuple(reversed(selected)),
        summary_items=summary_items,
    )


def project_authorized_turn_record(
    pool: ConnectionPool,
    snapshot: AuthorizationSnapshot,
    record: TurnRecord,
) -> AuthorizedTurnContext:
    """Remove historical answer material whose dependencies are revoked."""

    authorized = _citation_dependencies_are_authorized(
        pool, snapshot, record.citation_dependencies
    )
    return AuthorizedTurnContext(
        turn_id=record.turn_id,
        user_message=record.user_message,
        standalone_question=record.standalone_question if authorized else None,
        assistant_message=record.assistant_message if authorized else None,
        citation_dependencies=(record.citation_dependencies if authorized else ()),
        marker=(
            None if authorized else "historical_evidence_no_longer_authorized"
        ),
    )


def project_authorized_summary_items(
    pool: ConnectionPool,
    snapshot: AuthorizationSnapshot,
    summary: ConversationSummary | None,
) -> tuple[SummaryItem, ...]:
    """Remove summary items whose evidence dependencies are revoked."""

    if summary is None:
        return ()
    return tuple(
        item
        for item in summary.items
        if _citation_dependencies_are_authorized(
            pool, snapshot, item.citation_dependencies
        )
    )


def _citation_dependencies_are_authorized(
    pool: ConnectionPool,
    snapshot: AuthorizationSnapshot,
    dependencies: tuple[CitationDependency, ...],
) -> bool:
    return all(
        source_document_is_authorized(
            pool,
            snapshot,
            knowledge_source=dependency.knowledge_source.value,
            document_id=dependency.document_id,
        )
        for dependency in dependencies
    )


def begin_turn(
    thread: ThreadState,
    principal_id: str,
    turn_id: str,
    *,
    input_fingerprint: str | None = None,
) -> ThreadState:
    """Bind or resume one Turn without reexecuting completed Turn IDs."""

    if not principal_id.strip() or not turn_id.strip():
        raise ValueError("Principal and Turn ID must not be empty")
    if thread.principal_id is not None and thread.principal_id != principal_id:
        raise ThreadPrincipalMismatchError("thread_principal_mismatch")
    if any(record.turn_id == turn_id for record in thread.turn_records):
        return thread
    if thread.active_turn_id not in (None, turn_id):
        raise ThreadBusyError("thread_busy")
    if (
        thread.active_turn_id == turn_id
        and thread.active_turn_fingerprint != input_fingerprint
    ):
        raise TurnResumeIncompatibleError("turn_resume_incompatible")
    return thread.model_copy(
        update={
            "principal_id": principal_id,
            "active_turn_id": turn_id,
            "active_turn_fingerprint": input_fingerprint,
        }
    )


def commit_turn(thread: ThreadState, record: TurnRecord) -> ThreadState:
    """Atomically append one terminal record and clear the active Turn."""

    existing = next(
        (
            value
            for value in thread.turn_records
            if value.turn_id == record.turn_id
        ),
        None,
    )
    if existing is not None:
        if existing != record:
            raise ValueError("completed Turn ID has a different result")
        return thread
    if thread.active_turn_id != record.turn_id:
        raise ValueError("record does not match the active Turn")
    return thread.model_copy(
        update={
            "turn_records": thread.turn_records + (record,),
            "active_turn_id": None,
            "active_turn_fingerprint": None,
        }
    )


def thread_memory_tokens(thread: ThreadState) -> int:
    """Apply the versioned chars-div-4 approximation to semantic Memory."""

    return _approximate_tokens(thread.model_dump_json())


def active_thread_memory_tokens(
    thread: ThreadState, current_user_message: str
) -> int:
    """Count semantic Memory plus the current message against one budget."""

    return thread_memory_tokens(thread) + _approximate_tokens(
        current_user_message
    )


def oldest_compaction_prefix(
    thread: ThreadState,
    current_user_message: str,
    *,
    token_limit: int,
    summary_token_reserve: int,
) -> tuple[TurnRecord, ...]:
    """Select the smallest oldest prefix that leaves bounded recent Turns."""

    if not 0 < summary_token_reserve < token_limit:
        raise ValueError("summary token reserve is invalid")
    for count in range(1, len(thread.turn_records) + 1):
        without_compacted = thread.model_copy(
            update={
                "summary": None,
                "turn_records": thread.turn_records[count:],
            }
        )
        if (
            active_thread_memory_tokens(
                without_compacted, current_user_message
            )
            + summary_token_reserve
            <= token_limit
        ):
            return thread.turn_records[:count]
    return thread.turn_records


def _approximate_tokens(value: str) -> int:
    return max(1, len(value) // 4)


def compact_thread_memory(
    thread: ThreadState,
    summary: ConversationSummary,
    compacted_turn_ids: tuple[str, ...],
) -> ThreadState:
    """Replace only an oldest complete Turn prefix with a validated summary."""

    prefix = tuple(record.turn_id for record in thread.turn_records)
    if (
        not compacted_turn_ids
        or prefix[: len(compacted_turn_ids)] != compacted_turn_ids
        or summary.covered_through_turn_id != compacted_turn_ids[-1]
    ):
        raise ValueError("summary coverage is not an oldest Turn prefix")
    allowed_ids = set(compacted_turn_ids)
    if thread.summary is not None:
        allowed_ids.update(
            turn_id
            for item in thread.summary.items
            for turn_id in item.source_turn_ids
        )
    if any(
        not set(item.source_turn_ids).issubset(allowed_ids)
        for item in summary.items
    ):
        raise ValueError("summary references uncompacted Turns")
    return thread.model_copy(
        update={
            "summary": summary,
            "turn_records": thread.turn_records[len(compacted_turn_ids) :],
        }
    )
