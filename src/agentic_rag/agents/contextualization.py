"""Structured Contextual Rewriter decision."""

import json
from hashlib import sha256
from typing import Annotated, Literal, Self

from langchain_core.language_models import BaseChatModel
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    computed_field,
    model_validator,
)

from agentic_rag.conversation import (
    AuthorizedThreadContext,
    AuthorizedTurnContext,
    ConversationSummary,
)


NonEmptyText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]


class ContextualizationConfig(BaseModel):
    """Pinned model and prompt/schema identity for rewriting."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: Literal["openai:gpt-5.4-mini-2026-03-17"] = (
        "openai:gpt-5.4-mini-2026-03-17"
    )
    prompt_version: Literal["contextual-rewrite-prompt-v1"] = (
        "contextual-rewrite-prompt-v1"
    )
    schema_version: Literal["contextual-rewrite-schema-v1"] = (
        "contextual-rewrite-schema-v1"
    )
    summary_prompt_version: Literal["conversation-summary-prompt-v1"] = (
        "conversation-summary-prompt-v1"
    )
    summary_schema_version: Literal["conversation-summary-schema-v1"] = (
        "conversation-summary-schema-v1"
    )
    context_token_limit: Literal[2000] = 2_000
    summary_token_limit: Literal[500] = 500
    summary_retry_limit: Literal[2] = 2

    @computed_field
    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.model_dump(exclude={"fingerprint"}),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return f"contextualization_{sha256(encoded).hexdigest()}"


class ContextualRewrite(BaseModel):
    """Inspectable rewrite without answers or free-text rationale."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    standalone_question: NonEmptyText
    depends_on_history: bool
    referenced_turn_ids: tuple[NonEmptyText, ...] = Field(max_length=20)
    clarification_needed: bool
    ambiguity_reason: NonEmptyText | None = None

    @model_validator(mode="after")
    def require_consistent_history_fields(self) -> Self:
        if self.depends_on_history != bool(self.referenced_turn_ids):
            raise ValueError("history dependency and Turn IDs disagree")
        if len(set(self.referenced_turn_ids)) != len(
            self.referenced_turn_ids
        ):
            raise ValueError("referenced Turn IDs must be distinct")
        if self.clarification_needed != (self.ambiguity_reason is not None):
            raise ValueError("clarification and ambiguity reason disagree")
        return self


def rewrite_question(
    model: BaseChatModel,
    user_message: str,
    context: AuthorizedThreadContext,
    config: ContextualizationConfig,
) -> ContextualRewrite:
    """Rewrite from only the currently authorized Thread projection."""

    output = model.with_structured_output(ContextualRewrite).invoke(
        [
            (
                "system",
                "Rewrite the current message into a standalone question using "
                "only the supplied authorized Thread context. Do not answer, "
                "add facts, or change user constraints. Preserve standalone "
                "questions and clear topic changes. Ask for clarification when "
                "the reference is ambiguous. Return no reasoning text.",
            ),
            (
                "human",
                json.dumps(
                    {
                        "user_message": user_message,
                        "authorized_context": context.model_dump(mode="json"),
                        "contract": {
                            "prompt": config.prompt_version,
                            "schema": config.schema_version,
                        },
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ),
        ]
    )
    rewrite = ContextualRewrite.model_validate(output)
    available = {turn.turn_id for turn in context.recent_turns}
    available.update(
        turn_id
        for item in context.summary_items
        for turn_id in item.source_turn_ids
    )
    if not set(rewrite.referenced_turn_ids).issubset(available):
        raise ValueError("rewrite references unavailable Turn history")
    if rewrite.depends_on_history and not available:
        raise ValueError("rewrite cannot depend on empty history")
    return rewrite


def summarize_turns(
    model: BaseChatModel,
    previous_summary: ConversationSummary | None,
    records: tuple[AuthorizedTurnContext, ...],
    config: ContextualizationConfig,
) -> ConversationSummary:
    """Summarize only complete Turns with inspectable source IDs."""

    if not records:
        raise ValueError("summary requires complete Turns")
    output = model.with_structured_output(ConversationSummary).invoke(
        [
            (
                "system",
                "Update the structured Conversation Summary from complete "
                "Turn Records. Preserve user constraints separately from "
                "evidence-derived claims and copy citation dependencies. Do "
                "not include raw Evidence, drafts, tools, or reasoning.",
            ),
            (
                "human",
                json.dumps(
                    {
                        "previous_summary": (
                            previous_summary.model_dump(mode="json")
                            if previous_summary is not None
                            else None
                        ),
                        "turn_records": [
                            record.model_dump(mode="json") for record in records
                        ],
                        "contract": {
                            "model": config.model_id,
                            "prompt": config.summary_prompt_version,
                            "schema": config.summary_schema_version,
                        },
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ),
        ]
    )
    summary = ConversationSummary.model_validate(output)
    if summary.model_version != config.model_id:
        raise ValueError("summary model version is invalid")
    if len(summary.model_dump_json()) // 4 > config.summary_token_limit:
        raise ValueError("summary exceeds its token budget")
    allowed = {record.turn_id for record in records}
    dependencies_by_turn = {
        record.turn_id: set(record.citation_dependencies)
        for record in records
    }
    if previous_summary is not None:
        for item in previous_summary.items:
            for turn_id in item.source_turn_ids:
                allowed.add(turn_id)
                dependencies_by_turn.setdefault(turn_id, set()).update(
                    item.citation_dependencies
                )
    if summary.covered_through_turn_id != records[-1].turn_id or any(
        not set(item.source_turn_ids).issubset(allowed)
        for item in summary.items
    ):
        raise ValueError("summary coverage is invalid")
    for item in summary.items:
        allowed_dependencies = set().union(
            *(dependencies_by_turn[turn_id] for turn_id in item.source_turn_ids)
        )
        if not set(item.citation_dependencies).issubset(
            allowed_dependencies
        ):
            raise ValueError("summary citation dependencies are invalid")
    return summary
