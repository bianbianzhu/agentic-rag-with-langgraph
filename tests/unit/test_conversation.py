"""L1 Conversation Thread binding, rewrite, and commit contracts."""

from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from agentic_rag.agents.contextualization import (
    ContextualizationConfig,
    ContextualRewrite,
    rewrite_question,
    summarize_turns,
)
from agentic_rag.conversation import (
    AuthorizedThreadContext,
    AuthorizedTurnContext,
    CitationDependency,
    ConversationSummary,
    SummaryItem,
    SummaryItemKind,
    ThreadBusyError,
    ThreadPrincipalMismatchError,
    ThreadState,
    TurnOutcome,
    TurnRecord,
    TurnResumeIncompatibleError,
    begin_turn,
    compact_thread_memory,
    commit_turn,
)
from agentic_rag.corpus import KnowledgeSource
from agentic_rag.corpus.models import SourceLocator


def test_first_turn_binds_principal_and_concurrent_turn_is_rejected() -> None:
    started = begin_turn(
        ThreadState(), "alice", "turn-1", input_fingerprint="input-a"
    )
    assert started.principal_id == "alice"
    assert started.active_turn_id == "turn-1"

    with pytest.raises(ThreadBusyError):
        begin_turn(started, "alice", "turn-2")
    with pytest.raises(ThreadPrincipalMismatchError):
        begin_turn(started, "bob", "turn-1")
    with pytest.raises(TurnResumeIncompatibleError):
        begin_turn(
            started,
            "alice",
            "turn-1",
            input_fingerprint="different-input",
        )


def test_completed_turn_id_returns_existing_record_without_reexecution() -> None:
    thread = begin_turn(ThreadState(), "alice", "turn-1")
    record = _record("turn-1")
    committed = commit_turn(thread, record)

    resumed = begin_turn(committed, "alice", "turn-1")

    assert resumed == committed
    assert resumed.turn_records == (record,)


def test_terminal_commit_clears_active_turn_and_is_atomic() -> None:
    thread = begin_turn(ThreadState(), "alice", "turn-1")

    committed = commit_turn(thread, _record("turn-1"))

    assert committed.active_turn_id is None
    assert len(committed.turn_records) == 1
    with pytest.raises(ValueError, match="active Turn"):
        commit_turn(committed, _record("turn-2"))


def test_turn_memory_rejects_raw_evidence_and_drafts() -> None:
    with pytest.raises(ValidationError):
        TurnRecord.model_validate(
            {
                **_record("turn-1").model_dump(),
                "evidence_set": {"raw": "forbidden"},
                "draft_answer": "forbidden",
            }
        )

    failed_payload = _record("turn-failed").model_dump()
    failed_payload["outcome"] = TurnOutcome.FAILED
    with pytest.raises(ValidationError, match="terminal reason"):
        TurnRecord.model_validate(failed_payload)


def test_contextual_rewrite_requires_valid_history_references() -> None:
    with pytest.raises(ValidationError):
        ContextualRewrite(
            standalone_question="What happened in staging?",
            depends_on_history=False,
            referenced_turn_ids=("turn-1",),
            clarification_needed=False,
        )


def test_contextual_rewrite_can_reference_a_compacted_summary_turn() -> None:
    model = Mock()
    model.with_structured_output.return_value.invoke.return_value = (
        ContextualRewrite(
            standalone_question="Did rollback fail in staging?",
            depends_on_history=True,
            referenced_turn_ids=("turn-compacted",),
            clarification_needed=False,
        )
    )
    context = AuthorizedThreadContext(
        recent_turns=(),
        summary_items=(
            SummaryItem(
                kind=SummaryItemKind.TOPIC,
                text="Rollback was discussed.",
                source_turn_ids=("turn-compacted",),
            ),
        ),
    )

    rewrite = rewrite_question(
        model,
        "Did it fail in staging?",
        context,
        ContextualizationConfig(),
    )

    assert rewrite.referenced_turn_ids == ("turn-compacted",)
    messages = model.with_structured_output.return_value.invoke.call_args.args[0]
    assert "exactly one prior topic or event" in messages[0][1]


def test_compaction_replaces_only_an_oldest_complete_turn_prefix() -> None:
    thread = commit_turn(
        begin_turn(ThreadState(), "alice", "turn-1"), _record("turn-1")
    )
    thread = commit_turn(
        begin_turn(thread, "alice", "turn-2"), _record("turn-2")
    )
    summary = ConversationSummary(
        model_version="summary-v1",
        items=(
            SummaryItem(
                kind=SummaryItemKind.TOPIC,
                text="Rollback was discussed.",
                source_turn_ids=("turn-1",),
            ),
        ),
        covered_through_turn_id="turn-1",
    )

    compacted = compact_thread_memory(thread, summary, ("turn-1",))

    assert compacted.summary == summary
    assert [record.turn_id for record in compacted.turn_records] == ["turn-2"]
    with pytest.raises(ValueError, match="oldest Turn prefix"):
        compact_thread_memory(thread, summary, ("turn-2",))
    with pytest.raises(ValidationError):
        ContextualRewrite(
            standalone_question="What happened?",
            depends_on_history=True,
            referenced_turn_ids=(),
            clarification_needed=False,
        )


def test_summary_requires_pinned_model_and_noninvented_dependencies() -> None:
    dependency = CitationDependency(
        knowledge_source=KnowledgeSource.OPERATIONAL_RUNBOOKS,
        document_id="document-1",
        source_revision="revision-1",
        source_locator=SourceLocator(section_path=("Rollback",)),
    )
    record = AuthorizedTurnContext(
        turn_id="turn-1",
        user_message="Why did rollback fail?",
        standalone_question="Why did rollback fail?",
        assistant_message="The schema changed.",
        citation_dependencies=(dependency,),
    )
    config = ContextualizationConfig()
    invented = dependency.model_copy(update={"document_id": "invented"})
    model = Mock()
    model.with_structured_output.return_value.invoke.return_value = (
        ConversationSummary(
            model_version=config.model_id,
            items=(
                SummaryItem(
                    kind=SummaryItemKind.EVIDENCE_CLAIM,
                    text="The schema changed.",
                    source_turn_ids=("turn-1",),
                    citation_dependencies=(invented,),
                ),
            ),
            covered_through_turn_id="turn-1",
        )
    )

    with pytest.raises(ValueError, match="dependencies"):
        summarize_turns(model, None, (record,), config)

    wrong_model = Mock()
    wrong_model.with_structured_output.return_value.invoke.return_value = (
        ConversationSummary(
            model_version="unpinned-model",
            items=(),
            covered_through_turn_id="turn-1",
        )
    )
    with pytest.raises(ValueError, match="model version"):
        summarize_turns(wrong_model, None, (record,), config)


def test_summary_items_distinguish_evidence_claims_from_user_constraints() -> None:
    with pytest.raises(ValidationError, match="citation dependencies"):
        SummaryItem(
            kind=SummaryItemKind.EVIDENCE_CLAIM,
            text="The schema changed.",
            source_turn_ids=("turn-1",),
        )


def _record(turn_id: str) -> TurnRecord:
    return TurnRecord(
        turn_id=turn_id,
        user_message="Why did rollback fail?",
        standalone_question="Why did rollback fail?",
        assistant_message="The schema changed.",
        outcome=TurnOutcome.ANSWERED,
    )
