"""Private node adapters for the top-level Turn graph."""

from langgraph.runtime import Runtime

from agentic_rag.authorization import (
    AuthorizationTransition,
    capture_authorization_snapshot,
    decide_authorization_transition,
)
from agentic_rag.citations import CitedAnswer, CitedAnswerStatus
from agentic_rag.conversation import (
    CitationDependency,
    ThreadState,
    TurnOutcome,
    TurnRecord,
    commit_turn,
)
from agentic_rag.graph._answer import answer_graph
from agentic_rag.graph._research import research_graph
from agentic_rag.graph.nodes import (
    checkpoint_turn_work,
    turn_requires_compaction,
)
from agentic_rag.graph.state import (
    AnswerGraphState,
    CurrentTurnWork,
    GraphState,
    ResearchCounters,
    ResearchGraphState,
)
from agentic_rag.retrieval import EvidenceSet
from agentic_rag.runtime import RuntimeContext


def capture_turn_authorization(
    state: GraphState, runtime: Runtime[RuntimeContext]
) -> dict[str, object]:
    """Bind all derived Turn work to one current Access Scope revision."""

    current = _current_work(state)
    thread = ThreadState.model_validate(state["thread"])
    pool = runtime.context.database_pool
    if pool is None:
        return _terminal_checkpoint(
            thread, current, "authorization_unavailable"
        )
    try:
        snapshot = capture_authorization_snapshot(
            pool, runtime.context.principal_id
        )
    except Exception:
        return _terminal_checkpoint(
            thread, current, "authorization_unavailable"
        )
    authorized = current.model_copy(
        update={
            "stage": "authorized",
            "authorization_snapshot": snapshot,
        }
    )
    return checkpoint_turn_work(thread, authorized)


def run_research_subgraph(
    state: GraphState, runtime: Runtime[RuntimeContext]
) -> dict[str, object]:
    """Run bounded research and merge only its declared counters and result."""

    current = _current_work(state)
    thread = ThreadState.model_validate(state["thread"])
    if current.authorization_snapshot is None:
        raise ValueError("research requires an Authorization Snapshot")
    if current.standalone_question is None:
        raise ValueError("research requires a Standalone Question")
    counters = ResearchCounters(
        model_calls=current.model_calls,
        retrieval_requests=current.retrieval_requests,
        research_iterations=current.research_iterations,
    )
    research_input: ResearchGraphState = {
        "question": current.standalone_question,
        "authorization_snapshot": current.authorization_snapshot,
        "counters": counters,
        "retrieval_results": [],
        "status": "pending",
    }
    try:
        result = research_graph.invoke(
            research_input,
            context=runtime.context,
        )
        consumed = ResearchCounters.model_validate(result["counters"])
    except Exception:
        return checkpoint_turn_work(
            thread,
            _authorizing_failure(current, "research_failed"),
        )

    updates: dict[str, object] = {
        "model_calls": consumed.model_calls,
        "retrieval_requests": consumed.retrieval_requests,
        "research_iterations": consumed.research_iterations,
    }
    status = result["status"]
    if status == "evidence_ready":
        evidence_set = EvidenceSet.model_validate(result["evidence_set"])
        updates.update(
            {"stage": "answering", "evidence_set": evidence_set}
        )
        return checkpoint_turn_work(
            thread, current.model_copy(update=updates)
        )

    outcome = {
        "direct": TurnOutcome.ANSWERED,
        "clarification": TurnOutcome.CLARIFICATION_REQUESTED,
        "refused": TurnOutcome.REFUSED,
        "incomplete": TurnOutcome.FAILED,
        "failed": TurnOutcome.FAILED,
    }.get(str(status), TurnOutcome.FAILED)
    internal_reason = str(
        result.get("terminal_reason")
        or result.get("failure_reason")
        or "research_failed"
    )
    reason = (
        None
        if outcome is TurnOutcome.ANSWERED
        else (
            "insufficient_evidence"
            if status == "incomplete"
            and internal_reason == "research_budget_exhausted"
            else internal_reason
        )
    )
    updates.update(
        {
            "stage": "authorizing",
            "assistant_message": str(
                result.get("response_text")
                or "I could not complete research safely."
            ),
            "outcome": outcome,
            "terminal_reason": reason,
        }
    )
    return checkpoint_turn_work(thread, current.model_copy(update=updates))


def run_answer_subgraph(
    state: GraphState, runtime: Runtime[RuntimeContext]
) -> dict[str, object]:
    """Run the buffered Answer subgraph within the remaining Turn budget."""

    current = _current_work(state)
    thread = ThreadState.model_validate(state["thread"])
    if current.authorization_snapshot is None or current.evidence_set is None:
        raise ValueError("answer requires authorized Evidence")
    if current.standalone_question is None:
        raise ValueError("answer requires a Standalone Question")
    answer_input: AnswerGraphState = {
        "question": current.standalone_question,
        "evidence_set": current.evidence_set,
        "authorization_snapshot": current.authorization_snapshot,
        "model_calls": current.model_calls,
        "repair_count": current.answer_repairs,
        "status": "pending",
    }
    try:
        result = answer_graph.invoke(
            answer_input,
            context=runtime.context,
        )
        answer = CitedAnswer.model_validate(result["cited_answer"])
        model_calls = int(result["model_calls"])
        repairs = int(result["repair_count"])
    except Exception:
        return checkpoint_turn_work(
            thread,
            _authorizing_failure(current, "answer_failed"),
        )

    outcome, reason = {
        CitedAnswerStatus.FACTUAL: (TurnOutcome.ANSWERED, None),
        CitedAnswerStatus.REFUSAL: (
            TurnOutcome.REFUSED,
            str(answer.refusal_reason),
        ),
        CitedAnswerStatus.INSUFFICIENT: (
            TurnOutcome.FAILED,
            str(answer.refusal_reason),
        ),
        CitedAnswerStatus.INCOMPLETE: (
            TurnOutcome.FAILED,
            str(result.get("failure_reason") or answer.refusal_reason),
        ),
    }[answer.status]
    authorizing = current.model_copy(
        update={
            "stage": "authorizing",
            "model_calls": model_calls,
            "answer_repairs": repairs,
            "cited_answer": answer,
            "assistant_message": answer.assistant_message,
            "outcome": outcome,
            "terminal_reason": reason,
        }
    )
    return checkpoint_turn_work(thread, authorizing)


def authorize_and_commit_turn(
    state: GraphState, runtime: Runtime[RuntimeContext]
) -> dict[str, object]:
    """Commit unchanged work, restart once, or fail closed on a second change."""

    current = _current_work(state)
    thread = ThreadState.model_validate(state["thread"])
    original = current.authorization_snapshot
    pool = runtime.context.database_pool
    rollback_thread = current.pre_compaction_thread or thread
    if original is None or pool is None:
        return _commit_failure(
            rollback_thread, current, "authorization_unavailable"
        )
    try:
        latest = capture_authorization_snapshot(
            pool, runtime.context.principal_id
        )
        transition = decide_authorization_transition(
            original, latest, current.authorization_restarts
        )
    except Exception:
        return _commit_failure(
            rollback_thread, current, "authorization_context_mismatch"
        )

    if transition is AuthorizationTransition.RESTART:
        return checkpoint_turn_work(
            rollback_thread, current.restart_after_authorization_change()
        )
    if transition is AuthorizationTransition.FAIL:
        return _commit_failure(
            rollback_thread, current, "authorization_changed_twice"
        )
    try:
        dependencies = _citation_dependencies(current)
    except ValueError:
        return _commit_failure(
            thread, current, "citation_dependency_mismatch"
        )
    record = TurnRecord(
        turn_id=current.turn_id,
        user_message=current.user_message,
        standalone_question=str(current.standalone_question),
        assistant_message=str(current.assistant_message),
        outcome=current.outcome or TurnOutcome.FAILED,
        terminal_reason=current.terminal_reason,
        citation_dependencies=dependencies,
    )
    return _committed_state(thread, record)


def route_after_turn_prepare(state: GraphState) -> str:
    raw_current = state["current_turn"]
    if raw_current is None:
        return "end"
    current = CurrentTurnWork.model_validate(raw_current)
    return (
        "complete_turn"
        if current.stage == "terminal"
        else "capture_turn_authorization"
    )


def route_after_contextualization(state: GraphState) -> str:
    current = _current_work(state)
    if current.stage == "terminal":
        return "complete_turn"
    if current.stage == "authorizing":
        return "authorize_and_commit_turn"
    return "run_research_subgraph"


def route_after_authorization_capture(state: GraphState) -> str:
    current = _current_work(state)
    if current.stage == "terminal":
        return "complete_turn"
    if turn_requires_compaction(state):
        return "compact_context"
    return "contextualize_turn"


def route_after_research(state: GraphState) -> str:
    return (
        "run_answer_subgraph"
        if _current_work(state).stage == "answering"
        else "authorize_and_commit_turn"
    )


def route_after_final_authorization(state: GraphState) -> str:
    raw_current = state["current_turn"]
    if raw_current is None:
        return "end"
    current = CurrentTurnWork.model_validate(raw_current)
    if current.stage != "prepared":
        raise ValueError("final authorization produced an invalid transition")
    return "prepare_turn"


def _current_work(state: GraphState) -> CurrentTurnWork:
    raw = state["current_turn"]
    if raw is None:
        raise ValueError("current_turn is required")
    return CurrentTurnWork.model_validate(raw)


def _terminal_checkpoint(
    thread: ThreadState, current: CurrentTurnWork, reason: str
) -> dict[str, object]:
    terminal = current.model_copy(
        update={
            "stage": "terminal",
            "standalone_question": current.standalone_question
            or current.user_message,
            "assistant_message": "I could not complete this request safely.",
            "outcome": TurnOutcome.FAILED,
            "terminal_reason": reason,
        }
    )
    return checkpoint_turn_work(thread, terminal)


def _authorizing_failure(
    current: CurrentTurnWork, reason: str
) -> CurrentTurnWork:
    return current.model_copy(
        update={
            "stage": "authorizing",
            "evidence_set": None,
            "cited_answer": None,
            "assistant_message": "I could not complete this request safely.",
            "outcome": TurnOutcome.FAILED,
            "terminal_reason": reason,
        }
    )


def _citation_dependencies(
    current: CurrentTurnWork,
) -> tuple[CitationDependency, ...]:
    answer = current.cited_answer
    evidence_set = current.evidence_set
    if answer is None or answer.status is not CitedAnswerStatus.FACTUAL:
        return ()
    if evidence_set is None:
        raise ValueError("factual answer has no Evidence Set")
    by_chunk = {item.chunk_id: item for item in evidence_set.evidence_items}
    dependencies: list[CitationDependency] = []
    for citation in answer.citations:
        item = by_chunk.get(citation.chunk_id)
        if item is None or (
            item.document_id,
            item.source_revision,
            item.source_locator,
        ) != (
            citation.document_id,
            citation.source_revision,
            citation.source_locator,
        ):
            raise ValueError("citation does not match final Evidence")
        dependencies.append(
            CitationDependency(
                knowledge_source=item.knowledge_source,
                document_id=item.document_id,
                source_revision=item.source_revision,
                source_locator=item.source_locator,
            )
        )
    return tuple(dependencies)


def _commit_failure(
    thread: ThreadState, current: CurrentTurnWork, reason: str
) -> dict[str, object]:
    record = TurnRecord(
        turn_id=current.turn_id,
        user_message=current.user_message,
        standalone_question=current.standalone_question
        or current.user_message,
        assistant_message="I could not complete this request safely.",
        outcome=TurnOutcome.FAILED,
        terminal_reason=reason,
    )
    return _committed_state(thread, record)


def _committed_state(
    thread: ThreadState, record: TurnRecord
) -> dict[str, object]:
    completed = commit_turn(thread, record)
    return {
        "thread": completed.model_dump(mode="json"),
        "current_turn": None,
    }
