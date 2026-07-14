"""Node adapters for the Reference System graph."""

from langgraph.runtime import Runtime

from agentic_rag.agents.answering import (
    VerificationDecision,
    generate_answer_draft,
    verify_answer_draft,
)
from agentic_rag.authorization import AuthorizationSnapshot
from agentic_rag.citations import (
    CitationDraft,
    CitationHydrationError,
    CitationMapping,
    CitationValidation,
    hydrate_citation_mappings,
    incomplete_answer,
    render_cited_answer,
    validate_citation_draft,
)
from agentic_rag.conversation import ThreadState
from agentic_rag.graph.state import AnswerGraphState, CurrentTurnWork, GraphState
from agentic_rag.retrieval import EvidenceSet
from agentic_rag.runtime import RuntimeContext


def complete_turn(
    state: GraphState, runtime: Runtime[RuntimeContext]
) -> dict[str, dict[str, object]]:
    """Complete the deterministic Chapter 01 Turn."""

    raw_current_turn = state["current_turn"]
    if raw_current_turn is None:
        raise ValueError("current_turn is required")

    thread = ThreadState.model_validate(state["thread"])
    current_turn = CurrentTurnWork.model_validate(raw_current_turn)

    return {
        "thread": thread.model_copy(
            update={"completed_turns": thread.completed_turns + 1}
        ).model_dump(mode="json"),
        "current_turn": current_turn.model_copy(
            update={
                "assistant_message": "The Reference System development loop is ready.",
                "status": "answered",
            }
        ).model_dump(mode="json"),
    }


def hydrate_answer_citations(
    state: AnswerGraphState, runtime: Runtime[RuntimeContext]
) -> dict[str, object]:
    """Recheck authority before assigning code-owned citation mappings."""

    evidence_set = EvidenceSet.model_validate(state["evidence_set"])
    if not evidence_set.complete:
        return {"failure_reason": "evidence_set_incomplete"}
    pool = runtime.context.database_pool
    if pool is None:
        raise ValueError("database_pool is required for the Answer subgraph")
    snapshot = AuthorizationSnapshot.model_validate(
        state["authorization_snapshot"]
    )
    if snapshot.principal_id != runtime.context.principal_id:
        return {"failure_reason": "authorization_context_mismatch"}
    try:
        mappings = hydrate_citation_mappings(
            pool, snapshot, evidence_set.evidence_items
        )
    except CitationHydrationError:
        return {"failure_reason": "citation_hydration_failed"}
    return {
        "citation_mappings": [
            mapping.model_dump(mode="json") for mapping in mappings
        ],
        "failure_reason": None,
    }


def generate_answer(
    state: AnswerGraphState, runtime: Runtime[RuntimeContext]
) -> dict[str, object]:
    """Generate the first buffered structured answer draft."""

    return _generate_or_repair(state, runtime, repair_feedback=None)


def repair_answer(
    state: AnswerGraphState, runtime: Runtime[RuntimeContext]
) -> dict[str, object]:
    """Use the one allowed repair against the unchanged Evidence Set."""

    feedback = "structured_output_invalid"
    raw_validation = state.get("validation")
    raw_verification = state.get("verification")
    if raw_verification is not None:
        verification = VerificationDecision.model_validate(raw_verification)
        feedback = "unsupported_claim_indexes:" + ",".join(
            str(index) for index in verification.unsupported_claim_indexes
        )
    elif raw_validation is not None:
        validation = CitationValidation.model_validate(raw_validation)
        feedback = "citation_errors:" + ",".join(
            error.value for error in validation.errors
        )
    result = _generate_or_repair(
        state, runtime, repair_feedback=feedback
    )
    result["repair_count"] = state["repair_count"] + 1
    return result


def validate_answer_citations(
    state: AnswerGraphState,
) -> dict[str, object]:
    """Run deterministic citation integrity before semantic verification."""

    evidence_set = EvidenceSet.model_validate(state["evidence_set"])
    draft = CitationDraft.model_validate(
        _required(state.get("draft"), "draft")
    )
    mappings = _mappings(state)
    validation = validate_citation_draft(
        draft, evidence_set.evidence_items, mappings
    )
    return {"validation": validation.model_dump(mode="json")}


def verify_answer(
    state: AnswerGraphState, runtime: Runtime[RuntimeContext]
) -> dict[str, object]:
    """Judge whether cited Evidence semantically supports every claim."""

    model = runtime.context.answer_model
    if model is None:
        raise ValueError("answer_model is required for the Answer subgraph")
    evidence_set = EvidenceSet.model_validate(state["evidence_set"])
    draft = CitationDraft.model_validate(
        _required(state.get("draft"), "draft")
    )
    try:
        decision = verify_answer_draft(
            model, draft, evidence_set.evidence_items, _mappings(state)
        )
    except Exception:
        return {
            "verification": None,
            "failure_reason": "verification_failed",
        }
    return {
        "verification": decision.model_dump(mode="json"),
        "failure_reason": None,
    }


def finalize_answer(
    state: AnswerGraphState, runtime: Runtime[RuntimeContext]
) -> dict[str, object]:
    """Move only the validated and verified answer into user projection."""

    evidence_set = EvidenceSet.model_validate(state["evidence_set"])
    snapshot = AuthorizationSnapshot.model_validate(
        state["authorization_snapshot"]
    )
    pool = runtime.context.database_pool
    if pool is None:
        raise ValueError("database_pool is required for the Answer subgraph")
    try:
        current_mappings = hydrate_citation_mappings(
            pool, snapshot, evidence_set.evidence_items
        )
    except CitationHydrationError:
        return _incomplete_result("authorization_changed_before_release")
    if current_mappings != _mappings(state):
        return _incomplete_result("citation_mapping_changed_before_release")

    draft = CitationDraft.model_validate(
        _required(state.get("draft"), "draft")
    )
    validation = CitationValidation.model_validate(
        _required(state.get("validation"), "validation")
    )
    verification = VerificationDecision.model_validate(
        _required(state.get("verification"), "verification")
    )
    answer = render_cited_answer(
        draft,
        _mappings(state),
        validation,
        semantically_supported=verification.supported,
    )
    return {
        "cited_answer": answer.model_dump(mode="json"),
        "status": "answered",
        "failure_reason": None,
    }


def finish_incomplete_answer(state: AnswerGraphState) -> dict[str, object]:
    """Return a safe terminal without exposing any buffered draft text."""

    return _incomplete_result(state.get("failure_reason"))


def _incomplete_result(failure_reason: str | None) -> dict[str, object]:
    answer = incomplete_answer()
    return {
        "cited_answer": answer.model_dump(mode="json"),
        "status": "incomplete",
        "failure_reason": failure_reason,
    }


def route_after_hydration(state: AnswerGraphState) -> str:
    return "finish_incomplete" if state.get("failure_reason") else "generate"


def route_after_generation(state: AnswerGraphState) -> str:
    if state.get("draft") is not None:
        return "validate"
    return "repair" if state["repair_count"] < 1 else "finish_incomplete"


def route_after_validation(state: AnswerGraphState) -> str:
    validation = CitationValidation.model_validate(
        _required(state.get("validation"), "validation")
    )
    if validation.valid:
        return "verify"
    return "repair" if state["repair_count"] < 1 else "finish_incomplete"


def route_after_verification(state: AnswerGraphState) -> str:
    raw_verification = state.get("verification")
    if raw_verification is None:
        return "finish_incomplete"
    verification = VerificationDecision.model_validate(raw_verification)
    if verification.supported:
        return "finalize"
    return "repair" if state["repair_count"] < 1 else "finish_incomplete"


def route_after_repair(state: AnswerGraphState) -> str:
    return "validate" if state.get("draft") is not None else "finish_incomplete"


def _generate_or_repair(
    state: AnswerGraphState,
    runtime: Runtime[RuntimeContext],
    *,
    repair_feedback: str | None,
) -> dict[str, object]:
    model = runtime.context.answer_model
    if model is None:
        raise ValueError("answer_model is required for the Answer subgraph")
    evidence_set = EvidenceSet.model_validate(state["evidence_set"])
    try:
        draft = generate_answer_draft(
            model,
            state["question"],
            evidence_set.evidence_items,
            _mappings(state),
            repair_feedback=repair_feedback,
        )
    except Exception:
        return {
            "draft": None,
            "validation": None,
            "verification": None,
            "failure_reason": "generation_failed",
        }
    return {
        "draft": draft.model_dump(mode="json"),
        "validation": None,
        "verification": None,
        "failure_reason": None,
    }


def _mappings(state: AnswerGraphState) -> tuple[CitationMapping, ...]:
    return tuple(
        CitationMapping.model_validate(mapping)
        for mapping in state.get("citation_mappings", ())
    )


def _required(value: object | None, key: str) -> object:
    if value is None:
        raise ValueError(f"{key} is required")
    return value
