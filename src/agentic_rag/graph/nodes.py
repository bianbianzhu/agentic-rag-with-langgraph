"""Node adapters for the Reference System graph."""

from langgraph.graph import END
from langgraph.runtime import Runtime

from agentic_rag.agents.answering import (
    VerificationDecision,
    generate_answer_draft,
    verify_answer_draft,
)
from agentic_rag.agents.research import (
    EvidenceAssessment,
    PlanAction,
    QueryRefinement,
    ResearchPlan,
    assess_evidence,
    create_research_plan,
    refine_research_query,
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
from agentic_rag.corpus import KnowledgeSource
from agentic_rag.graph.state import (
    AnswerGraphState,
    CurrentTurnWork,
    GraphState,
    ResearchCounters,
    ResearchGraphState,
)
from agentic_rag.retrieval import (
    EvidenceSet,
    RetrievalRequest,
    RetrievalResult,
    RetrievalStage,
    RetrievalStatus,
    assemble_evidence_set,
    retrieve,
)
from agentic_rag.retrieval.evidence import expand_evidence_set_context
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


def route_after_research_plan(state: ResearchGraphState) -> str:
    """Enter retrieval only for a pending retrieval plan."""

    return "retrieve" if state["status"] == "pending" else END


def route_after_research_assessment(state: ResearchGraphState) -> str:
    """Take only the code-selected bounded refinement edge."""

    return "refine" if state.get("next_node") == "refine" else END


def plan_research(
    state: ResearchGraphState, runtime: Runtime[RuntimeContext]
) -> dict[str, object]:
    """Create a bounded retrieval plan or a structured terminal."""

    snapshot = AuthorizationSnapshot.model_validate(
        state["authorization_snapshot"]
    )
    if snapshot.principal_id != runtime.context.principal_id:
        return _research_terminal(
            "refused", "authorization_context_mismatch"
        )
    model = runtime.context.research_model
    if model is None:
        raise ValueError("research_model is required for the Research subgraph")
    counters, terminal = _consume_research_operation(
        state, runtime, "model_calls"
    )
    if terminal is not None:
        return terminal
    try:
        plan = create_research_plan(
            model, state["question"], runtime.context.research_config
        )
    except Exception:
        return _research_terminal("failed", "planning_failed", counters)
    if _research_deadline_exceeded(runtime.context):
        return _research_terminal(
            "incomplete", "deadline_exceeded", counters
        )

    result: dict[str, object] = {
        "plan": plan.model_dump(mode="json"),
        "counters": counters.model_dump(mode="json"),
        "failure_reason": None,
    }
    if plan.action is PlanAction.RETRIEVE:
        result.update(
            {
                "active_query": plan.query,
                "knowledge_sources": [
                    source.value for source in plan.knowledge_sources
                ],
                "status": "pending",
                "next_node": "retrieve",
            }
        )
        return result

    status, response = {
        PlanAction.DIRECT: (
            "direct",
            "Hello. Ask me about engineering docs or operational runbooks.",
        ),
        PlanAction.CLARIFICATION: (
            "clarification",
            "Please clarify which system, rollback, or environment you mean.",
        ),
        PlanAction.REFUSAL: (
            "refused",
            "I can only help with the configured engineering documentation.",
        ),
    }[plan.action]
    result.update(
        {
            "status": status,
            "response_text": response,
            "terminal_reason": str(
                _required(plan.terminal_reason, "terminal_reason")
            ),
            "next_node": "end",
        }
    )
    return result


def retrieve_research_evidence(
    state: ResearchGraphState, runtime: Runtime[RuntimeContext]
) -> dict[str, object]:
    """Execute one code-bounded authorized Retrieval Request."""

    context = runtime.context
    if (
        context.database_pool is None
        or context.embedder is None
        or context.reranker is None
        or context.retrieval_config is None
    ):
        raise ValueError("retrieval dependencies are required")
    counters, terminal = _consume_research_operations(
        state, runtime, ("retrieval_requests", "model_calls")
    )
    if terminal is not None:
        return terminal
    snapshot = AuthorizationSnapshot.model_validate(
        state["authorization_snapshot"]
    )
    query = str(_required(state.get("active_query"), "active_query"))
    sources = tuple(
        KnowledgeSource(value)
        for value in state.get("knowledge_sources", ())
    )
    request = RetrievalRequest(
        request_id=f"research-{counters.retrieval_requests}",
        query=query,
        knowledge_sources=sources,
        dense_candidate_limit=50,
        lexical_candidate_limit=50,
        result_limit=8,
    )
    result = retrieve(
        context.database_pool,
        snapshot,
        request,
        context.retrieval_config,
        context.embedder,
        context.reranker,
    )
    if (
        result.candidate_counts.reranked == 0
        and result.failed_stage is not RetrievalStage.RERANK
    ):
        counters = counters.model_copy(
            update={"model_calls": counters.model_calls - 1}
        )
    if _research_deadline_exceeded(context):
        return _research_terminal(
            "incomplete", "deadline_exceeded", counters
        )
    previous = tuple(
        RetrievalResult.model_validate(value)
        for value in state["retrieval_results"]
    )
    all_results = previous + (result,)
    evidence_set = expand_evidence_set_context(
        context.database_pool,
        snapshot,
        assemble_evidence_set(all_results, context.retrieval_config),
        context.retrieval_config,
    )
    if _research_deadline_exceeded(context):
        return _research_terminal(
            "incomplete", "deadline_exceeded", counters
        )
    update: dict[str, object] = {
        "retrieval_results": [
            value.model_dump(mode="json") for value in all_results
        ],
        "evidence_set": evidence_set.model_dump(mode="json"),
        "counters": counters.model_dump(mode="json"),
    }
    if result.status is RetrievalStatus.FAILED or not evidence_set.complete:
        update.update(
            _research_terminal(
                "failed",
                (
                    result.error_code.value
                    if result.error_code is not None
                    else "evidence_assembly_failed"
                ),
                counters,
            )
        )
    else:
        update.update(
            {"status": "pending", "next_node": "assess", "failure_reason": None}
        )
    return update


def assess_research_evidence(
    state: ResearchGraphState, runtime: Runtime[RuntimeContext]
) -> dict[str, object]:
    """Assess Evidence and select success, one refinement, or incomplete."""

    model = runtime.context.research_model
    if model is None:
        raise ValueError("research_model is required for the Research subgraph")
    counters, terminal = _consume_research_operation(
        state, runtime, "model_calls"
    )
    if terminal is not None:
        return terminal
    evidence_set = EvidenceSet.model_validate(
        _required(state.get("evidence_set"), "evidence_set")
    )
    try:
        assessment = assess_evidence(
            model,
            state["question"],
            evidence_set.evidence_items,
            runtime.context.research_config,
        )
    except Exception:
        return _research_terminal(
            "failed", "evidence_assessment_failed", counters
        )
    if _research_deadline_exceeded(runtime.context):
        return _research_terminal(
            "incomplete", "deadline_exceeded", counters
        )
    result: dict[str, object] = {
        "assessment": assessment.model_dump(mode="json"),
        "counters": counters.model_dump(mode="json"),
    }
    if assessment.sufficient:
        result.update(
            {
                "status": "evidence_ready",
                "next_node": "end",
                "failure_reason": None,
            }
        )
        return result

    budget = runtime.context.execution_budget
    can_refine = (
        counters.research_iterations < budget.research_iteration_limit
        and counters.retrieval_requests < budget.retrieval_request_limit
        and counters.model_calls < budget.model_call_limit
        and not _research_deadline_exceeded(runtime.context)
    )
    if can_refine:
        result.update({"status": "pending", "next_node": "refine"})
    else:
        result.update(
            _research_terminal(
                "incomplete", "research_budget_exhausted", counters
            )
        )
    return result


def refine_research_query_node(
    state: ResearchGraphState, runtime: Runtime[RuntimeContext]
) -> dict[str, object]:
    """Consume the one research iteration and create a replacement query."""

    model = runtime.context.research_model
    if model is None:
        raise ValueError("research_model is required for the Research subgraph")
    counters, terminal = _consume_research_operations(
        state, runtime, ("research_iterations", "model_calls")
    )
    if terminal is not None:
        return terminal
    evidence_set = EvidenceSet.model_validate(
        _required(state.get("evidence_set"), "evidence_set")
    )
    previous_query = str(
        _required(state.get("active_query"), "active_query")
    )
    try:
        refinement = refine_research_query(
            model,
            state["question"],
            previous_query,
            evidence_set.evidence_items,
            runtime.context.research_config,
        )
    except Exception:
        return _research_terminal("failed", "refinement_failed", counters)
    if _research_deadline_exceeded(runtime.context):
        return _research_terminal(
            "incomplete", "deadline_exceeded", counters
        )
    return {
        "refinement": refinement.model_dump(mode="json"),
        "active_query": refinement.query,
        "counters": counters.model_dump(mode="json"),
        "status": "pending",
        "next_node": "retrieve",
        "failure_reason": None,
    }


def route_after_research_retrieval(state: ResearchGraphState) -> str:
    return "assess" if state.get("next_node") == "assess" else END


def route_after_research_refinement(state: ResearchGraphState) -> str:
    return "retrieve" if state.get("next_node") == "retrieve" else END


def _consume_research_operation(
    state: ResearchGraphState,
    runtime: Runtime[RuntimeContext],
    operation: str,
) -> tuple[ResearchCounters, dict[str, object] | None]:
    return _consume_research_operations(state, runtime, (operation,))


def _consume_research_operations(
    state: ResearchGraphState,
    runtime: Runtime[RuntimeContext],
    operations: tuple[str, ...],
) -> tuple[ResearchCounters, dict[str, object] | None]:
    counters = ResearchCounters.model_validate(state["counters"])
    if _research_deadline_exceeded(runtime.context):
        return counters, _research_terminal(
            "incomplete", "deadline_exceeded", counters
        )
    budget = runtime.context.execution_budget
    limits = {
        "model_calls": budget.model_call_limit,
        "retrieval_requests": budget.retrieval_request_limit,
        "research_iterations": budget.research_iteration_limit,
    }
    values = counters.model_dump()
    for operation in operations:
        if values[operation] >= limits[operation]:
            return counters, _research_terminal(
                "incomplete", f"{operation}_exhausted", counters
            )
    for operation in operations:
        values[operation] += 1
    return ResearchCounters.model_validate(values), None


def _research_deadline_exceeded(context: RuntimeContext) -> bool:
    return (
        context.deadline_at is not None
        and context.clock() >= context.deadline_at
    )


def _research_terminal(
    status: str,
    reason: str,
    counters: ResearchCounters | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "status": status,
        "next_node": "end",
        "failure_reason": reason,
    }
    if status == "incomplete":
        result["response_text"] = (
            "I could not find enough authorized evidence within the research "
            "limits."
        )
    elif status == "failed":
        result["response_text"] = "I could not complete research safely."
    elif status == "refused":
        result["response_text"] = "I cannot perform that request safely."
    if counters is not None:
        result["counters"] = counters.model_dump(mode="json")
    return result
