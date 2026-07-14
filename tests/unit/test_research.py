"""L1 contracts and routes for bounded research."""

import pytest
from pydantic import ValidationError

from agentic_rag.agents.research import (
    EvidenceAssessment,
    PlanAction,
    PlanTerminalReason,
    QueryRefinement,
    ResearchAgentConfig,
    ResearchPlan,
)
from agentic_rag.corpus import KnowledgeSource
from agentic_rag.graph.nodes import (
    route_after_research_assessment,
    route_after_research_plan,
)
from agentic_rag.graph.state import ResearchCounters, ResearchGraphState
from agentic_rag.runtime import TurnExecutionBudget


def test_research_plan_rejects_authority_and_budget_fields() -> None:
    with pytest.raises(ValidationError):
        ResearchPlan.model_validate(
            {
                "action": "retrieve",
                "query": "rollback failure",
                "knowledge_sources": ["engineering-docs"],
                "principal_id": "alice",
                "result_limit": 50,
            }
        )


def test_retrieval_plan_requires_query_and_distinct_sources() -> None:
    with pytest.raises(ValidationError):
        ResearchPlan(
            action=PlanAction.RETRIEVE,
            knowledge_sources=(KnowledgeSource.ENGINEERING_DOCS,),
        )
    with pytest.raises(ValidationError):
        ResearchPlan(
            action=PlanAction.RETRIEVE,
            query="rollback failure",
            knowledge_sources=(
                KnowledgeSource.ENGINEERING_DOCS,
                KnowledgeSource.ENGINEERING_DOCS,
            ),
        )


def test_terminal_plan_requires_matching_structured_reason() -> None:
    with pytest.raises(ValidationError):
        ResearchPlan(
            action=PlanAction.CLARIFICATION,
            terminal_reason=PlanTerminalReason.GREETING,
        )

    plan = ResearchPlan(
        action=PlanAction.CLARIFICATION,
        terminal_reason=PlanTerminalReason.AMBIGUOUS_REQUEST,
    )
    assert plan.query is None
    assert plan.knowledge_sources == ()


def test_research_decisions_are_strict_and_bounded() -> None:
    with pytest.raises(ValidationError):
        EvidenceAssessment.model_validate(
            {"sufficient": True, "reasoning": "hidden rationale"}
        )
    with pytest.raises(ValidationError):
        QueryRefinement(query=" ")
    with pytest.raises(ValidationError):
        ResearchCounters(model_calls=-1)
    with pytest.raises(ValidationError):
        TurnExecutionBudget(retrieval_request_limit=3)
    assert ResearchAgentConfig().fingerprint.startswith("research_")


def test_plan_route_skips_retrieval_for_clarification() -> None:
    state = _state()
    state["plan"] = ResearchPlan(
        action=PlanAction.CLARIFICATION,
        terminal_reason=PlanTerminalReason.AMBIGUOUS_REQUEST,
    )
    state["status"] = "clarification"

    assert route_after_research_plan(state) == "__end__"


def test_assessment_route_allows_only_one_refinement_edge() -> None:
    state = _state()
    state["assessment"] = EvidenceAssessment(sufficient=False)
    state["next_node"] = "refine"

    assert route_after_research_assessment(state) == "refine"

    state["counters"] = ResearchCounters(
        model_calls=4,
        retrieval_requests=2,
        research_iterations=1,
    )
    state["status"] = "incomplete"
    state["next_node"] = "end"
    assert route_after_research_assessment(state) == "__end__"


def _state() -> ResearchGraphState:
    return {
        "question": "Why did the rollback fail?",
        "authorization_snapshot": {
            "principal_id": "alice",
            "revision": "auth_" + "0" * 64,
        },
        "counters": ResearchCounters(),
        "retrieval_results": [],
        "status": "pending",
    }
