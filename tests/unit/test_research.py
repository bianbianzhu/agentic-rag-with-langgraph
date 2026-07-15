"""L1 contracts and routes for bounded research."""

import json
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from agentic_rag.agents.research import (
    EvidenceAssessment,
    PlanAction,
    PlanTerminalReason,
    QueryRefinement,
    ResearchAgentConfig,
    ResearchPlan,
    assess_evidence,
    create_research_plan,
    refine_research_query,
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


def test_live_plan_prompt_declares_action_fields_and_source_scope() -> None:
    model = Mock()
    model.with_structured_output.return_value.invoke.return_value = ResearchPlan(
        action=PlanAction.RETRIEVE,
        query="settlement impact",
        knowledge_sources=(KnowledgeSource.OPERATIONAL_RUNBOOKS,),
    )

    create_research_plan(
        model,
        "What was the confidential settlement impact?",
        ResearchAgentConfig(),
    )

    messages = model.with_structured_output.return_value.invoke.call_args.args[0]
    system_prompt = messages[0][1]
    assert "For retrieve:" in system_prompt
    assert "For direct, clarification, or refusal:" in system_prompt
    assert "settlement" in system_prompt
    assert "merchants, amount, delay, and duration" in system_prompt
    assert "explicit first search phrase" in system_prompt
    assert "operational-runbooks" in system_prompt
    assert "settlement impact or status" in system_prompt


def test_evidence_assessment_receives_the_active_search_stage() -> None:
    model = Mock()
    model.with_structured_output.return_value.invoke.return_value = (
        EvidenceAssessment(sufficient=False)
    )

    assess_evidence(
        model,
        "First search for the exact phrase 'empty legacy query'.",
        "empty legacy query",
        (),
        ResearchAgentConfig(),
    )

    messages = model.with_structured_output.return_value.invoke.call_args.args[0]
    system_prompt = messages[0][1]
    payload = json.loads(messages[1][1])
    assert "ordered first search" in system_prompt
    assert "public incident summary" in system_prompt
    assert payload["active_query"] == "empty legacy query"


def test_final_evidence_confirmation_uses_a_stricter_sufficiency_prompt() -> None:
    model = Mock()
    model.with_structured_output.return_value.invoke.return_value = (
        EvidenceAssessment(sufficient=True)
    )

    with pytest.raises(ValueError, match="empty Evidence"):
        assess_evidence(
            model,
            "Which worker version was incompatible?",
            "legacy worker incompatible schema v2",
            (),
            ResearchAgentConfig(),
            confirmation=True,
        )

    messages = model.with_structured_output.return_value.invoke.call_args.args[0]
    assert "final confirmation" in messages[0][1]


def test_plan_repair_requests_exactly_one_structured_object() -> None:
    model = Mock()
    model.with_structured_output.return_value.invoke.return_value = ResearchPlan(
        action=PlanAction.RETRIEVE,
        query="settlement impact",
        knowledge_sources=(KnowledgeSource.OPERATIONAL_RUNBOOKS,),
    )

    create_research_plan(
        model,
        "What was the settlement impact?",
        ResearchAgentConfig(),
        repair=True,
    )

    messages = model.with_structured_output.return_value.invoke.call_args.args[0]
    payload = json.loads(messages[1][1])
    assert "exactly one JSON object" in messages[0][1]
    assert payload["repair_invalid_output"] is True


def test_refinement_drops_a_failed_ordered_exact_phrase() -> None:
    model = Mock()
    model.with_structured_output.return_value.invoke.return_value = QueryRefinement(
        query="legacy worker incompatible schema v2"
    )

    refine_research_query(
        model,
        "First search for 'empty legacy query', then find the legacy worker.",
        '"empty legacy query" legacy worker',
        (),
        ResearchAgentConfig(),
    )

    messages = model.with_structured_output.return_value.invoke.call_args.args[0]
    assert "omit that failed exact phrase" in messages[0][1]


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
