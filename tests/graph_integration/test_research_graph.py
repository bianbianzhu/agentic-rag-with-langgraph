"""L3 verification for the compiled bounded Research subgraph."""

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from psycopg_pool import ConnectionPool
from pydantic import Field

from agentic_rag.agents.research import (
    EvidenceAssessment,
    PlanAction,
    PlanTerminalReason,
    QueryRefinement,
    ResearchPlan,
)
from agentic_rag.authorization import capture_authorization_snapshot
from agentic_rag.corpus import KnowledgeSource
from agentic_rag.database import apply_migrations, open_database_pool
from agentic_rag.graph._research import research_graph
from agentic_rag.graph.state import ResearchCounters, ResearchGraphState
from agentic_rag.retrieval import (
    RerankCandidate,
    RerankItem,
    RerankOutput,
    RetrievalConfig,
)
from agentic_rag.runtime import RuntimeContext, TurnExecutionBudget
from tests.support.reference_fixture import FixtureEmbedder, load_reference_snapshot


REPOSITORY_ROOT = Path(__file__).parents[2]
DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://agentic_rag@127.0.0.1:55432/agentic_rag",
)
RETRIEVAL_DATABASE_URL = os.environ.get(
    "TEST_RETRIEVAL_DATABASE_URL",
    "postgresql://agentic_rag_retrieval@127.0.0.1:55432/agentic_rag",
)


class DeterministicResearchModel(BaseChatModel):
    responses: list[Any] = Field(default_factory=list)
    requested_schemas: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "deterministic-research"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: object | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise AssertionError("structured output is required")

    def with_structured_output(
        self,
        schema: dict[str, Any] | type,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[Any, Any]:
        assert include_raw is False
        self.requested_schemas.append(
            schema.__name__ if isinstance(schema, type) else "json-schema"
        )

        def next_response(_: object) -> Any:
            if not self.responses:
                raise AssertionError("unexpected model call")
            response = self.responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

        return RunnableLambda(next_response)


@pytest.fixture(scope="module")
def database_pool() -> Iterator[ConnectionPool]:
    pool = open_database_pool(DATABASE_URL)
    apply_migrations(pool, REPOSITORY_ROOT / "migrations")
    yield pool
    pool.close()


@pytest.fixture(scope="module")
def retrieval_pool(
    database_pool: ConnectionPool,
) -> Iterator[ConnectionPool]:
    pool = open_database_pool(RETRIEVAL_DATABASE_URL)
    yield pool
    pool.close()


@pytest.fixture(autouse=True)
def empty_research_state(database_pool: ConnectionPool) -> None:
    with database_pool.connection() as connection:
        connection.execute(
            """
            TRUNCATE access_grants, principal_group_memberships, groups,
                principals, sync_reports, indexed_chunks, source_documents,
                knowledge_sources CASCADE
            """
        )


@pytest.fixture
def research_input(
    database_pool: ConnectionPool,
    retrieval_pool: ConnectionPool,
) -> tuple[ResearchGraphState, RuntimeContext]:
    load_reference_snapshot(database_pool)
    snapshot = capture_authorization_snapshot(retrieval_pool, "bob")
    state: ResearchGraphState = {
        "question": "Why did the payments rollback fail?",
        "authorization_snapshot": snapshot.model_dump(mode="json"),
        "counters": ResearchCounters(),
        "retrieval_results": [],
        "status": "pending",
    }
    return state, RuntimeContext(
        principal_id="bob",
        database_pool=retrieval_pool,
        embedder=FixtureEmbedder(),
        reranker=scenario_reranker,
        retrieval_config=RetrievalConfig(
            embedding_model="deterministic-test-v1"
        ),
    )


def test_research_graph_returns_real_authorized_evidence(
    research_input: tuple[ResearchGraphState, RuntimeContext],
) -> None:
    state, base_context = research_input
    model = DeterministicResearchModel(
        responses=[_plan("rollback failure"), EvidenceAssessment(sufficient=True)]
    )

    result = _invoke(state, base_context, model)

    assert result["status"] == "evidence_ready"
    assert result["counters"] == {
        "model_calls": 3,
        "retrieval_requests": 1,
        "research_iterations": 0,
    }
    assert result["evidence_set"]["complete"] is True
    assert result["evidence_set"]["evidence_items"]


def test_research_graph_retries_one_malformed_plan_within_budget(
    research_input: tuple[ResearchGraphState, RuntimeContext],
) -> None:
    state, base_context = research_input
    model = DeterministicResearchModel(
        responses=[
            ValueError("malformed structured output"),
            _plan("rollback failure"),
            EvidenceAssessment(sufficient=True),
        ]
    )

    result = _invoke(state, base_context, model)

    assert result["status"] == "evidence_ready"
    assert result["counters"] == {
        "model_calls": 4,
        "retrieval_requests": 1,
        "research_iterations": 0,
    }
    assert model.requested_schemas[:2] == ["ResearchPlan", "ResearchPlan"]


def test_research_graph_refines_once_then_succeeds(
    research_input: tuple[ResearchGraphState, RuntimeContext],
) -> None:
    state, base_context = research_input
    model = DeterministicResearchModel(
        responses=[
            _plan("empty first query"),
            EvidenceAssessment(sufficient=False),
            QueryRefinement(query="rollback failure"),
            EvidenceAssessment(sufficient=True),
        ]
    )

    result = _invoke(state, base_context, model)

    assert result["status"] == "evidence_ready"
    assert result["counters"] == {
        "model_calls": 6,
        "retrieval_requests": 2,
        "research_iterations": 1,
    }
    assert [item["status"] for item in result["retrieval_results"]] == [
        "no_evidence",
        "completed",
    ]
    assert model.requested_schemas == [
        "ResearchPlan",
        "EvidenceAssessment",
        "QueryRefinement",
        "EvidenceAssessment",
    ]


def test_repeated_empty_retrieval_stops_at_research_budget(
    research_input: tuple[ResearchGraphState, RuntimeContext],
) -> None:
    state, base_context = research_input
    model = DeterministicResearchModel(
        responses=[
            _plan("empty first query"),
            EvidenceAssessment(sufficient=False),
            QueryRefinement(query="empty second query"),
            EvidenceAssessment(sufficient=False),
        ]
    )

    result = _invoke(state, base_context, model)

    assert result["status"] == "incomplete"
    assert result["failure_reason"] == "research_budget_exhausted"
    assert result["counters"] == {
        "model_calls": 6,
        "retrieval_requests": 2,
        "research_iterations": 1,
    }
    assert result["evidence_set"]["evidence_items"] == []


def test_model_cannot_mark_empty_evidence_sufficient(
    research_input: tuple[ResearchGraphState, RuntimeContext],
) -> None:
    state, base_context = research_input
    model = DeterministicResearchModel(
        responses=[
            _plan("empty first query"),
            EvidenceAssessment(sufficient=True),
        ]
    )

    result = _invoke(state, base_context, model)

    assert result["status"] == "failed"
    assert result["failure_reason"] == "evidence_assessment_failed"
    assert result["counters"] == {
        "model_calls": 3,
        "retrieval_requests": 1,
        "research_iterations": 0,
    }


def test_refinement_must_change_the_query_before_second_retrieval(
    research_input: tuple[ResearchGraphState, RuntimeContext],
) -> None:
    state, base_context = research_input
    model = DeterministicResearchModel(
        responses=[
            _plan("empty first query"),
            EvidenceAssessment(sufficient=False),
            QueryRefinement(query="empty first query"),
        ]
    )

    result = _invoke(state, base_context, model)

    assert result["status"] == "failed"
    assert result["failure_reason"] == "refinement_failed"
    assert result["counters"] == {
        "model_calls": 4,
        "retrieval_requests": 1,
        "research_iterations": 1,
    }


def test_retrieval_failure_is_an_explicit_terminal(
    research_input: tuple[ResearchGraphState, RuntimeContext],
) -> None:
    state, base_context = research_input
    model = DeterministicResearchModel(responses=[_plan("fail retrieval")])

    result = _invoke(state, base_context, model)

    assert result["status"] == "failed"
    assert result["failure_reason"] == "rerank_failed"
    assert result["counters"] == {
        "model_calls": 2,
        "retrieval_requests": 1,
        "research_iterations": 0,
    }
    assert model.requested_schemas == ["ResearchPlan"]


def test_exceptional_retrieval_preserves_reserved_turn_counters(
    research_input: tuple[ResearchGraphState, RuntimeContext],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, base_context = research_input
    model = DeterministicResearchModel(responses=[_plan("rollback failure")])

    def fail_retrieval(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected retrieval failure")

    monkeypatch.setattr(
        "agentic_rag.graph.nodes.retrieve", fail_retrieval
    )
    result = _invoke(state, base_context, model)

    assert result["status"] == "failed"
    assert result["failure_reason"] == "retrieval_failed"
    assert result["counters"] == {
        "model_calls": 2,
        "retrieval_requests": 1,
        "research_iterations": 0,
    }


def test_model_budget_is_reserved_before_reranker_call(
    research_input: tuple[ResearchGraphState, RuntimeContext],
) -> None:
    state, base_context = research_input
    model = DeterministicResearchModel(responses=[_plan("rollback failure")])

    def forbidden_reranker(
        _query: str, _candidates: tuple[RerankCandidate, ...]
    ) -> RerankOutput:
        raise AssertionError("reranker must not run past the model budget")

    context = _context(
        base_context,
        model,
        execution_budget=TurnExecutionBudget(model_call_limit=1),
        reranker=forbidden_reranker,
    )

    result = research_graph.invoke(state, context=context)

    assert result["status"] == "incomplete"
    assert result["failure_reason"] == "model_calls_exhausted"
    assert result["counters"] == {
        "model_calls": 1,
        "retrieval_requests": 0,
        "research_iterations": 0,
    }
    assert result["retrieval_results"] == []


def test_deadline_exhaustion_stops_before_model_or_retrieval(
    research_input: tuple[ResearchGraphState, RuntimeContext],
) -> None:
    state, base_context = research_input
    model = DeterministicResearchModel(responses=[_plan("rollback failure")])
    context = _context(base_context, model, clock=lambda: 2.0, deadline_at=1.0)

    result = research_graph.invoke(state, context=context)

    assert result["status"] == "incomplete"
    assert result["failure_reason"] == "deadline_exceeded"
    assert result["counters"] == ResearchCounters().model_dump()
    assert model.requested_schemas == []


def test_model_call_crossing_deadline_cannot_publish_success(
    research_input: tuple[ResearchGraphState, RuntimeContext],
) -> None:
    state, base_context = research_input
    model = DeterministicResearchModel(
        responses=[
            ResearchPlan(
                action=PlanAction.DIRECT,
                terminal_reason=PlanTerminalReason.GREETING,
            )
        ]
    )
    readings = iter((0.0, 0.0, 2.0))
    context = _context(
        base_context,
        model,
        clock=lambda: next(readings),
        deadline_at=1.0,
    )

    result = research_graph.invoke(state, context=context)

    assert result["status"] == "incomplete"
    assert result["failure_reason"] == "deadline_exceeded"
    assert result["counters"]["model_calls"] == 1
    assert result["counters"]["retrieval_requests"] == 0


def test_clarification_skips_retrieval(
    research_input: tuple[ResearchGraphState, RuntimeContext],
) -> None:
    state, base_context = research_input
    model = DeterministicResearchModel(
        responses=[
            ResearchPlan(
                action=PlanAction.CLARIFICATION,
                terminal_reason=PlanTerminalReason.AMBIGUOUS_REQUEST,
            )
        ]
    )

    result = _invoke(state, base_context, model)

    assert result["status"] == "clarification"
    assert result["counters"]["retrieval_requests"] == 0
    assert result["retrieval_results"] == []


def scenario_reranker(
    query: str, candidates: tuple[RerankCandidate, ...]
) -> RerankOutput:
    if query == "fail retrieval":
        return RerankOutput(
            items=(RerankItem(chunk_id="unknown-chunk", score=1),)
        )
    score = 0.4 if query.startswith("empty") else 1.0
    return RerankOutput(
        items=tuple(
            RerankItem(
                chunk_id=candidate.chunk_id,
                score=score - index / (10 * len(candidates)),
            )
            for index, candidate in enumerate(candidates)
        )
    )


def _plan(query: str) -> ResearchPlan:
    return ResearchPlan(
        action=PlanAction.RETRIEVE,
        query=query,
        knowledge_sources=(KnowledgeSource.ENGINEERING_DOCS,),
    )


def _invoke(
    state: ResearchGraphState,
    base_context: RuntimeContext,
    model: DeterministicResearchModel,
):
    return research_graph.invoke(state, context=_context(base_context, model))


def _context(
    base: RuntimeContext,
    model: DeterministicResearchModel,
    **overrides: Any,
) -> RuntimeContext:
    values = {
        "principal_id": base.principal_id,
        "database_pool": base.database_pool,
        "research_model": model,
        "embedder": base.embedder,
        "reranker": base.reranker,
        "retrieval_config": base.retrieval_config,
        "execution_budget": base.execution_budget,
        "deadline_at": base.deadline_at,
    }
    values.update(overrides)
    return RuntimeContext(**values)
