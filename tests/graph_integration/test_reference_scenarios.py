"""L3 canonical adversarial and failure scenarios through the compiled graph."""

import logging
import os
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from psycopg_pool import ConnectionPool
from pydantic import Field

from agentic_rag.agents.answering import VerificationDecision
from agentic_rag.agents.contextualization import ContextualRewrite
from agentic_rag.agents.research import (
    EvidenceAssessment,
    PlanAction,
    PlanTerminalReason,
    QueryRefinement,
    ResearchPlan,
)
from agentic_rag.authorization import (
    capture_authorization_snapshot,
    source_document_is_authorized,
)
from agentic_rag.citations import (
    CitationDraft,
    DraftClaim,
    DraftDisposition,
)
from agentic_rag.conversation import (
    ThreadPrincipalMismatchError,
    ThreadState,
    TurnOutcome,
)
from agentic_rag.corpus import KnowledgeSource
from agentic_rag.database import apply_migrations, open_database_pool
from agentic_rag.graph._turn import build_turn_graph
from agentic_rag.graph.state import CurrentTurnWork, GraphState
from agentic_rag.retrieval import (
    RerankCandidate,
    RerankItem,
    RerankOutput,
    RetrievalConfig,
)
from agentic_rag.runtime import RuntimeContext
from tests.support.reference_fixture import (
    FixtureEmbedder,
    load_reference_scenario,
    resolve_evidence_aliases,
)
from tests.support.scenarios import (
    ReferenceScenario,
    ScenarioTurn,
    load_reference_scenarios,
)


REPOSITORY_ROOT = Path(__file__).parents[2]
DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://agentic_rag@127.0.0.1:55432/agentic_rag",
)
RETRIEVAL_DATABASE_URL = os.environ.get(
    "TEST_RETRIEVAL_DATABASE_URL",
    "postgresql://agentic_rag_retrieval@127.0.0.1:55432/agentic_rag",
)
SCENARIOS = load_reference_scenarios()


class ScriptedModel(BaseChatModel):
    responses: list[Any] = Field(default_factory=list)
    requested_schemas: list[str] = Field(default_factory=list)
    structured_inputs: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "reference-scenario-script"

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

        def next_response(value: object) -> Any:
            self.structured_inputs.append(value)
            if not self.responses:
                raise AssertionError("unexpected scenario model call")
            response = self.responses.pop(0)
            return response() if callable(response) else response

        return RunnableLambda(next_response)


@dataclass(frozen=True)
class ScenarioScript:
    research_model: ScriptedModel
    answer_model: ScriptedModel
    contextualization_model: ScriptedModel
    reranker: Callable[[str, tuple[RerankCandidate, ...]], RerankOutput]


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
def empty_scenario_state(database_pool: ConnectionPool) -> None:
    with database_pool.connection() as connection:
        connection.execute(
            """
            TRUNCATE access_grants, principal_group_memberships, groups,
                principals, sync_reports, indexed_chunks, source_documents,
                knowledge_sources CASCADE
            """
        )


def test_canonical_manifest_set_is_thirteen_scenarios_and_fifteen_turns() -> None:
    assert len(SCENARIOS) == 13
    assert sum(len(scenario.turns) for scenario in SCENARIOS) == 15
    assert {scenario.id for scenario in SCENARIOS} == {
        "auth_change",
        "budget",
        "clarification",
        "document_injection",
        "greeting",
        "happy",
        "multi_turn",
        "poisoned_fact",
        "refine",
        "relevant_not_allowed",
        "repair",
        "retrieval_failure",
        "thread_mismatch",
    }
    assert [scenario.id for scenario in SCENARIOS if scenario.known_limitation] == [
        "poisoned_fact"
    ]


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda value: value.id)
def test_compiled_reference_scenario(
    scenario: ReferenceScenario,
    database_pool: ConnectionPool,
    retrieval_pool: ConnectionPool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="agentic_rag")
    load_reference_scenario(
        database_pool,
        scenario.fixture_setup.snapshot,
        scenario.fixture_setup.overlay,
    )
    aliases = _resolve_scenario_aliases(database_pool, scenario)
    script = _scenario_script(scenario, aliases, database_pool)
    graph = build_turn_graph(
        InMemorySaver(),
        interrupt_after=(
            "run_research_subgraph",
            "run_answer_subgraph",
            "authorize_and_commit_turn",
        ),
    )
    config: RunnableConfig = {
        "configurable": {"thread_id": f"scenario-{scenario.id}"}
    }
    previous_thread: dict[str, object] | ThreadState = ThreadState()

    for index, turn in enumerate(scenario.turns):
        principal_id = turn.principal_id or scenario.principal_id
        before_calls = _model_call_count(script)
        input_state: GraphState = {
            "thread": previous_thread,
            "current_turn": {
                "turn_id": turn.turn_id,
                "user_message": turn.user_message,
            },
        }
        if scenario.id == "thread_mismatch" and index == 1:
            with pytest.raises(
                ThreadPrincipalMismatchError,
                match="thread_principal_mismatch",
            ) as mismatch:
                graph.invoke(
                    cast(GraphState, {"current_turn": input_state["current_turn"]}),
                    config=config,
                    context=_runtime(script, retrieval_pool, principal_id),
                )
            assert _model_call_count(script) == before_calls
            latest = graph.get_state(config).values
            latest_thread = ThreadState.model_validate(latest["thread"])
            assert latest_thread.principal_id == "alice"
            assert len(latest_thread.turn_records) == 1
            rejected_work = CurrentTurnWork.model_validate(
                latest["current_turn"]
            )
            _assert_rejected_mismatch_turn(
                turn,
                mismatch.value,
                rejected_work,
                aliases,
            )
            continue

        states = _run_interrupted_turn(
            graph,
            input_state if index == 0 else cast(
                GraphState, {"current_turn": input_state["current_turn"]}
            ),
            config,
            _runtime(script, retrieval_pool, principal_id),
        )
        final = states[-1]
        previous_thread = cast(dict[str, object], final["thread"])
        _assert_turn(scenario, turn, states, aliases)

    assert not script.research_model.responses
    assert not script.answer_model.responses
    assert not script.contextualization_model.responses
    if scenario.id == "document_injection":
        assert "SYSTEM OVERRIDE" in str(script.answer_model.structured_inputs)
        _assert_document_content_cannot_self_grant(database_pool)
    _assert_scenario_non_disclosure(
        scenario,
        script,
        previous_thread,
        caplog.text,
        aliases,
        database_pool,
    )


def _run_interrupted_turn(
    graph: Any,
    input_state: GraphState,
    config: RunnableConfig,
    context: RuntimeContext,
) -> list[dict[str, Any]]:
    result = graph.invoke(input_state, config=config, context=context)
    states = [result]
    while result["current_turn"] is not None:
        result = graph.invoke(None, config=config, context=context)
        states.append(result)
    return states


def _assert_turn(
    scenario: ReferenceScenario,
    turn: ScenarioTurn,
    states: list[dict[str, Any]],
    aliases: dict[str, str],
) -> None:
    final_thread = ThreadState.model_validate(states[-1]["thread"])
    record = final_thread.turn_records[-1]
    expected = turn.expected
    assert record.outcome == expected.outcome
    assert record.terminal_reason == expected.terminal_reason
    for claim in expected.required_claims:
        assert claim.statement in record.assistant_message
    for claim in expected.forbidden_claims:
        assert claim.statement not in record.assistant_message

    authorizing = next(
        (
            CurrentTurnWork.model_validate(state["current_turn"])
            for state in reversed(states[:-1])
            if state["current_turn"] is not None
            and state["current_turn"]["stage"] == "authorizing"
        ),
        None,
    )
    assert authorizing is not None
    assert authorizing.retrieval_requests == expected.retrieval_requests
    assert authorizing.research_iterations == expected.research_iterations
    assert authorizing.answer_repairs == expected.answer_repairs
    assert authorizing.authorization_restarts == expected.authorization_restarts

    evidence_ids = (
        {item.chunk_id for item in authorizing.evidence_set.evidence_items}
        if authorizing.evidence_set is not None
        else set()
    )
    expected_evidence = {
        aliases[alias]
        for alias in (
            expected.evidence_aliases + expected.untrusted_evidence_aliases
        )
    }
    assert evidence_ids == expected_evidence
    forbidden_ids = {
        aliases[alias] for alias in expected.forbidden_evidence_aliases
    }
    assert evidence_ids.isdisjoint(forbidden_ids)

    cited_ids = (
        {mapping.chunk_id for mapping in authorizing.cited_answer.citations}
        if authorizing.cited_answer is not None
        else set()
    )
    assert cited_ids == {
        aliases[alias] for alias in expected.evidence_aliases
    }
    serialized = final_thread.model_dump_json()
    if scenario.id == "auth_change":
        assert "184 merchants" not in serialized
        assert "AUD 2.4 million" not in serialized
    if scenario.id == "document_injection":
        assert "SYSTEM OVERRIDE" not in serialized


def _scenario_script(
    scenario: ReferenceScenario,
    aliases: dict[str, str],
    database_pool: ConnectionPool,
) -> ScenarioScript:
    research: list[Any] = []
    answers: list[Any] = []
    contextualization: list[Any] = []
    targets: dict[str, tuple[str, ...]] = {}
    scenario_id = scenario.id

    def plan(
        query: str, *sources: KnowledgeSource
    ) -> ResearchPlan:
        return ResearchPlan(
            action=PlanAction.RETRIEVE,
            query=query,
            knowledge_sources=sources,
        )

    if scenario_id in {"happy", "repair", "thread_mismatch"}:
        research += [
            plan("production rollback root cause", KnowledgeSource.ENGINEERING_DOCS),
            EvidenceAssessment(sufficient=True),
        ]
        targets["production rollback root cause"] = (
            aliases["D2#rollback-incompatibility"],
        )
        if scenario_id == "repair":
            answers += [
                _draft("The rollback failed because of an invented network outage.", "E8"),
                _draft(_required_statement(scenario.turns[0]), "E1"),
                VerificationDecision(supported=True),
            ]
        else:
            answers += [
                _draft(_required_statement(scenario.turns[0]), "E1"),
                VerificationDecision(supported=True),
            ]
    elif scenario_id == "greeting":
        research.append(
            ResearchPlan(
                action=PlanAction.DIRECT,
                terminal_reason=PlanTerminalReason.GREETING,
            )
        )
    elif scenario_id == "clarification":
        research.append(
            ResearchPlan(
                action=PlanAction.CLARIFICATION,
                terminal_reason=PlanTerminalReason.AMBIGUOUS_REQUEST,
            )
        )
    elif scenario_id == "refine":
        research += [
            plan("empty legacy query", KnowledgeSource.ENGINEERING_DOCS),
            EvidenceAssessment(sufficient=False),
            QueryRefinement(query="legacy worker version"),
            EvidenceAssessment(sufficient=True),
        ]
        targets["empty legacy query"] = ()
        targets["legacy worker version"] = (
            aliases["D3#legacy-worker-version"],
        )
        answers += [
            _draft(_required_statement(scenario.turns[0]), "E1"),
            VerificationDecision(supported=True),
        ]
    elif scenario_id == "auth_change":
        research += [
            plan("confidential settlement impact", KnowledgeSource.OPERATIONAL_RUNBOOKS),
            EvidenceAssessment(sufficient=True),
            plan("public settlement status", KnowledgeSource.ENGINEERING_DOCS),
            EvidenceAssessment(sufficient=True),
        ]
        targets["confidential settlement impact"] = (
            aliases["D8#settlement-impact"],
        )
        targets["public settlement status"] = (
            aliases["D4#public-incident-summary"],
        )

        def revoke_finance_grant() -> VerificationDecision:
            with database_pool.connection() as connection:
                connection.execute(
                    """
                    DELETE FROM access_grants
                    WHERE grant_type = 'principal'
                      AND principal_id = 'alice'
                      AND document_id = (
                          SELECT document_id FROM source_documents
                          WHERE knowledge_source = 'operational-runbooks'
                            AND source_path = 'finance/settlement-impact.md'
                      )
                    """
                )
            return VerificationDecision(supported=True)

        answers += [
            _draft(_forbidden_statement(scenario.turns[0]), "E1"),
            revoke_finance_grant,
            _draft(_required_statement(scenario.turns[0]), "E1"),
            VerificationDecision(supported=True),
        ]
    elif scenario_id == "document_injection":
        research += [
            plan("idempotent replay", KnowledgeSource.ENGINEERING_DOCS),
            EvidenceAssessment(sufficient=True),
        ]
        targets["idempotent replay"] = (
            aliases["D5#valid-field-note"],
            aliases["D5#embedded-instruction"],
        )
        answers += [
            _draft(_required_statement(scenario.turns[0]), "E1"),
            VerificationDecision(supported=True),
        ]
    elif scenario_id == "poisoned_fact":
        research += [
            plan("legacy FAQ incident 4821", KnowledgeSource.ENGINEERING_DOCS),
            EvidenceAssessment(sufficient=True),
        ]
        targets["legacy FAQ incident 4821"] = (
            aliases["D6#false-compatibility-claim"],
        )
        answers += [
            _draft(_required_statement(scenario.turns[0]), "E1"),
            VerificationDecision(supported=True),
        ]
    elif scenario_id == "budget":
        research += [
            plan("empty lunar query", KnowledgeSource.ENGINEERING_DOCS),
            EvidenceAssessment(sufficient=False),
            QueryRefinement(query="empty lunar refined"),
            EvidenceAssessment(sufficient=False),
        ]
        targets["empty lunar query"] = ()
        targets["empty lunar refined"] = ()
    elif scenario_id == "retrieval_failure":
        research.append(
            plan("forced rerank failure", KnowledgeSource.ENGINEERING_DOCS)
        )
    elif scenario_id == "multi_turn":
        research += [
            plan("production rollback root cause", KnowledgeSource.ENGINEERING_DOCS),
            EvidenceAssessment(sufficient=True),
            plan("staging migration timeout", KnowledgeSource.OPERATIONAL_RUNBOOKS),
            EvidenceAssessment(sufficient=True),
        ]
        targets["production rollback root cause"] = (
            aliases["D2#rollback-incompatibility"],
        )
        targets["staging migration timeout"] = (
            aliases["D7#staging-timeout"],
        )
        answers += [
            _draft(_required_statement(scenario.turns[0]), "E1"),
            VerificationDecision(supported=True),
            _draft(_required_statement(scenario.turns[1]), "E1"),
            VerificationDecision(supported=True),
        ]
        contextualization.append(
            ContextualRewrite(
                standalone_question=(
                    "Did the production rollback failure also happen in staging?"
                ),
                depends_on_history=True,
                referenced_turn_ids=(scenario.turns[0].turn_id,),
                clarification_needed=False,
            )
        )
    elif scenario_id == "relevant_not_allowed":
        research += [
            plan("empty recovery steps", KnowledgeSource.OPERATIONAL_RUNBOOKS),
            EvidenceAssessment(sufficient=False),
            QueryRefinement(query="empty exact recovery procedure"),
            EvidenceAssessment(sufficient=False),
        ]
        targets["empty recovery steps"] = ()
        targets["empty exact recovery procedure"] = ()
    else:
        raise AssertionError(f"missing deterministic script for {scenario_id}")

    return ScenarioScript(
        research_model=ScriptedModel(responses=research),
        answer_model=ScriptedModel(responses=answers),
        contextualization_model=ScriptedModel(responses=contextualization),
        reranker=_scenario_reranker(targets),
    )


def _scenario_reranker(
    targets: dict[str, tuple[str, ...]],
) -> Callable[[str, tuple[RerankCandidate, ...]], RerankOutput]:
    def rerank(
        query: str, candidates: tuple[RerankCandidate, ...]
    ) -> RerankOutput:
        if query == "forced rerank failure":
            return RerankOutput(
                items=(RerankItem(chunk_id="unknown-chunk", score=1),)
            )
        preferred = targets.get(query, ())
        order = {chunk_id: index for index, chunk_id in enumerate(preferred)}
        ranked = sorted(
            candidates,
            key=lambda candidate: (
                order.get(candidate.chunk_id, len(order)),
                candidate.chunk_id,
            ),
        )
        return RerankOutput(
            items=tuple(
                RerankItem(
                    chunk_id=candidate.chunk_id,
                    score=(
                        1 - order[candidate.chunk_id] / 20
                        if candidate.chunk_id in order
                        else 0.4
                    ),
                )
                for candidate in ranked
            )
        )

    return rerank


def _runtime(
    script: ScenarioScript,
    pool: ConnectionPool,
    principal_id: str,
) -> RuntimeContext:
    return RuntimeContext(
        principal_id=principal_id,
        database_pool=pool,
        research_model=script.research_model,
        answer_model=script.answer_model,
        contextualization_model=script.contextualization_model,
        embedder=FixtureEmbedder(),
        reranker=script.reranker,
        retrieval_config=RetrievalConfig(
            embedding_model="deterministic-test-v1"
        ),
    )


def _resolve_scenario_aliases(
    pool: ConnectionPool, scenario: ReferenceScenario
) -> dict[str, str]:
    aliases = {
        alias
        for turn in scenario.turns
        for alias in (
            turn.expected.evidence_aliases
            + turn.expected.untrusted_evidence_aliases
            + turn.expected.forbidden_evidence_aliases
        )
    }
    return resolve_evidence_aliases(pool, tuple(sorted(aliases)))


def _draft(text: str, key: str) -> CitationDraft:
    return CitationDraft(
        disposition=DraftDisposition.FACTUAL,
        claims=(DraftClaim(text=text, citation_keys=(key,)),),
    )


def _required_statement(turn: ScenarioTurn) -> str:
    assert turn.expected.required_claims
    return turn.expected.required_claims[0].statement


def _forbidden_statement(turn: ScenarioTurn) -> str:
    assert turn.expected.forbidden_claims
    return turn.expected.forbidden_claims[0].statement


def _model_call_count(script: ScenarioScript) -> int:
    return sum(
        len(model.requested_schemas)
        for model in (
            script.research_model,
            script.answer_model,
            script.contextualization_model,
        )
    )


def _assert_document_content_cannot_self_grant(
    pool: ConnectionPool,
) -> None:
    with pool.connection() as connection:
        row = connection.execute(
            """
            SELECT document_id
            FROM source_documents
            WHERE knowledge_source = 'engineering-docs'
              AND source_path = 'payments/migration-field-notes.md'
            """
        ).fetchone()
    assert row is not None
    assert not source_document_is_authorized(
        pool,
        capture_authorization_snapshot(pool, "carol"),
        knowledge_source="engineering-docs",
        document_id=str(row[0]),
    )


def _assert_rejected_mismatch_turn(
    turn: ScenarioTurn,
    error: ThreadPrincipalMismatchError,
    work: CurrentTurnWork,
    aliases: dict[str, str],
) -> None:
    expected = turn.expected
    assert expected.outcome is TurnOutcome.REFUSED
    assert expected.terminal_reason == str(error)
    assert work.stage == "new"
    assert work.standalone_question is None
    assert work.retrieval_requests == expected.retrieval_requests == 0
    assert work.research_iterations == expected.research_iterations == 0
    assert work.answer_repairs == expected.answer_repairs == 0
    assert work.authorization_restarts == expected.authorization_restarts == 0
    error_projection = str(error)
    assert error_projection == "thread_principal_mismatch"
    for claim in expected.required_claims + expected.forbidden_claims:
        assert claim.statement not in error_projection
    for alias in expected.forbidden_evidence_aliases:
        assert aliases[alias] not in error_projection


def _assert_scenario_non_disclosure(
    scenario: ReferenceScenario,
    script: ScenarioScript,
    final_thread: dict[str, object] | ThreadState,
    logs: str,
    aliases: dict[str, str],
    pool: ConnectionPool,
) -> None:
    if scenario.id == "relevant_not_allowed":
        projection = " ".join(
            (
                ThreadState.model_validate(final_thread).model_dump_json(),
                str(script.research_model.structured_inputs),
                str(script.answer_model.structured_inputs),
                logs,
            )
        )
        forbidden = (
            "Payments Rollback Recovery",
            "payments/rollback-recovery.md",
            "payments-on-call",
            "alice",
            "inaccessible",
            _forbidden_statement(scenario.turns[0]),
        )
    elif scenario.id == "auth_change":
        projection = " ".join(
            (
                ThreadState.model_validate(final_thread).model_dump_json(),
                str(script.research_model.structured_inputs[-2:]),
                str(script.answer_model.structured_inputs[-2:]),
                logs,
            )
        )
        forbidden = (
            "Settlement Impact",
            "finance/settlement-impact.md",
            "direct grant",
            "inaccessible",
            _forbidden_statement(scenario.turns[0]),
        )
    else:
        return
    forbidden_aliases = {
        alias
        for turn in scenario.turns
        for alias in turn.expected.forbidden_evidence_aliases
    }
    chunk_ids = {aliases[alias] for alias in forbidden_aliases}
    document_ids = _document_ids_for_chunks(pool, chunk_ids)
    for value in forbidden + tuple(chunk_ids) + tuple(document_ids):
        assert value not in projection
    for field in (
        "inaccessible_count",
        "unauthorized_count",
        "forbidden_count",
    ):
        assert field not in projection
    for count in {len(chunk_ids), len(document_ids)}:
        assert not re.search(
            rf"\b{count}\s+(?:inaccessible|unauthorized|forbidden)\b",
            projection,
            re.IGNORECASE,
        )


def _document_ids_for_chunks(
    pool: ConnectionPool,
    chunk_ids: set[str],
) -> set[str]:
    document_ids: set[str] = set()
    with pool.connection() as connection:
        for chunk_id in chunk_ids:
            row = connection.execute(
                "SELECT document_id FROM indexed_chunks WHERE chunk_id = %s",
                (chunk_id,),
            ).fetchone()
            assert row is not None
            document_ids.add(str(row[0]))
    return document_ids
