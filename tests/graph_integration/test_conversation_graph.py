"""L3 checkpoint verification for multi-turn terminal commit."""

import os
from collections.abc import Iterator
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
    ResearchPlan,
)
from agentic_rag.authorization import (
    AuthorizationSnapshot,
    capture_authorization_snapshot,
)
from agentic_rag.conversation import (
    CitationDependency,
    ConversationSummary,
    SummaryItem,
    SummaryItemKind,
    ThreadPrincipalMismatchError,
    ThreadState,
    TurnOutcome,
    TurnRecord,
    TurnResumeIncompatibleError,
    begin_turn,
    commit_turn,
    project_authorized_thread_context,
)
from agentic_rag.corpus import KnowledgeSource
from agentic_rag.corpus.models import SourceLocator
from agentic_rag.citations import (
    CitationDraft,
    DraftClaim,
    DraftDisposition,
)
from agentic_rag.database import apply_migrations, open_database_pool
from agentic_rag.graph._turn import build_turn_graph
from agentic_rag.graph.state import GraphState
from agentic_rag.retrieval import (
    RerankCandidate,
    RerankItem,
    RerankOutput,
    RetrievalConfig,
)
from agentic_rag.runtime import RuntimeContext, TurnExecutionBudget
from tests.support.reference_fixture import (
    FixtureEmbedder,
    load_reference_snapshot,
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


class DeterministicContextModel(BaseChatModel):
    responses: list[Any] = Field(default_factory=list)
    inputs: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "deterministic-context"

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
        def next_response(value: object) -> Any:
            self.inputs.append(value)
            response = self.responses.pop(0)
            return response() if callable(response) else response

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
def empty_thread_data(database_pool: ConnectionPool) -> None:
    with database_pool.connection() as connection:
        connection.execute(
            """
            TRUNCATE access_grants, principal_group_memberships, groups,
                principals, sync_reports, indexed_chunks, source_documents,
                knowledge_sources CASCADE
            """
        )


def test_checkpointer_reuses_thread_and_completed_turn_is_idempotent() -> None:
    graph = build_turn_graph(InMemorySaver())
    config: RunnableConfig = {
        "configurable": {"thread_id": "thread-1"}
    }
    first = graph.invoke(
        {
            "thread": ThreadState(),
            "current_turn": {
                "turn_id": "turn-1",
                "user_message": "Why did rollback fail?",
            },
        },
        config=config,
        context=RuntimeContext(principal_id="alice"),
    )
    assert first["current_turn"] is None

    second_input = cast(
        GraphState,
        {
            "current_turn": {
                "turn_id": "turn-2",
                "user_message": "Did it happen in staging?",
            }
        },
    )
    second = graph.invoke(
        second_input,
        config=config,
        context=RuntimeContext(principal_id="alice"),
    )
    replay = graph.invoke(
        cast(
            GraphState,
            {
                "current_turn": {
                    "turn_id": "turn-2",
                    "user_message": "MUST_NOT_REEXECUTE",
                }
            },
        ),
        config=config,
        context=RuntimeContext(principal_id="alice"),
    )

    thread = ThreadState.model_validate(second["thread"])
    replayed = ThreadState.model_validate(replay["thread"])
    assert [record.turn_id for record in thread.turn_records] == [
        "turn-1",
        "turn-2",
    ]
    assert replayed == thread
    assert "MUST_NOT_REEXECUTE" not in replayed.model_dump_json()


def test_checkpointed_thread_rejects_another_principal() -> None:
    graph = build_turn_graph(InMemorySaver())
    config: RunnableConfig = {
        "configurable": {"thread_id": "thread-principal"}
    }
    graph.invoke(
        {
            "thread": ThreadState(),
            "current_turn": {
                "turn_id": "turn-1",
                "user_message": "First",
            },
        },
        config=config,
        context=RuntimeContext(principal_id="alice"),
    )

    with pytest.raises(ThreadPrincipalMismatchError):
        graph.invoke(
            cast(
                GraphState,
                {
                    "current_turn": {
                        "turn_id": "turn-2",
                        "user_message": "Continue",
                    }
                },
            ),
            config=config,
            context=RuntimeContext(principal_id="bob"),
        )


def test_revoked_historical_answer_is_removed_from_active_context(
    database_pool: ConnectionPool,
    retrieval_pool: ConnectionPool,
) -> None:
    load_reference_snapshot(database_pool)
    snapshot = capture_authorization_snapshot(retrieval_pool, "alice")
    dependency = _finance_dependency(database_pool)
    thread = commit_turn(
        begin_turn(ThreadState(), "alice", "turn-finance"),
        TurnRecord(
            turn_id="turn-finance",
            user_message="What was the settlement impact?",
            standalone_question="What was the settlement impact?",
            assistant_message="AUTHORIZED_HISTORY_CANARY",
            outcome=TurnOutcome.ANSWERED,
            citation_dependencies=(dependency,),
        ),
    )
    before = project_authorized_thread_context(
        retrieval_pool,
        snapshot,
        thread,
        current_user_message="What about now?",
    )
    assert before.recent_turns[0].assistant_message == (
        "AUTHORIZED_HISTORY_CANARY"
    )

    with database_pool.connection() as connection:
        connection.execute(
            """
            DELETE FROM access_grants
            WHERE grant_type = 'principal' AND principal_id = 'alice'
              AND document_id = %s
            """,
            (dependency.document_id,),
        )
    after = project_authorized_thread_context(
        retrieval_pool,
        snapshot,
        thread,
        current_user_message="What about now?",
    )

    projected = after.recent_turns[0]
    assert projected.user_message == "What was the settlement impact?"
    assert projected.assistant_message is None
    assert projected.standalone_question is None
    assert projected.citation_dependencies == ()
    assert projected.marker == "historical_evidence_no_longer_authorized"
    assert "AUTHORIZED_HISTORY_CANARY" not in after.model_dump_json()


def test_second_turn_records_authorized_standalone_rewrite(
    database_pool: ConnectionPool,
    retrieval_pool: ConnectionPool,
) -> None:
    load_reference_snapshot(database_pool)
    model = DeterministicContextModel(
        responses=[
            ContextualRewrite(
                standalone_question=(
                    "Did the payments rollback also fail in staging?"
                ),
                depends_on_history=True,
                referenced_turn_ids=("turn-1",),
                clarification_needed=False,
            )
        ]
    )
    graph = build_turn_graph(InMemorySaver())
    config: RunnableConfig = {
        "configurable": {"thread_id": "thread-rewrite"}
    }
    context = RuntimeContext(
        principal_id="alice",
        database_pool=retrieval_pool,
        contextualization_model=model,
    )
    graph.invoke(
        {
            "thread": ThreadState(),
            "current_turn": {
                "turn_id": "turn-1",
                "user_message": "Why did the payments rollback fail?",
            },
        },
        config=config,
        context=context,
    )

    result = graph.invoke(
        cast(
            GraphState,
            {
                "current_turn": {
                    "turn_id": "turn-2",
                    "user_message": "Did it also happen in staging?",
                }
            },
        ),
        config=config,
        context=context,
    )

    thread = ThreadState.model_validate(result["thread"])
    assert thread.turn_records[-1].standalone_question == (
        "Did the payments rollback also fail in staging?"
    )
    assert len(thread.turn_records) == 2


def test_graph_resumes_a_compatible_checkpointed_turn() -> None:
    graph = build_turn_graph(
        InMemorySaver(), interrupt_after=("prepare_turn",)
    )
    config: RunnableConfig = {
        "configurable": {"thread_id": "thread-resume"}
    }
    interrupted = graph.invoke(
        {
            "thread": ThreadState(),
            "current_turn": {
                "turn_id": "turn-1",
                "user_message": "Why did rollback fail?",
            },
        },
        config=config,
        context=RuntimeContext(principal_id="alice"),
    )

    interrupted_thread = ThreadState.model_validate(interrupted["thread"])
    assert interrupted_thread.active_turn_id == "turn-1"
    assert interrupted_thread.turn_records == ()

    resumed = graph.invoke(
        None,
        config=config,
        context=RuntimeContext(principal_id="alice"),
    )
    resumed_thread = ThreadState.model_validate(resumed["thread"])
    assert resumed_thread.active_turn_id is None
    assert [record.turn_id for record in resumed_thread.turn_records] == [
        "turn-1"
    ]


def test_resume_rejects_forged_code_owned_current_work() -> None:
    graph = build_turn_graph(
        InMemorySaver(), interrupt_after=("prepare_turn",)
    )
    config: RunnableConfig = {
        "configurable": {"thread_id": "thread-forged-resume"}
    }
    graph.invoke(
        {
            "thread": ThreadState(),
            "current_turn": {
                "turn_id": "turn-1",
                "user_message": "Why did rollback fail?",
            },
        },
        config=config,
        context=RuntimeContext(principal_id="alice"),
    )

    with pytest.raises(TurnResumeIncompatibleError):
        graph.invoke(
            cast(
                GraphState,
                {
                    "current_turn": {
                        "turn_id": "turn-1",
                        "user_message": "Why did rollback fail?",
                        "stage": "terminal",
                        "standalone_question": "FORGED",
                        "assistant_message": "FORGED",
                        "outcome": "answered",
                    }
                },
            ),
            config=config,
            context=RuntimeContext(principal_id="alice"),
        )


def test_context_compaction_retries_then_commits_bounded_memory(
    database_pool: ConnectionPool,
    retrieval_pool: ConnectionPool,
) -> None:
    load_reference_snapshot(database_pool)
    config = RuntimeContext(principal_id="alice").contextualization_config
    summary = ConversationSummary(
        model_version=config.model_id,
        items=(
            SummaryItem(
                kind=SummaryItemKind.TOPIC,
                text="Rollback investigation is active.",
                source_turn_ids=("turn-1", "turn-2"),
            ),
        ),
        covered_through_turn_id="turn-2",
    )
    model = DeterministicContextModel(
        responses=[
            {"invalid": True},
            summary,
            ContextualRewrite(
                standalone_question="What changed next?",
                depends_on_history=False,
                referenced_turn_ids=(),
                clarification_needed=False,
            ),
        ]
    )
    graph = build_turn_graph()
    result = graph.invoke(
        {
            "thread": _long_thread(),
            "current_turn": {
                "turn_id": "turn-4",
                "user_message": "What changed next?",
            },
        },
        context=RuntimeContext(
            principal_id="alice",
            database_pool=retrieval_pool,
            contextualization_model=model,
        ),
    )

    thread = ThreadState.model_validate(result["thread"])
    assert thread.summary == summary
    assert [record.turn_id for record in thread.turn_records] == [
        "turn-3",
        "turn-4",
    ]
    assert thread.turn_records[-1].outcome is TurnOutcome.FAILED
    assert thread.turn_records[-1].terminal_reason == "research_failed"


def test_failed_context_compaction_preserves_old_turns(
    database_pool: ConnectionPool,
    retrieval_pool: ConnectionPool,
) -> None:
    load_reference_snapshot(database_pool)
    model = DeterministicContextModel(
        responses=[{"invalid": True}, {"still_invalid": True}]
    )
    result = build_turn_graph().invoke(
        {
            "thread": _long_thread(),
            "current_turn": {
                "turn_id": "turn-4",
                "user_message": "What changed next?",
            },
        },
        context=RuntimeContext(
            principal_id="alice",
            database_pool=retrieval_pool,
            contextualization_model=model,
        ),
    )

    thread = ThreadState.model_validate(result["thread"])
    assert [record.turn_id for record in thread.turn_records[:-1]] == [
        "turn-1",
        "turn-2",
        "turn-3",
    ]
    assert thread.turn_records[-1].outcome is TurnOutcome.FAILED
    assert thread.turn_records[-1].terminal_reason == (
        "context_compaction_failed"
    )


def test_compaction_and_rewrite_share_one_model_call_budget(
    database_pool: ConnectionPool,
    retrieval_pool: ConnectionPool,
) -> None:
    load_reference_snapshot(database_pool)
    config = RuntimeContext(principal_id="alice").contextualization_config
    model = DeterministicContextModel(
        responses=[
            {"invalid": True},
            ConversationSummary(
                model_version=config.model_id,
                items=(
                    SummaryItem(
                        kind=SummaryItemKind.TOPIC,
                        text="Rollback investigation is active.",
                        source_turn_ids=("turn-1", "turn-2"),
                    ),
                ),
                covered_through_turn_id="turn-2",
            ),
        ]
    )
    result = build_turn_graph().invoke(
        {
            "thread": _long_thread(),
            "current_turn": {
                "turn_id": "turn-4",
                "user_message": "What changed next?",
            },
        },
        context=RuntimeContext(
            principal_id="alice",
            database_pool=retrieval_pool,
            contextualization_model=model,
            execution_budget=TurnExecutionBudget(model_call_limit=1),
        ),
    )

    thread = ThreadState.model_validate(result["thread"])
    assert thread.turn_records[-1].outcome is TurnOutcome.FAILED
    assert thread.turn_records[-1].terminal_reason == "model_calls_exhausted"
    assert len(model.inputs) == 1


def test_single_oversized_historical_turn_is_compacted(
    database_pool: ConnectionPool,
    retrieval_pool: ConnectionPool,
) -> None:
    load_reference_snapshot(database_pool)
    config = RuntimeContext(principal_id="alice").contextualization_config
    model = DeterministicContextModel(
        responses=[
            ConversationSummary(
                model_version=config.model_id,
                items=(
                    SummaryItem(
                        kind=SummaryItemKind.TOPIC,
                        text="Rollback investigation is active.",
                        source_turn_ids=("turn-1",),
                    ),
                ),
                covered_through_turn_id="turn-1",
            ),
            ContextualRewrite(
                standalone_question="What changed next?",
                depends_on_history=True,
                referenced_turn_ids=("turn-1",),
                clarification_needed=False,
            ),
        ]
    )
    oversized_record = _long_thread().turn_records[0].model_copy(
        update={
            "user_message": "q" * 3_000,
            "standalone_question": "q" * 3_000,
            "assistant_message": "a" * 3_000,
        }
    )
    oversized = ThreadState(
        principal_id="alice", turn_records=(oversized_record,)
    )

    result = build_turn_graph().invoke(
        {
            "thread": oversized,
            "current_turn": {
                "turn_id": "turn-2",
                "user_message": "What changed next?",
            },
        },
        context=RuntimeContext(
            principal_id="alice",
            database_pool=retrieval_pool,
            contextualization_model=model,
        ),
    )

    thread = ThreadState.model_validate(result["thread"])
    assert thread.summary is not None
    assert [record.turn_id for record in thread.turn_records] == ["turn-2"]
    assert thread.turn_records[0].standalone_question == "What changed next?"


def test_resume_after_compaction_preserves_work_and_model_budget(
    database_pool: ConnectionPool,
    retrieval_pool: ConnectionPool,
) -> None:
    load_reference_snapshot(database_pool)
    context_config = RuntimeContext(
        principal_id="alice"
    ).contextualization_config
    model = DeterministicContextModel(
        responses=[
            {"invalid": True},
            ConversationSummary(
                model_version=context_config.model_id,
                items=(
                    SummaryItem(
                        kind=SummaryItemKind.TOPIC,
                        text="Rollback investigation is active.",
                        source_turn_ids=("turn-1", "turn-2"),
                    ),
                ),
                covered_through_turn_id="turn-2",
            ),
            ContextualRewrite(
                standalone_question="What changed next?",
                depends_on_history=False,
                referenced_turn_ids=(),
                clarification_needed=False,
            ),
        ]
    )
    graph = build_turn_graph(
        InMemorySaver(), interrupt_after=("compact_context",)
    )
    run_config: RunnableConfig = {
        "configurable": {"thread_id": "thread-resume-after-compaction"}
    }
    runtime = RuntimeContext(
        principal_id="alice",
        database_pool=retrieval_pool,
        contextualization_model=model,
    )

    interrupted = graph.invoke(
        {
            "thread": _long_thread(),
            "current_turn": {
                "turn_id": "turn-4",
                "user_message": "What changed next?",
            },
        },
        config=run_config,
        context=runtime,
    )

    assert interrupted["current_turn"]["model_calls"] == 1
    assert interrupted["current_turn"]["compaction_attempts"] == 1
    assert interrupted["thread"]["summary"] is None

    compacted = graph.invoke(None, config=run_config, context=runtime)
    assert compacted["current_turn"]["model_calls"] == 2
    assert compacted["current_turn"]["compaction_attempts"] == 2
    assert compacted["thread"]["summary"] is not None

    resumed = graph.invoke(None, config=run_config, context=runtime)
    thread = ThreadState.model_validate(resumed["thread"])
    assert resumed["current_turn"] is None
    assert thread.turn_records[-1].turn_id == "turn-4"
    assert len(model.inputs) == 3


def test_compaction_never_sends_revoked_history_to_the_model(
    database_pool: ConnectionPool,
    retrieval_pool: ConnectionPool,
) -> None:
    load_reference_snapshot(database_pool)
    dependency = _finance_dependency(database_pool)
    records = list(_long_thread().turn_records)
    records[0] = records[0].model_copy(
        update={
            "assistant_message": "REVOKED_COMPACTION_CANARY" + "a" * 1_300,
            "citation_dependencies": (dependency,),
        }
    )
    thread = ThreadState(principal_id="alice", turn_records=tuple(records))
    with database_pool.connection() as connection:
        connection.execute(
            """
            DELETE FROM access_grants
            WHERE grant_type = 'principal' AND principal_id = 'alice'
              AND document_id = %s
            """,
            (dependency.document_id,),
        )

    config = RuntimeContext(principal_id="alice").contextualization_config
    model = DeterministicContextModel(
        responses=[
            ConversationSummary(
                model_version=config.model_id,
                items=(
                    SummaryItem(
                        kind=SummaryItemKind.TOPIC,
                        text="Rollback investigation is active.",
                        source_turn_ids=("turn-1", "turn-2"),
                    ),
                ),
                covered_through_turn_id="turn-2",
            ),
            ContextualRewrite(
                standalone_question="What changed next?",
                depends_on_history=False,
                referenced_turn_ids=(),
                clarification_needed=False,
            ),
        ]
    )
    build_turn_graph().invoke(
        {
            "thread": thread,
            "current_turn": {
                "turn_id": "turn-4",
                "user_message": "What changed next?",
            },
        },
        context=RuntimeContext(
            principal_id="alice",
            database_pool=retrieval_pool,
            contextualization_model=model,
        ),
    )

    assert "REVOKED_COMPACTION_CANARY" not in str(model.inputs[0])
    assert "historical_evidence_no_longer_authorized" in str(model.inputs[0])


def test_authorization_change_discards_work_and_restarts_once(
    database_pool: ConnectionPool,
    retrieval_pool: ConnectionPool,
) -> None:
    load_reference_snapshot(database_pool)
    research_model = DeterministicContextModel(
        responses=[
            _research_plan(),
            EvidenceAssessment(sufficient=True),
            _research_plan(),
            EvidenceAssessment(sufficient=True),
        ]
    )
    grant: tuple[str, str] | None = None

    def change_scope_once() -> VerificationDecision:
        nonlocal grant
        grant = _grant_unrelated_document(database_pool, "bob")
        return VerificationDecision(supported=True)

    answer_model = DeterministicContextModel(
        responses=[
            _answer_draft("DISCARDED_ANSWER_CANARY"),
            change_scope_once,
            _answer_draft("AUTHORIZED_RESTARTED_ANSWER"),
            VerificationDecision(supported=True),
        ]
    )

    result = build_turn_graph().invoke(
        {
            "thread": ThreadState(),
            "current_turn": {
                "turn_id": "turn-authorization-restart",
                "user_message": "Why did the payments rollback fail?",
            },
        },
        context=_full_turn_context(
            retrieval_pool, research_model, answer_model
        ),
    )

    thread = ThreadState.model_validate(result["thread"])
    record = thread.turn_records[-1]
    assert grant is not None
    assert record.outcome is TurnOutcome.ANSWERED
    assert "AUTHORIZED_RESTARTED_ANSWER" in record.assistant_message
    assert "DISCARDED_ANSWER_CANARY" not in thread.model_dump_json()
    assert record.citation_dependencies
    assert len(research_model.inputs) == 4
    assert len(answer_model.inputs) == 4


def test_second_authorization_change_fails_without_stale_output(
    database_pool: ConnectionPool,
    retrieval_pool: ConnectionPool,
) -> None:
    load_reference_snapshot(database_pool)
    research_model = DeterministicContextModel(
        responses=[
            _research_plan(),
            EvidenceAssessment(sufficient=True),
            _research_plan(),
            EvidenceAssessment(sufficient=True),
        ]
    )
    grant: tuple[str, str] | None = None

    def first_change() -> VerificationDecision:
        nonlocal grant
        grant = _grant_unrelated_document(database_pool, "bob")
        return VerificationDecision(supported=True)

    def second_change() -> VerificationDecision:
        assert grant is not None
        with database_pool.connection() as connection:
            connection.execute(
                """
                DELETE FROM access_grants
                WHERE knowledge_source = %s AND document_id = %s
                  AND grant_type = 'principal' AND principal_id = 'bob'
                """,
                grant,
            )
        return VerificationDecision(supported=True)

    answer_model = DeterministicContextModel(
        responses=[
            _answer_draft("FIRST_DISCARDED_CANARY"),
            first_change,
            _answer_draft("SECOND_DISCARDED_CANARY"),
            second_change,
        ]
    )

    result = build_turn_graph().invoke(
        {
            "thread": ThreadState(),
            "current_turn": {
                "turn_id": "turn-authorization-fail",
                "user_message": "Why did the payments rollback fail?",
            },
        },
        context=_full_turn_context(
            retrieval_pool, research_model, answer_model
        ),
    )

    thread = ThreadState.model_validate(result["thread"])
    record = thread.turn_records[-1]
    serialized = thread.model_dump_json()
    assert record.outcome is TurnOutcome.FAILED
    assert record.terminal_reason == "authorization_changed_twice"
    assert record.citation_dependencies == ()
    assert "FIRST_DISCARDED_CANARY" not in serialized
    assert "SECOND_DISCARDED_CANARY" not in serialized


def test_revoked_evidence_is_discarded_before_a_safe_restart_result(
    database_pool: ConnectionPool,
    retrieval_pool: ConnectionPool,
) -> None:
    load_reference_snapshot(database_pool)
    research_model = DeterministicContextModel(
        responses=[
            _research_plan(),
            EvidenceAssessment(sufficient=True),
            ResearchPlan(
                action=PlanAction.DIRECT,
                terminal_reason=PlanTerminalReason.GREETING,
            ),
        ]
    )

    def revoke_engineering_scope() -> VerificationDecision:
        with database_pool.connection() as connection:
            connection.execute(
                """
                DELETE FROM principal_group_memberships
                WHERE principal_id = 'bob'
                  AND group_id = 'payments-engineering'
                """
            )
        return VerificationDecision(supported=True)

    answer_model = DeterministicContextModel(
        responses=[
            _answer_draft("REVOKED_EVIDENCE_ANSWER_CANARY"),
            revoke_engineering_scope,
        ]
    )
    result = build_turn_graph().invoke(
        {
            "thread": ThreadState(),
            "current_turn": {
                "turn_id": "turn-revocation",
                "user_message": "Why did the payments rollback fail?",
            },
        },
        context=RuntimeContext(
            principal_id="bob",
            database_pool=retrieval_pool,
            research_model=research_model,
            answer_model=answer_model,
            embedder=FixtureEmbedder(),
            reranker=_rerank_private_first,
            retrieval_config=RetrievalConfig(
                embedding_model="deterministic-test-v1"
            ),
        ),
    )

    thread = ThreadState.model_validate(result["thread"])
    record = thread.turn_records[-1]
    assert record.outcome is TurnOutcome.ANSWERED
    assert record.assistant_message.startswith("Hello.")
    assert record.citation_dependencies == ()
    assert "REVOKED_EVIDENCE_ANSWER_CANARY" not in thread.model_dump_json()
    assert len(research_model.inputs) == 3
    assert len(answer_model.inputs) == 2
    assert "Worker v1.8 expects" in str(answer_model.inputs[0])


def test_scope_change_during_clarification_restarts_before_commit(
    database_pool: ConnectionPool,
    retrieval_pool: ConnectionPool,
) -> None:
    load_reference_snapshot(database_pool)
    grant: tuple[str, str] | None = None

    def change_scope_during_rewrite() -> ContextualRewrite:
        nonlocal grant
        grant = _grant_unrelated_document(database_pool, "bob")
        return _clarification_rewrite()

    model = DeterministicContextModel(
        responses=[change_scope_during_rewrite, _clarification_rewrite()]
    )
    thread = ThreadState(
        principal_id="bob",
        turn_records=(
            TurnRecord(
                turn_id="turn-1",
                user_message="The rollback failed.",
                standalone_question="Why did the rollback fail?",
                assistant_message="A prior safe answer.",
                outcome=TurnOutcome.ANSWERED,
            ),
        ),
    )

    result = build_turn_graph().invoke(
        {
            "thread": thread,
            "current_turn": {
                "turn_id": "turn-2",
                "user_message": "Did it happen there too?",
            },
        },
        context=RuntimeContext(
            principal_id="bob",
            database_pool=retrieval_pool,
            contextualization_model=model,
        ),
    )

    completed = ThreadState.model_validate(result["thread"])
    record = completed.turn_records[-1]
    assert grant is not None
    assert len(model.inputs) == 2
    assert record.outcome is TurnOutcome.CLARIFICATION_REQUESTED
    assert record.terminal_reason == "clarification_needed"


def test_authorization_restart_rolls_back_and_rebuilds_compaction(
    database_pool: ConnectionPool,
    retrieval_pool: ConnectionPool,
) -> None:
    load_reference_snapshot(database_pool)
    dependency = _finance_dependency(database_pool)
    records = list(_long_thread().turn_records)
    records[0] = records[0].model_copy(
        update={
            "assistant_message": "SUMMARY_SCOPE_CANARY" + "a" * 1_300,
            "citation_dependencies": (dependency,),
        }
    )
    thread = ThreadState(principal_id="alice", turn_records=tuple(records))
    config = RuntimeContext(principal_id="alice").contextualization_config
    first_summary = ConversationSummary(
        model_version=config.model_id,
        items=(
            SummaryItem(
                kind=SummaryItemKind.TOPIC,
                text="SUMMARY_SCOPE_CANARY",
                source_turn_ids=("turn-1", "turn-2"),
            ),
        ),
        covered_through_turn_id="turn-2",
    )
    rebuilt_summary = ConversationSummary(
        model_version=config.model_id,
        items=(
            SummaryItem(
                kind=SummaryItemKind.TOPIC,
                text="Authorized rollback discussion.",
                source_turn_ids=("turn-1", "turn-2"),
            ),
        ),
        covered_through_turn_id="turn-2",
    )
    context_model = DeterministicContextModel(
        responses=[
            first_summary,
            _standalone_rewrite("What changed next?"),
            rebuilt_summary,
            _standalone_rewrite("What changed next?"),
        ]
    )
    research_model = DeterministicContextModel(
        responses=[
            _research_plan(),
            EvidenceAssessment(sufficient=True),
            ResearchPlan(
                action=PlanAction.DIRECT,
                terminal_reason=PlanTerminalReason.GREETING,
            ),
        ]
    )

    def revoke_summary_dependency() -> VerificationDecision:
        with database_pool.connection() as connection:
            connection.execute(
                """
                DELETE FROM access_grants
                WHERE grant_type = 'principal' AND principal_id = 'alice'
                  AND document_id = %s
                """,
                (dependency.document_id,),
            )
        return VerificationDecision(supported=True)

    answer_model = DeterministicContextModel(
        responses=[
            _answer_draft("DISCARDED_AFTER_COMPACTION"),
            revoke_summary_dependency,
        ]
    )
    result = build_turn_graph().invoke(
        {
            "thread": thread,
            "current_turn": {
                "turn_id": "turn-4",
                "user_message": "What changed next?",
            },
        },
        context=RuntimeContext(
            principal_id="alice",
            database_pool=retrieval_pool,
            contextualization_model=context_model,
            research_model=research_model,
            answer_model=answer_model,
            embedder=FixtureEmbedder(),
            reranker=_rerank_all,
            retrieval_config=RetrievalConfig(
                embedding_model="deterministic-test-v1"
            ),
        ),
    )

    completed = ThreadState.model_validate(result["thread"])
    assert completed.summary == rebuilt_summary
    assert [record.turn_id for record in completed.turn_records] == [
        "turn-3",
        "turn-4",
    ]
    assert completed.turn_records[-1].outcome is TurnOutcome.ANSWERED
    assert "SUMMARY_SCOPE_CANARY" not in completed.model_dump_json()
    assert "SUMMARY_SCOPE_CANARY" in str(context_model.inputs[0])
    assert "SUMMARY_SCOPE_CANARY" not in str(context_model.inputs[2])
    assert "DISCARDED_AFTER_COMPACTION" not in completed.model_dump_json()


def test_second_change_rolls_back_recompacted_thread_before_failure(
    database_pool: ConnectionPool,
    retrieval_pool: ConnectionPool,
) -> None:
    load_reference_snapshot(database_pool)
    dependency = _finance_dependency(database_pool)
    records = list(_long_thread().turn_records)
    records[0] = records[0].model_copy(
        update={"citation_dependencies": (dependency,)}
    )
    thread = ThreadState(principal_id="alice", turn_records=tuple(records))
    config = RuntimeContext(principal_id="alice").contextualization_config
    context_model = DeterministicContextModel(
        responses=[
            _topic_summary(config.model_id, "FIRST_SUMMARY_CANARY"),
            _standalone_rewrite("What changed next?"),
            _topic_summary(config.model_id, "SECOND_STALE_SUMMARY_CANARY"),
            _standalone_rewrite("What changed next?"),
        ]
    )

    def second_scope_change() -> ResearchPlan:
        with database_pool.connection() as connection:
            connection.execute(
                """
                INSERT INTO access_grants (
                    knowledge_source, document_id, grant_type, principal_id
                ) VALUES (%s, %s, 'principal', 'alice')
                """,
                (dependency.knowledge_source.value, dependency.document_id),
            )
        return ResearchPlan(
            action=PlanAction.DIRECT,
            terminal_reason=PlanTerminalReason.GREETING,
        )

    research_model = DeterministicContextModel(
        responses=[
            _research_plan(),
            EvidenceAssessment(sufficient=True),
            second_scope_change,
        ]
    )

    def first_scope_change() -> VerificationDecision:
        with database_pool.connection() as connection:
            connection.execute(
                """
                DELETE FROM access_grants
                WHERE grant_type = 'principal' AND principal_id = 'alice'
                  AND document_id = %s
                """,
                (dependency.document_id,),
            )
        return VerificationDecision(supported=True)

    answer_model = DeterministicContextModel(
        responses=[
            _answer_draft("DISCARDED_BEFORE_SECOND_CHANGE"),
            first_scope_change,
        ]
    )
    result = build_turn_graph().invoke(
        {
            "thread": thread,
            "current_turn": {
                "turn_id": "turn-4",
                "user_message": "What changed next?",
            },
        },
        context=RuntimeContext(
            principal_id="alice",
            database_pool=retrieval_pool,
            contextualization_model=context_model,
            research_model=research_model,
            answer_model=answer_model,
            embedder=FixtureEmbedder(),
            reranker=_rerank_all,
            retrieval_config=RetrievalConfig(
                embedding_model="deterministic-test-v1"
            ),
        ),
    )

    completed = ThreadState.model_validate(result["thread"])
    serialized = completed.model_dump_json()
    assert completed.summary is None
    assert [record.turn_id for record in completed.turn_records] == [
        "turn-1",
        "turn-2",
        "turn-3",
        "turn-4",
    ]
    assert completed.turn_records[-1].outcome is TurnOutcome.FAILED
    assert completed.turn_records[-1].terminal_reason == (
        "authorization_changed_twice"
    )
    assert "FIRST_SUMMARY_CANARY" not in serialized
    assert "SECOND_STALE_SUMMARY_CANARY" not in serialized
    assert "DISCARDED_BEFORE_SECOND_CHANGE" not in serialized


def test_authorized_projection_keeps_only_a_contiguous_newest_suffix(
    retrieval_pool: ConnectionPool,
) -> None:
    thread = ThreadState(
        principal_id="alice",
        turn_records=(
            TurnRecord(
                turn_id="turn-old",
                user_message="old",
                standalone_question="old",
                assistant_message="old",
                outcome=TurnOutcome.ANSWERED,
            ),
            TurnRecord(
                turn_id="turn-new",
                user_message="new" * 500,
                standalone_question="new" * 500,
                assistant_message="new" * 500,
                outcome=TurnOutcome.ANSWERED,
            ),
        ),
    )
    snapshot = AuthorizationSnapshot(
        principal_id="alice", revision="auth_" + "0" * 64
    )

    projected = project_authorized_thread_context(
        retrieval_pool,
        snapshot,
        thread,
        current_user_message="current",
        token_limit=300,
    )

    assert projected.recent_turns == ()


def test_authorized_projection_rejects_an_oversized_summary(
    retrieval_pool: ConnectionPool,
) -> None:
    thread = ThreadState(
        principal_id="alice",
        summary=ConversationSummary(
            model_version="summary-model",
            items=(
                SummaryItem(
                    kind=SummaryItemKind.TOPIC,
                    text="x" * 1_000,
                    source_turn_ids=("turn-1",),
                ),
            ),
            covered_through_turn_id="turn-1",
        ),
    )
    snapshot = AuthorizationSnapshot(
        principal_id="alice", revision="auth_" + "0" * 64
    )

    with pytest.raises(ValueError, match="Summary exceeds"):
        project_authorized_thread_context(
            retrieval_pool,
            snapshot,
            thread,
            current_user_message="current",
            token_limit=100,
        )


def _long_thread() -> ThreadState:
    records = tuple(
        TurnRecord(
            turn_id=f"turn-{index}",
            user_message=f"Question {index}: " + "q" * 1_300,
            standalone_question=f"Question {index}: " + "q" * 1_300,
            assistant_message=f"Answer {index}: " + "a" * 1_300,
            outcome=TurnOutcome.ANSWERED,
        )
        for index in range(1, 4)
    )
    return ThreadState(principal_id="alice", turn_records=records)


def _finance_dependency(pool: ConnectionPool) -> CitationDependency:
    with pool.connection() as connection:
        row = connection.execute(
            """
            SELECT source_document.document_id,
                   indexed_chunk.source_revision
            FROM source_documents AS source_document
            JOIN indexed_chunks AS indexed_chunk
              ON indexed_chunk.knowledge_source = source_document.knowledge_source
             AND indexed_chunk.document_id = source_document.document_id
            WHERE source_document.knowledge_source = 'operational-runbooks'
              AND source_document.source_path = 'finance/settlement-impact.md'
            LIMIT 1
            """
        ).fetchone()
    assert row is not None
    return CitationDependency(
        knowledge_source=KnowledgeSource.OPERATIONAL_RUNBOOKS,
        document_id=str(row[0]),
        source_revision=str(row[1]),
        source_locator=SourceLocator(section_path=("Confidential impact",)),
    )


def _research_plan() -> ResearchPlan:
    return ResearchPlan(
        action=PlanAction.RETRIEVE,
        query="rollback failure",
        knowledge_sources=(KnowledgeSource.ENGINEERING_DOCS,),
    )


def _answer_draft(text: str) -> CitationDraft:
    return CitationDraft(
        disposition=DraftDisposition.FACTUAL,
        claims=(DraftClaim(text=text, citation_keys=("E1",)),),
    )


def _clarification_rewrite() -> ContextualRewrite:
    return ContextualRewrite(
        standalone_question="Did the rollback happen in the referenced place?",
        depends_on_history=True,
        referenced_turn_ids=("turn-1",),
        clarification_needed=True,
        ambiguity_reason="The place is ambiguous.",
    )


def _standalone_rewrite(question: str) -> ContextualRewrite:
    return ContextualRewrite(
        standalone_question=question,
        depends_on_history=False,
        referenced_turn_ids=(),
        clarification_needed=False,
    )


def _topic_summary(model_id: str, text: str) -> ConversationSummary:
    return ConversationSummary(
        model_version=model_id,
        items=(
            SummaryItem(
                kind=SummaryItemKind.TOPIC,
                text=text,
                source_turn_ids=("turn-1", "turn-2"),
            ),
        ),
        covered_through_turn_id="turn-2",
    )


def _full_turn_context(
    pool: ConnectionPool,
    research_model: BaseChatModel,
    answer_model: BaseChatModel,
) -> RuntimeContext:
    return RuntimeContext(
        principal_id="bob",
        database_pool=pool,
        research_model=research_model,
        answer_model=answer_model,
        embedder=FixtureEmbedder(),
        reranker=_rerank_all,
        retrieval_config=RetrievalConfig(
            embedding_model="deterministic-test-v1"
        ),
    )


def _rerank_all(
    _query: str, candidates: tuple[RerankCandidate, ...]
) -> RerankOutput:
    return RerankOutput(
        items=tuple(
            RerankItem(
                chunk_id=candidate.chunk_id,
                score=1 - index / (2 * len(candidates)),
            )
            for index, candidate in enumerate(candidates)
        )
    )


def _rerank_private_first(
    _query: str, candidates: tuple[RerankCandidate, ...]
) -> RerankOutput:
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            candidate.source_path != "payments/rollback-worker-compatibility.md",
            candidate.chunk_id,
        ),
    )
    return RerankOutput(
        items=tuple(
            RerankItem(
                chunk_id=candidate.chunk_id,
                score=1 - index / (2 * len(ordered)),
            )
            for index, candidate in enumerate(ordered)
        )
    )


def _grant_unrelated_document(
    pool: ConnectionPool, principal_id: str
) -> tuple[str, str]:
    with pool.connection() as connection:
        row = connection.execute(
            """
            SELECT knowledge_source, document_id
            FROM source_documents
            WHERE NOT source_document_is_authorized(
                %s, knowledge_source, document_id
            )
            ORDER BY knowledge_source, document_id
            LIMIT 1
            """,
            (principal_id,),
        ).fetchone()
        assert row is not None
        identity = (str(row[0]), str(row[1]))
        connection.execute(
            """
            INSERT INTO access_grants (
                knowledge_source, document_id, grant_type, principal_id
            ) VALUES (%s, %s, 'principal', %s)
            """,
            (*identity, principal_id),
        )
    return identity
