"""Live OpenAI target harness over the compiled graph and real PostgreSQL."""

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, Iterator, cast
from uuid import uuid4

import langsmith as ls
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from psycopg_pool import ConnectionPool

from agentic_rag.citations import CitedAnswer
from agentic_rag.conversation import (
    ThreadState,
    TurnOutcome,
    active_thread_memory_tokens,
)
from agentic_rag.database import apply_migrations, open_database_pool
from agentic_rag.graph._turn import build_turn_graph
from agentic_rag.graph.state import CurrentTurnWork, GraphState
from agentic_rag.retrieval import Reranker, RetrievalConfig
from agentic_rag.retrieval.reranking import build_reranker
from agentic_rag.runtime import RuntimeContext
from evals.contracts import (
    ActualCitation,
    ActualCounters,
    ActualEvidence,
    ActualTurn,
    TargetCommand,
    TargetOutput,
)
from evals.config import (
    CHAT_MODEL_ID,
    EMBEDDING_MODEL_ID,
    MODEL_TEMPERATURE,
    RERANKER_MODEL_ID,
)
from evals.dataset import validate_target_command
from evals.fixtures import prepare_live_fixture, resolve_all_evidence_aliases


REPOSITORY_ROOT = Path(__file__).parents[1]
DEFAULT_DATABASE_URL = (
    "postgresql://agentic_rag@127.0.0.1:55432/agentic_rag"
)
DEFAULT_RETRIEVAL_DATABASE_URL = (
    "postgresql://agentic_rag_retrieval@127.0.0.1:55432/agentic_rag"
)


@dataclass
class LiveTarget:
    write_pool: ConnectionPool
    retrieval_pool: ConnectionPool
    chat_model: BaseChatModel
    embedder: Embeddings
    reranker: Reranker

    def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        command = validate_target_command(inputs)
        prepare_live_fixture(
            self.write_pool,
            snapshot=command.fixture_setup.snapshot,
            overlay=command.fixture_setup.overlay,
            embedder=self.embedder,
        )
        aliases_by_chunk = resolve_all_evidence_aliases(self.write_pool)
        graph = build_turn_graph(
            InMemorySaver(),
            interrupt_after=("run_answer_subgraph",),
        )
        config: RunnableConfig = {
            "configurable": {
                "thread_id": f"evaluation-{command.scenario_id}-{uuid4()}"
            }
        }
        results = []
        authorization_revoked = False
        for index, turn in enumerate(command.turns):
            runtime = self._runtime(command.fixture_setup.principal_id)
            thread_for_budget = ThreadState()
            if index:
                thread_for_budget = ThreadState.model_validate(
                    graph.get_state(config).values["thread"]
                )
            thread_token_count = active_thread_memory_tokens(
                thread_for_budget,
                turn.user_message,
            )
            input_state: GraphState = {
                "thread": ThreadState(),
                "current_turn": {
                    "turn_id": turn.turn_id,
                    "user_message": turn.user_message,
                },
            }
            if index:
                input_state = cast(
                    GraphState,
                    {"current_turn": input_state["current_turn"]},
                )
            with ls.trace(
                "evaluation_turn",
                inputs={
                    "scenario_id": command.scenario_id,
                    "turn_id": turn.turn_id,
                },
            ) as turn_trace:
                result, authorization_revoked = self._run_turn(
                    graph,
                    input_state,
                    config,
                    runtime,
                    command,
                    aliases_by_chunk,
                    authorization_revoked,
                    thread_token_count,
                )
                turn_trace.end(
                    outputs={
                        "turn_id": turn.turn_id,
                        "outcome": result.outcome.value,
                    }
                )
            results.append(result)
        return TargetOutput(
            scenario_id=command.scenario_id,
            turns=tuple(results),
        ).model_dump(mode="json")

    def _runtime(self, principal_id: str) -> RuntimeContext:
        return RuntimeContext(
            principal_id=principal_id,
            database_pool=self.retrieval_pool,
            research_model=self.chat_model,
            answer_model=self.chat_model,
            contextualization_model=self.chat_model,
            embedder=self.embedder,
            reranker=self.reranker,
            retrieval_config=RetrievalConfig(
                embedding_model=EMBEDDING_MODEL_ID
            ),
        )

    def _run_turn(
        self,
        graph: Any,
        input_state: GraphState,
        config: RunnableConfig,
        runtime: RuntimeContext,
        command: TargetCommand,
        aliases_by_chunk: dict[str, str],
        authorization_revoked: bool,
        thread_token_count: int,
    ) -> tuple[ActualTurn, bool]:
        started_at = monotonic()
        first_progress: float | None = None
        trajectory: list[str] = []
        selected_sources: set[str] = set()
        latest_authorizing: CurrentTurnWork | None = None
        next_input: GraphState | None = input_state
        while True:
            segment_nodes: list[str] = []
            for _, update in graph.stream(
                next_input,
                config=config,
                context=runtime,
                stream_mode="updates",
                subgraphs=True,
            ):
                if first_progress is None:
                    first_progress = monotonic() - started_at
                for node, payload in update.items():
                    trajectory.append(node)
                    segment_nodes.append(node)
                    if isinstance(payload, dict):
                        selected_sources.update(
                            str(value)
                            for value in payload.get("knowledge_sources", ())
                        )
                        raw_work = payload.get("current_turn")
                        if raw_work is not None:
                            work = CurrentTurnWork.model_validate(raw_work)
                            if work.stage == "authorizing":
                                latest_authorizing = work
            state = graph.get_state(config).values
            raw_current = state.get("current_turn")
            if raw_current is None:
                break
            current = CurrentTurnWork.model_validate(raw_current)
            if (
                command.fixture_setup.authorization_event
                == "revoke_alice_finance_after_first_verification"
                and not authorization_revoked
                and "run_answer_subgraph" in segment_nodes
                and current.authorization_restarts == 0
            ):
                self._revoke_alice_finance()
                authorization_revoked = True
            next_input = None
        if first_progress is None or latest_authorizing is None:
            raise ValueError("evaluation Turn produced no observable result")
        thread = ThreadState.model_validate(state["thread"])
        record = thread.turn_records[-1]
        return (
            _project_actual_turn(
                record.turn_id,
                record.assistant_message,
                record.outcome,
                record.terminal_reason,
                latest_authorizing,
                aliases_by_chunk,
                selected_sources,
                trajectory,
                first_progress,
                monotonic() - started_at,
                thread_token_count,
            ),
            authorization_revoked,
        )

    def _revoke_alice_finance(self) -> None:
        with self.write_pool.connection() as connection:
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


@contextmanager
def open_live_target() -> Iterator[LiveTarget]:
    """Construct live dependencies from process secrets and close both pools."""

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for L4 evaluation")
    write_pool = open_database_pool(
        os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    )
    apply_migrations(write_pool, REPOSITORY_ROOT / "migrations")
    retrieval_pool = open_database_pool(
        os.environ.get(
            "RETRIEVAL_DATABASE_URL",
            DEFAULT_RETRIEVAL_DATABASE_URL,
        )
    )
    try:
        chat_model = init_chat_model(
            CHAT_MODEL_ID,
            temperature=MODEL_TEMPERATURE,
        )
        reranker_model = init_chat_model(
            RERANKER_MODEL_ID,
            temperature=MODEL_TEMPERATURE,
        )
        embedder = init_embeddings(EMBEDDING_MODEL_ID)
        yield LiveTarget(
            write_pool=write_pool,
            retrieval_pool=retrieval_pool,
            chat_model=chat_model,
            embedder=embedder,
            reranker=build_reranker(reranker_model),
        )
    finally:
        retrieval_pool.close()
        write_pool.close()


def _project_actual_turn(
    turn_id: str,
    assistant_message: str,
    outcome: TurnOutcome,
    terminal_reason: str | None,
    work: CurrentTurnWork,
    aliases_by_chunk: dict[str, str],
    selected_sources: set[str],
    trajectory: list[str],
    first_progress: float,
    latency: float,
    thread_token_count: int,
) -> ActualTurn:
    if work.standalone_question is None:
        raise ValueError("evaluation Turn has no Standalone Question")
    evidence_items = (
        work.evidence_set.evidence_items
        if work.evidence_set is not None
        else ()
    )
    evidence = tuple(
        ActualEvidence(
            alias=aliases_by_chunk.get(
                item.chunk_id, f"unlabeled:{item.chunk_id}"
            ),
            chunk_id=item.chunk_id,
            document_id=item.document_id,
            knowledge_source=item.knowledge_source.value,
            source_revision=item.source_revision,
            processing_revision=item.processing_revision,
            corpus_revision=item.corpus_revision,
            source_path=item.source_path,
            title=item.title,
            source_locator=item.source_locator,
            content="\n\n".join(
                value
                for value in (item.chunk_text, item.context_text)
                if value
            ),
        )
        for item in evidence_items
    )
    cited = work.cited_answer
    cited_answer = (
        CitedAnswer.model_validate(cited) if cited is not None else None
    )
    citations = tuple(
        ActualCitation(
            key=mapping.key,
            alias=aliases_by_chunk.get(
                mapping.chunk_id, f"unlabeled:{mapping.chunk_id}"
            ),
            chunk_id=mapping.chunk_id,
            document_id=mapping.document_id,
            source_revision=mapping.source_revision,
            source_path=mapping.source_path,
            title=mapping.title,
            source_locator=mapping.source_locator,
        )
        for mapping in (cited_answer.citations if cited_answer else ())
    )
    events = []
    if work.authorization_restarts:
        events.append("authorization_restarted")
    if terminal_reason == "insufficient_evidence":
        events.append("insufficient_authorized_evidence")
    return ActualTurn(
        turn_id=turn_id,
        assistant_message=assistant_message,
        standalone_question=work.standalone_question,
        outcome=outcome,
        terminal_reason=terminal_reason,
        final_evidence=evidence,
        citations=citations,
        counters=ActualCounters(
            model_calls=work.model_calls,
            retrieval_requests=work.retrieval_requests,
            research_iterations=work.research_iterations,
            answer_repairs=work.answer_repairs,
            authorization_restarts=work.authorization_restarts,
        ),
        selected_knowledge_sources=tuple(sorted(selected_sources)),
        trajectory_nodes=tuple(trajectory),
        security_events=tuple(events),
        evidence_token_count=(
            work.evidence_set.token_count
            if work.evidence_set is not None
            else 0
        ),
        thread_token_count=thread_token_count,
        answer_token_count=max(1, len(assistant_message) // 4),
        first_progress_seconds=first_progress,
        latency_seconds=latency,
    )
