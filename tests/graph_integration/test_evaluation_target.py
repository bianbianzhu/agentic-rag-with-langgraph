"""L3 evaluation-target projection over the real graph and PostgreSQL."""

import os
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from psycopg_pool import ConnectionPool
from pydantic import Field

from agentic_rag.agents.answering import VerificationDecision
from agentic_rag.agents.research import (
    EvidenceAssessment,
    PlanAction,
    ResearchPlan,
)
from agentic_rag.citations import CitationDraft, DraftClaim, DraftDisposition
from agentic_rag.corpus import KnowledgeSource
from agentic_rag.database import apply_migrations, open_database_pool
from agentic_rag.retrieval import (
    RerankCandidate,
    RerankItem,
    RerankOutput,
)
from evals.contracts import TargetOutput
from evals.dataset import build_golden_examples
from evals.target import LiveTarget
from tests.support.reference_fixture import FixtureEmbedder


REPOSITORY_ROOT = Path(__file__).parents[2]
DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://agentic_rag@127.0.0.1:55432/agentic_rag",
)
RETRIEVAL_DATABASE_URL = os.environ.get(
    "TEST_RETRIEVAL_DATABASE_URL",
    "postgresql://agentic_rag_retrieval@127.0.0.1:55432/agentic_rag",
)


class ScriptedModel(BaseChatModel):
    responses: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "evaluation-target-script"

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
        def next_response(_: object) -> Any:
            if not self.responses:
                raise AssertionError("unexpected model call")
            return self.responses.pop(0)

        return RunnableLambda(next_response)


def test_evaluation_target_returns_structured_aliases_and_trajectory() -> None:
    write_pool = open_database_pool(DATABASE_URL)
    retrieval_pool = open_database_pool(RETRIEVAL_DATABASE_URL)
    apply_migrations(write_pool, REPOSITORY_ROOT / "migrations")
    model = ScriptedModel(
        responses=[
            ResearchPlan(
                action=PlanAction.RETRIEVE,
                query="production rollback undefined_column",
                knowledge_sources=(KnowledgeSource.ENGINEERING_DOCS,),
            ),
            EvidenceAssessment(sufficient=True),
            CitationDraft(
                disposition=DraftDisposition.FACTUAL,
                claims=(
                    DraftClaim(
                        text=(
                            "Production worker v1.8 queried rollback_token, "
                            "received undefined_column, and failed before "
                            "sending a provider request."
                        ),
                        citation_keys=("E1",),
                    ),
                ),
            ),
            VerificationDecision(supported=True),
        ]
    )
    target = LiveTarget(
        write_pool=write_pool,
        retrieval_pool=retrieval_pool,
        chat_model=model,
        embedder=FixtureEmbedder(),
        reranker=_rerank_rollback_first,
    )
    command = next(
        example.inputs
        for example in build_golden_examples()
        if example.inputs.scenario_id == "happy"
    )

    try:
        output = TargetOutput.model_validate(
            target(command.model_dump(mode="json"))
        )
    finally:
        retrieval_pool.close()
        write_pool.close()

    assert not model.responses
    assert output.errors == ()
    assert output.turns[0].outcome.value == "answered"
    assert output.turns[0].final_evidence[0].alias == (
        "D2#rollback-incompatibility"
    )
    assert output.turns[0].citations[0].alias == (
        "D2#rollback-incompatibility"
    )
    assert "engineering-docs" in output.turns[0].selected_knowledge_sources
    assert "run_answer_subgraph" in output.turns[0].trajectory_nodes


def _rerank_rollback_first(
    query: str,
    candidates: tuple[RerankCandidate, ...],
) -> RerankOutput:
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            "undefined_column" not in candidate.content,
            candidate.chunk_id,
        ),
    )
    return RerankOutput(
        items=tuple(
            RerankItem(
                chunk_id=candidate.chunk_id,
                score=1.0 if index == 0 else 0.4,
            )
            for index, candidate in enumerate(ranked)
        )
    )
