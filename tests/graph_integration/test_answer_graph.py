"""L3 verification for the compiled Answer subgraph."""

import json
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

from agentic_rag.agents.answering import VerificationDecision
from agentic_rag.authorization import (
    AuthorizationSnapshot,
    capture_authorization_snapshot,
)
from agentic_rag.citations import (
    CitationDraft,
    CitedAnswer,
    CitedAnswerStatus,
    DraftClaim,
    DraftDisposition,
)
from agentic_rag.corpus import KnowledgeSource
from agentic_rag.database import apply_migrations, open_database_pool
from agentic_rag.graph._answer import answer_graph
from agentic_rag.graph.state import AnswerGraphState
from agentic_rag.retrieval import (
    EvidenceSet,
    RerankCandidate,
    RerankItem,
    RerankOutput,
    RetrievalConfig,
    RetrievalRequest,
    assemble_evidence_set,
    retrieve,
)
from agentic_rag.retrieval.evidence import expand_evidence_set_context
from agentic_rag.runtime import RuntimeContext
from tests.support.reference_fixture import (
    FIXTURE_ROOT,
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


class DeterministicAnswerModel(BaseChatModel):
    """Schema-aware local model double; retrieval and graph stay real."""

    responses: list[Any] = Field(default_factory=list)
    requested_schemas: list[str] = Field(default_factory=list)
    structured_inputs: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "deterministic-answer"

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
                raise AssertionError("unexpected model call")
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
def empty_retrieval_state(database_pool: ConnectionPool) -> None:
    with database_pool.connection() as connection:
        connection.execute(
            """
            TRUNCATE access_grants, principal_group_memberships, groups,
                principals, sync_reports, indexed_chunks, source_documents,
                knowledge_sources CASCADE
            """
        )


@pytest.fixture
def answer_input(
    database_pool: ConnectionPool,
    retrieval_pool: ConnectionPool,
) -> tuple[AnswerGraphState, RuntimeContext]:
    evidence_set, snapshot = _retrieve_fixture_evidence(
        database_pool, retrieval_pool
    )
    state: AnswerGraphState = {
        "question": "Why does rollback_token cause undefined_column?",
        "evidence_set": evidence_set.model_dump(mode="json"),
        "authorization_snapshot": snapshot.model_dump(mode="json"),
        "repair_count": 0,
        "status": "pending",
    }
    return state, RuntimeContext(
        principal_id="bob",
        database_pool=retrieval_pool,
    )


def test_compiled_answer_graph_releases_only_verified_cited_answer(
    answer_input: tuple[AnswerGraphState, RuntimeContext],
) -> None:
    state, context = answer_input
    model = DeterministicAnswerModel(
        responses=[
            _draft("The old worker reads a removed rollback_token column.", "E1"),
            VerificationDecision(supported=True),
        ]
    )

    result = answer_graph.invoke(
        state,
        context=RuntimeContext(
            principal_id=context.principal_id,
            database_pool=context.database_pool,
            answer_model=model,
        ),
    )

    answer = CitedAnswer.model_validate(result["cited_answer"])
    assert answer.status is CitedAnswerStatus.FACTUAL
    assert answer.assistant_message.endswith("[E1]")
    assert [citation.key for citation in answer.citations] == ["E1"]
    assert result["repair_count"] == 0
    assert model.requested_schemas == [
        "CitationDraft",
        "VerificationDecision",
    ]


def test_invalid_citation_gets_one_repair_against_same_evidence(
    answer_input: tuple[AnswerGraphState, RuntimeContext],
) -> None:
    state, context = answer_input
    invalid_text = "UNVERIFIED_DRAFT_CANARY"
    repaired_text = "The migration removed rollback_token before worker rollout."
    model = DeterministicAnswerModel(
        responses=[
            _draft(invalid_text, "E8"),
            _draft(repaired_text, "E1"),
            VerificationDecision(supported=True),
        ]
    )

    result = answer_graph.invoke(
        state,
        context=RuntimeContext(
            principal_id=context.principal_id,
            database_pool=context.database_pool,
            answer_model=model,
        ),
    )

    answer = CitedAnswer.model_validate(result["cited_answer"])
    assert answer.status is CitedAnswerStatus.FACTUAL
    assert repaired_text in answer.assistant_message
    assert invalid_text not in answer.assistant_message
    assert result["repair_count"] == 1
    assert model.requested_schemas == [
        "CitationDraft",
        "CitationDraft",
        "VerificationDecision",
    ]


def test_second_invalid_draft_ends_in_safe_incomplete_projection(
    answer_input: tuple[AnswerGraphState, RuntimeContext],
) -> None:
    state, context = answer_input
    invalid_text = "PERSISTENT_UNVERIFIED_CANARY"
    model = DeterministicAnswerModel(
        responses=[_draft(invalid_text, "E8"), _draft(invalid_text, "E8")]
    )

    result = answer_graph.invoke(
        state,
        context=RuntimeContext(
            principal_id=context.principal_id,
            database_pool=context.database_pool,
            answer_model=model,
        ),
    )

    answer = CitedAnswer.model_validate(result["cited_answer"])
    assert answer.status is CitedAnswerStatus.INCOMPLETE
    assert answer.citations == ()
    assert invalid_text not in answer.assistant_message
    assert result["repair_count"] == 1
    assert model.requested_schemas == ["CitationDraft", "CitationDraft"]


def test_semantically_unsupported_claim_gets_one_repair(
    answer_input: tuple[AnswerGraphState, RuntimeContext],
) -> None:
    state, context = answer_input
    unsupported_text = "The migration failed because of a network outage."
    repaired_text = "The old worker still queried the removed column."
    model = DeterministicAnswerModel(
        responses=[
            _draft(unsupported_text, "E1"),
            VerificationDecision(
                supported=False, unsupported_claim_indexes=(0,)
            ),
            _draft(repaired_text, "E1"),
            VerificationDecision(supported=True),
        ]
    )

    result = answer_graph.invoke(
        state,
        context=RuntimeContext(
            principal_id=context.principal_id,
            database_pool=context.database_pool,
            answer_model=model,
        ),
    )

    answer = CitedAnswer.model_validate(result["cited_answer"])
    assert answer.status is CitedAnswerStatus.FACTUAL
    assert repaired_text in answer.assistant_message
    assert unsupported_text not in answer.assistant_message
    assert result["repair_count"] == 1
    assert model.requested_schemas == [
        "CitationDraft",
        "VerificationDecision",
        "CitationDraft",
        "VerificationDecision",
    ]
    repair_messages = model.structured_inputs[2]
    repair_payload = json.loads(repair_messages[1][1])
    assert repair_payload["repair_feedback"] == (
        "unsupported_claim_indexes:0"
    )


def test_runtime_principal_mismatch_fails_before_generation(
    answer_input: tuple[AnswerGraphState, RuntimeContext],
) -> None:
    state, context = answer_input
    model = DeterministicAnswerModel(
        responses=[_draft("This must never be generated.", "E1")]
    )

    result = answer_graph.invoke(
        state,
        context=RuntimeContext(
            principal_id="alice",
            database_pool=context.database_pool,
            answer_model=model,
        ),
    )

    answer = CitedAnswer.model_validate(result["cited_answer"])
    assert answer.status is CitedAnswerStatus.INCOMPLETE
    assert result["failure_reason"] == "authorization_context_mismatch"
    assert model.requested_schemas == []


def test_grant_revocation_during_answering_blocks_final_release(
    answer_input: tuple[AnswerGraphState, RuntimeContext],
    database_pool: ConnectionPool,
) -> None:
    state, context = answer_input

    def revoke_then_verify() -> VerificationDecision:
        with database_pool.connection() as connection:
            connection.execute("DELETE FROM access_grants")
        return VerificationDecision(supported=True)

    model = DeterministicAnswerModel(
        responses=[
            _draft("AUTHORIZED_THEN_REVOKED_CANARY", "E1"),
            revoke_then_verify,
        ]
    )

    result = answer_graph.invoke(
        state,
        context=RuntimeContext(
            principal_id=context.principal_id,
            database_pool=context.database_pool,
            answer_model=model,
        ),
    )

    answer = CitedAnswer.model_validate(result["cited_answer"])
    assert answer.status is CitedAnswerStatus.INCOMPLETE
    assert "AUTHORIZED_THEN_REVOKED_CANARY" not in answer.assistant_message
    assert result["failure_reason"] == "authorization_changed_before_release"


def test_out_of_range_verifier_index_fails_without_releasing_draft(
    answer_input: tuple[AnswerGraphState, RuntimeContext],
) -> None:
    state, context = answer_input
    draft_text = "MALFORMED_VERIFIER_CANARY"
    model = DeterministicAnswerModel(
        responses=[
            _draft(draft_text, "E1"),
            VerificationDecision(
                supported=False, unsupported_claim_indexes=(999,)
            ),
        ]
    )

    result = answer_graph.invoke(
        state,
        context=RuntimeContext(
            principal_id=context.principal_id,
            database_pool=context.database_pool,
            answer_model=model,
        ),
    )

    answer = CitedAnswer.model_validate(result["cited_answer"])
    assert answer.status is CitedAnswerStatus.INCOMPLETE
    assert draft_text not in answer.assistant_message
    assert result["failure_reason"] == "verification_failed"
    assert result["repair_count"] == 0


def _draft(text: str, key: str) -> CitationDraft:
    return CitationDraft(
        disposition=DraftDisposition.FACTUAL,
        claims=(DraftClaim(text=text, citation_keys=(key,)),),
    )


def _retrieve_fixture_evidence(
    database_pool: ConnectionPool,
    retrieval_pool: ConnectionPool,
) -> tuple[EvidenceSet, AuthorizationSnapshot]:
    labels = json.loads(
        (FIXTURE_ROOT / "trusted/retrieval-labels.json").read_text(
            encoding="utf-8"
        )
    )
    case = labels["cases"]["rollback-incompatibility"]
    expected = case["expected_evidence"]
    load_reference_snapshot(database_pool, case["snapshot"])
    snapshot = capture_authorization_snapshot(
        retrieval_pool, case["principal_id"]
    )
    with database_pool.connection() as connection:
        row = connection.execute(
            """
            SELECT indexed_chunk.chunk_id
            FROM indexed_chunks AS indexed_chunk
            JOIN source_documents AS source_document
              ON source_document.knowledge_source = indexed_chunk.knowledge_source
             AND source_document.document_id = indexed_chunk.document_id
            WHERE indexed_chunk.knowledge_source = %(knowledge_source)s
              AND source_document.source_path = %(source_path)s
              AND indexed_chunk.heading_path = %(section_path)s
            """,
            {
                "knowledge_source": expected["knowledge_source"],
                "source_path": expected["source_path"],
                "section_path": expected["section_path"],
            },
        ).fetchone()
        assert row is not None
        expected_chunk_id = str(row[0])

    def labeled_reranker(
        _query: str, candidates: tuple[RerankCandidate, ...]
    ) -> RerankOutput:
        ordered = sorted(
            candidates,
            key=lambda candidate: (
                candidate.chunk_id != expected_chunk_id,
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

    config = RetrievalConfig(embedding_model="deterministic-test-v1")
    result = retrieve(
        retrieval_pool,
        snapshot,
        RetrievalRequest(
            request_id="answer-graph-evidence",
            query=case["query"],
            knowledge_sources=tuple(
                KnowledgeSource(value) for value in case["knowledge_sources"]
            ),
            dense_candidate_limit=50,
            lexical_candidate_limit=50,
            result_limit=case["result_limit"],
        ),
        config,
        FixtureEmbedder(),
        labeled_reranker,
    )
    evidence_set = expand_evidence_set_context(
        retrieval_pool,
        snapshot,
        assemble_evidence_set((result,), config),
        config,
    )
    assert isinstance(evidence_set, EvidenceSet)
    assert evidence_set.complete
    assert evidence_set.evidence_items[0].chunk_id == expected_chunk_id
    return evidence_set, snapshot
