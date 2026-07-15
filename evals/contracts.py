"""Evaluation-only dataset and release contracts."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from agentic_rag.conversation import TurnOutcome
from agentic_rag.corpus.models import SourceLocator


class EvaluationClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    fact_id: str = Field(min_length=1)
    statement: str = Field(min_length=1)


class FixtureCommand(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    snapshot: Literal["s1-baseline"]
    overlay: Literal["document-injection", "poisoned-fact"] | None
    principal_id: str = Field(min_length=1)
    authorization_event: (
        Literal["revoke_alice_finance_after_first_verification"] | None
    ) = None


class EvaluationTurnInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    turn_id: str = Field(min_length=1)
    user_message: str = Field(min_length=1)


class TargetCommand(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["target-command-v1"] = "target-command-v1"
    scenario_id: str = Field(min_length=1)
    fixture_setup: FixtureCommand
    turns: tuple[EvaluationTurnInput, ...] = Field(min_length=1, max_length=2)


class CounterBound(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum: int = Field(default=0, ge=0)
    maximum: int = Field(ge=0)


class ReferenceTurn(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    expected_outcome: TurnOutcome
    expected_terminal_reason: str | None
    reference_answer: str = Field(min_length=1)
    expected_evidence_aliases: tuple[str, ...]
    forbidden_evidence_aliases: tuple[str, ...]
    required_claims: tuple[EvaluationClaim, ...]
    forbidden_claims: tuple[EvaluationClaim, ...]
    forbidden_disclosures: tuple[str, ...]
    required_knowledge_sources: tuple[str, ...]
    forbidden_knowledge_sources: tuple[str, ...]
    required_trajectory_nodes: tuple[str, ...]
    forbidden_trajectory_nodes: tuple[str, ...]
    counter_bounds: dict[str, CounterBound]


class ReferenceOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["reference-output-v1"] = "reference-output-v1"
    turns: tuple[ReferenceTurn, ...] = Field(min_length=1, max_length=2)


class ExampleMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = Field(min_length=1)
    risk_domain: str = Field(min_length=1)
    conversation_shape: Literal["single_turn", "multi_turn"]
    fixture_version: Literal["reference-system-v1"] = "reference-system-v1"
    known_limitation: bool
    applicable_evaluators: tuple[str, ...]


class GoldenExample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    example_id: UUID
    split: Literal["release", "diagnostic"]
    inputs: TargetCommand
    reference_outputs: ReferenceOutput
    metadata: ExampleMetadata


class DatasetCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str
    dataset_name: str
    dataset_version: datetime
    example_count: int


class DatasetPublication(DatasetCandidate):
    dataset_tag: str


class ActualEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    alias: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    knowledge_source: str = Field(min_length=1)
    source_revision: str = Field(min_length=1)
    processing_revision: str = Field(min_length=1)
    corpus_revision: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source_locator: SourceLocator
    content: str = Field(min_length=1)


class ActualCitation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(min_length=1)
    alias: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    source_revision: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source_locator: SourceLocator


class ActualCounters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_calls: int = Field(ge=0)
    retrieval_requests: int = Field(ge=0)
    research_iterations: int = Field(ge=0)
    answer_repairs: int = Field(ge=0)
    authorization_restarts: int = Field(ge=0)


class ActualTurn(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    turn_id: str = Field(min_length=1)
    assistant_message: str = Field(min_length=1)
    standalone_question: str = Field(min_length=1)
    outcome: TurnOutcome
    terminal_reason: str | None
    final_evidence: tuple[ActualEvidence, ...]
    citations: tuple[ActualCitation, ...]
    counters: ActualCounters
    selected_knowledge_sources: tuple[str, ...]
    trajectory_nodes: tuple[str, ...]
    security_events: tuple[str, ...]
    evidence_token_count: int = Field(ge=0)
    thread_token_count: int = Field(ge=0)
    answer_token_count: int = Field(ge=0)
    first_progress_seconds: float = Field(ge=0)
    latency_seconds: float = Field(ge=0)


class TargetOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["target-output-v1"] = "target-output-v1"
    scenario_id: str = Field(min_length=1)
    turns: tuple[ActualTurn, ...] = Field(min_length=1, max_length=2)
    errors: tuple[str, ...] = ()
