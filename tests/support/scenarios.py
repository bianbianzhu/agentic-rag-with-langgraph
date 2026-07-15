"""Strict loader for deterministic Reference System scenario manifests."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agentic_rag.conversation import TurnOutcome
from tests.support.reference_fixture import FIXTURE_ROOT


class ClaimExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    statement: str = Field(min_length=1)


class TurnExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: TurnOutcome
    terminal_reason: str | None
    evidence_aliases: tuple[str, ...]
    untrusted_evidence_aliases: tuple[str, ...] = ()
    forbidden_evidence_aliases: tuple[str, ...]
    required_claims: tuple[ClaimExpectation, ...]
    forbidden_claims: tuple[ClaimExpectation, ...]
    retrieval_requests: int = Field(ge=0, le=2)
    research_iterations: int = Field(ge=0, le=1)
    answer_repairs: int = Field(ge=0, le=1)
    authorization_restarts: int = Field(default=0, ge=0, le=1)


class ScenarioTurn(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    turn_id: str = Field(min_length=1)
    principal_id: str | None = None
    user_message: str = Field(min_length=1)
    expected: TurnExpectation


class FixtureSetup(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    snapshot: Literal["s1-baseline"]
    overlay: Literal["document-injection", "poisoned-fact"] | None


class ReferenceScenario(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal["reference-scenario-v1"]
    id: str = Field(min_length=1)
    fixture_setup: FixtureSetup
    principal_id: str = Field(min_length=1)
    authorization_event: (
        Literal["revoke_alice_finance_after_first_verification"] | None
    ) = None
    known_limitation: bool = False
    turns: tuple[ScenarioTurn, ...] = Field(min_length=1, max_length=2)


def load_reference_scenarios() -> tuple[ReferenceScenario, ...]:
    """Load every canonical manifest in stable scenario-ID order."""

    scenarios = []
    for path in (FIXTURE_ROOT / "scenarios").glob("*.json"):
        scenario = ReferenceScenario.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        if path.stem != scenario.id:
            raise ValueError("scenario filename and ID disagree")
        scenarios.append(scenario)
    return tuple(sorted(scenarios, key=lambda scenario: scenario.id))
