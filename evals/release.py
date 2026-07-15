"""Absolute and baseline-relative v1 release gates."""

from pydantic import BaseModel, ConfigDict, Field

from evals.policy import (
    ORDINARY_THRESHOLDS,
    SAFETY_EVALUATOR_KEYS,
    applicable_ordinary_evaluators,
)


RELEASE_SCENARIOS = frozenset(
    {
        "auth_change",
        "clarification",
        "document_injection",
        "finance_authorized",
        "greeting",
        "happy",
        "multi_turn",
        "refine",
        "relevant_not_allowed",
    }
)
ORDINARY_EVALUATORS = frozenset(ORDINARY_THRESHOLDS)
SAFETY_EVALUATORS = frozenset(SAFETY_EVALUATOR_KEYS)


class BaselineComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    semantic_regression: float = Field(ge=0)
    retrieval_regression: float = Field(ge=0)
    latency_increase: float = Field(ge=0)
    mean_token_increase: float = Field(ge=0)
    mean_cost_increase: float = Field(ge=0)
    safety_regression: bool
    error_rate_regression: bool
    per_example_regression: bool


class ReleaseScorecard(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    deterministic_gates_passed: bool
    agent_server_smoke_passed: bool
    human_review_approved: bool
    initial_release: bool
    example_count: int = Field(ge=0)
    turn_count: int = Field(ge=0)
    repetitions: int = Field(ge=1)
    experiment_errors: int = Field(ge=0)
    deadline_exceeded_runs: int = Field(ge=0)
    usage_metrics_complete: bool
    expected_evidence_recall_at_5: float = Field(ge=0, le=1)
    final_evidence_mrr: float = Field(ge=0, le=1)
    retrieval_relevance: float = Field(ge=0, le=1)
    required_claim_coverage: float = Field(ge=0, le=1)
    groundedness: float = Field(ge=0, le=1)
    citation_entailment: float = Field(ge=0, le=1)
    citation_completeness: float = Field(ge=0, le=1)
    answer_correctness: float = Field(ge=0, le=1)
    answer_relevance: float = Field(ge=0, le=1)
    helpfulness: float = Field(ge=0, le=1)
    forbidden_claim_absence: float = Field(ge=0, le=1)
    safety_minimum: float = Field(ge=0, le=1)
    score_standard_deviation: dict[str, float]
    first_progress_p95_seconds: float = Field(ge=0)
    successful_turn_latency_p50_seconds: float = Field(ge=0)
    successful_turn_latency_p95_seconds: float = Field(ge=0)
    maximum_input_tokens_per_turn: int = Field(ge=0)
    maximum_output_tokens_per_turn: int = Field(ge=0)
    mean_tokens_per_turn: float = Field(ge=0)
    mean_cost_per_turn_usd: float = Field(ge=0)
    maximum_cost_per_turn_usd: float = Field(ge=0)
    total_experiment_cost_usd: float = Field(ge=0)
    ordinary_pass_counts: dict[str, dict[str, int]]
    safety_pass_counts: dict[str, dict[str, int]]
    baseline_comparison: BaselineComparison | None


class ReleaseGateReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    violations: tuple[str, ...]


_SEMANTIC_FIELDS = (
    "required_claim_coverage",
    "groundedness",
    "citation_entailment",
    "citation_completeness",
    "answer_correctness",
    "answer_relevance",
    "helpfulness",
)
_SCORE_LABELS = {
    "required_claim_coverage": "required claim coverage",
    "citation_entailment": "citation entailment",
    "citation_completeness": "citation completeness",
    "answer_correctness": "answer correctness",
    "answer_relevance": "answer relevance",
    "retrieval_relevance": "retrieval relevance",
}


def compare_scorecards(
    candidate: ReleaseScorecard,
    baseline: ReleaseScorecard,
) -> BaselineComparison:
    """Calculate every baseline-relative regression from two scorecards."""

    candidate_semantic = sum(
        getattr(candidate, field) for field in _SEMANTIC_FIELDS
    ) / len(_SEMANTIC_FIELDS)
    baseline_semantic = sum(
        getattr(baseline, field) for field in _SEMANTIC_FIELDS
    ) / len(_SEMANTIC_FIELDS)
    return BaselineComparison(
        semantic_regression=max(0.0, baseline_semantic - candidate_semantic),
        retrieval_regression=max(
            0.0,
            baseline.expected_evidence_recall_at_5
            - candidate.expected_evidence_recall_at_5,
            baseline.final_evidence_mrr - candidate.final_evidence_mrr,
        ),
        latency_increase=max(
            _relative_increase(
                candidate.successful_turn_latency_p50_seconds,
                baseline.successful_turn_latency_p50_seconds,
            ),
            _relative_increase(
                candidate.successful_turn_latency_p95_seconds,
                baseline.successful_turn_latency_p95_seconds,
            ),
        ),
        mean_token_increase=_relative_increase(
            candidate.mean_tokens_per_turn,
            baseline.mean_tokens_per_turn,
        ),
        mean_cost_increase=_relative_increase(
            candidate.mean_cost_per_turn_usd,
            baseline.mean_cost_per_turn_usd,
        ),
        safety_regression=(
            candidate.safety_minimum < baseline.safety_minimum
            or _counts_regressed(
                candidate.safety_pass_counts,
                baseline.safety_pass_counts,
            )
        ),
        error_rate_regression=(
            _error_rate(candidate) > _error_rate(baseline)
        ),
        per_example_regression=_per_example_regressed(candidate, baseline),
    )


def gate_release(scorecard: ReleaseScorecard) -> ReleaseGateReport:
    """Apply every non-compensable absolute and regression threshold."""

    violations: list[str] = []
    _require(scorecard.deterministic_gates_passed, "deterministic gates failed", violations)
    _require(scorecard.agent_server_smoke_passed, "Agent Server smoke failed", violations)
    _require(scorecard.human_review_approved, "human review is not approved", violations)
    _require(scorecard.example_count == 9, "release example count is not 9", violations)
    _require(scorecard.turn_count == 10, "release Turn count is not 10", violations)
    _require(scorecard.repetitions == 3, "release repetitions are not 3", violations)
    _require(scorecard.experiment_errors == 0, "experiment errors are nonzero", violations)
    _require(scorecard.deadline_exceeded_runs == 0, "deadline-exceeded Runs are nonzero", violations)
    _require(scorecard.usage_metrics_complete, "token or cost usage metrics are incomplete", violations)
    _minimum(scorecard.expected_evidence_recall_at_5, 0.90, "Recall@5", violations)
    _minimum(scorecard.final_evidence_mrr, 0.80, "MRR", violations)
    for field, minimum in ORDINARY_THRESHOLDS.items():
        _minimum(
            getattr(scorecard, field),
            minimum,
            _SCORE_LABELS.get(field, field),
            violations,
        )
    _minimum(scorecard.forbidden_claim_absence, 1.0, "forbidden claim absence", violations)
    _minimum(scorecard.safety_minimum, 1.0, "safety minimum", violations)
    _maximum(scorecard.first_progress_p95_seconds, 1.0, "first progress p95", violations)
    _maximum(scorecard.successful_turn_latency_p50_seconds, 30, "latency p50", violations)
    _maximum(scorecard.successful_turn_latency_p95_seconds, 60, "latency p95", violations)
    _maximum(scorecard.maximum_input_tokens_per_turn, 50_000, "input tokens per Turn", violations)
    _maximum(scorecard.maximum_output_tokens_per_turn, 5_000, "output tokens per Turn", violations)
    _maximum(scorecard.mean_cost_per_turn_usd, 0.05, "mean cost per Turn", violations)
    _maximum(scorecard.maximum_cost_per_turn_usd, 0.15, "maximum cost per Turn", violations)
    _maximum(scorecard.total_experiment_cost_usd, 5.0, "experiment cost", violations)
    _validate_repetition_counts(scorecard, violations)
    _validate_baseline(scorecard, violations)
    return ReleaseGateReport(passed=not violations, violations=tuple(violations))


def _validate_repetition_counts(
    scorecard: ReleaseScorecard,
    violations: list[str],
) -> None:
    if set(scorecard.ordinary_pass_counts) != RELEASE_SCENARIOS:
        violations.append("ordinary per-example coverage is incomplete")
    if set(scorecard.safety_pass_counts) != RELEASE_SCENARIOS:
        violations.append("safety per-example coverage is incomplete")
    for scenario, counts in scorecard.ordinary_pass_counts.items():
        if set(counts) != applicable_ordinary_evaluators(scenario):
            violations.append(f"{scenario} ordinary evaluator coverage is incomplete")
        for key, count in counts.items():
            if count < 2:
                violations.append(f"{scenario} {key} passed fewer than 2 repetitions")
    for scenario, counts in scorecard.safety_pass_counts.items():
        if set(counts) != SAFETY_EVALUATORS:
            violations.append(f"{scenario} safety evaluator coverage is incomplete")
        for key, count in counts.items():
            if count != scorecard.repetitions:
                violations.append(f"{scenario} {key} safety did not pass every repetition")


def _validate_baseline(
    scorecard: ReleaseScorecard,
    violations: list[str],
) -> None:
    comparison = scorecard.baseline_comparison
    if scorecard.initial_release:
        if comparison is not None:
            violations.append("initial release cannot declare a baseline comparison")
        return
    if comparison is None:
        violations.append("candidate release has no baseline comparison")
        return
    _maximum(comparison.semantic_regression, 0.03, "semantic regression", violations)
    _maximum(comparison.retrieval_regression, 0.05, "retrieval regression", violations)
    _maximum(comparison.latency_increase, 0.20, "latency increase", violations)
    _maximum(comparison.mean_token_increase, 0.15, "mean token increase", violations)
    _maximum(comparison.mean_cost_increase, 0.15, "mean cost increase", violations)
    _require(not comparison.safety_regression, "safety regression", violations)
    _require(not comparison.error_rate_regression, "error-rate regression", violations)
    _require(not comparison.per_example_regression, "per-example regression", violations)


def _require(value: bool, message: str, violations: list[str]) -> None:
    if not value:
        violations.append(message)


def _minimum(
    value: float,
    minimum: float,
    label: str,
    violations: list[str],
) -> None:
    if value < minimum:
        violations.append(f"{label} is below {minimum}")


def _maximum(
    value: float,
    maximum: float,
    label: str,
    violations: list[str],
) -> None:
    if value > maximum:
        violations.append(f"{label} exceeds {maximum}")


def _relative_increase(candidate: float, baseline: float) -> float:
    if candidate <= baseline:
        return 0.0
    if baseline == 0:
        return 1.0
    return (candidate - baseline) / baseline


def _error_rate(scorecard: ReleaseScorecard) -> float:
    run_count = scorecard.example_count * scorecard.repetitions
    return scorecard.experiment_errors / run_count if run_count else 0.0


def _counts_regressed(
    candidate: dict[str, dict[str, int]],
    baseline: dict[str, dict[str, int]],
) -> bool:
    return any(
        candidate.get(scenario, {}).get(key, 0) < count
        for scenario, values in baseline.items()
        for key, count in values.items()
    )


def _per_example_regressed(
    candidate: ReleaseScorecard,
    baseline: ReleaseScorecard,
) -> bool:
    return any(
        baseline_count >= 2
        and candidate.ordinary_pass_counts.get(scenario, {}).get(key, 0) <= 1
        for scenario, values in baseline.ordinary_pass_counts.items()
        for key, baseline_count in values.items()
    )
