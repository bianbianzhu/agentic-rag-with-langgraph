"""L1 non-compensable release scorecard gates."""

from evals.policy import applicable_ordinary_evaluators

from evals.release import (
    BaselineComparison,
    ReleaseScorecard,
    compare_scorecards,
    gate_release,
)


RELEASE_SCENARIOS = (
    "auth_change",
    "clarification",
    "document_injection",
    "finance_authorized",
    "greeting",
    "happy",
    "multi_turn",
    "refine",
    "relevant_not_allowed",
)
ORDINARY = (
    "required_claim_coverage",
    "groundedness",
    "citation_entailment",
    "citation_completeness",
    "answer_correctness",
    "answer_relevance",
    "helpfulness",
    "retrieval_relevance",
)
SAFETY = (
    "terminal_contract",
    "safety_contract",
    "citation_contract",
    "forbidden_claim_absence",
)


def _passing_scorecard() -> ReleaseScorecard:
    return ReleaseScorecard(
        deterministic_gates_passed=True,
        agent_server_smoke_passed=True,
        human_review_approved=True,
        initial_release=True,
        example_count=9,
        turn_count=10,
        repetitions=3,
        experiment_errors=0,
        deadline_exceeded_runs=0,
        usage_metrics_complete=True,
        expected_evidence_recall_at_5=0.95,
        final_evidence_mrr=0.9,
        retrieval_relevance=0.9,
        required_claim_coverage=0.95,
        groundedness=0.95,
        citation_entailment=0.95,
        citation_completeness=0.95,
        answer_correctness=0.9,
        answer_relevance=0.85,
        helpfulness=0.85,
        forbidden_claim_absence=1.0,
        safety_minimum=1.0,
        score_standard_deviation={key: 0.01 for key in ORDINARY},
        first_progress_p95_seconds=0.5,
        successful_turn_latency_p50_seconds=10,
        successful_turn_latency_p95_seconds=20,
        maximum_input_tokens_per_turn=10_000,
        maximum_output_tokens_per_turn=1_000,
        mean_tokens_per_turn=2_000,
        mean_cost_per_turn_usd=0.02,
        maximum_cost_per_turn_usd=0.04,
        total_experiment_cost_usd=1.0,
        ordinary_pass_counts={
            scenario: {
                key: 3 for key in applicable_ordinary_evaluators(scenario)
            }
            for scenario in RELEASE_SCENARIOS
        },
        safety_pass_counts={
            scenario: {key: 3 for key in SAFETY}
            for scenario in RELEASE_SCENARIOS
        },
        baseline_comparison=None,
    )


def test_release_gate_accepts_the_initial_scorecard_at_all_floors() -> None:
    report = gate_release(_passing_scorecard())

    assert report.passed is True
    assert report.violations == ()


def test_one_safety_failure_cannot_be_compensated_by_high_means() -> None:
    scorecard = _passing_scorecard()
    safety = {
        scenario: dict(values)
        for scenario, values in scorecard.safety_pass_counts.items()
    }
    safety["auth_change"]["safety_contract"] = 2
    unsafe = scorecard.model_copy(
        update={
            "safety_minimum": 0.99,
            "safety_pass_counts": safety,
        }
    )

    report = gate_release(unsafe)

    assert report.passed is False
    assert any("safety" in violation.lower() for violation in report.violations)


def test_candidate_requires_a_baseline_comparison_within_tolerance() -> None:
    candidate = _passing_scorecard().model_copy(
        update={
            "initial_release": False,
            "baseline_comparison": BaselineComparison(
                semantic_regression=0.04,
                retrieval_regression=0.0,
                latency_increase=0.0,
                mean_token_increase=0.0,
                mean_cost_increase=0.0,
                safety_regression=False,
                error_rate_regression=False,
                per_example_regression=False,
            ),
        }
    )

    report = gate_release(candidate)

    assert report.passed is False
    assert "semantic regression exceeds 0.03" in report.violations


def test_baseline_comparison_detects_mean_token_and_per_example_regressions() -> None:
    baseline = _passing_scorecard()
    ordinary = {
        scenario: dict(values)
        for scenario, values in baseline.ordinary_pass_counts.items()
    }
    ordinary["happy"]["groundedness"] = 1
    candidate = baseline.model_copy(
        update={
            "initial_release": False,
            "mean_tokens_per_turn": 2_400,
            "ordinary_pass_counts": ordinary,
        }
    )

    comparison = compare_scorecards(candidate, baseline)

    assert comparison.mean_token_increase == 0.2
    assert comparison.per_example_regression is True
