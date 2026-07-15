"""L1 aggregation tests for LangSmith experiment reports."""

from types import SimpleNamespace
from typing import cast

import pytest

from evals.policy import applicable_ordinary_evaluators
from evals.release import ORDINARY_EVALUATORS, RELEASE_SCENARIOS, SAFETY_EVALUATORS
from evals.report import (
    _usage_metrics,
    build_diagnostic_report,
    build_release_scorecard,
)


class RecordingClient:
    def __init__(self, runs: list[SimpleNamespace]) -> None:
        self.runs = runs

    def list_runs(self, **kwargs: object) -> list[SimpleNamespace]:
        assert kwargs == {"project_name": "release-experiment"}
        return self.runs


def test_report_aggregates_all_release_examples_repetitions_and_usage() -> None:
    rows = []
    llm_runs = []
    for scenario in sorted(RELEASE_SCENARIOS):
        for repetition in range(3):
            trace_id = f"{scenario}-{repetition}"
            turn_count = 2 if scenario == "multi_turn" else 1
            rows.append(
                {
                    "run": SimpleNamespace(
                        trace_id=trace_id,
                        error=None,
                        outputs=_target_output(scenario, turn_count),
                    ),
                    "example": SimpleNamespace(metadata={"scenario_id": scenario}),
                    "evaluation_results": SimpleNamespace(
                        results=[
                            SimpleNamespace(key=key, score=1.0)
                            for key in (
                                *applicable_ordinary_evaluators(scenario),
                                *SAFETY_EVALUATORS,
                                "expected_evidence_recall_at_5",
                                "final_evidence_mrr",
                            )
                        ]
                    ),
                }
            )
            for turn_index in range(turn_count):
                turn_run_id = f"{trace_id}-turn-{turn_index}"
                llm_runs.extend(
                    [
                        SimpleNamespace(
                            id=turn_run_id,
                            name="evaluation_turn",
                            run_type="chain",
                            parent_run_id=trace_id,
                            prompt_tokens=None,
                            completion_tokens=None,
                            total_cost=None,
                        ),
                        SimpleNamespace(
                            id=f"{turn_run_id}-llm",
                            name="ChatOpenAI",
                            run_type="llm",
                            parent_run_id=turn_run_id,
                            prompt_tokens=100,
                            completion_tokens=20,
                            total_cost=0.01,
                        ),
                    ]
                )

    scorecard = build_release_scorecard(
        rows,
        client=RecordingClient(llm_runs),
        experiment_name="release-experiment",
        deterministic_gates_passed=True,
        agent_server_smoke_passed=True,
        human_review_approved=True,
        initial_release=True,
    )

    assert scorecard.example_count == 9
    assert scorecard.turn_count == 10
    assert scorecard.repetitions == 3
    assert scorecard.safety_minimum == 1.0
    assert scorecard.score_standard_deviation["groundedness"] == 0.0
    assert scorecard.usage_metrics_complete is True
    assert scorecard.maximum_input_tokens_per_turn == 100
    assert scorecard.maximum_output_tokens_per_turn == 20
    assert scorecard.mean_tokens_per_turn == 120
    assert scorecard.total_experiment_cost_usd == pytest.approx(0.3)
    assert all(
        count == 3
        for counts in scorecard.safety_pass_counts.values()
        for count in counts.values()
    )
    assert set(scorecard.ordinary_pass_counts["greeting"]) == {
        "answer_relevance",
        "helpfulness",
        "retrieval_relevance",
    }


def test_report_safety_minimum_includes_deterministic_safety_feedback() -> None:
    trace_id = "happy-0"
    scorecard = build_release_scorecard(
        [
            {
                "run": SimpleNamespace(
                    trace_id=trace_id,
                    error=None,
                    outputs=_target_output("happy", 1),
                ),
                "example": SimpleNamespace(metadata={"scenario_id": "happy"}),
                "evaluation_results": SimpleNamespace(
                    results=[
                        SimpleNamespace(key="terminal_contract", score=0.0),
                        SimpleNamespace(key="forbidden_claim_absence", score=1.0),
                    ]
                ),
            }
        ],
        client=RecordingClient(_usage_runs(trace_id)),
        experiment_name="release-experiment",
        deterministic_gates_passed=True,
        agent_server_smoke_passed=True,
        human_review_approved=True,
        initial_release=True,
    )

    assert scorecard.safety_minimum == 0.0


def test_usage_maximum_is_not_hidden_by_averaging_two_turns() -> None:
    runs = []
    for index, prompt_tokens in enumerate((90_000, 0)):
        turn_id = f"turn-{index}"
        runs.extend(
            [
                SimpleNamespace(
                    id=turn_id,
                    name="evaluation_turn",
                    run_type="chain",
                    parent_run_id="root",
                    prompt_tokens=None,
                    completion_tokens=None,
                    total_cost=None,
                ),
                SimpleNamespace(
                    id=f"{turn_id}-llm",
                    name="ChatOpenAI",
                    run_type="llm",
                    parent_run_id=turn_id,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=10,
                    total_cost=0.01,
                ),
            ]
        )

    usage = _usage_metrics(
        RecordingClient(runs),
        "release-experiment",
        expected_turn_runs=2,
    )

    assert usage["maximum_input_tokens_per_turn"] == 90_000


def test_diagnostic_report_exposes_poisoned_correctness_failure() -> None:
    rows = [
        {
            "run": SimpleNamespace(error=None, outputs={"ok": True}),
            "evaluation_results": SimpleNamespace(
                results=[
                    SimpleNamespace(key="answer_correctness", score=0.0),
                    SimpleNamespace(key="groundedness", score=score),
                    SimpleNamespace(key="citation_entailment", score=1.0),
                    SimpleNamespace(key="terminal_contract", score=1.0),
                    SimpleNamespace(key="safety_contract", score=1.0),
                    SimpleNamespace(key="citation_contract", score=1.0),
                    SimpleNamespace(key="forbidden_claim_absence", score=1.0),
                ]
            ),
        }
        for score in (0.8, 1.0, 0.9)
    ]

    report = build_diagnostic_report(rows)
    means = cast(dict[str, float], report["means"])
    variation = cast(dict[str, float], report["standard_deviation"])

    assert report["passed"] is True
    assert means["answer_correctness"] == 0.0
    assert means["groundedness"] == 0.9
    assert variation["groundedness"] > 0


def _target_output(scenario: str, turn_count: int) -> dict[str, object]:
    turns = []
    for index in range(turn_count):
        turns.append(
            {
                "turn_id": f"turn-{index + 1}",
                "assistant_message": "A verified answer.",
                "standalone_question": "What happened?",
                "outcome": "answered",
                "terminal_reason": "answered",
                "final_evidence": [],
                "citations": [],
                "counters": {
                    "model_calls": 1,
                    "retrieval_requests": 1,
                    "research_iterations": 0,
                    "answer_repairs": 0,
                    "authorization_restarts": 0,
                },
                "selected_knowledge_sources": [],
                "trajectory_nodes": [],
                "security_events": [],
                "evidence_token_count": 0,
                "thread_token_count": 0,
                "answer_token_count": 10,
                "first_progress_seconds": 0.1,
                "latency_seconds": 1.0,
            }
        )
    return {
        "schema_version": "target-output-v1",
        "scenario_id": scenario,
        "turns": turns,
        "errors": [],
    }


def _usage_runs(trace_id: str) -> list[SimpleNamespace]:
    turn_id = f"{trace_id}-turn"
    return [
        SimpleNamespace(
            id=turn_id,
            name="evaluation_turn",
            run_type="chain",
            parent_run_id=trace_id,
            prompt_tokens=None,
            completion_tokens=None,
            total_cost=None,
        ),
        SimpleNamespace(
            id=f"{turn_id}-llm",
            name="ChatOpenAI",
            run_type="llm",
            parent_run_id=turn_id,
            prompt_tokens=100,
            completion_tokens=20,
            total_cost=0.01,
        ),
    ]
