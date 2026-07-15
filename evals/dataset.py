"""Build and publish the single versioned LangSmith golden dataset."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from evals.contracts import (
    CounterBound,
    DatasetCandidate,
    DatasetPublication,
    EvaluationClaim,
    EvaluationTurnInput,
    ExampleMetadata,
    FixtureCommand,
    GoldenExample,
    ReferenceOutput,
    ReferenceTurn,
    TargetCommand,
)
from evals.policy import applicable_evaluator_keys


DATASET_NAME = "agentic-rag-reference-v1"
DATASET_TAG = "release-v1"
FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures/reference-system"
GOLDEN_PATH = FIXTURE_ROOT / "evaluation/live-golden.json"
SCENARIO_ROOT = FIXTURE_ROOT / "scenarios"
_GLOBAL_COUNTER_MAXIMA = {
    "model_calls": 10,
    "retrieval_requests": 2,
    "research_iterations": 1,
    "answer_repairs": 1,
    "authorization_restarts": 1,
}


def build_golden_examples() -> tuple[GoldenExample, ...]:
    """Compose self-contained examples from canonical scenario fixtures."""

    golden = _read_json(GOLDEN_PATH)
    if golden.get("version") != "live-golden-v1":
        raise ValueError("unsupported live golden version")
    finance = golden["finance_authorized"]
    examples = []
    for entry in golden["entries"]:
        scenario_id = entry["scenario_id"]
        scenario = (
            finance
            if scenario_id == "finance_authorized"
            else _read_json(SCENARIO_ROOT / f"{scenario_id}.json")
        )
        examples.append(_build_example(scenario, entry, golden))
    return tuple(examples)


def publish_golden_candidate(client: Any) -> DatasetCandidate:
    """Upsert one review candidate without moving the release tag."""

    examples = build_golden_examples()
    if client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
    else:
        dataset = client.create_dataset(
            DATASET_NAME,
            description=(
                "Northstar Labs Agentic RAG v1 release and diagnostic examples."
            ),
            metadata={"fixture_version": "reference-system-v1"},
        )
    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {
                "id": example.example_id,
                "inputs": example.inputs.model_dump(mode="json"),
                "outputs": example.reference_outputs.model_dump(mode="json"),
                "metadata": example.metadata.model_dump(mode="json"),
            }
            for example in examples
        ],
    )
    release_ids = [
        example.example_id for example in examples if example.split == "release"
    ]
    diagnostic_ids = [
        example.example_id
        for example in examples
        if example.split == "diagnostic"
    ]
    client.update_dataset_splits(
        dataset_id=dataset.id,
        split_name="release",
        example_ids=release_ids,
    )
    client.update_dataset_splits(
        dataset_id=dataset.id,
        split_name="diagnostic",
        example_ids=diagnostic_ids,
    )
    versions = tuple(client.list_dataset_versions(dataset_id=dataset.id))
    if not versions:
        raise ValueError("LangSmith returned no dataset version")
    latest = max(versions, key=lambda version: version.as_of)
    return DatasetCandidate(
        dataset_id=str(dataset.id),
        dataset_name=DATASET_NAME,
        dataset_version=latest.as_of,
        example_count=len(examples),
    )


def tag_golden_dataset(
    client: Any,
    *,
    dataset_id: str,
    dataset_version: datetime,
    human_approved: bool,
) -> DatasetPublication:
    """Move release-v1 only after the candidate report was manually approved."""

    if not human_approved:
        raise ValueError("human approval is required before dataset tagging")
    client.update_dataset_tag(
        dataset_id=dataset_id,
        as_of=dataset_version,
        tag=DATASET_TAG,
    )
    return DatasetPublication(
        dataset_id=dataset_id,
        dataset_name=DATASET_NAME,
        dataset_version=dataset_version,
        dataset_tag=DATASET_TAG,
        example_count=len(build_golden_examples()),
    )


def validate_target_command(inputs: dict[str, Any]) -> TargetCommand:
    """Accept only an exact trusted command from the local golden fixture."""

    command = TargetCommand.model_validate(inputs)
    expected = {
        example.inputs.scenario_id: example.inputs
        for example in build_golden_examples()
    }.get(command.scenario_id)
    if command != expected:
        raise ValueError("target command does not match the trusted golden fixture")
    return command


def _build_example(
    scenario: dict[str, Any],
    entry: dict[str, Any],
    golden: dict[str, Any],
) -> GoldenExample:
    turns = scenario["turns"]
    aligned_keys = (
        "reference_answers",
        "required_knowledge_sources",
        "forbidden_knowledge_sources",
        "required_trajectory_nodes",
        "forbidden_trajectory_nodes",
        "forbidden_disclosures",
        "counter_bounds",
    )
    if any(len(entry[key]) != len(turns) for key in aligned_keys):
        raise ValueError("live golden Turn metadata is misaligned")
    reference_turns = []
    for index, turn in enumerate(turns):
        expected = turn["expected"]
        bounds = {
            field: CounterBound(maximum=maximum)
            for field, maximum in _GLOBAL_COUNTER_MAXIMA.items()
        }
        bounds.update(
            {
                field: CounterBound.model_validate(value)
                for field, value in entry["counter_bounds"][index].items()
            }
        )
        reference_turns.append(
            ReferenceTurn(
                expected_outcome=expected["outcome"],
                expected_terminal_reason=expected["terminal_reason"],
                reference_answer=entry["reference_answers"][index],
                expected_evidence_aliases=expected["evidence_aliases"],
                forbidden_evidence_aliases=expected[
                    "forbidden_evidence_aliases"
                ],
                required_claims=tuple(
                    EvaluationClaim(
                        fact_id=claim["id"], statement=claim["statement"]
                    )
                    for claim in expected["required_claims"]
                ),
                forbidden_claims=tuple(
                    EvaluationClaim(
                        fact_id=claim["id"], statement=claim["statement"]
                    )
                    for claim in expected["forbidden_claims"]
                ),
                forbidden_disclosures=tuple(
                    dict.fromkeys(
                        [
                            *golden["global_disclosure_canaries"],
                            *entry["forbidden_disclosures"][index],
                            *(
                                disclosure
                                for alias in expected[
                                    "forbidden_evidence_aliases"
                                ]
                                for disclosure in golden[
                                    "forbidden_evidence_disclosures"
                                ].get(alias, ())
                            ),
                        ]
                    )
                ),
                required_knowledge_sources=entry[
                    "required_knowledge_sources"
                ][index],
                forbidden_knowledge_sources=entry[
                    "forbidden_knowledge_sources"
                ][index],
                required_trajectory_nodes=entry[
                    "required_trajectory_nodes"
                ][index],
                forbidden_trajectory_nodes=entry[
                    "forbidden_trajectory_nodes"
                ][index],
                counter_bounds=bounds,
            )
        )
    fixture = scenario["fixture_setup"]
    scenario_id = scenario["id"]
    return GoldenExample(
        example_id=uuid5(NAMESPACE_URL, f"{DATASET_NAME}/{scenario_id}"),
        split=entry["split"],
        inputs=TargetCommand(
            scenario_id=scenario_id,
            fixture_setup=FixtureCommand(
                snapshot=fixture["snapshot"],
                overlay=fixture["overlay"],
                principal_id=scenario["principal_id"],
                authorization_event=scenario.get("authorization_event"),
            ),
            turns=tuple(
                EvaluationTurnInput(
                    turn_id=turn["turn_id"],
                    user_message=turn["user_message"],
                )
                for turn in turns
            ),
        ),
        reference_outputs=ReferenceOutput(turns=tuple(reference_turns)),
        metadata=ExampleMetadata(
            scenario_id=scenario_id,
            risk_domain=entry["risk_domain"],
            conversation_shape=(
                "multi_turn" if len(turns) > 1 else "single_turn"
            ),
            known_limitation=scenario.get("known_limitation", False),
            applicable_evaluators=applicable_evaluator_keys(scenario_id),
        ),
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
