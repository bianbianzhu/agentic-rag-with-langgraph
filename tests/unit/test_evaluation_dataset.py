"""L1 contracts for the versioned LangSmith golden dataset."""

from uuid import UUID

import pytest

from evals.dataset import (
    DATASET_NAME,
    DATASET_TAG,
    build_golden_examples,
    publish_golden_candidate,
    tag_golden_dataset,
    validate_target_command,
)


def test_golden_dataset_has_ten_examples_and_eleven_turns() -> None:
    examples = build_golden_examples()

    assert len(examples) == 10
    assert sum(len(example.inputs.turns) for example in examples) == 11
    assert {example.inputs.scenario_id for example in examples} == {
        "auth_change",
        "clarification",
        "document_injection",
        "finance_authorized",
        "greeting",
        "happy",
        "multi_turn",
        "poisoned_fact",
        "refine",
        "relevant_not_allowed",
    }
    assert {example.inputs.scenario_id for example in examples}.isdisjoint(
        {"budget", "repair", "retrieval_failure", "thread_mismatch"}
    )
    assert len({example.example_id for example in examples}) == 10
    assert all(isinstance(example.example_id, UUID) for example in examples)


def test_golden_examples_keep_commands_references_and_metadata_separate() -> None:
    examples = {
        example.inputs.scenario_id: example
        for example in build_golden_examples()
    }
    poisoned = examples["poisoned_fact"]
    finance = examples["finance_authorized"]

    assert poisoned.split == "diagnostic"
    assert poisoned.metadata.known_limitation is True
    assert "undefined_column" in (
        poisoned.reference_outputs.turns[0].reference_answer
    )
    assert all(
        example.split == "release"
        for scenario_id, example in examples.items()
        if scenario_id != "poisoned_fact"
    )
    assert finance.inputs.fixture_setup.principal_id == "carol"
    assert finance.inputs.turns[0].user_message == (
        "For the confidential settlement impact, how many merchants and how "
        "much money were affected, and for how long?"
    )
    assert finance.reference_outputs.turns[0].expected_evidence_aliases == (
        "D8#settlement-impact",
    )
    assert finance.reference_outputs.turns[0].required_claims[0].fact_id == "F5"
    assert set(examples["greeting"].metadata.applicable_evaluators).isdisjoint(
        {"answer_correctness", "groundedness", "citation_entailment"}
    )
    assert examples["refine"].inputs.turns[0].user_message.startswith(
        "First search for the exact phrase 'empty legacy query'."
    )
    assert examples["multi_turn"].inputs.turns[1].user_message == (
        "For the payments rollback we just discussed, did staging fail for "
        "the same reason or for a different one?"
    )
    assert "payments/rollback-recovery.md" in (
        finance.reference_outputs.turns[0].forbidden_disclosures
    )
    for scenario_id in (
        "auth_change",
        "document_injection",
        "relevant_not_allowed",
        "finance_authorized",
    ):
        assert examples[scenario_id].reference_outputs.turns[
            0
        ].forbidden_disclosures
    for example in examples.values():
        inputs = example.inputs.model_dump(mode="json")
        assert "required_claims" not in str(inputs)
        assert "evidence_aliases" not in str(inputs)
        for turn in example.reference_outputs.turns:
            for claim in turn.required_claims + turn.forbidden_claims:
                assert claim.statement.strip()


def test_tagging_requires_explicit_human_golden_approval() -> None:
    with pytest.raises(ValueError, match="human approval"):
        tag_golden_dataset(
            object(),
            dataset_id="dataset-1",
            dataset_version=__import__("datetime").datetime(2026, 7, 15),
            human_approved=False,
        )


def test_target_command_cannot_change_the_trusted_principal() -> None:
    command = build_golden_examples()[0].inputs.model_dump(mode="json")
    command["fixture_setup"]["principal_id"] = "carol"

    with pytest.raises(ValueError, match="trusted golden fixture"):
        validate_target_command(command)


class RecordingClient:
    def __init__(self) -> None:
        self.created_examples: list[dict[str, object]] = []
        self.splits: dict[str, list[UUID]] = {}
        self.tags: list[tuple[str, str]] = []
        self.updated_examples: list[UUID] = []

    def has_dataset(self, *, dataset_name: str) -> bool:
        assert dataset_name == DATASET_NAME
        return False

    def create_dataset(self, dataset_name: str, **kwargs: object):
        assert dataset_name == DATASET_NAME
        return type("Dataset", (), {"id": "dataset-1"})()

    def create_examples(self, *, dataset_id: str, examples: list[dict[str, object]]):
        assert dataset_id == "dataset-1"
        self.created_examples = examples

    def list_examples(self, **kwargs: object):
        return iter(())

    def update_example(self, example_id: UUID, **kwargs: object) -> None:
        self.updated_examples.append(example_id)

    def update_dataset_splits(
        self,
        *,
        dataset_id: str,
        split_name: str,
        example_ids: list[UUID],
    ) -> None:
        assert dataset_id == "dataset-1"
        self.splits[split_name] = example_ids

    def list_dataset_versions(self, *, dataset_id: str):
        assert dataset_id == "dataset-1"
        version = type(
            "Version",
            (),
            {"as_of": __import__("datetime").datetime(2026, 7, 15)},
        )()
        return iter((version,))

    def update_dataset_tag(
        self,
        *,
        dataset_id: str,
        as_of: object,
        tag: str,
    ) -> None:
        assert dataset_id == "dataset-1"
        self.tags.append((str(as_of), tag))


def test_publish_creates_an_untagged_candidate_then_acceptance_tags_exact_version() -> None:
    client = RecordingClient()

    candidate = publish_golden_candidate(client)

    assert candidate.dataset_name == DATASET_NAME
    assert len(client.created_examples) == 10
    assert len(client.splits["release"]) == 9
    assert len(client.splits["diagnostic"]) == 1
    assert client.tags == []

    result = tag_golden_dataset(
        client,
        dataset_id=candidate.dataset_id,
        dataset_version=candidate.dataset_version,
        human_approved=True,
    )

    assert result.dataset_tag == DATASET_TAG
    assert client.tags[0][1] == DATASET_TAG


def test_publish_updates_existing_fixed_id_examples_instead_of_conflicting() -> None:
    client = RecordingClient()
    client.has_dataset = lambda **kwargs: True  # type: ignore[method-assign]
    client.read_dataset = lambda **kwargs: type(  # type: ignore[attr-defined]
        "Dataset", (), {"id": "dataset-1"}
    )()
    existing_ids = [example.example_id for example in build_golden_examples()]
    client.list_examples = lambda **kwargs: iter(  # type: ignore[method-assign]
        type("Example", (), {"id": example_id})()
        for example_id in existing_ids
    )

    publish_golden_candidate(client)

    assert client.created_examples == []
    assert client.updated_examples == existing_ids
