"""L1 command and artifact tests for the L4 evaluation entrypoint."""

import json
import importlib
import sys
from types import SimpleNamespace
from typing import Any, cast

import langsmith as ls
import pytest

from evals.run import (
    _dataset_tag_for_version,
    _read_baseline,
    _require_current_release_tag,
    build_experiment_metadata,
    main,
)


def test_preview_is_local_and_describes_the_frozen_dataset(capsys: pytest.CaptureFixture[str]) -> None:
    result = main(["preview"])

    output = json.loads(capsys.readouterr().out)
    assert result == 0
    assert output == {
        "dataset_name": "agentic-rag-reference-v1",
        "dataset_tag": "release-v1",
        "diagnostic_examples": 1,
        "examples": 10,
        "release_examples": 9,
        "turns": 11,
    }


def test_importing_the_cli_does_not_construct_a_langsmith_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ls,
        "Client",
        lambda *args, **kwargs: pytest.fail("unexpected LangSmith client"),
    )
    sys.modules.pop("evals.run", None)

    importlib.import_module("evals.run")


def test_experiment_requires_both_live_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY and LANGSMITH_API_KEY"):
        main(
            [
                "experiment",
                "--dataset-version",
                "2026-07-15T00:00:00+00:00",
                "--initial-release",
            ]
        )


def test_experiment_metadata_records_reproducible_configuration() -> None:
    metadata = build_experiment_metadata(
        git_sha="abc123",
        resolved_dataset_version="2026-07-15T00:00:00+00:00",
        corpus_revisions={"s1-baseline": {"engineering-docs": "corpus-1"}},
    )

    assert metadata["models"] == [
        "openai:gpt-5.4-mini-2026-03-17",
        "openai:gpt-5.4-nano-2026-03-17",
        "openai:text-embedding-3-small",
    ]
    assert metadata["dataset_tag"] == "release-v1"
    assert metadata["dataset_version"] == "2026-07-15T00:00:00+00:00"
    assert metadata["git_sha"] == "abc123"
    assert metadata["corpus_revisions"] == {
        "s1-baseline": {"engineering-docs": "corpus-1"}
    }
    assert str(metadata["processing_revision"]).startswith("proc_")
    assert str(metadata["retrieval_fingerprint"]).startswith("retrieval_")
    assert metadata["evaluators"] == [
        "terminal_contract",
        "safety_contract",
        "citation_contract",
        "expected_evidence_recall_at_5",
        "final_evidence_mrr",
        "answer_correctness",
        "groundedness",
        "answer_relevance",
        "helpfulness",
        "required_claim_coverage",
        "forbidden_claim_absence",
        "citation_entailment",
        "citation_completeness",
        "retrieval_relevance",
    ]
    tools = cast(list[dict[str, Any]], metadata["tools"])
    tool_parameters = tools[0]["parameters"]
    assert "principal" not in json.dumps(tool_parameters).lower()
    assert "scope" not in json.dumps(tool_parameters).lower()


def test_baseline_must_be_approved_and_use_the_same_dataset_version(
    tmp_path,
) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "schema_version": "release-report-v1",
                "manual_review_status": "approved",
                "dataset_name": "agentic-rag-reference-v1",
                "dataset_tag": "release-v1",
                "dataset_version": "2026-07-14T00:00:00+00:00",
                "gate": {"passed": True, "violations": []},
                "scorecard": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="dataset versions differ"):
        _read_baseline(
            baseline,
            __import__("datetime").datetime.fromisoformat(
                "2026-07-15T00:00:00+00:00"
            ),
        )


def test_experiment_records_release_tag_only_when_it_resolves_to_exact_version() -> None:
    version = __import__("datetime").datetime.fromisoformat(
        "2026-07-15T00:00:00+00:00"
    )
    client = type(
        "Client",
        (),
        {
            "read_dataset_version": lambda self, **kwargs: SimpleNamespace(
                as_of=version
            )
        },
    )()

    assert _dataset_tag_for_version(cast(Any, client), version) == "release-v1"


def test_baseline_version_must_still_be_the_current_release_tag() -> None:
    expected = __import__("datetime").datetime.fromisoformat(
        "2026-07-15T00:00:00+00:00"
    )
    moved = __import__("datetime").datetime.fromisoformat(
        "2026-07-16T00:00:00+00:00"
    )
    client = type(
        "Client",
        (),
        {
            "read_dataset_version": lambda self, **kwargs: SimpleNamespace(
                as_of=moved
            )
        },
    )()

    with pytest.raises(RuntimeError, match="current release-v1"):
        _require_current_release_tag(cast(Any, client), expected)
