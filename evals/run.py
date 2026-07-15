"""Command-line entrypoint for the versioned LangSmith release workflow."""

import argparse
from datetime import UTC, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING, Any, Sequence, cast

import langsmith as ls
from langsmith.utils import LangSmithNotFoundError

from agentic_rag.corpus.identity import processing_revision
from agentic_rag.observability import trace_anonymizer
from agentic_rag.retrieval import RetrievalConfig
from evals.config import (
    CHAT_MODEL_ID,
    EMBEDDING_MODEL_ID,
    LIVE_PROCESSING_CONFIG,
    MODEL_TEMPERATURE,
    RERANKER_MODEL_ID,
)
from evals.dataset import (
    DATASET_NAME,
    DATASET_TAG,
    build_golden_examples,
    publish_golden_candidate,
    tag_golden_dataset,
)
from evals.evaluators import code_evaluators, semantic_evaluators
from evals.fixtures import prepare_live_fixture
from evals.policy import EVALUATOR_IDENTITIES
from evals.release import (
    ReleaseGateReport,
    ReleaseScorecard,
    compare_scorecards,
    gate_release,
)
from evals.report import build_diagnostic_report, build_release_scorecard


if TYPE_CHECKING:
    from evals.target import LiveTarget


REPOSITORY_ROOT = Path(__file__).parents[1]
DEFAULT_ARTIFACT_DIRECTORY = REPOSITORY_ROOT / "evals/artifacts"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preview":
        print(json.dumps(_preview(), indent=2, sort_keys=True))
        return 0
    if args.command == "publish":
        _require_environment("LANGSMITH_API_KEY")
        candidate = publish_golden_candidate(_client())
        print(candidate.model_dump_json(indent=2))
        return 0
    if args.command == "experiment":
        _require_environment("OPENAI_API_KEY", "LANGSMITH_API_KEY")
        return _run_experiment(args)
    if args.command == "accept":
        _require_environment("LANGSMITH_API_KEY")
        return _accept_candidate(args)
    raise RuntimeError("unknown evaluation command")


def build_experiment_metadata(
    *,
    git_sha: str,
    resolved_dataset_version: str,
    corpus_revisions: dict[str, dict[str, str]],
    dataset_tag: str | None = DATASET_TAG,
) -> dict[str, object]:
    """Return only non-secret identities required to reproduce a release run."""

    return {
        "models": [CHAT_MODEL_ID, RERANKER_MODEL_ID, EMBEDDING_MODEL_ID],
        "model_parameters": {"temperature": MODEL_TEMPERATURE},
        "prompts": [
            "contextual-rewrite-prompt-v2",
            "conversation-summary-prompt-v1",
            "research-plan-prompt-v2",
            "evidence-assessment-prompt-v2",
            "query-refinement-prompt-v1",
            "answer-generation-v2",
            "answer-verification-v1",
            "listwise-v2",
            "openevals-0.2.0",
            "route-quality-judge-v1",
            "forbidden-claim-judge-v2",
        ],
        "tools": [
            {
                "name": "authorized_hybrid_retrieval",
                "description": "Retrieve bounded Evidence under code-owned Access Scope.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "knowledge_sources": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["query", "knowledge_sources"],
                },
            }
        ],
        "git_sha": git_sha,
        "fixture_version": "reference-system-v1",
        "dataset_tag": dataset_tag or "candidate",
        "dataset_version": resolved_dataset_version,
        "corpus_revisions": corpus_revisions,
        "processing_revision": processing_revision(LIVE_PROCESSING_CONFIG),
        "retrieval_fingerprint": _retrieval_fingerprint(),
        "evaluators": list(EVALUATOR_IDENTITIES),
        "judge_model": CHAT_MODEL_ID,
    }


def write_release_artifacts(
    directory: Path,
    *,
    dataset_version: datetime,
    dataset_tag: str | None,
    experiment_name: str,
    experiment_url: str | None,
    diagnostic_experiment_name: str,
    diagnostic_experiment_url: str | None,
    diagnostic_report: dict[str, object],
    scorecard: ReleaseScorecard,
    gate: ReleaseGateReport,
) -> tuple[Path, Path]:
    """Persist a reproducible review surface without secrets or raw traces."""

    directory.mkdir(parents=True, exist_ok=True)
    accepted = dataset_tag == DATASET_TAG and scorecard.human_review_approved
    stem = "accepted-release-report" if accepted else "candidate-report"
    generated_at = datetime.now(UTC).isoformat()
    payload = {
        "schema_version": "release-report-v1",
        "generated_at": generated_at,
        "manual_review_status": "approved" if accepted else "pending",
        "dataset_name": DATASET_NAME,
        "dataset_version": dataset_version.isoformat(),
        "dataset_tag": dataset_tag,
        "experiment_name": experiment_name,
        "experiment_url": experiment_url,
        "diagnostic_experiment_name": diagnostic_experiment_name,
        "diagnostic_experiment_url": diagnostic_experiment_url,
        "diagnostic_report": diagnostic_report,
        "scorecard": scorecard.model_dump(mode="json"),
        "gate": gate.model_dump(mode="json"),
    }
    json_path = directory / f"{stem}.json"
    markdown_path = directory / f"{stem}.md"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    violations = "\n".join(
        f"- {violation}" for violation in gate.violations
    ) or "- None"
    markdown_path.write_text(
        "\n".join(
            (
                "# Agentic RAG v1 release report",
                "",
                f"- Dataset version: `{dataset_version.isoformat()}`",
                f"- Dataset tag: `{dataset_tag or 'candidate'}`",
                f"- Manual review: `{'approved' if accepted else 'pending'}`",
                f"- Release experiment: {experiment_url or experiment_name}",
                f"- Diagnostic experiment: {diagnostic_experiment_url or diagnostic_experiment_name}",
                f"- Release gate: `{'PASS' if gate.passed else 'FAIL'}`",
                "",
                "## Violations",
                "",
                violations,
                "",
                "## Poisoned-source diagnostic",
                "",
                "```json",
                json.dumps(diagnostic_report, indent=2, sort_keys=True),
                "```",
                "",
                "## Scorecard",
                "",
                "```json",
                scorecard.model_dump_json(indent=2),
                "```",
                "",
            )
        ),
        encoding="utf-8",
    )
    return json_path, markdown_path


def _run_experiment(args: argparse.Namespace) -> int:
    from evals.target import open_live_target

    if args.initial_release == bool(args.baseline_report):
        raise RuntimeError(
            "choose exactly one of --initial-release or --baseline-report"
        )
    dataset_version = _parse_dataset_version(args.dataset_version)
    baseline = _read_baseline(args.baseline_report, dataset_version)
    deterministic, smoke = _run_local_gates()
    if not deterministic or not smoke:
        raise RuntimeError("L1-L3 or Agent Server smoke gates failed")
    client = _client()
    if baseline is None:
        version = client.read_dataset_version(
            dataset_name=DATASET_NAME,
            as_of=dataset_version,
        )
        examples_as_of: datetime | str = version.as_of
    else:
        version = _require_current_release_tag(client, dataset_version)
        examples_as_of = DATASET_TAG
    release_examples = list(
        client.list_examples(
            dataset_name=DATASET_NAME,
            as_of=examples_as_of,
            splits=["release"],
        )
    )
    diagnostic_examples = list(
        client.list_examples(
            dataset_name=DATASET_NAME,
            as_of=examples_as_of,
            splits=["diagnostic"],
        )
    )
    with open_live_target() as target:
        dataset_tag = _dataset_tag_for_version(client, version.as_of)
        metadata = build_experiment_metadata(
            git_sha=_git_sha(),
            resolved_dataset_version=version.as_of.isoformat(),
            corpus_revisions=_collect_corpus_revisions(target),
            dataset_tag=dataset_tag,
        )
        evaluators = [
            *code_evaluators(),
            *semantic_evaluators(target.chat_model),
        ]
        release_results = _evaluate(
            client,
            target,
            release_examples,
            evaluators,
            metadata={**metadata, "dataset_split": "release"},
            prefix="agentic-rag-reference-v1-release-candidate",
        )
        release_rows = list(release_results)
        diagnostic_results = _evaluate(
            client,
            target,
            diagnostic_examples,
            evaluators,
            metadata={**metadata, "dataset_split": "diagnostic"},
            prefix="agentic-rag-reference-v1-poisoned-diagnostic",
        )
        diagnostic_rows = list(diagnostic_results)
    scorecard = build_release_scorecard(
        release_rows,
        client=client,
        experiment_name=release_results.experiment_name,
        deterministic_gates_passed=deterministic,
        agent_server_smoke_passed=smoke,
        human_review_approved=False,
        initial_release=args.initial_release,
    )
    if baseline is not None:
        scorecard = scorecard.model_copy(
            update={
                "baseline_comparison": compare_scorecards(scorecard, baseline)
            }
        )
    gate = gate_release(scorecard)
    review_gate = gate_release(
        scorecard.model_copy(update={"human_review_approved": True})
    )
    diagnostic_report = build_diagnostic_report(diagnostic_rows)
    review_ready = review_gate.passed and diagnostic_report["passed"] is True
    paths = write_release_artifacts(
        args.artifact_directory,
        dataset_version=version.as_of,
        dataset_tag=None,
        experiment_name=release_results.experiment_name,
        experiment_url=release_results.url,
        diagnostic_experiment_name=diagnostic_results.experiment_name,
        diagnostic_experiment_url=diagnostic_results.url,
        diagnostic_report=diagnostic_report,
        scorecard=scorecard,
        gate=gate,
    )
    print(
        json.dumps(
            {
                "review_ready": review_ready,
                "artifacts": [str(path) for path in paths],
                "release_experiment_url": release_results.url,
                "diagnostic_experiment_url": diagnostic_results.url,
                "next": (
                    "inspect both LangSmith experiments, then run accept"
                    if review_ready
                    else "fix automated release violations before review"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if review_ready else 1


def _accept_candidate(args: argparse.Namespace) -> int:
    if not args.human_approved:
        raise RuntimeError("--human-approved is required to accept a candidate")
    payload = _read_report(args.candidate_report)
    if (
        payload.get("manual_review_status") != "pending"
        or payload.get("dataset_name") != DATASET_NAME
        or payload.get("dataset_tag") is not None
    ):
        raise RuntimeError("candidate report is not an untagged review artifact")
    dataset_version = _parse_dataset_version(str(payload["dataset_version"]))
    scorecard = ReleaseScorecard.model_validate(payload["scorecard"]).model_copy(
        update={"human_review_approved": True}
    )
    diagnostic_report = cast(dict[str, object], payload["diagnostic_report"])
    if diagnostic_report.get("passed") is not True:
        raise RuntimeError("poisoned-source diagnostic gates failed")
    gate = gate_release(scorecard)
    if not gate.passed:
        raise RuntimeError(f"candidate release gates failed: {gate.violations}")
    client = _client()
    dataset = client.read_dataset(dataset_name=DATASET_NAME)
    resolved = client.read_dataset_version(
        dataset_id=dataset.id,
        as_of=dataset_version,
    )
    if resolved.as_of != dataset_version:
        raise RuntimeError("candidate dataset version no longer resolves exactly")
    tag_golden_dataset(
        client,
        dataset_id=str(dataset.id),
        dataset_version=dataset_version,
        human_approved=True,
    )
    paths = write_release_artifacts(
        args.artifact_directory,
        dataset_version=dataset_version,
        dataset_tag=DATASET_TAG,
        experiment_name=str(payload["experiment_name"]),
        experiment_url=cast(str | None, payload["experiment_url"]),
        diagnostic_experiment_name=str(payload["diagnostic_experiment_name"]),
        diagnostic_experiment_url=cast(
            str | None, payload["diagnostic_experiment_url"]
        ),
        diagnostic_report=diagnostic_report,
        scorecard=scorecard,
        gate=gate,
    )
    print(
        json.dumps(
            {
                "accepted": True,
                "dataset_tag": DATASET_TAG,
                "artifacts": [str(path) for path in paths],
                "next": "set the accepted release experiment as LangSmith baseline",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _evaluate(
    client: ls.Client,
    target: "LiveTarget",
    examples: list[Any],
    evaluators: list[Any],
    *,
    metadata: dict[str, object],
    prefix: str,
) -> Any:
    return cast(Any, client).evaluate(
        target,
        data=examples,
        evaluators=evaluators,
        metadata=metadata,
        experiment_prefix=prefix,
        # Every target Run resets the same PostgreSQL fixture. Serial execution
        # prevents overlays and grant mutations from leaking across examples.
        max_concurrency=1,
        num_repetitions=3,
        error_handling="log",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preview", help="validate and summarize local golden data")
    commands.add_parser("publish", help="publish an untagged review candidate")
    experiment = commands.add_parser(
        "experiment",
        help="run release and poisoned-source experiments on one exact version",
    )
    experiment.add_argument("--dataset-version", required=True)
    experiment.add_argument("--initial-release", action="store_true")
    experiment.add_argument("--baseline-report", type=Path)
    experiment.add_argument(
        "--artifact-directory",
        type=Path,
        default=DEFAULT_ARTIFACT_DIRECTORY,
    )
    accept = commands.add_parser(
        "accept",
        help="tag a manually inspected candidate and emit the accepted baseline",
    )
    accept.add_argument("--candidate-report", type=Path, required=True)
    accept.add_argument("--human-approved", action="store_true")
    accept.add_argument(
        "--artifact-directory",
        type=Path,
        default=DEFAULT_ARTIFACT_DIRECTORY,
    )
    return parser


def _preview() -> dict[str, object]:
    examples = build_golden_examples()
    return {
        "dataset_name": DATASET_NAME,
        "dataset_tag": DATASET_TAG,
        "examples": len(examples),
        "turns": sum(len(example.inputs.turns) for example in examples),
        "release_examples": sum(example.split == "release" for example in examples),
        "diagnostic_examples": sum(
            example.split == "diagnostic" for example in examples
        ),
    }


def _client() -> ls.Client:
    client = ls.Client(anonymizer=trace_anonymizer())
    ls.configure(client=client, enabled=True)
    return client


def _require_environment(*names: str) -> None:
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"{' and '.join(missing)} are required")


def _run_local_gates() -> tuple[bool, bool]:
    pytest = str(Path(sys.executable).with_name("pytest"))
    pyright = str(Path(sys.executable).with_name("pyright"))
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required for lock verification")
    deterministic_commands = (
        [pytest, "-q", "--ignore=tests/smoke"],
        [pyright],
        [uv, "lock", "--check"],
    )
    deterministic = all(_run(command) for command in deterministic_commands)
    smoke = deterministic and _run(
        [pytest, "tests/smoke/test_agent_server.py", "-q"]
    )
    return deterministic, smoke


def _run(command: list[str]) -> bool:
    return subprocess.run(command, cwd=REPOSITORY_ROOT, check=False).returncode == 0


def _collect_corpus_revisions(
    target: "LiveTarget",
) -> dict[str, dict[str, str]]:
    fixture_variants = (
        ("s1-baseline", None),
        ("s1-baseline", "document-injection"),
        ("s1-baseline", "poisoned-fact"),
    )
    return {
        f"{snapshot}:{overlay or 'none'}": prepare_live_fixture(
            target.write_pool,
            snapshot=snapshot,
            overlay=overlay,
            embedder=target.embedder,
        )
        for snapshot, overlay in fixture_variants
    }


def _retrieval_fingerprint() -> str:
    payload = {
        "config": RetrievalConfig(
            embedding_model=EMBEDDING_MODEL_ID
        ).model_dump(mode="json"),
        "dense_candidate_limit": 50,
        "lexical_candidate_limit": 50,
        "result_limit": 8,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return f"retrieval_{sha256(serialized.encode()).hexdigest()}"


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _parse_dataset_version(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise RuntimeError("dataset version must be an ISO-8601 timestamp") from error


def _dataset_tag_for_version(
    client: ls.Client,
    dataset_version: datetime,
) -> str | None:
    try:
        tagged = client.read_dataset_version(
            dataset_name=DATASET_NAME,
            tag=DATASET_TAG,
        )
    except LangSmithNotFoundError:
        return None
    return DATASET_TAG if tagged.as_of == dataset_version else None


def _require_current_release_tag(
    client: ls.Client,
    expected_dataset_version: datetime,
) -> Any:
    try:
        tagged = client.read_dataset_version(
            dataset_name=DATASET_NAME,
            tag=DATASET_TAG,
        )
    except LangSmithNotFoundError as error:
        raise RuntimeError("current release-v1 dataset tag is missing") from error
    if tagged.as_of != expected_dataset_version:
        raise RuntimeError(
            "baseline dataset version is not the current release-v1 version"
        )
    return tagged


def _read_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "release-report-v1":
        raise RuntimeError("release report contract is invalid")
    return payload


def _read_baseline(
    path: Path | None,
    expected_dataset_version: datetime,
) -> ReleaseScorecard | None:
    if path is None:
        return None
    payload = _read_report(path)
    if (
        payload.get("manual_review_status") != "approved"
        or payload.get("dataset_name") != DATASET_NAME
        or payload.get("dataset_tag") != DATASET_TAG
    ):
        raise RuntimeError("baseline report is not manually approved")
    baseline_version = _parse_dataset_version(str(payload["dataset_version"]))
    if baseline_version != expected_dataset_version:
        raise RuntimeError("candidate and baseline dataset versions differ")
    gate = ReleaseGateReport.model_validate(payload["gate"])
    if not gate.passed:
        raise RuntimeError("baseline report did not pass release gates")
    return ReleaseScorecard.model_validate(payload["scorecard"])


if __name__ == "__main__":
    raise SystemExit(main())
