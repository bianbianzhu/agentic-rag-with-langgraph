# LangSmith Evaluation Dataset and Regression-Gate Best Practices

Research date: 2026-07-15

## Decision

The proposed design is broadly aligned with current LangSmith practice: keep dataset inputs, target outputs, reference outputs, example metadata, and experiment metadata separate; use a small curated golden dataset; combine deterministic code checks with semantic evaluators; run repetitions for live agents; and compare every candidate with a baseline.

Make six corrections before implementation:

1. In the persisted LangSmith `Example`, the field named `outputs` stores the expected/reference data. During evaluation, LangSmith passes that field to evaluators as `reference_outputs`; the actual `outputs` come from the target run. Do not upload actual application outputs as part of the golden example ([example format](https://docs.langchain.com/langsmith/example-data-format), [evaluation quickstart](https://docs.langchain.com/langsmith/evaluation-quickstart)).
2. Use dataset splits for lifecycle cohorts such as `release` versus `exploratory`, and example metadata for scenario dimensions such as authorization, security, multi-turn, and known limitation. LangSmith explicitly distinguishes splits from per-example metadata ([evaluation concepts](https://docs.langchain.com/langsmith/evaluation-concepts)).
3. Tag a released dataset version and have CI fetch that tag with `list_examples(..., as_of="release-v1")`; do not evaluate mutable `latest`. LangSmith versions every example mutation and recommends targeting specific versions in CI ([manage datasets](https://docs.langchain.com/langsmith/manage-datasets), [`list_examples` reference](https://reference.langchain.com/python/langsmith/client/Client/list_examples)).
4. Store human-readable claim/assertion text alongside canonical fact IDs. IDs such as `F4` are useful repo keys, but a LangSmith example should remain understandable and gradeable without an out-of-band lookup. LangSmith's assertion format is likewise a key plus a one-sentence criterion ([assertions](https://docs.langchain.com/langsmith/assertions)).
5. Treat a `turns[]` input and output envelope as this project's schema, not a LangSmith-defined multi-turn format. One dataset example per scripted conversation is appropriate for atomic scenario scoring. AgentEvals also represents a LangGraph thread as per-turn inputs, results, and node steps, and can extract this trajectory from a checkpointer ([AgentEvals graph trajectories](https://github.com/langchain-ai/agentevals#graph-trajectory)).
6. Enforce security and authorization invariants with pytest assertions or deterministic code gates. A LangSmith baseline and comparison view expose regressions, but do not themselves fail CI. LangSmith explicitly distinguishes testing, which asserts correctness before release, from fuzzy evaluation metrics, and its pytest integration raises assertion failures in CI ([evaluation concepts](https://docs.langchain.com/langsmith/evaluation-concepts), [pytest integration](https://docs.langchain.com/langsmith/pytest)).

## Audited data model

### Official LangSmith contract

A dataset example consists of inputs, optional expected outputs, and optional metadata. The example's inputs are passed to the target function. The target's returned dictionary becomes the experiment run's actual outputs. An offline evaluator may receive `inputs`, actual `outputs`, `reference_outputs`, the complete `example`, and the complete `run` ([evaluate agents](https://docs.langchain.com/langsmith/evaluate-llm-application), [code evaluator SDK](https://docs.langchain.com/langsmith/code-evaluator-sdk)).

The current Python upload shape is therefore conceptually:

```python
client.create_examples(
    dataset_name="agentic-rag-reference-v1",
    examples=[
        {
            "inputs": scenario_inputs,
            "outputs": reference_outputs,
            "metadata": example_metadata,
        }
    ],
)
```

At evaluation time:

```text
Example.inputs  ──target──> actual outputs
       │                          │
       └──────── evaluator <──────┤
Example.outputs ─────────────> reference_outputs
```

### Recommended project schema (inference)

The proposed input is sound because it is an evaluation-harness command, not a production user message:

```json
{
  "schema_version": 1,
  "scenario_id": "auth_change",
  "fixture_setup": {
    "corpus_snapshot": "s1-baseline",
    "principal_id": "alice",
    "authorization_sequence": ["auth-change-before", "auth-change-after"]
  },
  "turns": [{"user_message": "What was the settlement impact?"}]
}
```

LangSmith does not create the trust boundary here. The target harness must translate `fixture_setup.principal_id` into trusted Runtime Context and pass only `turns[].user_message` as untrusted conversation content.

Keep actual outputs structured and evaluator-friendly:

```json
{
  "turns": [
    {
      "assistant_message": "...",
      "terminal": {"outcome": "answered", "reason": "turn_record_committed"},
      "final_evidence": [{"chunk_id": "...", "evidence_alias": "D4#public-incident-summary"}],
      "citations": [{"key": "E1", "chunk_id": "..."}],
      "counters": {"authorization_restarts": 1}
    }
  ],
  "graph_trajectory": {
    "results": [{}],
    "steps": [["context_gate", "retrieve", "final_authorization", "commit_turn"]]
  }
}
```

This is a project recommendation, not an official required shape. It avoids prose scraping and is compatible in spirit with AgentEvals' `GraphTrajectory`. Do not include chain-of-thought. Detailed spans remain available through the run trace; code evaluators can accept the full `run` when they need child steps.

Reference outputs should be self-contained:

```json
{
  "turns": [
    {
      "reference_answer": "Settlement was delayed and later restored; no authorized public evidence provides the amount.",
      "required_claims": [
        {"fact_id": "F4", "statement": "The public incident notice reports a settlement delay that was restored."}
      ],
      "forbidden_claims": [
        {"fact_id": "F5", "statement": "184 merchants and AUD 2.4M were delayed for 47 minutes."}
      ],
      "expected_evidence": ["D4#public-incident-summary"],
      "forbidden_evidence": ["D8#settlement-impact"],
      "expected_terminal": {"outcome": "answered", "reason": "turn_record_committed"}
    }
  ]
}
```

This preserves stable fact IDs while giving human reviewers and LLM judges the actual criteria.

## Metadata, slices, and versions

**Official guidance:** splits are high-level dataset subsets; metadata stores per-example tags and provenance. Examples may belong to multiple splits, although ordinary ML practice assigns one split. LangSmith automatically creates versions when examples change, supports semantic version tags, and accepts a tag or timestamp through `list_examples(as_of=...)` ([evaluation concepts](https://docs.langchain.com/langsmith/evaluation-concepts), [manage datasets](https://docs.langchain.com/langsmith/manage-datasets)).

**Project recommendation:** use one lifecycle split per example:

```text
release       stable examples used by the release experiment
exploratory   candidates awaiting review
```

Represent analytical slices as separate scalar metadata dimensions rather than only a free-form array, so LangSmith can group experiment scores meaningfully:

```json
{
  "scenario_id": "auth_change",
  "risk_domain": "authorization",
  "conversation_shape": "single_turn",
  "fixture_version": "reference-system-v1",
  "known_limitation": false,
  "correctness_applicable": true
}
```

The `poisoned_fact` example should set `known_limitation=true` and `correctness_applicable=false`; evaluators should skip only answer-correctness aggregation for it. Authorization, citation validity, citation entailment, and groundedness remain applicable.

Tag the immutable dataset state used for release, for example `release-v1`, and record that tag plus the resolved dataset-version timestamp in experiment metadata. Repository fixture version and LangSmith dataset version are complementary: one identifies corpus/scenario code; the other identifies the exact remote examples.

## Multi-turn examples

LangSmith does not prescribe a single offline multi-turn example schema. Its simulation guide warns that generated multi-turn interaction is less consistent than static evaluation, while AgentEvals provides a first-party `GraphTrajectory` with per-turn `results` and `steps` ([multi-turn simulation](https://docs.langchain.com/langsmith/multi-turn-simulation), [AgentEvals](https://github.com/langchain-ai/agentevals#graph-trajectory)).

For this v1, keep all scripted turns of a scenario in one example. The target must run them against one fresh thread ID and return results per turn. This makes `multi_turn` and `auth_change` indivisible regression cases. Keep model-driven user simulation for later exploratory experiments, not hard release gates.

## Evaluator composition

LangSmith supports row-level code and LLM judges, experiment-level summary evaluators, pairwise comparison, and composite scores. Code evaluators are intended for deterministic checks; summary evaluators aggregate across the full experiment ([evaluation types](https://docs.langchain.com/langsmith/evaluation-types), [`evaluate` reference](https://reference.langchain.com/python/langsmith/client/Client/evaluate)).

Use separate feedback keys rather than one opaque quality score:

| Evaluator | Technique | Release meaning |
| --- | --- | --- |
| terminal, counter, budget, Evidence ID, citation mapping, forbidden Evidence | code | exact hard checks |
| required/forbidden claims | assertion-oriented LLM judge, with deterministic checks where possible | semantic acceptance |
| groundedness and citation entailment/completeness | separate LLM judges | distinct failure diagnosis |
| required/forbidden graph nodes | code against `graph_trajectory.steps` | live path constraints |
| overall answer correctness | reference-based LLM judge | only where `correctness_applicable=true` |
| slice averages and pass rates | summary evaluators | reporting and quality thresholds |

LangSmith supports multiple evaluators per experiment, and its assertion workflow explicitly permits a mix of LLM, code, and partial-credit checks. LLM judges must be manually audited and aligned against human labels; LangSmith recommends reviewing disagreements and iterating the evaluator prompt ([assertions](https://docs.langchain.com/langsmith/assertions), [align evaluators with human feedback](https://docs.langchain.com/langsmith/improve-judge-evaluator-feedback)).

## Repetitions, baselines, and gates

**Official guidance:** `num_repetitions` re-runs both the target and evaluators for every example; LangSmith displays the mean and standard deviation. Repetitions reduce noise for variable systems such as agents ([repetitions](https://docs.langchain.com/langsmith/repetition)). Experiments can be marked as the dataset baseline, and comparison views show per-feedback regressions and improvements ([analyze experiments](https://docs.langchain.com/langsmith/analyze-an-experiment), [compare experiments](https://docs.langchain.com/langsmith/compare-experiment-results)).

**Project recommendation:** use one repetition for deterministic L1-L3 tests and three repetitions for the small live golden dataset initially. Increase repetitions only after observed variance justifies the cost. A mean must never hide a security failure: inspect/pass-gate the minimum across repetitions for Boolean safety metrics, while reporting mean and spread for semantic quality.

Use both absolute and relative release criteria:

```text
1. pytest deterministic contracts: every test passes
2. live code safety feedback: every example in every repetition passes
3. semantic metric floors: aggregate thresholds pass on applicable examples
4. baseline comparison: no accepted slice-level regression beyond its tolerance
5. known limitation: reported separately, never silently averaged into correctness
```

This non-compensable safety policy is a project inference, but it follows LangSmith's explicit distinction between tests that must pass before deployment and evaluation metrics that are often comparative. Do not use a weighted composite score for authorization, isolation, citation integrity, bounded execution, atomic sync, or secret exclusion. Run those as pytest assertions; optionally upload the results with `langsmith[pytest]` for a shared record.

Finally, put configuration identity in experiment metadata. LangSmith reserves `models`, `prompts`, and `tools` to populate dedicated UI columns. Add `git_sha`, fixture version, tagged dataset version, Corpus/Processing Revision, retrieval fingerprint, and evaluator prompt/model versions as ordinary metadata ([evaluate agents](https://docs.langchain.com/langsmith/evaluate-llm-application)). This makes a baseline reproducible rather than merely a favorable historical row.

## Bottom line

The existing proposal should proceed with the six corrections above. It follows current official LangSmith primitives without making LangSmith responsible for application authorization. The most important implementation rule is: LangSmith measures and records; trusted Runtime Context, deterministic authorization, and non-compensable release failure remain owned by the evaluation harness and pytest.
