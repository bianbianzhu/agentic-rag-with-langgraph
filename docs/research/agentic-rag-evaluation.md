# Agentic RAG Evaluation Methods

Research date: 2026-07-13

## Decision

Use two evaluation lanes:

1. **Deterministic `pytest` gates** for contracts and mechanically checkable behavior. These run on every pull request and must not require an LLM judge.
2. **LangSmith offline experiments** for semantic quality and live-model behavior. Run them on a versioned dataset, compare them with a baseline, and use repetitions where model variance matters.

This split follows LangSmith's own distinction: unit tests are rule-based, consistent checks suitable for CI, while regression tests compare application versions on curated datasets. LangSmith supports code rules, LLM-as-judge, pairwise, and summary evaluators rather than requiring one evaluator type for every property ([evaluation types](https://docs.langchain.com/langsmith/evaluation-types), [evaluation overview](https://docs.langchain.com/langsmith/evaluation)).

Keep the target output structured so evaluators do not scrape prose:

```text
inputs:            messages, principal
outputs:           answer, retrieved_hits, citations, messages/trajectory
reference_outputs: relevant_chunk_ids, expected_sources/tools,
                   reference_answer, expected_trajectory, assertions
```

## Evaluation matrix

| Area | Deterministic `pytest` | LangSmith offline evaluation |
| --- | --- | --- |
| Retrieval | Against a frozen corpus, assert ACL filtering, stable chunk IDs, deduplication, top-k bounds, and metric implementations. Compute label-based `Recall@k`, `Precision@k`, `MRR`, and, only if graded relevance labels exist, `NDCG@k`. | Store relevant chunk/document IDs in `reference_outputs`; use code evaluators per example and a summary evaluator for dataset aggregates. Add an LLM judge only for semantic retrieval relevance when exhaustive labels are unavailable. LangSmith explicitly supports custom code evaluators and dataset-level summary evaluators; its RAG guidance separately defines retrieval relevance as retrieved context versus input ([RAG tutorial](https://docs.langchain.com/langsmith/evaluate-rag-tutorial), [summary evaluators](https://docs.langchain.com/langsmith/summary)). |
| Source and tool selection | With a stubbed model/router, assert the selected source, allowed tool names, exact security-sensitive arguments, and that forbidden tools/sources are never called. Also assert retry/tool-call budgets. | For cases with one required path, use a code evaluator or AgentEvals trajectory matching. `create_trajectory_match_evaluator` supports `strict`, `unordered`, `subset`, and `superset`, plus tool-argument matching controls. For cases with several valid paths, use an LLM judge with a rubric or OpenEvals' `TOOL_SELECTION_PROMPT` ([trajectory evaluation](https://docs.langchain.com/langsmith/trajectory-evals), [AgentEvals source](https://github.com/langchain-ai/agentevals), [OpenEvals source](https://github.com/langchain-ai/openevals)). |
| Multi-turn behavior and trajectory | Use scripted turns and fixed model/tool doubles to assert thread persistence, thread isolation, principal isolation, follow-up resolution, bounded loops, checkpoint resume, and deterministic graph transitions. | Evaluate fixed message histories/reference trajectories first. AgentEvals can match tool-call trajectories or LangGraph node trajectories. For broader conversational behavior, use `openevals.simulators.run_multiturn_simulation` with `create_llm_simulated_user`, then judge the resulting trajectory; OpenEvals also provides `TASK_COMPLETION_PROMPT` and `KNOWLEDGE_RETENTION_PROMPT`. Simulation is less consistent than static examples, so it belongs in experiments rather than the hard unit lane ([multi-turn simulation](https://docs.langchain.com/langsmith/multi-turn-simulation), [AgentEvals source](https://github.com/langchain-ai/agentevals), [OpenEvals source](https://github.com/langchain-ai/openevals)). |
| Answer correctness | Assert output schema, refusal shape, empty-evidence behavior, and exact results only for deterministic fixtures. Do not assert natural-language equality from a live model. | Compare answer to `reference_outputs["answer"]` with an LLM judge. The current reusable API is `openevals.llm.create_llm_as_judge` with `openevals.prompts.CORRECTNESS_PROMPT`; LangSmith's RAG guidance classifies correctness as response versus reference answer ([OpenEvals source](https://github.com/langchain-ai/openevals), [RAG tutorial](https://docs.langchain.com/langsmith/evaluate-rag-tutorial)). |
| Groundedness | Assert that generation receives only the selected evidence and that an evidence-free path abstains. These checks validate wiring, not semantic entailment. | Judge answer versus retrieved context with `create_llm_as_judge` and `RAG_GROUNDEDNESS_PROMPT`. This evaluator does not require a reference answer because it compares output with context ([OpenEvals source](https://github.com/langchain-ai/openevals), [RAG tutorial](https://docs.langchain.com/langsmith/evaluate-rag-tutorial)). |
| Citation validity | Parse citations and assert every citation resolves to a retrieved, ACL-authorized chunk; IDs and source metadata match; quoted spans occur in the cited chunk; and no invented URI is emitted. These are code checks. | Add two semantic scores where needed: **citation entailment** (does the cited passage support the associated claim?) and **citation completeness** (do all externally verifiable claims carry support?). Implement them as custom LLM judges or assertion evaluators. LangSmith's assertion workflow explicitly shows `must_cite_source` and `must_not_invent_url`, and permits code, regex/schema, or LLM scoring ([assertions](https://docs.langchain.com/langsmith/assertions)). Groundedness alone must not be reported as citation validity because it does not verify citation-to-claim mapping. |
| Latency | Use fake clocks to test deadlines/timeouts and counters to test retry budgets. A small local performance test may catch algorithmic explosions, but live wall-clock latency must not be a unit-test gate. | LangSmith experiments expose latency, first-token latency, token counts, costs, error rate, and feedback statistics. After `evaluate`, fetch aggregate statistics with `Client.read_project(project_name=results.experiment_name, include_stats=True)`; compare `latency_p50`/`latency_p99` to a baseline or explicit budget ([performance metrics](https://docs.langchain.com/langsmith/fetch-perf-metrics-experiment)). |
| Token usage and estimated cost | With a fake model response, assert context/token budget enforcement and that the workflow stops before configured limits. Do not hard-code live tokenizer/pricing results as unit tests. | LangSmith records token usage and automatically estimates cost for supported traced LangChain/provider calls when token counts, model/provider, and prices are available; custom run costs can be submitted manually. Cost is an estimate tied to the configured price table, not an invoice ([cost tracking](https://docs.langchain.com/langsmith/cost-tracking)). Track per-experiment totals and per-example outliers. |

## Current Python APIs and packages

Use these APIs rather than the older `langchain.evaluation`/classic evaluator surface:

- `langsmith`: `Client.create_dataset`, `Client.create_examples`, and `Client.evaluate`/`Client.aevaluate`. `evaluate` accepts `evaluators`, `summary_evaluators`, `num_repetitions`, concurrency, and experiment metadata ([Python API reference](https://reference.langchain.com/python/langsmith/client/Client/evaluate)).
- `langsmith[pytest]`: optional LangSmith tracking for pytest via `@pytest.mark.langsmith` and `langsmith.testing`; `LANGSMITH_TEST_TRACKING=false` provides a local dry run. The documented integration requires `langsmith>=0.3.4` ([pytest integration](https://docs.langchain.com/langsmith/pytest)). Plain `pytest` remains sufficient for the deterministic lane.
- `openevals`: reusable LLM judges and prompts. Relevant imports are `create_llm_as_judge`, `CORRECTNESS_PROMPT`, `RAG_GROUNDEDNESS_PROMPT`, `RAG_RETRIEVAL_RELEVANCE_PROMPT`, `TOOL_SELECTION_PROMPT`, `TASK_COMPLETION_PROMPT`, and `KNOWLEDGE_RETENTION_PROMPT`; its simulator provides `run_multiturn_simulation` and `create_llm_simulated_user` ([OpenEvals source](https://github.com/langchain-ai/openevals)).
- `agentevals`: deterministic message/tool trajectory matching, LLM trajectory judges, and LangGraph trajectory extraction/matching. The main Python entry points include `create_trajectory_match_evaluator` and `create_trajectory_llm_as_judge` ([AgentEvals source](https://github.com/langchain-ai/agentevals), [LangChain agent-eval guide](https://docs.langchain.com/oss/python/langchain/test/evals)).

Suggested development dependencies:

```bash
uv add --dev pytest "langsmith[pytest]" openevals agentevals
```

The application should still run without a LangSmith API key. Only tracked tests and offline experiments require LangSmith access.

## Dataset and regression policy

Create one small, versioned golden dataset with slices for single-turn, follow-up/coreference, cross-document, unanswerable, unauthorized, source-routing, and retry/fallback cases. Each example should label only what is knowable: relevant IDs, required/allowed tools, a reference answer or assertions, and an expected trajectory when there is genuinely one correct path. Start with manually curated examples; LangSmith recommends examples of good retrievals, answers, tool selection, and trajectories for critical components ([evaluation concepts](https://docs.langchain.com/langsmith/evaluation-concepts)).

Run regressions in two tiers:

1. **Per pull request:** plain deterministic pytest; optionally upload results with the LangSmith pytest plugin. Fail on any contract, ACL, citation-integrity, or bounded-execution violation.
2. **Prompt/model/retrieval changes and scheduled runs:** `Client.aevaluate` on the golden dataset, with `num_repetitions` for non-deterministic evaluators. Record model, prompt, index/corpus version, and git revision in experiment metadata. LangSmith supports setting an experiment baseline and highlights score regressions; its experiment view also includes latency, token, and cost metrics ([experiment analysis](https://docs.langchain.com/langsmith/analyze-an-experiment), [experiment configuration](https://docs.langchain.com/langsmith/experiment-configuration)).

Gate the second tier on aggregate minimums and explicit safety invariants, not on every individual LLM-judge result. Investigate per-example regressions in LangSmith's comparison view before accepting a new baseline ([compare experiments](https://docs.langchain.com/langsmith/compare-experiment-results)).

## Recommended v1 scorecard

- Retrieval: `Recall@k`, `MRR`, and retrieval relevance; add `NDCG@k` only after graded labels exist.
- Routing: expected/allowed source and tool selection, argument validity, forbidden-tool rate, tool-call count.
- Conversation: follow-up resolution, knowledge retention, task completion, thread/principal isolation.
- Answer: correctness, groundedness, refusal correctness, citation validity, citation completeness.
- System: p50/p99 latency, first-token latency when streaming, total/input/output tokens, estimated cost, errors, retries.

Do not collapse these into one opaque "RAG quality" number. Keep component scores visible so a generation improvement cannot hide a retrieval, authorization, or citation regression.

