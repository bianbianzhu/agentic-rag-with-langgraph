# Chapter 11: Evaluation and Release Evidence

## Outcome

The Reference System now has a release workflow that keeps deterministic safety
and live semantic quality in separate, non-compensating lanes. L1-L3 and the
Agent Server smoke must pass before a paid L4 experiment starts. A strong judge
score can never excuse an authorization, citation, isolation, budget, atomicity,
or secret-exclusion failure.

The live lane uses one LangSmith dataset, an exact dataset version, real OpenAI
models, real PostgreSQL retrieval, three repetitions, code evaluators, distinct
LLM judges, and a machine-readable release report. It does not deploy the
application.

## The two-lane gate

```text
L1 unit + L2 data + L3 graph + Agent Server smoke
                         |
                         | all pass
                         v
              exact LangSmith dataset version
                    /                 \
          release split           diagnostic split
          9 examples              poisoned_fact
          10 Turns                1 example / 1 Turn
                    \                 /
                     v               v
                  review artifacts + traces
                             |
                      human acceptance
                             |
                     tag release-v1
```

The release split owns release correctness and performance floors. The
diagnostic split proves a different point: an answer can be authorized,
grounded, and correctly cited while still repeating a false source. A separate
truth-oriented reference makes answer correctness fail visibly, while the
release aggregate excludes the entire diagnostic split. Its terminal,
authorization, citation, and forbidden-claim safety checks must still pass all
three repetitions.

## One self-contained golden dataset

`agentic-rag-reference-v1` contains ten Examples and eleven Turns:

- nine release Examples: `happy`, `greeting`, `clarification`, `refine`,
  `auth_change`, `document_injection`, `multi_turn`,
  `relevant_not_allowed`, and `finance_authorized`;
- one diagnostic Example: `poisoned_fact`; and
- both `multi_turn` messages in one Example and one fresh Thread.

Each Example keeps three roles separate:

| Field | Role |
| --- | --- |
| `inputs` | trusted fixture command plus untrusted user messages |
| `outputs` | reference outcomes, Evidence aliases, claims, routes, and bounds |
| `metadata` | scenario dimensions, fixture identity, and applicability |

The target rejects any command that does not exactly match the local golden
fixture. LangSmith supplies work; it does not establish Principal authority.
Only the command's locally verified `principal_id` becomes Runtime Context, and
only `turns[].user_message` becomes Untrusted Content.

## The live target

The L4 target uses:

- `openai:gpt-5.4-mini-2026-03-17` for agent decisions and answers;
- `openai:gpt-5.4-nano-2026-03-17` for listwise reranking;
- `openai:text-embedding-3-small` for live fixture embeddings;
- the compiled top-level LangGraph with a fresh in-memory checkpointer; and
- the same PostgreSQL authorization predicate and retrieval implementation used
  by L2 and L3.

Its structured output contains only checkable, chain-of-thought-free data:
assistant messages, Standalone Questions, final authorized Evidence, citation
mappings, selected Knowledge Sources, node names, counters, security events,
latency, and per-Turn budget measurements. Detailed spans remain in LangSmith.

`auth_change` revokes Alice's direct finance grant after the first verified
draft. The final authorization gate must discard that draft, restart once, and
return only public Evidence.

## Evaluators

Code evaluators enforce hard structure:

- terminal outcome and reason;
- forbidden Evidence, Knowledge Sources, trajectory nodes, claims, and
  disclosure canaries;
- citation keys resolving only to final Evidence;
- expected-Evidence Recall@5 and final-Evidence MRR; and
- model, retrieval, refinement, repair, authorization-restart, Evidence item,
  Evidence token, Thread token, answer token, and 90-second Turn limits.

Distinct OpenEvals judges score answer correctness, groundedness, relevance,
helpfulness, required-claim coverage, forbidden-claim absence, citation
entailment, citation completeness, and retrieval relevance. The report records
means and population standard deviations rather than hiding repetition
variation behind one number.

Every release safety evaluator must pass every repetition. Every ordinary
evaluator that is applicable to a release Example must pass at least two of
three repetitions. Greeting, clarification, and safe authorization abstention
use answer relevance, helpfulness, and retrieval relevance; factual and
citation judges return an explicit not-applicable result. The applicability
matrix is stored in each Example's metadata and is also the release gate's
coverage contract. Aggregate floors, latency, token, cost, and accepted-baseline
regression limits are then applied without a composite score.

## Candidate before tag

`release-v1` is never moved by an experiment. Dataset publication, evaluation,
and acceptance are deliberately separate:

```bash
# 1. Local validation; no credentials, database, model, or network required.
uv run python -m evals.run preview

# 2. Upsert one untagged candidate. Save its exact dataset_version.
uv run python -m evals.run publish

# 3. Run release and diagnostic experiments against that exact timestamp.
uv run python -m evals.run experiment \
  --dataset-version <ISO-8601 timestamp> \
  --initial-release

# A later candidate must use the version currently tagged release-v1.
uv run python -m evals.run experiment \
  --dataset-version <ISO-8601 timestamp> \
  --baseline-report evals/artifacts/accepted-release-report.json

# 4. Inspect every golden output, citation mapping, score, explanation, and trace.
# Only then certify the review and move release-v1 to the exact candidate version.
uv run python -m evals.run accept \
  --candidate-report evals/artifacts/candidate-report.json \
  --human-approved
```

The accepted JSON report records the dataset timestamp, tag, both experiment
links, diagnostic contrast, score variation, release scorecard, baseline
comparison, and every gate violation. Secrets and raw traces are excluded. Set
the accepted release experiment as the LangSmith baseline in the UI only after
the `accept` command passes.

For a baseline comparison, the command resolves `release-v1` again and rejects
an accepted report whose timestamp is no longer the tag's current version. It
then loads release Examples by the tag, not by mutable `latest` or a historical
timestamp supplied only by the report.

## Reproducible experiment identity

Experiment metadata records reserved `models`, `prompts`, and `tools` fields,
plus git SHA, fixture version, exact dataset version, Corpus Revisions,
Processing Revision, retrieval fingerprint, evaluator identities, and judge
model. The advertised retrieval tool schema contains query intent and Knowledge
Sources, never Principal or Access Scope arguments.

The client-side trace anonymizer removes OpenAI keys, LangSmith keys, and
database credentials before upload. Local artifacts contain aggregates and
LangSmith links, not secrets.

## Run all non-live verification

```bash
docker compose up -d postgres
uv run pytest -q --ignore=tests/smoke
uv run pytest tests/smoke/test_agent_server.py -q
uv run pyright
uv lock --check
git diff --check
```

L4 additionally requires `OPENAI_API_KEY` and `LANGSMITH_API_KEY`. Production
deployment, production authentication and RLS, scaling, online monitoring, and
operations remain outside development v1.

## Official references

- [Run an evaluation with the LangSmith SDK](https://docs.langchain.com/langsmith/evaluate-llm-application)
- [Manage and version datasets](https://docs.langchain.com/langsmith/manage-datasets)
- [LangSmith `Client.evaluate` reference](https://reference.langchain.com/python/langsmith/client/Client/evaluate)
- [Fetch experiment performance metrics](https://docs.langchain.com/langsmith/fetch-perf-metrics-experiment)
