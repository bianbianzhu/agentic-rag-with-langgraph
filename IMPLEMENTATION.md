# Reference System Implementation Entry Point

This is the canonical entry point for implementing the locally runnable v1 Reference System. Start every implementation session here. Do not restart architecture discovery unless implementation exposes a genuinely missing decision.

## Read in this order

1. [`AGENTS.md`](AGENTS.md) for repository rules.
2. [`CONTEXT.md`](CONTEXT.md) for the domain language that code, tests, and course material must use.
3. The [Wayfinder map](.scratch/production-agentic-rag-v1/map.md) for the low-resolution decision index and explicit v1 exclusions.
4. [Define codebase and course boundaries](.scratch/production-agentic-rag-v1/issues/11-define-codebase-and-course-boundaries.md) for package ownership, model defaults, repository layout, test layers, and the authoritative eleven-chapter sequence.
5. Only the decision tickets linked for the current chapter below. Open research reports or prototypes only when a linked decision points to them or a current implementation fact must be verified.

When artifacts differ, use this priority: repository instructions, resolved decision tickets, research reports, then explanatory prototypes. A resolved project decision may intentionally override a research recommendation.

## How to continue

1. Find the single `next` row in the ledger below. Earlier rows must be `verified`; later rows remain `blocked`.
2. Read that chapter's linked decisions and inspect the current code and tests. Do not preload every ticket.
3. Invoke `$tdd` for behavior and integration work, then `$implement` for the bounded chapter scope. Use `$to-tickets` first only if the current chapter cannot fit one implementation session.
4. State a short plan with a concrete verification command for every step.
5. Add the smallest failing test for an invalid or unsafe case, make it pass, then add the happy path. Use the real PostgreSQL boundary at L2/L3; do not create a fake retriever.
6. Run the narrow changed tests, then every previously available non-live test layer. Never use an L4 score to excuse an L1-L3 failure.
7. Write or update the English `docs/course/NN-*.md` chapter after behavior is verified. Maintain the ignored local `*.zh.md` counterpart when requested.
8. Record the exact commands and results in the implementation handoff. Change the current row to `verified`, put its immutable `course-*` tag in the ledger only after the user has approved creating the tag, and promote exactly one following row from `blocked` to `next`.
9. Stop at the chapter boundary. Do not pull later chapter features forward.

## Progress ledger

| Chapter | Status | Required decision context | Minimum verification reached | Course tag |
| --- | --- | --- | --- | --- |
| 01 System contract and development loop | `verified` | [development contracts](.scratch/production-agentic-rag-v1/issues/01-verify-current-langgraph-development-contracts.md), [graph prototype decision](.scratch/production-agentic-rag-v1/issues/07-prototype-the-reference-system-graph.md), [security controls](.scratch/production-agentic-rag-v1/issues/09-define-authorization-and-security-controls.md), [codebase boundaries](.scratch/production-agentic-rag-v1/issues/11-define-codebase-and-course-boundaries.md) | L1 plus compiled-graph import and local Agent Server load | — |
| 02 Corpus as a transaction | `verified` | [retrieval stack](.scratch/production-agentic-rag-v1/issues/04-choose-the-local-retrieval-stack.md), [Corpus Sync](.scratch/production-agentic-rag-v1/issues/05-define-the-corpus-sync-contract.md), [fixtures](.scratch/production-agentic-rag-v1/issues/12-define-the-sample-corpus-and-fixtures.md) | L1 + L2 sync and atomic-failure cases | — |
| 03 Authorization before relevance | `verified` | [security controls](.scratch/production-agentic-rag-v1/issues/09-define-authorization-and-security-controls.md), [fixtures](.scratch/production-agentic-rag-v1/issues/12-define-the-sample-corpus-and-fixtures.md) | L1 + L2 grant matrix and default deny | — |
| 04 Authorized hybrid retrieval | `verified` | [retrieval stack](.scratch/production-agentic-rag-v1/issues/04-choose-the-local-retrieval-stack.md), [retrieval contract](.scratch/production-agentic-rag-v1/issues/06-define-the-evidence-and-retrieval-contract.md), [security controls](.scratch/production-agentic-rag-v1/issues/09-define-authorization-and-security-controls.md) | L1 + L2 dense/lexical shared-scope retrieval | — |
| 05 Reranking and the Evidence Set | `next` | [retrieval contract](.scratch/production-agentic-rag-v1/issues/06-define-the-evidence-and-retrieval-contract.md), [quality gates](.scratch/production-agentic-rag-v1/issues/10-define-the-evaluation-dataset-and-quality-gates.md), [model defaults](.scratch/production-agentic-rag-v1/issues/11-define-codebase-and-course-boundaries.md) | L1 + L2 stable reranking and bounded Evidence | — |
| 06 Answers that can be checked | `blocked` | [retrieval and citation contract](.scratch/production-agentic-rag-v1/issues/06-define-the-evidence-and-retrieval-contract.md), [graph decision](.scratch/production-agentic-rag-v1/issues/07-prototype-the-reference-system-graph.md), [security controls](.scratch/production-agentic-rag-v1/issues/09-define-authorization-and-security-controls.md) | L1 + L3 verified-answer and citation failures | — |
| 07 Bounded agentic research | `blocked` | [graph decision](.scratch/production-agentic-rag-v1/issues/07-prototype-the-reference-system-graph.md), [quality gates](.scratch/production-agentic-rag-v1/issues/10-define-the-evaluation-dataset-and-quality-gates.md) | L1 + L3 routing, refinement, and budget terminals | — |
| 08 Multi-turn memory without authority drift | `blocked` | [graph decision](.scratch/production-agentic-rag-v1/issues/07-prototype-the-reference-system-graph.md), [Thread policy](.scratch/production-agentic-rag-v1/issues/08-define-conversation-thread-context-policy.md), [security controls](.scratch/production-agentic-rag-v1/issues/09-define-authorization-and-security-controls.md) | L1 + L3 two-Turn, resume, and commit behavior | — |
| 09 Authorization changes mid-conversation | `blocked` | [Thread policy](.scratch/production-agentic-rag-v1/issues/08-define-conversation-thread-context-policy.md), [security controls](.scratch/production-agentic-rag-v1/issues/09-define-authorization-and-security-controls.md), [quality gates](.scratch/production-agentic-rag-v1/issues/10-define-the-evaluation-dataset-and-quality-gates.md) | L1-L3 restart, revocation, and mismatch cases | — |
| 10 Adversarial and failure scenarios | `blocked` | [security controls](.scratch/production-agentic-rag-v1/issues/09-define-authorization-and-security-controls.md), [quality gates](.scratch/production-agentic-rag-v1/issues/10-define-the-evaluation-dataset-and-quality-gates.md), [fixtures](.scratch/production-agentic-rag-v1/issues/12-define-the-sample-corpus-and-fixtures.md) | Complete deterministic L1-L3 security suite | — |
| 11 Evaluation and release evidence | `blocked` | [evaluation methods](.scratch/production-agentic-rag-v1/issues/03-establish-agentic-rag-evaluation-methods.md), [development contracts](.scratch/production-agentic-rag-v1/issues/01-verify-current-langgraph-development-contracts.md), [quality gates](.scratch/production-agentic-rag-v1/issues/10-define-the-evaluation-dataset-and-quality-gates.md) | L1-L3, Agent Server smoke, tagged L4 experiment | — |

Allowed statuses are `blocked`, `next`, `in_progress`, and `verified`. Keep exactly one `next` or `in_progress` row until all chapters are verified.

## Current next action

The next implementation session begins Chapter 05 only: Reranking and the Evidence Set. Read its three linked decisions before planning, then add the global structured reranker, candidate validation, context expansion, bounded Evidence assembly, deterministic failure, and complete retrieval fingerprints. Chapters 01 through 04 are verified; do not alter their public seams unless Chapter 05 evidence requires a compatible extension.

v1 ends after Chapter 11. Deployment, a custom web UI/API, production authentication, production RLS, scaling, online monitoring, and operations remain out of scope for this implementation route.
