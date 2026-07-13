# AGENTS.md

## Project

Engineering-first course for building a multi-agent RAG system with LangGraph. Separate retrieval, embeddings, indexing, reranking, generation, and verification into agents. Cover core modules through deployment and monitoring.

## Language

- Project internals: English only.
- Documentation: English first, then Chinese counterpart.
- Chinese docs: `*.zh.md`; local-only, ignored by Git.

## Simplicity First

Minimum code that solves the request. Nothing speculative.

- No unrequested features.
- No abstraction for single-use code.
- No unrequested flexibility or configuration.
- No handling for impossible cases.
- If 200 lines can be 50, rewrite.

If a senior engineer would call it overcomplicated, simplify.

## Surgical Changes

Touch only what the request requires. Clean only your own mess.

- Do not improve adjacent code, comments, or formatting.
- Do not refactor working code.
- Match existing style.
- Mention unrelated dead code; do not remove it.
- Remove only imports, variables, or functions made unused by your change.

Every changed line must trace to the request.

## Goal-Driven Execution

Define verifiable success criteria. Work until verified.

- Validation: test invalid inputs, then pass tests.
- Bug fix: reproduce with a test, then pass it.
- Refactor: tests pass before and after.

For multi-step work, state a short plan:

```text
1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
3. [Step] -> verify: [check]
```

Prefer concrete checks over vague goals.
