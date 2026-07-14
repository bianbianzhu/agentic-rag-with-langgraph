# Chapter 01: System Contract and Development Loop

## Outcome

The Reference System now has one installable Python package, one compiled LangGraph export, typed mutable State, typed trusted Runtime Context, and a credential-free local development loop. This chapter deliberately answers no knowledge question yet; it proves the application boundary that every later tracer bullet will extend.

## The first trust boundary

`GraphState` is checkpointable application data. The Agent Server may receive it as JSON, so the graph validates nested state contracts before using them. `RuntimeContext` is immutable trusted input for one run. The current Principal identifier enters through Runtime Context, never through the user message or checkpointed State.

```text
Agent Server JSON                         trusted run context
        |                                principal_id="alice"
        v                                       |
  GraphState ----------------------------------+
        |
        v
 compiled engineering_assistant graph
        |
        v
 deterministic answered Current Turn Work
```

The Chapter 01 graph has one deterministic node. It increments the minimal Thread counter and writes a fixed development-loop answer. That behavior is scaffolding, not the final answer pipeline. Current Turn Work remains visible for Studio inspection in this chapter; later chapters replace it with bounded research, verified answers, and terminal Turn Record commit.

## Repository surfaces

- `src/agentic_rag/runtime.py` owns the frozen Runtime Context.
- `src/agentic_rag/conversation.py` owns the minimal Thread State.
- `src/agentic_rag/graph/` owns Graph State, the node adapter, and the compiled `graph` export.
- `langgraph.json` registers that export as `engineering_assistant`.
- `.env.example` documents names without supplying secrets. Core Chapter 01 execution uses `LANGSMITH_TRACING=false` and needs no API key.
- `tests/unit/` observes only the Runtime Context and compiled-graph seams.

## Run the loop

Install the locked environment and run the deterministic checks:

```bash
uv sync --group dev
uv run pytest -q
uv run pyright
```

Start the local Agent Server without opening a browser automatically:

```bash
uv run langgraph dev --no-browser
```

The server exposes the graph at `http://127.0.0.1:2024` and prints a LangSmith Studio URL. Use Studio's graph view because the Reference System intentionally uses custom State rather than `MessagesState`.

A run supplies State and Context separately:

```python
from langgraph_sdk import get_sync_client

client = get_sync_client(url="http://127.0.0.1:2024")
thread = client.threads.create()
result = client.runs.wait(
    thread["thread_id"],
    "engineering_assistant",
    input={
        "thread": {"completed_turns": 0},
        "current_turn": {"user_message": "Start the reference system"},
    },
    context={"principal_id": "alice"},
)
```

The result is deterministic and requires neither OpenAI nor LangSmith credentials. Agent Server owns local checkpoints; the compiled application graph does not create its own checkpointer.

## Verification

Chapter 01 is complete when all of the following are true:

- an empty Principal identifier is rejected;
- missing Current Turn Work is rejected;
- model objects and Agent Server-shaped JSON both complete the same deterministic Turn;
- the full L1 suite passes;
- Pyright reports no errors;
- `langgraph dev` imports `engineering_assistant`, becomes healthy, and completes the SDK Turn through a reused server Thread boundary.

PostgreSQL, Corpus Sync, Access Grants, retrieval, model calls, citations, multi-turn semantic memory, and deployment are not Chapter 01 concerns.
