# Current LangGraph development contracts

Checked against the official Python documentation and references on 2026-07-13.

## Recommendation

Use an exported, already-compiled `StateGraph` as the application boundary, register it in `langgraph.json`, and run it with `langgraph dev`. Let Agent Server own checkpointing and its store; use LangGraph state for mutable conversation data and typed runtime context for immutable per-run dependencies. Exercise the server through LangSmith Studio and `langgraph-sdk`, not through a second FastAPI wrapper.

## Packages

The supported local server requires Python 3.11 or newer and the `inmem` CLI extra. The application itself needs `langgraph`; the Python SDK is only needed by API clients or integration tests. The official starter currently uses `langgraph>=1.0.0` and `langgraph-cli[inmem]>=0.4.14` ([local-server guide](https://docs.langchain.com/oss/python/langgraph/local-server), [official starter `pyproject.toml`](https://github.com/langchain-ai/new-langgraph-project/blob/main/pyproject.toml)).

For this OpenAI-backed application, add `langchain[openai]`. Application code can use `init_chat_model` and `init_embeddings`, but the OpenAI provider integration is still installed underneath by that extra; model-provider integrations are separate packages ([model initialization](https://docs.langchain.com/oss/python/langchain/models), [`init_embeddings` reference](https://reference.langchain.com/python/langchain/embeddings/base/init_embeddings)).

Suggested ownership:

| Package | Purpose | Dependency group |
| --- | --- | --- |
| `langgraph>=1.0` | Graph API and runtime | application |
| `langchain[openai]` | Unified LangChain model factories plus the OpenAI integration | application |
| `langgraph-cli[inmem]>=0.4.14` | `langgraph dev` and hot reload | development |
| `langgraph-sdk` | Typed Agent Server client used by tests or another client process | development until a client ships |

Do not add a separate checkpointer package merely to support `langgraph dev`; Agent Server injects its checkpointer and store ([Agent Server graph loading](https://docs.langchain.com/langsmith/agent-server)).

## Application and graph export

`langgraph.json` is the CLI/application manifest. Its required core is a dependency location and a mapping from a stable graph ID to a Python module export. The graph mapping may point to a compiled graph or a factory, but exporting a compiled graph is the official recommendation because it is loaded once; a factory is for genuine per-run topology construction and is invoked repeatedly ([application structure](https://docs.langchain.com/langsmith/application-structure), [Agent Server graph loading](https://docs.langchain.com/langsmith/agent-server)).

```json
{
  "$schema": "https://langgra.ph/schema.json",
  "dependencies": ["."],
  "graphs": {
    "agent": "./src/agentic_rag/graph.py:graph"
  },
  "env": ".env"
}
```

The Python export should therefore have this shape:

```python
builder = StateGraph(State, context_schema=Context)
# add nodes and edges
graph = builder.compile(name="agentic-rag")
```

Do not pass an application-created checkpointer or store to `compile()` when this export is loaded by Agent Server. The server must inject those resources so its thread, history, interrupt, and time-travel operations share the same persistence layer ([Agent Server graph loading](https://docs.langchain.com/langsmith/agent-server)).

## State, runtime context, and configuration

These are distinct contracts:

- **State** is mutable graph data. On a stateful server run it is checkpointed into a Conversation Thread and becomes the short-term memory used by later runs on that thread.
- **Runtime context** is immutable data/dependencies for one run. Define it with `StateGraph(..., context_schema=Context)`, accept `runtime: Runtime[Context]` in nodes, and read `runtime.context`. `config_schema` is deprecated; `context_schema` is its replacement ([`StateGraph` reference](https://reference.langchain.com/python/langgraph/graph/state/StateGraph), [Graph API runtime context](https://docs.langchain.com/oss/python/langgraph/graph-api)).
- **`RunnableConfig`** carries execution controls and tracing/configurable fields. A node that needs it should accept a separate `config: RunnableConfig` parameter; it is not a field on `Runtime` ([`Runtime` reference](https://reference.langchain.com/python/langgraph/runtime/Runtime)).
- **Assistant context** is a saved runtime-context configuration for a graph. Agent Server creates a default assistant for each registered graph; additional assistants can store different context values. The SDK also accepts `context` on a run for a per-run value/override ([assistant configuration](https://docs.langchain.com/langsmith/configuration-cloud), [`runs.stream` reference](https://reference.langchain.com/python/langgraph-sdk/_sync/runs/SyncRunsClient/stream)).

For the reference system, conversation messages, retrieval attempts, evidence, and answer state belong in `State`; a request principal and other immutable request dependencies belong in `Context`. Environment variables and package/graph discovery belong in `langgraph.json`/`.env`, not graph state.

## Conversation Threads and development persistence

A Thread is the stateful conversation container. Create one with `client.threads.create()`, then submit every turn as a run against the same `thread_id`; each run starts from that thread's current state and checkpoints its updates for the next turn. Passing `None` as the thread ID creates a stateless run and does not persist its output ([thread semantics](https://docs.langchain.com/langsmith/use-threads), [stateless streaming](https://docs.langchain.com/langsmith/streaming#stateless-runs)).

`langgraph dev` is called an in-memory development server because it is a lightweight single local process, but current official documentation says it writes checkpoint, memory, assistant, and related server data under `.langgraph_api` in the working directory. That makes local threads durable across server restarts unless the directory is removed. This local-disk backend is for development/testing, not a production persistence guarantee ([data storage and privacy](https://docs.langchain.com/langsmith/data-storage-and-privacy), [local-server guide](https://docs.langchain.com/oss/python/langgraph/local-server)).

## Streaming

Use the Agent Server streaming contract through `langgraph-sdk`:

```python
from langgraph_sdk import get_client

client = get_client(url="http://localhost:2024")
thread = await client.threads.create()

async for chunk in client.runs.stream(
    thread["thread_id"],
    "agent",
    input={"messages": [{"role": "user", "content": "..."}]},
    context={"principal_id": "..."},
    stream_mode=["messages-tuple", "updates", "custom"],
):
    handle(chunk.event, chunk.data)
```

The server supports `values` (full state), `updates` (node state deltas), `messages-tuple` (LLM tokens plus metadata), `custom`, `debug`, and `events`, and accepts multiple modes in one run. Use `messages-tuple` for answer tokens, `updates` for coarse workflow progress, and `custom` only for deliberate application progress events ([Streaming API](https://docs.langchain.com/langsmith/streaming)).

`client.runs.stream()` creates and streams one run. `client.runs.join_stream()` joins an already-running run but does not replay output emitted before joining. `client.threads.join_stream()` is a long-lived SSE stream across all runs on a thread and can resume from `Last-Event-ID`; it is useful for a future chat client, but is unnecessary for Studio-only v1 development ([Streaming API: join and thread streams](https://docs.langchain.com/langsmith/streaming#join-and-stream)).

## Agent Server API

`langgraph dev` defaults to `http://127.0.0.1:2024`, watches source files, and serves OpenAPI at `/docs`. Its API surface manages assistants, threads, thread runs, stateless runs, cron jobs, the cross-thread store, and system endpoints. Use the graph ID (`"agent"` above) to address its automatically created default assistant, or a UUID to address a separately configured assistant ([local-server guide](https://docs.langchain.com/oss/python/langgraph/local-server), [Agent Server API reference](https://docs.langchain.com/langsmith/server-api-ref), [assistants overview](https://docs.langchain.com/langsmith/assistants)).

The API and SDK are the callable application boundary already supplied by Agent Server. A parallel FastAPI wrapper would duplicate thread/run/stream semantics and should not be part of v1.

## LangSmith Studio

The current product name is **LangSmith Studio**. `langgraph dev` opens a Studio URL whose `baseUrl` points back to the local Agent Server. Studio can visualize the graph, stream runs, manage assistants and threads, inspect intermediate state, set breakpoints, edit state, and fork or rerun from a checkpoint ([Studio overview](https://docs.langchain.com/langsmith/studio), [Studio workflows](https://docs.langchain.com/langsmith/use-studio)). Chat mode requires a state compatible with `MessagesState`; graph mode works with arbitrary graph state ([Studio overview](https://docs.langchain.com/langsmith/studio)).

Studio is hosted at `smith.langchain.com`, but its browser code connects directly to the local server. With tracing disabled, application inputs and graph data stay in the local Agent Server; set `LANGSMITH_TRACING=false` explicitly when that is required. A LangSmith API key is therefore an observability/tracing dependency, not a requirement that core graph execution should assume ([Studio quickstart](https://docs.langchain.com/langsmith/quick-start-studio), [data storage and privacy](https://docs.langchain.com/langsmith/data-storage-and-privacy)). Safari may require `langgraph dev --tunnel` because it blocks the plain-HTTP localhost connection ([Studio quickstart](https://docs.langchain.com/langsmith/quick-start-studio)).

## Resulting v1 contract

1. Export one compiled graph and register it as `agent` in `langgraph.json`.
2. Run it with Python 3.11+ and `langgraph dev`; do not wrap or replace Agent Server.
3. Let Agent Server inject persistence; keep Conversation Thread turns on one server `thread_id`.
4. Use typed `State`, typed `context_schema`, and `Runtime[Context]`; do not use deprecated `config_schema`.
5. Use `langgraph-sdk` for API integration tests and streaming; use LangSmith Studio for interactive local development.
6. Treat `.langgraph_api` as disposable development data and keep core execution functional with LangSmith tracing disabled.

