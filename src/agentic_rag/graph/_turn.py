"""Private top-level Turn graph construction."""

from collections.abc import Sequence

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from agentic_rag.graph._turn_nodes import (
    authorize_and_commit_turn,
    capture_turn_authorization,
    route_after_authorization_capture,
    route_after_contextualization,
    route_after_final_authorization,
    route_after_research,
    route_after_turn_prepare,
    run_answer_subgraph,
    run_research_subgraph,
)
from agentic_rag.graph.nodes import (
    compact_context,
    complete_turn,
    contextualize_turn,
    prepare_turn,
    route_after_compaction,
)
from agentic_rag.graph.state import GraphState
from agentic_rag.runtime import RuntimeContext


def build_turn_graph(
    checkpointer: BaseCheckpointSaver | None = None,
    *,
    interrupt_after: Sequence[str] | None = None,
):
    """Compile the one Turn graph, optionally with a test checkpointer."""

    builder = StateGraph(GraphState, context_schema=RuntimeContext)
    builder.add_node("prepare_turn", prepare_turn)
    builder.add_node("compact_context", compact_context)
    builder.add_node("contextualize_turn", contextualize_turn)
    builder.add_node("capture_turn_authorization", capture_turn_authorization)
    builder.add_node("run_research_subgraph", run_research_subgraph)
    builder.add_node("run_answer_subgraph", run_answer_subgraph)
    builder.add_node("authorize_and_commit_turn", authorize_and_commit_turn)
    builder.add_node("complete_turn", complete_turn)
    builder.add_edge(START, "prepare_turn")
    builder.add_conditional_edges(
        "prepare_turn",
        route_after_turn_prepare,
        {
            "capture_turn_authorization": "capture_turn_authorization",
            "complete_turn": "complete_turn",
            "end": END,
        },
    )
    builder.add_conditional_edges(
        "compact_context",
        route_after_compaction,
        {
            "compact_context": "compact_context",
            "contextualize_turn": "contextualize_turn",
            "complete_turn": "complete_turn",
            "authorize_and_commit_turn": "authorize_and_commit_turn",
        },
    )
    builder.add_conditional_edges(
        "contextualize_turn",
        route_after_contextualization,
        {
            "run_research_subgraph": "run_research_subgraph",
            "authorize_and_commit_turn": "authorize_and_commit_turn",
            "complete_turn": "complete_turn",
        },
    )
    builder.add_conditional_edges(
        "capture_turn_authorization",
        route_after_authorization_capture,
        {
            "compact_context": "compact_context",
            "contextualize_turn": "contextualize_turn",
            "complete_turn": "complete_turn",
        },
    )
    builder.add_conditional_edges(
        "run_research_subgraph",
        route_after_research,
        {
            "run_answer_subgraph": "run_answer_subgraph",
            "authorize_and_commit_turn": "authorize_and_commit_turn",
        },
    )
    builder.add_edge("run_answer_subgraph", "authorize_and_commit_turn")
    builder.add_conditional_edges(
        "authorize_and_commit_turn",
        route_after_final_authorization,
        {"prepare_turn": "prepare_turn", "end": END},
    )
    builder.add_edge("complete_turn", END)
    return builder.compile(
        name="engineering-assistant",
        checkpointer=checkpointer,
        interrupt_after=(list(interrupt_after) if interrupt_after else None),
    )
