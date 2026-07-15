"""Private top-level Turn graph construction."""

from collections.abc import Sequence

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from agentic_rag.graph.nodes import (
    compact_context,
    complete_turn,
    contextualize_turn,
    prepare_turn,
    route_after_compaction,
    route_after_prepare,
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
    builder.add_node("complete_turn", complete_turn)
    builder.add_edge(START, "prepare_turn")
    builder.add_conditional_edges(
        "prepare_turn",
        route_after_prepare,
        {
            "compact_context": "compact_context",
            "contextualize_turn": "contextualize_turn",
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
        },
    )
    builder.add_edge("contextualize_turn", "complete_turn")
    builder.add_edge("complete_turn", END)
    return builder.compile(
        name="engineering-assistant",
        checkpointer=checkpointer,
        interrupt_after=(list(interrupt_after) if interrupt_after else None),
    )
