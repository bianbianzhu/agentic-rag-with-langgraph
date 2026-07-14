"""Compiled Reference System graph export."""

from langgraph.graph import END, START, StateGraph

from agentic_rag.graph.nodes import complete_turn
from agentic_rag.graph.state import GraphState
from agentic_rag.runtime import RuntimeContext

_builder = StateGraph(GraphState, context_schema=RuntimeContext)
_builder.add_node("complete_turn", complete_turn)
_builder.add_edge(START, "complete_turn")
_builder.add_edge("complete_turn", END)

graph = _builder.compile(name="engineering-assistant")
