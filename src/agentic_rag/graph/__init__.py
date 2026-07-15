"""Compiled Reference System graph export."""

from agentic_rag.graph._turn import build_turn_graph
from agentic_rag.observability import configure_tracing


configure_tracing()
graph = build_turn_graph()
