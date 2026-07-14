"""Private compiled Answer subgraph."""

from langgraph.graph import END, START, StateGraph

from agentic_rag.graph.nodes import (
    finalize_answer,
    finish_incomplete_answer,
    generate_answer,
    hydrate_answer_citations,
    repair_answer,
    route_after_generation,
    route_after_hydration,
    route_after_repair,
    route_after_validation,
    route_after_verification,
    validate_answer_citations,
    verify_answer,
)
from agentic_rag.graph.state import AnswerGraphState
from agentic_rag.runtime import RuntimeContext


_builder = StateGraph(AnswerGraphState, context_schema=RuntimeContext)
_builder.add_node("hydrate_citations", hydrate_answer_citations)
_builder.add_node("generate", generate_answer)
_builder.add_node("validate", validate_answer_citations)
_builder.add_node("verify", verify_answer)
_builder.add_node("repair", repair_answer)
_builder.add_node("finalize", finalize_answer)
_builder.add_node("finish_incomplete", finish_incomplete_answer)

_builder.add_edge(START, "hydrate_citations")
_builder.add_conditional_edges("hydrate_citations", route_after_hydration)
_builder.add_conditional_edges("generate", route_after_generation)
_builder.add_conditional_edges("validate", route_after_validation)
_builder.add_conditional_edges("verify", route_after_verification)
_builder.add_conditional_edges("repair", route_after_repair)
_builder.add_edge("finalize", END)
_builder.add_edge("finish_incomplete", END)

answer_graph = _builder.compile(name="answer-subgraph")
