"""Private compiled bounded Research subgraph."""

from langgraph.graph import END, START, StateGraph

from agentic_rag.graph.nodes import (
    assess_research_evidence,
    plan_research,
    refine_research_query_node,
    retrieve_research_evidence,
    route_after_research_assessment,
    route_after_research_plan,
    route_after_research_refinement,
    route_after_research_retrieval,
)
from agentic_rag.graph.state import ResearchGraphState
from agentic_rag.runtime import RuntimeContext


_builder = StateGraph(ResearchGraphState, context_schema=RuntimeContext)
_builder.add_node("plan", plan_research)
_builder.add_node("retrieve", retrieve_research_evidence)
_builder.add_node("assess", assess_research_evidence)
_builder.add_node("refine", refine_research_query_node)

_builder.add_edge(START, "plan")
_builder.add_conditional_edges("plan", route_after_research_plan)
_builder.add_conditional_edges("retrieve", route_after_research_retrieval)
_builder.add_conditional_edges("assess", route_after_research_assessment)
_builder.add_conditional_edges("refine", route_after_research_refinement)

research_graph = _builder.compile(name="research-subgraph")
