"""Node adapters for the Reference System graph."""

from langgraph.runtime import Runtime

from agentic_rag.conversation import ThreadState
from agentic_rag.graph.state import CurrentTurnWork
from agentic_rag.graph.state import GraphState
from agentic_rag.runtime import RuntimeContext


def complete_turn(
    state: GraphState, runtime: Runtime[RuntimeContext]
) -> dict[str, dict[str, object]]:
    """Complete the deterministic Chapter 01 Turn."""

    raw_current_turn = state["current_turn"]
    if raw_current_turn is None:
        raise ValueError("current_turn is required")

    thread = ThreadState.model_validate(state["thread"])
    current_turn = CurrentTurnWork.model_validate(raw_current_turn)

    return {
        "thread": thread.model_copy(
            update={"completed_turns": thread.completed_turns + 1}
        ).model_dump(mode="json"),
        "current_turn": current_turn.model_copy(
            update={
                "assistant_message": "The Reference System development loop is ready.",
                "status": "answered",
            }
        ).model_dump(mode="json"),
    }
