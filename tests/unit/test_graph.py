import pytest

from agentic_rag.conversation import ThreadState
from agentic_rag.graph import graph
from agentic_rag.graph.state import CurrentTurnWork
from agentic_rag.runtime import RuntimeContext


def test_graph_rejects_missing_current_turn() -> None:
    with pytest.raises(ValueError, match="current_turn"):
        graph.invoke(
            {"thread": ThreadState(), "current_turn": None},
            context=RuntimeContext(principal_id="alice"),
        )


def test_graph_completes_deterministic_turn() -> None:
    result = graph.invoke(
        {
            "thread": ThreadState(),
            "current_turn": CurrentTurnWork(user_message="Start the reference system"),
        },
        context=RuntimeContext(principal_id="alice"),
    )

    assert result == {
        "thread": ThreadState(completed_turns=1),
        "current_turn": CurrentTurnWork(
            user_message="Start the reference system",
            assistant_message="The Reference System development loop is ready.",
            status="answered",
        ),
    }


def test_graph_accepts_json_shaped_agent_server_input() -> None:
    result = graph.invoke(
        {
            "thread": {"completed_turns": 0},
            "current_turn": {"user_message": "Start the reference system"},
        },
        context=RuntimeContext(principal_id="alice"),
    )

    assert result["thread"] == ThreadState(completed_turns=1)
    assert result["current_turn"] == CurrentTurnWork(
        user_message="Start the reference system",
        assistant_message="The Reference System development loop is ready.",
        status="answered",
    )
