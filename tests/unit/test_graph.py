import pytest

from agentic_rag.conversation import (
    ThreadState,
    TurnResumeIncompatibleError,
)
from agentic_rag.graph import graph
from agentic_rag.graph.state import CurrentTurnWork
from agentic_rag.runtime import RuntimeContext


def test_graph_rejects_missing_current_turn() -> None:
    with pytest.raises(ValueError, match="current_turn"):
        graph.invoke(
            {"thread": ThreadState(), "current_turn": None},
            context=RuntimeContext(principal_id="alice"),
        )


def test_graph_rejects_a_user_message_that_cannot_fit_context_budget() -> None:
    with pytest.raises(ValueError, match="at most 4000"):
        CurrentTurnWork(turn_id="turn-1", user_message="x" * 4_001)


def test_graph_rejects_an_incompatible_checkpoint_schema() -> None:
    with pytest.raises(
        TurnResumeIncompatibleError, match="turn_resume_incompatible"
    ):
        graph.invoke(
            {
                "thread": {"schema_version": "thread-state-v0"},
                "current_turn": {
                    "turn_id": "turn-1",
                    "user_message": "Resume",
                },
            },
            context=RuntimeContext(principal_id="alice"),
        )


def test_graph_completes_deterministic_turn() -> None:
    result = graph.invoke(
        {
            "thread": ThreadState(),
            "current_turn": CurrentTurnWork(
                turn_id="turn-1",
                user_message="Start the reference system",
            ),
        },
        context=RuntimeContext(principal_id="alice"),
    )

    thread = ThreadState.model_validate(result["thread"])
    assert thread.principal_id == "alice"
    assert len(thread.turn_records) == 1
    assert thread.turn_records[0].turn_id == "turn-1"
    assert result["current_turn"] is None


def test_graph_accepts_json_shaped_agent_server_input() -> None:
    result = graph.invoke(
        {
            "thread": {},
            "current_turn": {
                "turn_id": "turn-1",
                "user_message": "Start the reference system",
            },
        },
        context=RuntimeContext(principal_id="alice"),
    )

    assert len(result["thread"]["turn_records"]) == 1
    assert result["current_turn"] is None


def test_graph_reuses_json_safe_thread_state_for_next_turn() -> None:
    first_result = graph.invoke(
        {
            "thread": {},
            "current_turn": {
                "turn_id": "turn-1",
                "user_message": "First Turn",
            },
        },
        context=RuntimeContext(principal_id="alice"),
    )

    second_result = graph.invoke(
        {
            "thread": first_result["thread"],
            "current_turn": {
                "turn_id": "turn-2",
                "user_message": "Second Turn",
            },
        },
        context=RuntimeContext(principal_id="alice"),
    )

    thread = ThreadState.model_validate(second_result["thread"])
    assert [record.turn_id for record in thread.turn_records] == [
        "turn-1",
        "turn-2",
    ]
    assert thread.turn_records[-1].outcome.value == "failed"
    assert thread.turn_records[-1].terminal_reason == (
        "contextualization_unavailable"
    )
