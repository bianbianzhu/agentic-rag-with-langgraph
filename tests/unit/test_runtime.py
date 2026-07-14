import pytest

from agentic_rag.runtime import RuntimeContext, TurnExecutionBudget


def test_runtime_context_rejects_empty_principal_id() -> None:
    with pytest.raises(ValueError, match="principal_id"):
        RuntimeContext(principal_id=" ")


def test_runtime_context_owns_a_bounded_deadline() -> None:
    context = RuntimeContext(
        principal_id="alice",
        clock=lambda: 100.0,
        execution_budget=TurnExecutionBudget(deadline_seconds=30),
    )

    assert context.deadline_at == 130.0
    assert context.execution_budget.fingerprint.startswith("turn_budget_")
    assert (
        context.execution_budget.fingerprint
        != TurnExecutionBudget(deadline_seconds=20).fingerprint
    )


def test_explicit_deadline_cannot_extend_the_turn_budget() -> None:
    with pytest.raises(ValueError, match="deadline_at"):
        RuntimeContext(
            principal_id="alice",
            clock=lambda: 100.0,
            execution_budget=TurnExecutionBudget(deadline_seconds=30),
            deadline_at=131.0,
        )
