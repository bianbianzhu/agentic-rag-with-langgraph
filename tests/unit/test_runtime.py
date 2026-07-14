import pytest

from agentic_rag.runtime import RuntimeContext


def test_runtime_context_rejects_empty_principal_id() -> None:
    with pytest.raises(ValueError, match="principal_id"):
        RuntimeContext(principal_id=" ")
