"""L1 secret exclusion at the client-side trace boundary."""

import langsmith as ls
import pytest

from agentic_rag.observability import configure_tracing, trace_anonymizer


def test_trace_anonymizer_redacts_credentials_before_upload() -> None:
    openai_canary = "sk-proj-" + "a" * 32
    langsmith_canary = "lsv2_pt_" + "b" * 32 + "_test"
    database_canary = "postgresql://reader:secret-canary@db/reference"

    redacted = trace_anonymizer()(
        {
            "authorized_content": "rollback failed after schema drift",
            "openai_api_key": openai_canary,
            "langsmith_api_key": langsmith_canary,
            "database_url": database_canary,
        }
    )

    serialized = str(redacted)
    assert "rollback failed after schema drift" in serialized
    assert openai_canary not in serialized
    assert langsmith_canary not in serialized
    assert "postgresql://" not in serialized
    assert "secret-canary" not in serialized
    assert serialized.count("[SECRET_DETECTED]") == 3


def test_configure_tracing_explicitly_enables_the_development_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured: dict[str, object] = {}
    client_arguments: dict[str, object] = {}
    client = object()
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setattr(
        ls,
        "Client",
        lambda **kwargs: client_arguments.update(kwargs) or client,
    )
    monkeypatch.setattr(
        ls,
        "configure",
        lambda **kwargs: configured.update(kwargs),
    )

    configure_tracing()

    assert configured["enabled"] is True
    assert configured["client"] is client
    assert callable(client_arguments["anonymizer"])
