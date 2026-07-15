"""L1 authorization contract verification."""

import pytest
from pydantic import ValidationError

from agentic_rag.authorization import (
    AccessGrant,
    AccessGrantType,
    AuthorizationSnapshot,
    AuthorizationTransition,
    Principal,
    decide_authorization_transition,
)


def test_authorization_contracts_reject_agent_authored_authority() -> None:
    with pytest.raises(ValidationError):
        Principal.model_validate(
            {
                "principal_id": "alice",
                "group_ids": ["payments-engineering"],
                "sql": "SELECT true",
            }
        )

    with pytest.raises(ValidationError):
        AccessGrant(
            knowledge_source="engineering-docs",
            document_id="doc_example",
            grant_type=AccessGrantType.PRINCIPAL,
        )

    with pytest.raises(ValidationError):
        AuthorizationSnapshot.model_validate(
            {
                "principal_id": "alice",
                "revision": "auth_" + "0" * 64,
                "group_ids": ["payments-engineering"],
            }
        )


def test_authorization_change_restarts_once_then_fails_closed() -> None:
    original = AuthorizationSnapshot(
        principal_id="alice", revision="auth_" + "0" * 64
    )
    unchanged = AuthorizationSnapshot(
        principal_id="alice", revision=original.revision
    )
    changed = AuthorizationSnapshot(
        principal_id="alice", revision="auth_" + "1" * 64
    )

    assert (
        decide_authorization_transition(original, unchanged, 0)
        is AuthorizationTransition.COMMIT
    )
    assert (
        decide_authorization_transition(original, changed, 0)
        is AuthorizationTransition.RESTART
    )
    assert (
        decide_authorization_transition(original, changed, 1)
        is AuthorizationTransition.FAIL
    )


def test_authorization_transition_rejects_principal_mismatch() -> None:
    original = AuthorizationSnapshot(
        principal_id="alice", revision="auth_" + "0" * 64
    )
    other = AuthorizationSnapshot(
        principal_id="bob", revision="auth_" + "0" * 64
    )

    with pytest.raises(ValueError, match="authorization_context_mismatch"):
        decide_authorization_transition(original, other, 0)
