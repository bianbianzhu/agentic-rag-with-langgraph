"""L1 authorization contract verification."""

import pytest
from pydantic import ValidationError

from agentic_rag.authorization import (
    AccessGrant,
    AccessGrantType,
    AuthorizationSnapshot,
    Principal,
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
