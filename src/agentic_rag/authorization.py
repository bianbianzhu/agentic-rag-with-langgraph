"""Trusted Principal, Access Grant, and Authorization Snapshot contracts."""

import json
from enum import StrEnum
from hashlib import sha256
from typing import Annotated, Self

from psycopg_pool import ConnectionPool
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)


Identifier = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]


class AccessGrantType(StrEnum):
    """The three relations that may grant Source Document access."""

    ORGANIZATION_PUBLIC = "organization-public"
    PRINCIPAL = "principal"
    GROUP = "group"


class AuthorizationTransition(StrEnum):
    """Code-owned final action after current-scope comparison."""

    COMMIT = "commit"
    RESTART = "restart"
    FAIL = "fail"


class Principal(BaseModel):
    """One trusted Principal and current Group memberships."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    principal_id: Identifier
    group_ids: tuple[Identifier, ...] = ()


class AccessGrant(BaseModel):
    """One trusted relation from a Source Document to an allowed subject."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    knowledge_source: Identifier
    document_id: Identifier
    grant_type: AccessGrantType
    principal_id: Identifier | None = None
    group_id: Identifier | None = None

    @model_validator(mode="after")
    def validate_subject(self) -> Self:
        has_principal = self.principal_id is not None
        has_group = self.group_id is not None
        valid_subject = {
            AccessGrantType.ORGANIZATION_PUBLIC: not has_principal and not has_group,
            AccessGrantType.PRINCIPAL: has_principal and not has_group,
            AccessGrantType.GROUP: has_group and not has_principal,
        }[self.grant_type]
        if not valid_subject:
            raise ValueError("Access Grant subject does not match its type")
        return self


class AuthorizationSnapshot(BaseModel):
    """Versioned effective Access Scope for one trusted Principal."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    principal_id: Identifier
    revision: str = Field(pattern=r"^auth_[0-9a-f]{64}$")


class UnknownPrincipalError(ValueError):
    """Trusted Runtime Context named no provisioned Principal."""


def decide_authorization_transition(
    original: AuthorizationSnapshot,
    current: AuthorizationSnapshot,
    authorization_restarts: int,
) -> AuthorizationTransition:
    """Commit unchanged work, otherwise restart once and then fail closed."""

    if original.principal_id != current.principal_id:
        raise ValueError("authorization_context_mismatch")
    if authorization_restarts not in (0, 1):
        raise ValueError("authorization restart count is invalid")
    if original.revision == current.revision:
        return AuthorizationTransition.COMMIT
    if authorization_restarts == 0:
        return AuthorizationTransition.RESTART
    return AuthorizationTransition.FAIL


def capture_authorization_snapshot(
    pool: ConnectionPool, principal_id: str
) -> AuthorizationSnapshot:
    """Capture a content-free revision of one Principal's effective Access Scope."""

    principal = Principal(principal_id=principal_id)
    with pool.connection() as connection:
        exists = connection.execute(
            "SELECT 1 FROM principals WHERE principal_id = %s",
            (principal.principal_id,),
        ).fetchone()
        if exists is None:
            raise UnknownPrincipalError("unknown Principal")
        authorized_source_documents = connection.execute(
            """
            SELECT source_document.knowledge_source,
                   source_document.document_id
            FROM source_documents AS source_document
            WHERE source_document_is_authorized(
                %(principal_id)s,
                source_document.knowledge_source,
                source_document.document_id
            )
            ORDER BY source_document.knowledge_source,
                     source_document.document_id
            """,
            {"principal_id": principal.principal_id},
        ).fetchall()
    serialized_scope = json.dumps(
        {
            "principal_id": principal.principal_id,
            "source_documents": authorized_source_documents,
        },
        separators=(",", ":"),
    )
    revision = f"auth_{sha256(serialized_scope.encode()).hexdigest()}"
    return AuthorizationSnapshot(
        principal_id=principal.principal_id,
        revision=revision,
    )


def source_document_is_authorized(
    pool: ConnectionPool,
    snapshot: AuthorizationSnapshot,
    *,
    knowledge_source: str,
    document_id: str,
) -> bool:
    """Evaluate the shared default-deny predicate for one Source Document."""

    if not knowledge_source.strip() or not document_id.strip():
        raise ValueError("Source Document identity must not be empty")
    with pool.connection() as connection:
        row = connection.execute(
            """
            SELECT source_document_is_authorized(
                %(principal_id)s,
                %(knowledge_source)s,
                %(document_id)s
            )
            """,
            {
                "principal_id": snapshot.principal_id,
                "knowledge_source": knowledge_source,
                "document_id": document_id,
            },
        ).fetchone()
    return bool(row and row[0])


def authorization_snapshot_is_current(
    pool: ConnectionPool, snapshot: AuthorizationSnapshot
) -> bool:
    """Revalidate that a Principal's effective Access Scope is unchanged."""

    try:
        current = capture_authorization_snapshot(pool, snapshot.principal_id)
    except UnknownPrincipalError:
        return False
    return current.revision == snapshot.revision
