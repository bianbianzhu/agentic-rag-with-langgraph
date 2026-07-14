"""Shared typed loader for the canonical PostgreSQL reference fixture."""

import json
from hashlib import sha256
from pathlib import Path

from langchain_core.embeddings import Embeddings
from psycopg_pool import ConnectionPool

from agentic_rag.authorization import AccessGrant, Principal
from agentic_rag.corpus import (
    KnowledgeSource,
    ProcessingConfig,
    sync_knowledge_source,
)


FIXTURE_ROOT = Path(__file__).parents[2] / "fixtures/reference-system"
PROCESSING_CONFIG = ProcessingConfig(
    parser_version="markdown-v1",
    chunker_version="heading-v1",
    embedding_model="deterministic-test-v1",
)


class FixtureEmbedder(Embeddings):
    """Stable local vectors shared by frozen L2 fixtures."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [byte / 255 for byte in sha256(text.encode()).digest()[:4]]


def load_reference_snapshot(
    pool: ConnectionPool, snapshot: str = "s1-baseline"
) -> None:
    sync_reference_snapshot(pool, snapshot)
    provision_reference_access_scope(pool, snapshot)


def sync_reference_snapshot(pool: ConnectionPool, snapshot: str) -> None:
    fixture = _grant_fixture()
    for knowledge_source in KnowledgeSource:
        assignments = fixture["sets"][snapshot][knowledge_source.value]
        revisions = {
            source_path: sha256(
                json.dumps(
                    fixture["policies"][policy_name],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            for source_path, policy_name in assignments.items()
        }
        sync_knowledge_source(
            pool,
            knowledge_source,
            FIXTURE_ROOT / "corpus/snapshots" / snapshot / knowledge_source.value,
            revisions,
            PROCESSING_CONFIG,
            FixtureEmbedder(),
        )


def provision_reference_access_scope(
    pool: ConnectionPool, grant_set: str
) -> None:
    principals_fixture = json.loads(
        (FIXTURE_ROOT / "trusted/principals.json").read_text(encoding="utf-8")
    )
    principals = [
        Principal.model_validate(item)
        for item in principals_fixture["principals"]
    ]
    fixture = _grant_fixture()
    with pool.connection() as connection:
        for principal in principals:
            connection.execute(
                "INSERT INTO principals (principal_id) VALUES (%s)",
                (principal.principal_id,),
            )
        group_ids = sorted(
            {
                group_id
                for principal in principals
                for group_id in principal.group_ids
            }
        )
        for group_id in group_ids:
            connection.execute(
                "INSERT INTO groups (group_id) VALUES (%s)", (group_id,)
            )
        for principal in principals:
            for group_id in principal.group_ids:
                connection.execute(
                    """
                    INSERT INTO principal_group_memberships (
                        principal_id, group_id
                    ) VALUES (%s, %s)
                    """,
                    (principal.principal_id, group_id),
                )
        for knowledge_source, assignments in fixture["sets"][grant_set].items():
            for source_path, policy_name in assignments.items():
                document_row = connection.execute(
                    """
                    SELECT document_id FROM source_documents
                    WHERE knowledge_source = %s AND source_path = %s
                    """,
                    (knowledge_source, source_path),
                ).fetchone()
                assert document_row is not None
                for relation in fixture["policies"][policy_name]:
                    grant = AccessGrant(
                        knowledge_source=knowledge_source,
                        document_id=str(document_row[0]),
                        **relation,
                    )
                    connection.execute(
                        """
                        INSERT INTO access_grants (
                            knowledge_source, document_id, grant_type,
                            principal_id, group_id
                        ) VALUES (%s, %s, %s, %s, %s)
                        """,
                        (
                            grant.knowledge_source,
                            grant.document_id,
                            grant.grant_type.value,
                            grant.principal_id,
                            grant.group_id,
                        ),
                    )


def _grant_fixture():
    return json.loads(
        (FIXTURE_ROOT / "trusted/access-grant-sets.json").read_text(
            encoding="utf-8"
        )
    )
