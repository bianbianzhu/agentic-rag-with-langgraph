"""Shared typed loader for the canonical PostgreSQL reference fixture."""

import json
import shutil
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

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


def load_reference_scenario(
    pool: ConnectionPool,
    snapshot: str,
    overlay: str | None,
) -> None:
    """Publish one fixed snapshot plus an optional trusted attack overlay."""

    if overlay is None:
        load_reference_snapshot(pool, snapshot)
        return
    grants = _grant_fixture()
    overlays = _overlay_fixture()["overlays"]
    if overlay not in overlays:
        raise ValueError("unknown reference overlay")
    with TemporaryDirectory() as temporary_directory:
        temporary_root = Path(temporary_directory)
        for knowledge_source in KnowledgeSource:
            source_root = (
                FIXTURE_ROOT
                / "corpus/snapshots"
                / snapshot
                / knowledge_source.value
            )
            composed_root = temporary_root / knowledge_source.value
            shutil.copytree(source_root, composed_root)
            overlay_root = (
                FIXTURE_ROOT
                / "corpus/overlays"
                / overlay
                / knowledge_source.value
            )
            if overlay_root.exists():
                shutil.copytree(
                    overlay_root, composed_root, dirs_exist_ok=True
                )
            assignments = _grant_assignments(
                grants["sets"][snapshot][knowledge_source.value],
                overlays[overlay],
                knowledge_source.value,
            )
            revisions = {
                source_path: sha256(
                    json.dumps(
                        grants["policies"][policy_name],
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                for source_path, policy_name in assignments.items()
            }
            sync_knowledge_source(
                pool,
                knowledge_source,
                composed_root,
                revisions,
                PROCESSING_CONFIG,
                FixtureEmbedder(),
            )
    provision_reference_access_scope(pool, snapshot, overlay=overlay)


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
    pool: ConnectionPool,
    grant_set: str,
    *,
    overlay: str | None = None,
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
        overlay_assignments = (
            _overlay_fixture()["overlays"][overlay]
            if overlay is not None
            else {}
        )
        for knowledge_source, base_assignments in fixture["sets"][
            grant_set
        ].items():
            assignments = _grant_assignments(
                base_assignments,
                overlay_assignments,
                knowledge_source,
            )
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


def _overlay_fixture():
    return json.loads(
        (FIXTURE_ROOT / "trusted/overlay-grants.json").read_text(
            encoding="utf-8"
        )
    )


def _grant_assignments(
    base_assignments: dict[str, str],
    overlay_assignments: dict[str, dict[str, str]],
    knowledge_source: str,
) -> dict[str, str]:
    assignments = dict(base_assignments)
    assignments.update(overlay_assignments.get(knowledge_source, {}))
    return assignments


def resolve_evidence_aliases(
    pool: ConnectionPool, aliases: tuple[str, ...]
) -> dict[str, str]:
    """Resolve trusted test aliases to current real Indexed Chunk IDs."""

    registry = json.loads(
        (FIXTURE_ROOT / "trusted/evidence-aliases.json").read_text(
            encoding="utf-8"
        )
    )["aliases"]
    resolved: dict[str, str] = {}
    with pool.connection() as connection:
        for alias in aliases:
            try:
                identity = registry[alias]
            except KeyError:
                raise ValueError("unknown Evidence alias") from None
            rows = connection.execute(
                """
                SELECT indexed_chunk.chunk_id
                FROM indexed_chunks AS indexed_chunk
                JOIN source_documents AS source_document
                  ON source_document.knowledge_source =
                     indexed_chunk.knowledge_source
                 AND source_document.document_id = indexed_chunk.document_id
                WHERE indexed_chunk.knowledge_source = %(knowledge_source)s
                  AND source_document.source_path = %(source_path)s
                  AND indexed_chunk.heading_path = %(section_path)s
                """,
                identity,
            ).fetchall()
            if len(rows) != 1:
                raise ValueError("Evidence alias does not resolve exactly once")
            resolved[alias] = str(rows[0][0])
    return resolved
