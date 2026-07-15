"""Neutral composition and authorization helpers for the reference fixture."""

from collections.abc import Iterable
import json
import shutil
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from langchain_core.embeddings import Embeddings
from psycopg_pool import ConnectionPool

from agentic_rag.authorization import AccessGrant, Principal
from agentic_rag.corpus import (
    KnowledgeSource,
    ProcessingConfig,
    sync_knowledge_source,
)


FIXTURE_ROOT = Path(__file__).parent / "reference-system"


def sync_reference_fixture(
    pool: ConnectionPool,
    *,
    snapshot: str,
    overlay: str | None,
    processing_config: ProcessingConfig,
    embedder: Embeddings,
) -> dict[str, str]:
    """Compose and publish one trusted snapshot/overlay combination."""

    grants = _read_json(FIXTURE_ROOT / "trusted/access-grant-sets.json")
    overlays = _read_json(FIXTURE_ROOT / "trusted/overlay-grants.json")[
        "overlays"
    ]
    if snapshot not in grants["sets"]:
        raise ValueError("unknown reference snapshot")
    if overlay is not None and overlay not in overlays:
        raise ValueError("unknown reference overlay")
    active_overlay = overlays.get(overlay, {})
    revisions: dict[str, str] = {}
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
            if overlay is not None:
                overlay_root = (
                    FIXTURE_ROOT
                    / "corpus/overlays"
                    / overlay
                    / knowledge_source.value
                )
                if overlay_root.exists():
                    shutil.copytree(
                        overlay_root,
                        composed_root,
                        dirs_exist_ok=True,
                    )
            assignments = _grant_assignments(
                grants["sets"][snapshot][knowledge_source.value],
                active_overlay,
                knowledge_source.value,
            )
            scope_revisions = {
                source_path: _policy_revision(grants["policies"][policy_name])
                for source_path, policy_name in assignments.items()
            }
            report = sync_knowledge_source(
                pool,
                knowledge_source,
                composed_root,
                scope_revisions,
                processing_config,
                embedder,
            )
            if report.corpus_revision is None:
                raise ValueError("reference Corpus Sync did not publish")
            revisions[knowledge_source.value] = report.corpus_revision
    return revisions


def provision_reference_access_scope(
    pool: ConnectionPool,
    grant_set: str,
    *,
    overlay: str | None,
    reset: bool,
) -> None:
    """Install Principals, groups, and grants from trusted fixture manifests."""

    grants = _read_json(FIXTURE_ROOT / "trusted/access-grant-sets.json")
    overlays = _read_json(FIXTURE_ROOT / "trusted/overlay-grants.json")[
        "overlays"
    ]
    if grant_set not in grants["sets"]:
        raise ValueError("unknown reference grant set")
    if overlay is not None and overlay not in overlays:
        raise ValueError("unknown reference overlay")
    principals = tuple(
        Principal.model_validate(value)
        for value in _read_json(FIXTURE_ROOT / "trusted/principals.json")[
            "principals"
        ]
    )
    with pool.connection() as connection:
        if reset:
            connection.execute(
                """
                TRUNCATE access_grants, principal_group_memberships,
                    groups, principals CASCADE
                """
            )
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
                "INSERT INTO groups (group_id) VALUES (%s)",
                (group_id,),
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
        active_overlay = overlays.get(overlay, {})
        for knowledge_source, base_assignments in grants["sets"][
            grant_set
        ].items():
            assignments = _grant_assignments(
                base_assignments,
                active_overlay,
                knowledge_source,
            )
            for source_path, policy_name in assignments.items():
                row = connection.execute(
                    """
                    SELECT document_id FROM source_documents
                    WHERE knowledge_source = %s AND source_path = %s
                    """,
                    (knowledge_source, source_path),
                ).fetchone()
                if row is None:
                    raise ValueError("grant document is not published")
                for relation in grants["policies"][policy_name]:
                    grant = AccessGrant(
                        knowledge_source=knowledge_source,
                        document_id=str(row[0]),
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


def resolve_reference_evidence_aliases(
    pool: ConnectionPool,
    aliases: Iterable[str] | None = None,
    *,
    require_all: bool = True,
) -> dict[str, str]:
    """Resolve trusted aliases to exactly one current Indexed Chunk each."""

    registry = _read_json(
        FIXTURE_ROOT / "trusted/evidence-aliases.json"
    )["aliases"]
    selected = tuple(registry) if aliases is None else tuple(aliases)
    resolved: dict[str, str] = {}
    with pool.connection() as connection:
        for alias in selected:
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
            if not rows and not require_all:
                continue
            if len(rows) != 1:
                raise ValueError("Evidence alias does not resolve exactly once")
            resolved[alias] = str(rows[0][0])
    return resolved


def _grant_assignments(
    base_assignments: dict[str, str],
    overlay_assignments: dict[str, dict[str, str]],
    knowledge_source: str,
) -> dict[str, str]:
    assignments = dict(base_assignments)
    assignments.update(overlay_assignments.get(knowledge_source, {}))
    return assignments


def _policy_revision(policy: list[dict[str, object]]) -> str:
    encoded = json.dumps(
        policy,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
