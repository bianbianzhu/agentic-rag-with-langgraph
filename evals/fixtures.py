"""Live adapter for the shared canonical reference fixture."""

from langchain_core.embeddings import Embeddings
from psycopg_pool import ConnectionPool

from evals.config import LIVE_PROCESSING_CONFIG
from fixtures.reference_support import (
    provision_reference_access_scope,
    resolve_reference_evidence_aliases,
    sync_reference_fixture,
)


def prepare_live_fixture(
    pool: ConnectionPool,
    *,
    snapshot: str,
    overlay: str | None,
    embedder: Embeddings,
) -> dict[str, str]:
    """Publish one live fixture and reset its trusted Access Scope."""

    revisions = sync_reference_fixture(
        pool,
        snapshot=snapshot,
        overlay=overlay,
        processing_config=LIVE_PROCESSING_CONFIG,
        embedder=embedder,
    )
    provision_reference_access_scope(
        pool,
        snapshot,
        overlay=overlay,
        reset=True,
    )
    return revisions


def resolve_all_evidence_aliases(pool: ConnectionPool) -> dict[str, str]:
    """Map every currently published chunk ID back to its trusted alias."""

    by_alias = resolve_reference_evidence_aliases(pool, require_all=False)
    return {chunk_id: alias for alias, chunk_id in by_alias.items()}
