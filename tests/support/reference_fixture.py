"""Deterministic adapter for the shared canonical reference fixture."""

from hashlib import sha256

from langchain_core.embeddings import Embeddings
from psycopg_pool import ConnectionPool

from agentic_rag.corpus import ProcessingConfig
from fixtures.reference_support import (
    provision_reference_access_scope as _provision_access_scope,
)
from fixtures.reference_support import (
    FIXTURE_ROOT,
    resolve_reference_evidence_aliases,
    sync_reference_fixture,
)


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
    pool: ConnectionPool,
    snapshot: str = "s1-baseline",
) -> None:
    sync_reference_snapshot(pool, snapshot)
    provision_reference_access_scope(pool, snapshot)


def load_reference_scenario(
    pool: ConnectionPool,
    snapshot: str,
    overlay: str | None,
) -> None:
    """Publish one snapshot plus optional trusted attack overlay."""

    sync_reference_fixture(
        pool,
        snapshot=snapshot,
        overlay=overlay,
        processing_config=PROCESSING_CONFIG,
        embedder=FixtureEmbedder(),
    )
    provision_reference_access_scope(pool, snapshot, overlay=overlay)


def sync_reference_snapshot(pool: ConnectionPool, snapshot: str) -> None:
    sync_reference_fixture(
        pool,
        snapshot=snapshot,
        overlay=None,
        processing_config=PROCESSING_CONFIG,
        embedder=FixtureEmbedder(),
    )


def provision_reference_access_scope(
    pool: ConnectionPool,
    grant_set: str,
    *,
    overlay: str | None = None,
) -> None:
    _provision_access_scope(
        pool,
        grant_set,
        overlay=overlay,
        reset=False,
    )


def resolve_evidence_aliases(
    pool: ConnectionPool,
    aliases: tuple[str, ...],
) -> dict[str, str]:
    """Resolve trusted test aliases to current real Indexed Chunk IDs."""

    return resolve_reference_evidence_aliases(pool, aliases)
