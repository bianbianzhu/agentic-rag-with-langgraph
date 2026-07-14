import pytest

from agentic_rag.corpus.identity import (
    indexed_chunk_id,
    processing_revision,
    source_document_id,
    source_revision,
)
from agentic_rag.corpus.models import KnowledgeSource, ProcessingConfig


@pytest.mark.parametrize("source_path", ["/absolute.md", "../escape.md", "a/../b.md"])
def test_source_document_id_rejects_non_relative_paths(source_path: str) -> None:
    with pytest.raises(ValueError, match="source-relative"):
        source_document_id(KnowledgeSource.ENGINEERING_DOCS, source_path)


def test_corpus_identities_change_only_with_their_owned_inputs() -> None:
    document_id = source_document_id(
        KnowledgeSource.ENGINEERING_DOCS, "payments/schema-migration.md"
    )
    assert document_id == source_document_id(
        KnowledgeSource.ENGINEERING_DOCS, "payments/schema-migration.md"
    )
    assert document_id != source_document_id(
        KnowledgeSource.ENGINEERING_DOCS, "payments/schema-migration-v2.md"
    )

    first_source_revision = source_revision(b"revision one", "access-v1")
    assert first_source_revision == source_revision(b"revision one", "access-v1")
    assert first_source_revision != source_revision(b"revision two", "access-v1")
    assert first_source_revision != source_revision(b"revision one", "access-v2")

    config = ProcessingConfig(
        parser_version="markdown-v1",
        chunker_version="headings-v1",
        embedding_model="deterministic-test-v1",
    )
    processing = processing_revision(config)
    assert processing == processing_revision(config)
    assert processing != processing_revision(
        config.model_copy(update={"chunker_version": "headings-v2"})
    )

    assert indexed_chunk_id(document_id, first_source_revision, processing, 0) != (
        indexed_chunk_id(document_id, first_source_revision, processing, 1)
    )
