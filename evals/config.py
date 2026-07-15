"""Pinned non-secret identities shared by the live evaluation workflow."""

from agentic_rag.corpus import ProcessingConfig


CHAT_MODEL_ID = "openai:gpt-5.4-mini-2026-03-17"
RERANKER_MODEL_ID = "openai:gpt-5.4-nano-2026-03-17"
EMBEDDING_MODEL_ID = "openai:text-embedding-3-small"
LIVE_PROCESSING_CONFIG = ProcessingConfig(
    parser_version="markdown-v1",
    chunker_version="heading-v1",
    embedding_model=EMBEDDING_MODEL_ID,
)
