"""L1 prompt contract for the live listwise reranker."""

from unittest.mock import Mock
import json

from agentic_rag.retrieval import RerankCandidate, RerankItem, RerankOutput
from agentic_rag.retrieval.reranking import build_reranker


def test_live_reranker_separates_the_exact_candidate_id_permutation() -> None:
    model = Mock()
    model.with_structured_output.return_value.invoke.return_value = RerankOutput(
        items=(
            RerankItem(chunk_id="chunk-a", score=1.0),
            RerankItem(chunk_id="chunk-b", score=0.0),
        )
    )
    candidates = (
        RerankCandidate(
            chunk_id="chunk-a",
            content="A",
            title="A",
            source_path="a.md",
        ),
        RerankCandidate(
            chunk_id="chunk-b",
            content="B",
            title="B",
            source_path="b.md",
        ),
    )

    build_reranker(model)("rollback", candidates)

    messages = model.with_structured_output.return_value.invoke.call_args.args[0]
    system_prompt = messages[0][1]
    payload = json.loads(messages[1][1])
    assert "Never duplicate" in system_prompt
    assert "0.8 or higher" in system_prompt
    assert "title names the queried subject" in system_prompt
    assert payload["candidate_count"] == 2
    assert payload["candidate_ids"] == ["chunk-a", "chunk-b"]
