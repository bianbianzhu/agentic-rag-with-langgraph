"""Deterministic bounded Evidence Set assembly."""

import json
from hashlib import sha256
from math import ceil
from time import perf_counter
from typing import Sequence

from psycopg import Error as PsycopgError
from psycopg_pool import ConnectionPool

from agentic_rag.authorization import AuthorizationSnapshot
from agentic_rag.retrieval.models import (
    EvidenceItem,
    EvidenceSet,
    EvidenceSetBudget,
    EvidenceSetRequestOutcome,
    RetrievalConfig,
    RetrievalErrorCode,
    RetrievalResult,
    RetrievalStage,
    RetrievalStatus,
)


class _ContextError(Exception):
    """Internal signal for stale or unauthorized context expansion."""


def assemble_evidence_set(
    results: Sequence[RetrievalResult],
    config: RetrievalConfig,
    *,
    item_limit: int = 8,
) -> EvidenceSet:
    """Deduplicate and round-robin successful request results."""

    budget = EvidenceSetBudget(
        item_limit=item_limit,
        token_limit=config.evidence_token_limit,
    )
    merged_items = _merge_provenance(results)
    selected: list[EvidenceItem] = []
    selected_ids: set[str] = set()
    token_count = 0
    item_budget_exhausted = False
    token_budget_exhausted = False
    completed = [
        result
        for result in results
        if result.status is RetrievalStatus.COMPLETED
    ]

    round_count = max(
        (len(result.evidence_items) for result in completed), default=0
    )
    for item_index in range(round_count):
        for result in completed:
            if item_index >= len(result.evidence_items):
                continue
            chunk_id = result.evidence_items[item_index].chunk_id
            if chunk_id in selected_ids:
                continue
            item = merged_items[chunk_id]
            item_tokens = count_evidence_tokens(item.chunk_text)
            exceeds_items = len(selected) >= budget.item_limit
            exceeds_tokens = token_count + item_tokens > budget.token_limit
            item_budget_exhausted |= exceeds_items
            token_budget_exhausted |= exceeds_tokens
            if exceeds_items or exceeds_tokens:
                continue
            selected.append(item)
            selected_ids.add(chunk_id)
            token_count += item_tokens

    return EvidenceSet(
        evidence_items=tuple(selected),
        request_outcomes=tuple(
            EvidenceSetRequestOutcome(
                request_id=result.request_id,
                retrieval_config_fingerprint=(
                    result.retrieval_config_fingerprint
                ),
                status=result.status,
                failed_stage=result.failed_stage,
                error_code=result.error_code,
            )
            for result in results
        ),
        budget=budget,
        evidence_config_fingerprint=_evidence_config_fingerprint(
            config, budget
        ),
        complete=all(
            result.status is not RetrievalStatus.FAILED for result in results
        ),
        token_count=token_count,
        item_budget_exhausted=item_budget_exhausted,
        token_budget_exhausted=token_budget_exhausted,
        context_expanded=False,
        context_ms=0,
        failed_stage=None,
        error_code=None,
    )


def expand_evidence_set_context(
    pool: ConnectionPool,
    snapshot: AuthorizationSnapshot,
    evidence_set: EvidenceSet,
    config: RetrievalConfig,
) -> EvidenceSet:
    """Hydrate context only for final selected Evidence Items."""

    if _evidence_config_fingerprint(
        config, evidence_set.budget
    ) != evidence_set.evidence_config_fingerprint:
        raise ValueError("Evidence Set and context configuration disagree")
    if evidence_set.context_expanded or not evidence_set.complete:
        return evidence_set

    started = perf_counter()
    if not evidence_set.evidence_items or config.context_window == 0:
        return evidence_set.model_copy(
            update={
                "context_expanded": True,
                "context_ms": _elapsed_ms(started),
            }
        )

    token_limit = evidence_set.budget.token_limit
    token_count = evidence_set.token_count
    token_budget_exhausted = evidence_set.token_budget_exhausted
    expanded: list[EvidenceItem] = []
    try:
        with pool.connection() as connection:
            with connection.transaction():
                connection.execute(
                    "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"
                )
                for item in evidence_set.evidence_items:
                    rows = connection.execute(
                        """
                        SELECT neighbor.chunk_id, neighbor.content
                        FROM indexed_chunks AS matched
                        JOIN knowledge_sources AS knowledge_source
                          ON knowledge_source.knowledge_source =
                             matched.knowledge_source
                        JOIN indexed_chunks AS neighbor
                          ON neighbor.knowledge_source = matched.knowledge_source
                         AND neighbor.document_id = matched.document_id
                         AND neighbor.source_revision = matched.source_revision
                         AND neighbor.processing_revision =
                             matched.processing_revision
                         AND neighbor.ordinal BETWEEN
                             matched.ordinal - %(context_window)s
                             AND matched.ordinal + %(context_window)s
                        WHERE matched.chunk_id = %(chunk_id)s
                          AND knowledge_source.corpus_revision =
                              %(corpus_revision)s
                          AND source_document_is_authorized(
                              %(principal_id)s,
                              matched.knowledge_source,
                              matched.document_id
                          )
                          AND source_document_is_authorized(
                              %(principal_id)s,
                              neighbor.knowledge_source,
                              neighbor.document_id
                          )
                        ORDER BY neighbor.ordinal, neighbor.chunk_id
                        """,
                        {
                            "chunk_id": item.chunk_id,
                            "context_window": config.context_window,
                            "corpus_revision": item.corpus_revision,
                            "principal_id": snapshot.principal_id,
                        },
                    ).fetchall()
                    if not rows or item.chunk_id not in {
                        str(row[0]) for row in rows
                    }:
                        raise _ContextError

                    context_parts: list[str] = []
                    context_chunk_ids: list[str] = []
                    for chunk_id_value, content_value in rows:
                        chunk_id = str(chunk_id_value)
                        if chunk_id == item.chunk_id:
                            continue
                        content = str(content_value)
                        previous = "\n\n".join(context_parts)
                        current = "\n\n".join((*context_parts, content))
                        added_tokens = count_evidence_tokens(current) - (
                            count_evidence_tokens(previous) if previous else 0
                        )
                        if token_count + added_tokens > token_limit:
                            token_budget_exhausted = True
                            continue
                        token_count += added_tokens
                        context_parts.append(content)
                        context_chunk_ids.append(chunk_id)
                    expanded.append(
                        item.model_copy(
                            update={
                                "context_text": (
                                    "\n\n".join(context_parts)
                                    if context_parts
                                    else None
                                ),
                                "context_chunk_ids": tuple(context_chunk_ids),
                            }
                        )
                    )
    except (PsycopgError, _ContextError):
        return evidence_set.model_copy(
            update={
                "evidence_items": (),
                "complete": False,
                "token_count": 0,
                "context_ms": _elapsed_ms(started),
                "failed_stage": RetrievalStage.CONTEXT,
                "error_code": RetrievalErrorCode.CORPUS_UNAVAILABLE,
            }
        )

    return evidence_set.model_copy(
        update={
            "evidence_items": tuple(expanded),
            "token_count": token_count,
            "token_budget_exhausted": token_budget_exhausted,
            "context_expanded": True,
            "context_ms": _elapsed_ms(started),
        }
    )


def count_evidence_tokens(text: str) -> int:
    """Apply the fingerprinted chars-div-4-v1 token approximation."""

    return max(1, ceil(len(text) / 4))


def _merge_provenance(
    results: Sequence[RetrievalResult],
) -> dict[str, EvidenceItem]:
    merged: dict[str, EvidenceItem] = {}
    provenance_keys: dict[str, set[str]] = {}
    for result in results:
        for item in result.evidence_items:
            if item.chunk_id not in merged:
                merged[item.chunk_id] = item.model_copy(
                    update={"context_text": None, "context_chunk_ids": ()}
                )
                provenance_keys[item.chunk_id] = {
                    value.model_dump_json() for value in item.provenance
                }
                continue
            current = merged[item.chunk_id]
            additions = tuple(
                value
                for value in item.provenance
                if value.model_dump_json()
                not in provenance_keys[item.chunk_id]
            )
            if additions:
                merged[item.chunk_id] = current.model_copy(
                    update={"provenance": current.provenance + additions}
                )
                provenance_keys[item.chunk_id].update(
                    value.model_dump_json() for value in additions
                )
    return merged


def _elapsed_ms(started: float) -> float:
    return (perf_counter() - started) * 1_000


def _evidence_config_fingerprint(
    config: RetrievalConfig, budget: EvidenceSetBudget
) -> str:
    serialized = json.dumps(
        {
            "contract_version": config.contract_version,
            "context_window": config.context_window,
            "item_limit": budget.item_limit,
            "token_counting_semantics": config.token_counting_semantics,
            "token_limit": budget.token_limit,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"evidence_{sha256(serialized.encode()).hexdigest()}"
