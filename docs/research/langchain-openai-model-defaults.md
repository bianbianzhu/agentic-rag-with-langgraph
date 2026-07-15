# LangChain and OpenAI Model Defaults for the Reference System

Research date: 2026-07-15

> Project decision: the research below originally recommended GPT-5.6 Terra as the simplest single-model default. The accepted v1 boundary decision instead pins GPT-5.4 mini for Agent roles and GPT-5.4 nano for reranking to preserve the existing Turn and experiment cost headroom. See the final codebase and course boundary ticket; treat its project decision as authoritative while retaining this report as the external-facts record.

## Decision

Use one OpenAI chat model family and one embedding model in v1:

| Role | Project default | Why |
| --- | --- | --- |
| contextualization, planning, evidence assessment, answer generation, verification | `openai:gpt-5.6-terra` through `init_chat_model` | OpenAI positions Terra as the intelligence/cost balance and specifically recommends it for conversational interfaces. One model keeps the course and evaluation matrix small. |
| global reranking | the same `gpt-5.6-terra`, returning an ordered list of candidate IDs with Structured Outputs | There is no current standalone OpenAI reranking model/API for arbitrary application candidates. Reusing Terra avoids a second provider or local model. |
| indexing and query embeddings | `text-embedding-3-small`, native dimensions | It is OpenAI's smaller, lower-cost v3 embedding model and is sufficient as the v1 starting point. Promote to `text-embedding-3-large` only if retrieval evaluation demonstrates a material gap. |
| deterministic tests | model doubles plus a deterministic reranker callable | Exact graph paths and exact reranked ordering cannot depend on a live model. |

The recommended chat defaults are `reasoning_effort="low"`, `use_responses_api=True`, `max_completion_tokens=4096`, `timeout=30`, and `max_retries=2`. Omit `temperature`. The first two choices follow current OpenAI guidance; the numeric limits are project defaults bounded by the already-decided 5,000 aggregate output-token and 90-second Turn budgets. Change them only through versioned configuration and evaluation.

Do not split v1 across Sol, Terra, and Luna by role. OpenAI recommends starting experiments with flagship `gpt-5.6`, and offers Terra or Luna for lower cost and latency, but a multi-model router would add configuration and evaluation work before this small reference system proves that it needs it ([latest model guide](https://developers.openai.com/api/docs/guides/latest-model), [building agents](https://developers.openai.com/tracks/building-agents)).

## Current LangChain construction

**Official facts:** `init_chat_model` is imported from `langchain.chat_models`. The OpenAI integration remains a required dependency even though application code does not import `ChatOpenAI` directly. LangChain documents either `pip install -U "langchain[openai]"` or the dedicated `langchain-openai` package, and recommends a provider-prefixed identifier such as `openai:gpt-5.5` when the provider should be explicit ([models](https://docs.langchain.com/oss/python/langchain/models), [providers and models](https://docs.langchain.com/oss/python/concepts/providers-and-models), [`init_chat_model` reference](https://reference.langchain.com/python/langchain/chat_models/base/init_chat_model)).

For this uv project, install with one command:

```bash
uv add "langchain[openai]"
```

Then construct the fixed live dependencies once and place them in trusted `RuntimeContext`:

```python
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings

chat_model = init_chat_model(
    "openai:gpt-5.6-terra",
    reasoning_effort="low",
    use_responses_api=True,
    max_completion_tokens=4096,
    timeout=30,
    max_retries=2,
)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
```

Provider prefixes are not mandatory when LangChain can infer the provider, but they prevent ambiguity and make the configuration fingerprint readable. New OpenAI model names are normally passed through without waiting for a LangChain release. Keep `configurable_fields=None`, the default: LangChain warns that unrestricted runtime configurability can let untrusted input replace credentials or redirect `base_url` ([`init_chat_model` reference](https://reference.langchain.com/python/langchain/chat_models/base/init_chat_model)).

## Chat-model choice

**Official facts:** the current OpenAI family is GPT-5.6. `gpt-5.6-sol` is the highest-capability model, `gpt-5.6-terra` balances intelligence and cost, and `gpt-5.6-luna` targets efficient, cost-sensitive, high-volume workloads. OpenAI recommends the Responses API for reasoning, tool use, and multi-turn agent workflows. It recommends medium reasoning as a balanced starting point generally, and low reasoning for latency-sensitive workloads; higher effort should be justified by evaluations ([latest model guide](https://developers.openai.com/api/docs/guides/latest-model), [Terra model](https://developers.openai.com/api/docs/models/gpt-5.6-terra), [Luna model](https://developers.openai.com/api/docs/models/gpt-5.6-luna)).

**Project recommendation:** Terra with low reasoning is the simplest cost-bounded choice for the conversational reference system. The graph already supplies bounded retries, retrieval, verification, and a 90-second deadline, so the model should not spend medium reasoning on every small routing schema by default. Run the L4 LangSmith experiment before changing this default. If quality fails, first test Terra at medium effort; test Sol only after that. Record the provider-returned resolved model metadata in traces because an undated model ID can change behind the service.

## Embeddings

**Official facts:** OpenAI calls `text-embedding-3-small` its improved small embedding model and `text-embedding-3-large` its most capable embedding model for English and non-English tasks ([small model](https://developers.openai.com/api/docs/models/text-embedding-3-small), [large model](https://developers.openai.com/api/docs/models/text-embedding-3-large)). LangChain's current integration is `OpenAIEmbeddings` from `langchain_openai`; it supports v3 models and an optional `dimensions` argument ([LangChain OpenAI embeddings](https://docs.langchain.com/oss/python/integrations/embeddings/openai)).

**Project recommendation:** start with `text-embedding-3-small` and its native dimensions. Avoid a custom dimension until retrieval evaluation shows a storage or latency need. Changing either model or dimensions creates a new Processing Revision and requires re-embedding; include both in the retrieval configuration fingerprint. Cosine and dot-product rankings are equivalent for OpenAI's normalized embeddings, but keep the chosen pgvector operator explicit and tested ([OpenAI embeddings guide](https://developers.openai.com/api/docs/guides/embeddings)).

## One global reranker without another provider

As of the research date, OpenAI's public model catalog exposes embedding models and File Search performs internal ranking, but it does not expose a dedicated standalone reranker model/API for arbitrary candidate lists ([model catalog](https://developers.openai.com/api/docs/models), [File Search](https://developers.openai.com/api/docs/guides/tools-file-search)). LangChain has provider-specific and local rerank integrations; its generic LLM listwise reranker demonstrates the Structured Output pattern but currently lives in `langchain-classic` ([`LLMListwiseRerank` reference](https://reference.langchain.com/python/langchain-classic/retrievers/document_compressors/listwise_rerank/LLMListwiseRerank)).

Do not add `langchain-classic` just for this wrapper. Keep the already-approved small `Reranker` callable and implement it locally:

1. Authorization-filter dense and lexical retrieval in SQL, fuse with RRF, then pass only the bounded authorized candidates to the reranker.
2. Send candidate ID, source ID, title, and bounded chunk text; request an ordered subset of candidate IDs using a Pydantic Structured Output schema.
3. Reject unknown IDs, duplicates, excessive output, and schema failure as `rerank_failed`. Do not silently fall back to RRF.
4. Assign ranks deterministically from the returned order; never authorize or generate facts in the reranker.

This is a project design, not an OpenAI or LangChain prescription. It preserves one global semantic reranker while satisfying the v1 prohibition on a second provider and local models. L2/L3 use the deterministic implementation; L4 exercises the live Terra implementation.

## Structured-output and parameter caveats

LangChain supports Pydantic Structured Outputs through `with_structured_output(..., method="json_schema")`, while OpenAI recommends Structured Outputs over plain JSON mode when application code consumes the result ([LangChain OpenAI integration](https://docs.langchain.com/oss/python/integrations/chat/openai), [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)). Apply that to every Agent decision and the reranker.

Keep schemas small, frozen, and closed to extra fields. Prefer explicit outcome enums such as `sufficient`, `refine`, and `incomplete` over webs of optional fields. Handle model refusal separately. Schema validity is not semantic correctness: code must still check candidate IDs, citation IDs, budgets, and authorization. OpenAI supports only a subset of JSON Schema in strict mode, and LangChain notes stricter limitations for tool schemas, so avoid clever schema features and defaults.

Use `max_completion_tokens`, not the deprecated `max_tokens`. Give reasoning enough output headroom: too small a completion budget can be consumed by reasoning before visible structured output is produced. Do not set `temperature=0` as a universal determinism switch; omit unsupported sampling controls for this reasoning family, validate schemas, and use deterministic doubles where exact repeatability is required ([LangChain OpenAI integration](https://docs.langchain.com/oss/python/integrations/chat/openai)).

## Versioned configuration versus secrets

Commit and fingerprint these non-secret values:

- chat model ID, Responses API choice, reasoning effort, token limit, timeout, and retries;
- embedding model ID and dimensions;
- reranker model ID, prompt/schema version, candidate budget, result budget, and failure policy;
- every Agent prompt/schema version and the retrieval configuration fingerprint;
- LangSmith project name and tracing enabled/disabled flag, if environment-specific overrides are documented rather than committed as values.

Never commit `OPENAI_API_KEY` or `LANGSMITH_API_KEY`. Load them from the environment or an ignored `.env`; a committed `.env.example` contains names only. LangChain documents `OPENAI_API_KEY` for OpenAI and `LANGSMITH_API_KEY` plus `LANGSMITH_TRACING=true` for tracing ([ChatOpenAI setup](https://docs.langchain.com/oss/python/integrations/chat/openai)). Database URLs containing credentials are secrets as well.

Do not expose `api_key`, `base_url`, model ID, reasoning effort, or budgets as user- or graph-state-configurable fields. They belong to the frozen trusted `RuntimeContext`; State may record their non-secret identity for reproducibility but must never checkpoint the objects or credentials.

## Implementation checkpoint

Proceed with Terra/low, Responses API, small embeddings, and a local Structured Output reranker callable. Before calling these defaults accepted, run the tagged L4 LangSmith dataset and compare at least Terra/low versus Terra/medium on answer quality, reranking quality, latency, and cost. A model upgrade or role split is justified only by that evidence.
