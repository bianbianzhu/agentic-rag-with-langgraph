# Agentic RAG Course

This context defines the shared language for the production-oriented Agentic RAG system developed throughout the course.

## Language

**Reference System**:
The single end-to-end Agentic RAG application that the course progressively designs, implements, and operates.
_Avoid_: Demo, toy app, sample bot

**Agentic RAG**:
A retrieval-augmented generation system that makes bounded runtime decisions about whether, where, and how to retrieve evidence before answering.
_Avoid_: RAG chatbot, autonomous RAG

**Agent**:
A unit that makes bounded, non-deterministic decisions from context, such as selecting tools or choosing the next step.
_Avoid_: Component, service, pipeline stage

**Component**:
A deterministic unit that performs one defined operation, such as embedding, indexing, retrieval execution, or reranking.
_Avoid_: Agent

**Engineering Knowledge Assistant**:
The user-facing Reference System for multi-turn questions over versioned technical documentation, architecture guidance, and operational runbooks.
_Avoid_: Chatbot, support bot

**Conversation Thread**:
A persistent sequence of related user questions and assistant answers bound to one Principal, whose currently authorized context can inform later questions in the same sequence.
_Avoid_: Session, chat history

**Thread Memory**:
The bounded conversation information retained across turns within one Conversation Thread.
_Avoid_: Chat history, long-term memory, graph state

**Turn**:
One identified attempt within a Conversation Thread to handle a Principal's message and reach a terminal outcome, including any safe resume of that attempt.
_Avoid_: Run, request, message

**Turn Record**:
The durable semantic record of one completed exchange, linking the original question, its standalone interpretation, and the final Cited Answer or refusal.
_Avoid_: Message pair, run, checkpoint

**Standalone Question**:
The current user's question rewritten only as needed to resolve references to authorized Thread Memory while preserving the user's intent and constraints.
_Avoid_: Search query, query rewrite, clarified question

**Conversation Summary**:
A bounded, provenance-aware representation of older completed Turn Records used to preserve relevant context after those records leave active Thread Memory.
_Avoid_: Chat summary, transcript summary, long-term memory

**Current Turn Work**:
Transient information created while answering one user turn that is not valid as memory for a later turn.
_Avoid_: Scratchpad, working memory, intermediate state

**Turn Execution Budget**:
A versioned set of code-owned limits on one Turn's model calls, Retrieval Requests, research iterations, Evidence volume, concurrency, retries, and elapsed time.
_Avoid_: Prompt instruction, recursion limit, token limit

**Context Projection**:
A bounded, role-specific view of currently authorized Thread Memory and Current Turn Work supplied to one Agent or Component.
_Avoid_: Prompt context, chat history, graph state

**Untrusted Content**:
User-supplied or Source Document content that may express intent or provide Evidence but has no authority to change a Principal, Access Scope, tool availability, execution budget, or system contract.
_Avoid_: Instructions, trusted context, safe document

**Access Scope**:
The set of Source Documents a Principal may retrieve within the single organization served by the Reference System, derived only from trusted organization-public, direct Principal, and Group Access Grants.
_Avoid_: Tenant, visibility filter

**Principal**:
The runtime representation of the authenticated person whose Access Scope constrains retrieval in the Reference System.
_Avoid_: User, account, identity

**Access Grant**:
A trusted authorization relation that permits organization-wide, direct Principal, or Group access to one Source Document; absence of a matching Access Grant denies access by default.
_Avoid_: Permission prompt, Agent-selected filter, chunk ACL

**Authorization Snapshot**:
The versioned effective Access Scope captured for one Turn and revalidated before Evidence enters a model Context Projection and before its Cited Answer is returned.
_Avoid_: Cached permissions, Agent context, login session

**Knowledge Source**:
A bounded internal corpus with a distinct retrieval purpose, specifically Engineering Docs or Operational Runbooks in v1.
_Avoid_: Data source, collection, knowledge base

**Source Document**:
One file within a Knowledge Source, identified by its normalized source-relative path; content changes preserve its identity, while a move or rename creates a new Source Document.
_Avoid_: Document, file, record

**Source Revision**:
A version of a Source Document determined by its source content and Access Scope metadata.
_Avoid_: Source fingerprint, file version

**Processing Revision**:
A version of indexed output determined by the parsing, chunking, and embedding semantics applied to a Source Revision.
_Avoid_: Processing fingerprint, pipeline version

**Indexed Chunk**:
A revision-scoped passage produced from one Source Document; its identity includes the Source Revision, Processing Revision, and ordinal within that output.
_Avoid_: Chunk, passage, vector record

**Evidence Item**:
An authorized Indexed Chunk used as one atomic unit of support for an answer; it preserves the matched chunk's identity and source location even when neighboring context is included for readability.
_Avoid_: Search result, context, document

**Source Locator**:
A structured position within a Source Document that identifies where an Evidence Item originated and supports deterministic citation rendering.
_Avoid_: URL, line number, generated citation

**Retrieval Request**:
A bounded request to find evidence for one query or subquestion in selected Knowledge Sources; it expresses retrieval intent but carries no authority to choose a Principal or expand Access Scope.
_Avoid_: Search query, tool call, retrieval config

**Retrieval Result**:
The structured outcome of one Retrieval Request, distinguishing completed retrieval, no evidence, and failure while carrying authorized Evidence Items and non-prompt diagnostics.
_Avoid_: Results, context list, tool output

**Retrieval Provenance**:
Metadata that records how a Retrieval Request produced and ranked an Evidence Item without becoming part of the supporting evidence itself.
_Avoid_: Relevance, evidence metadata, model reasoning

**Evidence Set**:
A bounded collection of authorized Evidence Items assembled across Retrieval Results for one answer attempt, preserving request coverage, provenance, and completeness.
_Avoid_: Context window, retrieved documents, search results

**Cited Answer**:
An answer whose factual claims reference authorized Evidence Items through validated citation keys and a structured citation map.
_Avoid_: Response, sourced answer, answer with links

**Corpus Sync**:
A repeatable batch reconciliation that makes an indexed Knowledge Source match its current documents through incremental additions, updates, and deletions.
_Avoid_: Batch sync, ingestion run, real-time sync

**Corpus Revision**:
The complete indexed state of one Knowledge Source produced by a successfully published Corpus Sync.
_Avoid_: Index version, sync version, snapshot

**Sync Report**:
The durable outcome record for one attempted Corpus Sync, including its status, change counts, and any failure summary.
_Avoid_: Sync log, ingestion report, audit log
