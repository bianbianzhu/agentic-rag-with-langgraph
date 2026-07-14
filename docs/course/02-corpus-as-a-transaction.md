# Chapter 02: Corpus as a Transaction

## Outcome

The Reference System can now reconcile either v1 Knowledge Source into local
PostgreSQL with pgvector. Corpus Sync is synchronous batch work: it discovers a
complete source snapshot, processes every changed Source Document, and then
publishes additions, replacements, and deletions in one transaction.

This chapter indexes Untrusted Content. It does not retrieve that content or
decide who may read it. Trusted Access Scope metadata enters as a revision value
owned by the caller; Chapter 03 will persist and enforce the actual grants.

## Identity and revisions

A Source Document identity is derived from two code-owned inputs:

```text
Knowledge Source + normalized source-relative path -> Source Document ID
```

Editing a Source Document therefore preserves identity. A rename produces one
deletion and one addition. Two independent revisions decide whether processing
is needed:

```text
raw source bytes + trusted Access Scope revision -> Source Revision
parser + chunker + embedding model              -> Processing Revision
```

If either revision changes, Corpus Sync replaces every Indexed Chunk for that
Source Document. If neither changes, it skips parsing and embedding. Indexed
Chunk identity includes the Source Document, both revisions, and its output
ordinal; old and new processing output cannot accidentally share an identifier.

## Publication boundary

The expensive work happens before publication:

```text
complete discovery
      |
      v
validate UTF-8 and trusted metadata
      |
      v
parse headings -> Indexed Chunks -> embeddings
      |
      v
per-Knowledge-Source advisory lock
      |
      v
one PostgreSQL transaction
  delete absent Source Documents
  replace changed Source Documents and Indexed Chunks
  update Corpus Revision
  insert successful Sync Report
```

Only a complete snapshot may infer deletions. An escaping symlink, missing
trusted metadata, invalid Markdown input, or exhausted embedding retry budget
fails the whole attempt before publication. The earlier Corpus Revision remains
queryable without partial Source Documents or mixed Indexed Chunk revisions. A
failed Sync Report is then committed separately and contains the failed path and
a bounded summary, never source text or embeddings.

## Frozen Northstar Labs snapshots

The synthetic CC BY 4.0 corpus lives under
`fixtures/reference-system/corpus/`:

- `s0-initial` contains six Source Documents across Engineering Docs and
  Operational Runbooks.
- `s1-baseline` updates the schema migration, changes only the incident
  announcement's Access Scope revision, renames the legacy-worker Source
  Document, preserves three Source Documents, and includes one unsupported text
  source.
- the `invalid` overlay supplies a supported Markdown Source Document with
  invalid UTF-8.

Trusted grant policies and their S0/S1 assignments live separately in
`fixtures/reference-system/trusted/access-grant-sets.json`; Source Document
content cannot declare or modify them.

The resulting aggregate reports are deterministic:

| Snapshot | Added | Updated | Deleted | Unchanged | Ignored |
| --- | ---: | ---: | ---: | ---: | ---: |
| S0 | 6 | 0 | 0 | 0 | 0 |
| S1 after S0 | 1 | 2 | 1 | 3 | 1 |

S1 demonstrates why content hashes alone are insufficient: the public incident
announcement keeps identical bytes and Source Document identity, but its trusted
Access Scope revision changes, so its Source Revision and Indexed Chunks are
replaced.

## Run the local verification

Start the one development persistence service and run deterministic L1 and L2
checks:

```bash
docker compose up -d postgres
uv run pytest tests/unit/test_corpus_identity.py -q
uv run pytest tests/data_integration/test_corpus_sync.py -q
uv run pytest -q
uv run pyright
```

The default test connection is
`postgresql://agentic_rag@127.0.0.1:55432/agentic_rag`. Set
`TEST_DATABASE_URL` to use another disposable local database. No OpenAI or
LangSmith credential is used by Chapter 02; the tests provide deterministic
embedding vectors.

## Verification

Chapter 02 is complete when identity functions reject unsafe paths, S0 and S1
produce the exact reports above, unchanged Source Documents perform no embedding
work, a Processing Revision change rebuilds all affected Source Documents,
transient embedding
failures stop after three attempts, escaping symlinks fail closed, and invalid
UTF-8 leaves the previously published Source Documents, Indexed Chunks, and
Corpus Revision byte-for-byte unchanged while preserving a durable failed Sync
Report.

Access Grant persistence, authorized retrieval, rank fusion, reranking, answer
generation, and deployment remain outside this chapter.
