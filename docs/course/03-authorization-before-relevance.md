# Chapter 03: Authorization Before Relevance

## Outcome

The Reference System now derives each Principal's Access Scope from trusted
PostgreSQL relations before any later relevance query may return data. A Source
Document is allowed only when at least one explicit organization-public, direct
Principal, or Group Access Grant matches. No matching relation means default
deny.

Authentication remains upstream of development-only v1. Runtime Context
supplies a Principal identifier; the authorization Component verifies that the
Principal was provisioned. User messages, Source Document content, Agent output,
and graph State cannot create a Principal, Group membership, or Access Grant.

## Trusted data is separate from content

The source corpus remains Untrusted Content. Trusted fixtures are stored outside
every Knowledge Source root:

```text
fixtures/reference-system/
├── corpus/                         Untrusted Content
└── trusted/
    ├── principals.json             Principal + Group membership
    └── access-grant-sets.json      Source Document Access Grants
```

The database mirrors that separation:

```text
principals ──< principal_group_memberships >── groups

source_documents ──< access_grants
                       ├── organization-public
                       ├── principal:<principal_id>
                       └── group:<group_id>
```

Indexed Chunks do not carry copied authorization fields. They inherit access
through their Source Document foreign key, so one predicate remains authoritative
when an Indexed Chunk is retrieved, expanded, or hydrated later.

## One shared SQL predicate

PostgreSQL owns one parameterized `source_document_is_authorized(...)` function.
The authorization module calls it while hiding Group resolution and default-deny
mechanics from callers. The function first proves that the bound `principal_id`
names a provisioned Principal, then requires one of the three matching Access
Grant relations for the current Source Document.

```text
known Principal
      AND
(
  organization-public
  OR direct Principal match
  OR trusted Group membership match
)
```

The Principal is always a bound SQL parameter; it is never interpolated from
text. An input such as `alice' OR true --` therefore resolves to no Principal
instead of changing the query. Later dense retrieval, lexical retrieval, context
expansion, citation hydration, and history revalidation must call this same
database predicate inside their private queries before rows leave PostgreSQL.

## Authorization Snapshot

`capture_authorization_snapshot(...)` verifies the Principal and hashes the
sorted identities in the currently effective Access Scope into a content-free
revision. The snapshot contains only:

```text
principal_id + auth revision
```

It contains no Group list, Access Grant details, Source Document identities, or
content. `authorization_snapshot_is_current(...)` captures the effective scope
again and compares revisions. Removing Alice's direct settlement grant changes
her revision; the stale snapshot no longer validates. Chapter 09 will connect
this revalidation seam to bounded graph restart and final-answer checks.

## Northstar Labs grant matrix

The S1 fixture proves every v1 grant relation:

| Source Document | Relation | Alice | Bob | Carol |
| --- | --- | :---: | :---: | :---: |
| D1–D3 Engineering Docs | `payments-engineering` Group | allow | allow | deny |
| D4 incident announcement | organization-public | allow | allow | allow |
| D7 rollback Runbook | `payments-on-call` Group | allow | deny | deny |
| D8 settlement impact | `finance` Group + direct Alice | allow | deny | allow |

Alice demonstrates multiple Group membership and a direct grant. Bob proves that
relevant Operational Runbook content is still forbidden without the on-call
Group. Carol proves that finance access does not imply engineering access. A
known Principal with no matching Access Grant receives only `False`; inaccessible
identities, counts, and Access Grant details are not returned.

## Run the verification

Start the local PostgreSQL/pgvector service, then run L1, L2, and accumulated
checks:

```bash
docker compose up -d postgres
uv run pytest tests/unit/test_authorization.py -q
uv run pytest tests/data_integration/test_authorization_postgres.py -q
uv run pytest -q
uv run pyright
```

The checks reject unknown or injected Principal identifiers and schema fields,
prove default deny, exercise the exact Alice/Bob/Carol matrix through real
PostgreSQL, and invalidate a snapshot after an effective Access Scope change.
They use no model or live credential.

Hybrid relevance ranking, read-only online retrieval, Evidence construction,
authorization-aware graph routing, mid-Turn restart, authentication, production
RLS, and deployment remain outside this chapter.
