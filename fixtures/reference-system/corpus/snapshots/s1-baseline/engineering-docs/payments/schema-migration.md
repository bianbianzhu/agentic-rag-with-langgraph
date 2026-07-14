# Payments Schema Migration

## Schema change

Schema v2 renames `rollback_token` to `reversal_token`.

## Rollback incompatibility

Production worker v1.8 still queries `rollback_token`, receives PostgreSQL
`undefined_column`, and fails before sending a provider request.
