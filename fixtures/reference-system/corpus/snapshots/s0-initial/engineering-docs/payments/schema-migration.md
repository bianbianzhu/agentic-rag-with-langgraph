# Payments Schema Migration

## Schema change

Schema v1 stores the provider token in `rollback_token`. The planned schema v2
migration will rename that column to `reversal_token`.

## Rollback compatibility

Worker v1.8 still reads `rollback_token` and must be upgraded with the schema.
