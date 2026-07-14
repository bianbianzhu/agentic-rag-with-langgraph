# Payments Rollback Recovery

## Diagnosis procedure

Check the rollback worker logs and confirm the database schema version.

## Recovery procedure

Pause the consumer, deploy the schema-v2-compatible worker, run the schema
check, and replay pending jobs. Replay is idempotent by `rollback_job_id`.

## Staging timeout

Staging failed because its migration job timed out before schema v2 completed.
