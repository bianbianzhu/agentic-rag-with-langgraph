# Payments Platform Overview

## Platform topology

The Payments Platform records rollback jobs in PostgreSQL and sends approved
reversals to the payment provider through a rollback worker.
