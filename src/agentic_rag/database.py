"""PostgreSQL pool and migration acquisition."""

from pathlib import Path
from typing import LiteralString, cast

from psycopg import sql
from psycopg_pool import ConnectionPool


def open_database_pool(database_url: str) -> ConnectionPool:
    """Open the one PostgreSQL connection pool used by application components."""

    if not database_url.strip():
        raise ValueError("database_url must not be empty")
    return ConnectionPool(
        conninfo=database_url, min_size=1, max_size=4, open=True
    )


def apply_migrations(pool: ConnectionPool, migrations_root: Path) -> None:
    """Apply each forward-only SQL migration exactly once."""

    migration_paths = sorted(migrations_root.glob("[0-9][0-9][0-9][0-9]_*.sql"))
    with pool.connection() as connection:
        with connection.transaction():
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version text PRIMARY KEY,
                    applied_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )
            applied = {
                row[0]
                for row in connection.execute(
                    "SELECT version FROM schema_migrations"
                ).fetchall()
            }
            for migration_path in migration_paths:
                if migration_path.name in applied:
                    continue
                connection.execute(
                    sql.SQL(
                        cast(
                            LiteralString,
                            migration_path.read_text(encoding="utf-8"),
                        )
                    )
                )
                connection.execute(
                    "INSERT INTO schema_migrations (version) VALUES (%s)",
                    (migration_path.name,),
                )
