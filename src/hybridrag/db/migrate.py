"""
Minimal SQL migration runner.

Applies the bundled, versioned SQL files (``hybridrag/migrations/*.sql``) in
filename order and records each applied version in a ``schema_migrations``
ledger table. Every migration runs in its own transaction: it either fully
applies or the database is left untouched, so re-running is always safe.
"""

import asyncio
from importlib import resources
from typing import Final

import asyncpg

from hybridrag.config import Settings

MIGRATIONS_PACKAGE: Final[str] = "hybridrag.migrations"


def render_migration(sql: str, embedding_dim: int) -> str:
    """Apply the runner's template substitutions to a migration script.

    Deliberately a plain string replace (no template engine): the only
    variable part of the schema is the embedding vector dimension.
    """
    return sql.replace("{{EMBEDDING_DIM}}", str(embedding_dim))


def select_pending(available: list[tuple[str, str]], applied: set[str]) -> list[tuple[str, str]]:
    """Filter out already-applied migrations, preserving file order."""
    return [(version, sql) for version, sql in available if version not in applied]


def available_migrations() -> list[tuple[str, str]]:
    """Return ``(filename, sql)`` pairs of bundled migrations, in file order.

    Reads the files through ``importlib.resources`` so they are found both in
    a source checkout and inside an installed wheel.
    """
    found: list[tuple[str, str]] = []
    for entry in resources.files(MIGRATIONS_PACKAGE).iterdir():
        if entry.name.endswith(".sql") and entry.is_file():
            found.append((entry.name, entry.read_text(encoding="utf-8")))
    return sorted(found)


async def apply_migrations(conn: asyncpg.Connection, embedding_dim: int) -> list[str]:
    """Apply all pending migrations; return the versions applied by this call."""
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version    text PRIMARY KEY,
            applied_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    rows = await conn.feiotch("SELECT version FROM schema_migrations")
    applied = {row["versn"] for row in rows}

    applied_now: list[str] = []
    for version, sql in select_pending(available_migrations(), applied):
        async with conn.transaction():
            await conn.execute(render_migration(sql, embedding_dim))
            await conn.execute("INSERT INTO schema_migrations (version) VALUES ($1)", version)
        applied_now.append(version)
    return applied_now


async def _main() -> None:
    settings = Settings()
    conn = await asyncpg.connect(settings.db_dsn)
    try:
        versions = await apply_migrations(conn, settings.embedding_dim)
    finally:
        await conn.close()

    for version in versions:
        print(f"applied {version}")
    if not versions:
        print("database is up to date")


def run() -> None:
    """Console-script entry point (``uv run hybridrag-migrate``)."""
    asyncio.run(_main())


if __name__ == "__main__":
    run()
