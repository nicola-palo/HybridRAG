from collections.abc import AsyncIterator

import asyncpg
import pytest

from hybridrag.db.migrate import (
    apply_migrations,
    available_migrations,
    render_migration,
    select_pending,
)


def test_render_migration_substitutes_vector_dimension() -> None:
    assert render_migration("vector({{EMBEDDING_DIM}}) NOT NULL", 1536) == ("vector(1536) NOT NULL")


def test_available_migrations_are_discovered_in_order() -> None:
    versions = [version for version, _ in available_migrations()]

    assert versions == sorted(versions)
    assert versions  # the bundle ships at least one migration


def test_select_pending_preserves_order_and_filters_applied() -> None:
    available = sorted([("0002_b.sql", "B"), ("0001_a.sql", "A"), ("0003_c.sql", "C")])

    pending = select_pending(available, {"0001_a.sql"})

    assert [version for version, _ in pending] == ["0002_b.sql", "0003_c.sql"]


TEST_EMBEDDING_DIM = 8  # small on purpose: keeps the test container cheap


@pytest.fixture()
async def db_conn() -> AsyncIterator[asyncpg.Connection]:
    from testcontainers.community.postgres import PostgresContainer

    container: PostgresContainer | None = None
    try:
        # Container construction already opens the Docker client: everything
        # that touches the daemon stays inside this try, so a stopped daemon
        # skips the test instead of erroring it.
        container = PostgresContainer(
            "pgvector/pgvector:pg16", username="test", password="test", dbname="test"
        )
        container.start()
        host = container.get_container_host_ip()
        port = container.get_exposed_port(5432)
    except Exception as exc:
        pytest.skip(f"Docker unavailable: {exc}")
    assert container is not None

    conn = await asyncpg.connect(f"postgresql://test:test@{host}:{port}/test")
    try:
        yield conn
    finally:
        await conn.close()
        container.stop()


async def test_migrations_apply_idempotently(
    db_conn: asyncpg.Connection,
) -> None:
    first_run = await apply_migrations(db_conn, embedding_dim=TEST_EMBEDDING_DIM)
    assert first_run == ["0001_init.sql"]

    second_run = await apply_migrations(db_conn, embedding_dim=TEST_EMBEDDING_DIM)
    assert second_run == []


async def test_schema_supports_hybrid_rows(db_conn: asyncpg.Connection) -> None:
    await apply_migrations(db_conn, embedding_dim=TEST_EMBEDDING_DIM)

    extensions = {row["extname"] for row in await db_conn.fetch("SELECT extname FROM pg_extension")}
    assert {"vector", "pg_trgm"} <= extensions

    document_id = await db_conn.fetchval(
        "INSERT INTO documents (source_path, content, content_hash) "
        "VALUES ('docs/demo.md', 'irrelevant body', 'hash0') RETURNING id"
    )
    await db_conn.execute(
        "INSERT INTO chunks (document_id, ordinal, content, embedding) "
        "VALUES ($1, 0, 'Hybrid retrieval rocks', $2::vector)",
        document_id,
        "[1,2,3,4,5,6,7,8]",
    )

    row = await db_conn.fetchrow(
        "SELECT tsv, embedding FROM chunks WHERE document_id = $1", document_id
    )
    assert row is not None
    assert "retrieval" in row["tsv"]  # generated tsvector indexed the content
    assert row["embedding"].startswith("[")  # pgvector text representation
