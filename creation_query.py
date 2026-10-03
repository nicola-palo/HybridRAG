import asyncio

import asyncpg

from hybridrag.config import Settings


async def main() -> None:
    settings = Settings()
    conn = await asyncpg.connect(settings.db_dsn)

    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    print(">>>pgvector extension enabled successfully!")

    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
