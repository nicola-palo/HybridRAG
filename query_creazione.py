import asyncio
import asyncpg

async def main():
    conn = await asyncpg.connect(
        host="localhost",
        port=5432,
        user="rag_user",
        password="rag_password",
        database="rag_db"
    )


    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    print(">>>Estensione pgvector abilitata con successo!")

    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())