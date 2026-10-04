"""Typed application configuration, loaded from the environment.

Follows the 12-factor methodology: configuration lives in environment
variables (prefix ``RAG_``), optionally backed by a local ``.env`` file that
is never committed. See ``.env.example`` for the expected variables.
"""

from typing import Self
from urllib.parse import quote

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All settings required by the application, validated at startup.

    Every field has a development-only default; environment variables
    (``RAG_``-prefixed) always win over defaults, so real deployments and the
    docker-compose stack (which fails to start unconfigured) stay explicit.
    """

    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    db_host: str = "localhost"
    db_port: int = Field(default=5432, ge=1, le=65535)
    db_user: str = "rag_user"
    db_password: str = "rag_password"
    db_name: str = "rag_db"
    # Dimension of the embedding model's vectors. HNSW indexes support up to
    # 2000 dimensions; changing the model requires re-embedding and a schema
    # migration. bge-m3 = 1024, nomic-embed-text = 768,
    # text-embedding-3-small = 1536.
    embedding_dim: int = Field(default=1024, ge=1, le=2000)
    # Chunking: target size per chunk (estimated tokens) and trailing context
    # repeated between consecutive chunks. The heuristic assumes ~4 characters
    # per token; swap in a real tokenizer if precision ever matters.
    chunk_target_tokens: int = Field(default=512, ge=16, le=4096)
    chunk_overlap_tokens: int = Field(default=64, ge=0, le=1024)

    @model_validator(mode="after")
    def _overlap_below_target(self) -> Self:
        if self.chunk_overlap_tokens >= self.chunk_target_tokens:
            msg = "chunk_overlap_tokens must be smaller than chunk_target_tokens"
            raise ValueError(msg)
        return self

    @property
    def db_dsn(self) -> str:
        """PostgreSQL connection string, safe for special characters."""
        return (
            f"postgresql://{quote(self.db_user, safe='')}:{quote(self.db_password, safe='')}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )
