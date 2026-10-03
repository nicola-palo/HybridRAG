from pathlib import Path

import pytest
from pydantic import ValidationError

from hybridrag.config import Settings


@pytest.fixture(autouse=True)
def _isolate_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run from an empty directory so no real .env file is ever picked up."""
    monkeypatch.chdir(tmp_path)


def test_defaults_and_dsn_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_DB_PASSWORD", "s3cret p@ss")

    settings = Settings()

    assert settings.db_host == "localhost"
    assert settings.db_port == 5432
    assert settings.db_user == "rag_user"
    assert settings.db_dsn == "postgresql://rag_user:s3cret%20p%40ss@localhost:5432/rag_db"


def test_environment_variables_beat_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_DB_HOST", "db.internal")
    monkeypatch.setenv("RAG_DB_PORT", "6543")
    monkeypatch.setenv("RAG_DB_NAME", "other_db")

    settings = Settings()

    assert settings.db_host == "db.internal"
    assert settings.db_port == 6543
    assert settings.db_name == "other_db"


def test_out_of_range_port_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_DB_PORT", "99999")

    with pytest.raises(ValidationError):
        Settings()
