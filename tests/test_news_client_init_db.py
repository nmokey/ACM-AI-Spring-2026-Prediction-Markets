"""Tests for nlp.news_client.init_db."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from nlp import news_client


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "headlines.db"
    monkeypatch.setattr(news_client, "DB_PATH", db_path)
    return db_path


def test_init_db_returns_connection(temp_db):
    conn = news_client.init_db()
    assert isinstance(conn, sqlite3.Connection)
    conn.close()


def test_init_db_creates_headlines_table(temp_db):
    conn = news_client.init_db()
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='headlines'"
    )
    assert cur.fetchone() is not None
    conn.close()


def test_init_db_schema_columns(temp_db):
    conn = news_client.init_db()
    cols = {row[1]: row[2] for row in conn.execute("PRAGMA table_info(headlines)")}
    assert cols == {
        "id": "TEXT",
        "text": "TEXT",
        "source": "TEXT",
        "url": "TEXT",
        "timestamp": "TEXT",
        "query": "TEXT",
    }
    pk_cols = [row[1] for row in conn.execute("PRAGMA table_info(headlines)") if row[5] == 1]
    assert pk_cols == ["id"]
    conn.close()


def test_init_db_is_idempotent(temp_db):
    conn1 = news_client.init_db()
    conn1.execute(
        "INSERT INTO headlines (id, text, source, url, timestamp, query) "
        "VALUES ('a', 't', 's', 'u', '2026-01-01T00:00:00Z', 'q')"
    )
    conn1.commit()
    conn1.close()

    conn2 = news_client.init_db()
    count = conn2.execute("SELECT COUNT(*) FROM headlines").fetchone()[0]
    assert count == 1
    conn2.close()


def test_init_db_text_not_null(temp_db):
    conn = news_client.init_db()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO headlines (id, text, source, url, timestamp, query) "
            "VALUES ('x', NULL, 's', 'u', 't', 'q')"
        )
    conn.close()
