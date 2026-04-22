"""Tests for nlp.news_client: _fetch_gdelt, fetch_headlines, store_headlines,
get_recent_headlines, _extract_query."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from nlp import news_client


# ──────────────────────────── Fixtures ────────────────────────────


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(news_client, "DB_PATH", tmp_path / "headlines.db")


@pytest.fixture
def no_key_client(temp_db, monkeypatch):
    monkeypatch.setattr(news_client, "GNEWS_API_KEY", "")
    c = news_client.NewsClient()
    yield c
    c.conn.close()


@pytest.fixture
def keyed_client(temp_db, monkeypatch):
    monkeypatch.setattr(news_client, "GNEWS_API_KEY", "FAKE_KEY")
    c = news_client.NewsClient()
    yield c
    c.conn.close()


def _mock_resp(payload):
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status = MagicMock()
    return resp


def _sample_headline(hid="abc", timestamp="2026-04-20T12:00:00Z"):
    return {
        "id": hid,
        "text": "hello world",
        "source": "Reuters",
        "url": f"https://r/{hid}",
        "timestamp": timestamp,
        "query": "bitcoin",
    }


# ──────────────────────────── _fetch_gdelt ────────────────────────────

GDELT_SAMPLE = {
    "articles": [
        {
            "title": "BTC surges past $70k",
            "domain": "reuters.com",
            "url": "https://reuters.com/a",
            "seendate": "20260420T120000Z",
        },
        {
            "title": "Rain expected in NYC",
            "domain": "nytimes.com",
            "url": "https://nytimes.com/b",
            "seendate": "20260420T130000Z",
        },
    ]
}


def _patch_gdelt(monkeypatch, payload, captured=None):
    def fake_get(url, params=None, **kw):
        if captured is not None:
            captured["url"] = url
            captured["params"] = params
        return _mock_resp(payload)
    monkeypatch.setattr(news_client.requests, "get", fake_get)


def test_fetch_gdelt_returns_dicts_with_required_keys(no_key_client, monkeypatch):
    _patch_gdelt(monkeypatch, GDELT_SAMPLE)
    result = no_key_client._fetch_gdelt("bitcoin", 10)
    assert len(result) == 2
    assert set(result[0].keys()) == {"id", "text", "source", "url", "timestamp", "query"}


def test_fetch_gdelt_parses_title_and_domain(no_key_client, monkeypatch):
    _patch_gdelt(monkeypatch, GDELT_SAMPLE)
    result = no_key_client._fetch_gdelt("bitcoin", 10)
    assert result[0]["text"] == "BTC surges past $70k"
    assert result[0]["source"] == "reuters.com"
    assert result[0]["url"] == "https://reuters.com/a"
    assert result[0]["query"] == "bitcoin"


def test_fetch_gdelt_preserves_seendate(no_key_client, monkeypatch):
    _patch_gdelt(monkeypatch, GDELT_SAMPLE)
    result = no_key_client._fetch_gdelt("bitcoin", 10)
    assert result[0]["timestamp"] == "20260420T120000Z"


def test_fetch_gdelt_assigns_unique_ids(no_key_client, monkeypatch):
    _patch_gdelt(monkeypatch, GDELT_SAMPLE)
    result = no_key_client._fetch_gdelt("bitcoin", 10)
    ids = [h["id"] for h in result]
    assert len(set(ids)) == len(ids)
    assert all(i for i in ids)


def test_fetch_gdelt_calls_correct_endpoint(no_key_client, monkeypatch):
    captured = {}
    _patch_gdelt(monkeypatch, GDELT_SAMPLE, captured)
    no_key_client._fetch_gdelt("bitcoin price", 5)
    assert "gdeltproject.org" in captured["url"]
    assert captured["params"]["query"] == "bitcoin price"
    assert captured["params"]["maxrecords"] == 5
    assert captured["params"]["format"] == "json"


def test_fetch_gdelt_returns_empty_on_network_error(no_key_client, monkeypatch):
    import requests as _r

    def boom(*a, **k):
        raise _r.ConnectionError("network down")

    monkeypatch.setattr(news_client.requests, "get", boom)
    assert no_key_client._fetch_gdelt("bitcoin", 10) == []


# ──────────────────────────── fetch_headlines ────────────────────────────


def test_fetch_headlines_falls_back_to_gdelt_when_no_key(no_key_client, monkeypatch):
    called = {}

    def fake_gdelt(query, max_results):
        called["args"] = (query, max_results)
        return [_sample_headline("g1")]

    monkeypatch.setattr(no_key_client, "_fetch_gdelt", fake_gdelt)
    result = no_key_client.fetch_headlines("bitcoin", 5)
    assert called["args"] == ("bitcoin", 5)
    assert result[0]["id"] == "g1"


GNEWS_SAMPLE = {
    "articles": [
        {
            "title": "BTC rises",
            "description": "Bitcoin climbs 5%",
            "source": {"name": "CoinDesk"},
            "url": "https://cd/a",
            "publishedAt": "2026-04-20T12:00:00Z",
        }
    ]
}


def test_fetch_headlines_uses_gnews_when_key_present(keyed_client, monkeypatch):
    captured = {}

    def fake_get(url, params=None, **kw):
        captured["url"] = url
        captured["params"] = params
        return _mock_resp(GNEWS_SAMPLE)

    monkeypatch.setattr(keyed_client.session, "get", fake_get)
    result = keyed_client.fetch_headlines("bitcoin", 5)
    assert "gnews.io" in captured["url"]
    assert captured["params"]["apikey"] == "FAKE_KEY"
    assert captured["params"]["q"] == "bitcoin"
    assert captured["params"]["max"] == 5
    assert len(result) == 1


def test_fetch_headlines_concatenates_title_and_description(keyed_client, monkeypatch):
    monkeypatch.setattr(keyed_client.session, "get", lambda *a, **k: _mock_resp(GNEWS_SAMPLE))
    result = keyed_client.fetch_headlines("bitcoin", 5)
    assert "BTC rises" in result[0]["text"]
    assert "Bitcoin climbs 5%" in result[0]["text"]


def test_fetch_headlines_parses_gnews_fields(keyed_client, monkeypatch):
    monkeypatch.setattr(keyed_client.session, "get", lambda *a, **k: _mock_resp(GNEWS_SAMPLE))
    result = keyed_client.fetch_headlines("bitcoin", 5)
    h = result[0]
    assert h["source"] == "CoinDesk"
    assert h["url"] == "https://cd/a"
    assert h["timestamp"] == "2026-04-20T12:00:00Z"
    assert h["query"] == "bitcoin"


# ──────────────────────────── store_headlines ────────────────────────────


def test_store_headlines_inserts_rows(no_key_client):
    n = no_key_client.store_headlines([_sample_headline("a"), _sample_headline("b")])
    assert n == 2
    count = no_key_client.conn.execute("SELECT COUNT(*) FROM headlines").fetchone()[0]
    assert count == 2


def test_store_headlines_skips_duplicate_ids(no_key_client):
    no_key_client.store_headlines([_sample_headline("a")])
    n = no_key_client.store_headlines([_sample_headline("a"), _sample_headline("b")])
    assert n == 1
    count = no_key_client.conn.execute("SELECT COUNT(*) FROM headlines").fetchone()[0]
    assert count == 2


def test_store_headlines_empty_list_returns_zero(no_key_client):
    assert no_key_client.store_headlines([]) == 0


def test_store_headlines_persists_all_fields(no_key_client):
    no_key_client.store_headlines([_sample_headline("a")])
    row = no_key_client.conn.execute(
        "SELECT id, text, source, url, timestamp, query FROM headlines WHERE id='a'"
    ).fetchone()
    assert row == ("a", "hello world", "Reuters", "https://r/a", "2026-04-20T12:00:00Z", "bitcoin")


# ──────────────────────────── get_recent_headlines ────────────────────────────


def test_get_recent_headlines_returns_all_when_no_filter(no_key_client):
    no_key_client.store_headlines([_sample_headline("a"), _sample_headline("b")])
    result = no_key_client.get_recent_headlines()
    assert len(result) == 2
    assert isinstance(result[0], dict)
    assert set(result[0].keys()) >= {"id", "text", "source", "url", "timestamp", "query"}


def test_get_recent_headlines_filters_by_since_iso(no_key_client):
    old = _sample_headline("old", timestamp="2026-01-01T00:00:00Z")
    new = _sample_headline("new", timestamp="2026-06-01T00:00:00Z")
    no_key_client.store_headlines([old, new])
    result = no_key_client.get_recent_headlines(since_iso="2026-03-01T00:00:00Z")
    ids = [h["id"] for h in result]
    assert ids == ["new"]


def test_get_recent_headlines_empty_store_returns_empty(no_key_client):
    assert no_key_client.get_recent_headlines() == []


# ──────────────────────────── _extract_query ────────────────────────────


def test_extract_query_is_shorter_than_input():
    title = "Will BTC exceed $70k by April 20?"
    q = news_client._extract_query(title)
    assert len(q) < len(title)
    assert "?" not in q
    assert "will" not in q.lower()


def test_extract_query_crypto_btc_maps_to_bitcoin():
    q = news_client._extract_query("Will BTC exceed $70k?").lower()
    assert "bitcoin" in q or "btc" in q


def test_extract_query_weather_extracts_location_and_topic():
    q = news_client._extract_query("Will it rain in New York today?").lower()
    assert "new york" in q
    assert "rain" in q or "weather" in q


def test_extract_query_nonempty():
    assert news_client._extract_query("Will the Lakers win tonight?").strip() != ""
