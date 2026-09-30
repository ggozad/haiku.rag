import shutil
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from httpx import ASGITransport

from haiku.rag.client import HaikuRAG
from haiku.rag.curate.api.server import APIState, build_app
from haiku.rag.curate.sweep import sweep
from tests.curate.test_flags import CLEAN, GARBLED
from tests.curate.test_sweep import (
    _import,
    _rewrite,
    _set_metadata,
    _writer_config,
    curate,  # noqa: F401
)


def _client(config, repository, auth_token=None) -> httpx.AsyncClient:
    app = build_app(APIState(config, repository), auth_token=auth_token)
    return httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    )


@pytest.fixture
async def populated(curate):  # noqa: F811
    """wiki holds a garbled update and a repeated footer; papers is empty."""
    paths, config, repository = curate
    footer = "Confidential. Do not distribute outside the organisation."
    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        doc_id = await _import(rag, "file:///wiki/a.pdf", CLEAN, [0, 1])
        for n in range(5):
            await _import(rag, f"file:///wiki/{n}.pdf", [f"Body {n}.", footer], [2, 3])
    await sweep(config, repository)
    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        await _rewrite(rag, doc_id, GARBLED, [5, 6])
    await sweep(config, repository)
    return config, repository, doc_id, footer


async def test_health_is_open_without_a_token(populated):
    config, repository, _, _ = populated
    async with _client(config, repository, auth_token="secret") as client:
        health = await client.get("/health")
        databases = await client.get("/databases")
    assert health.status_code == 200
    assert {d["database"] for d in health.json()["databases"]} == {"wiki", "papers"}
    assert databases.status_code == 401


async def test_databases_summarise_each_database(populated):
    config, repository, _, _ = populated
    async with _client(config, repository, auth_token="secret") as client:
        response = await client.get(
            "/databases", headers={"Authorization": "Bearer secret"}
        )
    wiki, papers = response.json()
    assert wiki["database"] == "wiki" and wiki["documents"] == 6
    assert wiki["last_status"] == "ok" and wiki["open_flags"] >= 2
    assert papers["documents"] == 0 and papers["last_status"] == "unchanged"


async def test_flags_filter_and_acknowledge(populated):
    config, repository, doc_id, _ = populated
    async with _client(config, repository) as client:
        [bad] = (await client.get("/flags", params={"kind": "bad_update"})).json()
        assert bad["subject"] == "file:///wiki/a.pdf"
        assert bad["document_id"] == doc_id
        [repeated] = (
            await client.get("/flags", params={"kind": "repeated_chunk"})
        ).json()
        assert repeated["document_id"] is None
        assert sorted(m["uri"] for m in repeated["members"]) == [
            f"file:///wiki/{n}.pdf" for n in range(5)
        ]
        assert all("title" in m for m in repeated["members"])
        assert (await client.get("/flags", params={"database": "papers"})).json() == []

        acknowledged = await client.post(
            f"/flags/{bad['id']}/acknowledge", json={"note": "known"}
        )
        missing = await client.post("/flags/999/acknowledge", json={})
        still_open = await client.get(
            "/flags", params={"kind": "bad_update", "status": "open"}
        )
    assert acknowledged.json()["status"] == "acknowledged"
    assert acknowledged.json()["note"] == "known"
    assert missing.status_code == 404
    assert still_open.json() == []


async def test_reopen_an_acknowledged_flag(populated):
    config, repository, _, _ = populated
    async with _client(config, repository) as client:
        [bad] = (await client.get("/flags", params={"kind": "bad_update"})).json()
        not_acknowledged = await client.post(f"/flags/{bad['id']}/reopen")
        await client.post(f"/flags/{bad['id']}/acknowledge", json={"note": "known"})
        reopened = await client.post(f"/flags/{bad['id']}/reopen")
        missing = await client.post("/flags/999/reopen")
    assert not_acknowledged.status_code == 409
    assert reopened.status_code == 200
    assert reopened.json()["status"] == "open"
    assert reopened.json()["note"] is None
    assert missing.status_code == 404


async def test_repeated_chunk_text_is_read_from_the_database(populated):
    config, repository, _, footer = populated
    async with _client(config, repository) as client:
        [repeated] = (
            await client.get("/flags", params={"kind": "repeated_chunk"})
        ).json()
        text = await client.get(f"/flags/{repeated['id']}/text")
        [bad] = (await client.get("/flags", params={"kind": "bad_update"})).json()
        not_text = await client.get(f"/flags/{bad['id']}/text")
        missing = await client.get("/flags/999/text")
    assert text.json() == {"text": footer}
    assert not_text.status_code == 404
    assert missing.status_code == 404


async def test_changes_since(populated):
    config, repository, doc_id, _ = populated
    async with _client(config, repository) as client:
        everything = (await client.get("/changes")).json()
        future = await client.get("/changes", params={"since": "9999-01-01T00:00:00Z"})
        papers = await client.get("/changes", params={"database": "papers"})
    kinds = [(c["kind"], c["document_id"] == doc_id) for c in everything]
    assert kinds.count(("added", False)) == 5
    assert ("added", True) in kinds and ("updated", True) in kinds
    assert future.json() == [] and papers.json() == []


async def test_deletions_appear_in_changes(populated, curate):  # noqa: F811
    paths, config, repository = curate
    _, _, doc_id, _ = populated
    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        await rag.delete_document(doc_id)
    await sweep(config, repository)
    async with _client(config, repository) as client:
        changes = (await client.get("/changes")).json()
    assert changes[-1]["kind"] == "deleted" and changes[-1]["document_id"] == doc_id


async def test_documents_and_history(populated):
    config, repository, doc_id, _ = populated
    async with _client(config, repository) as client:
        documents = (await client.get("/documents", params={"database": "wiki"})).json()
        history = await client.get(f"/documents/wiki/{doc_id}/history")
        empty = await client.get("/documents/wiki/nope/history")
    by_id = {d["document_id"]: d for d in documents}
    assert len(by_id) == 6
    assert by_id[doc_id]["open_flags"] == ["bad_update"]
    assert set(by_id[doc_id]) >= {"isolation", "chunk_stats", "replacement_chars"}
    footer_docs = [d for d in documents if d["document_id"] != doc_id]
    assert all("repeated_chunk" in d["open_flags"] for d in footer_docs)
    first, second = history.json()
    assert "centroid" not in first
    assert first["ended_sweep"] == second["became_current_sweep"]
    assert first["ended_at"] == second["became_current_at"]
    assert first["became_current_at"] < first["ended_at"]
    assert second["ended_at"] is None
    assert empty.status_code == 404


async def test_watch_list(populated):
    config, repository, _, _ = populated
    async with _client(config, repository) as client:
        added = await client.post(
            "/watched", json={"database": "wiki", "uri": "file:///wiki/a.pdf"}
        )
        renoted = await client.post(
            "/watched",
            json={"database": "wiki", "uri": "file:///wiki/a.pdf", "note": "golden"},
        )
        rewatched = await client.post(
            "/watched", json={"database": "wiki", "uri": "file:///wiki/a.pdf"}
        )
        listed = (await client.get("/watched")).json()
        removed = await client.delete(
            "/watched", params={"database": "wiki", "uri": "file:///wiki/a.pdf"}
        )
        again = await client.delete(
            "/watched", params={"database": "wiki", "uri": "file:///wiki/a.pdf"}
        )
    assert added.status_code == renoted.status_code == rewatched.status_code == 201
    assert [(w["uri"], w["note"]) for w in listed] == [("file:///wiki/a.pdf", "golden")]
    assert removed.status_code == 204 and again.status_code == 404


async def test_unknown_database_is_not_found(populated):
    config, repository, _, _ = populated
    async with _client(config, repository) as client:
        responses = [
            await client.get("/documents", params={"database": "nope"}),
            await client.get("/flags", params={"database": "nope"}),
            await client.get("/changes", params={"database": "nope"}),
            await client.get("/documents/nope/x/history"),
            await client.post("/watched", json={"database": "nope", "uri": "u"}),
        ]
    assert [r.status_code for r in responses] == [404] * 5
    assert "unknown database 'nope'" in responses[0].json()["detail"]


async def test_metadata_only_changes_are_not_changes(populated, curate):  # noqa: F811
    paths, config, repository = curate
    _, _, doc_id, _ = populated
    async with _client(config, repository) as client:
        before = (await client.get("/changes")).json()
    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        await _set_metadata(rag, doc_id, {"department": "ops"})
    await sweep(config, repository)
    async with _client(config, repository) as client:
        after = (await client.get("/changes")).json()
    assert after == before


async def test_repeated_text_gone_since_the_sweep(populated, curate):  # noqa: F811
    paths, _, _ = curate
    config, repository, _, _ = populated
    async with _client(config, repository) as client:
        [repeated] = (
            await client.get("/flags", params={"kind": "repeated_chunk"})
        ).json()
    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        for member in repeated["members"]:
            await rag.delete_document(member["document_id"])
    async with _client(config, repository) as client:
        response = await client.get(f"/flags/{repeated['id']}/text")
    assert response.status_code == 404
    assert response.json()["detail"] == "the text is no longer present"


async def test_repeated_text_of_an_unavailable_database(populated, curate):  # noqa: F811
    paths, _, _ = curate
    config, repository, _, _ = populated
    shutil.rmtree(paths["wiki"])
    async with _client(config, repository) as client:
        [repeated] = (
            await client.get("/flags", params={"kind": "repeated_chunk"})
        ).json()
        response = await client.get(f"/flags/{repeated['id']}/text")
    assert response.status_code == 503
    assert response.json()["detail"].startswith("database 'wiki' does not exist")


async def test_changes_since_compares_instants_not_strings(populated):
    config, repository, _, _ = populated
    async with _client(config, repository) as client:
        everything = (await client.get("/changes")).json()
        first = everything[0]["at"]
        # The same instant written in another offset still includes it.
        shifted = datetime.fromisoformat(first).astimezone(timezone(timedelta(hours=3)))
        same = (
            await client.get("/changes", params={"since": shifted.isoformat()})
        ).json()
        naive = (
            await client.get(
                "/changes",
                params={
                    "since": datetime.fromisoformat(first)
                    .replace(tzinfo=None)
                    .isoformat()
                },
            )
        ).json()
        malformed = await client.get("/changes", params={"since": "yesterday"})
    assert same == everything
    assert naive == everything
    assert malformed.status_code == 422


async def test_records_outside_the_curated_databases_are_hidden(populated):
    config, repository, _, _ = populated
    await repository.watch("wiki", "file:///wiki/a.pdf")
    narrowed = config.model_copy(
        update={"curate": config.curate.model_copy(update={"databases": ["papers"]})}
    )
    async with _client(config, repository) as client:
        [wiki_flag, *_] = (await client.get("/flags")).json()
    async with _client(narrowed, repository) as client:
        flags = (await client.get("/flags")).json()
        watched = (await client.get("/watched")).json()
        acknowledge = await client.post(
            f"/flags/{wiki_flag['id']}/acknowledge", json={}
        )
        text = await client.get(f"/flags/{wiki_flag['id']}/text")
        unwatch = await client.delete(
            "/watched", params={"database": "wiki", "uri": "file:///wiki/a.pdf"}
        )
    assert flags == [] and watched == []
    assert acknowledge.status_code == 404 and text.status_code == 404
    assert unwatch.status_code == 404
    [still] = [f for f in await repository.flags() if f.id == wiki_flag["id"]]
    assert still.status.value == "open"


async def test_cross_database_flags_need_every_member_in_scope(curate):  # noqa: F811
    paths, config, repository = curate
    for name in ("wiki", "papers"):
        async with HaikuRAG(paths[name], _writer_config()) as rag:
            doc = await _import(rag, f"file:///{name}/a.pdf", CLEAN, [0, 1])
            await _set_metadata(rag, doc, {"md5": "same-bytes"})
    await sweep(config, repository)
    narrowed = config.model_copy(
        update={"curate": config.curate.model_copy(update={"databases": ["wiki"]})}
    )
    async with _client(config, repository) as client:
        everywhere = (
            await client.get("/flags", params={"kind": "duplicate_group"})
        ).json()
    async with _client(narrowed, repository) as client:
        narrowly = (
            await client.get("/flags", params={"kind": "duplicate_group"})
        ).json()
    assert [f["database"] for f in everywhere] == [None]
    assert narrowly == []


async def test_database_filters_include_cross_database_groups(curate):  # noqa: F811
    paths, config, repository = curate
    for name in ("wiki", "papers"):
        async with HaikuRAG(paths[name], _writer_config()) as rag:
            doc = await _import(rag, f"file:///{name}/a.pdf", CLEAN, [0, 1])
            await _set_metadata(rag, doc, {"md5": "same-bytes"})
    await sweep(config, repository)
    async with _client(config, repository) as client:
        wiki = (await client.get("/flags", params={"database": "wiki"})).json()
        papers = (await client.get("/flags", params={"database": "papers"})).json()
        summaries = (await client.get("/databases")).json()
    assert [f["database"] for f in wiki] == [None]
    assert [f["id"] for f in papers] == [f["id"] for f in wiki]
    assert [s["open_flags"] for s in summaries] == [1, 1]
