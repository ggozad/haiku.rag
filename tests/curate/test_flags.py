from haiku.rag.client import HaikuRAG
from haiku.rag.curate.store.models import FlagKind, FlagStatus
from haiku.rag.curate.sweep import sweep
from tests.curate.test_sweep import (
    _import,
    _rewrite,
    _set_metadata,
    _writer_config,
    curate,  # noqa: F401
)

CLEAN = ["Alpha beta gamma delta.", "Epsilon zeta eta theta."]
GARBLED = ["Wkh#txlfn eurzq ir{.", "Mxps#ryhu wkh odc|."]


async def _flags(repository, kind: FlagKind):
    return await repository.flags(kind=kind)


async def test_bad_update_is_raised_then_superseded_by_a_fixed_upload(curate):  # noqa: F811
    paths, config, repository = curate
    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        doc_id = await _import(rag, "file:///wiki/a.pdf", CLEAN, [0, 1])
    await sweep(config, repository)
    assert await _flags(repository, FlagKind.BAD_UPDATE) == []

    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        await _rewrite(rag, doc_id, GARBLED, [5, 6])
    await sweep(config, repository)

    [flag] = await _flags(repository, FlagKind.BAD_UPDATE)
    assert flag.status is FlagStatus.OPEN
    assert flag.database == "wiki" and flag.subject == "file:///wiki/a.pdf"
    assert [r["metric"] for r in flag.reasons] == ["centroid_cosine"]

    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        await _rewrite(rag, doc_id, CLEAN, [0, 1])
    await sweep(config, repository)

    [flag] = await _flags(repository, FlagKind.BAD_UPDATE)
    assert flag.status is FlagStatus.SUPERSEDED


async def test_acknowledged_rewrite_becomes_the_baseline(curate):  # noqa: F811
    paths, config, repository = curate
    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        doc_id = await _import(rag, "file:///wiki/a.pdf", CLEAN, [0, 1])
    await sweep(config, repository)
    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        await _rewrite(rag, doc_id, GARBLED, [5, 6])
    await sweep(config, repository)
    [flag] = await _flags(repository, FlagKind.BAD_UPDATE)
    await repository.acknowledge(flag.id, "rewritten on purpose")

    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        await _rewrite(rag, doc_id, GARBLED[::-1], [6, 5])
    await sweep(config, repository)

    assert [f.status for f in await _flags(repository, FlagKind.BAD_UPDATE)] == [
        FlagStatus.ACKNOWLEDGED
    ]


async def test_missing_metadata_resolves_and_reopens(curate):  # noqa: F811
    paths, config, repository = curate
    config.curate.required_metadata = ["department"]
    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        doc_id = await _import(rag, "file:///wiki/a.pdf", CLEAN, [0, 1])
    await sweep(config, repository)
    [flag] = await _flags(repository, FlagKind.MISSING_METADATA)
    assert flag.status is FlagStatus.OPEN

    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        await _set_metadata(rag, doc_id, {"department": "ops"})
    await sweep(config, repository)
    [flag] = await _flags(repository, FlagKind.MISSING_METADATA)
    assert flag.status is FlagStatus.RESOLVED

    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        await _set_metadata(rag, doc_id, {})
    await sweep(config, repository)
    [reopened] = await _flags(repository, FlagKind.MISSING_METADATA)
    assert reopened.id == flag.id and reopened.status is FlagStatus.OPEN


async def test_acknowledged_flag_stays_acknowledged(curate):  # noqa: F811
    paths, config, repository = curate
    config.curate.required_metadata = ["department"]
    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        doc_id = await _import(rag, "file:///wiki/a.pdf", CLEAN, [0, 1])
    await sweep(config, repository)
    [flag] = await _flags(repository, FlagKind.MISSING_METADATA)
    await repository.acknowledge(flag.id, "not needed here")

    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        await _set_metadata(rag, doc_id, {"department": "ops"})
    await sweep(config, repository)
    await sweep(config, repository)

    [flag] = await _flags(repository, FlagKind.MISSING_METADATA)
    assert flag.status is FlagStatus.ACKNOWLEDGED
    assert flag.note == "not needed here"


async def test_detectors_run_on_an_unchanged_database(curate):  # noqa: F811
    paths, config, repository = curate
    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        await _import(rag, "file:///wiki/a.pdf", CLEAN, [0, 1])
    await sweep(config, repository)
    config.curate.required_metadata = ["department"]

    results = await sweep(config, repository)

    assert {r.database: r.status for r in results}["wiki"] == "unchanged"
    assert len(await _flags(repository, FlagKind.MISSING_METADATA)) == 1


async def test_watched_document_changes_and_deletion(curate):  # noqa: F811
    paths, config, repository = curate
    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        doc_id = await _import(rag, "file:///wiki/a.pdf", CLEAN, [0, 1])
    await sweep(config, repository)
    await repository.watch("wiki", "file:///wiki/a.pdf", "golden")
    await sweep(config, repository)
    assert await _flags(repository, FlagKind.WATCHED_CHANGE) == []

    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        await _rewrite(rag, doc_id, CLEAN[::-1], [1, 0])
    await sweep(config, repository)
    [changed] = await _flags(repository, FlagKind.WATCHED_CHANGE)
    assert changed.status is FlagStatus.OPEN and changed.reasons == []

    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        await rag.delete_document(doc_id)
    await sweep(config, repository)
    [changed] = await _flags(repository, FlagKind.WATCHED_CHANGE)
    [deleted] = await _flags(repository, FlagKind.WATCHED_DELETION)
    assert changed.status is FlagStatus.SUPERSEDED
    assert deleted.status is FlagStatus.OPEN
    assert deleted.fingerprint_id == changed.fingerprint_id

    await repository.unwatch("wiki", "file:///wiki/a.pdf")
    await sweep(config, repository)
    [deleted] = await _flags(repository, FlagKind.WATCHED_DELETION)
    assert deleted.status is FlagStatus.RESOLVED


async def test_watched_deletion_is_superseded_when_the_document_returns(curate):  # noqa: F811
    paths, config, repository = curate
    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        doc_id = await _import(rag, "file:///wiki/a.pdf", CLEAN, [0, 1])
    await sweep(config, repository)
    await repository.watch("wiki", "file:///wiki/a.pdf")
    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        await rag.delete_document(doc_id)
    await sweep(config, repository)
    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        await _import(rag, "file:///wiki/a.pdf", CLEAN, [0, 1])
    await sweep(config, repository)

    [deleted] = await _flags(repository, FlagKind.WATCHED_DELETION)
    [returned] = await _flags(repository, FlagKind.WATCHED_CHANGE)
    assert deleted.status is FlagStatus.SUPERSEDED
    assert returned.status is FlagStatus.OPEN


async def test_repeated_chunk_text_across_documents(curate):  # noqa: F811
    paths, config, repository = curate
    footer = "Confidential. Do not distribute outside the organisation."
    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        for n in range(5):
            await _import(rag, f"file:///wiki/{n}.pdf", [f"Body {n}.", footer], [n, 7])
    await sweep(config, repository)

    [flag] = await _flags(repository, FlagKind.REPEATED_CHUNK)
    assert flag.members is not None and len(flag.members) == 5


async def test_near_duplicate_documents_in_one_database(curate):  # noqa: F811
    paths, config, repository = curate
    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        for uri in ("file:///wiki/a.pdf", "file:///wiki/copy-of-a.pdf"):
            await _import(rag, uri, ["One.", "Two.", "Three."], [0, 1, 2])
    await sweep(config, repository)

    [flag] = await _flags(repository, FlagKind.DUPLICATE_GROUP)
    assert flag.database == "wiki"
    assert flag.members is not None and len(flag.members) == 2


async def test_exact_copies_across_databases_and_their_resolution(curate):  # noqa: F811
    paths, config, repository = curate
    ids = {}
    for name in ("wiki", "papers"):
        async with HaikuRAG(paths[name], _writer_config()) as rag:
            ids[name] = await _import(rag, f"file:///{name}/a.pdf", CLEAN, [0, 1])
            await _set_metadata(rag, ids[name], {"md5": "same-bytes"})
    await sweep(config, repository)

    [flag] = await _flags(repository, FlagKind.DUPLICATE_GROUP)
    assert flag.database is None
    assert flag.members == [
        {"database": "papers", "document_id": ids["papers"]},
        {"database": "wiki", "document_id": ids["wiki"]},
    ]

    async with HaikuRAG(paths["papers"], _writer_config()) as rag:
        await rag.delete_document(ids["papers"])
    await sweep(config, repository)

    [flag] = await _flags(repository, FlagKind.DUPLICATE_GROUP)
    assert flag.status is FlagStatus.RESOLVED


async def test_isolation_is_written_when_a_database_changes(curate):  # noqa: F811
    paths, config, repository = curate
    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        a = await _import(rag, "file:///wiki/a.pdf", CLEAN, [0, 1])
        b = await _import(rag, "file:///wiki/b.pdf", CLEAN, [2, 3])
    await sweep(config, repository)

    scores = await repository.isolation("wiki")

    assert scores[a] == scores[b] == 1.0
    assert await repository.isolation("papers") == {}


async def test_editing_a_watch_note_keeps_its_open_flag(curate):  # noqa: F811
    paths, config, repository = curate
    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        doc_id = await _import(rag, "file:///wiki/a.pdf", CLEAN, [0, 1])
    await sweep(config, repository)
    await repository.watch("wiki", "file:///wiki/a.pdf", "golden")
    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        await _rewrite(rag, doc_id, CLEAN[::-1], [1, 0])
    await sweep(config, repository)

    await repository.watch("wiki", "file:///wiki/a.pdf", "golden, owned by ops")
    await sweep(config, repository)

    [changed] = await _flags(repository, FlagKind.WATCHED_CHANGE)
    assert changed.status is FlagStatus.OPEN


async def test_repeated_chunk_length_ignores_whitespace(curate):  # noqa: F811
    paths, config, repository = curate
    short = "Page 1 of 10 total."
    assert len(short) < config.curate.repeated_chunks.min_chars
    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        for n in range(5):
            padded = short.replace(" ", "   ")
            await _import(rag, f"file:///wiki/{n}.pdf", [f"Body {n}.", padded], [n, 7])
    await sweep(config, repository)

    assert await _flags(repository, FlagKind.REPEATED_CHUNK) == []
