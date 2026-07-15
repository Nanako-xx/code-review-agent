from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from review_agent.memory_models import (
    MemoryMode,
    RepositoryKnowledgeCapability,
)
from review_agent.memory_store import MemoryStore
from review_agent.repository_cache import (
    CAPABILITY_METADATA,
    RepositoryCacheConflictError,
    RepositoryCacheStatus,
    RepositoryKnowledgeArtifact,
    RepositoryKnowledgeCache,
    build_repository_knowledge_key,
    digest_repository_configuration,
    digest_repository_input,
    validate_repository_knowledge_manifest,
)


REPOSITORY_KEY = "a" * 64
SHA_A = "1" * 40
SHA_B = "2" * 40
CREATED_AT = "2026-07-14T00:00:00Z"


def _key(
    *,
    repository_key: str = REPOSITORY_KEY,
    revision_binding: str = "head@" + SHA_A,
    capability: RepositoryKnowledgeCapability = RepositoryKnowledgeCapability.SYMBOL_INDEX,
    analyzer_name: str = "python-ast",
    analyzer_version: str = "3.12-v1",
    configuration: object = None,
    inputs: object = None,
):
    return build_repository_knowledge_key(
        repository_key=repository_key,
        revision_binding=revision_binding,
        capability=capability,
        analyzer_name=analyzer_name,
        analyzer_version=analyzer_version,
        configuration={"lsp_status": "unavailable"} if configuration is None else configuration,
        inputs={"paths": ["app.py"]} if inputs is None else inputs,
    )


def _artifact(content: bytes = b'{"symbols":[]}') -> RepositoryKnowledgeArtifact:
    return RepositoryKnowledgeArtifact(
        content=content,
        content_type="application/vnd.review-agent.symbol-index+json",
        artifact_schema="symbol_index_v1",
        summary_hash=hashlib.sha256(b"no symbols").hexdigest(),
    )


def _cache(store, mode: MemoryMode) -> RepositoryKnowledgeCache:
    return RepositoryKnowledgeCache(
        store,
        mode=mode,
        clock=lambda: CREATED_AT,
    )


def test_key_builder_binds_every_exact_cache_dimension() -> None:
    baseline = _key()
    variants = (
        _key(repository_key="b" * 64),
        _key(revision_binding="head@" + SHA_B),
        _key(capability=RepositoryKnowledgeCapability.FILE_INDEX),
        _key(analyzer_name="ripgrep"),
        _key(analyzer_version="3.12-v2"),
        _key(configuration={"lsp_status": "available"}),
        _key(inputs={"paths": ["other.py"]}),
    )

    assert len({baseline.key_hash, *(item.key_hash for item in variants)}) == 8
    assert baseline.configuration_digest == digest_repository_configuration(
        {"lsp_status": "unavailable"}
    )
    assert baseline.input_digest == digest_repository_input({"paths": ["app.py"]})
    assert baseline.to_dict() == {
        "schema_version": 1,
        "key_hash": baseline.key_hash,
        "repository_key": REPOSITORY_KEY,
        "revision_binding": "head@" + SHA_A,
        "capability": "symbol_index",
        "analyzer_name": "python-ast",
        "analyzer_version": "3.12-v1",
        "configuration_digest": baseline.configuration_digest,
        "input_digest": baseline.input_digest,
    }


def test_capability_metadata_covers_the_final_cache_taxonomy() -> None:
    assert set(CAPABILITY_METADATA) == set(RepositoryKnowledgeCapability)
    for capability, metadata in CAPABILITY_METADATA.items():
        assert metadata.capability is capability
        assert metadata.artifact_schema == capability.value + "_v1"
        assert metadata.content_type.startswith("application/vnd.review-agent.")


def test_exact_hit_and_same_content_blob_reuse_keep_distinct_revision_manifests(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "memory")
    cache = _cache(store, MemoryMode.READ_WRITE)
    first_key = _key(revision_binding="head@" + SHA_A)
    second_key = _key(revision_binding="head@" + SHA_B)
    build_count = 0

    def build() -> RepositoryKnowledgeArtifact:
        nonlocal build_count
        build_count += 1
        return _artifact()

    first = cache.get_or_build(first_key, build, review_id="review-001")
    exact_hit = cache.get_or_build(
        first_key,
        lambda: pytest.fail("an exact cache hit must not rebuild"),
        review_id="review-002",
    )
    second_revision = cache.get_or_build(second_key, build, review_id="review-003")

    assert build_count == 2
    assert first.provenance.status is RepositoryCacheStatus.MISS
    assert exact_hit.provenance.status is RepositoryCacheStatus.HIT
    assert second_revision.provenance.status is RepositoryCacheStatus.MISS
    assert first.content == exact_hit.content == second_revision.content
    assert first.entry is not None and exact_hit.entry is not None
    assert second_revision.entry is not None
    assert first.entry.entry_id == exact_hit.entry.entry_id
    assert first.entry.entry_id != second_revision.entry.entry_id
    assert first.entry.blob_hash == second_revision.entry.blob_hash
    assert first.entry.key.revision_binding == "head@" + SHA_A
    assert second_revision.entry.key.revision_binding == "head@" + SHA_B
    assert len(store.list_knowledge_entries(REPOSITORY_KEY)) == 2
    validate_repository_knowledge_manifest(first.entry, first_key, first.content)


def test_off_read_and_read_write_modes_have_exact_cross_run_behavior(
    tmp_path: Path,
) -> None:
    class ExplodingStore:
        def __getattribute__(self, name):
            raise AssertionError("off mode touched the cross-run store")

    off = _cache(ExplodingStore(), MemoryMode.OFF)
    off_result = off.get_or_build(_key(), _artifact, review_id="review-off")
    assert off_result.provenance.status is RepositoryCacheStatus.OFF
    assert off_result.provenance.persistent is False
    assert off_result.entry is None

    store = MemoryStore(tmp_path / "memory")
    writer = _cache(store, MemoryMode.READ_WRITE)
    hit_key = _key()
    persisted = writer.get_or_build(hit_key, _artifact)
    assert persisted.entry is not None

    reader = _cache(store, MemoryMode.READ)
    hit = reader.get_or_build(
        hit_key,
        lambda: pytest.fail("read mode must use an exact hit"),
        review_id="review-read-hit",
    )
    miss_key = _key(inputs={"paths": ["missing.py"]})
    miss = reader.get_or_build(miss_key, _artifact, review_id="review-read-miss")

    assert hit.provenance.status is RepositoryCacheStatus.HIT
    assert hit.provenance.persistent is True
    assert hit.provenance.session_pinned is True
    assert miss.provenance.status is RepositoryCacheStatus.MISS
    assert miss.provenance.persistent is False
    assert miss.entry is None
    assert store.find_knowledge_by_key(miss_key) is None

    written = writer.get_or_build(miss_key, _artifact)
    assert written.provenance.persistent is True
    assert store.find_knowledge_by_key(miss_key) == written.entry


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_missing_or_corrupt_blob_is_rebuilt_and_never_returned_as_stale(
    tmp_path: Path,
    damage: str,
) -> None:
    store = MemoryStore(tmp_path / "memory")
    cache = _cache(store, MemoryMode.READ_WRITE)
    key = _key()
    original = cache.get_or_build(key, _artifact)
    assert original.entry is not None
    blob_path = store.blob_path(original.entry.blob_hash)
    if damage == "missing":
        blob_path.unlink()
    else:
        blob_path.write_bytes(b"corrupt cache bytes")

    builds = 0

    def rebuild() -> RepositoryKnowledgeArtifact:
        nonlocal builds
        builds += 1
        return _artifact()

    rebuilt = cache.get_or_build(key, rebuild, review_id="review-rebuild")
    hit = cache.get_or_build(
        key,
        lambda: pytest.fail("a repaired cache entry must hit"),
    )

    assert builds == 1
    assert rebuilt.provenance.status is RepositoryCacheStatus.REBUILD
    assert rebuilt.content == _artifact().content
    assert rebuilt.provenance.corruption_reason == "corruption"
    assert rebuilt.provenance.persistent is True
    assert hit.provenance.status is RepositoryCacheStatus.HIT
    assert store.read_blob(original.entry.blob_hash) == _artifact().content


def test_read_mode_corruption_rebuild_is_session_only(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory")
    key = _key()
    persisted = _cache(store, MemoryMode.READ_WRITE).get_or_build(key, _artifact)
    assert persisted.entry is not None
    store.blob_path(persisted.entry.blob_hash).unlink()

    rebuilt = _cache(store, MemoryMode.READ).get_or_build(key, _artifact)

    assert rebuilt.provenance.status is RepositoryCacheStatus.REBUILD
    assert rebuilt.provenance.persistent is False
    assert rebuilt.entry is None
    assert not store.blob_path(persisted.entry.blob_hash).exists()


def test_immutable_exact_key_rejects_different_content(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory")
    cache = _cache(store, MemoryMode.READ_WRITE)
    key = _key()
    first = cache.write(key, _artifact(b'{"symbols":[]}'))
    assert first.entry is not None

    with pytest.raises(RepositoryCacheConflictError, match="immutable"):
        cache.write(key, _artifact(b'{"symbols":["changed"]}'))

    assert store.read_blob(first.entry.blob_hash) == b'{"symbols":[]}'


def test_hash_valid_but_invalid_artifact_is_replaced_by_deterministic_rebuild(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "memory")
    cache = _cache(store, MemoryMode.READ_WRITE)
    key = _key()
    invalid = cache.write(key, _artifact(b"not-json"))
    assert invalid.entry is not None

    def require_symbol_json(content: bytes) -> None:
        payload = json.loads(content.decode("utf-8"))
        if set(payload) != {"symbols"} or not isinstance(payload["symbols"], list):
            raise ValueError("invalid symbol payload")

    rebuilt = cache.get_or_build(
        key,
        _artifact,
        validator=require_symbol_json,
        review_id="review-valid-rebuild",
    )

    assert rebuilt.provenance.status is RepositoryCacheStatus.REBUILD
    assert rebuilt.provenance.persistent is True
    assert rebuilt.entry is not None
    assert rebuilt.entry.entry_id != invalid.entry.entry_id
    assert store.list_knowledge_entries(REPOSITORY_KEY) == (rebuilt.entry,)


def test_session_pin_prevents_cache_entry_and_blob_gc(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory")
    cache = _cache(store, MemoryMode.READ_WRITE)
    result = cache.get_or_build(_key(), _artifact, review_id="review-pinned")
    assert result.entry is not None

    retained = cache.gc_unpinned_entries(REPOSITORY_KEY, dry_run=False)
    assert retained.deleted_entry_ids == ()
    assert retained.retained_pinned_entry_ids == (result.entry.entry_id,)
    assert store.get_knowledge_entry(result.entry.entry_id).pinned_by_review_ids == (
        "review-pinned",
    )
    assert store.blob_path(result.entry.blob_hash).is_file()

    assert cache.unpin(result.entry.entry_id, "review-pinned") is True
    collected = cache.gc_unpinned_entries(REPOSITORY_KEY, dry_run=False)
    assert collected.deleted_entry_ids == (result.entry.entry_id,)
    assert collected.deleted_blob_hashes == (result.entry.blob_hash,)
    assert store.find_knowledge_entry(result.entry.entry_id) is None
    assert not store.blob_path(result.entry.blob_hash).exists()

    # Recreating the same immutable identity after collection is a new
    # generation-scoped write, not a replay of the original receipt.
    recreated = cache.get_or_build(_key(), _artifact)
    assert recreated.provenance.status is RepositoryCacheStatus.MISS
    assert recreated.provenance.persistent is True
    assert recreated.entry is not None
    assert recreated.entry.entry_id == result.entry.entry_id
    assert store.get_knowledge_entry(recreated.entry.entry_id) == recreated.entry


def test_provenance_records_key_analyzer_fallback_and_persistence(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory")
    cache = _cache(store, MemoryMode.READ_WRITE)
    key = _key()
    result = cache.get_or_build(
        key,
        _artifact,
        fallback_provenance={
            "lsp_status": "unavailable",
            "strategy": "python_ast+git_grep",
        },
    )

    payload = result.provenance.to_dict()
    assert payload["status"] == "miss"
    assert payload["key_hash"] == key.key_hash
    assert payload["entry_id"] == result.entry.entry_id
    assert payload["blob_hash"] == result.entry.blob_hash
    assert payload["analyzer"] == {"name": "python-ast", "version": "3.12-v1"}
    assert payload["fallback"] == {
        "lsp_status": "unavailable",
        "strategy": "python_ast+git_grep",
    }
    assert payload["persistent"] is True
