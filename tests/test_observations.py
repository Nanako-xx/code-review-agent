from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from review_agent.observations import ObservationStore


def test_observation_store_records_stable_id_and_raw_artifact(tmp_path: Path):
    store = ObservationStore(tmp_path)

    first = store.record(
        source="git.read_range",
        revision="head@abc",
        path="src/auth.py",
        line_start=1,
        line_end=2,
        raw_content="def check():\n    return True\n",
        context_view="src/auth.py:1-2 changed",
    )
    second = store.record(
        source="git.read_range",
        revision="head@abc",
        path="src/auth.py",
        line_start=1,
        line_end=2,
        raw_content="def check():\n    return True\n",
        context_view="src/auth.py:1-2 changed",
    )

    assert first.observation_id == second.observation_id
    assert first.content_hash == second.content_hash
    assert (tmp_path / "observations" / f"{first.observation_id}.txt").read_text(encoding="utf-8") == (
        "def check():\n    return True\n"
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "observations.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert records[-1]["observation_id"] == first.observation_id
    assert records[-1]["raw_artifact_ref"] == f"observations/{first.observation_id}.txt"
    assert len(records) == 1


def test_observation_store_returns_summary_map(tmp_path: Path):
    store = ObservationStore(tmp_path)
    observation = store.record(
        source="git.compare_base_head",
        revision="base..head",
        path="auth.py",
        line_start=None,
        line_end=None,
        raw_content="diff --git a/auth.py b/auth.py",
        context_view="auth.py changed between base and head",
    )

    assert store.summaries_by_id() == {observation.observation_id: "auth.py changed between base and head"}
    assert store.list_observations()[0].observation_id == observation.observation_id


def _record(tmp_path: Path, *, revision: str = "base..head"):
    return ObservationStore(tmp_path).record(
        source="git.compare_base_head",
        revision=revision,
        path="auth.py",
        line_start=None,
        line_end=None,
        raw_content="diff --git a/auth.py b/auth.py",
        context_view="auth.py changed",
    )


def _legacy_observation_id(
    source: str,
    revision: str,
    path: str | None,
    line_start: int | None,
    line_end: int | None,
    content_hash: str,
) -> str:
    seed = "|".join(
        [
            source,
            revision,
            path or "",
            "" if line_start is None else str(line_start),
            "" if line_end is None else str(line_end),
            content_hash,
        ]
    )
    return f"O-{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:12]}"


def test_observation_store_load_validates_and_restores_authorized_records(tmp_path: Path):
    observation = _record(tmp_path)

    loaded = ObservationStore.load(tmp_path, {"base..head"})

    assert loaded.list_observations() == [observation]
    assert loaded.summaries_by_id() == {observation.observation_id: "auth.py changed"}
    assert loaded.record(
        source=observation.source,
        revision=observation.revision,
        path=observation.path,
        line_start=observation.line_start,
        line_end=observation.line_end,
        raw_content="diff --git a/auth.py b/auth.py",
        context_view=observation.context_view,
    ) == observation
    assert len((tmp_path / "observations.jsonl").read_text(encoding="utf-8").splitlines()) == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "bad_json",
        "bad_id",
        "bad_hash",
        "missing_raw",
        "unauthorized_revision",
        "path_traversal",
        "noncanonical_path",
    ],
)
def test_observation_store_load_rejects_untrusted_records(tmp_path: Path, mutation: str):
    observation = _record(tmp_path)
    jsonl = tmp_path / "observations.jsonl"
    payload = json.loads(jsonl.read_text(encoding="utf-8"))
    if mutation == "bad_json":
        jsonl.write_text("{", encoding="utf-8")
    elif mutation == "bad_id":
        payload["observation_id"] = "O-tampered"
        jsonl.write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "bad_hash":
        (tmp_path / observation.raw_artifact_ref).write_text("tampered", encoding="utf-8")
    elif mutation == "missing_raw":
        (tmp_path / observation.raw_artifact_ref).unlink()
    elif mutation == "unauthorized_revision":
        payload["revision"] = "other"
        jsonl.write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "path_traversal":
        payload["raw_artifact_ref"] = "observations/../outside.txt"
        jsonl.write_text(json.dumps(payload), encoding="utf-8")
    else:
        alias = tmp_path / "observations" / "alias.txt"
        alias.write_bytes((tmp_path / observation.raw_artifact_ref).read_bytes())
        payload["raw_artifact_ref"] = "observations/alias.txt"
        jsonl.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        ObservationStore.load(tmp_path, {"base..head"})


def test_observation_store_load_rejects_duplicate_id_with_different_context(tmp_path: Path):
    _record(tmp_path)
    jsonl = tmp_path / "observations.jsonl"
    payload = json.loads(jsonl.read_text(encoding="utf-8"))
    conflicting = {**payload, "context_view": "different summary"}
    jsonl.write_text(
        json.dumps(payload) + "\n" + json.dumps(conflicting) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate observation ID"):
        ObservationStore.load(tmp_path, {"base..head"})


def test_observation_store_constructor_requires_explicit_load_for_existing_log(tmp_path: Path):
    _record(tmp_path)

    with pytest.raises(ValueError, match="use ObservationStore.load"):
        ObservationStore(tmp_path)


def test_observation_store_load_requires_regular_observations_directory(tmp_path: Path):
    (tmp_path / "observations.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="observations directory"):
        ObservationStore.load(tmp_path, {"base..head"})


def test_observation_store_round_trips_crlf_bytes(tmp_path: Path):
    raw_content = "first\r\nsecond\r\n"
    observation = ObservationStore(tmp_path).record(
        source="git.read_range",
        revision="head@abc",
        path="src/app.py",
        line_start=1,
        line_end=2,
        raw_content=raw_content,
        context_view="src/app.py:1-2",
    )

    assert (tmp_path / observation.raw_artifact_ref).read_bytes() == raw_content.encode(
        "utf-8"
    )
    assert ObservationStore.load(tmp_path, {"head@abc"}).list_observations() == [
        observation
    ]


def test_observation_store_load_rejects_symlinked_raw_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    observation = _record(tmp_path)
    artifact = tmp_path / observation.raw_artifact_ref
    target = tmp_path / "request.json"
    raw_bytes = artifact.read_bytes()
    target.write_bytes(raw_bytes)
    artifact.unlink()
    try:
        artifact.symlink_to(target)
    except OSError:
        artifact.write_bytes(raw_bytes)
        original_is_symlink = Path.is_symlink

        def report_artifact_as_symlink(path: Path) -> bool:
            return path == artifact or original_is_symlink(path)

        monkeypatch.setattr(Path, "is_symlink", report_artifact_as_symlink)

    with pytest.raises(ValueError, match="not regular"):
        ObservationStore.load(tmp_path, {"base..head"})


def test_observation_store_loads_legacy_id_and_windows_newline_artifact(tmp_path: Path):
    source = "git.read_range"
    revision = "head@abc"
    path = "src/app.py"
    raw_content = "first\r\nsecond\r\n"
    content_hash = hashlib.sha256(raw_content.encode("utf-8")).hexdigest()
    observation_id = _legacy_observation_id(
        source,
        revision,
        path,
        1,
        2,
        content_hash,
    )
    artifact_ref = f"observations/{observation_id}.txt"
    (tmp_path / "observations").mkdir()
    (tmp_path / artifact_ref).write_bytes(
        raw_content.replace("\n", "\r\n").encode("utf-8")
    )
    payload = {
        "observation_id": observation_id,
        "source": source,
        "revision": revision,
        "path": path,
        "line_start": 1,
        "line_end": 2,
        "content_hash": content_hash,
        "raw_artifact_ref": artifact_ref,
        "context_view": "src/app.py:1-2",
    }
    (tmp_path / "observations.jsonl").write_text(
        json.dumps(payload) + "\n",
        encoding="utf-8",
    )

    loaded = ObservationStore.load(tmp_path, {revision})

    assert loaded.list_observations()[0].observation_id == observation_id
    assert loaded.record(
        source=source,
        revision=revision,
        path=path,
        line_start=1,
        line_end=2,
        raw_content=raw_content,
        context_view="src/app.py:1-2",
    ).observation_id == observation_id
    assert len(
        (tmp_path / "observations.jsonl").read_text(encoding="utf-8").splitlines()
    ) == 1


def test_observation_ids_distinguish_delimiters_and_null_from_empty_path(tmp_path: Path):
    store = ObservationStore(tmp_path)
    observations = [
        store.record("a|b", "c", None, None, None, "same", "first"),
        store.record("a", "b|c", None, None, None, "same", "second"),
        store.record("source", "revision", None, None, None, "same", "third"),
        store.record("source", "revision", "", None, None, "same", "fourth"),
    ]

    assert len({item.observation_id for item in observations}) == 4
    assert ObservationStore.load(
        tmp_path,
        {"c", "b|c", "revision"},
    ).list_observations() == observations
