from pathlib import Path
import json

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
