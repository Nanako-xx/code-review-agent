from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
import json
import os

from review_agent.run_state import RunState, run_state_from_dict, run_state_to_dict
from review_agent.safe_io import atomic_write_text, fsync_parent_directory


class CheckpointStore:
    def __init__(self, repository_path: Path, review_id: str, *, create: bool = True) -> None:
        self.repository_path = Path(repository_path)
        self.review_id = review_id
        self.run_dir = self.repository_path / ".review-agent" / "runs" / review_id
        if create:
            self.run_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, filename: str, payload: dict[str, object]) -> Path:
        path = self.run_dir / filename
        content = json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        )
        _atomic_write_text(path, content)
        return path

    def read_json(self, filename: str) -> dict[str, object]:
        return json.loads((self.run_dir / filename).read_text(encoding="utf-8"))

    def write_state(self, state: RunState) -> Path:
        return self.write_json("state.json", run_state_to_dict(state))

    def read_state(self) -> RunState:
        return run_state_from_dict(self.read_json("state.json"))

    def append_jsonl(self, filename: str, payload: dict[str, object]) -> Path:
        path = self.run_dir / filename
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=_json_default))
            handle.write("\n")
        return path


def _json_default(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object is not JSON serializable: {type(value).__name__}")


def _atomic_write_text(path: Path, content: str) -> None:
    atomic_write_text(
        path,
        content,
        os_module=os,
        allow_legacy_extended_path=True,
    )


def _fsync_parent_directory(directory: Path) -> None:
    fsync_parent_directory(directory, os_module=os)
