from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
import json


class CheckpointStore:
    def __init__(self, repository_path: Path, review_id: str) -> None:
        self.run_dir = repository_path / ".review-agent" / "runs" / review_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, filename: str, payload: dict[str, object]) -> Path:
        path = self.run_dir / filename
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
        return path

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
