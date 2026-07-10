from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
import errno
import json
import os
import uuid

from review_agent.run_state import RunState, run_state_from_dict, run_state_to_dict


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
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_parent_directory(path.parent)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _fsync_parent_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    unsupported_errors = {
        errno.EACCES,
        errno.EBADF,
        errno.EINVAL,
        errno.ENOSYS,
        errno.EPERM,
        getattr(errno, "ENOTSUP", errno.EINVAL),
        getattr(errno, "EOPNOTSUPP", errno.EINVAL),
    }
    try:
        directory_descriptor = os.open(directory, flags)
    except OSError as error:
        if error.errno in unsupported_errors:
            return
        raise
    try:
        try:
            os.fsync(directory_descriptor)
        except OSError as error:
            if error.errno not in unsupported_errors:
                raise
    finally:
        os.close(directory_descriptor)
