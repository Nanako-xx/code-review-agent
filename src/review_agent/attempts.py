from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import stat
import uuid
from typing import Any, Mapping

from review_agent.checkpoint import _atomic_write_text, _fsync_parent_directory
from review_agent.run_state import RunPhase
from review_agent.session import SESSION_PHASES


@dataclass(frozen=True)
class AttemptWorkspace:
    """Isolated, auditable workspace for one phase or reviewer attempt."""

    run_dir: Path
    phase: RunPhase
    attempt: int
    reviewer_index: int | None = None

    def __post_init__(self) -> None:
        run_dir = Path(self.run_dir)
        if self.phase not in SESSION_PHASES:
            raise ValueError("phase must be one of the persisted SESSION_PHASES")
        if type(self.attempt) is not int or self.attempt < 1:
            raise ValueError("attempt must be a positive integer")
        if self.reviewer_index is not None:
            if self.phase is not RunPhase.REVIEWERS:
                raise ValueError("reviewer_index is valid only for the reviewers phase")
            if type(self.reviewer_index) is not int or self.reviewer_index < 0:
                raise ValueError("reviewer_index must be a non-negative integer")
        object.__setattr__(self, "run_dir", run_dir)

    @property
    def path(self) -> Path:
        path = self.run_dir / "attempts" / self.phase.value / str(self.attempt)
        if self.reviewer_index is not None:
            path /= f"reviewer-{self.reviewer_index}"
        return path

    def prepare(self) -> Path:
        self.path.mkdir(parents=True, exist_ok=True)
        return self.path

    def write_json(self, relative_path: str, payload: Mapping[str, Any]) -> Path:
        return self.write_text(
            relative_path,
            json.dumps(payload, indent=2, ensure_ascii=False),
        )

    def write_text(self, relative_path: str, content: str) -> Path:
        if not isinstance(content, str):
            raise ValueError("attempt artifact content must be text")
        destination = self._attempt_path(relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(destination, content)
        return destination

    def promote_file(
        self,
        source_relative_path: str,
        destination_relative_path: str,
    ) -> Path:
        """Atomically copy a completed attempt file into the authoritative run root."""

        source = self._regular_attempt_file(source_relative_path)
        destination = self._run_path(destination_relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.promote")
        try:
            with source.open("rb") as source_handle, staging.open("xb") as target_handle:
                shutil.copyfileobj(source_handle, target_handle)
                target_handle.flush()
                os.fsync(target_handle.fileno())
            os.replace(staging, destination)
            _fsync_parent_directory(destination.parent)
        finally:
            try:
                staging.unlink(missing_ok=True)
            except OSError:
                pass
        return destination

    def _attempt_path(self, relative_path: str) -> Path:
        relative = _canonical_relative_path(relative_path)
        root = self.path.resolve()
        candidate = self.path.joinpath(*PurePosixPath(relative).parts).resolve()
        _require_within(candidate, root, "attempt")
        return candidate

    def _run_path(self, relative_path: str) -> Path:
        relative = _canonical_relative_path(relative_path)
        if relative == "session.json" or relative.startswith("attempts/"):
            raise ValueError("attempt promotion destination is runtime-managed")
        root = self.run_dir.resolve()
        candidate = self.run_dir.joinpath(*PurePosixPath(relative).parts).resolve()
        _require_within(candidate, root, "Session run")
        return candidate

    def _regular_attempt_file(self, relative_path: str) -> Path:
        path = self._attempt_path(relative_path)
        try:
            metadata = path.lstat()
        except OSError as error:
            raise ValueError(f"attempt artifact does not exist: {relative_path}") from error
        if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
            raise ValueError(f"attempt artifact must be a regular file: {relative_path}")
        return path


def _canonical_relative_path(value: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("artifact path must be a non-empty canonical relative path")
    if "\\" in value:
        raise ValueError("artifact path must use forward-slash separators")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    parts = value.split("/")
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or posix.as_posix() != value
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError("artifact path must stay inside its managed directory")
    return value


def _require_within(candidate: Path, root: Path, label: str) -> None:
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"artifact path resolves outside the {label} directory") from error
