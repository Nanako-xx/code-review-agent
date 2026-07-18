from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from review_agent_eval.models import TraceRef, TraceType
from review_agent_eval import runner as runner_module


def _capture(
    workspace: Path,
    trace_value: str,
    *,
    max_trace_bytes: int = 1024 * 1024,
) -> Dict[str, Any]:
    submission = SimpleNamespace(
        trace_ref=TraceRef(type=TraceType.LOCAL_PATH, value=trace_value)
    )
    result = runner_module._capture_trace_summary(
        submission,
        workspace,
        max_trace_bytes=max_trace_bytes,
    )
    assert result is not None
    return result


def _assert_fail_closed(result: Dict[str, Any], secret: bytes) -> None:
    assert result["captured"] is False
    assert result["files"] == []
    assert result["total_bytes"] is None
    encoded = base64.b64encode(secret).decode("ascii")
    assert encoded not in str(result)


def test_trace_capture_rejects_hard_link_to_file_outside_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    trace_dir = workspace / "trace"
    trace_dir.mkdir(parents=True)
    secret = b"outside-hard-link-secret"
    outside = tmp_path / "outside-secret.txt"
    outside.write_bytes(secret)
    try:
        os.link(outside, trace_dir / "linked-secret.txt")
    except OSError as exc:
        pytest.skip(f"hard links are unavailable on this filesystem: {exc}")

    result = _capture(workspace, "trace")

    _assert_fail_closed(result, secret)


def test_trace_capture_rejects_symlink_or_reparse_point(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = b"outside-link-secret"
    (outside / "secret.txt").write_bytes(secret)
    trace_link = workspace / "trace"
    try:
        trace_link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory links are unavailable on this platform: {exc}")

    result = _capture(workspace, "trace")

    _assert_fail_closed(result, secret)


def test_trace_capture_rejects_parent_replacement_during_enumeration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    trace_dir = workspace / "trace"
    trace_dir.mkdir(parents=True)
    (trace_dir / "original.txt").write_bytes(b"original")
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    secret = b"replacement-secret"
    (replacement / "secret.txt").write_bytes(secret)
    displaced = workspace / "trace-displaced"

    original_scandir = os.scandir
    swapped = False

    def replacing_scandir(path: Any):
        nonlocal swapped
        if not swapped:
            trace_dir.rename(displaced)
            replacement.rename(trace_dir)
            swapped = True
        return original_scandir(path)

    monkeypatch.setattr(os, "scandir", replacing_scandir)

    result = _capture(workspace, "trace")

    assert swapped is True
    _assert_fail_closed(result, secret)


def test_trace_capture_rejects_file_identity_drift_while_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    trace_dir = workspace / "trace"
    trace_dir.mkdir(parents=True)
    trace_file = trace_dir / "events.jsonl"
    trace_file.write_bytes(b"initial-event\n")
    drift = b"drifted-event\n"

    real_sha256 = hashlib.sha256
    changed = False

    class _DriftingDigest:
        def __init__(self) -> None:
            self._delegate = real_sha256()

        def update(self, value: bytes) -> None:
            nonlocal changed
            self._delegate.update(value)
            if not changed:
                with trace_file.open("ab") as stream:
                    stream.write(drift)
                changed = True

        def hexdigest(self) -> str:
            return self._delegate.hexdigest()

    monkeypatch.setattr(runner_module.hashlib, "sha256", _DriftingDigest)

    result = _capture(workspace, "trace")

    assert changed is True
    _assert_fail_closed(result, drift)
