from __future__ import annotations

import io
from pathlib import Path, PurePosixPath
import subprocess
import tokenize

from review_agent.models import QualityGateResult


def detect_quality_gates(repo_path: Path, revision: str | None = None) -> list[str]:
    if revision is None:
        has_python = any(path.suffix == ".py" for path in repo_path.rglob("*.py"))
    else:
        has_python = bool(_python_blobs_at_revision(repo_path, revision))
    return ["python_compile"] if has_python else []


def run_python_compile_gate(
    repo_path: Path,
    revision: str | None = None,
) -> QualityGateResult:
    if revision is None:
        python_sources = [
            (str(path), path.read_bytes())
            for path in repo_path.rglob("*.py")
            if ".git" not in path.parts
        ]
    else:
        python_sources = [
            (
                f"{revision}:{relative_path}",
                _read_blob(repo_path, object_id),
            )
            for relative_path, object_id in _python_blobs_at_revision(
                repo_path,
                revision,
            )
        ]

    for filename, raw_source in python_sources:
        try:
            source = _decode_python_source(raw_source)
            compile(source, filename, "exec")
        except (LookupError, SyntaxError, UnicodeDecodeError) as exc:
            return _failed_python_compile_result(filename, exc)

    return QualityGateResult(
        name="python_compile",
        status="passed",
        command=["python", "-c", "compile(source, filename, 'exec')"],
        summary=f"Compiled {len(python_sources)} Python files",
    )


def _decode_python_source(raw_source: bytes) -> str:
    encoding, _ = tokenize.detect_encoding(io.BytesIO(raw_source).readline)
    return raw_source.decode(encoding)


def _failed_python_compile_result(
    filename: str,
    error: Exception,
) -> QualityGateResult:
    return QualityGateResult(
        name="python_compile",
        status="failed",
        command=["python", "-c", "compile(source, filename, 'exec')"],
        summary=f"{filename}: {type(error).__name__}: {error}",
    )


def _python_blobs_at_revision(
    repo_path: Path,
    revision: str,
) -> list[tuple[str, str]]:
    result = _run_git(
        repo_path,
        ["ls-tree", "-r", "-z", "--full-tree", revision, "--"],
    )
    python_blobs: list[tuple[str, str]] = []
    for record in result.stdout.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            _mode, object_type, object_id = metadata.split(b" ", 2)
        except ValueError as error:
            raise RuntimeError("git ls-tree returned a malformed entry") from error
        relative_path = raw_path.decode("utf-8", errors="surrogateescape")
        if object_type == b"blob" and PurePosixPath(relative_path).suffix == ".py":
            python_blobs.append((relative_path, object_id.decode("ascii")))
    return python_blobs


def _read_blob(repo_path: Path, object_id: str) -> bytes:
    return _run_git(repo_path, ["cat-file", "blob", object_id]).stdout


def _run_git(repo_path: Path, args: list[str]) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=Path(repo_path),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as error:
        raise RuntimeError(f"unable to execute Git: {error}") from error
    if result.returncode != 0:
        reason = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Git command failed: {reason or 'unknown Git error'}")
    return result
