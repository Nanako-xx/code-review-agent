from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any, Mapping
import uuid

from review_agent.git_repo import collect_complete_diff_bytes
from review_agent.pr_workspace import (
    ArtifactDescriptor,
    PRWorkspaceError,
    PRWorkspaceStore,
    SnapshotWorkspace,
)
from review_agent.safe_io import (
    SafeIOError,
    atomic_replace_bytes,
    canonical_json_bytes,
    canonical_relative_path,
    cleanup_staging_files,
    ensure_secure_directory,
    read_verified_bytes,
    strict_json_loads,
)
from review_agent.revision import RevisionResolver, canonical_repository_identity


class DiffArtifactError(ValueError):
    """The complete Git Diff could not be materialized or decoded."""


class DiffArtifactIntegrityError(DiffArtifactError):
    """Persisted Diff bytes or their mechanical index failed verification."""


DIFF_INDEX_SCHEMA = "diff_artifact_index_v1"
PATCH_RELATIVE_PATH = "DiffArtifact/diff.patch"
INDEX_RELATIVE_PATH = "DiffArtifact/index.json"
MAX_PAGE_BYTES = 50_000

_STABLE_ID = re.compile(r"\A[A-Z][A-Z0-9]*-[0-9a-f]{64}\Z")
_GIT_OBJECT_ID = re.compile(r"\A(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_HUNK_HEADER = re.compile(
    rb"\A@@ -(?P<old_start>[0-9]+)(?:,(?P<old_count>[0-9]+))? "
    rb"\+(?P<new_start>[0-9]+)(?:,(?P<new_count>[0-9]+))? @@"
)
_FILE_STATUSES = frozenset({"add", "modify", "delete", "rename", "copy"})


def _int(value: Any, field_name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise DiffArtifactIntegrityError(
            f"{field_name} must be an integer greater than or equal to {minimum}"
        )
    return value


def _text(value: Any, field_name: str) -> str:
    if type(value) is not str or not value:
        raise DiffArtifactIntegrityError(f"{field_name} must be non-empty text")
    return value


def _exact_object(
    payload: Any,
    fields: tuple[str, ...],
    context: str,
) -> dict[str, Any]:
    if type(payload) is not dict or set(payload) != set(fields):
        raise DiffArtifactIntegrityError(f"{context} has an invalid exact schema")
    return dict(payload)


@dataclass(frozen=True)
class DiffHunkIndex:
    hunk_index: int
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    byte_start: int
    byte_end: int

    def to_dict(self) -> dict[str, int]:
        return {
            "hunk_index": self.hunk_index,
            "old_start": self.old_start,
            "old_count": self.old_count,
            "new_start": self.new_start,
            "new_count": self.new_count,
            "byte_start": self.byte_start,
            "byte_end": self.byte_end,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DiffHunkIndex":
        value = _exact_object(
            payload,
            (
                "hunk_index",
                "old_start",
                "old_count",
                "new_start",
                "new_count",
                "byte_start",
                "byte_end",
            ),
            "Diff hunk index",
        )
        result = cls(
            hunk_index=_int(value["hunk_index"], "hunk_index"),
            old_start=_int(value["old_start"], "old_start"),
            old_count=_int(value["old_count"], "old_count"),
            new_start=_int(value["new_start"], "new_start"),
            new_count=_int(value["new_count"], "new_count"),
            byte_start=_int(value["byte_start"], "byte_start"),
            byte_end=_int(value["byte_end"], "byte_end", minimum=1),
        )
        if result.byte_end <= result.byte_start:
            raise DiffArtifactIntegrityError("Diff hunk byte span is empty or reversed")
        return result


@dataclass(frozen=True)
class DiffFileIndex:
    file_index: int
    path: str
    previous_path: str | None
    status: str
    additions: int
    deletions: int
    binary: bool
    submodule: bool
    byte_start: int
    byte_end: int
    hunks: tuple[DiffHunkIndex, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_index": self.file_index,
            "path": self.path,
            "previous_path": self.previous_path,
            "status": self.status,
            "additions": self.additions,
            "deletions": self.deletions,
            "binary": self.binary,
            "submodule": self.submodule,
            "byte_start": self.byte_start,
            "byte_end": self.byte_end,
            "hunks": [hunk.to_dict() for hunk in self.hunks],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DiffFileIndex":
        value = _exact_object(
            payload,
            (
                "file_index",
                "path",
                "previous_path",
                "status",
                "additions",
                "deletions",
                "binary",
                "submodule",
                "byte_start",
                "byte_end",
                "hunks",
            ),
            "Diff file index",
        )
        try:
            path = canonical_relative_path(value["path"])
            previous = value["previous_path"]
            if previous is not None:
                previous = canonical_relative_path(previous)
        except SafeIOError as error:
            raise DiffArtifactIntegrityError(str(error)) from error
        status = _text(value["status"], "status")
        if status not in _FILE_STATUSES:
            raise DiffArtifactIntegrityError("Diff file status is invalid")
        if type(value["binary"]) is not bool or type(value["submodule"]) is not bool:
            raise DiffArtifactIntegrityError("Diff file flags must be booleans")
        hunks_payload = value["hunks"]
        if type(hunks_payload) is not list:
            raise DiffArtifactIntegrityError("Diff file hunks must be an array")
        result = cls(
            file_index=_int(value["file_index"], "file_index"),
            path=path,
            previous_path=previous,
            status=status,
            additions=_int(value["additions"], "additions"),
            deletions=_int(value["deletions"], "deletions"),
            binary=value["binary"],
            submodule=value["submodule"],
            byte_start=_int(value["byte_start"], "byte_start"),
            byte_end=_int(value["byte_end"], "byte_end", minimum=1),
            hunks=tuple(DiffHunkIndex.from_dict(item) for item in hunks_payload),
        )
        if result.byte_end <= result.byte_start:
            raise DiffArtifactIntegrityError("Diff file byte span is empty or reversed")
        if (status in {"rename", "copy"}) != (previous is not None):
            raise DiffArtifactIntegrityError(
                "Diff rename/copy status and previous_path do not agree"
            )
        return result


@dataclass(frozen=True)
class DiffArtifactIndex:
    snapshot_id: str
    base_sha: str
    head_sha: str
    patch_artifact_id: str
    diff_sha256: str
    diff_size_bytes: int
    files: tuple[DiffFileIndex, ...]
    schema_version: str = DIFF_INDEX_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "base_sha": self.base_sha,
            "head_sha": self.head_sha,
            "patch_artifact_id": self.patch_artifact_id,
            "diff_sha256": self.diff_sha256,
            "diff_size_bytes": self.diff_size_bytes,
            "files": [file.to_dict() for file in self.files],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DiffArtifactIndex":
        value = _exact_object(
            payload,
            (
                "schema_version",
                "snapshot_id",
                "base_sha",
                "head_sha",
                "patch_artifact_id",
                "diff_sha256",
                "diff_size_bytes",
                "files",
            ),
            "DiffArtifact index",
        )
        if value["schema_version"] != DIFF_INDEX_SCHEMA:
            raise DiffArtifactIntegrityError("DiffArtifact index schema is unsupported")
        for field_name in ("snapshot_id", "patch_artifact_id"):
            if type(value[field_name]) is not str or _STABLE_ID.fullmatch(
                value[field_name]
            ) is None:
                raise DiffArtifactIntegrityError(
                    f"DiffArtifact {field_name} is invalid"
                )
        for field_name in ("base_sha", "head_sha"):
            if type(value[field_name]) is not str or _GIT_OBJECT_ID.fullmatch(
                value[field_name]
            ) is None:
                raise DiffArtifactIntegrityError(
                    f"DiffArtifact {field_name} is invalid"
                )
        digest = value["diff_sha256"]
        if type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise DiffArtifactIntegrityError("DiffArtifact content hash is invalid")
        files = value["files"]
        if type(files) is not list:
            raise DiffArtifactIntegrityError("DiffArtifact files must be an array")
        return cls(
            snapshot_id=value["snapshot_id"],
            base_sha=value["base_sha"],
            head_sha=value["head_sha"],
            patch_artifact_id=value["patch_artifact_id"],
            diff_sha256=digest,
            diff_size_bytes=_int(value["diff_size_bytes"], "diff_size_bytes"),
            files=tuple(DiffFileIndex.from_dict(item) for item in files),
            schema_version=value["schema_version"],
        )


@dataclass(frozen=True)
class DiffArtifact:
    snapshot: SnapshotWorkspace
    patch: ArtifactDescriptor
    index_artifact: ArtifactDescriptor
    index: DiffArtifactIndex


@dataclass(frozen=True)
class DiffPage:
    data: bytes
    cursor: int
    next_cursor: int | None
    has_more: bool
    total_bytes: int


def _line_records(content: bytes) -> tuple[tuple[int, int, bytes], ...]:
    records: list[tuple[int, int, bytes]] = []
    start = 0
    while start < len(content):
        newline = content.find(b"\n", start)
        end = len(content) if newline < 0 else newline + 1
        records.append((start, end, content[start:end]))
        start = end
    return tuple(records)


def _split_git_tokens(raw: bytes) -> tuple[bytes, ...]:
    tokens: list[bytes] = []
    index = 0
    while index < len(raw):
        while index < len(raw) and raw[index : index + 1] == b" ":
            index += 1
        if index >= len(raw):
            break
        start = index
        if raw[index : index + 1] == b'"':
            index += 1
            escaped = False
            while index < len(raw):
                value = raw[index]
                index += 1
                if escaped:
                    escaped = False
                    continue
                if value == 0x5C:
                    escaped = True
                elif value == 0x22:
                    break
            else:
                raise DiffArtifactIntegrityError("Git Diff contains an unterminated path")
            tokens.append(raw[start:index])
        else:
            while index < len(raw) and raw[index : index + 1] != b" ":
                index += 1
            tokens.append(raw[start:index])
    return tuple(tokens)


def _decode_git_path(token: bytes, *, prefix: bytes | None = None) -> str:
    value = token.strip()
    if value.startswith(b'"'):
        if not value.endswith(b'"'):
            raise DiffArtifactIntegrityError("Git Diff path quoting is invalid")
        source = value[1:-1]
        decoded = bytearray()
        index = 0
        escapes = {
            ord("a"): 0x07,
            ord("b"): 0x08,
            ord("t"): 0x09,
            ord("n"): 0x0A,
            ord("v"): 0x0B,
            ord("f"): 0x0C,
            ord("r"): 0x0D,
            ord('"'): 0x22,
            ord("\\"): 0x5C,
        }
        while index < len(source):
            current = source[index]
            index += 1
            if current != 0x5C:
                decoded.append(current)
                continue
            if index >= len(source):
                raise DiffArtifactIntegrityError("Git Diff path escape is incomplete")
            escaped = source[index]
            index += 1
            if escaped in escapes:
                decoded.append(escapes[escaped])
                continue
            if ord("0") <= escaped <= ord("7"):
                octal = bytearray([escaped])
                while (
                    len(octal) < 3
                    and index < len(source)
                    and ord("0") <= source[index] <= ord("7")
                ):
                    octal.append(source[index])
                    index += 1
                decoded.append(int(octal.decode("ascii"), 8))
                continue
            raise DiffArtifactIntegrityError("Git Diff path escape is unsupported")
        raw_path = bytes(decoded)
    else:
        raw_path = value
    if prefix is not None:
        if not raw_path.startswith(prefix):
            raise DiffArtifactIntegrityError("Git Diff path prefix is invalid")
        raw_path = raw_path[len(prefix) :]
    try:
        path = raw_path.decode("utf-8", "strict")
        return canonical_relative_path(path)
    except (UnicodeError, SafeIOError) as error:
        raise DiffArtifactIntegrityError("Git Diff path is not canonical UTF-8") from error


def _header_paths(header: bytes) -> tuple[str, str]:
    line = header[:-1] if header.endswith(b"\n") else header
    payload = line[len(b"diff --git ") :]
    tokens = _split_git_tokens(payload)
    if len(tokens) != 2:
        # Git quotes tabs, newlines, quotes, backslashes and non-ASCII bytes, but
        # it intentionally leaves ordinary spaces unquoted.  In that case the
        # unquoted header is still unambiguous because the second path starts at
        # the last `` b/`` delimiter.
        delimiter = payload.rfind(b" b/")
        if delimiter <= 0:
            raise DiffArtifactIntegrityError("Git Diff file header is invalid")
        tokens = (payload[:delimiter], payload[delimiter + 1 :])
    return (
        _decode_git_path(tokens[0], prefix=b"a/"),
        _decode_git_path(tokens[1], prefix=b"b/"),
    )


def _metadata_path(line: bytes, prefix: bytes) -> str:
    value = line[len(prefix) :]
    if value.endswith(b"\n"):
        value = value[:-1]
    return _decode_git_path(value)


def _hunk_from_records(
    records: tuple[tuple[int, int, bytes], ...],
    hunk_index: int,
    byte_end: int,
) -> tuple[DiffHunkIndex, int, int]:
    byte_start = records[0][0]
    header = records[0][2]
    match = _HUNK_HEADER.match(header.rstrip(b"\n"))
    if match is None:
        raise DiffArtifactIntegrityError("Git Diff hunk header is invalid")
    old_start = int(match.group("old_start"))
    old_count = int(match.group("old_count") or b"1")
    new_start = int(match.group("new_start"))
    new_count = int(match.group("new_count") or b"1")
    additions = 0
    deletions = 0
    old_seen = 0
    new_seen = 0
    for _start, _end, line in records[1:]:
        if line.startswith(b"+"):
            additions += 1
            new_seen += 1
        elif line.startswith(b"-"):
            deletions += 1
            old_seen += 1
        elif line.startswith(b" "):
            old_seen += 1
            new_seen += 1
        elif line.startswith(b"\\ No newline at end of file"):
            continue
        else:
            raise DiffArtifactIntegrityError("Git Diff hunk body is invalid")
    if old_seen != old_count or new_seen != new_count:
        raise DiffArtifactIntegrityError("Git Diff hunk line ranges do not match its body")
    return (
        DiffHunkIndex(
            hunk_index=hunk_index,
            old_start=old_start,
            old_count=old_count,
            new_start=new_start,
            new_count=new_count,
            byte_start=byte_start,
            byte_end=byte_end,
        ),
        additions,
        deletions,
    )


def _parse_file_section(
    content: bytes,
    records: tuple[tuple[int, int, bytes], ...],
    file_index: int,
    byte_end: int,
) -> DiffFileIndex:
    byte_start = records[0][0]
    old_header_path, new_header_path = _header_paths(records[0][2])
    status = "modify"
    previous_path: str | None = None
    current_path = new_header_path
    binary = False
    submodule = False
    hunk_record_indices: list[int] = []

    for record_index, (_start, _end, line) in enumerate(records[1:], start=1):
        if line.startswith(b"@@ "):
            hunk_record_indices.append(record_index)
            continue
        if line.startswith(b"new file mode "):
            status = "add"
        elif line.startswith(b"deleted file mode "):
            status = "delete"
            current_path = old_header_path
        elif line.startswith(b"rename from "):
            status = "rename"
            previous_path = _metadata_path(line, b"rename from ")
        elif line.startswith(b"rename to "):
            current_path = _metadata_path(line, b"rename to ")
        elif line.startswith(b"copy from "):
            status = "copy"
            previous_path = _metadata_path(line, b"copy from ")
        elif line.startswith(b"copy to "):
            current_path = _metadata_path(line, b"copy to ")
        if line.startswith(b"GIT binary patch") or line.startswith(b"Binary files "):
            binary = True
        if b"Subproject commit " in line or b"mode 160000" in line:
            submodule = True

    hunks: list[DiffHunkIndex] = []
    additions = 0
    deletions = 0
    for hunk_index, record_index in enumerate(hunk_record_indices):
        next_record = (
            hunk_record_indices[hunk_index + 1]
            if hunk_index + 1 < len(hunk_record_indices)
            else len(records)
        )
        hunk_end = records[next_record][0] if next_record < len(records) else byte_end
        hunk, hunk_additions, hunk_deletions = _hunk_from_records(
            records[record_index:next_record],
            hunk_index,
            hunk_end,
        )
        hunks.append(hunk)
        additions += hunk_additions
        deletions += hunk_deletions

    if status in {"rename", "copy"} and previous_path is None:
        previous_path = old_header_path
    if status not in {"rename", "copy"}:
        previous_path = None
    if content[byte_start:byte_end] != b"".join(
        record[2] for record in records
    ):
        raise DiffArtifactIntegrityError("Git Diff file span is not contiguous")
    return DiffFileIndex(
        file_index=file_index,
        path=current_path,
        previous_path=previous_path,
        status=status,
        additions=additions,
        deletions=deletions,
        binary=binary,
        submodule=submodule,
        byte_start=byte_start,
        byte_end=byte_end,
        hunks=tuple(hunks),
    )


def parse_diff_patch(
    patch: bytes,
    *,
    snapshot_id: str,
    base_sha: str,
    head_sha: str,
    patch_artifact_id: str,
) -> DiffArtifactIndex:
    if type(patch) is not bytes:
        raise DiffArtifactIntegrityError("Diff patch must be bytes")
    records = _line_records(patch)
    section_indices = [
        index
        for index, (_start, _end, line) in enumerate(records)
        if line.startswith(b"diff --git ")
    ]
    if patch and (not section_indices or section_indices[0] != 0):
        raise DiffArtifactIntegrityError("Git Diff has unindexed preamble bytes")
    files: list[DiffFileIndex] = []
    for file_index, record_index in enumerate(section_indices):
        next_record = (
            section_indices[file_index + 1]
            if file_index + 1 < len(section_indices)
            else len(records)
        )
        byte_end = records[next_record][0] if next_record < len(records) else len(patch)
        files.append(
            _parse_file_section(
                patch,
                records[record_index:next_record],
                file_index,
                byte_end,
            )
        )
    return DiffArtifactIndex(
        snapshot_id=snapshot_id,
        base_sha=base_sha,
        head_sha=head_sha,
        patch_artifact_id=patch_artifact_id,
        diff_sha256=hashlib.sha256(patch).hexdigest(),
        diff_size_bytes=len(patch),
        files=tuple(files),
    )


def validate_diff_index(index: DiffArtifactIndex, patch: bytes) -> None:
    if not isinstance(index, DiffArtifactIndex) or type(patch) is not bytes:
        raise DiffArtifactIntegrityError("Diff index validation input is invalid")
    if len(patch) != index.diff_size_bytes:
        raise DiffArtifactIntegrityError("Diff index size binding does not match")
    if hashlib.sha256(patch).hexdigest() != index.diff_sha256:
        raise DiffArtifactIntegrityError("Diff index content hash does not match")
    replayed = parse_diff_patch(
        patch,
        snapshot_id=index.snapshot_id,
        base_sha=index.base_sha,
        head_sha=index.head_sha,
        patch_artifact_id=index.patch_artifact_id,
    )
    if replayed.to_dict() != index.to_dict():
        raise DiffArtifactIntegrityError("Diff index replay or byte offsets do not match")


class DiffArtifactStore:
    def __init__(self, workspace_store: PRWorkspaceStore) -> None:
        if not isinstance(workspace_store, PRWorkspaceStore):
            raise DiffArtifactError("DiffArtifact requires a PRWorkspaceStore")
        self._workspace_store = workspace_store

    def materialize(
        self,
        repository_path: Path,
        snapshot: SnapshotWorkspace,
    ) -> DiffArtifact:
        diff_directory = snapshot.path / "DiffArtifact"
        try:
            self._workspace_store.verify_snapshot(snapshot)
            observed_repository = canonical_repository_identity(
                RevisionResolver().repository_identity(Path(repository_path))
            )
            if observed_repository != snapshot.workspace.resolved_pr.repository:
                raise DiffArtifactIntegrityError(
                    "Diff repository identity does not match the Snapshot"
                )
            ensure_secure_directory(diff_directory)
            cleanup_staging_files(diff_directory)
            patch = collect_complete_diff_bytes(
                Path(repository_path),
                snapshot.base_sha,
                snapshot.head_sha,
            )
            token = uuid.uuid4().hex
            patch_staging = diff_directory / f".stage-{token}.tmp"
            index_staging = diff_directory / f".stage-{uuid.uuid4().hex}.tmp"
            atomic_replace_bytes(patch_staging, patch)
            staged_patch = read_verified_bytes(
                patch_staging,
                hashlib.sha256(patch).hexdigest(),
            )
            patch_descriptor = self._workspace_store.describe_artifact(
                snapshot,
                PATCH_RELATIVE_PATH,
                staged_patch,
            )
            index = parse_diff_patch(
                staged_patch,
                snapshot_id=snapshot.snapshot_id,
                base_sha=snapshot.base_sha,
                head_sha=snapshot.head_sha,
                patch_artifact_id=patch_descriptor.artifact_id,
            )
            validate_diff_index(index, staged_patch)
            index_bytes = canonical_json_bytes(index.to_dict())
            atomic_replace_bytes(index_staging, index_bytes)
            staged_index = read_verified_bytes(
                index_staging,
                hashlib.sha256(index_bytes).hexdigest(),
            )
            if DiffArtifactIndex.from_dict(
                strict_json_loads(staged_index)
            ) != index:
                raise DiffArtifactIntegrityError(
                    "staged Diff index does not round-trip"
                )
            patch_artifact = self._workspace_store.publish_create_only(
                snapshot,
                PATCH_RELATIVE_PATH,
                staged_patch,
            )
            index_artifact = self._workspace_store.publish_create_only(
                snapshot,
                INDEX_RELATIVE_PATH,
                staged_index,
            )
        except (PRWorkspaceError, SafeIOError) as error:
            raise DiffArtifactIntegrityError(str(error)) from error
        finally:
            for local_name in ("patch_staging", "index_staging"):
                staging = locals().get(local_name)
                if isinstance(staging, Path):
                    try:
                        staging.unlink()
                    except FileNotFoundError:
                        pass
                    except OSError:
                        pass
        return DiffArtifact(
            snapshot=snapshot,
            patch=patch_artifact,
            index_artifact=index_artifact,
            index=index,
        )

    def load(self, snapshot: SnapshotWorkspace) -> DiffArtifact:
        try:
            patch_artifact = self._workspace_store.find_snapshot_artifact(
                snapshot, PATCH_RELATIVE_PATH
            )
            index_artifact = self._workspace_store.find_snapshot_artifact(
                snapshot, INDEX_RELATIVE_PATH
            )
            patch = read_verified_bytes(
                patch_artifact.path,
                patch_artifact.sha256,
                max_bytes=patch_artifact.size_bytes,
            )
            payload = self._workspace_store.read_verified_json(
                snapshot, index_artifact.artifact_id
            )
            index = DiffArtifactIndex.from_dict(payload)
        except (PRWorkspaceError, SafeIOError) as error:
            raise DiffArtifactIntegrityError(str(error)) from error
        if (
            index.snapshot_id != snapshot.snapshot_id
            or index.base_sha != snapshot.base_sha
            or index.head_sha != snapshot.head_sha
            or index.patch_artifact_id != patch_artifact.artifact_id
            or index.diff_sha256 != patch_artifact.sha256
            or index.diff_size_bytes != patch_artifact.size_bytes
        ):
            raise DiffArtifactIntegrityError("DiffArtifact Snapshot binding does not match")
        validate_diff_index(index, patch)
        return DiffArtifact(
            snapshot=snapshot,
            patch=patch_artifact,
            index_artifact=index_artifact,
            index=index,
        )

    def _verified_patch(self, artifact: DiffArtifact) -> bytes:
        if not isinstance(artifact, DiffArtifact):
            raise DiffArtifactIntegrityError("DiffArtifact handle is invalid")
        try:
            patch = read_verified_bytes(
                artifact.patch.path,
                artifact.patch.sha256,
                max_bytes=artifact.patch.size_bytes,
            )
        except SafeIOError as error:
            raise DiffArtifactIntegrityError(str(error)) from error
        validate_diff_index(artifact.index, patch)
        return patch

    def read_file(self, artifact: DiffArtifact, file_index: int) -> bytes:
        patch = self._verified_patch(artifact)
        if type(file_index) is not int or not 0 <= file_index < len(
            artifact.index.files
        ):
            raise DiffArtifactError("Diff file index is out of range")
        entry = artifact.index.files[file_index]
        return patch[entry.byte_start : entry.byte_end]

    def read_hunk(
        self,
        artifact: DiffArtifact,
        file_index: int,
        hunk_index: int,
    ) -> bytes:
        patch = self._verified_patch(artifact)
        if type(file_index) is not int or not 0 <= file_index < len(
            artifact.index.files
        ):
            raise DiffArtifactError("Diff file index is out of range")
        file_entry = artifact.index.files[file_index]
        if type(hunk_index) is not int or not 0 <= hunk_index < len(file_entry.hunks):
            raise DiffArtifactError("Diff hunk index is out of range")
        hunk = file_entry.hunks[hunk_index]
        return patch[hunk.byte_start : hunk.byte_end]

    def read_page(
        self,
        artifact: DiffArtifact,
        *,
        cursor: int = 0,
        max_bytes: int = MAX_PAGE_BYTES,
    ) -> DiffPage:
        patch = self._verified_patch(artifact)
        if type(cursor) is not int or not 0 <= cursor <= len(patch):
            raise DiffArtifactError("Diff page cursor is out of range")
        if type(max_bytes) is not int or not 1 <= max_bytes <= MAX_PAGE_BYTES:
            raise DiffArtifactError(
                f"Diff page max_bytes must be between 1 and {MAX_PAGE_BYTES}"
            )
        end = min(len(patch), cursor + max_bytes)
        has_more = end < len(patch)
        return DiffPage(
            data=patch[cursor:end],
            cursor=cursor,
            next_cursor=end if has_more else None,
            has_more=has_more,
            total_bytes=len(patch),
        )


__all__ = [
    "DiffArtifact",
    "DiffArtifactError",
    "DiffArtifactIndex",
    "DiffArtifactIntegrityError",
    "DiffArtifactStore",
    "DiffFileIndex",
    "DiffHunkIndex",
    "DiffPage",
    "parse_diff_patch",
    "validate_diff_index",
]
