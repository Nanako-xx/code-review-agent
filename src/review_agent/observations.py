from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from review_agent.checkpoint import _fsync_parent_directory


DEFAULT_MAX_OBSERVATION_LOG_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_OBSERVATIONS = 4096
DEFAULT_MAX_RAW_ARTIFACT_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_TOTAL_RAW_ARTIFACT_BYTES = 64 * 1024 * 1024


def _storage_path(path: Path) -> Path:
    """Use the extended-length namespace for Windows filesystem syscalls."""

    raw = os.fspath(path)
    if os.name != "nt" or raw.startswith("\\\\?\\"):
        return Path(raw)
    absolute = os.path.abspath(raw)
    if absolute.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + absolute[2:])
    return Path("\\\\?\\" + absolute)


def _path_is_symlink(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
    except OSError:
        pass
    return os.path.islink(_storage_path(path))


@dataclass(frozen=True)
class Observation:
    observation_id: str
    source: str
    revision: str
    path: str | None
    line_start: int | None
    line_end: int | None
    content_hash: str
    raw_artifact_ref: str
    context_view: str


class ObservationStore:
    def __init__(self, run_dir: Path, *, _create: bool = True) -> None:
        self.run_dir = Path(run_dir)
        self.observations_dir = self.run_dir / "observations"
        self.jsonl_path = self.run_dir / "observations.jsonl"
        self._observations: list[Observation] = []
        self._by_id: dict[str, Observation] = {}
        if _create:
            self.observations_dir.mkdir(parents=True, exist_ok=True)
            if self.jsonl_path.exists() and self.jsonl_path.stat().st_size:
                raise ValueError(
                    "observations.jsonl already exists; use ObservationStore.load() "
                    "with an explicit revision allowlist"
                )
            self.jsonl_path.touch(exist_ok=True)

    @classmethod
    def load(
        cls,
        run_dir: Path,
        expected_revisions: set[str],
        *,
        max_log_bytes: int = DEFAULT_MAX_OBSERVATION_LOG_BYTES,
        max_observations: int = DEFAULT_MAX_OBSERVATIONS,
        max_raw_artifact_bytes: int = DEFAULT_MAX_RAW_ARTIFACT_BYTES,
        max_total_raw_bytes: int = DEFAULT_MAX_TOTAL_RAW_ARTIFACT_BYTES,
    ) -> "ObservationStore":
        revisions = _expected_revisions(expected_revisions)
        _bounded_positive_int(
            max_log_bytes,
            "max_log_bytes",
            maximum=64 * 1024 * 1024,
        )
        _bounded_positive_int(
            max_observations,
            "max_observations",
            maximum=100_000,
        )
        _bounded_positive_int(
            max_raw_artifact_bytes,
            "max_raw_artifact_bytes",
            maximum=64 * 1024 * 1024,
        )
        _bounded_positive_int(
            max_total_raw_bytes,
            "max_total_raw_bytes",
            maximum=512 * 1024 * 1024,
        )
        store = cls(run_dir, _create=False)
        _validate_observations_directory(store.run_dir)

        observations: list[Observation] = []
        by_id: dict[str, Observation] = {}
        try:
            log_bytes = _read_regular_file_bytes(
                store.jsonl_path,
                "observations.jsonl",
                max_bytes=max_log_bytes,
            )
            lines = log_bytes.decode("utf-8").splitlines()
        except (OSError, UnicodeError) as error:
            raise ValueError(f"unable to read observations.jsonl: {error}") from error
        if len(lines) > max_observations:
            raise ValueError("observations.jsonl exceeds the bounded record count")

        hydration_plan: list[tuple[Observation, Path, bool]] = []
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                raise ValueError(
                    f"observations.jsonl line {line_number} must not be blank"
                )
            try:
                payload = json.loads(
                    line,
                    object_pairs_hook=_reject_duplicate_json_keys,
                )
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"observations.jsonl line {line_number} is invalid JSON: "
                    f"{error.msg}"
                ) from error
            observation = _observation_from_dict(
                payload,
                context=f"observations.jsonl line {line_number}",
            )
            if observation.revision not in revisions:
                raise ValueError(
                    f"observation {observation.observation_id} uses unauthorized "
                    f"revision binding: {observation.revision}"
                )
            legacy_id = _validate_observation_id(observation)
            existing = by_id.get(observation.observation_id)
            if existing is not None:
                if existing != observation:
                    raise ValueError(
                        "duplicate observation ID points to different content: "
                        f"{observation.observation_id}"
                    )
                continue
            raw_path = _safe_raw_artifact_path(
                store.run_dir,
                observation.raw_artifact_ref,
                observation.observation_id,
            )
            observations.append(observation)
            by_id[observation.observation_id] = observation
            hydration_plan.append((observation, raw_path, legacy_id))

        # Complete count and declared-size authorization before materializing
        # any raw artifact.  The descriptor reads below remain independently
        # bounded to close growth races after this preflight.
        total_declared_bytes = 0
        for observation, raw_path, _legacy_id in hydration_plan:
            raw_size = _regular_file_size(
                raw_path,
                observation.raw_artifact_ref,
            )
            if raw_size > max_raw_artifact_bytes:
                raise ValueError("raw observation artifact exceeds its byte bound")
            total_declared_bytes += raw_size
            if total_declared_bytes > max_total_raw_bytes:
                raise ValueError("raw observation artifacts exceed the total byte bound")

        total_read_bytes = 0
        for observation, raw_path, legacy_id in hydration_plan:
            raw_bytes = _read_regular_file_bytes(
                raw_path,
                observation.raw_artifact_ref,
                max_bytes=max_raw_artifact_bytes,
            )
            total_read_bytes += len(raw_bytes)
            if total_read_bytes > max_total_raw_bytes:
                raise ValueError("raw observation artifacts exceed the total byte bound")
            _validate_raw_artifact_hash(
                raw_bytes,
                observation,
                allow_legacy_newline_translation=legacy_id,
            )

        store._observations = observations
        store._by_id = by_id
        return store

    def record(
        self,
        source: str,
        revision: str,
        path: str | None,
        line_start: int | None,
        line_end: int | None,
        raw_content: str,
        context_view: str,
    ) -> Observation:
        for field_name, value in (("source", source), ("revision", revision)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be a non-empty string")
        if path is not None and not isinstance(path, str):
            raise ValueError("path must be a string or null")
        _validate_line_range(line_start, line_end, "observation")
        if not isinstance(raw_content, str) or not isinstance(context_view, str):
            raise ValueError("raw_content and context_view must be strings")
        raw_bytes = raw_content.encode("utf-8")
        content_hash = _sha256_bytes(raw_bytes)
        observation_id = _observation_id(
            source=source,
            revision=revision,
            path=path,
            line_start=line_start,
            line_end=line_end,
            content_hash=content_hash,
        )
        artifact_ref = f"observations/{observation_id}.txt"
        observation = Observation(
            observation_id=observation_id,
            source=source,
            revision=revision,
            path=path,
            line_start=line_start,
            line_end=line_end,
            content_hash=content_hash,
            raw_artifact_ref=artifact_ref,
            context_view=context_view,
        )
        existing = self._by_id.get(observation_id)
        if existing is not None:
            if existing != observation:
                raise ValueError(
                    f"observation ID collision with different content: {observation_id}"
                )
            self._validate_existing_artifact(existing, legacy_id=False)
            return existing

        legacy_observation_id = _legacy_observation_id(
            source=source,
            revision=revision,
            path=path,
            line_start=line_start,
            line_end=line_end,
            content_hash=content_hash,
        )
        legacy_existing = (
            self._by_id.get(legacy_observation_id)
            if legacy_observation_id is not None
            else None
        )
        if legacy_existing is not None:
            if not _same_observation_payload(legacy_existing, observation):
                raise ValueError(
                    "legacy observation ID collision with different content: "
                    f"{legacy_observation_id}"
                )
            self._validate_existing_artifact(legacy_existing, legacy_id=True)
            return legacy_existing

        raw_path = _safe_raw_artifact_path(
            self.run_dir,
            artifact_ref,
            observation_id,
        )
        if os.path.exists(_storage_path(raw_path)):
            existing_content = _read_regular_file_bytes(raw_path, artifact_ref)
            if existing_content != raw_bytes:
                raise ValueError(
                    f"raw observation artifact already exists with different content: "
                    f"{artifact_ref}"
                )
        else:
            _atomic_write_bytes(raw_path, raw_bytes)
        _append_jsonl_record(self.jsonl_path, asdict(observation))
        self._observations.append(observation)
        self._by_id[observation_id] = observation
        return observation

    def _validate_existing_artifact(
        self,
        observation: Observation,
        *,
        legacy_id: bool,
    ) -> None:
        raw_path = _safe_raw_artifact_path(
            self.run_dir,
            observation.raw_artifact_ref,
            observation.observation_id,
        )
        raw_bytes = _read_regular_file_bytes(raw_path, observation.raw_artifact_ref)
        _validate_raw_artifact_hash(
            raw_bytes,
            observation,
            allow_legacy_newline_translation=legacy_id,
        )

    def list_observations(self) -> list[Observation]:
        return list(self._observations)

    def summaries_by_id(self) -> dict[str, str]:
        return {observation.observation_id: observation.context_view for observation in self._observations}


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _observation_id(
    source: str,
    revision: str,
    path: str | None,
    line_start: int | None,
    line_end: int | None,
    content_hash: str,
) -> str:
    seed = json.dumps(
        [source, revision, path, line_start, line_end, content_hash],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"O-{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:32]}"


def _legacy_observation_id(
    source: str,
    revision: str,
    path: str | None,
    line_start: int | None,
    line_end: int | None,
    content_hash: str,
) -> str | None:
    if path == "" or any("|" in value for value in (source, revision, path or "")):
        return None
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


def _validate_observation_id(observation: Observation) -> bool:
    calculated_id = _observation_id(
        source=observation.source,
        revision=observation.revision,
        path=observation.path,
        line_start=observation.line_start,
        line_end=observation.line_end,
        content_hash=observation.content_hash,
    )
    if calculated_id == observation.observation_id:
        return False
    legacy_id = _legacy_observation_id(
        source=observation.source,
        revision=observation.revision,
        path=observation.path,
        line_start=observation.line_start,
        line_end=observation.line_end,
        content_hash=observation.content_hash,
    )
    if legacy_id == observation.observation_id:
        return True
    raise ValueError(
        f"observation ID validation failed: {observation.observation_id}"
    )


def _same_observation_payload(left: Observation, right: Observation) -> bool:
    return (
        left.source == right.source
        and left.revision == right.revision
        and left.path == right.path
        and left.line_start == right.line_start
        and left.line_end == right.line_end
        and left.content_hash == right.content_hash
        and left.context_view == right.context_view
    )


_OBSERVATION_FIELDS = {
    "observation_id",
    "source",
    "revision",
    "path",
    "line_start",
    "line_end",
    "content_hash",
    "raw_artifact_ref",
    "context_view",
}


def _observation_from_dict(value: Any, *, context: str) -> Observation:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    missing = _OBSERVATION_FIELDS - set(value)
    if missing:
        raise ValueError(
            f"{context} is missing required field(s): {', '.join(sorted(missing))}"
        )
    unexpected = set(value) - _OBSERVATION_FIELDS
    if unexpected:
        raise ValueError(
            f"{context} contains unsupported field(s): "
            f"{', '.join(sorted(str(item) for item in unexpected))}"
        )
    for field_name in (
        "observation_id",
        "source",
        "revision",
        "content_hash",
        "raw_artifact_ref",
        "context_view",
    ):
        if not isinstance(value[field_name], str):
            raise ValueError(f"{context}.{field_name} must be a string")
    for field_name in ("source", "revision"):
        if not value[field_name]:
            raise ValueError(f"{context}.{field_name} must be a non-empty string")
    path = value["path"]
    if path is not None and not isinstance(path, str):
        raise ValueError(f"{context}.path must be a string or null")
    line_start = value["line_start"]
    line_end = value["line_end"]
    _validate_line_range(line_start, line_end, context)
    content_hash = value["content_hash"]
    if len(content_hash) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in content_hash
    ):
        raise ValueError(f"{context}.content_hash must be a SHA-256 digest")
    return Observation(
        observation_id=value["observation_id"],
        source=value["source"],
        revision=value["revision"],
        path=path,
        line_start=line_start,
        line_end=line_end,
        content_hash=content_hash,
        raw_artifact_ref=value["raw_artifact_ref"],
        context_view=value["context_view"],
    )


def _validate_line_range(line_start: Any, line_end: Any, context: str) -> None:
    for name, value in (("line_start", line_start), ("line_end", line_end)):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"{context}.{name} must be a positive integer or null")
    if (line_start is None) != (line_end is None):
        raise ValueError(f"{context} line_start and line_end must both be set or null")
    if line_start is not None and line_end < line_start:
        raise ValueError(f"{context}.line_end must be greater than or equal to line_start")


def _expected_revisions(value: set[str]) -> frozenset[str]:
    if not isinstance(value, set) or not value:
        raise ValueError("expected_revisions must be a non-empty set of revision bindings")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError("expected_revisions must contain non-empty strings")
    return frozenset(value)


def _safe_raw_artifact_path(
    run_dir: Path,
    artifact_ref: str,
    observation_id: str,
) -> Path:
    expected_ref = f"observations/{observation_id}.txt"
    if artifact_ref != expected_ref:
        raise ValueError(
            f"raw_artifact_ref must use the canonical observation path: {expected_ref}"
        )
    observations_dir = _validate_observations_directory(run_dir)
    return observations_dir / f"{observation_id}.txt"


def _validate_observations_directory(run_dir: Path) -> Path:
    root = Path(run_dir).resolve(strict=True)
    observations_dir = Path(run_dir) / "observations"
    try:
        directory_metadata = os.lstat(_storage_path(observations_dir))
    except OSError as error:
        raise ValueError("observations directory is missing") from error
    if not stat.S_ISDIR(directory_metadata.st_mode) or _path_is_symlink(
        observations_dir
    ):
        raise ValueError("observations directory must be a regular directory")
    resolved_observations_dir = observations_dir.resolve(strict=True)
    try:
        resolved_observations_dir.relative_to(root)
    except ValueError as error:
        raise ValueError(
            "observations directory resolves outside the Session run directory"
        ) from error
    return observations_dir


def _read_regular_file_bytes(
    path: Path,
    artifact_ref: str,
    *,
    max_bytes: int | None = None,
) -> bytes:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    storage_path = _storage_path(path)
    try:
        descriptor = os.open(storage_path, flags)
    except OSError as error:
        raise ValueError(
            f"raw observation artifact is missing or not regular: {artifact_ref}"
        ) from error
    try:
        try:
            opened_metadata = os.fstat(descriptor)
            path_metadata = os.lstat(storage_path)
        except OSError as error:
            raise ValueError(
                f"unable to inspect raw observation artifact: {artifact_ref}"
            ) from error
        if (
            not stat.S_ISREG(opened_metadata.st_mode)
            or not stat.S_ISREG(path_metadata.st_mode)
            or stat.S_ISLNK(path_metadata.st_mode)
            or _path_is_symlink(path)
            or not os.path.samestat(path_metadata, opened_metadata)
        ):
            raise ValueError(
                f"raw observation artifact is missing or not regular: {artifact_ref}"
            )
        if max_bytes is not None and opened_metadata.st_size > max_bytes:
            raise ValueError(
                f"raw observation artifact exceeds the byte bound: {artifact_ref}"
            )

        chunks: list[bytes] = []
        total = 0
        while True:
            read_size = 1024 * 1024
            if max_bytes is not None:
                read_size = min(read_size, max_bytes + 1 - total)
            try:
                chunk = os.read(descriptor, read_size)
            except OSError as error:
                raise ValueError(
                    f"unable to read raw observation artifact: {artifact_ref}"
                ) from error
            if not chunk:
                break
            total += len(chunk)
            if max_bytes is not None and total > max_bytes:
                raise ValueError(
                    f"raw observation artifact exceeds the byte bound: {artifact_ref}"
                )
            chunks.append(chunk)

        try:
            final_opened = os.fstat(descriptor)
            final_path = os.lstat(storage_path)
        except OSError as error:
            raise ValueError(
                f"unable to inspect raw observation artifact: {artifact_ref}"
            ) from error
        if (
            not _same_file_snapshot(opened_metadata, final_opened)
            or not stat.S_ISREG(final_path.st_mode)
            or stat.S_ISLNK(final_path.st_mode)
            or _path_is_symlink(path)
            or not os.path.samestat(final_opened, final_path)
        ):
            raise ValueError(
                f"raw observation artifact changed while reading: {artifact_ref}"
            )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _regular_file_size(path: Path, artifact_ref: str) -> int:
    try:
        metadata = os.lstat(_storage_path(path))
    except OSError as error:
        raise ValueError(
            f"raw observation artifact is missing or not regular: {artifact_ref}"
        ) from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or _path_is_symlink(path)
    ):
        raise ValueError(
            f"raw observation artifact is missing or not regular: {artifact_ref}"
        )
    return metadata.st_size


def _same_file_snapshot(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        os.path.samestat(left, right)
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
        and left.st_ctime_ns == right.st_ctime_ns
    )


def _bounded_positive_int(value: Any, field_name: str, *, maximum: int) -> int:
    if type(value) is not int or value < 1 or value > maximum:
        raise ValueError(f"{field_name} must be a positive bounded integer")
    return value


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError("JSON objects must not contain duplicate keys")
        value[key] = child
    return value


def _validate_raw_artifact_hash(
    raw_bytes: bytes,
    observation: Observation,
    *,
    allow_legacy_newline_translation: bool,
) -> None:
    try:
        raw_text = raw_bytes.decode("utf-8")
    except UnicodeError as error:
        raise ValueError(
            f"unable to read raw observation artifact: {observation.raw_artifact_ref}"
        ) from error
    if _sha256_bytes(raw_bytes) == observation.content_hash:
        return
    if (
        allow_legacy_newline_translation
        and _sha256(raw_text.replace("\r\n", "\n")) == observation.content_hash
    ):
        return
    raise ValueError(
        f"observation raw artifact hash mismatch: {observation.raw_artifact_ref}"
    )


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".tmp-{uuid.uuid4().hex[:12]}.tmp")
    storage_temporary = _storage_path(temporary)
    storage_path = _storage_path(path)
    try:
        with storage_temporary.open("wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(storage_temporary, storage_path)
        _fsync_parent_directory(_storage_path(path.parent))
    finally:
        try:
            storage_temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _append_jsonl_record(path: Path, payload: dict[str, Any]) -> None:
    content = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    storage_path = _storage_path(path)
    descriptor = os.open(storage_path, flags, 0o666)
    try:
        opened_metadata = os.fstat(descriptor)
        if not stat.S_ISREG(opened_metadata.st_mode):
            raise ValueError("observations.jsonl must be a regular file")
        current_metadata = os.lstat(storage_path)
        if (
            not stat.S_ISREG(current_metadata.st_mode)
            or _path_is_symlink(path)
            or not os.path.samestat(current_metadata, opened_metadata)
        ):
            raise ValueError("observations.jsonl changed while opening")
        written = 0
        while written < len(content):
            count = os.write(descriptor, content[written:])
            if count <= 0:
                raise OSError("unable to append observations.jsonl")
            written += count
        os.fsync(descriptor)
        final_metadata = os.lstat(storage_path)
        if not os.path.samestat(final_metadata, opened_metadata):
            raise ValueError("observations.jsonl changed while appending")
    finally:
        os.close(descriptor)
    _fsync_parent_directory(_storage_path(path.parent))


def _is_regular_file(path: Path) -> bool:
    try:
        metadata = os.lstat(_storage_path(path))
    except OSError:
        return False
    return stat.S_ISREG(metadata.st_mode) and not _path_is_symlink(path)
