from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
import errno
import hashlib
import hmac
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
from typing import Any, Mapping
import unicodedata
import uuid


class SafeIOError(ValueError):
    """A managed path or durable file failed a fail-closed safety check."""


_STAGING_NAME = re.compile(r"\A\.stage-[0-9a-f]{32}\.tmp\Z")
_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
_WINDOWS_RESERVED_NAMES = frozenset(
    {
        "aux",
        "con",
        "nul",
        "prn",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
)
_WINDOWS_FORBIDDEN_CHARACTERS = frozenset('<>:"|?*')
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400


def _storage_path(path: Path) -> Path:
    """Use Windows' extended namespace only after managed-path validation."""

    raw = os.fspath(path)
    if os.name != "nt" or raw.startswith("\\\\?\\"):
        return Path(raw)
    absolute = os.path.abspath(raw)
    if absolute.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + absolute[2:])
    return Path("\\\\?\\" + absolute)


def _json_ready(value: Any, context: str = "value") -> Any:
    if value is None or type(value) in {str, int, bool}:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SafeIOError(f"{context} contains a non-finite number")
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value), context)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise SafeIOError(f"{context} contains a non-string object key")
            result[key] = _json_ready(item, f"{context}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _json_ready(item, f"{context}[{index}]")
            for index, item in enumerate(value)
        ]
    raise SafeIOError(f"{context} contains a non-JSON value")


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON for manifests and content identities."""

    try:
        return json.dumps(
            _json_ready(value),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        if isinstance(error, SafeIOError):
            raise
        raise SafeIOError("value cannot be encoded as canonical JSON") from error


def strict_json_loads(raw: str | bytes | bytearray) -> Any:
    if isinstance(raw, (bytes, bytearray)):
        try:
            text = bytes(raw).decode("utf-8", "strict")
        except UnicodeError as error:
            raise SafeIOError("JSON file must be valid UTF-8") from error
    elif type(raw) is str:
        text = raw
    else:
        raise SafeIOError("JSON input must be text or UTF-8 bytes")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SafeIOError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(token: str) -> None:
        raise SafeIOError(f"JSON contains a non-finite number: {token}")

    try:
        return json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except SafeIOError:
        raise
    except (json.JSONDecodeError, RecursionError, TypeError, ValueError) as error:
        raise SafeIOError("file contains invalid JSON") from error


def canonical_relative_path(value: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise SafeIOError("path must be a non-empty canonical relative path")
    if "\\" in value or "\x00" in value:
        raise SafeIOError("path must use safe forward-slash separators")
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
        raise SafeIOError("path must stay inside its managed directory")
    for part in parts:
        normalized = unicodedata.normalize("NFKC", part)
        basename = part.split(".", 1)[0].casefold()
        if (
            normalized != part
            or part != part.strip()
            or part.endswith((".", " "))
            or basename in _WINDOWS_RESERVED_NAMES
            or any(character in _WINDOWS_FORBIDDEN_CHARACTERS for character in part)
            or any(ord(character) < 32 for character in part)
        ):
            raise SafeIOError("path contains a non-canonical Windows name")
    return value


def metadata_is_reparse_point(metadata: Any) -> bool:
    attributes = getattr(metadata, "st_file_attributes", 0)
    return bool(attributes & _FILE_ATTRIBUTE_REPARSE_POINT)


def _lstat(path: Path) -> os.stat_result | None:
    try:
        return os.lstat(_storage_path(path))
    except FileNotFoundError:
        return None
    except OSError as error:
        raise SafeIOError(f"managed path is unavailable: {path}") from error


def _assert_not_link_or_reparse(path: Path, metadata: os.stat_result) -> None:
    if stat.S_ISLNK(metadata.st_mode) or metadata_is_reparse_point(metadata):
        raise SafeIOError(f"managed path must not contain a link or reparse point: {path}")


def _assert_exact_child_name(
    parent: Path,
    name: str,
    *,
    allow_unreadable: bool = False,
) -> None:
    metadata = _lstat(parent)
    if metadata is None:
        return
    _assert_not_link_or_reparse(parent, metadata)
    if not stat.S_ISDIR(metadata.st_mode):
        raise SafeIOError(f"managed path parent is not a directory: {parent}")
    wanted = unicodedata.normalize("NFKC", name).casefold()
    try:
        with os.scandir(_storage_path(parent)) as entries:
            for entry in entries:
                observed = unicodedata.normalize("NFKC", entry.name).casefold()
                if observed == wanted and entry.name != name:
                    raise SafeIOError(
                        f"managed path uses a case alias for existing name: {name}"
                    )
    except SafeIOError:
        raise
    except PermissionError:
        if allow_unreadable:
            return
        raise SafeIOError(f"unable to inspect managed directory: {parent}") from None
    except OSError as error:
        raise SafeIOError(f"unable to inspect managed directory: {parent}") from error


def _absolute_real_path(path: Path) -> Path:
    raw = os.fspath(path)
    if raw.startswith("\\\\?\\"):
        raise SafeIOError("extended-length paths are not allowed for managed roots")
    if "\x00" in raw:
        raise SafeIOError("managed root path contains a null character")
    return Path(os.path.abspath(raw))


def ensure_secure_directory(path: Path) -> Path:
    """Create a real directory chain without links, junctions, or case aliases."""

    target = _absolute_real_path(Path(path))
    anchor = Path(target.anchor)
    current = anchor
    relative_parts = target.parts[1:] if target.anchor else target.parts
    anchor_metadata = _lstat(anchor)
    if anchor_metadata is None or not stat.S_ISDIR(anchor_metadata.st_mode):
        raise SafeIOError("managed root anchor is unavailable")
    _assert_not_link_or_reparse(anchor, anchor_metadata)

    for part in relative_parts:
        canonical_relative_path(part)
        _assert_exact_child_name(current, part, allow_unreadable=True)
        child = current / part
        metadata = _lstat(child)
        if metadata is None:
            try:
                _storage_path(child).mkdir()
            except FileExistsError:
                metadata = _lstat(child)
            except OSError as error:
                raise SafeIOError(f"unable to create managed directory: {child}") from error
            else:
                metadata = _lstat(child)
        if metadata is None:
            raise SafeIOError(f"managed directory disappeared: {child}")
        _assert_not_link_or_reparse(child, metadata)
        if not stat.S_ISDIR(metadata.st_mode):
            raise SafeIOError(f"managed path is not a directory: {child}")
        current = child
    return target


def resolve_managed_path(root: Path, relative_path: str) -> Path:
    managed_root = ensure_secure_directory(root)
    relative = canonical_relative_path(relative_path)
    current = managed_root
    parts = relative.split("/")
    for index, part in enumerate(parts):
        _assert_exact_child_name(current, part)
        candidate = current / part
        metadata = _lstat(candidate)
        if metadata is not None:
            _assert_not_link_or_reparse(candidate, metadata)
            if index < len(parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
                raise SafeIOError(f"managed path parent is not a directory: {candidate}")
        current = candidate
    try:
        current.relative_to(managed_root)
    except ValueError as error:
        raise SafeIOError("managed path escapes its root") from error
    return current


def fsync_parent_directory(directory: Path, *, os_module: Any = os) -> None:
    flags = os_module.O_RDONLY | getattr(os_module, "O_DIRECTORY", 0)
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
        descriptor = os_module.open(_storage_path(directory), flags)
    except OSError as error:
        if error.errno in unsupported_errors:
            return
        raise
    try:
        try:
            os_module.fsync(descriptor)
        except OSError as error:
            if error.errno not in unsupported_errors:
                raise
    finally:
        os_module.close(descriptor)


def _staging_path(destination: Path) -> Path:
    return destination.with_name(f".stage-{uuid.uuid4().hex}.tmp")


def atomic_replace_bytes(
    path: Path,
    content: bytes,
    *,
    os_module: Any = os,
    allow_legacy_extended_path: bool = False,
) -> None:
    if type(content) is not bytes:
        raise SafeIOError("atomic file content must be bytes")
    destination = Path(path)
    if not allow_legacy_extended_path:
        ensure_secure_directory(destination.parent)
        canonical_relative_path(destination.name)
        _assert_exact_child_name(destination.parent, destination.name)
        existing = _lstat(destination)
        if existing is not None:
            _assert_not_link_or_reparse(destination, existing)
            if not stat.S_ISREG(existing.st_mode):
                raise SafeIOError(
                    "atomic replacement destination must be a regular file"
                )
    staging = _staging_path(destination)
    storage_staging = _storage_path(staging)
    storage_destination = _storage_path(destination)
    try:
        with storage_staging.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os_module.fsync(handle.fileno())
        os_module.replace(storage_staging, storage_destination)
        fsync_parent_directory(destination.parent, os_module=os_module)
    finally:
        try:
            storage_staging.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass


def atomic_write_text(
    path: Path,
    content: str,
    *,
    os_module: Any = os,
    allow_legacy_extended_path: bool = False,
) -> None:
    if type(content) is not str:
        raise SafeIOError("atomic text content must be a string")
    atomic_replace_bytes(
        Path(path),
        content.encode("utf-8"),
        os_module=os_module,
        allow_legacy_extended_path=allow_legacy_extended_path,
    )


def publish_create_only_bytes(
    path: Path,
    content: bytes,
    *,
    os_module: Any = os,
) -> None:
    if type(content) is not bytes:
        raise SafeIOError("create-only file content must be bytes")
    destination = Path(path)
    ensure_secure_directory(destination.parent)
    canonical_relative_path(destination.name)
    _assert_exact_child_name(destination.parent, destination.name)
    if _lstat(destination) is not None:
        raise SafeIOError(f"create-only destination already exists: {destination.name}")

    staging = _staging_path(destination)
    storage_staging = _storage_path(staging)
    storage_destination = _storage_path(destination)
    published = False
    try:
        with storage_staging.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os_module.fsync(handle.fileno())
        try:
            os_module.link(storage_staging, storage_destination)
        except FileExistsError as error:
            raise SafeIOError(
                f"create-only destination already exists: {destination.name}"
            ) from error
        published = True
        storage_staging.unlink()
        fsync_parent_directory(destination.parent, os_module=os_module)
    finally:
        try:
            storage_staging.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            if not published:
                raise


def assert_regular_file(path: Path) -> Path:
    candidate = Path(path)
    metadata = _lstat(candidate)
    if metadata is None:
        raise SafeIOError(f"required regular file does not exist: {candidate}")
    _assert_not_link_or_reparse(candidate, metadata)
    if not stat.S_ISREG(metadata.st_mode):
        raise SafeIOError(f"required path is not a regular file: {candidate}")
    return candidate


def sha256_hex(content: bytes) -> str:
    if type(content) is not bytes:
        raise SafeIOError("SHA-256 content must be bytes")
    return hashlib.sha256(content).hexdigest()


def read_verified_bytes(
    path: Path,
    expected_sha256: str,
    *,
    max_bytes: int | None = None,
) -> bytes:
    if type(expected_sha256) is not str or _SHA256.fullmatch(expected_sha256) is None:
        raise SafeIOError("expected hash must be a full lowercase SHA-256 digest")
    candidate = assert_regular_file(path)
    storage_candidate = _storage_path(candidate)
    try:
        if max_bytes is not None and storage_candidate.stat().st_size > max_bytes:
            raise SafeIOError("regular file exceeds its configured size bound")
        content = storage_candidate.read_bytes()
    except SafeIOError:
        raise
    except OSError as error:
        raise SafeIOError(f"unable to read regular file: {candidate}") from error
    if max_bytes is not None and len(content) > max_bytes:
        raise SafeIOError("regular file grew beyond its configured size bound")
    actual = sha256_hex(content)
    if not hmac.compare_digest(actual, expected_sha256):
        raise SafeIOError("regular file content hash does not match")
    return content


def read_strict_json(path: Path, *, max_bytes: int = 16 * 1024 * 1024) -> Any:
    candidate = assert_regular_file(path)
    storage_candidate = _storage_path(candidate)
    try:
        if storage_candidate.stat().st_size > max_bytes:
            raise SafeIOError("JSON file exceeds its configured size bound")
        content = storage_candidate.read_bytes()
    except SafeIOError:
        raise
    except OSError as error:
        raise SafeIOError(f"unable to read JSON file: {candidate}") from error
    if len(content) > max_bytes:
        raise SafeIOError("JSON file grew beyond its configured size bound")
    return strict_json_loads(content)


def cleanup_staging_files(directory: Path) -> tuple[Path, ...]:
    root = ensure_secure_directory(directory)
    removed: list[Path] = []
    try:
        with os.scandir(_storage_path(root)) as iterator:
            entries = tuple(root / entry.name for entry in iterator)
    except OSError as error:
        raise SafeIOError("unable to enumerate staging directory") from error
    for entry in entries:
        if _STAGING_NAME.fullmatch(entry.name) is None:
            continue
        metadata = _lstat(entry)
        if metadata is None:
            continue
        if (
            stat.S_ISLNK(metadata.st_mode)
            or metadata_is_reparse_point(metadata)
            or not stat.S_ISREG(metadata.st_mode)
        ):
            raise SafeIOError("staging cleanup target must be a regular file")
        try:
            _storage_path(entry).unlink()
        except OSError as error:
            raise SafeIOError("unable to remove interrupted staging file") from error
        removed.append(entry)
    if removed:
        fsync_parent_directory(root)
    return tuple(sorted(removed, key=lambda item: item.name))


__all__ = [
    "SafeIOError",
    "assert_regular_file",
    "atomic_replace_bytes",
    "atomic_write_text",
    "canonical_json_bytes",
    "canonical_relative_path",
    "cleanup_staging_files",
    "ensure_secure_directory",
    "fsync_parent_directory",
    "metadata_is_reparse_point",
    "publish_create_only_bytes",
    "read_strict_json",
    "read_verified_bytes",
    "resolve_managed_path",
    "sha256_hex",
    "strict_json_loads",
]
