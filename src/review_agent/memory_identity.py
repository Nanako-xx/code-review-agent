from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import os
from pathlib import Path
import re
import sys
from typing import Dict, Mapping, Optional, Union

from review_agent.revision import (
    RepositoryIdentity,
    normalize_repository_identity_path,
    normalize_repository_origin,
    sanitize_origin_url,
)


MEMORY_ROOT_ENVIRONMENT_VARIABLE = "REVIEW_AGENT_MEMORY_ROOT"
REPOSITORY_IDENTITY_SCHEMA = "repository_identity_v1"
REPOSITORY_RELINK_SCHEMA = "repository_relink_v1"
_APPLICATION_DIRECTORY = "code-review-agent"
_MEMORY_DIRECTORY = "memory"
_REPOSITORIES_DIRECTORY = "repositories"
_REPOSITORY_KEY_PATTERN = re.compile(r"^[0-9a-f]{64}$")

PathInput = Union[str, os.PathLike]


class MemoryIdentityError(ValueError):
    """A stable, non-sensitive Memory root or repository identity error."""


class MemoryRootSource(str, Enum):
    CLI_OVERRIDE = "cli_override"
    ENVIRONMENT = "environment"
    PLATFORM_DEFAULT = "platform_default"


@dataclass(frozen=True)
class ResolvedMemoryRoot:
    path: str
    source: MemoryRootSource

    def __fspath__(self) -> str:
        return self.path


@dataclass(frozen=True)
class RepositoryIdentityDescriptor:
    repository_key: str
    canonical_path: str
    git_common_dir: str
    origin_url: Optional[str]
    schema: str = REPOSITORY_IDENTITY_SCHEMA

    def __post_init__(self) -> None:
        _validate_repository_key(self.repository_key)
        if self.schema != REPOSITORY_IDENTITY_SCHEMA:
            raise MemoryIdentityError("unsupported repository identity schema")
        if not Path(self.canonical_path).is_absolute():
            raise MemoryIdentityError("repository canonical path must be absolute")
        if not Path(self.git_common_dir).is_absolute():
            raise MemoryIdentityError("Git common directory must be absolute")
        if self.origin_url != sanitize_origin_url(self.origin_url):
            raise MemoryIdentityError("repository origin metadata is not sanitized")

    def to_payload(self) -> Dict[str, object]:
        return {
            "schema": self.schema,
            "repository_key": self.repository_key,
            "canonical_path": self.canonical_path,
            "git_common_dir": self.git_common_dir,
            "origin_url": self.origin_url,
        }


@dataclass(frozen=True)
class RepositoryMemoryNamespace:
    repository_key: str
    memory_root: str
    namespace_path: str
    metadata: RepositoryIdentityDescriptor

    def __post_init__(self) -> None:
        _validate_repository_key(self.repository_key)
        if self.metadata.repository_key != self.repository_key:
            raise MemoryIdentityError("namespace and metadata repository keys differ")
        root = Path(self.memory_root)
        namespace = Path(self.namespace_path)
        if not root.is_absolute() or not namespace.is_absolute():
            raise MemoryIdentityError("memory namespace paths must be absolute")
        if not _path_is_within(namespace, root):
            raise MemoryIdentityError("memory namespace escapes the configured root")


@dataclass(frozen=True)
class RepositoryRelinkDescriptor:
    old_identity: RepositoryIdentityDescriptor
    new_identity: RepositoryIdentityDescriptor
    operation: str = "explicit_relink"
    schema: str = REPOSITORY_RELINK_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != REPOSITORY_RELINK_SCHEMA:
            raise MemoryIdentityError("unsupported repository relink schema")
        if self.operation != "explicit_relink":
            raise MemoryIdentityError("repository relink must be explicit")
        if self.old_identity.repository_key == self.new_identity.repository_key:
            raise MemoryIdentityError(
                "repository relink requires different repository keys"
            )

    def to_payload(self) -> Dict[str, object]:
        return {
            "schema": self.schema,
            "operation": self.operation,
            "old_identity": self.old_identity.to_payload(),
            "new_identity": self.new_identity.to_payload(),
        }


class MemoryRootResolver:
    def resolve(
        self,
        cli_override: Optional[PathInput] = None,
        *,
        env: Optional[Mapping[str, str]] = None,
        platform_name: Optional[str] = None,
        home: Optional[PathInput] = None,
        create: bool = True,
    ) -> ResolvedMemoryRoot:
        environment = os.environ if env is None else env
        if cli_override is not None:
            requested_root = cli_override
            source = MemoryRootSource.CLI_OVERRIDE
        elif MEMORY_ROOT_ENVIRONMENT_VARIABLE in environment:
            requested_root = environment[MEMORY_ROOT_ENVIRONMENT_VARIABLE]
            source = MemoryRootSource.ENVIRONMENT
        else:
            requested_root = _platform_default_root(
                environment,
                platform_name=sys.platform if platform_name is None else platform_name,
                home=Path.home() if home is None else Path(home),
            )
            source = MemoryRootSource.PLATFORM_DEFAULT

        canonical_root = _canonicalize_root(requested_root, create=create)
        return ResolvedMemoryRoot(path=str(canonical_root), source=source)


def resolve_memory_root(
    cli_override: Optional[PathInput] = None,
    *,
    env: Optional[Mapping[str, str]] = None,
    platform_name: Optional[str] = None,
    home: Optional[PathInput] = None,
    create: bool = True,
) -> Path:
    """Resolve and optionally create the canonical application Memory root."""

    resolution = MemoryRootResolver().resolve(
        cli_override,
        env=env,
        platform_name=platform_name,
        home=home,
        create=create,
    )
    return Path(resolution.path)


def repository_key(identity: RepositoryIdentity) -> str:
    """Create a clone-local key while sharing linked Git worktrees."""

    try:
        common_dir = normalize_repository_identity_path(identity.git_common_dir)
    except (OSError, ValueError) as error:
        raise MemoryIdentityError("invalid Git common directory identity") from error
    normalized_origin = normalize_repository_origin(identity.origin_url) or ""
    material = f"{common_dir}\0{normalized_origin}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def build_repository_identity_descriptor(
    identity: RepositoryIdentity,
) -> RepositoryIdentityDescriptor:
    key = repository_key(identity)
    canonical_path = _canonical_identity_path(
        identity.canonical_path,
        field_name="repository canonical path",
    )
    git_common_dir = _canonical_identity_path(
        identity.git_common_dir,
        field_name="Git common directory",
    )
    return RepositoryIdentityDescriptor(
        repository_key=key,
        canonical_path=canonical_path,
        git_common_dir=git_common_dir,
        origin_url=sanitize_origin_url(identity.origin_url),
    )


def repository_namespace_path(
    memory_root: Union[PathInput, ResolvedMemoryRoot],
    key: str,
) -> Path:
    """Return a contained namespace path without creating repository storage."""

    _validate_repository_key(key)
    root_value: PathInput
    if isinstance(memory_root, ResolvedMemoryRoot):
        root_value = memory_root.path
    else:
        root_value = memory_root
    root = _canonicalize_root(root_value, create=False)

    candidate = root / _REPOSITORIES_DIRECTORY / key
    current = root
    for component in (_REPOSITORIES_DIRECTORY, key):
        current = current / component
        if current.is_symlink():
            raise MemoryIdentityError(
                "memory namespace cannot traverse a symbolic link"
            )
        if current.exists() and not current.is_dir():
            raise MemoryIdentityError("memory namespace component is not a directory")

    try:
        canonical_candidate = candidate.resolve(strict=False)
    except OSError as error:
        raise MemoryIdentityError("unable to canonicalize memory namespace") from error
    if not _path_is_within(canonical_candidate, root):
        raise MemoryIdentityError("memory namespace escapes the configured root")
    return canonical_candidate


def build_repository_memory_namespace(
    identity: RepositoryIdentity,
    memory_root: Union[PathInput, ResolvedMemoryRoot],
) -> RepositoryMemoryNamespace:
    metadata = build_repository_identity_descriptor(identity)
    namespace = repository_namespace_path(memory_root, metadata.repository_key)
    root = _canonicalize_root(
        memory_root.path if isinstance(memory_root, ResolvedMemoryRoot) else memory_root,
        create=False,
    )
    return RepositoryMemoryNamespace(
        repository_key=metadata.repository_key,
        memory_root=str(root),
        namespace_path=str(namespace),
        metadata=metadata,
    )


def build_relink_descriptor(
    old_identity: RepositoryIdentityDescriptor,
    new_identity: RepositoryIdentityDescriptor,
) -> RepositoryRelinkDescriptor:
    """Describe an explicit migration; no origin-only lookup is performed."""

    return RepositoryRelinkDescriptor(
        old_identity=old_identity,
        new_identity=new_identity,
    )


def _platform_default_root(
    env: Mapping[str, str],
    *,
    platform_name: str,
    home: Path,
) -> Path:
    home_path = _absolute_platform_base(home, "home directory")
    normalized_platform = platform_name.casefold()
    if normalized_platform.startswith("win"):
        local_app_data = _optional_absolute_platform_base(env.get("LOCALAPPDATA"))
        base = (
            local_app_data
            if local_app_data is not None
            else home_path / "AppData" / "Local"
        )
        return base / _APPLICATION_DIRECTORY / _MEMORY_DIRECTORY
    if normalized_platform == "darwin":
        return (
            home_path
            / "Library"
            / "Application Support"
            / _APPLICATION_DIRECTORY
            / _MEMORY_DIRECTORY
        )

    xdg_state_home = _optional_absolute_platform_base(env.get("XDG_STATE_HOME"))
    base = (
        xdg_state_home
        if xdg_state_home is not None
        else home_path / ".local" / "state"
    )
    return base / _APPLICATION_DIRECTORY / _MEMORY_DIRECTORY


def _optional_absolute_platform_base(value: Optional[str]) -> Optional[Path]:
    if value is None or not value.strip() or "\0" in value:
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        return None
    return candidate


def _absolute_platform_base(value: PathInput, label: str) -> Path:
    text = str(value)
    candidate = Path(text)
    if not text.strip() or "\0" in text or not candidate.is_absolute():
        raise MemoryIdentityError(f"{label} must be an absolute path")
    return candidate


def _canonicalize_root(value: PathInput, *, create: bool) -> Path:
    text = str(value)
    candidate = Path(text)
    if (
        not text.strip()
        or "\0" in text
        or not candidate.is_absolute()
        or ".." in candidate.parts
    ):
        raise MemoryIdentityError("memory root must be a non-empty absolute path")

    try:
        canonical = candidate.resolve(strict=False)
    except OSError as error:
        raise MemoryIdentityError("unable to canonicalize memory root") from error
    if canonical.exists() and not canonical.is_dir():
        raise MemoryIdentityError("memory root exists but is not a directory")
    if create:
        try:
            canonical.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise MemoryIdentityError(
                "memory root cannot safely create its parent directories"
            ) from error
        try:
            canonical = canonical.resolve(strict=True)
        except OSError as error:
            raise MemoryIdentityError("unable to canonicalize memory root") from error
        if not canonical.is_dir():
            raise MemoryIdentityError("memory root exists but is not a directory")
    if not canonical.is_absolute():
        raise MemoryIdentityError("memory root must resolve to an absolute path")
    return canonical


def _canonical_identity_path(value: str, *, field_name: str) -> str:
    text = str(value)
    candidate = Path(text)
    if not text or "\0" in text or not candidate.is_absolute():
        raise MemoryIdentityError(f"{field_name} must be absolute")
    try:
        return str(candidate.resolve(strict=False))
    except OSError as error:
        raise MemoryIdentityError(f"unable to canonicalize {field_name}") from error


def _validate_repository_key(key: str) -> None:
    if _REPOSITORY_KEY_PATTERN.fullmatch(key) is None:
        raise MemoryIdentityError(
            "repository key must be 64 lowercase hexadecimal characters"
        )


def _path_is_within(candidate: Path, root: Path) -> bool:
    try:
        canonical_candidate = os.path.normcase(str(candidate.resolve(strict=False)))
        canonical_root = os.path.normcase(str(root.resolve(strict=False)))
        return os.path.commonpath((canonical_candidate, canonical_root)) == canonical_root
    except (OSError, ValueError):
        return False


__all__ = [
    "MEMORY_ROOT_ENVIRONMENT_VARIABLE",
    "MemoryIdentityError",
    "MemoryRootResolver",
    "MemoryRootSource",
    "RepositoryIdentityDescriptor",
    "RepositoryMemoryNamespace",
    "RepositoryRelinkDescriptor",
    "ResolvedMemoryRoot",
    "build_relink_descriptor",
    "build_repository_identity_descriptor",
    "build_repository_memory_namespace",
    "repository_key",
    "repository_namespace_path",
    "resolve_memory_root",
]
