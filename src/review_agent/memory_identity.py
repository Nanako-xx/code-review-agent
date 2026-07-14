from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import hmac
import os
from pathlib import Path
import re
import stat
import sys
from typing import Dict, Mapping, Optional, Tuple, Union

from review_agent.revision import (
    RepositoryIdentity,
    RepositoryLayout,
    RevisionResolver,
    normalize_repository_identity_path,
    normalize_repository_origin,
    sanitize_origin_url,
)


MEMORY_ROOT_ENVIRONMENT_VARIABLE = "REVIEW_AGENT_MEMORY_ROOT"
REPOSITORY_IDENTITY_SCHEMA = "repository_identity_v1"
REPOSITORY_IDENTITY_CORE_SCHEMA = "repository_identity_core_v1"
REPOSITORY_RELINK_SCHEMA = "repository_relink_v1"
WINDOWS_DIRECTORY_PATH_LIMIT = 247
WINDOWS_FILE_PATH_LIMIT = 259

_APPLICATION_DIRECTORY = "code-review-agent"
_MEMORY_DIRECTORY = "memory"
_REPOSITORIES_DIRECTORY = "repositories"
_REPOSITORY_KEY_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400
_SHORT_RANDOM_TOKEN = "r" * 32
_SHA256_PLACEHOLDER = "0" * 64

PathInput = Union[str, os.PathLike]
IdentityInput = Union[
    RepositoryIdentity,
    "RepositoryIdentityCore",
    "RepositoryIdentityDescriptor",
    "VerifiedRepositoryIdentity",
]


class MemoryIdentityError(ValueError):
    """A stable, non-sensitive Memory root or repository identity error."""


class MemoryRootSource(str, Enum):
    CLI_OVERRIDE = "cli_override"
    ENVIRONMENT = "environment"
    PLATFORM_DEFAULT = "platform_default"


class PathBudgetKind(str, Enum):
    DIRECTORY = "directory"
    FILE = "file"


@dataclass(frozen=True)
class ResolvedMemoryRoot:
    path: str
    source: MemoryRootSource

    def __fspath__(self) -> str:
        return self.path


@dataclass(frozen=True)
class RepositoryIdentityCore:
    """Location-independent repository identity authority."""

    normalized_git_common_dir: str
    normalized_origin_url: Optional[str]
    schema: str = REPOSITORY_IDENTITY_CORE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != REPOSITORY_IDENTITY_CORE_SCHEMA:
            raise MemoryIdentityError("unsupported repository identity core schema")
        try:
            normalized_common = normalize_repository_identity_path(
                self.normalized_git_common_dir
            )
        except (OSError, ValueError) as error:
            raise MemoryIdentityError("invalid repository identity core") from error
        normalized_origin = normalize_repository_origin(self.normalized_origin_url)
        if (
            normalized_common != self.normalized_git_common_dir
            or normalized_origin != self.normalized_origin_url
        ):
            raise MemoryIdentityError("repository identity core is not canonical")

    @classmethod
    def from_components(
        cls,
        git_common_dir: PathInput,
        origin_url: Optional[str],
    ) -> "RepositoryIdentityCore":
        try:
            normalized_common = normalize_repository_identity_path(git_common_dir)
        except (OSError, ValueError) as error:
            raise MemoryIdentityError("invalid Git common directory identity") from error
        return cls(
            normalized_git_common_dir=normalized_common,
            normalized_origin_url=normalize_repository_origin(origin_url),
        )

    @property
    def canonical_material(self) -> bytes:
        return (
            f"{self.normalized_git_common_dir}\0"
            f"{self.normalized_origin_url or ''}"
        ).encode("utf-8")

    @property
    def core_hash(self) -> str:
        return hashlib.sha256(self.canonical_material).hexdigest()

    def to_payload(self) -> Dict[str, object]:
        return {
            "schema": self.schema,
            "git_common_dir": self.normalized_git_common_dir,
            "origin_url": self.normalized_origin_url,
            "core_hash": self.core_hash,
        }


@dataclass(frozen=True)
class RepositoryIdentityDescriptor:
    repository_key: str = field(compare=False)
    canonical_path: str = field(compare=False)
    git_common_dir: str = field(compare=False)
    origin_url: Optional[str] = field(compare=False)
    schema: str = field(default=REPOSITORY_IDENTITY_SCHEMA, compare=False)
    _core: RepositoryIdentityCore = field(init=False, repr=False, compare=True)

    def __post_init__(self) -> None:
        if self.schema != REPOSITORY_IDENTITY_SCHEMA:
            raise MemoryIdentityError("unsupported repository identity schema")
        _validate_repository_key(self.repository_key)
        if self.origin_url is not None and not isinstance(self.origin_url, str):
            raise MemoryIdentityError("repository origin metadata is invalid")
        if self.origin_url != sanitize_origin_url(self.origin_url):
            raise MemoryIdentityError("repository origin metadata is not sanitized")

        canonical_path = _canonical_identity_path(
            self.canonical_path,
            field_name="repository canonical path",
        )
        git_common_dir = _canonical_identity_path(
            self.git_common_dir,
            field_name="Git common directory",
        )
        core = RepositoryIdentityCore.from_components(
            git_common_dir,
            self.origin_url,
        )
        if not hmac.compare_digest(self.repository_key, core.core_hash):
            raise MemoryIdentityError(
                "repository key does not match identity core"
            )

        object.__setattr__(self, "canonical_path", canonical_path)
        object.__setattr__(self, "git_common_dir", git_common_dir)
        object.__setattr__(self, "origin_url", core.normalized_origin_url)
        object.__setattr__(self, "_core", core)

    @property
    def core(self) -> RepositoryIdentityCore:
        return self._core

    @property
    def core_hash(self) -> str:
        return self._core.core_hash

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
    ) -> "RepositoryIdentityDescriptor":
        expected = {
            "schema",
            "repository_key",
            "canonical_path",
            "git_common_dir",
            "origin_url",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise MemoryIdentityError("repository identity payload is invalid")
        if not all(
            isinstance(payload[field], str)
            for field in (
                "schema",
                "repository_key",
                "canonical_path",
                "git_common_dir",
            )
        ):
            raise MemoryIdentityError("repository identity payload is invalid")
        origin = payload["origin_url"]
        if origin is not None and not isinstance(origin, str):
            raise MemoryIdentityError("repository identity payload is invalid")
        return cls(
            repository_key=str(payload["repository_key"]),
            canonical_path=str(payload["canonical_path"]),
            git_common_dir=str(payload["git_common_dir"]),
            origin_url=origin,
            schema=str(payload["schema"]),
        )

    def to_payload(self) -> Dict[str, object]:
        return {
            "schema": self.schema,
            "repository_key": self.repository_key,
            "canonical_path": self.canonical_path,
            "git_common_dir": self.git_common_dir,
            "origin_url": self.origin_url,
        }


@dataclass(frozen=True)
class VerifiedRepositoryIdentity:
    """A descriptor whose repository key is verified against its core."""

    descriptor: RepositoryIdentityDescriptor = field(compare=False)
    core: RepositoryIdentityCore = field(init=False, compare=True)

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, RepositoryIdentityDescriptor):
            raise MemoryIdentityError("verified repository identity is invalid")
        if self.descriptor.repository_key != self.descriptor.core_hash:
            raise MemoryIdentityError("verified repository identity is invalid")
        object.__setattr__(self, "core", self.descriptor.core)

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
    ) -> "VerifiedRepositoryIdentity":
        return cls(RepositoryIdentityDescriptor.from_payload(payload))

    @property
    def repository_key(self) -> str:
        return self.descriptor.repository_key

    @property
    def core_hash(self) -> str:
        return self.core.core_hash

    def to_payload(self) -> Dict[str, object]:
        return self.descriptor.to_payload()


@dataclass(frozen=True)
class VerifiedRepositoryLocator:
    """Live worktree locations verified by Git for one identity core."""

    identity: VerifiedRepositoryIdentity
    canonical_worktree_path: str
    git_common_dir: str
    worktree_paths: Tuple[str, ...]
    git_dirs: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.identity, VerifiedRepositoryIdentity):
            raise MemoryIdentityError("verified repository locator is invalid")
        current = _canonical_identity_path(
            self.canonical_worktree_path,
            field_name="repository canonical path",
        )
        common = _canonical_identity_path(
            self.git_common_dir,
            field_name="Git common directory",
        )
        worktrees = _canonical_path_tuple(self.worktree_paths)
        git_dirs = _canonical_path_tuple(self.git_dirs)
        if normalize_repository_identity_path(current) != (
            normalize_repository_identity_path(
                self.identity.descriptor.canonical_path
            )
        ):
            raise MemoryIdentityError("verified repository locator is inconsistent")
        if normalize_repository_identity_path(common) != (
            self.identity.core.normalized_git_common_dir
        ):
            raise MemoryIdentityError("verified repository locator is inconsistent")
        normalized_worktrees = {
            normalize_repository_identity_path(path) for path in worktrees
        }
        normalized_git_dirs = {
            normalize_repository_identity_path(path) for path in git_dirs
        }
        if (
            normalize_repository_identity_path(current) not in normalized_worktrees
            or normalize_repository_identity_path(common) not in normalized_git_dirs
        ):
            raise MemoryIdentityError("verified repository locator is inconsistent")
        object.__setattr__(self, "canonical_worktree_path", current)
        object.__setattr__(self, "git_common_dir", common)
        object.__setattr__(self, "worktree_paths", worktrees)
        object.__setattr__(self, "git_dirs", git_dirs)

    @property
    def protected_paths(self) -> Tuple[str, ...]:
        protected = {self.git_common_dir, *self.git_dirs, *self.worktree_paths}
        for worktree in self.worktree_paths:
            location = Path(worktree)
            protected.add(str(location / ".git"))
            protected.add(str(location / ".review-agent"))
        return _canonical_path_tuple(tuple(protected))


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


@dataclass(frozen=True)
class PathBudgetEntry:
    label: str
    relative_path: str
    kind: PathBudgetKind
    utf16_code_units: int
    limit: int
    within_limit: bool


@dataclass(frozen=True)
class PathBudgetReport:
    platform_name: str
    enforced: bool
    namespace_units: int
    entries: Tuple[PathBudgetEntry, ...]
    directory_limit: int = WINDOWS_DIRECTORY_PATH_LIMIT
    file_limit: int = WINDOWS_FILE_PATH_LIMIT

    @property
    def maximum_directory_units(self) -> int:
        return max(
            entry.utf16_code_units
            for entry in self.entries
            if entry.kind is PathBudgetKind.DIRECTORY
        )

    @property
    def maximum_file_units(self) -> int:
        return max(
            entry.utf16_code_units
            for entry in self.entries
            if entry.kind is PathBudgetKind.FILE
        )

    @property
    def windows_compatible(self) -> bool:
        return all(entry.within_limit for entry in self.entries)

    @property
    def within_budget(self) -> bool:
        return self.windows_compatible if self.enforced else True


@dataclass(frozen=True)
class RepositoryMemoryNamespacePlan:
    namespace: RepositoryMemoryNamespace
    locator: VerifiedRepositoryLocator
    path_budget: PathBudgetReport
    platform_name: str

    def __post_init__(self) -> None:
        if self.namespace.repository_key != self.locator.identity.repository_key:
            raise MemoryIdentityError("memory namespace plan identity is inconsistent")
        if not self.path_budget.within_budget:
            raise MemoryIdentityError(
                "memory namespace exceeds the bounded Windows path policy"
            )

    @property
    def repository_key(self) -> str:
        return self.namespace.repository_key

    @property
    def memory_root(self) -> str:
        return self.namespace.memory_root

    @property
    def namespace_path(self) -> str:
        return self.namespace.namespace_path

    @property
    def metadata(self) -> RepositoryIdentityDescriptor:
        return self.namespace.metadata


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


class RepositoryMemoryNamespacePlanner:
    def __init__(
        self,
        revision_resolver: Optional[RevisionResolver] = None,
    ) -> None:
        self._revision_resolver = revision_resolver or RevisionResolver()

    def plan(
        self,
        identity: IdentityInput,
        memory_root: Optional[Union[PathInput, ResolvedMemoryRoot]] = None,
        *,
        env: Optional[Mapping[str, str]] = None,
        platform_name: Optional[str] = None,
        home: Optional[PathInput] = None,
    ) -> RepositoryMemoryNamespacePlan:
        selected_platform = sys.platform if platform_name is None else platform_name
        if memory_root is None:
            resolution = MemoryRootResolver().resolve(
                env=env,
                platform_name=selected_platform,
                home=home,
                create=False,
            )
            root = Path(resolution.path)
        else:
            root_value = (
                memory_root.path
                if isinstance(memory_root, ResolvedMemoryRoot)
                else memory_root
            )
            root = _canonicalize_root(
                root_value,
                create=False,
                require_directory=False,
            )

        locator = verify_repository_locator(
            identity,
            revision_resolver=self._revision_resolver,
        )
        _reject_protected_path_overlap(root, locator.protected_paths)
        if root.exists() and not root.is_dir():
            raise MemoryIdentityError("memory root exists but is not a directory")

        namespace_path = repository_namespace_path(
            root,
            locator.identity.repository_key,
        )
        namespace = RepositoryMemoryNamespace(
            repository_key=locator.identity.repository_key,
            memory_root=str(root),
            namespace_path=str(namespace_path),
            metadata=locator.identity.descriptor,
        )
        path_budget = build_path_budget_report(
            namespace_path,
            platform_name=selected_platform,
        )
        if not path_budget.within_budget:
            raise MemoryIdentityError(
                "memory namespace exceeds the bounded Windows path policy"
            )
        return RepositoryMemoryNamespacePlan(
            namespace=namespace,
            locator=locator,
            path_budget=path_budget,
            platform_name=selected_platform,
        )


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


def build_repository_identity_core(identity: IdentityInput) -> RepositoryIdentityCore:
    if isinstance(identity, RepositoryIdentityCore):
        return identity
    if isinstance(identity, VerifiedRepositoryIdentity):
        return identity.core
    if isinstance(identity, RepositoryIdentityDescriptor):
        return identity.core
    if not isinstance(identity, RepositoryIdentity):
        raise MemoryIdentityError("repository identity is invalid")
    return RepositoryIdentityCore.from_components(
        identity.git_common_dir,
        identity.origin_url,
    )


def repository_identity_core_hash(identity: IdentityInput) -> str:
    return build_repository_identity_core(identity).core_hash


def canonical_repository_identity_core_hash(identity: IdentityInput) -> str:
    return repository_identity_core_hash(identity)


def repository_key(identity: IdentityInput) -> str:
    """Create a clone-local key while sharing linked Git worktrees."""

    return repository_identity_core_hash(identity)


def build_repository_identity_descriptor(
    identity: RepositoryIdentity,
) -> RepositoryIdentityDescriptor:
    core = build_repository_identity_core(identity)
    return RepositoryIdentityDescriptor(
        repository_key=core.core_hash,
        canonical_path=_canonical_identity_path(
            identity.canonical_path,
            field_name="repository canonical path",
        ),
        git_common_dir=_canonical_identity_path(
            identity.git_common_dir,
            field_name="Git common directory",
        ),
        origin_url=sanitize_origin_url(identity.origin_url),
    )


def hydrate_repository_identity_descriptor(
    payload: Mapping[str, object],
) -> RepositoryIdentityDescriptor:
    return RepositoryIdentityDescriptor.from_payload(payload)


def verify_repository_identity(
    identity: IdentityInput,
    *,
    revision_resolver: Optional[RevisionResolver] = None,
) -> VerifiedRepositoryIdentity:
    return verify_repository_locator(
        identity,
        revision_resolver=revision_resolver,
    ).identity


def verify_repository_locator(
    identity: IdentityInput,
    *,
    revision_resolver: Optional[RevisionResolver] = None,
) -> VerifiedRepositoryLocator:
    claimed = _coerce_identity_descriptor(identity)
    resolver = revision_resolver or RevisionResolver()
    try:
        observed = build_repository_identity_descriptor(
            resolver.repository_identity(Path(claimed.canonical_path))
        )
        layout = resolver.repository_layout(Path(observed.canonical_path))
    except (OSError, RuntimeError, ValueError) as error:
        raise MemoryIdentityError("unable to verify repository locator") from error

    if (
        claimed.core != observed.core
        or normalize_repository_identity_path(claimed.canonical_path)
        != normalize_repository_identity_path(observed.canonical_path)
    ):
        raise MemoryIdentityError("repository locator does not match identity core")
    if normalize_repository_identity_path(layout.git_common_dir) != (
        observed.core.normalized_git_common_dir
    ):
        raise MemoryIdentityError("repository locator does not match identity core")

    verified = VerifiedRepositoryIdentity(observed)
    return VerifiedRepositoryLocator(
        identity=verified,
        canonical_worktree_path=observed.canonical_path,
        git_common_dir=layout.git_common_dir,
        worktree_paths=layout.worktree_paths,
        git_dirs=layout.git_dirs,
    )


def repository_namespace_path(
    memory_root: Union[PathInput, ResolvedMemoryRoot],
    key: str,
) -> Path:
    """Return a contained namespace path without creating repository storage."""

    _validate_repository_key(key)
    root_value = memory_root.path if isinstance(memory_root, ResolvedMemoryRoot) else memory_root
    root = _canonicalize_root(root_value, create=False)
    candidate = root / _REPOSITORIES_DIRECTORY / key
    _assert_no_symlink_or_reparse_components(candidate)
    for component in (root / _REPOSITORIES_DIRECTORY, candidate):
        if component.exists() and not component.is_dir():
            raise MemoryIdentityError("memory namespace component is not a directory")
    canonical_candidate = Path(os.path.normpath(str(candidate)))
    if not _path_is_within(canonical_candidate, root):
        raise MemoryIdentityError("memory namespace escapes the configured root")
    return canonical_candidate


def build_repository_memory_namespace(
    identity: RepositoryIdentity,
    memory_root: Union[PathInput, ResolvedMemoryRoot],
) -> RepositoryMemoryNamespace:
    """Direct, unbound namespace construction retained for existing callers."""

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


def plan_repository_memory_namespace(
    identity: IdentityInput,
    memory_root: Optional[Union[PathInput, ResolvedMemoryRoot]] = None,
    *,
    env: Optional[Mapping[str, str]] = None,
    platform_name: Optional[str] = None,
    home: Optional[PathInput] = None,
    revision_resolver: Optional[RevisionResolver] = None,
) -> RepositoryMemoryNamespacePlan:
    return RepositoryMemoryNamespacePlanner(revision_resolver).plan(
        identity,
        memory_root,
        env=env,
        platform_name=platform_name,
        home=home,
    )


def materialize_repository_memory_namespace(
    plan: RepositoryMemoryNamespacePlan,
    *,
    revision_resolver: Optional[RevisionResolver] = None,
) -> RepositoryMemoryNamespace:
    if not isinstance(plan, RepositoryMemoryNamespacePlan):
        raise MemoryIdentityError("memory namespace plan is invalid")

    current_locator = verify_repository_locator(
        plan.locator.identity,
        revision_resolver=revision_resolver,
    )
    root = _canonicalize_root(
        plan.memory_root,
        create=False,
        require_directory=False,
    )
    _reject_protected_path_overlap(root, current_locator.protected_paths)
    if root.exists() and not root.is_dir():
        raise MemoryIdentityError("memory root exists but is not a directory")
    namespace = repository_namespace_path(root, plan.repository_key)
    if _normalized_path(namespace) != _normalized_path(Path(plan.namespace_path)):
        raise MemoryIdentityError("memory namespace plan path changed")
    report = build_path_budget_report(
        namespace,
        platform_name=plan.platform_name,
    )
    if not report.within_budget:
        raise MemoryIdentityError(
            "memory namespace exceeds the bounded Windows path policy"
        )

    try:
        root.mkdir(parents=True, exist_ok=True)
        _assert_materialized_directory(root)
        repositories = root / _REPOSITORIES_DIRECTORY
        repositories.mkdir(exist_ok=True)
        _assert_materialized_directory(repositories)
        namespace.mkdir(exist_ok=True)
        _assert_materialized_directory(namespace)
    except MemoryIdentityError:
        raise
    except OSError as error:
        raise MemoryIdentityError("memory namespace cannot be materialized safely") from error
    return plan.namespace


def build_path_budget_report(
    namespace_path: PathInput,
    *,
    platform_name: Optional[str] = None,
) -> PathBudgetReport:
    namespace_text = str(namespace_path)
    namespace = Path(namespace_text)
    if not namespace_text or "\0" in namespace_text or not namespace.is_absolute():
        raise MemoryIdentityError("memory namespace path must be absolute")
    selected_platform = sys.platform if platform_name is None else platform_name
    enforced = selected_platform.casefold().startswith("win")

    token = _SHORT_RANDOM_TOKEN
    digest = _SHA256_PLACEHOLDER
    directory_specs = (
        ("memory_root", namespace.parent.parent, "../.."),
        ("repositories", namespace.parent, ".."),
        ("namespace", namespace, "."),
        ("blob_root", namespace / "blobs", "blobs"),
        ("blob_sha256", namespace / "blobs" / "sha256", "blobs/sha256"),
        ("blob_shard", namespace / "blobs" / "sha256" / "00", "blobs/sha256/00"),
        ("blob_temp_root", namespace / "blobs" / ".tmp", "blobs/.tmp"),
    )
    file_specs = (
        ("database", namespace / "memory.sqlite3", "memory.sqlite3"),
        ("database_wal", namespace / "memory.sqlite3-wal", "memory.sqlite3-wal"),
        ("database_shm", namespace / "memory.sqlite3-shm", "memory.sqlite3-shm"),
        ("database_journal", namespace / "memory.sqlite3-journal", "memory.sqlite3-journal"),
        ("repository_lock", namespace / ".memory-store.lock", ".memory-store.lock"),
        ("blob_lock", namespace / "blobs" / ".blob-store.lock", "blobs/.blob-store.lock"),
        ("final_blob", namespace / "blobs" / "sha256" / "00" / digest, f"blobs/sha256/00/{digest}"),
        ("blob_temp", namespace / "blobs" / ".tmp" / f".tmp-{token}.tmp", f"blobs/.tmp/.tmp-{token}.tmp"),
        ("blob_repair", namespace / "blobs" / ".tmp" / f".repair-{token}.tmp", f"blobs/.tmp/.repair-{token}.tmp"),
        ("migration_database", namespace / f".{token}.migration.sqlite3", f".{token}.migration.sqlite3"),
        ("migration_wal", namespace / f".{token}.migration.sqlite3-wal", f".{token}.migration.sqlite3-wal"),
        ("migration_shm", namespace / f".{token}.migration.sqlite3-shm", f".{token}.migration.sqlite3-shm"),
    )

    entries = []
    for label, path, relative in directory_specs:
        units = _utf16_code_units(str(path))
        entries.append(
            PathBudgetEntry(
                label=label,
                relative_path=relative,
                kind=PathBudgetKind.DIRECTORY,
                utf16_code_units=units,
                limit=WINDOWS_DIRECTORY_PATH_LIMIT,
                within_limit=units <= WINDOWS_DIRECTORY_PATH_LIMIT,
            )
        )
    for label, path, relative in file_specs:
        units = _utf16_code_units(str(path))
        entries.append(
            PathBudgetEntry(
                label=label,
                relative_path=relative,
                kind=PathBudgetKind.FILE,
                utf16_code_units=units,
                limit=WINDOWS_FILE_PATH_LIMIT,
                within_limit=units <= WINDOWS_FILE_PATH_LIMIT,
            )
        )
    return PathBudgetReport(
        platform_name=selected_platform,
        enforced=enforced,
        namespace_units=_utf16_code_units(str(namespace)),
        entries=tuple(entries),
    )


def build_relink_descriptor(
    old_identity: Union[RepositoryIdentityDescriptor, VerifiedRepositoryIdentity],
    new_identity: Union[RepositoryIdentityDescriptor, VerifiedRepositoryIdentity],
) -> RepositoryRelinkDescriptor:
    """Describe an explicit migration; no origin-only lookup is performed."""

    old_descriptor = (
        old_identity.descriptor
        if isinstance(old_identity, VerifiedRepositoryIdentity)
        else old_identity
    )
    new_descriptor = (
        new_identity.descriptor
        if isinstance(new_identity, VerifiedRepositoryIdentity)
        else new_identity
    )
    return RepositoryRelinkDescriptor(
        old_identity=old_descriptor,
        new_identity=new_descriptor,
    )


def _coerce_identity_descriptor(identity: IdentityInput) -> RepositoryIdentityDescriptor:
    if isinstance(identity, VerifiedRepositoryIdentity):
        return identity.descriptor
    if isinstance(identity, RepositoryIdentityDescriptor):
        return identity
    if isinstance(identity, RepositoryIdentity):
        return build_repository_identity_descriptor(identity)
    raise MemoryIdentityError("repository identity is invalid")


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
        return home_path / "Library" / "Application Support" / _APPLICATION_DIRECTORY / _MEMORY_DIRECTORY

    xdg_state_home = _optional_absolute_platform_base(env.get("XDG_STATE_HOME"))
    base = xdg_state_home if xdg_state_home is not None else home_path / ".local" / "state"
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


def _canonicalize_root(
    value: PathInput,
    *,
    create: bool,
    require_directory: bool = True,
) -> Path:
    text = str(value)
    candidate = Path(text)
    if (
        not text.strip()
        or "\0" in text
        or not candidate.is_absolute()
        or ".." in candidate.parts
    ):
        raise MemoryIdentityError("memory root must be a non-empty absolute path")
    _assert_no_symlink_or_reparse_components(candidate)
    canonical = Path(os.path.normpath(str(candidate)))
    if require_directory and canonical.exists() and not canonical.is_dir():
        raise MemoryIdentityError("memory root exists but is not a directory")
    if create:
        try:
            canonical.mkdir(parents=True, exist_ok=True)
            _assert_materialized_directory(canonical)
        except MemoryIdentityError:
            raise
        except OSError as error:
            raise MemoryIdentityError(
                "memory root cannot safely create its parent directories"
            ) from error
    return canonical


def _canonical_identity_path(value: PathInput, *, field_name: str) -> str:
    text = str(value)
    candidate = Path(text)
    if not text or "\0" in text or not candidate.is_absolute():
        raise MemoryIdentityError(f"{field_name} must be absolute")
    try:
        return str(candidate.resolve(strict=False))
    except OSError as error:
        raise MemoryIdentityError(f"unable to canonicalize {field_name}") from error


def _canonical_path_tuple(values: Tuple[str, ...]) -> Tuple[str, ...]:
    canonical: Dict[str, str] = {}
    for value in values:
        path = _canonical_identity_path(value, field_name="repository locator path")
        canonical[normalize_repository_identity_path(path)] = path
    return tuple(canonical[key] for key in sorted(canonical))


def _validate_repository_key(key: str) -> None:
    if not isinstance(key, str) or _REPOSITORY_KEY_PATTERN.fullmatch(key) is None:
        raise MemoryIdentityError(
            "repository key must be 64 lowercase hexadecimal characters"
        )


def _assert_no_symlink_or_reparse_components(candidate: Path) -> None:
    if not candidate.is_absolute():
        raise MemoryIdentityError("memory path must be absolute")
    current = Path(candidate.anchor)
    parts = candidate.parts[1:] if candidate.anchor else candidate.parts
    for part in parts:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            break
        except OSError as error:
            raise MemoryIdentityError("unable to inspect memory path safety") from error
        attributes = getattr(metadata, "st_file_attributes", 0)
        if stat.S_ISLNK(metadata.st_mode) or (
            attributes & _FILE_ATTRIBUTE_REPARSE_POINT
        ):
            raise MemoryIdentityError(
                "memory path cannot traverse a symbolic link or reparse point"
            )


def _assert_materialized_directory(path: Path) -> None:
    _assert_no_symlink_or_reparse_components(path)
    if not path.is_dir():
        raise MemoryIdentityError("memory namespace component is not a directory")


def _reject_protected_path_overlap(root: Path, protected_paths: Tuple[str, ...]) -> None:
    for protected in protected_paths:
        if _paths_overlap(root, Path(protected)):
            raise MemoryIdentityError(
                "memory root overlaps protected repository paths"
            )


def _paths_overlap(left: Path, right: Path) -> bool:
    left_value = _normalized_path(left)
    right_value = _normalized_path(right)
    try:
        common = os.path.commonpath((left_value, right_value))
    except ValueError:
        return False
    return common == left_value or common == right_value


def _path_is_within(candidate: Path, root: Path) -> bool:
    candidate_value = _normalized_path(candidate)
    root_value = _normalized_path(root)
    try:
        return os.path.commonpath((candidate_value, root_value)) == root_value
    except ValueError:
        return False


def _normalized_path(path: Path) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


def _utf16_code_units(value: str) -> int:
    try:
        return len(value.encode("utf-16-le")) // 2
    except UnicodeEncodeError as error:
        raise MemoryIdentityError("memory namespace contains unsupported path text") from error


__all__ = [
    "MEMORY_ROOT_ENVIRONMENT_VARIABLE",
    "REPOSITORY_IDENTITY_CORE_SCHEMA",
    "REPOSITORY_IDENTITY_SCHEMA",
    "WINDOWS_DIRECTORY_PATH_LIMIT",
    "WINDOWS_FILE_PATH_LIMIT",
    "MemoryIdentityError",
    "MemoryRootResolver",
    "MemoryRootSource",
    "PathBudgetEntry",
    "PathBudgetKind",
    "PathBudgetReport",
    "RepositoryIdentityCore",
    "RepositoryIdentityDescriptor",
    "RepositoryMemoryNamespace",
    "RepositoryMemoryNamespacePlan",
    "RepositoryMemoryNamespacePlanner",
    "RepositoryRelinkDescriptor",
    "ResolvedMemoryRoot",
    "VerifiedRepositoryIdentity",
    "VerifiedRepositoryLocator",
    "build_path_budget_report",
    "build_relink_descriptor",
    "build_repository_identity_core",
    "build_repository_identity_descriptor",
    "build_repository_memory_namespace",
    "canonical_repository_identity_core_hash",
    "hydrate_repository_identity_descriptor",
    "materialize_repository_memory_namespace",
    "plan_repository_memory_namespace",
    "repository_identity_core_hash",
    "repository_key",
    "repository_namespace_path",
    "resolve_memory_root",
    "verify_repository_identity",
    "verify_repository_locator",
]
