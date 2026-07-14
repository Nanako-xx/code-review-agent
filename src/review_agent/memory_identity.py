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

if os.name == "nt":
    import ctypes
    from ctypes import wintypes

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
_WINDOWS_RESERVED_COMPONENTS = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{index}" for index in range(1, 10)}
    | {f"lpt{index}" for index in range(1, 10)}
)

if os.name == "nt":
    _WIN_FILE_ATTRIBUTE_DIRECTORY = 0x10
    _WIN_FILE_ATTRIBUTE_NORMAL = 0x80
    _WIN_FILE_SHARE_ALL = 0x1 | 0x2 | 0x4
    _WIN_FILE_LIST_DIRECTORY = 0x1
    _WIN_FILE_TRAVERSE = 0x20
    _WIN_FILE_READ_ATTRIBUTES = 0x80
    _WIN_SYNCHRONIZE = 0x00100000
    _WIN_OPEN_EXISTING = 3
    _WIN_FILE_OPEN = 1
    _WIN_FILE_OPEN_IF = 3
    _WIN_FILE_DIRECTORY_FILE = 0x1
    _WIN_FILE_SYNCHRONOUS_IO_NONALERT = 0x20
    _WIN_FILE_OPEN_FOR_BACKUP_INTENT = 0x4000
    _WIN_FILE_OPEN_REPARSE_POINT = 0x00200000
    _WIN_FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
    _WIN_OBJ_CASE_INSENSITIVE = 0x40
    _WIN_ERROR_FILE_NOT_FOUND = 2
    _WIN_ERROR_PATH_NOT_FOUND = 3

    class _WindowsUnicodeString(ctypes.Structure):
        _fields_ = [
            ("Length", wintypes.USHORT),
            ("MaximumLength", wintypes.USHORT),
            ("Buffer", wintypes.LPWSTR),
        ]

    class _WindowsObjectAttributes(ctypes.Structure):
        _fields_ = [
            ("Length", wintypes.ULONG),
            ("RootDirectory", wintypes.HANDLE),
            ("ObjectName", ctypes.POINTER(_WindowsUnicodeString)),
            ("Attributes", wintypes.ULONG),
            ("SecurityDescriptor", wintypes.LPVOID),
            ("SecurityQualityOfService", wintypes.LPVOID),
        ]

    class _WindowsIoStatusBlock(ctypes.Structure):
        _fields_ = [
            ("Status", wintypes.LPVOID),
            ("Information", ctypes.c_size_t),
        ]

    class _WindowsByHandleFileInformation(ctypes.Structure):
        _fields_ = [
            ("dwFileAttributes", wintypes.DWORD),
            ("ftCreationTime", wintypes.FILETIME),
            ("ftLastAccessTime", wintypes.FILETIME),
            ("ftLastWriteTime", wintypes.FILETIME),
            ("dwVolumeSerialNumber", wintypes.DWORD),
            ("nFileSizeHigh", wintypes.DWORD),
            ("nFileSizeLow", wintypes.DWORD),
            ("nNumberOfLinks", wintypes.DWORD),
            ("nFileIndexHigh", wintypes.DWORD),
            ("nFileIndexLow", wintypes.DWORD),
        ]

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
class _PhysicalPathAncestor:
    identity: Tuple[int, int]
    suffix_parts: Tuple[str, ...]


@dataclass(frozen=True)
class _PhysicalPathDescriptor:
    ancestors: Tuple[_PhysicalPathAncestor, ...]
    missing_parts: Tuple[str, ...]

    @property
    def deepest_identity(self) -> Tuple[int, int]:
        return self.ancestors[0].identity


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
        except (OSError, ValueError):
            raise MemoryIdentityError("invalid repository identity core") from None
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
        except (OSError, ValueError):
            raise MemoryIdentityError("invalid Git common directory identity") from None
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
        if not isinstance(self.metadata, RepositoryIdentityDescriptor):
            raise MemoryIdentityError("namespace metadata is invalid")
        _validate_repository_key(self.repository_key)
        if self.metadata.repository_key != self.repository_key:
            raise MemoryIdentityError("namespace and metadata repository keys differ")
        root, namespace = _validated_repository_namespace_paths(
            self.repository_key,
            self.memory_root,
            self.namespace_path,
        )
        object.__setattr__(self, "memory_root", str(root))
        object.__setattr__(self, "namespace_path", str(namespace))


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
    except (OSError, RuntimeError, ValueError):
        raise MemoryIdentityError("unable to verify repository locator") from None

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


def validate_repository_memory_namespace(
    namespace: RepositoryMemoryNamespace,
) -> RepositoryMemoryNamespace:
    """Revalidate a namespace immediately before filesystem authority use."""

    if not isinstance(namespace, RepositoryMemoryNamespace):
        raise MemoryIdentityError("repository memory namespace is invalid")
    root, path = _validated_repository_namespace_paths(
        namespace.repository_key,
        namespace.memory_root,
        namespace.namespace_path,
    )
    if str(root) != namespace.memory_root or str(path) != namespace.namespace_path:
        raise MemoryIdentityError("repository memory namespace is not canonical")
    if namespace.metadata.repository_key != namespace.repository_key:
        raise MemoryIdentityError("namespace and metadata repository keys differ")
    return namespace


def _validated_repository_namespace_paths(
    key: str,
    memory_root: PathInput,
    namespace_path: PathInput,
) -> Tuple[Path, Path]:
    root = _canonicalize_root(
        memory_root,
        create=False,
        require_directory=False,
    )
    raw_namespace = _absolute_platform_base(
        namespace_path,
        "memory namespace",
    )
    if ".." in raw_namespace.parts:
        raise MemoryIdentityError("memory namespace escapes the configured root")
    _assert_no_symlink_or_reparse_components(raw_namespace)
    canonical_namespace = Path(os.path.normpath(str(raw_namespace)))
    expected = repository_namespace_path(root, key)
    if _normalized_path(canonical_namespace) != _normalized_path(expected):
        raise MemoryIdentityError(
            "memory namespace must use its canonical repository location"
        )
    return root, expected


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

    root_descriptor = _physical_path_descriptor(root)
    namespace_descriptor = _physical_path_descriptor(namespace)
    if not _physical_path_is_ancestor(root_descriptor, namespace_descriptor):
        raise MemoryIdentityError("memory namespace plan path changed")
    _reject_protected_path_overlap(
        root,
        current_locator.protected_paths,
        root_descriptor=root_descriptor,
    )

    try:
        _secure_materialize_directory(
            namespace,
            expected=namespace_descriptor,
        )
        _assert_materialized_directory(root)
        _assert_materialized_directory(root / _REPOSITORIES_DIRECTORY)
        _assert_materialized_directory(namespace)
    except MemoryIdentityError:
        raise
    except OSError:
        raise MemoryIdentityError("memory namespace cannot be materialized safely") from None
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
    _validate_memory_path_components(candidate)
    _assert_no_symlink_or_reparse_components(
        candidate,
        blocked_component_message=(
            "memory root cannot safely create its parent directories"
            if create
            else "memory path component is not a directory"
        ),
    )
    canonical = Path(os.path.normpath(str(candidate)))
    if require_directory and canonical.exists() and not canonical.is_dir():
        raise MemoryIdentityError("memory root exists but is not a directory")
    if create:
        try:
            _secure_materialize_directory(canonical)
            _assert_materialized_directory(canonical)
        except MemoryIdentityError:
            raise
        except OSError:
            raise MemoryIdentityError(
                "memory root cannot safely create its parent directories"
            ) from None
    return canonical


def _canonical_identity_path(value: PathInput, *, field_name: str) -> str:
    text = str(value)
    candidate = Path(text)
    if not text or "\0" in text or not candidate.is_absolute():
        raise MemoryIdentityError(f"{field_name} must be absolute")
    try:
        return str(candidate.resolve(strict=False))
    except OSError:
        raise MemoryIdentityError(f"unable to canonicalize {field_name}") from None


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


def _assert_no_symlink_or_reparse_components(
    candidate: Path,
    *,
    blocked_component_message: str = "memory path component is not a directory",
) -> None:
    if not candidate.is_absolute():
        raise MemoryIdentityError("memory path must be absolute")
    current = Path(candidate.anchor)
    parts = candidate.parts[1:] if candidate.anchor else candidate.parts
    for index, part in enumerate(parts):
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            break
        except OSError:
            raise MemoryIdentityError("unable to inspect memory path safety") from None
        attributes = getattr(metadata, "st_file_attributes", 0)
        if stat.S_ISLNK(metadata.st_mode) or (
            attributes & _FILE_ATTRIBUTE_REPARSE_POINT
        ):
            raise MemoryIdentityError(
                "memory path cannot traverse a symbolic link or reparse point"
            )
        if index < len(parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise MemoryIdentityError(blocked_component_message)


def _assert_materialized_directory(path: Path) -> None:
    _assert_no_symlink_or_reparse_components(path)
    if not path.is_dir():
        raise MemoryIdentityError("memory namespace component is not a directory")


def _secure_materialize_directory(
    path: Path,
    expected: Optional[_PhysicalPathDescriptor] = None,
) -> None:
    """Create a directory chain relative to held, no-follow parent handles."""

    if not path.is_absolute():
        raise MemoryIdentityError("memory path must be absolute")
    expected_descriptor = expected or _physical_path_descriptor(path)
    if os.name == "nt":
        _secure_materialize_directory_windows(path, expected_descriptor)
    else:
        _secure_materialize_directory_posix(path, expected_descriptor)


def _secure_materialize_directory_posix(
    path: Path,
    expected: _PhysicalPathDescriptor,
) -> None:
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if (
        not no_follow
        or not directory
        or os.open not in os.supports_dir_fd
        or os.mkdir not in os.supports_dir_fd
    ):
        raise MemoryIdentityError("secure directory creation is unavailable")
    flags = os.O_RDONLY | directory | no_follow | getattr(os, "O_CLOEXEC", 0)
    parts = path.parts[1:] if path.anchor else path.parts
    current_fd: Optional[int] = None
    try:
        current_fd = os.open(path.anchor, flags)
        first_missing = len(parts)
        for index, part in enumerate(parts):
            try:
                child_fd = os.open(part, flags, dir_fd=current_fd)
            except FileNotFoundError:
                first_missing = index
                break
            except OSError:
                raise MemoryIdentityError(
                    "memory path cannot traverse a symbolic link or invalid component"
                ) from None
            os.close(current_fd)
            current_fd = child_fd

        remaining = tuple(
            _comparison_path_part(part) for part in parts[first_missing:]
        )
        _assert_secure_directory_anchor(
            expected,
            _posix_file_identity(current_fd),
            remaining,
        )

        for part in parts[first_missing:]:
            try:
                os.mkdir(part, mode=0o700, dir_fd=current_fd)
            except FileExistsError:
                pass
            except OSError:
                raise MemoryIdentityError(
                    "memory namespace cannot be materialized safely"
                ) from None
            try:
                child_fd = os.open(part, flags, dir_fd=current_fd)
            except OSError:
                raise MemoryIdentityError(
                    "memory path cannot traverse a symbolic link or invalid component"
                ) from None
            os.close(current_fd)
            current_fd = child_fd
    finally:
        if current_fd is not None:
            os.close(current_fd)


def _posix_file_identity(handle: int) -> Tuple[int, int]:
    try:
        metadata = os.fstat(handle)
    except OSError:
        raise MemoryIdentityError("unable to inspect memory path safety") from None
    if not stat.S_ISDIR(metadata.st_mode):
        raise MemoryIdentityError("memory path component is not a directory")
    return int(metadata.st_dev), int(metadata.st_ino)


def _assert_secure_directory_anchor(
    expected: _PhysicalPathDescriptor,
    observed_identity: Tuple[int, int],
    remaining_parts: Tuple[str, ...],
) -> None:
    if (
        observed_identity != expected.deepest_identity
        or remaining_parts != expected.missing_parts
    ):
        raise MemoryIdentityError("memory path changed during secure validation")


def _secure_materialize_directory_windows(
    path: Path,
    expected: _PhysicalPathDescriptor,
) -> None:
    if os.name != "nt":
        raise MemoryIdentityError("secure Windows directory creation is unavailable")
    parts = path.parts[1:] if path.anchor else path.parts
    current_handle = _windows_open_directory_root(Path(path.anchor))
    try:
        first_missing = len(parts)
        for index, part in enumerate(parts):
            child_handle = _windows_open_directory_component(
                current_handle,
                part,
                create=False,
            )
            if child_handle is None:
                first_missing = index
                break
            _windows_close_handle(current_handle)
            current_handle = child_handle

        remaining = tuple(
            _comparison_path_part(part) for part in parts[first_missing:]
        )
        _assert_secure_directory_anchor(
            expected,
            _windows_directory_identity(current_handle),
            remaining,
        )

        for part in parts[first_missing:]:
            child_handle = _windows_open_directory_component(
                current_handle,
                part,
                create=True,
            )
            if child_handle is None:
                raise MemoryIdentityError(
                    "memory namespace cannot be materialized safely"
                )
            _windows_close_handle(current_handle)
            current_handle = child_handle
    finally:
        _windows_close_handle(current_handle)


def _windows_open_directory_root(path: Path):
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    handle = create_file(
        str(path),
        _WIN_FILE_LIST_DIRECTORY
        | _WIN_FILE_TRAVERSE
        | _WIN_FILE_READ_ATTRIBUTES
        | _WIN_SYNCHRONIZE,
        _WIN_FILE_SHARE_ALL,
        None,
        _WIN_OPEN_EXISTING,
        _WIN_FILE_FLAG_BACKUP_SEMANTICS | _WIN_FILE_OPEN_REPARSE_POINT,
        None,
    )
    if handle == wintypes.HANDLE(-1).value:
        raise MemoryIdentityError("unable to inspect memory path safety")
    try:
        _windows_directory_identity(handle)
    except Exception:
        _windows_close_handle(handle)
        raise
    return handle


def _windows_open_directory_component(
    parent_handle,
    component: str,
    *,
    create: bool,
):
    _validate_windows_memory_component(component)
    try:
        encoded_length = len(component.encode("utf-16-le"))
    except UnicodeEncodeError:
        raise MemoryIdentityError("memory path component is invalid") from None
    if encoded_length > 65_532:
        raise MemoryIdentityError("memory path component is invalid")
    name_buffer = ctypes.create_unicode_buffer(component)
    name = _WindowsUnicodeString(
        Length=encoded_length,
        MaximumLength=encoded_length + 2,
        Buffer=ctypes.cast(name_buffer, wintypes.LPWSTR),
    )
    attributes = _WindowsObjectAttributes(
        Length=ctypes.sizeof(_WindowsObjectAttributes),
        RootDirectory=parent_handle,
        ObjectName=ctypes.pointer(name),
        Attributes=_WIN_OBJ_CASE_INSENSITIVE,
        SecurityDescriptor=None,
        SecurityQualityOfService=None,
    )
    io_status = _WindowsIoStatusBlock()
    child_handle = wintypes.HANDLE()
    ntdll = ctypes.WinDLL("ntdll")
    nt_create_file = ntdll.NtCreateFile
    nt_create_file.argtypes = (
        ctypes.POINTER(wintypes.HANDLE),
        wintypes.DWORD,
        ctypes.POINTER(_WindowsObjectAttributes),
        ctypes.POINTER(_WindowsIoStatusBlock),
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
    )
    nt_create_file.restype = ctypes.c_long
    status = nt_create_file(
        ctypes.byref(child_handle),
        _WIN_FILE_LIST_DIRECTORY
        | _WIN_FILE_TRAVERSE
        | _WIN_FILE_READ_ATTRIBUTES
        | _WIN_SYNCHRONIZE,
        ctypes.byref(attributes),
        ctypes.byref(io_status),
        None,
        _WIN_FILE_ATTRIBUTE_NORMAL,
        _WIN_FILE_SHARE_ALL,
        _WIN_FILE_OPEN_IF if create else _WIN_FILE_OPEN,
        _WIN_FILE_DIRECTORY_FILE
        | _WIN_FILE_SYNCHRONOUS_IO_NONALERT
        | _WIN_FILE_OPEN_FOR_BACKUP_INTENT
        | _WIN_FILE_OPEN_REPARSE_POINT,
        None,
        0,
    )
    if status < 0:
        status_to_error = ntdll.RtlNtStatusToDosError
        status_to_error.argtypes = (wintypes.ULONG,)
        status_to_error.restype = wintypes.ULONG
        error_code = int(status_to_error(ctypes.c_ulong(status).value))
        if not create and error_code in (
            _WIN_ERROR_FILE_NOT_FOUND,
            _WIN_ERROR_PATH_NOT_FOUND,
        ):
            return None
        raise MemoryIdentityError(
            "memory namespace cannot be materialized safely"
            if create
            else "unable to inspect memory path safety"
        )
    try:
        _windows_directory_identity(child_handle)
    except Exception:
        _windows_close_handle(child_handle)
        raise
    return child_handle


def _windows_directory_identity(handle) -> Tuple[int, int]:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_information = kernel32.GetFileInformationByHandle
    get_information.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(_WindowsByHandleFileInformation),
    )
    get_information.restype = wintypes.BOOL
    information = _WindowsByHandleFileInformation()
    if not get_information(handle, ctypes.byref(information)):
        raise MemoryIdentityError("unable to inspect memory path safety")
    if information.dwFileAttributes & _FILE_ATTRIBUTE_REPARSE_POINT:
        raise MemoryIdentityError(
            "memory path cannot traverse a symbolic link or reparse point"
        )
    if not information.dwFileAttributes & _WIN_FILE_ATTRIBUTE_DIRECTORY:
        raise MemoryIdentityError("memory path component is not a directory")
    return (
        int(information.dwVolumeSerialNumber),
        (int(information.nFileIndexHigh) << 32)
        | int(information.nFileIndexLow),
    )


def _windows_close_handle(handle) -> None:
    if os.name != "nt" or handle in (None, 0, wintypes.HANDLE(-1).value):
        return
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    close_handle(handle)


def _validate_memory_path_components(path: Path) -> None:
    if os.name != "nt":
        return
    parts = path.parts[1:] if path.anchor else path.parts
    for component in parts:
        _validate_windows_memory_component(component)


def _validate_windows_memory_component(component: str) -> None:
    if os.name != "nt":
        return
    if (
        not component
        or component in (".", "..")
        or "\\" in component
        or "/" in component
        or ":" in component
        or component.rstrip(" .") != component
    ):
        raise MemoryIdentityError("memory path component is invalid")
    stem = component.split(".", 1)[0].casefold()
    if stem in _WINDOWS_RESERVED_COMPONENTS:
        raise MemoryIdentityError("memory path component is invalid")
    try:
        component_units = len(component.encode("utf-16-le")) // 2
    except UnicodeEncodeError:
        raise MemoryIdentityError("memory path component is invalid") from None
    if component_units > 255:
        raise MemoryIdentityError("memory path component is invalid")


def _reject_protected_path_overlap(
    root: Path,
    protected_paths: Tuple[str, ...],
    *,
    root_descriptor: Optional[_PhysicalPathDescriptor] = None,
) -> None:
    checked_root = root_descriptor or _physical_path_descriptor(root)
    for protected in protected_paths:
        protected_descriptor = _physical_path_descriptor(Path(protected))
        if _physical_descriptors_overlap(checked_root, protected_descriptor):
            raise MemoryIdentityError(
                "memory root overlaps protected repository paths"
            )


def _paths_overlap(left: Path, right: Path) -> bool:
    left_descriptor = _physical_path_descriptor(left)
    right_descriptor = _physical_path_descriptor(right)
    return _physical_descriptors_overlap(left_descriptor, right_descriptor)


def _physical_descriptors_overlap(
    left_descriptor: _PhysicalPathDescriptor,
    right_descriptor: _PhysicalPathDescriptor,
) -> bool:
    return _physical_path_is_ancestor(
        left_descriptor,
        right_descriptor,
    ) or _physical_path_is_ancestor(
        right_descriptor,
        left_descriptor,
    )


def _path_is_within(candidate: Path, root: Path) -> bool:
    return _physical_path_is_ancestor(
        _physical_path_descriptor(root),
        _physical_path_descriptor(candidate),
    )


def _physical_path_descriptor(path: Path) -> _PhysicalPathDescriptor:
    if not path.is_absolute():
        raise MemoryIdentityError("memory path must be absolute")
    current = Path(os.path.normpath(str(path)))
    missing_parts = []
    while True:
        try:
            metadata = current.stat()
        except (FileNotFoundError, NotADirectoryError):
            parent = current.parent
            if parent == current:
                raise MemoryIdentityError("unable to inspect memory path safety") from None
            missing_parts.insert(0, _comparison_path_part(current.name))
            current = parent
            continue
        except OSError:
            raise MemoryIdentityError("unable to inspect memory path safety") from None
        if missing_parts and not stat.S_ISDIR(metadata.st_mode):
            raise MemoryIdentityError("memory path component is not a directory")
        break

    suffix = tuple(missing_parts)
    ancestors = []
    visited_paths = set()
    while True:
        normalized_current = _normalized_path(current)
        if normalized_current in visited_paths:
            raise MemoryIdentityError("unable to inspect memory path safety")
        visited_paths.add(normalized_current)
        try:
            metadata = current.stat()
        except OSError:
            raise MemoryIdentityError("unable to inspect memory path safety") from None
        ancestors.append(
            _PhysicalPathAncestor(
                identity=(int(metadata.st_dev), int(metadata.st_ino)),
                suffix_parts=suffix,
            )
        )
        parent = current.parent
        if parent == current:
            break
        suffix = (_comparison_path_part(current.name),) + suffix
        current = parent

    return _PhysicalPathDescriptor(
        ancestors=tuple(ancestors),
        missing_parts=tuple(missing_parts),
    )


def _physical_path_is_ancestor(
    ancestor: _PhysicalPathDescriptor,
    candidate: _PhysicalPathDescriptor,
) -> bool:
    if not ancestor.missing_parts:
        target = ancestor.deepest_identity
        return any(item.identity == target for item in candidate.ancestors)

    base = ancestor.deepest_identity
    for item in candidate.ancestors:
        if item.identity != base:
            continue
        expected = ancestor.missing_parts
        return item.suffix_parts[: len(expected)] == expected
    return False


def _comparison_path_part(value: str) -> str:
    return os.path.normcase(value)


def _normalized_path(path: Path) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


def _utf16_code_units(value: str) -> int:
    try:
        return len(value.encode("utf-16-le")) // 2
    except UnicodeEncodeError:
        raise MemoryIdentityError("memory namespace contains unsupported path text") from None


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
    "validate_repository_memory_namespace",
    "verify_repository_identity",
    "verify_repository_locator",
]
