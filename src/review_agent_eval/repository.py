"""Reproducible repository preparation and fail-closed Trial isolation.

The module deliberately keeps canonical, replayable manifests separate from
runtime path handles.  Git is used only as a bounded object reader (and, for
external acquisition, as a transport into a disposable quarantine).  A Trial
is never a clone or worktree: its object database and working tree are written
from a verified logical object closure with no checkout, filters, hooks,
alternates, hardlinks, remotes, or source locator.
"""

from __future__ import annotations

import errno
import hashlib
import ipaddress
import os
import re
import signal
import socket
import stat
import struct
import subprocess
import tempfile
import threading
import time
import unicodedata
import uuid
import zlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import (
    Any,
    BinaryIO,
    Callable,
    ClassVar,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)
from urllib.parse import urlsplit

from .artifacts import TrialManifest
from .cases import (
    SuiteCase,
    SuiteSource,
    is_windows_reserved_path_component,
)
from .models import (
    EvalInput,
    MAX_COUNTER,
    MAX_IDENTIFIER_CHARS,
    Repository,
    RepositorySource,
    RepositoryReviewTarget,
    SchemaError,
    TrialStatus,
    _JsonModel,
    _digest,
    _exact_fields,
    _git_object,
    _identifier,
    _integer,
    _object,
    _strict_json_loads,
    _string,
    canonical_json_bytes,
    canonical_sha256,
    stable_id,
)


def repository_from_eval_input(eval_input: EvalInput) -> Repository:
    """Return the Repository descriptor carried by a Repository Target.

    Repository preparation is a Target-specific operation.  Keeping this
    projection in one place prevents the old v1 ``eval_input.repository``
    shape from leaking back into the v2 runtime.
    """

    if not isinstance(eval_input, EvalInput):
        raise TypeError("eval_input must be an EvalInput")
    target = eval_input.review_target
    if not isinstance(target, RepositoryReviewTarget):
        raise RepositoryPreparationError(
            "Repository preparation requires a repository review target"
        )
    return target.repository


PREPARED_REPOSITORY_MANIFEST_SCHEMA_VERSION = (
    "prepared_repository_manifest_v1"
)
WORKSPACE_MANIFEST_SCHEMA_VERSION = "workspace_manifest_v1"
REPOSITORY_ACQUISITION_BINDING_SCHEMA_VERSION = (
    "repository_acquisition_binding_v1"
)
REPOSITORY_ISOLATION_POLICY_VERSION = "repository_isolation_v1"
REPOSITORY_PATH_POLICY_VERSION = "repository_path_policy_v1"
REPOSITORY_BUDGET_POLICY_VERSION = "repository_budget_policy_v1"
LOGICAL_GIT_SOURCE_VERSION = "logical_git_source_v1"
CACHE_INDEX_SCHEMA_VERSION = "repository_cache_index_v1"
REPOSITORY_RESERVATION_SCHEMA_VERSION = "repository_reservation_v1"

MAX_REPOSITORY_MANIFEST_BYTES = 128 * 1024
MAX_WORKSPACE_MANIFEST_BYTES = 128 * 1024
MAX_ACQUISITION_BINDING_BYTES = 64 * 1024
MAX_GIT_STDOUT_BYTES = 16 * 1024 * 1024
MAX_GIT_STDERR_BYTES = 256 * 1024
MAX_GIT_CONFIG_BYTES = 1024 * 1024
MAX_GIT_EXECUTABLE_BYTES = 512 * 1024 * 1024
MAX_GIT_METADATA_NODES = 500_000
MAX_GIT_OBJECTS = 100_000
MAX_GIT_BLOB_BYTES = 64 * 1024 * 1024
MAX_MATERIALIZED_FILES = 50_000
MAX_MATERIALIZED_BYTES = 512 * 1024 * 1024
MAX_CACHE_BYTES = 1024 * 1024 * 1024
MAX_PATH_DEPTH = 64
MAX_PATH_BYTES = 1024
MAX_PATH_COMPONENT_BYTES = 255
# A small Git tree DAG can expand into an enormous logical tree.  Count logical
# entries, not only unique objects/files, before materialization or replay.
MAX_LOGICAL_TREE_ENTRIES = MAX_MATERIALIZED_FILES * 2 + MAX_PATH_DEPTH
MAX_GIT_METADATA_OBJECT_BYTES = 8 * 1024 * 1024
MAX_TRIAL_ATTEMPT = 10_000
DEFAULT_GIT_TIMEOUT_SECONDS = 180.0
DEFAULT_LOCK_TIMEOUT_SECONDS = 180.0
DEFAULT_MAX_RETAINED_WORKSPACES = 3
DEFAULT_MAX_RETAINED_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_RETAINED_TTL_SECONDS = 24 * 60 * 60
MAX_RETENTION_TTL_SECONDS = 30 * 24 * 60 * 60
MAX_DATA_ROOT_BYTES = 64 * 1024 * 1024 * 1024
MAX_DATA_ROOT_NODES = 5_000_000
MAX_PREPARE_RESERVATION_BYTES = 5 * MAX_CACHE_BYTES
MAX_PREPARE_RESERVATION_NODES = 3 * MAX_GIT_METADATA_NODES

_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_GIT_OID_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_HEX_RE = re.compile(r"^[0-9a-f]+$")
_WINDOWS_FORBIDDEN = frozenset('<>:"\\|?*')
_VCS_METADATA = frozenset({".git", ".hg", ".svn", ".gitmodules"})
_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1\n"
_LFS_ATTRIBUTE_RE = re.compile(
    rb"(?:^|[ \t])(?:filter|diff|merge)=lfs(?:[ \t]|$)", re.IGNORECASE
)
_EXTERNAL_FILTER_ATTRIBUTE_RE = re.compile(
    rb"(?:^|[ \t])filter=(?!unset\b)[^ \t\r\n]+", re.IGNORECASE
)
_OPERATION_DIRECTORY_RE = re.compile(r"^operation-([0-9a-f]{32})$")
_SAFE_SOURCE_CONFIG_KEYS = frozenset(
    {
        "core.repositoryformatversion",
        "core.filemode",
        "core.bare",
        "core.logallrefupdates",
        "core.ignorecase",
        "core.precomposeunicode",
        "core.symlinks",
        "extensions.objectformat",
        "user.name",
        "user.email",
    }
)


class RepositoryPreparationError(RuntimeError):
    """Repository preparation could not produce a usable immutable source."""


class RepositoryIntegrityError(RepositoryPreparationError):
    """A declared revision, object closure, digest, or cache binding is wrong."""


class RepositorySecurityError(RepositoryPreparationError):
    """A filesystem, locator, process, or credential boundary is unsafe."""


class RepositoryPolicyError(RepositoryPreparationError):
    """Repository content is outside ``repository_isolation_v1``."""


class RepositoryLimitError(RepositoryPolicyError):
    """A fixed repository path or resource budget was exceeded."""


class RepositoryMode(str, Enum):
    """Whether repository sources may be acquired or only replayed from cache."""

    ACQUIRE = "acquire"
    CACHE_ONLY = "cache_only"


class RepositoryCacheStatus(str, Enum):
    """Stable result of a non-acquiring repository cache probe."""

    AVAILABLE = "available"
    MISSING = "missing"


@dataclass(frozen=True)
class CacheCheck:
    """Path-free result of validating one canonical repository cache binding.

    Corrupt, ambiguous, or unsafe cache state is never represented as a status:
    those conditions fail closed with the corresponding repository exception.
    ``MISSING`` therefore means only that no matching committed request index
    exists for the descriptor and current verified Git executable.
    """

    repository_descriptor_digest: str
    status: RepositoryCacheStatus
    request_id: Optional[str] = None
    cache_id: Optional[str] = None
    prepared_repository_id: Optional[str] = None
    manifest_digest: Optional[str] = None

    def __post_init__(self) -> None:
        _digest(
            self.repository_descriptor_digest,
            "cache check.repository_descriptor_digest",
        )
        if not isinstance(self.status, RepositoryCacheStatus):
            raise TypeError("cache check status must be a RepositoryCacheStatus")
        values = (
            self.request_id,
            self.cache_id,
            self.prepared_repository_id,
            self.manifest_digest,
        )
        if self.status is RepositoryCacheStatus.MISSING:
            if any(value is not None for value in values):
                raise ValueError("missing cache check may not claim cache identity")
            return
        if any(value is None for value in values):
            raise ValueError("available cache check requires complete cache identity")
        assert self.request_id is not None
        assert self.cache_id is not None
        assert self.prepared_repository_id is not None
        assert self.manifest_digest is not None
        _identifier(self.request_id, "cache check.request_id")
        _identifier(self.cache_id, "cache check.cache_id")
        _identifier(
            self.prepared_repository_id,
            "cache check.prepared_repository_id",
        )
        _digest(self.manifest_digest, "cache check.manifest_digest")

    @property
    def available(self) -> bool:
        return self.status is RepositoryCacheStatus.AVAILABLE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repository_descriptor_digest": self.repository_descriptor_digest,
            "status": self.status.value,
            "request_id": self.request_id,
            "cache_id": self.cache_id,
            "prepared_repository_id": self.prepared_repository_id,
            "manifest_digest": self.manifest_digest,
        }


class WorkspaceRetentionPolicy(str, Enum):
    DELETE_ALWAYS = "delete_always"
    RETAIN_ON_FAILURE = "retain_on_failure"


@dataclass(frozen=True)
class WorkspaceDiagnostic:
    code: str
    message: str

    def __post_init__(self) -> None:
        _identifier(self.code, "workspace diagnostic code")
        _string(self.message, "workspace diagnostic message", 4096)


def _fixed_isolation_policy() -> Dict[str, Any]:
    return {
        "schema_version": REPOSITORY_ISOLATION_POLICY_VERSION,
        "materialization": "verified_tree_manual_materialization",
        "git_object_storage": "independent_loose_objects_no_hardlinks",
        "git_remotes": "absent",
        "shared_alternates": "rejected",
        "common_directory": "rejected",
        "promisor_or_partial_clone": "rejected",
        "shallow_or_grafts": "rejected",
        "replace_refs": "rejected",
        "symlinks": "rejected",
        "gitlinks_or_submodules": "rejected",
        "nested_repositories": "rejected",
        "lfs": "rejected",
        "hooks": "never_executed",
        "external_filters": "never_executed",
        "remote_endpoint": "https_public_global_unicast_dns_pinned",
        "control_plane_access": "adapter_required_separate_os_identity_not_proven",
        "trial_network": "adapter_required_os_egress_not_proven",
    }


def _fixed_path_policy() -> Dict[str, Any]:
    return {
        "schema_version": REPOSITORY_PATH_POLICY_VERSION,
        "encoding": "utf8_nfc_posix_relative",
        "case_collision": "nfc_casefold_rejected",
        "windows_ads_and_devices": "rejected",
        "windows_trailing_dot_space": "rejected",
        "control_characters": "rejected",
        "max_depth": MAX_PATH_DEPTH,
        "max_path_bytes": MAX_PATH_BYTES,
        "max_component_bytes": MAX_PATH_COMPONENT_BYTES,
    }


def _budget_policy(
    *,
    object_count: int,
    blob_count: int,
    raw_object_bytes: int,
    materialized_files: int,
    materialized_bytes: int,
) -> Dict[str, Any]:
    return {
        "schema_version": REPOSITORY_BUDGET_POLICY_VERSION,
        "max_objects": MAX_GIT_OBJECTS,
        "max_blob_bytes": MAX_GIT_BLOB_BYTES,
        "max_materialized_files": MAX_MATERIALIZED_FILES,
        "max_materialized_bytes": MAX_MATERIALIZED_BYTES,
        "max_logical_tree_entries": MAX_LOGICAL_TREE_ENTRIES,
        "max_cache_bytes": MAX_CACHE_BYTES,
        "actual_objects": object_count,
        "actual_blobs": blob_count,
        "actual_raw_object_bytes": raw_object_bytes,
        "actual_materialized_files": materialized_files,
        "actual_materialized_bytes": materialized_bytes,
    }


def _validate_fixed_policy(
    value: Any, expected: Mapping[str, Any], context: str
) -> Dict[str, Any]:
    payload = _object(value, context)
    _exact_fields(payload, tuple(expected), context)
    if payload != dict(expected):
        raise SchemaError("%s must be the fixed official v1 policy" % context)
    return dict(payload)


def _validate_budget_policy(value: Any) -> Dict[str, Any]:
    payload = _object(value, "repository budget policy")
    expected_fields = (
        "schema_version",
        "max_objects",
        "max_blob_bytes",
        "max_materialized_files",
        "max_materialized_bytes",
        "max_logical_tree_entries",
        "max_cache_bytes",
        "actual_objects",
        "actual_blobs",
        "actual_raw_object_bytes",
        "actual_materialized_files",
        "actual_materialized_bytes",
    )
    _exact_fields(payload, expected_fields, "repository budget policy")
    fixed = {
        "schema_version": REPOSITORY_BUDGET_POLICY_VERSION,
        "max_objects": MAX_GIT_OBJECTS,
        "max_blob_bytes": MAX_GIT_BLOB_BYTES,
        "max_materialized_files": MAX_MATERIALIZED_FILES,
        "max_materialized_bytes": MAX_MATERIALIZED_BYTES,
        "max_logical_tree_entries": MAX_LOGICAL_TREE_ENTRIES,
        "max_cache_bytes": MAX_CACHE_BYTES,
    }
    for key, expected in fixed.items():
        if payload[key] != expected:
            raise SchemaError("repository budget policy is not official v1")
    maxima = {
        "actual_objects": MAX_GIT_OBJECTS,
        "actual_blobs": MAX_GIT_OBJECTS,
        "actual_raw_object_bytes": MAX_CACHE_BYTES,
        "actual_materialized_files": MAX_MATERIALIZED_FILES,
        "actual_materialized_bytes": MAX_MATERIALIZED_BYTES,
    }
    result = dict(fixed)
    for key, maximum in maxima.items():
        result[key] = _integer(payload[key], key, minimum=0, maximum=maximum)
    if result["actual_blobs"] > result["actual_objects"]:
        raise SchemaError("repository blob count exceeds object count")
    return result


def _prepared_repository_id_payload(
    *,
    source_digest: str,
    base_source_digest: str,
    head_source_digest: str,
    object_format: str,
    base_revision: str,
    head_revision: str,
    base_tree: str,
    head_tree: str,
    budget_policy: Mapping[str, Any],
) -> Tuple[Any, ...]:
    return (
        PREPARED_REPOSITORY_MANIFEST_SCHEMA_VERSION,
        LOGICAL_GIT_SOURCE_VERSION,
        source_digest,
        base_source_digest,
        head_source_digest,
        object_format,
        base_revision,
        head_revision,
        base_tree,
        head_tree,
        _fixed_isolation_policy(),
        _fixed_path_policy(),
        dict(budget_policy),
    )


@dataclass(frozen=True)
class PreparedRepositoryManifest(_JsonModel):
    """Canonical repository identity; it deliberately contains no ``Path``."""

    SCHEMA_VERSION: ClassVar[str] = PREPARED_REPOSITORY_MANIFEST_SCHEMA_VERSION

    schema_version: str
    prepared_repository_id: str
    logical_source_version: str
    source_digest: str
    base_source_digest: str
    head_source_digest: str
    object_format: str
    base_revision: str
    head_revision: str
    base_tree: str
    head_tree: str
    budget_policy: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.schema_version != self.SCHEMA_VERSION:
            raise SchemaError("prepared repository manifest has unknown schema_version")
        if self.logical_source_version != LOGICAL_GIT_SOURCE_VERSION:
            raise SchemaError("prepared repository has unknown logical source version")
        for name in (
            "source_digest",
            "base_source_digest",
            "head_source_digest",
        ):
            _digest(getattr(self, name), name)
        if self.object_format not in {"sha1", "sha256"}:
            raise SchemaError("object_format must be sha1 or sha256")
        expected_length = 40 if self.object_format == "sha1" else 64
        for name in ("base_revision", "head_revision", "base_tree", "head_tree"):
            value = _git_object(getattr(self, name), name)
            if len(value) != expected_length:
                raise SchemaError("Git object ID length does not match object_format")
        if self.base_revision == self.head_revision:
            raise SchemaError("prepared base and head revisions must differ")
        budget = _validate_budget_policy(self.budget_policy)
        object.__setattr__(self, "budget_policy", MappingProxyType(budget))
        expected_id = stable_id(
            "prepared-repository",
            *_prepared_repository_id_payload(
                source_digest=self.source_digest,
                base_source_digest=self.base_source_digest,
                head_source_digest=self.head_source_digest,
                object_format=self.object_format,
                base_revision=self.base_revision,
                head_revision=self.head_revision,
                base_tree=self.base_tree,
                head_tree=self.head_tree,
                budget_policy=budget,
            ),
        )
        if self.prepared_repository_id != expected_id:
            raise SchemaError("prepared_repository_id does not match canonical identity")

    @classmethod
    def create(
        cls,
        *,
        source_digest: str,
        base_source_digest: str,
        head_source_digest: str,
        object_format: str,
        base_revision: str,
        head_revision: str,
        base_tree: str,
        head_tree: str,
        budget_policy: Mapping[str, Any],
    ) -> "PreparedRepositoryManifest":
        identity = _prepared_repository_id_payload(
            source_digest=source_digest,
            base_source_digest=base_source_digest,
            head_source_digest=head_source_digest,
            object_format=object_format,
            base_revision=base_revision,
            head_revision=head_revision,
            base_tree=base_tree,
            head_tree=head_tree,
            budget_policy=budget_policy,
        )
        return cls(
            schema_version=cls.SCHEMA_VERSION,
            prepared_repository_id=stable_id("prepared-repository", *identity),
            logical_source_version=LOGICAL_GIT_SOURCE_VERSION,
            source_digest=source_digest,
            base_source_digest=base_source_digest,
            head_source_digest=head_source_digest,
            object_format=object_format,
            base_revision=base_revision,
            head_revision=head_revision,
            base_tree=base_tree,
            head_tree=head_tree,
            budget_policy=dict(budget_policy),
        )

    @classmethod
    def from_dict(cls, value: Any) -> "PreparedRepositoryManifest":
        payload = _object(value, "prepared repository manifest")
        fields = (
            "schema_version",
            "prepared_repository_id",
            "logical_source_version",
            "source_digest",
            "base_source_digest",
            "head_source_digest",
            "object_format",
            "base_revision",
            "head_revision",
            "base_tree",
            "head_tree",
            "isolation_policy",
            "path_policy",
            "budget_policy",
        )
        _exact_fields(payload, fields, "prepared repository manifest")
        _validate_fixed_policy(
            payload["isolation_policy"],
            _fixed_isolation_policy(),
            "repository isolation policy",
        )
        _validate_fixed_policy(
            payload["path_policy"], _fixed_path_policy(), "repository path policy"
        )
        return cls(
            schema_version=payload["schema_version"],
            prepared_repository_id=_identifier(
                payload["prepared_repository_id"], "prepared_repository_id"
            ),
            logical_source_version=payload["logical_source_version"],
            source_digest=_digest(payload["source_digest"], "source_digest"),
            base_source_digest=_digest(
                payload["base_source_digest"], "base_source_digest"
            ),
            head_source_digest=_digest(
                payload["head_source_digest"], "head_source_digest"
            ),
            object_format=payload["object_format"],
            base_revision=_git_object(payload["base_revision"], "base_revision"),
            head_revision=_git_object(payload["head_revision"], "head_revision"),
            base_tree=_git_object(payload["base_tree"], "base_tree"),
            head_tree=_git_object(payload["head_tree"], "head_tree"),
            budget_policy=_validate_budget_policy(payload["budget_policy"]),
        )

    @classmethod
    def from_json(cls, data: Any) -> "PreparedRepositoryManifest":
        return cls.from_dict(
            _strict_json_loads(
                data,
                MAX_REPOSITORY_MANIFEST_BYTES,
                "prepared repository manifest JSON",
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "prepared_repository_id": self.prepared_repository_id,
            "logical_source_version": self.logical_source_version,
            "source_digest": self.source_digest,
            "base_source_digest": self.base_source_digest,
            "head_source_digest": self.head_source_digest,
            "object_format": self.object_format,
            "base_revision": self.base_revision,
            "head_revision": self.head_revision,
            "base_tree": self.base_tree,
            "head_tree": self.head_tree,
            "isolation_policy": _fixed_isolation_policy(),
            "path_policy": _fixed_path_policy(),
            "budget_policy": dict(self.budget_policy),
        }


def _canonical_host(value: str) -> str:
    host = _string(value, "repository host", 253)
    if any(ord(character) < 33 or ord(character) == 127 for character in host):
        raise SchemaError("repository host contains whitespace or controls")
    try:
        canonical = host.encode("idna").decode("ascii").lower().rstrip(".")
    except UnicodeError as exc:
        raise SchemaError("repository host is not valid IDNA") from exc
    if not canonical or len(canonical) > 253:
        raise SchemaError("repository host is invalid")
    return canonical


def _validate_https_url(value: Any):
    url = _string(value, "remote repository URL", 8192)
    if any(character.isspace() or ord(character) < 32 for character in url):
        raise SchemaError("remote repository URL contains whitespace or controls")
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise SchemaError("remote repository URL is invalid") from exc
    if parsed.scheme != "https":
        raise SchemaError("repository_isolation_v1 permits HTTPS remotes only")
    if parsed.username is not None or parsed.password is not None or "@" in parsed.netloc:
        raise SchemaError("remote repository URL may not contain credentials")
    if not parsed.hostname:
        raise SchemaError("remote repository URL requires a host")
    _canonical_host(parsed.hostname)
    if port is not None and not 1 <= port <= 65535:
        raise SchemaError("remote repository URL port is invalid")
    if parsed.query or parsed.fragment:
        raise SchemaError("remote repository URL may not contain query or fragment")
    if not parsed.path.startswith("/") or parsed.path in {"", "/"}:
        raise SchemaError("remote repository URL requires an absolute repository path")
    return parsed


def _validate_https_origin(value: Any) -> str:
    origin = _string(value, "redirect origin", 1024)
    parsed = _validate_https_url(origin + "/sentinel" if origin.endswith(":") else origin)
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise SchemaError("redirect allowlist entries must be HTTPS origins")
    host = _canonical_host(parsed.hostname or "")
    port = parsed.port or 443
    return "https://%s%s" % (host, "" if port == 443 else ":%d" % port)


def _resolve_remote_endpoint(
    url: str,
    *,
    allowed_host: str,
    allowed_port: int,
) -> Tuple[str, ...]:
    """Resolve and pin one explicitly authorized public HTTPS endpoint.

    The returned addresses are later passed to libcurl through
    ``http.curloptResolve``.  Merely checking DNS here would leave a second
    resolver lookup vulnerable to rebinding between validation and fetch.
    """

    parsed = _validate_https_url(url)
    host = _canonical_host(parsed.hostname or "")
    port = parsed.port or 443
    if host != _canonical_host(allowed_host) or port != allowed_port:
        raise RepositorySecurityError(
            "remote repository endpoint is outside its acquisition allowlist"
        )

    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    try:
        if literal is not None:
            raw_addresses = (str(literal),)
        else:
            answers = socket.getaddrinfo(
                host,
                port,
                family=socket.AF_UNSPEC,
                type=socket.SOCK_STREAM,
                proto=socket.IPPROTO_TCP,
            )
            raw_addresses = tuple(answer[4][0] for answer in answers)
    except OSError as exc:
        raise RepositoryPreparationError(
            "remote repository endpoint could not be resolved"
        ) from exc

    addresses: Dict[bytes, str] = {}
    for raw in raw_addresses:
        try:
            address = ipaddress.ip_address(raw.split("%", 1)[0])
        except ValueError as exc:
            raise RepositorySecurityError(
                "remote repository DNS returned an invalid address"
            ) from exc
        if not address.is_global:
            raise RepositorySecurityError(
                "remote repository DNS returned a non-public address"
            )
        addresses[address.packed] = str(address)
        if len(addresses) > 32:
            raise RepositoryLimitError(
                "remote repository DNS answer exceeds its fixed address budget"
            )
    if not addresses:
        raise RepositoryPreparationError(
            "remote repository endpoint returned no usable address"
        )
    return tuple(addresses[key] for key in sorted(addresses))


def _curl_resolve_arguments(
    *, host: str, port: int, addresses: Sequence[str]
) -> List[str]:
    arguments: List[str] = []
    canonical_host = _canonical_host(host)
    for raw in addresses:
        address = ipaddress.ip_address(raw)
        rendered = "[%s]" % address if address.version == 6 else str(address)
        arguments.extend(
            [
                "-c",
                "http.curloptResolve=%s:%d:%s"
                % (canonical_host, port, rendered),
            ]
        )
    return arguments


def _is_reparse(info: os.stat_result) -> bool:
    return bool(getattr(info, "st_file_attributes", 0) & _REPARSE_POINT)


def _is_link_or_reparse(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or _is_reparse(info)


def _normalized_path(value: os.PathLike[str] | str) -> str:
    result = os.path.abspath(os.fspath(value))
    if os.name == "nt":
        if result.startswith("\\\\?\\UNC\\"):
            result = "\\\\" + result[8:]
        elif result.startswith("\\\\?\\"):
            result = result[4:]
    return os.path.normcase(os.path.normpath(result))


def _path_within(root: Path, target: Path) -> bool:
    root_text = _normalized_path(root)
    target_text = _normalized_path(target)
    try:
        return os.path.commonpath((root_text, target_text)) == root_text
    except ValueError:
        return False


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    left_inode = getattr(left, "st_ino", 0)
    right_inode = getattr(right, "st_ino", 0)
    if not left_inode or not right_inode:
        return False
    return (
        getattr(left, "st_dev", 0),
        left_inode,
    ) == (getattr(right, "st_dev", 0), right_inode)


def _absolute_path(value: os.PathLike[str] | str) -> Path:
    raw = os.fspath(value)
    if "\x00" in raw:
        raise RepositorySecurityError("filesystem path contains NUL")
    return Path(os.path.abspath(raw))


def _path_prefixes(path: Path) -> Iterator[Path]:
    parts = path.parts
    if not parts:
        return
    current = Path(parts[0])
    yield current
    for component in parts[1:]:
        current = current / component
        yield current


def _ensure_secure_directory(
    value: os.PathLike[str] | str,
    *,
    create: bool,
    context: str,
) -> Path:
    path = _absolute_path(value)
    if os.name == "nt" and str(path).startswith("\\\\"):
        raise RepositorySecurityError("%s may not use a UNC path" % context)
    for candidate in _path_prefixes(path):
        try:
            info = os.lstat(candidate)
        except FileNotFoundError:
            if not create:
                raise RepositorySecurityError("%s does not exist" % context)
            try:
                os.mkdir(candidate, 0o700)
            except FileExistsError:
                pass
            except OSError as exc:
                raise RepositorySecurityError(
                    "could not create secure %s" % context
                ) from exc
            try:
                info = os.lstat(candidate)
            except OSError as exc:
                raise RepositorySecurityError(
                    "could not verify newly created %s" % context
                ) from exc
        except OSError as exc:
            raise RepositorySecurityError("could not inspect %s" % context) from exc
        if _is_link_or_reparse(info):
            raise RepositorySecurityError(
                "%s may not traverse a symlink or reparse point" % context
            )
        if not stat.S_ISDIR(info.st_mode):
            raise RepositorySecurityError("%s must be a directory" % context)
    return path


def _assert_secure_directory(path: Path, context: str) -> os.stat_result:
    secured = _ensure_secure_directory(path, create=False, context=context)
    if _normalized_path(secured) != _normalized_path(path):
        raise RepositorySecurityError("%s resolved unexpectedly" % context)
    return os.lstat(secured)


def _secure_join(root: Path, relative_posix: str, context: str) -> Path:
    if type(relative_posix) is not str or not relative_posix:
        raise RepositorySecurityError("%s must be a relative path" % context)
    if relative_posix.startswith(("/", "\\")) or "\\" in relative_posix:
        raise RepositorySecurityError("%s must use relative POSIX form" % context)
    parts = relative_posix.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise RepositorySecurityError("%s contains traversal" % context)
    try:
        _validate_repo_path(parts, context)
    except RepositoryPolicyError as exc:
        raise RepositorySecurityError(
            "%s is not a portable canonical repository path" % context
        ) from exc
    target = root.joinpath(*parts)
    if not _path_within(root, target):
        raise RepositorySecurityError("%s escapes its configured root" % context)
    current = root
    _assert_secure_directory(root, "%s root" % context)
    for index, component in enumerate(parts):
        current = current / component
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            if index != len(parts) - 1:
                raise RepositorySecurityError("%s has a missing parent" % context)
            break
        except OSError as exc:
            raise RepositorySecurityError("could not inspect %s" % context) from exc
        if _is_link_or_reparse(info):
            raise RepositorySecurityError(
                "%s traverses a symlink or reparse point" % context
            )
        if index != len(parts) - 1 and not stat.S_ISDIR(info.st_mode):
            raise RepositorySecurityError("%s parent is not a directory" % context)
    return target


def _validate_direct_child(root: Path, target: Path, context: str) -> Tuple[Path, Path]:
    root = _absolute_path(root)
    target = _absolute_path(target)
    _assert_secure_directory(root, "%s root" % context)
    if _normalized_path(target.parent) != _normalized_path(root):
        raise RepositorySecurityError("%s must be a direct child of its root" % context)
    if not _path_within(root, target) or _normalized_path(root) == _normalized_path(target):
        raise RepositorySecurityError("%s escapes its root" % context)
    return root, target


@dataclass
class _SnapshotBudget:
    maximum_bytes: int
    maximum_nodes: int
    context: str
    bytes_copied: int = 0
    nodes_copied: int = 0

    def add_node(self) -> None:
        self.nodes_copied += 1
        if self.nodes_copied > self.maximum_nodes:
            raise RepositoryLimitError(
                "%s exceeds its filesystem-node budget" % self.context
            )

    def reserve_bytes(self, size: int) -> None:
        if size < 0:
            raise RepositorySecurityError(
                "%s contains a file with an invalid size" % self.context
            )
        self.bytes_copied += size
        if self.bytes_copied > self.maximum_bytes:
            raise RepositoryLimitError(
                "%s exceeds its snapshot byte budget" % self.context
            )


def _validate_snapshot_component(name: str, context: str) -> None:
    if type(name) is not str or not name or name in {".", ".."}:
        raise RepositorySecurityError("%s contains an invalid name" % context)
    if "/" in name or "\\" in name or "\x00" in name:
        raise RepositorySecurityError("%s contains path traversal syntax" % context)
    if unicodedata.normalize("NFC", name) != name:
        raise RepositoryPolicyError("%s contains a non-NFC name" % context)
    try:
        encoded = name.encode("utf-8", "strict")
    except UnicodeEncodeError as exc:
        raise RepositoryPolicyError("%s contains a non-UTF-8 name" % context) from exc
    if len(encoded) > MAX_PATH_COMPONENT_BYTES:
        raise RepositoryLimitError("%s contains an oversized name" % context)
    if any(ord(character) < 32 or ord(character) == 127 for character in name):
        raise RepositoryPolicyError("%s contains a control character" % context)


def _copy_regular_descriptor(
    source_descriptor: int,
    expected: os.stat_result,
    destination: Path,
    budget: _SnapshotBudget,
) -> None:
    opened = os.fstat(source_descriptor)
    if _is_link_or_reparse(opened) or not stat.S_ISREG(opened.st_mode):
        raise RepositorySecurityError(
            "%s changed into an unsafe file" % budget.context
        )
    if getattr(expected, "st_ino", 0) and not _same_identity(expected, opened):
        raise RepositorySecurityError(
            "%s file identity changed during snapshot" % budget.context
        )
    if opened.st_size > budget.maximum_bytes:
        raise RepositoryLimitError(
            "%s contains an oversized file" % budget.context
        )
    budget.reserve_bytes(int(opened.st_size))
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        destination_descriptor = os.open(destination, flags, 0o600)
    except OSError as exc:
        raise RepositorySecurityError(
            "could not create repository source snapshot"
        ) from exc
    copied = 0
    try:
        while True:
            chunk = os.read(source_descriptor, 1024 * 1024)
            if not chunk:
                break
            copied += len(chunk)
            if copied > opened.st_size:
                raise RepositorySecurityError(
                    "%s file grew during snapshot" % budget.context
                )
            offset = 0
            while offset < len(chunk):
                written = os.write(destination_descriptor, chunk[offset:])
                if written <= 0:
                    raise OSError("short snapshot write")
                offset += written
        after = os.fstat(source_descriptor)
        opened_mtime = int(
            getattr(opened, "st_mtime_ns", int(opened.st_mtime * 1_000_000_000))
        )
        after_mtime = int(
            getattr(after, "st_mtime_ns", int(after.st_mtime * 1_000_000_000))
        )
        if (
            copied != opened.st_size
            or after.st_size != opened.st_size
            or after_mtime != opened_mtime
            or (
                getattr(opened, "st_ino", 0)
                and not _same_identity(opened, after)
            )
        ):
            raise RepositorySecurityError(
                "%s file changed during snapshot" % budget.context
            )
        os.fsync(destination_descriptor)
    except OSError as exc:
        raise RepositoryPreparationError(
            "repository source snapshot write failed"
        ) from exc
    finally:
        os.close(destination_descriptor)


def _posix_directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )


def _copy_tree_from_posix_descriptor(
    source_descriptor: int,
    destination: Path,
    budget: _SnapshotBudget,
) -> None:
    before = os.fstat(source_descriptor)
    if _is_link_or_reparse(before) or not stat.S_ISDIR(before.st_mode):
        raise RepositorySecurityError(
            "%s contains an unsafe directory" % budget.context
        )
    try:
        entries = sorted(os.scandir(source_descriptor), key=lambda item: item.name)
    except OSError as exc:
        raise RepositorySecurityError(
            "could not enumerate %s" % budget.context
        ) from exc
    for entry in entries:
        name = entry.name
        _validate_snapshot_component(name, budget.context)
        budget.add_node()
        try:
            info = os.stat(
                name,
                dir_fd=source_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise RepositorySecurityError(
                "could not inspect %s node" % budget.context
            ) from exc
        if _is_link_or_reparse(info):
            raise RepositorySecurityError(
                "%s contains a symlink or reparse point" % budget.context
            )
        child_destination = destination / name
        if stat.S_ISDIR(info.st_mode):
            _mkdir_exclusive(child_destination)
            try:
                child_descriptor = os.open(
                    name,
                    _posix_directory_flags(),
                    dir_fd=source_descriptor,
                )
            except OSError as exc:
                raise RepositorySecurityError(
                    "could not open %s directory" % budget.context
                ) from exc
            try:
                opened = os.fstat(child_descriptor)
                if getattr(info, "st_ino", 0) and not _same_identity(info, opened):
                    raise RepositorySecurityError(
                        "%s directory identity changed" % budget.context
                    )
                _copy_tree_from_posix_descriptor(
                    child_descriptor, child_destination, budget
                )
            finally:
                os.close(child_descriptor)
        elif stat.S_ISREG(info.st_mode):
            flags = (
                os.O_RDONLY
                | getattr(os, "O_BINARY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0)
            )
            try:
                source_file = os.open(name, flags, dir_fd=source_descriptor)
            except OSError as exc:
                raise RepositorySecurityError(
                    "could not open %s file" % budget.context
                ) from exc
            try:
                _copy_regular_descriptor(
                    source_file, info, child_destination, budget
                )
            finally:
                os.close(source_file)
        else:
            raise RepositorySecurityError(
                "%s contains a special filesystem node" % budget.context
            )


def _open_posix_relative_directory(
    root: Path,
    relative_posix: str,
    *,
    expected_root_identity: Tuple[int, int],
    context: str,
) -> int:
    if os.name == "nt" or os.open not in getattr(os, "supports_dir_fd", set()):
        raise RepositorySecurityError(
            "descriptor-relative repository snapshots are unavailable"
        )
    _secure_join(root, relative_posix, context)
    try:
        descriptor = os.open(root, _posix_directory_flags())
    except OSError as exc:
        raise RepositorySecurityError("could not open %s root" % context) from exc
    try:
        root_info = os.fstat(descriptor)
        identity = (
            int(getattr(root_info, "st_dev", 0)),
            int(getattr(root_info, "st_ino", 0)),
        )
        if identity != expected_root_identity:
            raise RepositorySecurityError("%s root identity changed" % context)
        for component in relative_posix.split("/"):
            try:
                next_descriptor = os.open(
                    component,
                    _posix_directory_flags(),
                    dir_fd=descriptor,
                )
            except OSError as exc:
                raise RepositorySecurityError(
                    "%s path changed or became unsafe" % context
                ) from exc
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _windows_open_directory_handle(path: Path) -> int:
    if os.name != "nt":
        raise RepositorySecurityError("Windows directory handles are unavailable")
    try:
        import ctypes
        from ctypes import wintypes

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
        kernel32.SetHandleInformation.argtypes = (
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.DWORD,
        )
        kernel32.SetHandleInformation.restype = wintypes.BOOL
        handle = create_file(
            str(path),
            0x0080,
            0x00000001 | 0x00000002,
            None,
            3,
            0x02000000 | 0x00200000,
            None,
        )
        invalid = ctypes.c_void_p(-1).value
        value = int(getattr(handle, "value", handle) or 0)
        if not value or value == invalid:
            raise OSError(ctypes.get_last_error(), "CreateFileW directory failed")
        return value
    except (ImportError, OSError, TypeError, ValueError) as exc:
        raise RepositorySecurityError(
            "could not hold a Windows repository directory"
        ) from exc


def _windows_close_handle(handle: int) -> None:
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    close_handle(wintypes.HANDLE(handle))


def _windows_handle_path_and_attributes(handle: int) -> Tuple[Path, int]:
    try:
        import ctypes
        from ctypes import wintypes

        class ByHandleFileInformation(ctypes.Structure):
            _fields_ = (
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
            )

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_path = kernel32.GetFinalPathNameByHandleW
        get_path.argtypes = (
            wintypes.HANDLE,
            wintypes.LPWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
        )
        get_path.restype = wintypes.DWORD
        raw_handle = wintypes.HANDLE(handle)
        required = get_path(raw_handle, None, 0, 0)
        if not required:
            raise OSError(ctypes.get_last_error(), "GetFinalPathNameByHandleW failed")
        buffer = ctypes.create_unicode_buffer(required + 1)
        written = get_path(raw_handle, buffer, len(buffer), 0)
        if not written or written >= len(buffer):
            raise OSError(ctypes.get_last_error(), "GetFinalPathNameByHandleW failed")
        value = buffer.value
        get_info = kernel32.GetFileInformationByHandle
        get_info.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(ByHandleFileInformation),
        )
        get_info.restype = wintypes.BOOL
        information = ByHandleFileInformation()
        if not get_info(raw_handle, ctypes.byref(information)):
            raise OSError(
                ctypes.get_last_error(), "GetFileInformationByHandle failed"
            )
    except (ImportError, OSError, TypeError, ValueError) as exc:
        raise RepositorySecurityError(
            "could not verify a Windows repository directory handle"
        ) from exc
    if value.startswith("\\\\?\\UNC\\"):
        value = "\\\\" + value[8:]
    elif value.startswith("\\\\?\\"):
        value = value[4:]
    return Path(value), int(information.dwFileAttributes)


@contextmanager
def _guard_windows_directory_chain(path: Path) -> Iterator[None]:
    handles: List[int] = []
    try:
        for candidate in _path_prefixes(_absolute_path(path)):
            handle = _windows_open_directory_handle(candidate)
            handles.append(handle)
            actual, attributes = _windows_handle_path_and_attributes(handle)
            if (
                _normalized_path(actual) != _normalized_path(candidate)
                or attributes & _REPARSE_POINT
                or not attributes & 0x10
            ):
                raise RepositorySecurityError(
                    "Windows repository path changed or crossed a reparse point"
                )
        yield
    finally:
        for handle in reversed(handles):
            _windows_close_handle(handle)


def _windows_open_regular_descriptor(path: Path) -> int:
    try:
        import ctypes
        import msvcrt
        from ctypes import wintypes

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
            0x80000000,
            0x00000001,
            None,
            3,
            0x00200000 | 0x08000000,
            None,
        )
        invalid = ctypes.c_void_p(-1).value
        value = int(getattr(handle, "value", handle) or 0)
        if not value or value == invalid:
            raise OSError(ctypes.get_last_error(), "CreateFileW file failed")
        try:
            descriptor = msvcrt.open_osfhandle(
                value, os.O_RDONLY | getattr(os, "O_BINARY", 0)
            )
        except BaseException:
            _windows_close_handle(value)
            raise
        info = os.fstat(descriptor)
        if _is_link_or_reparse(info) or not stat.S_ISREG(info.st_mode):
            os.close(descriptor)
            raise OSError("Windows source file is unsafe")
        return descriptor
    except (ImportError, OSError, TypeError, ValueError) as exc:
        raise RepositorySecurityError(
            "could not hold a Windows repository source file"
        ) from exc


def _windows_current_user_sid() -> str:
    try:
        import ctypes
        from ctypes import wintypes

        advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = ()
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        advapi32.OpenProcessToken.argtypes = (
            wintypes.HANDLE,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.HANDLE),
        )
        advapi32.OpenProcessToken.restype = wintypes.BOOL
        advapi32.GetTokenInformation.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        )
        advapi32.GetTokenInformation.restype = wintypes.BOOL
        advapi32.ConvertSidToStringSidW.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(wintypes.LPWSTR),
        )
        advapi32.ConvertSidToStringSidW.restype = wintypes.BOOL
        kernel32.LocalFree.argtypes = (ctypes.c_void_p,)
        kernel32.LocalFree.restype = ctypes.c_void_p
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        token = wintypes.HANDLE()
        if not advapi32.OpenProcessToken(
            kernel32.GetCurrentProcess(), 0x0008, ctypes.byref(token)
        ):
            raise OSError(ctypes.get_last_error(), "OpenProcessToken failed")
        try:
            required = wintypes.DWORD()
            advapi32.GetTokenInformation(
                token, 1, None, 0, ctypes.byref(required)
            )
            if not required.value:
                raise OSError(
                    ctypes.get_last_error(), "GetTokenInformation size failed"
                )
            buffer = ctypes.create_string_buffer(required.value)
            if not advapi32.GetTokenInformation(
                token,
                1,
                buffer,
                required.value,
                ctypes.byref(required),
            ):
                raise OSError(
                    ctypes.get_last_error(), "GetTokenInformation failed"
                )
            sid_pointer = ctypes.cast(
                buffer, ctypes.POINTER(ctypes.c_void_p)
            ).contents.value
            string_sid = wintypes.LPWSTR()
            if not advapi32.ConvertSidToStringSidW(
                ctypes.c_void_p(sid_pointer), ctypes.byref(string_sid)
            ):
                raise OSError(
                    ctypes.get_last_error(), "ConvertSidToStringSidW failed"
                )
            try:
                return str(string_sid.value)
            finally:
                kernel32.LocalFree(string_sid)
        finally:
            kernel32.CloseHandle(token)
    except (ImportError, OSError, TypeError, ValueError) as exc:
        raise RepositorySecurityError(
            "could not determine the Windows repository owner identity"
        ) from exc


def _windows_path_owner_sid(path: Path) -> str:
    try:
        import ctypes
        from ctypes import wintypes

        advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        advapi32.GetNamedSecurityInfoW.argtypes = (
            wintypes.LPWSTR,
            ctypes.c_int,
            wintypes.DWORD,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        )
        advapi32.GetNamedSecurityInfoW.restype = wintypes.DWORD
        advapi32.ConvertSidToStringSidW.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(wintypes.LPWSTR),
        )
        advapi32.ConvertSidToStringSidW.restype = wintypes.BOOL
        kernel32.LocalFree.argtypes = (ctypes.c_void_p,)
        kernel32.LocalFree.restype = ctypes.c_void_p
        owner = ctypes.c_void_p()
        descriptor = ctypes.c_void_p()
        status = advapi32.GetNamedSecurityInfoW(
            str(path),
            1,
            0x00000001,
            ctypes.byref(owner),
            None,
            None,
            None,
            ctypes.byref(descriptor),
        )
        if status != 0:
            raise OSError(int(status), "GetNamedSecurityInfoW failed")
        try:
            string_sid = wintypes.LPWSTR()
            if not advapi32.ConvertSidToStringSidW(
                owner, ctypes.byref(string_sid)
            ):
                raise OSError(
                    ctypes.get_last_error(), "ConvertSidToStringSidW failed"
                )
            try:
                return str(string_sid.value)
            finally:
                kernel32.LocalFree(string_sid)
        finally:
            kernel32.LocalFree(descriptor)
    except (ImportError, OSError, TypeError, ValueError) as exc:
        raise RepositorySecurityError(
            "could not verify the Windows repository directory owner"
        ) from exc


def _replace_and_verify_windows_control_acl(
    path: Path, context: str, current_sid: str
) -> None:
    if _windows_path_owner_sid(path) != current_sid:
        raise RepositorySecurityError(
            "%s is not owned by the current Windows identity" % context
        )
    if re.fullmatch(r"S-[0-9-]+", current_sid) is None:
        raise RepositorySecurityError("current Windows SID is malformed")
    try:
        path_info = os.lstat(path)
    except OSError as exc:
        raise RepositorySecurityError("could not inspect %s ACL" % context) from exc
    is_directory = stat.S_ISDIR(path_info.st_mode)
    try:
        import ctypes
        from ctypes import wintypes

        advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = (
            wintypes.LPCWSTR,
            wintypes.DWORD,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(wintypes.DWORD),
        )
        advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = (
            wintypes.BOOL
        )
        advapi32.GetSecurityDescriptorDacl.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(wintypes.BOOL),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(wintypes.BOOL),
        )
        advapi32.GetSecurityDescriptorDacl.restype = wintypes.BOOL
        advapi32.SetNamedSecurityInfoW.argtypes = (
            wintypes.LPWSTR,
            ctypes.c_int,
            wintypes.DWORD,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        )
        advapi32.SetNamedSecurityInfoW.restype = wintypes.DWORD
        advapi32.GetNamedSecurityInfoW.argtypes = (
            wintypes.LPWSTR,
            ctypes.c_int,
            wintypes.DWORD,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        )
        advapi32.GetNamedSecurityInfoW.restype = wintypes.DWORD
        advapi32.GetSecurityDescriptorControl.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ushort),
            ctypes.POINTER(wintypes.DWORD),
        )
        advapi32.GetSecurityDescriptorControl.restype = wintypes.BOOL
        advapi32.GetAclInformation.argtypes = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.c_int,
        )
        advapi32.GetAclInformation.restype = wintypes.BOOL
        advapi32.GetAce.argtypes = (
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(ctypes.c_void_p),
        )
        advapi32.GetAce.restype = wintypes.BOOL
        advapi32.ConvertSidToStringSidW.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(wintypes.LPWSTR),
        )
        advapi32.ConvertSidToStringSidW.restype = wintypes.BOOL
        kernel32.LocalFree.argtypes = (ctypes.c_void_p,)
        kernel32.LocalFree.restype = ctypes.c_void_p

        inheritance = "OICI" if is_directory else ""
        sddl = "D:P(A;%s;FA;;;%s)(A;%s;FA;;;SY)(A;%s;FA;;;BA)" % (
            inheritance,
            current_sid,
            inheritance,
            inheritance,
        )
        expected_descriptor = ctypes.c_void_p()
        if not advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
            sddl, 1, ctypes.byref(expected_descriptor), None
        ):
            raise OSError(
                ctypes.get_last_error(), "could not build protected DACL"
            )
        try:
            present = wintypes.BOOL()
            defaulted = wintypes.BOOL()
            expected_acl = ctypes.c_void_p()
            if not advapi32.GetSecurityDescriptorDacl(
                expected_descriptor,
                ctypes.byref(present),
                ctypes.byref(expected_acl),
                ctypes.byref(defaulted),
            ) or not present.value or not expected_acl.value:
                raise OSError("protected DACL is unavailable")
            status = advapi32.SetNamedSecurityInfoW(
                str(path),
                1,
                0x00000004 | 0x80000000,
                None,
                None,
                expected_acl,
                None,
            )
            if status != 0:
                raise OSError(int(status), "SetNamedSecurityInfoW failed")
        finally:
            kernel32.LocalFree(expected_descriptor)

        actual_acl = ctypes.c_void_p()
        actual_descriptor = ctypes.c_void_p()
        status = advapi32.GetNamedSecurityInfoW(
            str(path),
            1,
            0x00000004,
            None,
            None,
            ctypes.byref(actual_acl),
            None,
            ctypes.byref(actual_descriptor),
        )
        if status != 0:
            raise OSError(int(status), "GetNamedSecurityInfoW failed")
        try:
            control = ctypes.c_ushort()
            revision = wintypes.DWORD()
            if not advapi32.GetSecurityDescriptorControl(
                actual_descriptor,
                ctypes.byref(control),
                ctypes.byref(revision),
            ) or not control.value & 0x1000:
                raise OSError("Windows control DACL is not protected")

            class AclSizeInformation(ctypes.Structure):
                _fields_ = (
                    ("AceCount", wintypes.DWORD),
                    ("AclBytesInUse", wintypes.DWORD),
                    ("AclBytesFree", wintypes.DWORD),
                )

            information = AclSizeInformation()
            if not advapi32.GetAclInformation(
                actual_acl,
                ctypes.byref(information),
                ctypes.sizeof(information),
                2,
            ) or information.AceCount != 3:
                raise OSError("Windows control DACL has unexpected ACEs")
            expected_sids = {
                current_sid,
                "S-1-5-18",
                "S-1-5-32-544",
            }
            actual_sids: Set[str] = set()
            for index in range(information.AceCount):
                ace = ctypes.c_void_p()
                if not advapi32.GetAce(
                    actual_acl, index, ctypes.byref(ace)
                ):
                    raise OSError("GetAce failed")
                ace_type, ace_flags, ace_size, access_mask = struct.unpack(
                    "<BBHI", ctypes.string_at(ace, 8)
                )
                if (
                    ace_type != 0
                    or ace_flags != (0x03 if is_directory else 0x00)
                    or ace_size < 16
                    or access_mask != 0x001F01FF
                ):
                    raise OSError("Windows control DACL ACE is not exact")
                sid_text = wintypes.LPWSTR()
                if not advapi32.ConvertSidToStringSidW(
                    ctypes.c_void_p(ace.value + 8), ctypes.byref(sid_text)
                ):
                    raise OSError("ConvertSidToStringSidW failed")
                try:
                    actual_sids.add(str(sid_text.value))
                finally:
                    kernel32.LocalFree(sid_text)
            if actual_sids != expected_sids:
                raise OSError("Windows control DACL trustee set is not exact")
        finally:
            kernel32.LocalFree(actual_descriptor)
    except (ImportError, OSError, TypeError, ValueError) as exc:
        raise RepositorySecurityError("could not harden %s ACL" % context) from exc
    if _windows_path_owner_sid(path) != current_sid:
        raise RepositorySecurityError(
            "%s owner changed while its ACL was hardened" % context
        )


def _harden_windows_control_directory(path: Path, context: str) -> None:
    _replace_and_verify_windows_control_acl(
        path, context, _windows_current_user_sid()
    )


def _harden_windows_control_tree(path: Path, context: str) -> None:
    if os.name != "nt":
        return
    current_sid = _windows_current_user_sid()
    stack = [path]
    nodes = 0
    while stack:
        current = stack.pop()
        try:
            info = os.lstat(current)
        except OSError as exc:
            raise RepositorySecurityError(
                "could not inspect %s ACL tree" % context
            ) from exc
        if _is_link_or_reparse(info) or not (
            stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)
        ):
            raise RepositorySecurityError(
                "%s ACL tree contains an unsafe node" % context
            )
        _replace_and_verify_windows_control_acl(
            current, context, current_sid
        )
        nodes += 1
        if nodes > MAX_DATA_ROOT_NODES:
            raise RepositoryLimitError(
                "%s ACL tree exceeds its node budget" % context
            )
        if stat.S_ISDIR(info.st_mode):
            try:
                children = list(os.scandir(current))
            except OSError as exc:
                raise RepositorySecurityError(
                    "could not enumerate %s ACL tree" % context
                ) from exc
            stack.extend(Path(child.path) for child in children)


def _secure_control_directory_authority(path: Path, context: str) -> None:
    info = _assert_secure_directory(path, context)
    if os.name == "nt":
        _harden_windows_control_directory(path, context)
        return
    current = int(getattr(os, "geteuid", lambda: -1)())
    owner = int(getattr(info, "st_uid", -2))
    if owner != current:
        raise RepositorySecurityError(
            "%s must be owned by the current user" % context
        )
    try:
        os.chmod(path, 0o700)
    except OSError as exc:
        raise RepositorySecurityError(
            "could not restrict %s permissions" % context
        ) from exc
    verified = os.lstat(path)
    if verified.st_mode & 0o077:
        raise RepositorySecurityError("%s permissions are not private" % context)


def _verify_control_directory_authority(path: Path, context: str) -> None:
    """Verify an already prepared control root without changing ACL/mode bits."""

    info = _assert_secure_directory(path, context)
    if os.name == "nt":
        if _windows_path_owner_sid(path) != _windows_current_user_sid():
            raise RepositorySecurityError(
                "%s must be owned by the current user" % context
            )
        return
    current = int(getattr(os, "geteuid", lambda: -1)())
    owner = int(getattr(info, "st_uid", -2))
    if owner != current or info.st_mode & 0o077:
        raise RepositorySecurityError(
            "%s permissions are not private" % context
        )


def _assert_suite_directory_authority(path: Path) -> None:
    info = _assert_secure_directory(path, "repository Suite root")
    if os.name == "nt":
        return
    current = int(getattr(os, "geteuid", lambda: -1)())
    owner = int(getattr(info, "st_uid", -2))
    if owner not in {0, current} or info.st_mode & 0o022:
        raise RepositorySecurityError(
            "repository Suite root owner or write permissions are unsafe"
        )


def _hold_windows_directory_chains(paths: Iterable[Path]) -> List[int]:
    if os.name != "nt":
        return []
    handles: List[int] = []
    seen: Set[str] = set()
    try:
        for path in paths:
            for candidate in _path_prefixes(_absolute_path(path)):
                normalized = _normalized_path(candidate)
                if normalized in seen:
                    continue
                handle = _windows_open_directory_handle(candidate)
                actual, attributes = _windows_handle_path_and_attributes(handle)
                if (
                    _normalized_path(actual) != normalized
                    or attributes & _REPARSE_POINT
                    or not attributes & 0x10
                ):
                    _windows_close_handle(handle)
                    raise RepositorySecurityError(
                        "Windows repository root chain is unsafe"
                    )
                handles.append(handle)
                seen.add(normalized)
        return handles
    except BaseException:
        for handle in reversed(handles):
            _windows_close_handle(handle)
        raise


def _copy_tree_windows(
    source: Path,
    destination: Path,
    budget: _SnapshotBudget,
) -> None:
    with _guard_windows_directory_chain(source):
        try:
            entries = sorted(os.scandir(source), key=lambda item: item.name)
        except OSError as exc:
            raise RepositorySecurityError(
                "could not enumerate %s" % budget.context
            ) from exc
        for entry in entries:
            name = entry.name
            _validate_snapshot_component(name, budget.context)
            budget.add_node()
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise RepositorySecurityError(
                    "could not inspect %s node" % budget.context
                ) from exc
            if _is_link_or_reparse(info):
                raise RepositorySecurityError(
                    "%s contains a symlink or reparse point" % budget.context
                )
            child_source = Path(entry.path)
            child_destination = destination / name
            if stat.S_ISDIR(info.st_mode):
                _mkdir_exclusive(child_destination)
                _copy_tree_windows(child_source, child_destination, budget)
            elif stat.S_ISREG(info.st_mode):
                descriptor = _windows_open_regular_descriptor(child_source)
                try:
                    _copy_regular_descriptor(
                        descriptor, info, child_destination, budget
                    )
                finally:
                    os.close(descriptor)
            else:
                raise RepositorySecurityError(
                    "%s contains a special filesystem node" % budget.context
                )


def _copy_relative_directory_snapshot(
    root: Path,
    relative_posix: str,
    destination: Path,
    *,
    expected_root_identity: Tuple[int, int],
    maximum_bytes: int,
    maximum_nodes: int,
    context: str,
) -> None:
    if os.path.lexists(destination):
        raise RepositorySecurityError("repository snapshot destination exists")
    _assert_secure_directory(destination.parent, "repository snapshot parent")
    _mkdir_exclusive(destination)
    budget = _SnapshotBudget(maximum_bytes, maximum_nodes, context)
    if os.name == "nt":
        source = _secure_join(root, relative_posix, context)
        with _guard_windows_directory_chain(root):
            if _filesystem_identity(root, "%s root" % context) != expected_root_identity:
                raise RepositorySecurityError("%s root identity changed" % context)
            _copy_tree_windows(source, destination, budget)
        return
    source_descriptor = _open_posix_relative_directory(
        root,
        relative_posix,
        expected_root_identity=expected_root_identity,
        context=context,
    )
    try:
        _copy_tree_from_posix_descriptor(source_descriptor, destination, budget)
    finally:
        os.close(source_descriptor)


def _copy_local_git_snapshot(
    root: Path,
    relative_posix: str,
    destination: Path,
    *,
    expected_root_identity: Tuple[int, int],
) -> None:
    context = "local Git source"
    if os.path.lexists(destination):
        raise RepositorySecurityError("local Git snapshot destination exists")
    _assert_secure_directory(destination.parent, "local Git snapshot parent")
    _mkdir_exclusive(destination)
    budget = _SnapshotBudget(MAX_CACHE_BYTES, MAX_GIT_METADATA_NODES, context)
    if os.name == "nt":
        source = _secure_join(root, relative_posix, context)
        with _guard_windows_directory_chain(root):
            if _filesystem_identity(root, "local Git source root") != expected_root_identity:
                raise RepositorySecurityError("local Git source root identity changed")
            dot_git = source / ".git"
            if os.path.lexists(dot_git):
                info = os.lstat(dot_git)
                if _is_link_or_reparse(info) or not stat.S_ISDIR(info.st_mode):
                    raise RepositorySecurityError(
                        "local .git indirection or reparse point is rejected"
                    )
                _copy_tree_windows(dot_git, destination, budget)
            else:
                _copy_tree_windows(source, destination, budget)
        return

    source_descriptor = _open_posix_relative_directory(
        root,
        relative_posix,
        expected_root_identity=expected_root_identity,
        context=context,
    )
    selected_descriptor = source_descriptor
    try:
        try:
            dot_git = os.stat(
                ".git", dir_fd=source_descriptor, follow_symlinks=False
            )
        except FileNotFoundError:
            dot_git = None
        except OSError as exc:
            raise RepositorySecurityError("could not inspect local .git") from exc
        if dot_git is not None:
            if _is_link_or_reparse(dot_git) or not stat.S_ISDIR(dot_git.st_mode):
                raise RepositorySecurityError(
                    "local .git indirection or symlink is rejected"
                )
            selected_descriptor = os.open(
                ".git", _posix_directory_flags(), dir_fd=source_descriptor
            )
        _copy_tree_from_posix_descriptor(
            selected_descriptor, destination, budget
        )
    finally:
        if selected_descriptor != source_descriptor:
            os.close(selected_descriptor)
        os.close(source_descriptor)


def _validate_component(name: str, context: str) -> bytes:
    if type(name) is not str or not name:
        raise RepositoryPolicyError("%s contains an empty path component" % context)
    if unicodedata.normalize("NFC", name) != name:
        raise RepositoryPolicyError("%s path is not NFC-normalized" % context)
    try:
        encoded = name.encode("utf-8", "strict")
    except UnicodeEncodeError as exc:
        raise RepositoryPolicyError("%s path is not strict UTF-8" % context) from exc
    if len(encoded) > MAX_PATH_COMPONENT_BYTES:
        raise RepositoryLimitError("%s path component exceeds byte budget" % context)
    if name in {".", ".."} or "/" in name or "\\" in name:
        raise RepositoryPolicyError("%s path component is unsafe" % context)
    if any(ord(character) < 32 or ord(character) == 127 for character in name):
        raise RepositoryPolicyError("%s path contains a control character" % context)
    if name.endswith((".", " ")):
        raise RepositoryPolicyError("%s path has a Windows trailing dot or space" % context)
    if any(character in _WINDOWS_FORBIDDEN for character in name):
        raise RepositoryPolicyError("%s path contains Windows ADS/forbidden syntax" % context)
    if is_windows_reserved_path_component(name):
        raise RepositoryPolicyError("%s path uses a Windows device name" % context)
    folded = unicodedata.normalize("NFC", name).casefold()
    if folded in _VCS_METADATA or folded == ".lfsconfig":
        raise RepositoryPolicyError(
            "%s contains nested repository, VCS metadata, or LFS metadata" % context
        )
    return encoded


def _validate_repo_path(parts: Sequence[str], context: str) -> bytes:
    if not parts or len(parts) > MAX_PATH_DEPTH:
        raise RepositoryLimitError("%s path exceeds the fixed depth budget" % context)
    encoded = b"/".join(_validate_component(part, context) for part in parts)
    if len(encoded) > MAX_PATH_BYTES:
        raise RepositoryLimitError("%s path exceeds the fixed byte budget" % context)
    return encoded


def _read_regular_file(
    path: Path,
    *,
    maximum: int,
    context: str,
) -> bytes:
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise RepositorySecurityError("could not inspect %s" % context) from exc
    if _is_link_or_reparse(before):
        raise RepositorySecurityError("%s is a symlink or reparse point" % context)
    if not stat.S_ISREG(before.st_mode):
        raise RepositorySecurityError("%s is not a regular file" % context)
    if before.st_size > maximum:
        raise RepositoryLimitError("%s exceeds its byte budget" % context)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RepositorySecurityError("could not open %s safely" % context) from exc
    try:
        opened = os.fstat(descriptor)
        if _is_link_or_reparse(opened) or not stat.S_ISREG(opened.st_mode):
            raise RepositorySecurityError("%s changed to an unsafe node" % context)
        if _same_identity(before, opened) is False and getattr(before, "st_ino", 0):
            raise RepositorySecurityError("%s changed during secure open" % context)
        chunks: List[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                raise RepositoryLimitError("%s exceeds its byte budget" % context)
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if opened.st_size != after.st_size or (
            _same_identity(opened, after) is False and getattr(opened, "st_ino", 0)
        ):
            raise RepositorySecurityError("%s changed while it was read" % context)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _sha256_regular_file(path: Path, *, maximum: int, context: str) -> str:
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise RepositorySecurityError("could not inspect %s" % context) from exc
    if _is_link_or_reparse(before) or not stat.S_ISREG(before.st_mode):
        raise RepositorySecurityError("%s is not a trusted regular file" % context)
    if before.st_size > maximum:
        raise RepositoryLimitError("%s exceeds its byte budget" % context)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RepositorySecurityError("could not open %s" % context) from exc
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if not _same_identity(before, opened):
            raise RepositorySecurityError("%s changed while it was opened" % context)
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                raise RepositoryLimitError("%s exceeds its byte budget" % context)
            digest.update(chunk)
        after = os.fstat(descriptor)
        if not _same_identity(opened, after) or opened.st_size != after.st_size:
            raise RepositorySecurityError("%s changed while it was hashed" % context)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _write_regular_file_exclusive(
    path: Path,
    data: bytes,
    *,
    mode: int = 0o600,
    fsync: bool = False,
) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, mode)
    except OSError as exc:
        raise RepositorySecurityError("could not create repository file safely") from exc
    try:
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            if written <= 0:
                raise OSError("short write")
            offset += written
        if fsync:
            os.fsync(descriptor)
    except OSError as exc:
        raise RepositoryPreparationError("repository file write failed") from exc
    finally:
        os.close(descriptor)


def _secure_tree_usage(
    root: Path,
    *,
    maximum_bytes: int,
    maximum_nodes: int,
    reject_links: bool,
    context: str,
) -> Tuple[int, int]:
    _assert_secure_directory(root, "%s root" % context)
    total = 0
    nodes = 0
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            entries = list(os.scandir(directory))
        except FileNotFoundError:
            # Concurrent prepare cleanup can remove an operation staging tree
            # after it was queued.  Quota accounting is repeated after each
            # prepare, so a vanished transient may be ignored safely.
            continue
        except OSError as exc:
            raise RepositorySecurityError("could not inspect %s" % context) from exc
        for entry in entries:
            nodes += 1
            if nodes > maximum_nodes:
                raise RepositoryLimitError(
                    "%s contains too many filesystem nodes" % context
                )
            try:
                info = entry.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise RepositorySecurityError(
                    "could not inspect %s node" % context
                ) from exc
            if _is_link_or_reparse(info):
                if reject_links:
                    raise RepositorySecurityError(
                        "%s contains a symlink or reparse point" % context
                    )
                continue
            if stat.S_ISDIR(info.st_mode):
                stack.append(Path(entry.path))
            elif stat.S_ISREG(info.st_mode):
                total += info.st_size
                if total > maximum_bytes:
                    return total, nodes
            else:
                raise RepositorySecurityError(
                    "%s contains a special node" % context
                )
    return total, nodes


def _secure_tree_size(root: Path, maximum: int) -> int:
    total, _nodes = _secure_tree_usage(
        root,
        maximum_bytes=maximum,
        maximum_nodes=MAX_GIT_METADATA_NODES,
        reject_links=False,
        context="workspace",
    )
    return total


@dataclass(frozen=True)
class _CommandResult:
    returncode: int
    stdout: bytes
    stderr_digest: str


class _GitCommandFailure(RepositoryPreparationError):
    def __init__(self, returncode: int, stderr_digest: str, *, timed_out: bool = False):
        self.returncode = returncode
        self.stderr_digest = stderr_digest
        self.timed_out = timed_out
        message = "Git command timed out" if timed_out else "Git command failed"
        super().__init__("%s (diagnostic %s)" % (message, stderr_digest))


def _digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _WindowsProcessJob:
    """Kill-on-close Job Object for one Git process tree."""

    def __init__(self) -> None:
        self.handle: Optional[int] = None
        if os.name != "nt":
            return
        import ctypes
        from ctypes import wintypes

        class BasicLimitInformation(ctypes.Structure):
            _fields_ = (
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            )

        class IoCounters(ctypes.Structure):
            _fields_ = tuple(
                (name, ctypes.c_ulonglong)
                for name in (
                    "ReadOperationCount",
                    "WriteOperationCount",
                    "OtherOperationCount",
                    "ReadTransferCount",
                    "WriteTransferCount",
                    "OtherTransferCount",
                )
            )

        class ExtendedLimitInformation(ctypes.Structure):
            _fields_ = (
                ("BasicLimitInformation", BasicLimitInformation),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            )

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_job = kernel32.CreateJobObjectW
        create_job.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
        create_job.restype = wintypes.HANDLE
        handle = create_job(None, None)
        handle_value = int(getattr(handle, "value", handle) or 0)
        if not handle_value:
            raise RepositorySecurityError("could not create Git process Job Object")
        information = ExtendedLimitInformation()
        information.BasicLimitInformation.LimitFlags = 0x00002000
        set_information = kernel32.SetInformationJobObject
        set_information.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        )
        set_information.restype = wintypes.BOOL
        if not set_information(
            wintypes.HANDLE(handle_value),
            9,
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            kernel32.CloseHandle(wintypes.HANDLE(handle_value))
            raise RepositorySecurityError("could not configure Git process Job Object")
        self.handle = handle_value

    def assign(self, process: subprocess.Popen[bytes]) -> None:
        if os.name != "nt" or self.handle is None:
            return
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        assign = kernel32.AssignProcessToJobObject
        assign.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        assign.restype = wintypes.BOOL
        process_handle = wintypes.HANDLE(int(getattr(process, "_handle")))
        if not assign(wintypes.HANDLE(self.handle), process_handle):
            raise RepositorySecurityError("could not contain Git process tree")

    def terminate(self) -> None:
        if os.name != "nt" or self.handle is None:
            return
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        terminate = kernel32.TerminateJobObject
        terminate.argtypes = (wintypes.HANDLE, wintypes.UINT)
        terminate.restype = wintypes.BOOL
        terminate(wintypes.HANDLE(self.handle), 1)

    def close(self) -> None:
        if os.name != "nt" or self.handle is None:
            return
        import ctypes
        from ctypes import wintypes

        ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(
            wintypes.HANDLE(self.handle)
        )
        self.handle = None


def _resume_windows_process(process: subprocess.Popen[bytes]) -> None:
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes

        raw_handle = int(getattr(process, "_handle"))
        ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
        resume = ntdll.NtResumeProcess
        resume.argtypes = (wintypes.HANDLE,)
        resume.restype = ctypes.c_long
        status = int(resume(wintypes.HANDLE(raw_handle)))
        if status < 0:
            raise OSError(status, "NtResumeProcess failed")
    except (ImportError, OSError, TypeError, ValueError) as exc:
        raise RepositorySecurityError(
            "could not resume the contained Windows Git process"
        ) from exc


class _GitRunner:
    """Small, bounded Git process boundary with a deliberately minimal env."""

    def __init__(
        self,
        control_root: Path,
        *,
        git_executable: os.PathLike[str] | str,
        timeout_seconds: float = DEFAULT_GIT_TIMEOUT_SECONDS,
        create_control_roots: bool = True,
    ) -> None:
        if type(create_control_roots) is not bool:
            raise TypeError("create_control_roots must be a bool")
        self.control_root = _ensure_secure_directory(
            control_root,
            create=create_control_roots,
            context="Git control root",
        )
        self.tmp_root = _ensure_secure_directory(
            self.control_root / "tmp",
            create=create_control_roots,
            context="Git temporary root",
        )
        self.home = _ensure_secure_directory(
            self.control_root / "home",
            create=create_control_roots,
            context="Git isolated home",
        )
        self.config_home = _ensure_secure_directory(
            self.control_root / "config",
            create=create_control_roots,
            context="Git isolated config",
        )
        self.hooks = _ensure_secure_directory(
            self.control_root / "hooks-disabled",
            create=create_control_roots,
            context="Git disabled hooks root",
        )
        executable_path = _absolute_path(git_executable)
        if not Path(os.fspath(git_executable)).is_absolute():
            raise ValueError("git_executable must be an explicit absolute path")
        try:
            info = os.lstat(executable_path)
        except OSError as exc:
            raise RepositorySecurityError("Git executable could not be verified") from exc
        if _is_link_or_reparse(info) or not stat.S_ISREG(info.st_mode):
            raise RepositorySecurityError("Git executable is not a regular file")
        if os.name != "nt":
            owner = int(getattr(info, "st_uid", -1))
            current = int(getattr(os, "geteuid", lambda: owner)())
            if owner not in {0, current} or info.st_mode & 0o022:
                raise RepositorySecurityError(
                    "Git executable owner or write permissions are unsafe"
                )
        self.executable = executable_path
        self.executable_identity = (
            int(getattr(info, "st_dev", 0)),
            int(getattr(info, "st_ino", 0)),
            int(info.st_size),
            int(getattr(info, "st_mtime_ns", int(info.st_mtime * 1_000_000_000))),
        )
        self.executable_sha256 = _sha256_regular_file(
            executable_path,
            maximum=MAX_GIT_EXECUTABLE_BYTES,
            context="Git executable",
        )
        self.timeout_seconds = _positive_timeout(timeout_seconds, "Git timeout")
        self.env = self._make_environment()
        self._operation_lease: Optional[_OperationLease] = None
        self.exec_path = self._discover_exec_path()
        self.upload_pack = self._find_git_helper("git-upload-pack")
        self.version = self._read_version()
        self._curlopt_resolve_supported: Optional[bool] = None

    def _assert_executable_identity(self) -> None:
        try:
            info = os.lstat(self.executable)
        except OSError as exc:
            raise RepositorySecurityError("Git executable became unavailable") from exc
        current = (
            int(getattr(info, "st_dev", 0)),
            int(getattr(info, "st_ino", 0)),
            int(info.st_size),
            int(getattr(info, "st_mtime_ns", int(info.st_mtime * 1_000_000_000))),
        )
        if _is_link_or_reparse(info) or current != self.executable_identity:
            raise RepositorySecurityError("Git executable identity changed")

    def _make_environment(self) -> Dict[str, str]:
        executable_parent = str(self.executable.parent)
        candidates = [executable_parent]
        if os.name == "nt":
            system_root = os.environ.get("SystemRoot", r"C:\\Windows")
            git_root = self.executable.parent.parent
            candidates.extend(
                [
                    str(git_root / "cmd"),
                    str(git_root / "mingw64" / "bin"),
                    str(git_root / "usr" / "bin"),
                    os.path.join(system_root, "System32"),
                    system_root,
                ]
            )
        else:
            candidates.extend(["/usr/bin", "/bin"])
        env = {
            "PATH": os.pathsep.join(dict.fromkeys(candidates)),
            "HOME": str(self.home),
            "XDG_CONFIG_HOME": str(self.config_home),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "GCM_INTERACTIVE": "Never",
            "GIT_ASKPASS": "",
            "SSH_ASKPASS": "",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_LFS_SKIP_SMUDGE": "1",
            "GIT_PROTOCOL_FROM_USER": "0",
            "GIT_PAGER": "cat",
            "PAGER": "cat",
            "LC_ALL": "C",
            "LANG": "C",
            "TMP": str(self.tmp_root),
            "TEMP": str(self.tmp_root),
            "TMPDIR": str(self.tmp_root),
            "USERPROFILE": str(self.home),
        }
        for key in ("SystemRoot", "WINDIR", "COMSPEC", "PATHEXT"):
            if key in os.environ:
                env[key] = os.environ[key]
        return env

    def _base_args(self, *, allow_https: bool = False, allow_file: bool = False) -> List[str]:
        args = [
            str(self.executable),
            "-c",
            "core.hooksPath=%s" % str(self.hooks),
            "-c",
            "credential.helper=",
            "-c",
            "filter.lfs.process=",
            "-c",
            "filter.lfs.smudge=",
            "-c",
            "filter.lfs.clean=",
            "-c",
            "filter.lfs.required=false",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "submodule.recurse=false",
            "-c",
            "protocol.allow=never",
            "-c",
            "http.followRedirects=false",
            "-c",
            "http.sslVerify=true",
        ]
        if allow_https:
            args.extend(["-c", "protocol.https.allow=always"])
        if allow_file:
            args.extend(["-c", "protocol.file.allow=always"])
        return args

    def _discover_exec_path(self) -> Path:
        result = self.run(
            ["--exec-path"],
            check=True,
            stdout_limit=4096,
            stderr_limit=MAX_GIT_STDERR_BYTES,
        )
        text = result.stdout.decode("ascii", "strict").strip()
        path = _absolute_path(text)
        _assert_secure_directory(path, "Git exec path")
        return path

    def _find_git_helper(self, name: str) -> Path:
        suffixes = (".exe", "") if os.name == "nt" else ("",)
        candidates = [
            directory / (name + suffix)
            for directory in (self.executable.parent, self.exec_path)
            for suffix in suffixes
        ]
        for candidate in candidates:
            try:
                info = os.lstat(candidate)
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise RepositorySecurityError(
                    "required Git helper could not be inspected"
                ) from exc
            if _is_link_or_reparse(info) or not stat.S_ISREG(info.st_mode):
                raise RepositorySecurityError("required Git helper is unsafe")
            return candidate
        raise RepositoryPreparationError("required Git helper is unavailable")

    def _read_version(self) -> str:
        result = self.run(
            ["--version"],
            check=True,
            stdout_limit=4096,
            stderr_limit=MAX_GIT_STDERR_BYTES,
        )
        try:
            value = result.stdout.decode("utf-8", "strict").strip()
        except UnicodeDecodeError as exc:
            raise RepositoryPreparationError("Git version was not UTF-8") from exc
        return _string(value, "Git version", 256)

    def require_curlopt_resolve(self) -> None:
        if self._curlopt_resolve_supported is None:
            try:
                result = self.run(
                    ["help", "--config"],
                    stdout_limit=MAX_GIT_CONFIG_BYTES,
                )
                names = {
                    line.decode("utf-8", "strict").strip()
                    for line in result.stdout.splitlines()
                    if line.strip()
                }
            except (RepositoryPreparationError, UnicodeDecodeError) as exc:
                raise RepositorySecurityError(
                    "Git remote resolution capability could not be verified"
                ) from exc
            self._curlopt_resolve_supported = "http.curloptResolve" in names
        if not self._curlopt_resolve_supported:
            raise RepositorySecurityError(
                "Git does not support pinned remote endpoint resolution"
            )

    @contextmanager
    def operation_lease(
        self, lease: _OperationLease
    ) -> Iterator[None]:
        if not isinstance(lease, _OperationLease):
            raise TypeError("Git operation lease must be an _OperationLease")
        if self._operation_lease is not None:
            raise RepositorySecurityError("Git operation lease is already bound")
        self._operation_lease = lease
        try:
            yield
        finally:
            self._operation_lease = None

    def _read_spooled(self, handle: BinaryIO, maximum: int) -> bytes:
        handle.flush()
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        if size > maximum:
            raise RepositoryLimitError("Git process output exceeds its fixed budget")
        handle.seek(0)
        return handle.read(maximum + 1)

    def _run_process(
        self,
        argv: Sequence[str],
        *,
        input_bytes: Optional[bytes],
        stdout_limit: int,
        stderr_limit: int,
        stdout_file: Optional[BinaryIO] = None,
        watch_root: Optional[Path] = None,
        watch_limit: Optional[int] = None,
    ) -> _CommandResult:
        if not argv or any(type(item) is not str for item in argv):
            raise RepositorySecurityError("Git argv must contain strings")
        self._assert_executable_identity()
        if input_bytes is not None and len(input_bytes) > MAX_CACHE_BYTES:
            raise RepositoryLimitError("Git input exceeds its fixed budget")
        own_stdout = stdout_file is None
        own_stderr = tempfile.TemporaryFile(dir=str(self.tmp_root))
        if own_stdout:
            stdout_file = tempfile.TemporaryFile(dir=str(self.tmp_root))
        assert stdout_file is not None
        process: Optional[subprocess.Popen[bytes]] = None
        job = _WindowsProcessJob()
        writer: Optional[threading.Thread] = None
        writer_errors: List[BaseException] = []
        reader: Optional[threading.Thread] = None
        reader_errors: List[BaseException] = []
        stdout_limit_exceeded = threading.Event()
        streamed_stdout_bytes = 0

        def terminate_tree() -> None:
            if process is None:
                return
            if os.name == "nt":
                job.terminate()
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if process.poll() is None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)

        try:
            popen_platform: Dict[str, Any] = {}
            if os.name == "nt":
                creationflags = (
                    getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                    | 0x00000004
                )
                popen_platform["creationflags"] = creationflags
                popen_platform["start_new_session"] = False
                if self._operation_lease is not None:
                    startupinfo = subprocess.STARTUPINFO()
                    startupinfo.lpAttributeList = {
                        "handle_list": [
                            self._operation_lease.windows_handle()
                        ]
                    }
                    popen_platform["startupinfo"] = startupinfo
            else:
                popen_platform["creationflags"] = 0
                popen_platform["start_new_session"] = True
                popen_platform["pass_fds"] = (
                    ()
                    if self._operation_lease is None
                    else (self._operation_lease.posix_descriptor(),)
                )
            process = subprocess.Popen(
                list(argv),
                stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL,
                stdout=(subprocess.PIPE if not own_stdout else stdout_file),
                stderr=own_stderr,
                shell=False,
                close_fds=True,
                env=dict(self.env),
                **popen_platform,
            )
            try:
                job.assign(process)
                _resume_windows_process(process)
            except BaseException:
                terminate_tree()
                raise
            if not own_stdout:
                assert process.stdout is not None

                def copy_output() -> None:
                    nonlocal streamed_stdout_bytes
                    try:
                        while True:
                            chunk = process.stdout.read(64 * 1024)
                            if not chunk:
                                break
                            next_size = streamed_stdout_bytes + len(chunk)
                            if next_size > stdout_limit:
                                stdout_limit_exceeded.set()
                                return
                            stdout_file.write(chunk)
                            streamed_stdout_bytes = next_size
                        stdout_file.flush()
                    except BaseException as exc:
                        reader_errors.append(exc)

                reader = threading.Thread(
                    target=copy_output,
                    name="repository-git-stdout",
                    daemon=True,
                )
                reader.start()
            if input_bytes is not None:
                assert process.stdin is not None

                def write_input() -> None:
                    try:
                        process.stdin.write(input_bytes)
                        process.stdin.close()
                    except BrokenPipeError:
                        pass
                    except BaseException as exc:
                        writer_errors.append(exc)

                writer = threading.Thread(
                    target=write_input,
                    name="repository-git-stdin",
                    daemon=True,
                )
                writer.start()
            deadline = time.monotonic() + self.timeout_seconds
            next_watch = 0.0
            while process.poll() is None:
                own_stderr.flush()
                if own_stdout:
                    stdout_file.flush()
                    if os.fstat(stdout_file.fileno()).st_size > stdout_limit:
                        terminate_tree()
                        raise RepositoryLimitError(
                            "Git process stdout exceeds its fixed budget"
                        )
                else:
                    if stdout_limit_exceeded.is_set():
                        terminate_tree()
                        raise RepositoryLimitError(
                            "Git process stdout exceeds its fixed budget"
                        )
                    if reader_errors:
                        terminate_tree()
                        raise RepositoryPreparationError(
                            "Git output reader failed"
                        ) from reader_errors[0]
                if os.fstat(own_stderr.fileno()).st_size > stderr_limit:
                    terminate_tree()
                    raise RepositoryLimitError(
                        "Git process stderr exceeds its fixed budget"
                    )
                now = time.monotonic()
                if (
                    watch_root is not None
                    and watch_limit is not None
                    and now >= next_watch
                ):
                    if _secure_tree_size(watch_root, watch_limit + 1) > watch_limit:
                        terminate_tree()
                        raise RepositoryLimitError(
                            "Git quarantine exceeds its fixed disk budget"
                        )
                    next_watch = now + 0.2
                if now >= deadline:
                    terminate_tree()
                    raise _GitCommandFailure(
                        -1, _digest_bytes(b"timeout"), timed_out=True
                    )
                time.sleep(0.02)
            if writer is not None:
                writer.join(timeout=5)
                if writer.is_alive():
                    terminate_tree()
                    raise RepositoryPreparationError(
                        "Git input writer did not terminate"
                    )
            if writer_errors:
                raise RepositoryPreparationError("Git input writer failed") from writer_errors[0]
            if reader is not None:
                reader.join(timeout=5)
                if reader.is_alive():
                    terminate_tree()
                    raise RepositoryPreparationError(
                        "Git output reader did not terminate"
                    )
                if stdout_limit_exceeded.is_set():
                    raise RepositoryLimitError(
                        "Git process stdout exceeds its fixed budget"
                    )
                if reader_errors:
                    raise RepositoryPreparationError(
                        "Git output reader failed"
                    ) from reader_errors[0]
            returncode = process.returncode
            if own_stdout:
                stdout_data = self._read_spooled(stdout_file, stdout_limit)
            else:
                stdout_file.flush()
                stdout_file.seek(0, os.SEEK_END)
                if stdout_file.tell() > stdout_limit:
                    raise RepositoryLimitError(
                        "Git process output exceeds its fixed budget"
                    )
                stdout_data = b""
            stderr_data = self._read_spooled(own_stderr, stderr_limit)
            stderr_digest = _digest_bytes(stderr_data)
            if returncode != 0:
                raise _GitCommandFailure(returncode, stderr_digest)
            return _CommandResult(returncode, stdout_data, stderr_digest)
        except FileNotFoundError as exc:
            raise RepositoryPreparationError("Git executable could not be launched") from exc
        finally:
            terminate_tree()
            if process is not None and process.stdout is not None:
                try:
                    process.stdout.close()
                except OSError:
                    pass
            if reader is not None and reader.is_alive():
                reader.join(timeout=5)
            job.close()
            if own_stdout:
                stdout_file.close()
            own_stderr.close()

    def run(
        self,
        args: Sequence[str],
        *,
        check: bool = True,
        stdout_limit: int = MAX_GIT_STDOUT_BYTES,
        stderr_limit: int = MAX_GIT_STDERR_BYTES,
        allow_https: bool = False,
        allow_file: bool = False,
        watch_root: Optional[Path] = None,
        watch_limit: Optional[int] = None,
    ) -> _CommandResult:
        result_args = self._base_args(allow_https=allow_https, allow_file=allow_file)
        result_args.extend(args)
        try:
            return self._run_process(
                result_args,
                input_bytes=None,
                stdout_limit=stdout_limit,
                stderr_limit=stderr_limit,
                watch_root=watch_root,
                watch_limit=watch_limit,
            )
        except _GitCommandFailure as exc:
            if check:
                raise
            # A nonzero result is useful to callers for ancestry checks.  The
            # command output is intentionally not returned to avoid locator
            # leakage through diagnostics.
            return _CommandResult(exc.returncode, b"", exc.stderr_digest)

    def run_input(
        self,
        args: Sequence[str],
        input_bytes: bytes,
        *,
        stdout_limit: int = MAX_GIT_STDOUT_BYTES,
        stderr_limit: int = MAX_GIT_STDERR_BYTES,
        allow_https: bool = False,
        allow_file: bool = False,
    ) -> _CommandResult:
        result_args = self._base_args(allow_https=allow_https, allow_file=allow_file)
        result_args.extend(args)
        return self._run_process(
            result_args,
            input_bytes=input_bytes,
            stdout_limit=stdout_limit,
            stderr_limit=stderr_limit,
        )

    def run_to_file(
        self,
        args: Sequence[str],
        output_path: Path,
        *,
        input_bytes: Optional[bytes] = None,
        stdout_limit: int,
        allow_https: bool = False,
        allow_file: bool = False,
    ) -> _CommandResult:
        parent = output_path.parent
        _assert_secure_directory(parent, "Git output parent")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
        try:
            descriptor = os.open(output_path, flags, 0o600)
            output_handle = os.fdopen(descriptor, "w+b")
        except OSError as exc:
            raise RepositorySecurityError("could not create bounded Git output") from exc
        result_args = self._base_args(allow_https=allow_https, allow_file=allow_file)
        result_args.extend(args)
        try:
            return self._run_process(
                result_args,
                input_bytes=input_bytes,
                stdout_limit=stdout_limit,
                stderr_limit=MAX_GIT_STDERR_BYTES,
                stdout_file=output_handle,
            )
        finally:
            output_handle.close()


def _positive_timeout(value: Any, context: str) -> float:
    if type(value) not in (int, float) or value <= 0 or value > 3600:
        raise ValueError("%s must be a positive bounded number" % context)
    return float(value)


@dataclass(frozen=True)
class _GitObject:
    oid: str
    object_type: str
    raw: bytes


@dataclass
class _RepositoryClosure:
    object_format: str
    objects: Dict[str, _GitObject]
    base_revision: str
    head_revision: str
    base_tree: str
    head_tree: str
    base_source_digest: str
    head_source_digest: str
    source_digest: str
    materialized_files: Tuple[Tuple[str, int, bytes, str], ...]
    materialized_bytes: int
    raw_object_bytes: int

    @property
    def object_count(self) -> int:
        return len(self.objects)

    @property
    def blob_count(self) -> int:
        return sum(1 for obj in self.objects.values() if obj.object_type == "blob")


def _object_hash(object_format: str, object_type: str, raw: bytes) -> str:
    header = ("%s %d\0" % (object_type, len(raw))).encode("ascii")
    digest = hashlib.sha1 if object_format == "sha1" else hashlib.sha256
    return digest(header + raw).hexdigest()


def _object_id_bytes(oid: str, object_format: str) -> bytes:
    expected = 40 if object_format == "sha1" else 64
    if len(oid) != expected or _GIT_OID_RE.fullmatch(oid) is None:
        raise RepositoryIntegrityError("Git object ID is not a full canonical ID")
    return bytes.fromhex(oid)


def _loose_object_bytes(object_format: str, object_type: str, raw: bytes) -> bytes:
    header = ("%s %d\0" % (object_type, len(raw))).encode("ascii")
    return zlib.compress(header + raw, 9)


def _logical_source_digest(objects: Mapping[str, _GitObject]) -> str:
    ordered = sorted(objects.values(), key=lambda item: (item.object_type, item.oid))
    digest = hashlib.sha256()
    digest.update(LOGICAL_GIT_SOURCE_VERSION.encode("ascii"))
    digest.update(b"\0")
    digest.update(struct.pack(">Q", len(ordered)))
    for item in ordered:
        type_bytes = item.object_type.encode("ascii")
        oid_bytes = item.oid.encode("ascii")
        digest.update(struct.pack(">I", len(type_bytes)))
        digest.update(type_bytes)
        digest.update(struct.pack(">I", len(oid_bytes)))
        digest.update(oid_bytes)
        digest.update(struct.pack(">Q", len(item.raw)))
        digest.update(item.raw)
    return digest.hexdigest()


def _subset_digest(objects: Mapping[str, _GitObject], reachable: Set[str]) -> str:
    return _logical_source_digest({oid: objects[oid] for oid in reachable})


def _parse_commit(obj: _GitObject, object_format: str) -> Tuple[str, Tuple[str, ...]]:
    if obj.object_type != "commit":
        raise RepositoryIntegrityError("declared revision is not a commit object")
    header, separator, _message = obj.raw.partition(b"\n\n")
    if not separator:
        raise RepositoryIntegrityError("commit object has no header terminator")
    trees: List[str] = []
    parents: List[str] = []
    for line in header.splitlines():
        if line.startswith(b"tree "):
            try:
                trees.append(line[5:].decode("ascii", "strict"))
            except UnicodeDecodeError as exc:
                raise RepositoryIntegrityError("commit tree ID is malformed") from exc
        elif line.startswith(b"parent "):
            try:
                parents.append(line[7:].decode("ascii", "strict"))
            except UnicodeDecodeError as exc:
                raise RepositoryIntegrityError("commit parent ID is malformed") from exc
    if len(trees) != 1:
        raise RepositoryIntegrityError("commit must contain exactly one tree header")
    tree = trees[0]
    _object_id_bytes(tree, object_format)
    for parent in parents:
        _object_id_bytes(parent, object_format)
    return tree, tuple(parents)


@dataclass(frozen=True)
class _TreeEntry:
    mode: int
    name: str
    name_bytes: bytes
    oid: str
    object_type: str


def _parse_tree(obj: _GitObject, object_format: str) -> Tuple[_TreeEntry, ...]:
    if obj.object_type != "tree":
        raise RepositoryIntegrityError("tree reference does not name a tree object")
    oid_size = 20 if object_format == "sha1" else 32
    data = obj.raw
    offset = 0
    entries: List[_TreeEntry] = []
    exact_names: Set[bytes] = set()
    folded_names: Set[str] = set()
    previous_sort_key: Optional[bytes] = None
    while offset < len(data):
        space = data.find(b" ", offset)
        nul = data.find(b"\0", space + 1 if space >= 0 else offset)
        if space <= offset or nul <= space + 1 or nul + 1 + oid_size > len(data):
            raise RepositoryIntegrityError("tree object has malformed entry framing")
        mode_bytes = data[offset:space]
        name_bytes = data[space + 1 : nul]
        oid = data[nul + 1 : nul + 1 + oid_size].hex()
        offset = nul + 1 + oid_size
        try:
            mode_text = mode_bytes.decode("ascii", "strict")
            mode = int(mode_text, 8)
            name = name_bytes.decode("utf-8", "strict")
        except (UnicodeDecodeError, ValueError) as exc:
            raise RepositoryPolicyError(
                "tree path/mode is not canonical UTF-8 Git data"
            ) from exc
        canonical_name = _validate_component(name, "repository tree")
        if canonical_name != name_bytes:
            raise RepositoryPolicyError("tree path is not canonical UTF-8")
        if mode in {0o100644, 0o100755}:
            object_type = "blob"
            sort_key = name_bytes + b"\0"
        elif mode == 0o040000:
            object_type = "tree"
            sort_key = name_bytes + b"/"
        elif mode == 0o120000:
            raise RepositoryPolicyError("symlink mode 120000 is rejected")
        elif mode == 0o160000:
            raise RepositoryPolicyError("submodule/gitlink mode 160000 is rejected")
        else:
            raise RepositoryPolicyError("tree contains a non-regular Git mode")
        folded = unicodedata.normalize("NFC", name).casefold()
        if name_bytes in exact_names or folded in folded_names:
            raise RepositoryPolicyError("tree has an NFC/casefold path collision")
        exact_names.add(name_bytes)
        folded_names.add(folded)
        if previous_sort_key is not None and sort_key <= previous_sort_key:
            raise RepositoryIntegrityError("tree entries are not in canonical Git order")
        previous_sort_key = sort_key
        entries.append(_TreeEntry(mode, name, name_bytes, oid, object_type))
    return tuple(entries)


def _check_blob_policy(raw: bytes, path_name: Optional[str] = None) -> None:
    if len(raw) > MAX_GIT_BLOB_BYTES:
        raise RepositoryLimitError("Git blob exceeds the fixed single-blob budget")
    if raw.startswith(_LFS_POINTER_PREFIX):
        raise RepositoryPolicyError("Git LFS pointer content is rejected")
    if path_name == ".gitattributes":
        if _LFS_ATTRIBUTE_RE.search(raw):
            raise RepositoryPolicyError("Git LFS attributes are rejected")
        if _EXTERNAL_FILTER_ATTRIBUTE_RE.search(raw):
            raise RepositoryPolicyError("external Git filter attributes are rejected")


def _closure_from_objects(
    objects: Mapping[str, _GitObject],
    *,
    object_format: str,
    base_revision: str,
    head_revision: str,
    expected_object_ids: Optional[Set[str]] = None,
) -> _RepositoryClosure:
    if object_format not in {"sha1", "sha256"}:
        raise RepositoryIntegrityError("unknown Git object format")
    expected_length = 40 if object_format == "sha1" else 64
    if len(base_revision) != expected_length or len(head_revision) != expected_length:
        raise RepositoryIntegrityError("revision length does not match object format")
    if base_revision == head_revision:
        raise RepositoryIntegrityError("base and head revisions must differ")
    if len(objects) > MAX_GIT_OBJECTS:
        raise RepositoryLimitError("Git closure exceeds the fixed object-count budget")
    canonical: Dict[str, _GitObject] = {}
    raw_object_bytes = 0
    for oid, obj in objects.items():
        if oid != obj.oid or len(oid) != expected_length or _GIT_OID_RE.fullmatch(oid) is None:
            raise RepositoryIntegrityError("Git object inventory contains an invalid ID")
        if obj.object_type not in {"blob", "tree", "commit"}:
            raise RepositoryPolicyError("Git closure contains an unsupported object type")
        if obj.object_type != "blob" and len(obj.raw) > MAX_GIT_METADATA_OBJECT_BYTES:
            raise RepositoryLimitError("Git metadata object exceeds its fixed budget")
        if obj.object_type == "blob":
            _check_blob_policy(obj.raw)
        if _object_hash(object_format, obj.object_type, obj.raw) != oid:
            raise RepositoryIntegrityError("Git object raw bytes do not match their ID")
        raw_object_bytes += len(obj.raw)
        if raw_object_bytes > MAX_CACHE_BYTES:
            raise RepositoryLimitError("Git closure exceeds the raw cache byte budget")
        canonical[oid] = obj
    if expected_object_ids is not None and set(canonical) != expected_object_ids:
        raise RepositoryIntegrityError("Git object inventory is not the exact closure")
    if base_revision not in canonical or head_revision not in canonical:
        raise RepositoryIntegrityError("declared revision object is missing")

    parsed_commits: Dict[str, Tuple[str, Tuple[str, ...]]] = {}
    parsed_trees: Dict[str, Tuple[_TreeEntry, ...]] = {}

    def commit_data(oid: str) -> Tuple[str, Tuple[str, ...]]:
        if oid not in canonical:
            raise RepositoryIntegrityError("commit closure references a missing object")
        if oid not in parsed_commits:
            parsed_commits[oid] = _parse_commit(canonical[oid], object_format)
        return parsed_commits[oid]

    def tree_data(oid: str) -> Tuple[_TreeEntry, ...]:
        if oid not in canonical:
            raise RepositoryIntegrityError("tree closure references a missing object")
        if oid not in parsed_trees:
            parsed_trees[oid] = _parse_tree(canonical[oid], object_format)
        return parsed_trees[oid]

    def reachable_from_commit(start: str) -> Set[str]:
        reachable: Set[str] = set()
        pending: List[Tuple[str, str]] = [(start, "commit")]
        while pending:
            oid, expected_type = pending.pop()
            if oid in reachable:
                continue
            obj = canonical.get(oid)
            if obj is None:
                raise RepositoryIntegrityError("Git closure references a missing object")
            if obj.object_type != expected_type:
                raise RepositoryIntegrityError("Git closure reference has the wrong type")
            reachable.add(oid)
            if expected_type == "commit":
                tree, parents = commit_data(oid)
                pending.append((tree, "tree"))
                pending.extend((parent, "commit") for parent in parents)
            elif expected_type == "tree":
                for entry in tree_data(oid):
                    pending.append((entry.oid, entry.object_type))
        return reachable

    base_reachable = reachable_from_commit(base_revision)
    head_reachable = reachable_from_commit(head_revision)
    union_reachable = base_reachable | head_reachable
    if set(canonical) != union_reachable:
        raise RepositoryIntegrityError("cache contains objects outside base/head closure")

    # Validate every reachable tree's component/collision policy and calculate
    # the maximum path depth/bytes by memoized DAG traversal.
    depth_stack: Set[str] = set()
    depth_memo: Dict[str, Tuple[int, int]] = {}

    def tree_extent(oid: str) -> Tuple[int, int]:
        if oid in depth_memo:
            return depth_memo[oid]
        if oid in depth_stack:
            raise RepositoryIntegrityError("Git tree graph contains a cycle")
        depth_stack.add(oid)
        max_depth = 0
        max_bytes = 0
        for entry in tree_data(oid):
            component_size = len(entry.name_bytes)
            if entry.object_type == "tree":
                child_depth, child_bytes = tree_extent(entry.oid)
                max_depth = max(max_depth, 1 + child_depth)
                max_bytes = max(max_bytes, component_size + 1 + child_bytes)
            else:
                max_depth = max(max_depth, 1)
                max_bytes = max(max_bytes, component_size)
                _check_blob_policy(canonical[entry.oid].raw, entry.name)
        depth_stack.remove(oid)
        if max_depth > MAX_PATH_DEPTH or max_bytes > MAX_PATH_BYTES:
            raise RepositoryLimitError("repository tree exceeds fixed path policy")
        depth_memo[oid] = (max_depth, max_bytes)
        return depth_memo[oid]

    for tree_oid, _parents in parsed_commits.values():
        tree_extent(tree_oid)

    base_tree, _ = commit_data(base_revision)
    head_tree, _ = commit_data(head_revision)
    materialized: List[Tuple[str, int, bytes, str]] = []
    materialized_bytes = 0
    materialized_entries = 0
    materialized_folded: Set[str] = set()

    def enumerate_tree(oid: str, prefix: Tuple[str, ...]) -> None:
        nonlocal materialized_bytes, materialized_entries
        for entry in tree_data(oid):
            materialized_entries += 1
            if materialized_entries > MAX_LOGICAL_TREE_ENTRIES:
                raise RepositoryLimitError(
                    "head tree exceeds logical entry expansion budget"
                )
            parts = (*prefix, entry.name)
            path_bytes = _validate_repo_path(parts, "head tree")
            path = path_bytes.decode("utf-8", "strict")
            folded = unicodedata.normalize("NFC", path).casefold()
            if folded in materialized_folded:
                raise RepositoryPolicyError("head tree has a path collision")
            if entry.object_type == "tree":
                enumerate_tree(entry.oid, parts)
                continue
            materialized_folded.add(folded)
            blob = canonical[entry.oid].raw
            _check_blob_policy(blob, entry.name)
            materialized_bytes += len(blob)
            if len(materialized) + 1 > MAX_MATERIALIZED_FILES:
                raise RepositoryLimitError("head tree exceeds fixed file-count budget")
            if materialized_bytes > MAX_MATERIALIZED_BYTES:
                raise RepositoryLimitError("head tree exceeds materialized byte budget")
            materialized.append((path, entry.mode, blob, entry.oid))

    enumerate_tree(head_tree, ())
    materialized.sort(key=lambda item: item[0].encode("utf-8"))
    return _RepositoryClosure(
        object_format=object_format,
        objects=canonical,
        base_revision=base_revision,
        head_revision=head_revision,
        base_tree=base_tree,
        head_tree=head_tree,
        base_source_digest=_subset_digest(canonical, base_reachable),
        head_source_digest=_subset_digest(canonical, head_reachable),
        source_digest=_subset_digest(canonical, union_reachable),
        materialized_files=tuple(materialized),
        materialized_bytes=materialized_bytes,
        raw_object_bytes=raw_object_bytes,
    )


def _git_config_bytes(object_format: str, *, bare: bool) -> bytes:
    lines = [
        "[core]",
        "\trepositoryformatversion = %d" % (1 if object_format == "sha256" else 0),
        "\tfilemode = %s" % ("true" if os.name != "nt" else "false"),
        "\tbare = %s" % ("true" if bare else "false"),
        "\tlogallrefupdates = false",
        "\thooksPath = hooks-disabled",
        "\tsymlinks = false",
        "[gc]",
        "\tauto = 0",
    ]
    if object_format == "sha256":
        lines.extend(["[extensions]", "\tobjectFormat = sha256"])
    return ("\n".join(lines) + "\n").encode("ascii")


def _mkdir_exclusive(path: Path, mode: int = 0o700) -> None:
    try:
        os.mkdir(path, mode)
    except OSError as exc:
        raise RepositorySecurityError("could not create repository directory safely") from exc
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise RepositorySecurityError("could not verify repository directory") from exc
    if _is_link_or_reparse(info) or not stat.S_ISDIR(info.st_mode):
        raise RepositorySecurityError("repository directory was replaced unsafely")


def _ensure_child_directory(parent: Path, name: str) -> Path:
    path = parent / name
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        _mkdir_exclusive(path)
        return path
    except OSError as exc:
        raise RepositorySecurityError("could not inspect repository directory") from exc
    if _is_link_or_reparse(info) or not stat.S_ISDIR(info.st_mode):
        raise RepositorySecurityError("repository path contains a link or special node")
    return path


def _write_loose_repository(
    repository_path: Path,
    closure: _RepositoryClosure,
    *,
    bare: bool,
) -> int:
    if os.path.lexists(repository_path):
        raise RepositorySecurityError("repository destination already exists")
    _assert_secure_directory(repository_path.parent, "repository destination parent")
    _mkdir_exclusive(repository_path)
    objects_root = _ensure_child_directory(repository_path, "objects")
    _ensure_child_directory(objects_root, "info")
    _ensure_child_directory(objects_root, "pack")
    refs_root = _ensure_child_directory(repository_path, "refs")
    heads_root = _ensure_child_directory(refs_root, "heads")
    hooks_root = _ensure_child_directory(repository_path, "hooks-disabled")
    del hooks_root
    _write_regular_file_exclusive(
        repository_path / "config",
        _git_config_bytes(closure.object_format, bare=bare),
        mode=0o600,
    )
    _write_regular_file_exclusive(
        repository_path / "HEAD", (closure.head_revision + "\n").encode("ascii")
    )
    _write_regular_file_exclusive(
        heads_root / "eval-base", (closure.base_revision + "\n").encode("ascii")
    )
    _write_regular_file_exclusive(
        heads_root / "eval-head", (closure.head_revision + "\n").encode("ascii")
    )
    cache_bytes = 0
    for obj in sorted(closure.objects.values(), key=lambda item: item.oid):
        prefix = _ensure_child_directory(objects_root, obj.oid[:2])
        loose = _loose_object_bytes(closure.object_format, obj.object_type, obj.raw)
        cache_bytes += len(loose)
        if cache_bytes > MAX_CACHE_BYTES:
            raise RepositoryLimitError("loose object database exceeds cache byte budget")
        _write_regular_file_exclusive(prefix / obj.oid[2:], loose, mode=0o444)
    # The fixed metadata is also part of the on-disk cache budget.
    cache_bytes = _secure_tree_size(repository_path, MAX_CACHE_BYTES)
    if cache_bytes > MAX_CACHE_BYTES:
        raise RepositoryLimitError("repository cache exceeds its fixed byte budget")
    return cache_bytes


def _decompress_loose_object(data: bytes) -> Tuple[str, bytes]:
    decompressor = zlib.decompressobj()
    maximum_output = max(MAX_GIT_BLOB_BYTES, MAX_GIT_METADATA_OBJECT_BYTES) + 1024
    try:
        decoded = decompressor.decompress(data, maximum_output + 1)
    except zlib.error as exc:
        raise RepositoryIntegrityError("loose Git object is not valid zlib data") from exc
    if (
        len(decoded) > maximum_output
        or not decompressor.eof
        or decompressor.unused_data
        or decompressor.unconsumed_tail
    ):
        raise RepositoryIntegrityError("loose Git object has trailing or oversized data")
    header, separator, raw = decoded.partition(b"\0")
    if not separator:
        raise RepositoryIntegrityError("loose Git object has no header")
    try:
        object_type_bytes, size_bytes = header.split(b" ", 1)
        object_type = object_type_bytes.decode("ascii", "strict")
        size = int(size_bytes.decode("ascii", "strict"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise RepositoryIntegrityError("loose Git object header is malformed") from exc
    if size != len(raw):
        raise RepositoryIntegrityError("loose Git object size header is wrong")
    return object_type, raw


def _read_loose_repository(
    repository_path: Path,
    *,
    object_format: str,
    base_revision: str,
    head_revision: str,
) -> Tuple[_RepositoryClosure, int]:
    _assert_secure_directory(repository_path, "canonical repository cache")
    expected_config = _git_config_bytes(object_format, bare=True)
    if _read_regular_file(
        repository_path / "config",
        maximum=MAX_GIT_CONFIG_BYTES,
        context="canonical repository config",
    ) != expected_config:
        raise RepositoryIntegrityError("canonical repository config was modified")
    expected_head = (head_revision + "\n").encode("ascii")
    if _read_regular_file(
        repository_path / "HEAD", maximum=128, context="canonical repository HEAD"
    ) != expected_head:
        raise RepositoryIntegrityError("canonical repository HEAD was modified")
    for name, revision in (("eval-base", base_revision), ("eval-head", head_revision)):
        if _read_regular_file(
            repository_path / "refs" / "heads" / name,
            maximum=128,
            context="canonical repository ref",
        ) != (revision + "\n").encode("ascii"):
            raise RepositoryIntegrityError("canonical repository ref was modified")

    objects_root = repository_path / "objects"
    _assert_secure_directory(objects_root, "canonical object database")
    objects: Dict[str, _GitObject] = {}
    cache_bytes = _secure_tree_size(repository_path, MAX_CACHE_BYTES)
    if cache_bytes > MAX_CACHE_BYTES:
        raise RepositoryLimitError("repository cache exceeds its fixed byte budget")
    try:
        entries = sorted(os.scandir(objects_root), key=lambda entry: entry.name)
    except OSError as exc:
        raise RepositorySecurityError("could not enumerate canonical object database") from exc
    for entry in entries:
        try:
            info = entry.stat(follow_symlinks=False)
        except OSError as exc:
            raise RepositorySecurityError("could not inspect canonical object database") from exc
        if _is_link_or_reparse(info) or not stat.S_ISDIR(info.st_mode):
            raise RepositorySecurityError("canonical object database contains unsafe node")
        if entry.name in {"info", "pack"}:
            try:
                if any(os.scandir(entry.path)):
                    raise RepositoryPolicyError(
                        "alternates, packs, promisor metadata, or grafts are rejected"
                    )
            except OSError as exc:
                raise RepositorySecurityError("could not inspect object metadata") from exc
            continue
        if len(entry.name) != 2 or _HEX_RE.fullmatch(entry.name) is None:
            raise RepositoryIntegrityError("canonical object directory is malformed")
        try:
            loose_entries = sorted(os.scandir(entry.path), key=lambda child: child.name)
        except OSError as exc:
            raise RepositorySecurityError("could not enumerate loose objects") from exc
        for loose_entry in loose_entries:
            try:
                loose_info = loose_entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise RepositorySecurityError("could not inspect loose object") from exc
            if _is_link_or_reparse(loose_info) or not stat.S_ISREG(loose_info.st_mode):
                raise RepositorySecurityError("loose object is a link or special node")
            oid = entry.name + loose_entry.name
            expected_length = 40 if object_format == "sha1" else 64
            if len(oid) != expected_length or _HEX_RE.fullmatch(oid) is None:
                raise RepositoryIntegrityError("loose object path is malformed")
            compressed = _read_regular_file(
                Path(loose_entry.path),
                maximum=MAX_CACHE_BYTES,
                context="loose Git object",
            )
            object_type, raw = _decompress_loose_object(compressed)
            if oid in objects:
                raise RepositoryIntegrityError("duplicate loose Git object")
            objects[oid] = _GitObject(oid, object_type, raw)
            if len(objects) > MAX_GIT_OBJECTS:
                raise RepositoryLimitError("cache exceeds object-count budget")
    closure = _closure_from_objects(
        objects,
        object_format=object_format,
        base_revision=base_revision,
        head_revision=head_revision,
    )
    return closure, cache_bytes


def _assert_no_repository_extensions(git_dir: Path) -> None:
    forbidden_files = (
        git_dir / "commondir",
        git_dir / "shallow",
        git_dir / "info" / "grafts",
        git_dir / "objects" / "info" / "alternates",
        git_dir / "objects" / "info" / "http-alternates",
    )
    for path in forbidden_files:
        if os.path.lexists(path):
            raise RepositorySecurityError(
                "alternates, common dir, shallow history, or grafts are rejected"
            )
    replace_root = git_dir / "refs" / "replace"
    if os.path.lexists(replace_root):
        raise RepositorySecurityError("Git replace refs are rejected")
    modules = git_dir / "modules"
    if os.path.lexists(modules):
        raise RepositorySecurityError(
            "nested repository/submodule metadata is rejected"
        )
    pack_root = git_dir / "objects" / "pack"
    if os.path.isdir(pack_root):
        try:
            entries = list(os.scandir(pack_root))
        except OSError as exc:
            raise RepositorySecurityError("could not inspect source packs") from exc
        for entry in entries:
            if entry.name.endswith(".promisor"):
                raise RepositorySecurityError(
                    "promisor/partial-clone objects are rejected"
                )


def _scan_metadata_tree(root: Path) -> None:
    _assert_secure_directory(root, "Git metadata root")
    stack = [root]
    nodes = 0
    while stack:
        directory = stack.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as exc:
            raise RepositorySecurityError("could not inspect Git metadata") from exc
        for entry in entries:
            nodes += 1
            if nodes > MAX_GIT_METADATA_NODES:
                raise RepositoryLimitError("Git metadata exceeds filesystem node budget")
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise RepositorySecurityError("could not inspect Git metadata node") from exc
            if _is_link_or_reparse(info):
                raise RepositorySecurityError(
                    "Git metadata contains a symlink or reparse point"
                )
            if stat.S_ISDIR(info.st_mode):
                stack.append(Path(entry.path))
            elif not stat.S_ISREG(info.st_mode):
                raise RepositorySecurityError("Git metadata contains a special node")


def _validate_source_config(runner: _GitRunner, git_dir: Path, object_format: str) -> None:
    config = git_dir / "config"
    if not os.path.lexists(config):
        raise RepositoryIntegrityError("source Git repository has no config")
    raw = _read_regular_file(
        config, maximum=MAX_GIT_CONFIG_BYTES, context="source Git config"
    )
    lowered = raw.lower()
    if b"[include" in lowered or b"include.path" in lowered:
        raise RepositorySecurityError("source Git config includes are rejected")
    try:
        result = runner.run(
            [
                "config",
                "--file",
                str(config),
                "--no-includes",
                "--null",
                "--name-only",
                "--list",
            ],
            stdout_limit=MAX_GIT_CONFIG_BYTES,
        )
    except _GitCommandFailure as exc:
        raise RepositoryIntegrityError("source Git config is malformed") from exc
    keys = [
        item.decode("utf-8", "strict").lower()
        for item in result.stdout.split(b"\0")
        if item
    ]
    if any(key not in _SAFE_SOURCE_CONFIG_KEYS for key in keys):
        raise RepositorySecurityError("source Git config authority is rejected")
    if object_format == "sha256" and "extensions.objectformat" not in keys:
        raise RepositoryIntegrityError("SHA-256 source is missing object-format declaration")
    if object_format == "sha1" and "extensions.objectformat" in keys:
        raise RepositoryIntegrityError("source object format conflicts with revisions")


def _locate_local_git_source(
    runner: _GitRunner, source_root: Path, object_format: str
) -> Path:
    _assert_secure_directory(source_root, "local Git source")
    dot_git = source_root / ".git"
    if os.path.lexists(dot_git):
        try:
            info = os.lstat(dot_git)
        except OSError as exc:
            raise RepositorySecurityError("could not inspect local .git") from exc
        if _is_link_or_reparse(info) or not stat.S_ISDIR(info.st_mode):
            raise RepositorySecurityError(
                "local .git indirection, symlink, or reparse point is rejected"
            )
        git_dir = dot_git
    elif os.path.isfile(source_root / "HEAD") and os.path.isdir(source_root / "objects"):
        git_dir = source_root
    else:
        raise RepositoryIntegrityError("local source is not a canonical Git repository")
    _scan_metadata_tree(git_dir)
    _assert_no_repository_extensions(git_dir)
    _validate_source_config(runner, git_dir, object_format)
    return git_dir


def _create_empty_git_repository(path: Path, object_format: str, *, bare: bool) -> None:
    if os.path.lexists(path):
        raise RepositorySecurityError("Git quarantine destination already exists")
    _assert_secure_directory(path.parent, "Git quarantine parent")
    _mkdir_exclusive(path)
    objects = _ensure_child_directory(path, "objects")
    _ensure_child_directory(objects, "info")
    _ensure_child_directory(objects, "pack")
    refs = _ensure_child_directory(path, "refs")
    _ensure_child_directory(refs, "heads")
    _ensure_child_directory(path, "hooks-disabled")
    _write_regular_file_exclusive(path / "config", _git_config_bytes(object_format, bare=bare))
    _write_regular_file_exclusive(path / "HEAD", b"ref: refs/heads/eval-head\n")


def _fetch_quarantine(
    runner: _GitRunner,
    quarantine: Path,
    *,
    locator: str,
    remote: bool,
    base_revision: str,
    head_revision: str,
    remote_host: Optional[str] = None,
    remote_port: Optional[int] = None,
    resolved_addresses: Sequence[str] = (),
) -> None:
    object_format = "sha1" if len(base_revision) == 40 else "sha256"
    _create_empty_git_repository(quarantine, object_format, bare=True)
    args: List[str] = []
    if remote:
        if remote_host is None or remote_port is None or not resolved_addresses:
            raise RepositorySecurityError(
                "remote Git fetch requires a pinned public endpoint"
            )
        args.extend(
            _curl_resolve_arguments(
                host=remote_host,
                port=remote_port,
                addresses=resolved_addresses,
            )
        )
    args.extend([
        "--git-dir",
        str(quarantine),
        "fetch",
        "--no-tags",
        "--no-recurse-submodules",
        "--no-write-fetch-head",
        "--force",
    ])
    if not remote:
        upload_pack = (
            runner.upload_pack.as_posix()
            if os.name == "nt"
            else str(runner.upload_pack)
        )
        args.extend(["--upload-pack", upload_pack])
    args.extend(
        [
            locator,
            "+%s:refs/heads/eval-base" % base_revision,
            "+%s:refs/heads/eval-head" % head_revision,
        ]
    )
    try:
        runner.run(
            args,
            allow_https=remote,
            allow_file=not remote,
            stdout_limit=MAX_GIT_STDOUT_BYTES,
            watch_root=quarantine,
            watch_limit=MAX_CACHE_BYTES,
        )
    except _GitCommandFailure as exc:
        raise RepositoryPreparationError("Git source acquisition failed") from exc
    _assert_no_repository_extensions(quarantine)
    config = _read_regular_file(
        quarantine / "config", maximum=MAX_GIT_CONFIG_BYTES, context="Git quarantine config"
    )
    if config != _git_config_bytes(object_format, bare=True):
        raise RepositorySecurityError("Git quarantine config was modified")
    for transient in (quarantine / "FETCH_HEAD", quarantine / "ORIG_HEAD"):
        if os.path.lexists(transient):
            try:
                info = os.lstat(transient)
            except OSError as exc:
                raise RepositorySecurityError("could not inspect Git transient metadata") from exc
            if _is_link_or_reparse(info) or not stat.S_ISREG(info.st_mode):
                raise RepositorySecurityError("Git transient metadata is unsafe")
            try:
                os.unlink(transient)
            except OSError as exc:
                raise RepositorySecurityError("could not remove Git transient metadata") from exc


def _read_exact(handle: BinaryIO, size: int) -> bytes:
    chunks: List[bytes] = []
    remaining = size
    while remaining:
        chunk = handle.read(min(1024 * 1024, remaining))
        if not chunk:
            raise RepositoryIntegrityError("Git batch object ended unexpectedly")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _extract_quarantine_closure(
    runner: _GitRunner,
    quarantine: Path,
    *,
    base_revision: str,
    head_revision: str,
) -> _RepositoryClosure:
    object_format = "sha1" if len(base_revision) == 40 else "sha256"
    try:
        result = runner.run(
            [
                "--git-dir",
                str(quarantine),
                "rev-list",
                "--objects",
                "--no-object-names",
                base_revision,
                head_revision,
            ],
            stdout_limit=MAX_GIT_OBJECTS * 70,
        )
    except _GitCommandFailure as exc:
        raise RepositoryIntegrityError("Git closure enumeration failed") from exc
    object_ids: Set[str] = set()
    expected_length = 40 if object_format == "sha1" else 64
    for line in result.stdout.splitlines():
        try:
            oid = line.decode("ascii", "strict").strip()
        except UnicodeDecodeError as exc:
            raise RepositoryIntegrityError("Git closure enumeration was not ASCII") from exc
        if len(oid) != expected_length or _GIT_OID_RE.fullmatch(oid) is None:
            raise RepositoryIntegrityError("Git closure enumeration returned a short ID")
        object_ids.add(oid)
        if len(object_ids) > MAX_GIT_OBJECTS:
            raise RepositoryLimitError("Git closure exceeds object-count budget")
    if base_revision not in object_ids or head_revision not in object_ids:
        raise RepositoryIntegrityError("Git closure omitted a declared revision")
    if not object_ids:
        raise RepositoryIntegrityError("Git closure is empty")

    batch_input = b"".join((oid.encode("ascii") + b"\n") for oid in sorted(object_ids))
    batch_output = runner.tmp_root / ("batch-" + uuid.uuid4().hex)
    try:
        runner.run_to_file(
            ["--git-dir", str(quarantine), "cat-file", "--batch"],
            batch_output,
            input_bytes=batch_input,
            stdout_limit=MAX_CACHE_BYTES + MAX_GIT_OBJECTS * 128,
        )
        objects: Dict[str, _GitObject] = {}
        raw_total = 0
        with open(batch_output, "rb", buffering=0) as handle:
            for _index in range(len(object_ids)):
                header = handle.readline(256)
                if not header or not header.endswith(b"\n"):
                    raise RepositoryIntegrityError("Git batch output has a malformed header")
                parts = header[:-1].split(b" ")
                if len(parts) != 3:
                    raise RepositoryIntegrityError("Git batch output has an invalid header")
                try:
                    actual_oid = parts[0].decode("ascii", "strict")
                    object_type = parts[1].decode("ascii", "strict")
                    size = int(parts[2].decode("ascii", "strict"))
                except (UnicodeDecodeError, ValueError) as exc:
                    raise RepositoryIntegrityError("Git batch output header is invalid") from exc
                if (
                    actual_oid not in object_ids
                    or actual_oid in objects
                    or object_type == "missing"
                    or size < 0
                ):
                    raise RepositoryIntegrityError("Git batch output omitted an object")
                maximum = (
                    MAX_GIT_BLOB_BYTES
                    if object_type == "blob"
                    else MAX_GIT_METADATA_OBJECT_BYTES
                )
                if object_type not in {"blob", "tree", "commit"} or size > maximum:
                    raise RepositoryPolicyError("Git closure contains an unsupported object")
                raw = _read_exact(handle, size)
                if handle.read(1) != b"\n":
                    raise RepositoryIntegrityError("Git batch object framing is invalid")
                raw_total += len(raw)
                if raw_total > MAX_CACHE_BYTES:
                    raise RepositoryLimitError("Git closure exceeds raw cache budget")
                objects[actual_oid] = _GitObject(actual_oid, object_type, raw)
            if set(objects) != object_ids:
                raise RepositoryIntegrityError("Git batch output omitted an object")
            if handle.read(1):
                raise RepositoryIntegrityError("Git batch output contains extra objects")
    finally:
        try:
            os.unlink(batch_output)
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise RepositorySecurityError("could not remove temporary Git batch output") from exc
    return _closure_from_objects(
        objects,
        object_format=object_format,
        base_revision=base_revision,
        head_revision=head_revision,
        expected_object_ids=object_ids,
    )


@dataclass(frozen=True)
class _FixtureFile:
    path: str
    parts: Tuple[str, ...]
    data: bytes


def _scan_fixture_snapshot(snapshot_root: Path) -> Tuple[_FixtureFile, ...]:
    _assert_secure_directory(snapshot_root, "fixture snapshot")
    files: List[_FixtureFile] = []
    total_bytes = 0
    folded_paths: Set[str] = set()
    stack: List[Tuple[Path, Tuple[str, ...]]] = [(snapshot_root, ())]
    while stack:
        directory, prefix = stack.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as exc:
            raise RepositorySecurityError("could not enumerate fixture snapshot") from exc
        sortable: List[Tuple[bytes, os.DirEntry[str], str]] = []
        local_folded: Set[str] = set()
        for entry in entries:
            name = entry.name
            encoded = _validate_component(name, "fixture")
            folded = unicodedata.normalize("NFC", name).casefold()
            if folded in local_folded:
                raise RepositoryPolicyError("fixture has an NFC/casefold collision")
            local_folded.add(folded)
            sortable.append((encoded, entry, name))
        sortable.sort(key=lambda item: item[0])
        directories: List[Tuple[Path, Tuple[str, ...]]] = []
        for _encoded, entry, name in sortable:
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise RepositorySecurityError("could not inspect fixture node") from exc
            if _is_link_or_reparse(info):
                raise RepositorySecurityError(
                    "fixture contains a symlink or reparse point"
                )
            parts = (*prefix, name)
            path_bytes = _validate_repo_path(parts, "fixture")
            path = path_bytes.decode("utf-8", "strict")
            folded_path = unicodedata.normalize("NFC", path).casefold()
            if folded_path in folded_paths:
                raise RepositoryPolicyError("fixture has an NFC/casefold path collision")
            folded_paths.add(folded_path)
            if stat.S_ISDIR(info.st_mode):
                directories.append((Path(entry.path), parts))
            elif stat.S_ISREG(info.st_mode):
                data = _read_regular_file(
                    Path(entry.path),
                    maximum=MAX_GIT_BLOB_BYTES,
                    context="fixture source file",
                )
                _check_blob_policy(data, name)
                total_bytes += len(data)
                if total_bytes > MAX_MATERIALIZED_BYTES:
                    raise RepositoryLimitError("fixture exceeds total source byte budget")
                if len(files) + 1 > MAX_MATERIALIZED_FILES:
                    raise RepositoryLimitError("fixture exceeds source file-count budget")
                files.append(_FixtureFile(path, parts, data))
            else:
                raise RepositorySecurityError("fixture contains a special filesystem node")
        # Reverse push keeps the eventual traversal stable, though final file
        # ordering is explicitly UTF-8 sorted below.
        stack.extend(reversed(directories))
    files.sort(key=lambda item: item.path.encode("utf-8"))
    return tuple(files)


def _add_object(
    objects: MutableMapping[str, _GitObject],
    object_format: str,
    object_type: str,
    raw: bytes,
) -> str:
    oid = _object_hash(object_format, object_type, raw)
    existing = objects.get(oid)
    candidate = _GitObject(oid, object_type, raw)
    if existing is not None and existing != candidate:
        raise RepositoryIntegrityError("Git object hash collision was detected")
    objects[oid] = candidate
    if len(objects) > MAX_GIT_OBJECTS:
        raise RepositoryLimitError("fixture exceeds Git object-count budget")
    return oid


def _build_fixture_tree(
    files: Sequence[_FixtureFile],
    object_format: str,
    objects: MutableMapping[str, _GitObject],
) -> str:
    root: Dict[str, Any] = {}
    for item in files:
        node = root
        for component in item.parts[:-1]:
            existing = node.setdefault(component, {})
            if type(existing) is not dict:
                raise RepositoryIntegrityError("fixture file/directory conflict")
            node = existing
        leaf = item.parts[-1]
        if leaf in node:
            raise RepositoryIntegrityError("fixture contains duplicate path")
        blob_oid = _add_object(objects, object_format, "blob", item.data)
        node[leaf] = (0o100644, blob_oid)

    def write_tree(node: Mapping[str, Any]) -> str:
        records: List[Tuple[bytes, bytes]] = []
        for name, value in node.items():
            name_bytes = name.encode("utf-8", "strict")
            if type(value) is dict:
                child_oid = write_tree(value)
                sort_key = name_bytes + b"/"
                record = b"40000 " + name_bytes + b"\0" + _object_id_bytes(
                    child_oid, object_format
                )
            else:
                mode, blob_oid = value
                sort_key = name_bytes + b"\0"
                record = ("%o " % mode).encode("ascii") + name_bytes + b"\0" + _object_id_bytes(
                    blob_oid, object_format
                )
            records.append((sort_key, record))
        raw = b"".join(record for _key, record in sorted(records, key=lambda item: item[0]))
        return _add_object(objects, object_format, "tree", raw)

    return write_tree(root)


_FIXTURE_IDENTITY = (
    "Review Agent Eval Fixture",
    "fixture@review-agent-eval.invalid",
    946684800,
    "+0000",
)


def _fixture_commit(
    objects: MutableMapping[str, _GitObject],
    object_format: str,
    *,
    tree: str,
    parent: Optional[str],
    message: bytes,
) -> str:
    name, email, timestamp, timezone = _FIXTURE_IDENTITY
    lines = [b"tree " + tree.encode("ascii")]
    if parent is not None:
        lines.append(b"parent " + parent.encode("ascii"))
    identity = ("%s <%s> %d %s" % (name, email, timestamp, timezone)).encode("ascii")
    lines.extend([b"author " + identity, b"committer " + identity, b"", message])
    raw = b"\n".join(lines)
    if not raw.endswith(b"\n"):
        raw += b"\n"
    return _add_object(objects, object_format, "commit", raw)


@dataclass(frozen=True)
class BuiltFixtureRepository:
    repository_path: Path
    base_revision: str
    head_revision: str
    base_tree: str
    head_tree: str
    base_source_digest: str
    head_source_digest: str
    source_digest: str
    object_format: str

    def to_repository(self, relative_path: str) -> Repository:
        """Return the sole canonical descriptor type from ``models``."""

        return Repository(
            source=RepositorySource.FIXTURE,
            path=relative_path,
            url=None,
            base_revision=self.base_revision,
            head_revision=self.head_revision,
        )


class FixtureRepositoryBuilder:
    """Build deterministic bare objects from complete ``base``/``head`` trees."""

    def __init__(self, *, object_format: str = "sha1") -> None:
        if object_format not in {"sha1", "sha256"}:
            raise ValueError("object_format must be sha1 or sha256")
        self.object_format = object_format

    def build(
        self, fixture_root: os.PathLike[str] | str, destination: os.PathLike[str] | str
    ) -> BuiltFixtureRepository:
        source = _ensure_secure_directory(
            fixture_root, create=False, context="fixture repository root"
        )
        destination_path = _absolute_path(destination)
        destination_parent = _ensure_secure_directory(
            destination_path.parent,
            create=True,
            context="fixture destination parent",
        )
        snapshot = destination_parent / (
            ".fixture-source.%s.staging" % uuid.uuid4().hex
        )
        try:
            _copy_relative_directory_snapshot(
                source.parent,
                source.name,
                snapshot,
                expected_root_identity=_filesystem_identity(
                    source.parent, "fixture source parent"
                ),
                maximum_bytes=MAX_MATERIALIZED_BYTES,
                maximum_nodes=MAX_GIT_METADATA_NODES,
                context="fixture repository source",
            )
            return self._build_from_snapshot(snapshot, destination_path)
        finally:
            if os.path.lexists(snapshot):
                _remove_tree_safely(destination_parent, snapshot)

    def _build_from_snapshot(
        self, fixture_root: os.PathLike[str] | str, destination: os.PathLike[str] | str
    ) -> BuiltFixtureRepository:
        root = _ensure_secure_directory(
            fixture_root, create=False, context="fixture repository root"
        )
        try:
            entries = list(os.scandir(root))
        except OSError as exc:
            raise RepositorySecurityError("could not inspect fixture repository") from exc
        names: Set[str] = set()
        for entry in entries:
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise RepositorySecurityError("could not inspect fixture root entry") from exc
            if _is_link_or_reparse(info):
                raise RepositorySecurityError("fixture root contains a link or reparse point")
            if not stat.S_ISDIR(info.st_mode):
                raise RepositoryPolicyError("fixture root must contain only base and head")
            names.add(entry.name)
        if names != {"base", "head"}:
            raise RepositoryPolicyError("fixture requires exactly base and head trees")
        base_files = _scan_fixture_snapshot(root / "base")
        head_files = _scan_fixture_snapshot(root / "head")
        objects: Dict[str, _GitObject] = {}
        base_tree = _build_fixture_tree(base_files, self.object_format, objects)
        head_tree = _build_fixture_tree(head_files, self.object_format, objects)
        base_revision = _fixture_commit(
            objects,
            self.object_format,
            tree=base_tree,
            parent=None,
            message=b"review-agent-eval fixture base",
        )
        head_revision = _fixture_commit(
            objects,
            self.object_format,
            tree=head_tree,
            parent=base_revision,
            message=b"review-agent-eval fixture head",
        )
        closure = _closure_from_objects(
            objects,
            object_format=self.object_format,
            base_revision=base_revision,
            head_revision=head_revision,
        )

        destination_path = _absolute_path(destination)
        parent = _ensure_secure_directory(
            destination_path.parent, create=True, context="fixture destination parent"
        )
        if os.path.lexists(destination_path):
            raise RepositoryPreparationError("fixture destination already exists")
        staging = parent / (".%s.%s.staging" % (destination_path.name, uuid.uuid4().hex))
        try:
            _write_loose_repository(staging, closure, bare=True)
            verified, _cache_bytes = _read_loose_repository(
                staging,
                object_format=self.object_format,
                base_revision=base_revision,
                head_revision=head_revision,
            )
            if verified.source_digest != closure.source_digest:
                raise RepositoryIntegrityError("fixture repository verification drifted")
            try:
                _rename_directory_no_replace(staging, destination_path)
            except OSError as exc:
                raise RepositoryPreparationError(
                    "could not atomically publish fixture repository"
                ) from exc
        finally:
            if os.path.lexists(staging):
                try:
                    _remove_tree_safely(parent, staging)
                except Exception:
                    pass
        return BuiltFixtureRepository(
            repository_path=destination_path,
            base_revision=base_revision,
            head_revision=head_revision,
            base_tree=base_tree,
            head_tree=head_tree,
            base_source_digest=closure.base_source_digest,
            head_source_digest=closure.head_source_digest,
            source_digest=closure.source_digest,
            object_format=self.object_format,
        )


def _unlink_or_remove_leaf(parent: Path, name: str, info: os.stat_result) -> None:
    target = parent / name
    if stat.S_ISDIR(info.st_mode) and not _is_link_or_reparse(info):
        try:
            os.rmdir(target)
        except OSError:
            raise
    else:
        os.unlink(target)


def _remove_tree_posix_fd(parent: Path, name: str) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        parent_fd = os.open(parent, flags)
    except OSError as exc:
        raise RepositorySecurityError("could not open controlled deletion root") from exc
    try:
        try:
            info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except OSError as exc:
            raise RepositorySecurityError("could not inspect controlled deletion target") from exc
        if _is_link_or_reparse(info) or not stat.S_ISDIR(info.st_mode):
            try:
                os.unlink(name, dir_fd=parent_fd)
            except OSError as exc:
                raise RepositorySecurityError("could not remove controlled deletion leaf") from exc
            return
        child_fd = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        try:
            opened = os.fstat(child_fd)
            if not _same_identity(info, opened):
                raise RepositorySecurityError("controlled deletion target changed identity")
            for child_name in os.listdir(child_fd):
                child_info = os.stat(child_name, dir_fd=child_fd, follow_symlinks=False)
                if _is_link_or_reparse(child_info) or not stat.S_ISDIR(child_info.st_mode):
                    os.unlink(child_name, dir_fd=child_fd)
                else:
                    _remove_tree_fd(child_fd, child_name)
            os.rmdir(name, dir_fd=parent_fd)
        finally:
            os.close(child_fd)
    finally:
        os.close(parent_fd)


def _remove_tree_fd(parent_fd: int, name: str) -> None:
    child_fd = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=parent_fd,
    )
    try:
        for child_name in os.listdir(child_fd):
            child_info = os.stat(child_name, dir_fd=child_fd, follow_symlinks=False)
            if _is_link_or_reparse(child_info) or not stat.S_ISDIR(child_info.st_mode):
                os.unlink(child_name, dir_fd=child_fd)
            else:
                _remove_tree_fd(child_fd, child_name)
        os.rmdir(name, dir_fd=parent_fd)
    finally:
        os.close(child_fd)


def _remove_tree_windows(root: Path, target: Path) -> None:
    # Windows has no portable dir_fd API in Python.  Every traversal step is
    # lstat(follow_symlinks=False) checked, and reparse points are removed as
    # leaves rather than traversed.  The caller holds the controlled root
    # namespace and the target is always a direct child of it.
    try:
        info = os.lstat(target)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise RepositorySecurityError("could not inspect controlled deletion target") from exc
    if _is_link_or_reparse(info) or not stat.S_ISDIR(info.st_mode):
        try:
            if stat.S_ISREG(info.st_mode):
                os.chmod(target, stat.S_IREAD | stat.S_IWRITE)
            os.unlink(target)
        except PermissionError:
            os.rmdir(target)
        except OSError as exc:
            raise RepositorySecurityError("could not remove controlled deletion leaf") from exc
        return
    try:
        entries = list(os.scandir(target))
    except OSError as exc:
        raise RepositorySecurityError("could not enumerate controlled deletion target") from exc
    for entry in entries:
        child = Path(entry.path)
        try:
            child_info = entry.stat(follow_symlinks=False)
        except OSError as exc:
            raise RepositorySecurityError("could not inspect controlled deletion node") from exc
        if _is_link_or_reparse(child_info) or not stat.S_ISDIR(child_info.st_mode):
            try:
                if stat.S_ISREG(child_info.st_mode):
                    os.chmod(child, stat.S_IREAD | stat.S_IWRITE)
                os.unlink(child)
            except PermissionError:
                os.rmdir(child)
        else:
            _remove_tree_windows(target, child)
    try:
        os.rmdir(target)
    except OSError as exc:
        raise RepositorySecurityError("could not remove controlled deletion directory") from exc


def _remove_tree_safely(root: Path, target: Path) -> None:
    root, target = _validate_direct_child(root, target, "controlled deletion")
    if not os.path.lexists(target):
        return
    if os.name == "nt":
        _remove_tree_windows(root, target)
    else:
        _remove_tree_posix_fd(root, target.name)


class _ProcessLock:
    def __init__(self, path: Path, timeout_seconds: float) -> None:
        self.path = path
        self.timeout_seconds = _positive_timeout(timeout_seconds, "lock timeout")
        self.handle: Optional[BinaryIO] = None
        self.locked = False
        self.identity: Optional[Tuple[int, int]] = None

    def __enter__(self) -> "_ProcessLock":
        parent = self.path.parent
        _assert_secure_directory(parent, "repository lock parent")
        if os.path.lexists(self.path):
            info = os.lstat(self.path)
            if _is_link_or_reparse(info) or not stat.S_ISREG(info.st_mode):
                raise RepositorySecurityError("repository lock is unsafe")
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.path, flags, 0o600)
            self.handle = os.fdopen(descriptor, "r+b", buffering=0)
        except OSError as exc:
            raise RepositorySecurityError("could not open repository process lock") from exc
        deadline = time.monotonic() + self.timeout_seconds
        try:
            while True:
                try:
                    if os.name == "nt":
                        import msvcrt

                        self.handle.seek(0)
                        if os.path.getsize(self.path) == 0:
                            self.handle.write(b"\0")
                            self.handle.flush()
                        self.handle.seek(0)
                        msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.locked = True
                    opened = os.fstat(self.handle.fileno())
                    current = os.lstat(self.path)
                    if not _same_identity(opened, current):
                        raise RepositorySecurityError(
                            "repository lock path changed identity"
                        )
                    self.identity = (
                        int(getattr(opened, "st_dev", 0)),
                        int(getattr(opened, "st_ino", 0)),
                    )
                    return self
                except (OSError, BlockingIOError):
                    if time.monotonic() >= deadline:
                        raise RepositoryPreparationError("repository cache lock timed out")
                    time.sleep(0.02)
        except Exception:
            self.__exit__(None, None, None)
            raise

    def assert_current(self) -> None:
        if self.handle is None or not self.locked or self.identity is None:
            raise RepositorySecurityError("repository lock is not held")
        opened = os.fstat(self.handle.fileno())
        try:
            current = os.lstat(self.path)
        except OSError as exc:
            raise RepositorySecurityError("repository lock path disappeared") from exc
        identity = (
            int(getattr(opened, "st_dev", 0)),
            int(getattr(opened, "st_ino", 0)),
        )
        if identity != self.identity or not _same_identity(opened, current):
            raise RepositorySecurityError("repository lock path changed identity")

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        if self.handle is None:
            return
        if self.locked:
            try:
                if os.name == "nt":
                    import msvcrt

                    self.handle.seek(0)
                    msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        self.handle.close()
        self.handle = None
        self.locked = False
        self.identity = None


def _windows_open_operation_sentinel(path: Path) -> int:
    if os.name != "nt":
        raise RepositorySecurityError("Windows operation sentinels are unavailable")
    try:
        import ctypes
        from ctypes import wintypes

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
        kernel32.SetHandleInformation.argtypes = (
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.DWORD,
        )
        kernel32.SetHandleInformation.restype = wintypes.BOOL
        handle = create_file(
            str(path),
            # FILE_READ_DATA makes this a data handle.  A metadata-only
            # FILE_READ_ATTRIBUTES handle does not reliably participate in
            # delete-sharing checks, so it cannot act as a child-held lease.
            0x00000001 | 0x0080,
            0x00000001 | 0x00000002,
            None,
            4,
            0x00200000 | 0x00000080,
            None,
        )
        invalid = ctypes.c_void_p(-1).value
        value = int(getattr(handle, "value", handle) or 0)
        if not value or value == invalid:
            raise OSError(
                ctypes.get_last_error(), "CreateFileW operation sentinel failed"
            )
        try:
            actual, attributes = _windows_handle_path_and_attributes(value)
            if (
                _normalized_path(actual) != _normalized_path(path)
                or attributes & _REPARSE_POINT
                or attributes & 0x10
            ):
                raise OSError("operation sentinel is unsafe")
            if not kernel32.SetHandleInformation(
                wintypes.HANDLE(value), 0x00000001, 0x00000001
            ):
                raise OSError(
                    ctypes.get_last_error(), "SetHandleInformation failed"
                )
            return value
        except BaseException:
            _windows_close_handle(value)
            raise
    except (ImportError, OSError, TypeError, ValueError) as exc:
        raise RepositorySecurityError(
            "could not create a Windows operation sentinel"
        ) from exc


def _windows_operation_sentinel_is_held(path: Path) -> bool:
    if os.name != "nt" or not os.path.lexists(path):
        return False
    try:
        import ctypes
        from ctypes import wintypes

        info = os.lstat(path)
        if _is_link_or_reparse(info) or not stat.S_ISREG(info.st_mode):
            raise RepositorySecurityError("operation sentinel is unsafe")
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
            0x00010000 | 0x0080,
            0x00000001 | 0x00000002 | 0x00000004,
            None,
            3,
            0x00200000,
            None,
        )
        invalid = ctypes.c_void_p(-1).value
        value = int(getattr(handle, "value", handle) or 0)
        if not value or value == invalid:
            error = ctypes.get_last_error()
            if error == 32:
                return True
            raise OSError(error, "operation sentinel probe failed")
        _windows_close_handle(value)
        return False
    except RepositorySecurityError:
        raise
    except (ImportError, OSError, TypeError, ValueError) as exc:
        raise RepositorySecurityError(
            "could not verify a Windows operation sentinel"
        ) from exc


def _operation_sentinel_is_free(path: Path) -> bool:
    if os.name != "nt":
        return True
    if _windows_operation_sentinel_is_held(path):
        return False
    if os.path.lexists(path):
        try:
            os.unlink(path)
        except OSError as exc:
            raise RepositorySecurityError(
                "could not remove an inactive operation sentinel"
            ) from exc
    return True


class _OperationLease:
    """A GC lease inherited by Git so parent death cannot expose live staging."""

    def __init__(self, lock_root: Path, key: str, timeout_seconds: float) -> None:
        if re.fullmatch(r"[0-9a-f]{32}", key) is None:
            raise RepositorySecurityError("operation lease key is invalid")
        self.lock_path = lock_root / ("operation-" + key + ".lock")
        self.sentinel_path = lock_root / ("operation-" + key + ".lease")
        self.timeout_seconds = _positive_timeout(
            timeout_seconds, "operation lease timeout"
        )
        self.process_lock = _ProcessLock(
            self.lock_path, self.timeout_seconds
        )
        self.sentinel_handle: Optional[int] = None

    def __enter__(self) -> "_OperationLease":
        self.process_lock.__enter__()
        try:
            if os.name == "nt":
                deadline = time.monotonic() + self.timeout_seconds
                while not _operation_sentinel_is_free(self.sentinel_path):
                    if time.monotonic() >= deadline:
                        raise RepositoryPreparationError(
                            "operation lease remains held by a child process"
                        )
                    time.sleep(0.02)
                self.sentinel_handle = _windows_open_operation_sentinel(
                    self.sentinel_path
                )
            return self
        except BaseException:
            self.process_lock.__exit__(None, None, None)
            raise

    def posix_descriptor(self) -> int:
        if os.name == "nt" or self.process_lock.handle is None:
            raise RepositorySecurityError("POSIX operation lease is unavailable")
        return self.process_lock.handle.fileno()

    def windows_handle(self) -> int:
        if os.name != "nt" or self.sentinel_handle is None:
            raise RepositorySecurityError("Windows operation lease is unavailable")
        return self.sentinel_handle

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        if self.sentinel_handle is not None:
            _windows_close_handle(self.sentinel_handle)
            self.sentinel_handle = None
        self.process_lock.__exit__(_type, _value, _traceback)
        if os.name == "nt" and os.path.lexists(self.sentinel_path):
            try:
                os.unlink(self.sentinel_path)
            except PermissionError:
                # A child inherited the no-delete handle and still owns the
                # operation.  Recovery will retry after that process exits.
                pass
            except OSError as exc:
                raise RepositorySecurityError(
                    "could not release Windows operation sentinel"
                ) from exc


def _atomic_write_control_file(path: Path, data: bytes) -> None:
    parent = path.parent
    _assert_secure_directory(parent, "control file parent")
    if os.path.lexists(path):
        existing = _read_regular_file(path, maximum=max(len(data), 1) + 1, context="control file")
        if existing == data:
            return
        raise RepositoryIntegrityError("immutable control file conflicts with its identity")
    temporary = parent / (".%s.%s.tmp" % (path.name, uuid.uuid4().hex))
    try:
        _write_regular_file_exclusive(temporary, data, mode=0o600, fsync=True)
        try:
            if os.name == "nt":
                os.rename(temporary, path)
            else:
                os.link(temporary, path, follow_symlinks=False)
                os.unlink(temporary)
        except FileExistsError as exc:
            raise RepositoryIntegrityError(
                "immutable control file was concurrently published"
            ) from exc
        except OSError as exc:
            raise RepositoryPreparationError("could not atomically publish control file") from exc
    finally:
        if os.path.lexists(temporary):
            try:
                os.unlink(temporary)
            except OSError:
                pass


def _rename_directory_no_replace(source: Path, target: Path) -> None:
    """Atomically publish one directory without ever replacing ``target``."""

    source = _absolute_path(source)
    target = _absolute_path(target)
    _assert_secure_directory(source, "directory publication source")
    _assert_secure_directory(source.parent, "directory publication source parent")
    _assert_secure_directory(target.parent, "directory publication target parent")
    if os.path.lexists(target):
        raise FileExistsError(errno.EEXIST, "directory target exists", str(target))
    if os.name == "nt":
        with _guard_windows_directory_chain(source.parent):
            with _guard_windows_directory_chain(target.parent):
                try:
                    os.rename(source, target)
                except PermissionError as exc:
                    if os.path.lexists(target):
                        raise FileExistsError(
                            errno.EEXIST,
                            "directory target exists",
                            str(target),
                        ) from exc
                    raise
        return

    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    target_bytes = os.fsencode(target)
    if hasattr(libc, "renameat2"):
        renameat2 = libc.renameat2
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            source_bytes,
            -100,
            target_bytes,
            1,
        )
    elif hasattr(libc, "renamex_np"):
        renamex = libc.renamex_np
        renamex.argtypes = (
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renamex.restype = ctypes.c_int
        result = renamex(source_bytes, target_bytes, 0x00000004)
    else:
        raise RepositorySecurityError(
            "atomic no-replace directory publication is unavailable"
        )
    if result != 0:
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(error, "directory target exists", str(target))
        raise OSError(error, "atomic no-replace directory publication failed")
    try:
        parent_descriptor = os.open(
            target.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except OSError as exc:
        raise RepositoryPreparationError(
            "could not persist directory publication metadata"
        ) from exc


@dataclass(frozen=True)
class PreparedRepository:
    """Runtime handle: canonical manifest plus the validated cache directory."""

    manifest: PreparedRepositoryManifest
    cache_path: Path
    repository: Repository
    acquisition_binding_digest: Optional[str]
    git_version: str
    git_executable_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, PreparedRepositoryManifest):
            raise TypeError("manifest must be a PreparedRepositoryManifest")
        if not isinstance(self.cache_path, Path) or not self.cache_path.is_absolute():
            raise TypeError("cache_path must be an absolute Path handle")
        if not isinstance(self.repository, Repository):
            raise TypeError("repository must be the canonical Repository")
        if (
            self.repository.base_revision != self.manifest.base_revision
            or self.repository.head_revision != self.manifest.head_revision
        ):
            raise RepositoryIntegrityError(
                "PreparedRepository Repository revisions do not match content"
            )
        if self.acquisition_binding_digest is not None:
            _digest(
                self.acquisition_binding_digest,
                "PreparedRepository.acquisition_binding_digest",
            )
        _string(self.git_version, "PreparedRepository.git_version", 256)
        _digest(
            self.git_executable_sha256,
            "PreparedRepository.git_executable_sha256",
        )

    @property
    def cache_id(self) -> str:
        return stable_id("repository-cache", self.manifest.prepared_repository_id)


def canonical_repository_path(value: Any) -> str:
    """Return one path under the exact policy used by prepared Git trees.

    The validator is public so evaluator components cannot silently develop a
    second notion of a canonical repository path.  It performs no correction:
    callers receive the original string or a policy/limit exception.
    """

    if type(value) is not str or not value:
        raise RepositoryPolicyError(
            "repository path must be a non-empty string"
        )
    encoded = _validate_repo_path(value.split("/"), "repository path")
    if encoded.decode("utf-8", "strict") != value:
        raise RepositoryPolicyError(
            "repository path must be canonical UTF-8 POSIX form"
        )
    return value


def _replay_byte_limit(value: Any, context: str, maximum: int) -> int:
    if type(value) is not int or value < 1 or value > maximum:
        raise RepositoryLimitError(
            "%s must be an integer from 1 through %d" % (context, maximum)
        )
    return value


def _tree_file_index(
    closure: _RepositoryClosure, tree_oid: str
) -> Mapping[str, str]:
    files: Dict[str, str] = {}
    logical_entries = 0

    def visit(oid: str, prefix: Tuple[str, ...]) -> None:
        nonlocal logical_entries
        tree = closure.objects.get(oid)
        if tree is None:
            raise RepositoryIntegrityError(
                "repository replay tree references a missing object"
            )
        for entry in _parse_tree(tree, closure.object_format):
            logical_entries += 1
            if logical_entries > MAX_LOGICAL_TREE_ENTRIES:
                raise RepositoryLimitError(
                    "repository replay tree exceeds logical entry expansion budget"
                )
            parts = (*prefix, entry.name)
            path_bytes = _validate_repo_path(parts, "repository replay tree")
            path = path_bytes.decode("utf-8", "strict")
            if entry.object_type == "tree":
                visit(entry.oid, parts)
                continue
            if path in files:
                raise RepositoryIntegrityError(
                    "repository replay tree contains a duplicate path"
                )
            files[path] = entry.oid

    visit(tree_oid, ())
    return MappingProxyType(dict(sorted(files.items())))


@dataclass(frozen=True)
class PreparedRepositoryReplay:
    """Read-only replay of the exact base/head objects in a prepared cache.

    Instances are issued only by :meth:`RepositoryPreparer.open_replay` after
    the cache index, manifest, complete object closure, and Git executable have
    been revalidated.  File reads come from that immutable in-memory closure;
    diffs use the same bounded, isolated Git process boundary used during
    preparation and never inspect a Trial workspace.
    """

    prepared_repository_id: str
    repository_descriptor_digest: str
    base_revision: str
    head_revision: str
    _git_dir: Path = field(repr=False, compare=False)
    _runner: _GitRunner = field(repr=False, compare=False)
    _open_check: Callable[[], None] = field(repr=False, compare=False)
    _verify_cache: Callable[[], None] = field(repr=False, compare=False)
    _objects: Mapping[str, _GitObject] = field(repr=False, compare=False)
    _files_by_revision: Mapping[str, Mapping[str, str]] = field(
        repr=False, compare=False
    )

    @classmethod
    def _from_verified(
        cls,
        prepared: PreparedRepository,
        closure: _RepositoryClosure,
        runner: _GitRunner,
        open_check: Callable[[], None],
        verify_cache: Callable[[], None],
    ) -> "PreparedRepositoryReplay":
        if (
            closure.base_revision != prepared.manifest.base_revision
            or closure.head_revision != prepared.manifest.head_revision
            or closure.source_digest != prepared.manifest.source_digest
        ):
            raise RepositoryIntegrityError(
                "repository replay closure does not match PreparedRepository"
            )
        base_files = _tree_file_index(closure, closure.base_tree)
        head_files = _tree_file_index(closure, closure.head_tree)
        files_by_revision: Mapping[str, Mapping[str, str]] = MappingProxyType(
            {
                closure.base_revision: base_files,
                closure.head_revision: head_files,
            }
        )
        return cls(
            prepared_repository_id=prepared.manifest.prepared_repository_id,
            repository_descriptor_digest=prepared.repository.digest(),
            base_revision=closure.base_revision,
            head_revision=closure.head_revision,
            _git_dir=prepared.cache_path,
            _runner=runner,
            _open_check=open_check,
            _verify_cache=verify_cache,
            _objects=MappingProxyType(dict(closure.objects)),
            _files_by_revision=files_by_revision,
        )

    def _files(self, revision: Any) -> Mapping[str, str]:
        self._open_check()
        if type(revision) is not str or revision not in self._files_by_revision:
            raise RepositoryIntegrityError(
                "repository replay revision must be the exact base or head"
            )
        return self._files_by_revision[revision]

    def paths(self, revision: str) -> Tuple[str, ...]:
        """Return a stable case-sensitive path catalog for one exact revision."""

        return tuple(self._files(revision))

    def contains_path(self, revision: str, path: str) -> bool:
        canonical = canonical_repository_path(path)
        return canonical in self._files(revision)

    def read_file(
        self,
        revision: str,
        path: str,
        *,
        max_bytes: int = MAX_GIT_BLOB_BYTES,
    ) -> Optional[bytes]:
        """Read a regular blob from an exact revision, or ``None`` if absent."""

        limit = _replay_byte_limit(
            max_bytes, "repository replay file byte limit", MAX_GIT_BLOB_BYTES
        )
        canonical = canonical_repository_path(path)
        oid = self._files(revision).get(canonical)
        if oid is None:
            return None
        obj = self._objects.get(oid)
        if obj is None or obj.object_type != "blob":
            raise RepositoryIntegrityError(
                "repository replay path does not resolve to a verified blob"
            )
        if len(obj.raw) > limit:
            raise RepositoryLimitError(
                "repository replay file exceeds its requested byte limit"
            )
        return obj.raw

    def diff(
        self,
        path: str,
        *,
        max_bytes: int = MAX_GIT_STDOUT_BYTES,
    ) -> bytes:
        """Replay the canonical full ``base..head`` diff for one path."""

        limit = _replay_byte_limit(
            max_bytes, "repository replay diff byte limit", MAX_GIT_STDOUT_BYTES
        )
        self._open_check()
        canonical = canonical_repository_path(path)
        self._verify_cache()
        try:
            try:
                result = self._runner.run(
                    [
                        "--literal-pathspecs",
                        "--git-dir",
                        str(self._git_dir),
                        "diff",
                        "--no-color",
                        "--no-ext-diff",
                        "--unified=3",
                        "%s..%s" % (self.base_revision, self.head_revision),
                        "--",
                        canonical,
                    ],
                    stdout_limit=limit,
                )
            except _GitCommandFailure as exc:
                raise RepositoryIntegrityError(
                    "canonical repository diff replay failed"
                ) from exc
        finally:
            self._verify_cache()
        return result.stdout


@dataclass(frozen=True)
class RepositoryAcquisitionBinding(_JsonModel):
    """Harness-only HTTPS attestation; it is never exposed to a Trial."""

    SCHEMA_VERSION: ClassVar[str] = REPOSITORY_ACQUISITION_BINDING_SCHEMA_VERSION

    repository: Repository
    expected_source_digest: str
    suite_source: SuiteSource
    allowed_host: str
    allowed_port: int
    redirect_allowlist: Tuple[str, ...] = ()
    schema_version: str = REPOSITORY_ACQUISITION_BINDING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != self.SCHEMA_VERSION:
            raise SchemaError("acquisition binding has unknown schema_version")
        if not isinstance(self.repository, Repository):
            raise SchemaError("acquisition repository must be a Repository")
        if (
            self.repository.source is not RepositorySource.GIT
            or self.repository.path is not None
            or self.repository.url is None
        ):
            raise SchemaError(
                "remote acquisition requires one canonical remote Repository"
            )
        _digest(self.expected_source_digest, "acquisition expected_source_digest")
        if not isinstance(self.suite_source, SuiteSource):
            raise SchemaError("acquisition suite_source must be a SuiteSource")
        if self.suite_source.license is None:
            raise SchemaError("remote acquisition requires licensed Suite provenance")
        parsed = _validate_https_url(self.repository.url)
        host = _canonical_host(self.allowed_host)
        port = _integer(
            self.allowed_port,
            "acquisition allowed_port",
            minimum=1,
            maximum=65535,
        )
        if parsed.hostname is None or _canonical_host(parsed.hostname) != host:
            raise SchemaError("acquisition URL is outside the host allowlist")
        if (parsed.port or 443) != port:
            raise SchemaError("acquisition URL is outside the port allowlist")
        if type(self.redirect_allowlist) is not tuple:
            raise SchemaError("redirect_allowlist must be a tuple")
        canonical_redirects = tuple(
            _validate_https_origin(value) for value in self.redirect_allowlist
        )
        if canonical_redirects != tuple(sorted(set(canonical_redirects))):
            raise SchemaError("redirect_allowlist must be unique and sorted")
        # Git cannot enforce an origin allowlist across redirects.  Official v1
        # therefore binds an explicit empty allowlist and disables redirects.
        if canonical_redirects:
            raise SchemaError("repository_isolation_v1 does not permit redirects")

    @classmethod
    def from_dict(cls, value: Any) -> "RepositoryAcquisitionBinding":
        payload = _object(value, "repository acquisition binding")
        fields = (
            "schema_version",
            "repository",
            "expected_source_digest",
            "suite_source",
            "allowed_host",
            "allowed_port",
            "redirect_allowlist",
        )
        _exact_fields(payload, fields, "repository acquisition binding")
        redirects = payload["redirect_allowlist"]
        if type(redirects) is not list:
            raise SchemaError("redirect_allowlist must be a JSON array")
        return cls(
            schema_version=payload["schema_version"],
            repository=Repository.from_dict(payload["repository"]),
            expected_source_digest=_digest(
                payload["expected_source_digest"],
                "acquisition expected_source_digest",
            ),
            suite_source=SuiteSource.from_dict(payload["suite_source"]),
            allowed_host=_string(
                payload["allowed_host"], "acquisition allowed_host", 253
            ),
            allowed_port=_integer(
                payload["allowed_port"],
                "acquisition allowed_port",
                minimum=1,
                maximum=65535,
            ),
            redirect_allowlist=tuple(redirects),
        )

    @classmethod
    def from_json(cls, data: Any) -> "RepositoryAcquisitionBinding":
        return cls.from_dict(
            _strict_json_loads(
                data,
                MAX_ACQUISITION_BINDING_BYTES,
                "repository acquisition binding JSON",
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "repository": self.repository.to_dict(),
            "expected_source_digest": self.expected_source_digest,
            "suite_source": self.suite_source.to_dict(),
            "allowed_host": self.allowed_host,
            "allowed_port": self.allowed_port,
            "redirect_allowlist": list(self.redirect_allowlist),
        }


def _workspace_binding_id(
    *,
    trial_manifest: TrialManifest,
    suite_case: SuiteCase,
    attempt: int,
    prepared_repository: PreparedRepositoryManifest,
    repository: Repository,
    acquisition_binding_digest: Optional[str],
    git_version: str,
    git_executable_sha256: str,
) -> str:
    return stable_id(
        "workspace-binding",
        WORKSPACE_MANIFEST_SCHEMA_VERSION,
        trial_manifest.digest(),
        suite_case.digest(),
        attempt,
        prepared_repository.digest(),
        repository.digest(),
        acquisition_binding_digest,
        git_version,
        git_executable_sha256,
    )


@dataclass(frozen=True)
class WorkspaceManifest(_JsonModel):
    """Canonical Trial/workspace binding with no denormalized repository copy."""

    SCHEMA_VERSION: ClassVar[str] = WORKSPACE_MANIFEST_SCHEMA_VERSION

    schema_version: str
    trial_manifest: TrialManifest
    suite_case: SuiteCase
    attempt: int
    prepared_repository: PreparedRepositoryManifest
    repository: Repository
    acquisition_binding_digest: Optional[str]
    git_version: str
    git_executable_sha256: str
    workspace_binding_id: str

    def __post_init__(self) -> None:
        if self.schema_version != self.SCHEMA_VERSION:
            raise SchemaError("workspace manifest has unknown schema_version")
        if not isinstance(self.trial_manifest, TrialManifest):
            raise SchemaError("workspace trial_manifest must be a TrialManifest")
        if not isinstance(self.suite_case, SuiteCase):
            raise SchemaError("workspace suite_case must be a SuiteCase")
        if not isinstance(self.prepared_repository, PreparedRepositoryManifest):
            raise SchemaError(
                "workspace prepared_repository must be a PreparedRepositoryManifest"
            )
        _integer(
            self.attempt,
            "workspace attempt",
            minimum=1,
            maximum=MAX_TRIAL_ATTEMPT,
        )
        if not isinstance(self.repository, Repository):
            raise SchemaError("workspace repository must be a Repository")
        if self.acquisition_binding_digest is not None:
            _digest(
                self.acquisition_binding_digest,
                "workspace acquisition_binding_digest",
            )
        _string(self.git_version, "workspace git_version", 256)
        _digest(
            self.git_executable_sha256,
            "workspace git_executable_sha256",
        )
        if (
            self.suite_case.task_id != self.trial_manifest.task_id
            or self.suite_case.canonical_case_digest
            != self.trial_manifest.canonical_case_digest
            or self.suite_case.eval_input_digest
            != self.trial_manifest.eval_input_digest
        ):
            raise SchemaError(
                "workspace SuiteCase does not match immutable TrialManifest"
            )
        if (
            self.repository.base_revision
            != self.prepared_repository.base_revision
            or self.repository.head_revision
            != self.prepared_repository.head_revision
        ):
            raise SchemaError(
                "workspace Repository revisions do not match prepared content"
            )
        expected = _workspace_binding_id(
            trial_manifest=self.trial_manifest,
            suite_case=self.suite_case,
            attempt=self.attempt,
            prepared_repository=self.prepared_repository,
            repository=self.repository,
            acquisition_binding_digest=self.acquisition_binding_digest,
            git_version=self.git_version,
            git_executable_sha256=self.git_executable_sha256,
        )
        if self.workspace_binding_id != expected:
            raise SchemaError("workspace_binding_id does not match canonical binding")
        if len(canonical_json_bytes(self)) > MAX_WORKSPACE_MANIFEST_BYTES:
            raise SchemaError("workspace manifest exceeds its canonical byte limit")

    @classmethod
    def create(
        cls,
        prepared: PreparedRepository,
        *,
        trial_manifest: TrialManifest,
        suite_case: SuiteCase,
        eval_input: EvalInput,
        attempt: int,
    ) -> "WorkspaceManifest":
        if not isinstance(prepared, PreparedRepository):
            raise SchemaError("workspace requires a PreparedRepository")
        if not isinstance(trial_manifest, TrialManifest):
            raise SchemaError("workspace requires a TrialManifest")
        if not isinstance(suite_case, SuiteCase):
            raise SchemaError("workspace requires a SuiteCase")
        if not isinstance(eval_input, EvalInput):
            raise SchemaError("workspace requires an EvalInput")
        if (
            eval_input.task_id != trial_manifest.task_id
            or eval_input.digest() != trial_manifest.eval_input_digest
            or eval_input.digest() != suite_case.eval_input_digest
        ):
            raise SchemaError("EvalInput does not match immutable Trial binding")
        try:
            input_repository = repository_from_eval_input(eval_input)
        except RepositoryPreparationError as exc:
            raise SchemaError(
                "workspace requires a Repository review target"
            ) from exc
        if input_repository != prepared.repository:
            raise SchemaError(
                "EvalInput Repository does not match PreparedRepository request"
            )
        binding_id = _workspace_binding_id(
            trial_manifest=trial_manifest,
            suite_case=suite_case,
            attempt=attempt,
            prepared_repository=prepared.manifest,
            repository=prepared.repository,
            acquisition_binding_digest=prepared.acquisition_binding_digest,
            git_version=prepared.git_version,
            git_executable_sha256=prepared.git_executable_sha256,
        )
        return cls(
            schema_version=cls.SCHEMA_VERSION,
            trial_manifest=trial_manifest,
            suite_case=suite_case,
            attempt=attempt,
            prepared_repository=prepared.manifest,
            repository=prepared.repository,
            acquisition_binding_digest=prepared.acquisition_binding_digest,
            git_version=prepared.git_version,
            git_executable_sha256=prepared.git_executable_sha256,
            workspace_binding_id=binding_id,
        )

    @classmethod
    def from_dict(cls, value: Any) -> "WorkspaceManifest":
        payload = _object(value, "workspace manifest")
        fields = (
            "schema_version",
            "trial_manifest",
            "suite_case",
            "attempt",
            "prepared_repository",
            "repository",
            "acquisition_binding_digest",
            "git_version",
            "git_executable_sha256",
            "workspace_binding_id",
        )
        _exact_fields(payload, fields, "workspace manifest")
        return cls(
            schema_version=payload["schema_version"],
            trial_manifest=TrialManifest.from_dict(payload["trial_manifest"]),
            suite_case=SuiteCase.from_dict(payload["suite_case"]),
            attempt=_integer(
                payload["attempt"],
                "workspace attempt",
                minimum=1,
                maximum=MAX_TRIAL_ATTEMPT,
            ),
            prepared_repository=PreparedRepositoryManifest.from_dict(
                payload["prepared_repository"]
            ),
            repository=Repository.from_dict(payload["repository"]),
            acquisition_binding_digest=(
                None
                if payload["acquisition_binding_digest"] is None
                else _digest(
                    payload["acquisition_binding_digest"],
                    "workspace acquisition_binding_digest",
                )
            ),
            git_version=_string(
                payload["git_version"], "workspace git_version", 256
            ),
            git_executable_sha256=_digest(
                payload["git_executable_sha256"],
                "workspace git_executable_sha256",
            ),
            workspace_binding_id=_identifier(
                payload["workspace_binding_id"], "workspace_binding_id"
            ),
        )

    @classmethod
    def from_json(cls, data: Any) -> "WorkspaceManifest":
        return cls.from_dict(
            _strict_json_loads(
                data, MAX_WORKSPACE_MANIFEST_BYTES, "workspace manifest JSON"
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "trial_manifest": self.trial_manifest.to_dict(),
            "suite_case": self.suite_case.to_dict(),
            "attempt": self.attempt,
            "prepared_repository": self.prepared_repository.to_dict(),
            "repository": self.repository.to_dict(),
            "acquisition_binding_digest": self.acquisition_binding_digest,
            "git_version": self.git_version,
            "git_executable_sha256": self.git_executable_sha256,
            "workspace_binding_id": self.workspace_binding_id,
        }


def _filesystem_identity(path: Path, context: str) -> Tuple[int, int]:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise RepositorySecurityError("%s is unavailable" % context) from exc
    if _is_link_or_reparse(info) or not stat.S_ISDIR(info.st_mode):
        raise RepositorySecurityError("%s is not a secure directory" % context)
    inode = int(getattr(info, "st_ino", 0) or 0)
    if inode == 0:
        raise RepositorySecurityError("%s has no stable filesystem identity" % context)
    return (int(getattr(info, "st_dev", 0) or 0), inode)


def _request_id(
    repository_descriptor_digest: str,
    acquisition_binding_digest: Optional[str],
    git_version: str,
    git_executable_sha256: str,
) -> str:
    return stable_id(
        "repository-request",
        CACHE_INDEX_SCHEMA_VERSION,
        repository_descriptor_digest,
        acquisition_binding_digest,
        git_version,
        git_executable_sha256,
        _fixed_isolation_policy(),
        _fixed_path_policy(),
        REPOSITORY_BUDGET_POLICY_VERSION,
    )


def _request_lock_key(request_id: str) -> str:
    _identifier(request_id, "repository request_id")
    return _digest_bytes(request_id.encode("ascii"))[:32]


def _cache_index_payload(
    *,
    request_id: str,
    repository_descriptor_digest: str,
    acquisition_binding_digest: Optional[str],
    git_version: str,
    git_executable_sha256: str,
    cache_id: str,
    prepared_repository_id: str,
    manifest_digest: str,
) -> Dict[str, Any]:
    return {
        "schema_version": CACHE_INDEX_SCHEMA_VERSION,
        "request_id": request_id,
        "repository_descriptor_digest": repository_descriptor_digest,
        "acquisition_binding_digest": acquisition_binding_digest,
        "git_version": git_version,
        "git_executable_sha256": git_executable_sha256,
        "cache_id": cache_id,
        "prepared_repository_id": prepared_repository_id,
        "manifest_digest": manifest_digest,
    }


def _load_cache_index(path: Path) -> Dict[str, Any]:
    payload = _object(
        _strict_json_loads(
            _read_regular_file(path, maximum=64 * 1024, context="repository cache index"),
            64 * 1024,
            "repository cache index JSON",
        ),
        "repository cache index",
    )
    fields = (
        "schema_version",
        "request_id",
        "repository_descriptor_digest",
        "acquisition_binding_digest",
        "git_version",
        "git_executable_sha256",
        "cache_id",
        "prepared_repository_id",
        "manifest_digest",
    )
    _exact_fields(payload, fields, "repository cache index")
    if payload["schema_version"] != CACHE_INDEX_SCHEMA_VERSION:
        raise RepositoryIntegrityError("repository cache index has unknown schema")
    _identifier(payload["request_id"], "repository request_id")
    if path.name != payload["request_id"] + ".json":
        raise RepositoryIntegrityError("repository cache index path drifted")
    _digest(
        payload["repository_descriptor_digest"],
        "repository index descriptor digest",
    )
    if payload["acquisition_binding_digest"] is not None:
        _digest(
            payload["acquisition_binding_digest"],
            "repository index acquisition digest",
        )
    _string(payload["git_version"], "repository index Git version", 256)
    _digest(
        payload["git_executable_sha256"],
        "repository index Git executable digest",
    )
    _identifier(payload["cache_id"], "repository cache_id")
    _identifier(payload["prepared_repository_id"], "prepared repository ID")
    _digest(payload["manifest_digest"], "repository manifest digest")
    return dict(payload)


def _repository_reservation_payload(request_id: str) -> Dict[str, Any]:
    request_lock_key = _request_lock_key(request_id)
    return {
        "schema_version": REPOSITORY_RESERVATION_SCHEMA_VERSION,
        "request_id": request_id,
        "request_lock_key": request_lock_key,
        "reserved_bytes": MAX_PREPARE_RESERVATION_BYTES,
        "reserved_nodes": MAX_PREPARE_RESERVATION_NODES,
    }


def _load_repository_reservation(path: Path) -> Dict[str, Any]:
    payload = _object(
        _strict_json_loads(
            _read_regular_file(
                path,
                maximum=16 * 1024,
                context="repository capacity reservation",
            ),
            16 * 1024,
            "repository capacity reservation JSON",
        ),
        "repository capacity reservation",
    )
    fields = (
        "schema_version",
        "request_id",
        "request_lock_key",
        "reserved_bytes",
        "reserved_nodes",
    )
    _exact_fields(payload, fields, "repository capacity reservation")
    if payload["schema_version"] != REPOSITORY_RESERVATION_SCHEMA_VERSION:
        raise RepositoryIntegrityError(
            "repository capacity reservation has unknown schema"
        )
    request_id = _identifier(payload["request_id"], "reservation request_id")
    request_lock_key = _string(
        payload["request_lock_key"], "reservation request lock key", 32
    )
    if (
        len(request_lock_key) != 32
        or _HEX_RE.fullmatch(request_lock_key) is None
        or request_lock_key != _request_lock_key(request_id)
        or path.name != request_lock_key + ".json"
    ):
        raise RepositoryIntegrityError(
            "repository capacity reservation identity drifted"
        )
    reserved_bytes = _integer(
        payload["reserved_bytes"],
        "reservation bytes",
        minimum=1,
        maximum=MAX_PREPARE_RESERVATION_BYTES,
    )
    reserved_nodes = _integer(
        payload["reserved_nodes"],
        "reservation nodes",
        minimum=1,
        maximum=MAX_PREPARE_RESERVATION_NODES,
    )
    if (
        reserved_bytes != MAX_PREPARE_RESERVATION_BYTES
        or reserved_nodes != MAX_PREPARE_RESERVATION_NODES
    ):
        raise RepositoryIntegrityError(
            "repository capacity reservation policy drifted"
        )
    return dict(payload)


@dataclass
class TrialWorkspace:
    """Runtime-only lease for one isolated, writable Trial workspace."""

    _path: Path
    manifest: WorkspaceManifest
    _preparer: "RepositoryPreparer" = field(repr=False)
    _directory_identity: Tuple[int, int] = field(repr=False)
    _terminal_status: Optional[TrialStatus] = field(
        default=None, init=False, repr=False
    )
    _closed: bool = field(default=False, init=False, repr=False)
    _retained: bool = field(default=False, init=False, repr=False)
    _entered: bool = field(default=False, init=False, repr=False)
    cleanup_diagnostic: Optional[WorkspaceDiagnostic] = field(
        default=None, init=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self._path, Path) or not self._path.is_absolute():
            raise TypeError("TrialWorkspace.path must be an absolute Path")
        if not isinstance(self.manifest, WorkspaceManifest):
            raise TypeError("TrialWorkspace.manifest must be a WorkspaceManifest")
        if (
            type(self._directory_identity) is not tuple
            or len(self._directory_identity) != 2
        ):
            raise TypeError("TrialWorkspace requires a stable directory identity")

    @property
    def path(self) -> Path:
        return self._path

    @property
    def retained(self) -> bool:
        return self._retained

    @property
    def closed(self) -> bool:
        return self._closed

    def validate(self) -> None:
        """Fail closed unless this is still the active published workspace."""

        if self._closed:
            raise RepositoryPreparationError("workspace lease is already closed")
        self._preparer._assert_workspace_lease(self)

    def read_file(self, relative_path: str) -> bytes:
        """Read one canonical workspace file through Repository safety gates."""

        self.validate()
        canonical = canonical_repository_path(relative_path)
        target = _secure_join(
            self._path,
            canonical,
            "active Trial workspace file",
        )
        if os.name == "nt":
            with _guard_windows_directory_chain(target.parent):
                return _read_regular_file(
                    target,
                    maximum=MAX_GIT_BLOB_BYTES,
                    context="active Trial workspace file",
                )
        return _read_regular_file(
            target,
            maximum=MAX_GIT_BLOB_BYTES,
            context="active Trial workspace file",
        )

    def record_terminal_status(self, status: TrialStatus) -> None:
        if self._closed:
            raise RepositoryPreparationError("closed workspace cannot record outcome")
        if not isinstance(status, TrialStatus) or status not in {
            TrialStatus.COMPLETED,
            TrialStatus.FAILED,
            TrialStatus.BLOCKED,
            TrialStatus.INVALID_OUTPUT,
            TrialStatus.INCOMPLETE,
        }:
            raise ValueError("workspace outcome must be a canonical terminal status")
        if self._terminal_status is not None and self._terminal_status is not status:
            raise RepositoryPreparationError("workspace terminal status is immutable")
        self._terminal_status = status

    def __enter__(self) -> "TrialWorkspace":
        if self._entered:
            raise RepositoryPreparationError("workspace lease is already entered")
        self.validate()
        self._entered = True
        return self

    def close(self) -> None:
        if self._closed:
            return
        if self._terminal_status is None:
            self._terminal_status = TrialStatus.COMPLETED
        self._preparer._release_workspace(self)

    def __exit__(self, exc_type: Any, _value: Any, _traceback: Any) -> bool:
        if exc_type is not None:
            self._terminal_status = TrialStatus.FAILED
        self.close()
        return False


class RepositoryPreparer:
    """Materialize canonical repository descriptors into isolated Trial roots."""

    _INTERNAL_DATA_NAMES = frozenset(
        {
            "repositories",
            "indexes",
            ".staging",
            ".locks",
            ".reservations",
            ".git-control",
        }
    )

    def __init__(
        self,
        *,
        suite_root: os.PathLike[str] | str,
        data_root: os.PathLike[str] | str,
        workspace_root: os.PathLike[str] | str,
        git_executable: os.PathLike[str] | str,
        allow_remote: bool = False,
        acquisition_bindings: Iterable[RepositoryAcquisitionBinding] = (),
        repository_mode: RepositoryMode = RepositoryMode.ACQUIRE,
        retention_policy: WorkspaceRetentionPolicy = WorkspaceRetentionPolicy.DELETE_ALWAYS,
        max_retained_workspaces: int = DEFAULT_MAX_RETAINED_WORKSPACES,
        max_retained_bytes: int = DEFAULT_MAX_RETAINED_BYTES,
        retention_ttl_seconds: int = DEFAULT_RETAINED_TTL_SECONDS,
        git_timeout_seconds: float = DEFAULT_GIT_TIMEOUT_SECONDS,
        lock_timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
    ) -> None:
        if type(allow_remote) is not bool:
            raise TypeError("allow_remote must be a bool")
        if not isinstance(repository_mode, RepositoryMode):
            raise TypeError("repository_mode must be a RepositoryMode")
        if not isinstance(retention_policy, WorkspaceRetentionPolicy):
            raise TypeError("retention_policy must be a WorkspaceRetentionPolicy")
        if type(max_retained_workspaces) is not int or not (
            1 <= max_retained_workspaces <= 100
        ):
            raise ValueError("max_retained_workspaces must be between 1 and 100")
        if type(max_retained_bytes) is not int or not (
            1 <= max_retained_bytes <= 64 * 1024 * 1024 * 1024
        ):
            raise ValueError("max_retained_bytes must be a positive bounded integer")
        if type(retention_ttl_seconds) is not int or not (
            1 <= retention_ttl_seconds <= MAX_RETENTION_TTL_SECONDS
        ):
            raise ValueError("retention_ttl_seconds must be a positive bounded integer")

        suite = _absolute_path(suite_root)
        data = _absolute_path(data_root)
        workspaces = _absolute_path(workspace_root)
        if data.name != ".eval-data":
            raise ValueError("RepositoryPreparer data_root must be an explicit .eval-data")
        if workspaces.name != ".eval-workspaces":
            raise ValueError(
                "RepositoryPreparer workspace_root must be an explicit .eval-workspaces"
            )
        lexical_roots = (("suite", suite), ("data", data), ("workspace", workspaces))
        for left_index, (left_name, left) in enumerate(lexical_roots):
            for right_name, right in lexical_roots[left_index + 1 :]:
                if _path_within(left, right) or _path_within(right, left):
                    raise RepositorySecurityError(
                        "%s and %s roots must be physically separate without overlap"
                        % (left_name, right_name)
                    )
        suite = _ensure_secure_directory(
            suite, create=False, context="repository Suite root"
        )
        create_control_roots = repository_mode is RepositoryMode.ACQUIRE
        data = _ensure_secure_directory(
            data,
            create=create_control_roots,
            context="repository .eval-data root",
        )
        workspaces = _ensure_secure_directory(
            workspaces,
            create=create_control_roots,
            context="repository .eval-workspaces root",
        )
        _assert_suite_directory_authority(suite)
        authority_check = (
            _secure_control_directory_authority
            if create_control_roots
            else _verify_control_directory_authority
        )
        authority_check(data, "repository .eval-data root")
        authority_check(workspaces, "repository .eval-workspaces root")
        self.suite_root = suite
        self.data_root = data
        self.workspace_root = workspaces
        self.allow_remote = allow_remote
        self.repository_mode = repository_mode
        self.retention_policy = retention_policy
        self.max_retained_workspaces = max_retained_workspaces
        self.max_retained_bytes = max_retained_bytes
        self.retention_ttl_seconds = retention_ttl_seconds
        self.lock_timeout_seconds = _positive_timeout(
            lock_timeout_seconds, "repository lock timeout"
        )
        self._root_identities = {
            "suite": _filesystem_identity(suite, "repository Suite root"),
            "data": _filesystem_identity(data, "repository data root"),
            "workspace": _filesystem_identity(workspaces, "repository workspace root"),
        }

        self.cache_root = _ensure_secure_directory(
            data / "repositories",
            create=create_control_roots,
            context="repository cache root",
        )
        self.index_root = _ensure_secure_directory(
            data / "indexes",
            create=create_control_roots,
            context="repository index root",
        )
        self.staging_root = _ensure_secure_directory(
            data / ".staging",
            create=create_control_roots,
            context="repository staging root",
        )
        self.lock_root = _ensure_secure_directory(
            data / ".locks",
            create=create_control_roots,
            context="repository lock root",
        )
        self.reservation_root = _ensure_secure_directory(
            data / ".reservations",
            create=create_control_roots,
            context="repository reservation root",
        )
        self.data_store_lock_path = self.lock_root / "data-store.lock"
        self.active_root = _ensure_secure_directory(
            workspaces / "active",
            create=create_control_roots,
            context="active workspace root",
        )
        self.retained_root = _ensure_secure_directory(
            workspaces / "retained",
            create=create_control_roots,
            context="retained workspace root",
        )
        self.trash_root = _ensure_secure_directory(
            workspaces / ".trash",
            create=create_control_roots,
            context="workspace trash root",
        )
        self._runner = _GitRunner(
            data / ".git-control",
            git_executable=git_executable,
            timeout_seconds=git_timeout_seconds,
            create_control_roots=create_control_roots,
        )
        bindings: Dict[str, RepositoryAcquisitionBinding] = {}
        for binding in acquisition_bindings:
            if not isinstance(binding, RepositoryAcquisitionBinding):
                raise TypeError(
                    "acquisition_bindings must contain RepositoryAcquisitionBinding"
                )
            binding_repository_digest = binding.repository.digest()
            if binding_repository_digest in bindings:
                raise ValueError("duplicate repository acquisition binding")
            bindings[binding_repository_digest] = binding
        self._acquisition_bindings = bindings
        if (
            self.allow_remote
            and self.repository_mode is RepositoryMode.ACQUIRE
        ):
            self._runner.require_curlopt_resolve()
        if os.name == "nt" and create_control_roots:
            for controlled_root, controlled_context in (
                (self.cache_root, "repository cache tree"),
                (self.index_root, "repository index tree"),
                (self.staging_root, "repository staging tree"),
                (self.lock_root, "repository lock tree"),
                (self.reservation_root, "repository reservation tree"),
                (self._runner.control_root, "Git control tree"),
                (self.active_root, "active workspace tree"),
                (self.retained_root, "retained workspace tree"),
                (self.trash_root, "workspace trash tree"),
            ):
                _harden_windows_control_tree(
                    controlled_root, controlled_context
                )
        self._active_leases: Dict[str, TrialWorkspace] = {}
        self._retained_leases: Dict[str, TrialWorkspace] = {}
        self._windows_root_handles = _hold_windows_directory_chains(
            (
                self.suite_root,
                self.data_root,
                self.workspace_root,
                self.cache_root,
                self.index_root,
                self.staging_root,
                self.lock_root,
                self.reservation_root,
                self.active_root,
                self.retained_root,
                self.trash_root,
                self._runner.control_root,
                self._runner.tmp_root,
                self._runner.home,
                self._runner.config_home,
                self._runner.hooks,
            )
        )
        self._closed = False
        try:
            # Cache-only consumers may create Trial workspaces, but they must
            # never repair, prune, acquire, or publish repository source data.
            # Prepare-mode startup retains the existing crash recovery policy.
            if self.repository_mode is RepositoryMode.ACQUIRE:
                self._prune_reservations()
                self._prune_staging()
                self._prune_orphan_caches()
                self._prune_trash()
                self._prune_retained()
            self._assert_data_root_budget()
        except BaseException:
            for handle in reversed(self._windows_root_handles):
                _windows_close_handle(handle)
            self._windows_root_handles = []
            self._closed = True
            raise

    def __enter__(self) -> "RepositoryPreparer":
        if self._closed:
            raise RepositoryPreparationError("RepositoryPreparer is closed")
        self._assert_roots()
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> bool:
        try:
            for lease in list(self._active_leases.values()):
                lease.close()
            if self.repository_mode is RepositoryMode.ACQUIRE:
                self._prune_trash()
        finally:
            self._closed = True
            for handle in reversed(self._windows_root_handles):
                _windows_close_handle(handle)
            self._windows_root_handles = []
        return False

    def _assert_roots(self) -> None:
        current = {
            "suite": _filesystem_identity(self.suite_root, "repository Suite root"),
            "data": _filesystem_identity(self.data_root, "repository data root"),
            "workspace": _filesystem_identity(
                self.workspace_root, "repository workspace root"
            ),
        }
        if current != self._root_identities:
            raise RepositorySecurityError("repository root identity changed")

    def _require_open(self) -> None:
        if self._closed:
            raise RepositoryPreparationError("RepositoryPreparer is closed")
        self._assert_roots()

    def _assert_data_root_budget(self) -> None:
        total_bytes, total_nodes = _secure_tree_usage(
            self.data_root,
            maximum_bytes=MAX_DATA_ROOT_BYTES,
            maximum_nodes=MAX_DATA_ROOT_NODES,
            reject_links=False,
            context="repository data root",
        )
        if total_bytes > MAX_DATA_ROOT_BYTES:
            raise RepositoryLimitError(
                "repository data root exceeds its global byte quota"
            )
        if total_nodes > MAX_DATA_ROOT_NODES:
            raise RepositoryLimitError(
                "repository data root exceeds its global node quota"
            )

    def _reservation_paths(self) -> Tuple[Path, ...]:
        try:
            entries = sorted(
                os.scandir(self.reservation_root), key=lambda item: item.name
            )
        except OSError as exc:
            raise RepositorySecurityError(
                "could not inspect repository reservations"
            ) from exc
        paths: List[Path] = []
        for entry in entries:
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise RepositorySecurityError(
                    "could not inspect repository reservation"
                ) from exc
            if re.fullmatch(
                r"\.[0-9a-f]{32}\.json\.[0-9a-f]{32}\.tmp",
                entry.name,
            ):
                if _is_link_or_reparse(info) or not stat.S_ISREG(info.st_mode):
                    raise RepositorySecurityError(
                        "repository reservation temp entry is unsafe"
                    )
                try:
                    os.unlink(entry.path)
                except OSError as exc:
                    raise RepositorySecurityError(
                        "could not remove stale reservation temp entry"
                    ) from exc
                continue
            if (
                _is_link_or_reparse(info)
                or not stat.S_ISREG(info.st_mode)
                or re.fullmatch(r"[0-9a-f]{32}\.json", entry.name) is None
            ):
                raise RepositorySecurityError(
                    "repository reservation root contains an unsafe entry"
                )
            paths.append(Path(entry.path))
        return tuple(paths)

    def _reservation_totals_locked(self) -> Tuple[int, int]:
        reserved_bytes = 0
        reserved_nodes = 0
        for path in self._reservation_paths():
            payload = _load_repository_reservation(path)
            reserved_bytes += int(payload["reserved_bytes"])
            reserved_nodes += int(payload["reserved_nodes"])
        return reserved_bytes, reserved_nodes

    @staticmethod
    def _unlink_reservation(path: Path) -> None:
        _load_repository_reservation(path)
        try:
            os.unlink(path)
        except OSError as exc:
            raise RepositorySecurityError(
                "could not release repository capacity reservation"
            ) from exc
        if os.name != "nt":
            try:
                descriptor = os.open(
                    path.parent,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                )
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            except OSError as exc:
                raise RepositoryPreparationError(
                    "could not persist repository reservation release"
                ) from exc

    def _prune_reservations(self) -> None:
        with _ProcessLock(
            self.data_store_lock_path, self.lock_timeout_seconds
        ):
            candidates = self._reservation_paths()
        for candidate in candidates:
            request_lock_key = candidate.stem
            request_lock_path = self.lock_root / (
                "request-" + request_lock_key + ".lock"
            )
            operation_lock_path = self.lock_root / (
                "operation-" + request_lock_key + ".lock"
            )
            operation_sentinel_path = self.lock_root / (
                "operation-" + request_lock_key + ".lease"
            )
            try:
                with _ProcessLock(
                    request_lock_path, min(self.lock_timeout_seconds, 0.05)
                ):
                    with _ProcessLock(
                        operation_lock_path,
                        min(self.lock_timeout_seconds, 0.05),
                    ):
                        if not _operation_sentinel_is_free(
                            operation_sentinel_path
                        ):
                            continue
                        with _ProcessLock(
                            self.data_store_lock_path,
                            self.lock_timeout_seconds,
                        ):
                            if not os.path.lexists(candidate):
                                continue
                            current = _load_repository_reservation(candidate)
                            if current["request_lock_key"] != request_lock_key:
                                raise RepositoryIntegrityError(
                                    "repository capacity reservation changed"
                                )
                            operation = self.staging_root / (
                                "operation-" + request_lock_key
                            )
                            if os.path.lexists(operation):
                                _remove_tree_safely(
                                    self.staging_root, operation
                                )
                            self._unlink_reservation(candidate)
            except RepositoryPreparationError as exc:
                if (
                    type(exc) is RepositoryPreparationError
                    and str(exc) == "repository cache lock timed out"
                ):
                    continue
                raise

    def _reserve_prepare_capacity(self, request_id: str) -> Path:
        request_lock_key = _request_lock_key(request_id)
        path = self.reservation_root / (request_lock_key + ".json")
        _validate_direct_child(
            self.reservation_root, path, "repository capacity reservation"
        )
        with _ProcessLock(
            self.data_store_lock_path, self.lock_timeout_seconds
        ):
            if os.path.lexists(path):
                self._unlink_reservation(path)
            current_bytes, current_nodes = _secure_tree_usage(
                self.data_root,
                maximum_bytes=MAX_DATA_ROOT_BYTES,
                maximum_nodes=MAX_DATA_ROOT_NODES,
                reject_links=False,
                context="repository data root",
            )
            reserved_bytes, reserved_nodes = self._reservation_totals_locked()
            if (
                current_bytes > MAX_DATA_ROOT_BYTES
                or current_bytes
                + reserved_bytes
                + MAX_PREPARE_RESERVATION_BYTES
                > MAX_DATA_ROOT_BYTES
            ):
                raise RepositoryLimitError(
                    "repository data root lacks global byte quota for prepare"
                )
            if (
                current_nodes > MAX_DATA_ROOT_NODES
                or current_nodes
                + reserved_nodes
                + MAX_PREPARE_RESERVATION_NODES
                > MAX_DATA_ROOT_NODES
            ):
                raise RepositoryLimitError(
                    "repository data root lacks global node quota for prepare"
                )
            _atomic_write_control_file(
                path,
                canonical_json_bytes(_repository_reservation_payload(request_id)),
            )
        return path

    def _release_prepare_capacity(self, path: Path) -> None:
        with _ProcessLock(
            self.data_store_lock_path, self.lock_timeout_seconds
        ):
            if os.path.lexists(path):
                self._unlink_reservation(path)
            self._assert_data_root_budget()

    def _prune_orphan_caches(self) -> None:
        with _ProcessLock(
            self.data_store_lock_path, self.lock_timeout_seconds
        ):
            if self._reservation_paths():
                return
            referenced: Set[str] = set()
            try:
                index_entries = sorted(
                    os.scandir(self.index_root), key=lambda item: item.name
                )
            except OSError as exc:
                raise RepositorySecurityError(
                    "could not inspect repository cache indexes"
                ) from exc
            for entry in index_entries:
                path = Path(entry.path)
                if re.fullmatch(
                    r"\..+\.json\.[0-9a-f]{32}\.tmp", entry.name
                ):
                    try:
                        info = entry.stat(follow_symlinks=False)
                    except OSError as exc:
                        raise RepositorySecurityError(
                            "could not inspect cache-index temp entry"
                        ) from exc
                    if _is_link_or_reparse(info) or not stat.S_ISREG(info.st_mode):
                        raise RepositorySecurityError(
                            "cache-index temp entry is unsafe"
                        )
                    try:
                        os.unlink(path)
                    except OSError as exc:
                        raise RepositorySecurityError(
                            "could not remove stale cache-index temp entry"
                        ) from exc
                    continue
                index = _load_cache_index(path)
                referenced.add(str(index["cache_id"]))
            try:
                cache_entries = sorted(
                    os.scandir(self.cache_root), key=lambda item: item.name
                )
            except OSError as exc:
                raise RepositorySecurityError(
                    "could not inspect repository content cache"
                ) from exc
            for entry in cache_entries:
                try:
                    info = entry.stat(follow_symlinks=False)
                except OSError as exc:
                    raise RepositorySecurityError(
                        "could not inspect repository cache entry"
                    ) from exc
                if _is_link_or_reparse(info) or not stat.S_ISDIR(info.st_mode):
                    raise RepositorySecurityError(
                        "repository cache contains an unsafe entry"
                    )
                _identifier(entry.name, "repository cache entry ID")
                if entry.name not in referenced:
                    _remove_tree_safely(self.cache_root, Path(entry.path))

    @staticmethod
    def _validate_declared_commit(
        runner: _GitRunner, git_dir: Path, revision: str
    ) -> None:
        try:
            result = runner.run(
                ["--git-dir", str(git_dir), "cat-file", "-t", revision],
                stdout_limit=64,
            )
        except _GitCommandFailure as exc:
            raise RepositoryIntegrityError(
                "declared revision is missing or unreadable"
            ) from exc
        if result.stdout.strip() != b"commit":
            raise RepositoryIntegrityError("declared revision is not a commit object")

    def _binding_for(
        self, descriptor: Repository
    ) -> Optional[RepositoryAcquisitionBinding]:
        if descriptor.url is None:
            return None
        try:
            _validate_https_url(descriptor.url)
        except SchemaError as exc:
            raise RepositorySecurityError(
                "remote repository URL violates the HTTPS credential-free policy"
            ) from exc
        if not self.allow_remote:
            raise RepositoryPreparationError(
                "remote repository prepare is not authorized"
            )
        binding = self._acquisition_bindings.get(descriptor.digest())
        if binding is None:
            raise RepositoryPreparationError(
                "remote repository acquisition attestation is required"
            )
        if binding.repository != descriptor:
            raise RepositoryIntegrityError(
                "remote acquisition binding does not match Repository"
            )
        self._runner.require_curlopt_resolve()
        return binding

    def _cache_binding_for(
        self, descriptor: Repository
    ) -> Optional[RepositoryAcquisitionBinding]:
        """Resolve configured provenance without authorizing source acquisition."""

        if descriptor.url is None:
            return None
        try:
            _validate_https_url(descriptor.url)
        except SchemaError as exc:
            raise RepositorySecurityError(
                "cached remote repository URL violates the HTTPS credential-free policy"
            ) from exc
        binding = self._acquisition_bindings.get(descriptor.digest())
        if binding is not None and binding.repository != descriptor:
            raise RepositoryIntegrityError(
                "cached remote acquisition binding does not match Repository"
            )
        return binding

    def _scan_cache_indexes(
        self,
        *,
        repository_descriptor_digest: str,
    ) -> Tuple[Dict[str, Any], ...]:
        """Read matching committed indexes without repairing the index tree."""

        try:
            entries = sorted(
                os.scandir(self.index_root), key=lambda item: item.name
            )
        except OSError as exc:
            raise RepositorySecurityError(
                "could not inspect repository cache indexes"
            ) from exc
        matches: List[Dict[str, Any]] = []
        for entry in entries:
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise RepositorySecurityError(
                    "could not inspect repository cache index"
                ) from exc
            if re.fullmatch(
                r"\..+\.json\.[0-9a-f]{32}\.tmp", entry.name
            ):
                if _is_link_or_reparse(info) or not stat.S_ISREG(info.st_mode):
                    raise RepositorySecurityError(
                        "repository cache-index temp entry is unsafe"
                    )
                # A cache-only process never adopts or removes an interrupted
                # writer's temporary control file.
                continue
            if (
                _is_link_or_reparse(info)
                or not stat.S_ISREG(info.st_mode)
                or not entry.name.endswith(".json")
            ):
                raise RepositorySecurityError(
                    "repository cache index tree contains an unsafe entry"
                )
            index = _load_cache_index(Path(entry.path))
            if (
                index["repository_descriptor_digest"]
                == repository_descriptor_digest
                and index["git_version"] == self._runner.version
                and index["git_executable_sha256"]
                == self._runner.executable_sha256
            ):
                matches.append(index)
        return tuple(matches)

    def _cached_index_for(
        self, descriptor: Repository
    ) -> Optional[Tuple[Dict[str, Any], Optional[str]]]:
        """Locate one exact request index without accessing the source."""

        repository_descriptor_digest = descriptor.digest()
        binding = self._cache_binding_for(descriptor)
        binding_digest = None if binding is None else binding.digest()
        if descriptor.url is None or binding is not None:
            request_id = _request_id(
                repository_descriptor_digest,
                binding_digest,
                self._runner.version,
                self._runner.executable_sha256,
            )
            index_path = self.index_root / (request_id + ".json")
            if not os.path.lexists(index_path):
                return None
            return _load_cache_index(index_path), binding_digest

        # A post-prepare process may intentionally receive no remote
        # acquisition authority.  It may reuse a single already-attested
        # binding, but ambiguity is never resolved by guessing.
        candidates = self._scan_cache_indexes(
            repository_descriptor_digest=repository_descriptor_digest,
        )
        if not candidates:
            return None
        if len(candidates) != 1:
            raise RepositoryIntegrityError(
                "cached repository acquisition binding is ambiguous"
            )
        candidate = candidates[0]
        candidate_binding_digest = candidate["acquisition_binding_digest"]
        if candidate_binding_digest is None:
            raise RepositoryIntegrityError(
                "cached remote repository lacks acquisition provenance"
            )
        return candidate, candidate_binding_digest

    def _load_cached_descriptor(
        self, descriptor: Repository
    ) -> Optional[Tuple[PreparedRepository, Dict[str, Any]]]:
        located = self._cached_index_for(descriptor)
        if located is None:
            return None
        index, acquisition_binding_digest = located
        prepared = self._load_cached_from_index(
            index,
            expected_request_id=index["request_id"],
            repository=descriptor,
            acquisition_binding_digest=acquisition_binding_digest,
        )
        return prepared, index

    def check_cached(self, descriptor: Repository) -> CacheCheck:
        """Validate cache availability without acquiring or publishing source data."""

        self._require_open()
        if not isinstance(descriptor, Repository):
            raise TypeError("descriptor must be the canonical Repository")
        repository_descriptor_digest = descriptor.digest()
        loaded = self._load_cached_descriptor(descriptor)
        if loaded is None:
            return CacheCheck(
                repository_descriptor_digest=repository_descriptor_digest,
                status=RepositoryCacheStatus.MISSING,
            )
        prepared, index = loaded
        return CacheCheck(
            repository_descriptor_digest=repository_descriptor_digest,
            status=RepositoryCacheStatus.AVAILABLE,
            request_id=index["request_id"],
            cache_id=prepared.cache_id,
            prepared_repository_id=prepared.manifest.prepared_repository_id,
            manifest_digest=index["manifest_digest"],
        )

    def require_cached(self, descriptor: Repository) -> PreparedRepository:
        """Return a fully verified cached repository or fail without acquisition."""

        self._require_open()
        if not isinstance(descriptor, Repository):
            raise TypeError("descriptor must be the canonical Repository")
        loaded = self._load_cached_descriptor(descriptor)
        if loaded is None:
            raise RepositoryPreparationError(
                "repository cache is not prepared; run prepare first"
            )
        return loaded[0]

    def _load_cached_bundle_from_index(
        self,
        index: Mapping[str, Any],
        *,
        expected_request_id: str,
        repository: Repository,
        acquisition_binding_digest: Optional[str],
    ) -> Tuple[PreparedRepository, _RepositoryClosure]:
        repository_descriptor_digest = repository.digest()
        if (
            index["request_id"] != expected_request_id
            or index["repository_descriptor_digest"]
            != repository_descriptor_digest
            or index["acquisition_binding_digest"] != acquisition_binding_digest
            or index["git_version"] != self._runner.version
            or index["git_executable_sha256"]
            != self._runner.executable_sha256
        ):
            raise RepositoryIntegrityError("repository cache index binding drifted")
        cache_id = index["cache_id"]
        entry = self.cache_root / cache_id
        _validate_direct_child(self.cache_root, entry, "repository cache entry")
        _assert_secure_directory(entry, "repository cache entry")
        manifest_path = entry / "manifest.json"
        manifest_bytes = _read_regular_file(
            manifest_path,
            maximum=MAX_REPOSITORY_MANIFEST_BYTES,
            context="prepared repository manifest",
        )
        if _digest_bytes(manifest_bytes) != index["manifest_digest"]:
            raise RepositoryIntegrityError("prepared repository manifest was modified")
        manifest = PreparedRepositoryManifest.from_json(manifest_bytes)
        if (
            manifest.prepared_repository_id != index["prepared_repository_id"]
            or stable_id("repository-cache", manifest.prepared_repository_id)
            != cache_id
        ):
            raise RepositoryIntegrityError("prepared repository cache identity drifted")
        repository_path = entry / "repository.git"
        closure, cache_bytes = _read_loose_repository(
            repository_path,
            object_format=manifest.object_format,
            base_revision=manifest.base_revision,
            head_revision=manifest.head_revision,
        )
        if (
            closure.source_digest != manifest.source_digest
            or closure.base_source_digest != manifest.base_source_digest
            or closure.head_source_digest != manifest.head_source_digest
            or closure.base_tree != manifest.base_tree
            or closure.head_tree != manifest.head_tree
            or closure.object_count != manifest.budget_policy["actual_objects"]
            or closure.blob_count != manifest.budget_policy["actual_blobs"]
            or closure.raw_object_bytes
            != manifest.budget_policy["actual_raw_object_bytes"]
            or closure.materialized_bytes
            != manifest.budget_policy["actual_materialized_bytes"]
            or len(closure.materialized_files)
            != manifest.budget_policy["actual_materialized_files"]
        ):
            raise RepositoryIntegrityError("prepared repository cache content drifted")
        if cache_bytes > MAX_CACHE_BYTES:
            raise RepositoryLimitError(
                "prepared repository cache exceeds its physical byte budget"
            )
        return (
            PreparedRepository(
                manifest=manifest,
                cache_path=repository_path,
                repository=repository,
                acquisition_binding_digest=acquisition_binding_digest,
                git_version=self._runner.version,
                git_executable_sha256=self._runner.executable_sha256,
            ),
            closure,
        )

    def _load_cached_from_index(
        self,
        index: Mapping[str, Any],
        *,
        expected_request_id: str,
        repository: Repository,
        acquisition_binding_digest: Optional[str],
    ) -> PreparedRepository:
        prepared, _closure = self._load_cached_bundle_from_index(
            index,
            expected_request_id=expected_request_id,
            repository=repository,
            acquisition_binding_digest=acquisition_binding_digest,
        )
        return prepared

    def _acquire_closure(
        self,
        descriptor: Repository,
        binding: Optional[RepositoryAcquisitionBinding],
        operation_root: Path,
    ) -> _RepositoryClosure:
        object_format = "sha1" if len(descriptor.base_revision) == 40 else "sha256"
        remote_host: Optional[str] = None
        remote_port: Optional[int] = None
        resolved_addresses: Tuple[str, ...] = ()
        if descriptor.source is RepositorySource.FIXTURE:
            assert descriptor.path is not None
            fixture_root = operation_root / "fixture-source"
            _copy_relative_directory_snapshot(
                self.suite_root,
                descriptor.path,
                fixture_root,
                expected_root_identity=self._root_identities["suite"],
                maximum_bytes=MAX_MATERIALIZED_BYTES,
                maximum_nodes=MAX_MATERIALIZED_FILES * 2 + MAX_PATH_DEPTH,
                context="fixture repository source",
            )
            authored = operation_root / "fixture.git"
            built = FixtureRepositoryBuilder(
                object_format=object_format
            )._build_from_snapshot(fixture_root, authored)
            if (
                built.base_revision != descriptor.base_revision
                or built.head_revision != descriptor.head_revision
            ):
                raise RepositoryIntegrityError(
                    "fixture revision binding does not match deterministic commits"
                )
            closure, _cache_bytes = _read_loose_repository(
                authored,
                object_format=object_format,
                base_revision=descriptor.base_revision,
                head_revision=descriptor.head_revision,
            )
            return closure

        if descriptor.path is not None:
            first_component = unicodedata.normalize(
                "NFC", descriptor.path.split("/", 1)[0]
            ).casefold()
            internal_names = {
                unicodedata.normalize("NFC", name).casefold()
                for name in self._INTERNAL_DATA_NAMES
            }
            if first_component in internal_names:
                raise RepositorySecurityError(
                    "local repository source may not enter Harness control data"
                )
            source = operation_root / "local-source.git"
            _copy_local_git_snapshot(
                self.data_root,
                descriptor.path,
                source,
                expected_root_identity=self._root_identities["data"],
            )
            git_dir = _locate_local_git_source(self._runner, source, object_format)
            self._validate_declared_commit(
                self._runner, git_dir, descriptor.base_revision
            )
            self._validate_declared_commit(
                self._runner, git_dir, descriptor.head_revision
            )
            locator = str(git_dir)
            remote = False
        else:
            if binding is None or descriptor.url is None:
                raise RepositoryPreparationError(
                    "remote repository acquisition binding is missing"
                )
            locator = descriptor.url
            remote = True
            remote_host = binding.allowed_host
            remote_port = binding.allowed_port
            resolved_addresses = _resolve_remote_endpoint(
                descriptor.url,
                allowed_host=remote_host,
                allowed_port=remote_port,
            )

        quarantine = operation_root / "quarantine.git"
        try:
            _fetch_quarantine(
                self._runner,
                quarantine,
                locator=locator,
                remote=remote,
                base_revision=descriptor.base_revision,
                head_revision=descriptor.head_revision,
                remote_host=remote_host,
                remote_port=remote_port,
                resolved_addresses=resolved_addresses,
            )
        except RepositoryPreparationError as exc:
            if not remote:
                raise RepositoryIntegrityError(
                    "local repository revisions could not be acquired"
                ) from exc
            raise
        self._validate_declared_commit(
            self._runner, quarantine, descriptor.base_revision
        )
        self._validate_declared_commit(
            self._runner, quarantine, descriptor.head_revision
        )
        closure = _extract_quarantine_closure(
            self._runner,
            quarantine,
            base_revision=descriptor.base_revision,
            head_revision=descriptor.head_revision,
        )
        if binding is not None and closure.source_digest != binding.expected_source_digest:
            raise RepositoryIntegrityError(
                "remote repository logical source digest does not match attestation"
            )
        return closure

    def _publish_cache(
        self,
        descriptor: Repository,
        binding: Optional[RepositoryAcquisitionBinding],
        closure: _RepositoryClosure,
        operation_root: Path,
    ) -> Tuple[PreparedRepository, Dict[str, Any]]:
        repository_descriptor_digest = descriptor.digest()
        binding_digest = None if binding is None else binding.digest()
        entry_staging = operation_root / "entry"
        _mkdir_exclusive(entry_staging)
        repository_path = entry_staging / "repository.git"
        _write_loose_repository(repository_path, closure, bare=True)
        budget = _budget_policy(
            object_count=closure.object_count,
            blob_count=closure.blob_count,
            raw_object_bytes=closure.raw_object_bytes,
            materialized_files=len(closure.materialized_files),
            materialized_bytes=closure.materialized_bytes,
        )
        manifest = PreparedRepositoryManifest.create(
            source_digest=closure.source_digest,
            base_source_digest=closure.base_source_digest,
            head_source_digest=closure.head_source_digest,
            object_format=closure.object_format,
            base_revision=closure.base_revision,
            head_revision=closure.head_revision,
            base_tree=closure.base_tree,
            head_tree=closure.head_tree,
            budget_policy=budget,
        )
        manifest_bytes = canonical_json_bytes(manifest)
        _write_regular_file_exclusive(
            entry_staging / "manifest.json", manifest_bytes, fsync=True
        )
        cache_id = stable_id("repository-cache", manifest.prepared_repository_id)
        target = self.cache_root / cache_id
        _validate_direct_child(self.cache_root, target, "repository cache publish")
        content_lock_path = self.lock_root / ("content-" + cache_id + ".lock")
        with _ProcessLock(
            content_lock_path, self.lock_timeout_seconds
        ) as content_lock:
            if os.path.lexists(target):
                _remove_tree_safely(operation_root, entry_staging)
                persisted_bytes = _read_regular_file(
                    target / "manifest.json",
                    maximum=MAX_REPOSITORY_MANIFEST_BYTES,
                    context="existing content-addressed repository manifest",
                )
                persisted = PreparedRepositoryManifest.from_json(persisted_bytes)
                if persisted != manifest:
                    raise RepositoryIntegrityError(
                        "content-addressed cache ID conflicts with existing content"
                    )
                manifest = persisted
                manifest_bytes = persisted_bytes
            else:
                content_lock.assert_current()
                try:
                    _rename_directory_no_replace(entry_staging, target)
                except OSError as exc:
                    raise RepositoryPreparationError(
                        "could not atomically publish repository cache"
                    ) from exc
        prepared = PreparedRepository(
            manifest=manifest,
            cache_path=target / "repository.git",
            repository=descriptor,
            acquisition_binding_digest=binding_digest,
            git_version=self._runner.version,
            git_executable_sha256=self._runner.executable_sha256,
        )
        index = _cache_index_payload(
            request_id="",  # Filled by the caller under the request lock.
            repository_descriptor_digest=repository_descriptor_digest,
            acquisition_binding_digest=binding_digest,
            git_version=self._runner.version,
            git_executable_sha256=self._runner.executable_sha256,
            cache_id=cache_id,
            prepared_repository_id=manifest.prepared_repository_id,
            manifest_digest=_digest_bytes(manifest_bytes),
        )
        return prepared, index

    def _verified_prepared_bundle(
        self, prepared: PreparedRepository
    ) -> Tuple[PreparedRepository, _RepositoryClosure]:
        self._require_open()
        if not isinstance(prepared, PreparedRepository):
            raise TypeError("prepared must be a PreparedRepository")
        expected_entry = self.cache_root / prepared.cache_id
        expected_cache_path = expected_entry / "repository.git"
        if _normalized_path(prepared.cache_path) != _normalized_path(
            expected_cache_path
        ):
            raise RepositorySecurityError(
                "PreparedRepository handle is outside its canonical cache entry"
            )
        _validate_direct_child(
            self.cache_root, expected_entry, "prepared repository cache handle"
        )
        _assert_secure_directory(expected_entry, "prepared repository cache entry")
        persisted_manifest = PreparedRepositoryManifest.from_json(
            _read_regular_file(
                expected_entry / "manifest.json",
                maximum=MAX_REPOSITORY_MANIFEST_BYTES,
                context="prepared repository handle manifest",
            )
        )
        if persisted_manifest != prepared.manifest:
            raise RepositoryIntegrityError(
                "PreparedRepository handle does not match its persisted manifest"
            )
        request_id = _request_id(
            prepared.repository.digest(),
            prepared.acquisition_binding_digest,
            prepared.git_version,
            prepared.git_executable_sha256,
        )
        index_path = self.index_root / (request_id + ".json")
        if not os.path.lexists(index_path):
            raise RepositoryIntegrityError(
                "PreparedRepository has no verified request-index binding"
            )
        verified_handle, closure = self._load_cached_bundle_from_index(
            _load_cache_index(index_path),
            expected_request_id=request_id,
            repository=prepared.repository,
            acquisition_binding_digest=prepared.acquisition_binding_digest,
        )
        if verified_handle != prepared:
            raise RepositoryIntegrityError(
                "PreparedRepository runtime provenance is not verified"
            )
        return verified_handle, closure

    def open_replay(
        self, prepared: PreparedRepository
    ) -> PreparedRepositoryReplay:
        """Open a verified, read-only base/head Evidence replay handle."""

        verified, closure = self._verified_prepared_bundle(prepared)

        def verify_cache() -> None:
            current, _current_closure = self._verified_prepared_bundle(verified)
            if current != verified:
                raise RepositoryIntegrityError(
                    "prepared repository replay binding changed"
                )

        return PreparedRepositoryReplay._from_verified(
            verified,
            closure,
            self._runner,
            self._require_open,
            verify_cache,
        )

    def open_replay_for(
        self, descriptor: Repository
    ) -> PreparedRepositoryReplay:
        """Open read-only base/head replay from an existing verified cache only."""

        return self.open_replay(self.require_cached(descriptor))

    def prepare(self, descriptor: Repository) -> PreparedRepository:
        self._require_open()
        if not isinstance(descriptor, Repository):
            raise TypeError("descriptor must be the canonical Repository")
        if self.repository_mode is RepositoryMode.CACHE_ONLY:
            return self.require_cached(descriptor)
        self._assert_data_root_budget()
        binding = self._binding_for(descriptor)
        repository_descriptor_digest = descriptor.digest()
        binding_digest = None if binding is None else binding.digest()
        request_id = _request_id(
            repository_descriptor_digest,
            binding_digest,
            self._runner.version,
            self._runner.executable_sha256,
        )
        index_path = self.index_root / (request_id + ".json")
        request_lock_key = _request_lock_key(request_id)
        lock_path = self.lock_root / ("request-" + request_lock_key + ".lock")
        with _ProcessLock(lock_path, self.lock_timeout_seconds) as request_lock:
            if os.path.lexists(index_path):
                index = _load_cache_index(index_path)
                return self._load_cached_from_index(
                    index,
                    expected_request_id=request_id,
                    repository=descriptor,
                    acquisition_binding_digest=binding_digest,
                )
            with _OperationLease(
                self.lock_root,
                request_lock_key,
                self.lock_timeout_seconds,
            ) as operation_lease, self._runner.operation_lease(operation_lease):
                reservation_path = self._reserve_prepare_capacity(request_id)
                operation_root = self.staging_root / (
                    "operation-" + request_lock_key
                )
                try:
                    _validate_direct_child(
                        self.staging_root,
                        operation_root,
                        "repository operation staging",
                    )
                    if os.path.lexists(operation_root):
                        _remove_tree_safely(self.staging_root, operation_root)
                    _mkdir_exclusive(operation_root)
                    closure = self._acquire_closure(
                        descriptor, binding, operation_root
                    )
                    prepared, index = self._publish_cache(
                        descriptor, binding, closure, operation_root
                    )
                    index["request_id"] = request_id
                    request_lock.assert_current()
                    operation_lease.process_lock.assert_current()
                    _atomic_write_control_file(
                        index_path, canonical_json_bytes(index)
                    )
                    loaded = self._load_cached_from_index(
                        index,
                        expected_request_id=request_id,
                        repository=descriptor,
                        acquisition_binding_digest=binding_digest,
                    )
                    if loaded.manifest != prepared.manifest:
                        raise RepositoryIntegrityError(
                            "published repository cache did not replay identically"
                        )
                    return loaded
                finally:
                    try:
                        if os.path.lexists(operation_root):
                            _remove_tree_safely(
                                self.staging_root, operation_root
                            )
                    finally:
                        self._release_prepare_capacity(reservation_path)

    @staticmethod
    def _materialize_tree(workspace: Path, closure: _RepositoryClosure) -> None:
        for relative, mode, data, _blob_oid in closure.materialized_files:
            parts = relative.split("/")
            parent = workspace
            for component in parts[:-1]:
                parent = _ensure_child_directory(parent, component)
            target = parent / parts[-1]
            _write_regular_file_exclusive(
                target,
                data,
                mode=0o755 if mode == 0o100755 else 0o644,
            )
            if os.name != "nt":
                os.chmod(target, 0o755 if mode == 0o100755 else 0o644)

    def trial_workspace(
        self,
        prepared: PreparedRepository,
        *,
        trial_manifest: TrialManifest,
        suite_case: SuiteCase,
        eval_input: EvalInput,
        attempt: int,
    ) -> TrialWorkspace:
        verified_handle, closure = self._verified_prepared_bundle(prepared)
        prepared = verified_handle
        manifest = WorkspaceManifest.create(
            prepared,
            trial_manifest=trial_manifest,
            suite_case=suite_case,
            eval_input=eval_input,
            attempt=attempt,
        )
        workspace_id = stable_id("workspace", manifest.workspace_binding_id)
        target = self.active_root / workspace_id
        _validate_direct_child(self.active_root, target, "Trial workspace")
        if workspace_id in self._active_leases or os.path.lexists(target):
            raise RepositoryPreparationError("Trial workspace already exists")

        staging = self.active_root / ("staging-" + uuid.uuid4().hex)
        _validate_direct_child(self.active_root, staging, "Trial staging workspace")
        _mkdir_exclusive(staging)
        try:
            git_dir = staging / ".git"
            _write_loose_repository(git_dir, closure, bare=False)
            self._materialize_tree(staging, closure)
            try:
                self._runner.run(
                    [
                        "--git-dir",
                        str(git_dir),
                        "--work-tree",
                        str(staging),
                        "read-tree",
                        closure.head_revision,
                    ],
                    stdout_limit=4096,
                )
                status = self._runner.run(
                    [
                        "--git-dir",
                        str(git_dir),
                        "--work-tree",
                        str(staging),
                        "status",
                        "--porcelain=v1",
                        "--untracked-files=all",
                    ],
                    stdout_limit=MAX_GIT_STDOUT_BYTES,
                )
            except _GitCommandFailure as exc:
                raise RepositoryIntegrityError(
                    "could not initialize isolated Trial Git index"
                ) from exc
            if status.stdout:
                raise RepositoryIntegrityError(
                    "manually materialized Trial workspace is not initially clean"
                )
            try:
                _rename_directory_no_replace(staging, target)
            except OSError as exc:
                raise RepositoryPreparationError(
                    "could not atomically publish Trial workspace"
                ) from exc
        finally:
            if os.path.lexists(staging):
                _remove_tree_safely(self.active_root, staging)
        lease = TrialWorkspace(
            _path=target,
            manifest=manifest,
            _preparer=self,
            _directory_identity=_filesystem_identity(
                target, "published Trial workspace"
            ),
        )
        self._active_leases[workspace_id] = lease
        return lease

    def _assert_workspace_lease(self, lease: TrialWorkspace) -> None:
        self._require_open()
        workspace_id = stable_id("workspace", lease.manifest.workspace_binding_id)
        if self._active_leases.get(workspace_id) is not lease:
            raise RepositorySecurityError("workspace lease is not active")
        expected = self.active_root / workspace_id
        _validate_direct_child(self.active_root, expected, "active Trial workspace")
        if _filesystem_identity(expected, "active Trial workspace") != lease._directory_identity:
            raise RepositorySecurityError("active Trial workspace identity changed")

    def _move_to_trash(self, source_root: Path, source: Path, name: str) -> Path:
        _validate_direct_child(source_root, source, "workspace trash source")
        target = self.trash_root / name
        _validate_direct_child(self.trash_root, target, "workspace trash")
        if os.path.lexists(target):
            raise RepositorySecurityError("workspace trash target already exists")
        try:
            _rename_directory_no_replace(source, target)
        except OSError as exc:
            raise RepositorySecurityError(
                "could not move workspace into controlled trash"
            ) from exc
        return target

    def _release_workspace(self, lease: TrialWorkspace) -> None:
        workspace_id = stable_id("workspace", lease.manifest.workspace_binding_id)
        if self._active_leases.get(workspace_id) is not lease:
            lease._closed = True
            return
        self._active_leases.pop(workspace_id, None)
        source = self.active_root / workspace_id
        retain = (
            self.retention_policy is WorkspaceRetentionPolicy.RETAIN_ON_FAILURE
            and lease._terminal_status
            in {
                TrialStatus.FAILED,
                TrialStatus.BLOCKED,
                TrialStatus.INVALID_OUTPUT,
                TrialStatus.INCOMPLETE,
            }
        )
        try:
            _validate_direct_child(self.active_root, source, "active Trial workspace")
            if _filesystem_identity(source, "active Trial workspace") != lease._directory_identity:
                raise RepositorySecurityError("active Trial workspace identity changed")
            if retain:
                target = self.retained_root / workspace_id
                _validate_direct_child(
                    self.retained_root, target, "retained Trial workspace"
                )
                if os.path.lexists(target):
                    raise RepositorySecurityError(
                        "retained workspace target already exists"
                    )
                _rename_directory_no_replace(source, target)
                os.utime(target, None)
                lease._path = target
                lease._directory_identity = _filesystem_identity(
                    target, "retained Trial workspace"
                )
                lease._retained = True
                self._retained_leases[workspace_id] = lease
                self._prune_retained()
            else:
                trashed = self._move_to_trash(
                    self.active_root, source, workspace_id
                )
                lease._path = trashed
                lease._directory_identity = _filesystem_identity(
                    trashed, "trashed Trial workspace"
                )
                _remove_tree_safely(self.trash_root, trashed)
        except Exception as exc:
            lease.cleanup_diagnostic = WorkspaceDiagnostic(
                code="cleanup_failed",
                message=str(exc)[:4096] or "workspace cleanup failed",
            )
        finally:
            lease._closed = True

    def _evict_retained(self, path: Path) -> None:
        workspace_id = path.name
        try:
            trashed = self._move_to_trash(
                self.retained_root, path, workspace_id
            )
            _remove_tree_safely(self.trash_root, trashed)
        finally:
            lease = self._retained_leases.pop(workspace_id, None)
            if lease is not None:
                lease._retained = False

    def _prune_staging(self) -> None:
        """Remove crash leftovers only while their request lock is unheld."""

        try:
            entries = list(os.scandir(self.staging_root))
        except OSError as exc:
            raise RepositorySecurityError(
                "could not inspect repository staging root"
            ) from exc
        for entry in entries:
            match = _OPERATION_DIRECTORY_RE.fullmatch(entry.name)
            if match is None:
                raise RepositorySecurityError(
                    "repository staging contains an unrecognized entry"
                )
            try:
                info = entry.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise RepositorySecurityError(
                    "could not inspect repository staging entry"
                ) from exc
            if _is_link_or_reparse(info) or not stat.S_ISDIR(info.st_mode):
                raise RepositorySecurityError(
                    "repository staging contains an unsafe entry"
                )
            request_lock_key = match.group(1)
            lock_path = self.lock_root / (
                "request-" + request_lock_key + ".lock"
            )
            operation_lock_path = self.lock_root / (
                "operation-" + request_lock_key + ".lock"
            )
            operation_sentinel_path = self.lock_root / (
                "operation-" + request_lock_key + ".lease"
            )
            try:
                with _ProcessLock(
                    lock_path, min(self.lock_timeout_seconds, 0.05)
                ):
                    with _ProcessLock(
                        operation_lock_path,
                        min(self.lock_timeout_seconds, 0.05),
                    ):
                        if not _operation_sentinel_is_free(
                            operation_sentinel_path
                        ):
                            continue
                        _remove_tree_safely(
                            self.staging_root,
                            Path(entry.path),
                        )
            except RepositoryPreparationError as exc:
                if (
                    type(exc) is RepositoryPreparationError
                    and str(exc) == "repository cache lock timed out"
                ):
                    continue
                raise

    def _prune_trash(self) -> None:
        try:
            entries = list(os.scandir(self.trash_root))
        except OSError as exc:
            raise RepositorySecurityError("could not inspect workspace trash") from exc
        for entry in entries:
            path = Path(entry.path)
            try:
                _validate_direct_child(
                    self.trash_root, path, "workspace trash garbage collection"
                )
                _remove_tree_safely(self.trash_root, path)
            except RepositoryPreparationError:
                # A Windows sharing violation can outlive the Trial process.
                # The path remains under the controlled trash root and is
                # retried on the next Preparer lifecycle.
                continue

    def _prune_retained(self) -> None:
        now = time.time()
        records: List[Tuple[float, str, Path, int]] = []
        try:
            entries = list(os.scandir(self.retained_root))
        except OSError as exc:
            raise RepositorySecurityError(
                "could not inspect retained workspaces"
            ) from exc
        for entry in entries:
            path = Path(entry.path)
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise RepositorySecurityError(
                    "could not inspect retained workspace"
                ) from exc
            if _is_link_or_reparse(info) or not stat.S_ISDIR(info.st_mode):
                self._evict_retained(path)
                continue
            size = _secure_tree_size(path, self.max_retained_bytes + 1)
            records.append((float(info.st_mtime), entry.name, path, size))
        records.sort(key=lambda item: (item[0], item[1]))
        kept: List[Tuple[float, str, Path, int]] = []
        for record in records:
            if now - record[0] > self.retention_ttl_seconds:
                self._evict_retained(record[2])
            else:
                kept.append(record)
        total = sum(item[3] for item in kept)
        while (
            len(kept) > self.max_retained_workspaces
            or total > self.max_retained_bytes
        ):
            oldest = kept.pop(0)
            total -= oldest[3]
            self._evict_retained(oldest[2])


__all__ = [
    "BuiltFixtureRepository",
    "CacheCheck",
    "FixtureRepositoryBuilder",
    "PreparedRepository",
    "PreparedRepositoryManifest",
    "PreparedRepositoryReplay",
    "RepositoryAcquisitionBinding",
    "RepositoryCacheStatus",
    "RepositoryIntegrityError",
    "RepositoryLimitError",
    "RepositoryMode",
    "RepositoryPolicyError",
    "RepositoryPreparer",
    "RepositoryPreparationError",
    "RepositorySecurityError",
    "TrialWorkspace",
    "WorkspaceDiagnostic",
    "WorkspaceManifest",
    "WorkspaceRetentionPolicy",
    "canonical_repository_path",
    "repository_from_eval_input",
]
