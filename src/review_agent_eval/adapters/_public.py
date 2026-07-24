"""Strict, offline preparation boundary for public evaluation datasets.

Dataset-specific adapters parse upstream records, but they all use this module
to bind exact source bytes, publish canonical ``EvalCase``/``SuiteManifest``
artifacts once, and retain an auditable record-level provenance receipt.

Nothing in this module performs network I/O.  Acquisition belongs to an
explicit prepare step; Trials consume only the verified Suite and repository
caches produced by that step.
"""

from __future__ import annotations

import errno
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
import stat
import tempfile
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Iterable, Mapping, Optional, Tuple
import unicodedata

from ..artifacts import (
    ArtifactError,
    _normalized_filesystem_path,
    _path_is_within,
    _windows_close_handle,
    _windows_open_directory_handle,
    _windows_raw_handle_attributes,
    _windows_raw_handle_path,
)
from ..cases import (
    MAX_SUITE_CASES,
    MAX_SUITE_MANIFEST_BYTES,
    MAX_SUITE_TOTAL_CASE_BYTES,
    PUBLIC_SUITE_PREPARATION_BINDING_SCHEMA_VERSION,
    SUITE_MANIFEST_SCHEMA_VERSION,
    CaseDimension,
    CaseSplit,
    PublicSuitePreparationBindingV2,
    SuiteCase,
    SuiteKind,
    SuiteManifest,
    SuiteSource,
    WireContractV2,
    _portable_case_path,
    validate_case_for_manifest,
)
from ..datasets import (
    CaseBank,
    _coerce_suite_root,
    _read_relative_regular_file,
    _secure_regular_file,
    _windows_descriptor_path,
)
from ..models import (
    MAX_COUNTER,
    MAX_EVAL_CASE_BYTES,
    MAX_IDENTIFIER_CHARS,
    MAX_URL_CHARS,
    EvalCase,
    FrozenContextReviewTarget,
    ReviewTargetKind,
    SchemaError,
    _JsonModel,
    _array,
    _check_model_size,
    _digest,
    _exact_fields,
    _identifier,
    _integer,
    _object,
    _optional_string,
    _strict_json_loads,
    _string,
    canonical_json,
    canonical_json_bytes,
    canonical_sha256,
)

if TYPE_CHECKING:
    from .swe_prbench import PreparedFrozenContextBundle


PUBLIC_SOURCE_MANIFEST_SCHEMA_VERSION = "public_dataset_source_v1"
PUBLIC_FILTER_MANIFEST_SCHEMA_VERSION = "public_dataset_filter_v1"
PUBLIC_PREPARATION_RECEIPT_SCHEMA_VERSION = (
    "public_dataset_preparation_receipt_v1"
)
PUBLIC_PREPARATION_PACKET_SCHEMA_VERSION = "public_dataset_preparation_packet_v1"

DEFAULT_PREPARATION_RECEIPT_PATH = "preparation_receipt.json"

MAX_PUBLIC_SOURCE_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_PUBLIC_FILTER_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_PUBLIC_PREPARATION_RECEIPT_BYTES = 256 * 1024 * 1024
MAX_PUBLIC_SOURCE_FILES = 65_536
MAX_PUBLIC_SOURCE_FILE_BYTES = 512 * 1024 * 1024
MAX_PUBLIC_SOURCE_TOTAL_BYTES = 4 * 1024 * 1024 * 1024
MAX_PUBLIC_STATISTICS = 512
MAX_PUBLIC_SELECTORS = 64
MAX_PUBLIC_SELECTOR_VALUES = 65_536
MAX_PUBLIC_RECORD_RECEIPTS = 262_144
MAX_PUBLIC_RECORD_JSON_BYTES = 4 * 1024 * 1024
MAX_PUBLIC_EXTRA_FILES = 262_144
MAX_PUBLIC_EXTRA_FILE_BYTES = 512 * 1024 * 1024
MAX_PUBLIC_EXTRA_TOTAL_BYTES = 4 * 1024 * 1024 * 1024

_REPARSE_POINT_FLAG = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)


class PublicDatasetError(SchemaError):
    """A public source, filter, format, or publication failed closed."""


class PublicSourceIntegrityError(PublicDatasetError):
    """The supplied source bytes do not match their immutable bindings."""


class PublicFormatError(PublicDatasetError):
    """An upstream record does not match the adapter's pinned schema."""


class PublicPreparationError(PublicDatasetError):
    """A canonical public Suite could not be published safely."""


class PublicConflictError(PublicPreparationError):
    """Create-only public output conflicts with an existing owner."""


class PublicPreconditionError(PublicPreparationError):
    """The host environment cannot begin the requested preparation."""


class PublicOperationalError(PublicPreparationError):
    """Publication failed because of an operational platform or I/O error."""


class PublicOptionalDependencyError(PublicPreconditionError):
    """A declared optional public-data format lacks its reader dependency."""


def verify_public_source_manifest_digest(
    manifest: "PublicSourceManifest", expected_source_manifest_digest: str
) -> str:
    """Bind a supplied manifest to a control-plane approved digest."""

    if not isinstance(manifest, PublicSourceManifest):
        raise PublicSourceIntegrityError(
            "source manifest digest verification requires PublicSourceManifest"
        )
    try:
        expected = _digest(
            expected_source_manifest_digest,
            "expected public source manifest digest",
        )
    except SchemaError as exc:
        raise PublicSourceIntegrityError(str(exc)) from exc
    actual = manifest.digest()
    if actual != expected:
        raise PublicSourceIntegrityError(
            "public source manifest digest does not match the expected digest"
        )
    return actual


def verify_public_filter_manifest_digest(
    manifest: "PublicFilterManifest", expected_profile_digest: str
) -> str:
    """Bind a canonical filter/profile manifest to an external trust anchor."""

    if not isinstance(manifest, PublicFilterManifest):
        raise PublicSourceIntegrityError(
            "filter manifest digest verification requires PublicFilterManifest"
        )
    try:
        expected = _digest(
            expected_profile_digest,
            "expected public profile digest",
        )
    except SchemaError as exc:
        raise PublicSourceIntegrityError(str(exc)) from exc
    actual = manifest.digest()
    if actual != expected:
        raise PublicSourceIntegrityError(
            "public filter manifest digest does not match the expected profile digest"
        )
    return actual


def _public_control_file(
    path: os.PathLike[str] | str,
    *,
    maximum_bytes: int,
    context: str,
) -> bytes:
    """Read one bounded control file without following links or reparses."""

    if not isinstance(path, (str, os.PathLike)):
        raise PublicSourceIntegrityError("%s path is invalid" % context)
    lexical = Path(os.path.abspath(os.fspath(path)))
    if not lexical.name:
        raise PublicSourceIntegrityError("%s path must name a file" % context)
    try:
        parent = _coerce_suite_root(lexical.parent)
    except SchemaError as exc:
        raise PublicSourceIntegrityError(
            "%s parent is not a secure directory" % context
        ) from exc
    if os.path.normcase(str(lexical.parent)) != os.path.normcase(str(parent)):
        raise PublicSourceIntegrityError(
            "%s path may not traverse a symlink or reparse point" % context
        )
    return _read_single_link_regular_file(
        parent,
        lexical.name,
        maximum_bytes,
        context,
    )


def read_public_source_manifest(
    path: os.PathLike[str] | str,
    *,
    expected_source_manifest_digest: str,
) -> "PublicSourceManifest":
    """Securely read canonical source-control JSON and verify its trust anchor."""

    raw = _public_control_file(
        path,
        maximum_bytes=MAX_PUBLIC_SOURCE_MANIFEST_BYTES,
        context="public source manifest control file",
    )
    try:
        manifest = PublicSourceManifest.from_json(raw)
    except SchemaError as exc:
        raise PublicSourceIntegrityError(
            "public source manifest control file is malformed"
        ) from exc
    if canonical_json_bytes(manifest.to_dict()) != raw:
        raise PublicSourceIntegrityError(
            "public source manifest control file is not canonical JSON"
        )
    verify_public_source_manifest_digest(
        manifest,
        expected_source_manifest_digest,
    )
    return manifest


def read_public_filter_manifest(
    path: os.PathLike[str] | str,
    *,
    expected_profile_digest: str,
) -> "PublicFilterManifest":
    """Securely read canonical filter JSON and verify the profile trust anchor."""

    raw = _public_control_file(
        path,
        maximum_bytes=MAX_PUBLIC_FILTER_MANIFEST_BYTES,
        context="public filter manifest control file",
    )
    try:
        manifest = PublicFilterManifest.from_json(raw)
    except SchemaError as exc:
        raise PublicSourceIntegrityError(
            "public filter manifest control file is malformed"
        ) from exc
    if canonical_json_bytes(manifest.to_dict()) != raw:
        raise PublicSourceIntegrityError(
            "public filter manifest control file is not canonical JSON"
        )
    verify_public_filter_manifest_digest(manifest, expected_profile_digest)
    return manifest


def _portable_path_key(path: str) -> Tuple[str, ...]:
    """Return the cross-platform identity used for immutable file manifests."""

    return tuple(
        unicodedata.normalize("NFC", component).casefold()
        for component in path.split("/")
    )


def _paths_overlap(first: Tuple[str, ...], second: Tuple[str, ...]) -> bool:
    common = min(len(first), len(second))
    return first[:common] == second[:common]


def _assert_no_portable_path_collisions(
    paths: Iterable[str],
    context: str,
    *,
    reserved: Iterable[str] = (),
) -> None:
    """Reject case/Unicode aliases and file-versus-directory ambiguity."""

    checked = tuple(paths)
    identities: list[Tuple[str, Tuple[str, ...]]] = []
    for path in checked:
        portable = _portable_case_path(path, context)
        identity = _portable_path_key(portable)
        for previous, previous_identity in identities:
            if _paths_overlap(identity, previous_identity):
                raise PublicDatasetError(
                    "%s contains a portable path collision between %r and %r"
                    % (context, previous, portable)
                )
        identities.append((portable, identity))

    reserved_identities = tuple(
        (path, _portable_path_key(_portable_case_path(path, context)))
        for path in reserved
    )
    for path, identity in identities:
        for reserved_path, reserved_identity in reserved_identities:
            if _paths_overlap(identity, reserved_identity):
                raise PublicDatasetError(
                    "%s path %r collides with reserved control-plane path %r"
                    % (context, path, reserved_path)
                )


def _lower_ascii_key(value: Any, context: str) -> str:
    result = _identifier(value, context)
    if not result[0].isascii() or not result[0].isalpha():
        raise PublicDatasetError("%s must start with an ASCII letter" % context)
    if result != result.casefold() or not all(
        character.isascii()
        and (character.islower() or character.isdigit() or character in "_.-")
        for character in result
    ):
        raise PublicDatasetError(
            "%s must be a lowercase ASCII grouping key" % context
        )
    return result


def _https_uri(value: Any, context: str) -> str:
    result = _string(value, context, MAX_URL_CHARS)
    if not result.startswith("https://"):
        raise PublicDatasetError("%s must use https" % context)
    if any(ord(character) < 32 or ord(character) == 127 for character in result):
        raise PublicDatasetError("%s may not contain controls" % context)
    return result


@dataclass(frozen=True)
class PublicSourceFile(_JsonModel):
    role: str
    path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _identifier(self.role, "public source file.role")
        _portable_case_path(self.path, "public source file.path")
        _integer(
            self.size_bytes,
            "public source file.size_bytes",
            minimum=1,
            maximum=MAX_PUBLIC_SOURCE_FILE_BYTES,
        )
        _digest(self.sha256, "public source file.sha256")

    @classmethod
    def from_dict(cls, value: Any) -> "PublicSourceFile":
        payload = _object(value, "public source file")
        _exact_fields(
            payload,
            ("role", "path", "size_bytes", "sha256"),
            "public source file",
        )
        return cls(
            role=_identifier(payload["role"], "public source file.role"),
            path=_portable_case_path(payload["path"], "public source file.path"),
            size_bytes=_integer(
                payload["size_bytes"],
                "public source file.size_bytes",
                minimum=1,
                maximum=MAX_PUBLIC_SOURCE_FILE_BYTES,
            ),
            sha256=_digest(payload["sha256"], "public source file.sha256"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class PublicStatistic(_JsonModel):
    name: str
    value: int

    def __post_init__(self) -> None:
        _lower_ascii_key(self.name, "public statistic.name")
        _integer(
            self.value,
            "public statistic.value",
            minimum=0,
            maximum=MAX_COUNTER,
        )

    @classmethod
    def from_dict(cls, value: Any) -> "PublicStatistic":
        payload = _object(value, "public statistic")
        _exact_fields(payload, ("name", "value"), "public statistic")
        return cls(
            name=_lower_ascii_key(payload["name"], "public statistic.name"),
            value=_integer(
                payload["value"],
                "public statistic.value",
                minimum=0,
                maximum=MAX_COUNTER,
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "value": self.value}


def _ordered_statistics(
    values: Iterable[PublicStatistic], context: str
) -> Tuple[PublicStatistic, ...]:
    result = tuple(values)
    if len(result) > MAX_PUBLIC_STATISTICS:
        raise PublicDatasetError(
            "%s exceeds the item limit of %d"
            % (context, MAX_PUBLIC_STATISTICS)
        )
    if any(not isinstance(item, PublicStatistic) for item in result):
        raise PublicDatasetError("%s must contain PublicStatistic values" % context)
    names = [item.name for item in result]
    if len(names) != len(set(names)):
        raise PublicDatasetError("%s contains duplicate names" % context)
    return tuple(sorted(result, key=lambda item: item.name))


@dataclass(frozen=True)
class PublicSourceManifest(_JsonModel):
    SCHEMA_VERSION: ClassVar[str] = PUBLIC_SOURCE_MANIFEST_SCHEMA_VERSION

    schema_version: str
    dataset_id: str
    dataset_version: str
    source_uri: str
    source_revision: str
    license: str
    files: Tuple[PublicSourceFile, ...]
    expected_statistics: Tuple[PublicStatistic, ...]

    def __post_init__(self) -> None:
        if self.schema_version != self.SCHEMA_VERSION:
            raise PublicDatasetError(
                "public source manifest has an unknown schema_version"
            )
        _identifier(self.dataset_id, "public source manifest.dataset_id")
        _identifier(self.dataset_version, "public source manifest.dataset_version")
        _https_uri(self.source_uri, "public source manifest.source_uri")
        _identifier(self.source_revision, "public source manifest.source_revision")
        _string(
            self.license,
            "public source manifest.license",
            MAX_IDENTIFIER_CHARS,
        )
        if not isinstance(self.files, tuple):
            raise PublicDatasetError("public source manifest.files must be a tuple")
        if not self.files:
            raise PublicDatasetError(
                "public source manifest.files must contain at least one file"
            )
        if len(self.files) > MAX_PUBLIC_SOURCE_FILES:
            raise PublicDatasetError(
                "public source manifest.files exceeds the item limit"
            )
        if any(not isinstance(item, PublicSourceFile) for item in self.files):
            raise PublicDatasetError(
                "public source manifest.files must contain PublicSourceFile values"
            )
        roles = [item.role for item in self.files]
        paths = [item.path for item in self.files]
        if len(roles) != len(set(roles)):
            raise PublicDatasetError(
                "public source manifest.files contains duplicate roles"
            )
        if len(paths) != len(set(paths)):
            raise PublicDatasetError(
                "public source manifest.files contains duplicate paths"
            )
        _assert_no_portable_path_collisions(
            paths, "public source manifest.files"
        )
        total = sum(item.size_bytes for item in self.files)
        if total > MAX_PUBLIC_SOURCE_TOTAL_BYTES:
            raise PublicDatasetError(
                "public source manifest files exceed the cumulative byte limit"
            )
        object.__setattr__(
            self, "files", tuple(sorted(self.files, key=lambda item: item.role))
        )
        object.__setattr__(
            self,
            "expected_statistics",
            _ordered_statistics(
                self.expected_statistics,
                "public source manifest.expected_statistics",
            ),
        )
        _check_model_size(
            self,
            MAX_PUBLIC_SOURCE_MANIFEST_BYTES,
            "public source manifest",
        )

    @classmethod
    def from_dict(cls, value: Any) -> "PublicSourceManifest":
        payload = _object(value, "public source manifest")
        _exact_fields(
            payload,
            (
                "schema_version",
                "dataset_id",
                "dataset_version",
                "source_uri",
                "source_revision",
                "license",
                "files",
                "expected_statistics",
            ),
            "public source manifest",
        )
        files = _array(
            payload["files"],
            "public source manifest.files",
            MAX_PUBLIC_SOURCE_FILES,
        )
        statistics = _array(
            payload["expected_statistics"],
            "public source manifest.expected_statistics",
            MAX_PUBLIC_STATISTICS,
        )
        return cls(
            schema_version=payload["schema_version"],
            dataset_id=_identifier(
                payload["dataset_id"], "public source manifest.dataset_id"
            ),
            dataset_version=_identifier(
                payload["dataset_version"],
                "public source manifest.dataset_version",
            ),
            source_uri=_https_uri(
                payload["source_uri"], "public source manifest.source_uri"
            ),
            source_revision=_identifier(
                payload["source_revision"],
                "public source manifest.source_revision",
            ),
            license=_string(
                payload["license"],
                "public source manifest.license",
                MAX_IDENTIFIER_CHARS,
            ),
            files=tuple(PublicSourceFile.from_dict(item) for item in files),
            expected_statistics=tuple(
                PublicStatistic.from_dict(item) for item in statistics
            ),
        )

    @classmethod
    def from_json(cls, data: Any) -> "PublicSourceManifest":
        return cls.from_dict(
            _strict_json_loads(
                data,
                MAX_PUBLIC_SOURCE_MANIFEST_BYTES,
                "public source manifest JSON",
            )
        )

    def file(self, role: str) -> PublicSourceFile:
        wanted = _identifier(role, "public source role")
        for item in self.files:
            if item.role == wanted:
                return item
        raise PublicDatasetError("public source manifest has no role %r" % wanted)

    def statistic(self, name: str) -> int:
        wanted = _lower_ascii_key(name, "public statistic name")
        for item in self.expected_statistics:
            if item.name == wanted:
                return item.value
        raise PublicDatasetError(
            "public source manifest has no expected statistic %r" % wanted
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "source_uri": self.source_uri,
            "source_revision": self.source_revision,
            "license": self.license,
            "files": [item.to_dict() for item in self.files],
            "expected_statistics": [
                item.to_dict() for item in self.expected_statistics
            ],
        }


@dataclass(frozen=True)
class PublicSelector(_JsonModel):
    name: str
    values: Tuple[str, ...]

    def __post_init__(self) -> None:
        _lower_ascii_key(self.name, "public selector.name")
        if not isinstance(self.values, tuple):
            raise PublicDatasetError("public selector.values must be a tuple")
        if not self.values:
            raise PublicDatasetError(
                "public selector.values must contain at least one explicit value"
            )
        if len(self.values) > MAX_PUBLIC_SELECTOR_VALUES:
            raise PublicDatasetError("public selector.values exceeds the item limit")
        checked = tuple(
            _string(value, "public selector value", MAX_IDENTIFIER_CHARS)
            for value in self.values
        )
        if len(checked) != len(set(checked)):
            raise PublicDatasetError("public selector.values contains duplicates")
        object.__setattr__(self, "values", tuple(sorted(checked)))

    @classmethod
    def from_dict(cls, value: Any) -> "PublicSelector":
        payload = _object(value, "public selector")
        _exact_fields(payload, ("name", "values"), "public selector")
        raw_values = _array(
            payload["values"],
            "public selector.values",
            MAX_PUBLIC_SELECTOR_VALUES,
        )
        return cls(
            name=_lower_ascii_key(payload["name"], "public selector.name"),
            values=tuple(
                _string(item, "public selector value", MAX_IDENTIFIER_CHARS)
                for item in raw_values
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "values": list(self.values)}


@dataclass(frozen=True)
class PublicFilterManifest(_JsonModel):
    SCHEMA_VERSION: ClassVar[str] = PUBLIC_FILTER_MANIFEST_SCHEMA_VERSION

    schema_version: str
    dataset_id: str
    selectors: Tuple[PublicSelector, ...]

    def __post_init__(self) -> None:
        if self.schema_version != self.SCHEMA_VERSION:
            raise PublicDatasetError(
                "public filter manifest has an unknown schema_version"
            )
        _identifier(self.dataset_id, "public filter manifest.dataset_id")
        if not isinstance(self.selectors, tuple):
            raise PublicDatasetError("public filter manifest.selectors must be a tuple")
        if len(self.selectors) > MAX_PUBLIC_SELECTORS:
            raise PublicDatasetError(
                "public filter manifest.selectors exceeds the item limit"
            )
        if any(not isinstance(item, PublicSelector) for item in self.selectors):
            raise PublicDatasetError(
                "public filter manifest.selectors must contain PublicSelector values"
            )
        names = [item.name for item in self.selectors]
        if len(names) != len(set(names)):
            raise PublicDatasetError(
                "public filter manifest.selectors contains duplicate names"
            )
        object.__setattr__(
            self,
            "selectors",
            tuple(sorted(self.selectors, key=lambda item: item.name)),
        )
        _check_model_size(
            self,
            MAX_PUBLIC_FILTER_MANIFEST_BYTES,
            "public filter manifest",
        )

    @classmethod
    def from_dict(cls, value: Any) -> "PublicFilterManifest":
        payload = _object(value, "public filter manifest")
        _exact_fields(
            payload,
            ("schema_version", "dataset_id", "selectors"),
            "public filter manifest",
        )
        selectors = _array(
            payload["selectors"],
            "public filter manifest.selectors",
            MAX_PUBLIC_SELECTORS,
        )
        return cls(
            schema_version=payload["schema_version"],
            dataset_id=_identifier(
                payload["dataset_id"], "public filter manifest.dataset_id"
            ),
            selectors=tuple(PublicSelector.from_dict(item) for item in selectors),
        )

    @classmethod
    def from_json(cls, data: Any) -> "PublicFilterManifest":
        return cls.from_dict(
            _strict_json_loads(
                data,
                MAX_PUBLIC_FILTER_MANIFEST_BYTES,
                "public filter manifest JSON",
            )
        )

    def values(self, name: str) -> Tuple[str, ...]:
        wanted = _lower_ascii_key(name, "public selector name")
        for item in self.selectors:
            if item.name == wanted:
                return item.values
        return ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "selectors": [item.to_dict() for item in self.selectors],
        }


@dataclass(frozen=True)
class VerifiedPublicSource:
    root: Path
    manifest: PublicSourceManifest

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path) or not self.root.is_absolute():
            raise PublicDatasetError("verified public source root must be absolute")
        if not isinstance(self.manifest, PublicSourceManifest):
            raise PublicDatasetError(
                "verified public source requires PublicSourceManifest"
            )

    @classmethod
    def open(
        cls,
        root: os.PathLike[str] | str,
        manifest: PublicSourceManifest,
        *,
        expected_source_manifest_digest: Optional[str] = None,
    ) -> "VerifiedPublicSource":
        if not isinstance(manifest, PublicSourceManifest):
            raise PublicDatasetError(
                "public source verification requires PublicSourceManifest"
            )
        if expected_source_manifest_digest is not None:
            verify_public_source_manifest_digest(
                manifest, expected_source_manifest_digest
            )
        verified = cls(_coerce_suite_root(root), manifest)
        for item in manifest.files:
            verified.read(item.role)
        return verified

    def read(self, role: str) -> bytes:
        binding = self.manifest.file(role)
        try:
            raw = _read_single_link_regular_file(
                self.root,
                binding.path,
                binding.size_bytes,
                "public source file %s" % binding.role,
            )
        except SchemaError as exc:
            raise PublicSourceIntegrityError(str(exc)) from exc
        if len(raw) != binding.size_bytes:
            raise PublicSourceIntegrityError(
                "public source file %s size does not match its manifest" % binding.role
            )
        if hashlib.sha256(raw).hexdigest() != binding.sha256:
            raise PublicSourceIntegrityError(
                "public source file %s hash does not match its manifest" % binding.role
            )
        return raw


@dataclass(frozen=True)
class PublicExtraFile(_JsonModel):
    """One immutable binding for adapter-published non-Case bytes."""

    path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _portable_case_path(self.path, "public extra file.path")
        _integer(
            self.size_bytes,
            "public extra file.size_bytes",
            minimum=0,
            maximum=MAX_PUBLIC_EXTRA_FILE_BYTES,
        )
        _digest(self.sha256, "public extra file.sha256")

    @classmethod
    def from_dict(cls, value: Any) -> "PublicExtraFile":
        payload = _object(value, "public extra file")
        _exact_fields(
            payload,
            ("path", "size_bytes", "sha256"),
            "public extra file",
        )
        return cls(
            path=_portable_case_path(
                payload["path"], "public extra file.path"
            ),
            size_bytes=_integer(
                payload["size_bytes"],
                "public extra file.size_bytes",
                minimum=0,
                maximum=MAX_PUBLIC_EXTRA_FILE_BYTES,
            ),
            sha256=_digest(payload["sha256"], "public extra file.sha256"),
        )

    @classmethod
    def from_bytes(cls, path: str, data: bytes) -> "PublicExtraFile":
        if type(data) is not bytes:
            raise PublicPreparationError(
                "public Suite extra file data must be bytes"
            )
        return cls(
            path=_portable_case_path(path, "public extra file.path"),
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


def _ordered_extra_files(
    values: Iterable[PublicExtraFile], context: str
) -> Tuple[PublicExtraFile, ...]:
    result = tuple(values)
    if len(result) > MAX_PUBLIC_EXTRA_FILES:
        raise PublicDatasetError("%s exceeds the item limit" % context)
    if any(not isinstance(item, PublicExtraFile) for item in result):
        raise PublicDatasetError(
            "%s must contain PublicExtraFile values" % context
        )
    _assert_no_portable_path_collisions(
        (item.path for item in result),
        context,
        reserved=("suite_manifest.json", DEFAULT_PREPARATION_RECEIPT_PATH, "cases"),
    )
    total = sum(item.size_bytes for item in result)
    if total > MAX_PUBLIC_EXTRA_TOTAL_BYTES:
        raise PublicDatasetError(
            "%s exceeds the cumulative byte limit" % context
        )
    return tuple(sorted(result, key=lambda item: item.path))


@dataclass(frozen=True)
class PublicRecordReceipt(_JsonModel):
    task_id: str
    truth_id: Optional[str]
    source_role: str
    record_pointer: str
    upstream_id: Optional[str]
    record_sha256: str
    record_json: str
    disposition: str
    reason: Optional[str]

    def __post_init__(self) -> None:
        _identifier(self.task_id, "public record receipt.task_id")
        if self.truth_id is not None:
            _identifier(self.truth_id, "public record receipt.truth_id")
        _identifier(self.source_role, "public record receipt.source_role")
        _string(
            self.record_pointer,
            "public record receipt.record_pointer",
            MAX_URL_CHARS,
        )
        if self.upstream_id is not None:
            _string(
                self.upstream_id,
                "public record receipt.upstream_id",
                MAX_URL_CHARS,
            )
        _digest(self.record_sha256, "public record receipt.record_sha256")
        record_json = _string(
            self.record_json,
            "public record receipt.record_json",
            MAX_PUBLIC_RECORD_JSON_BYTES,
        )
        record_bytes = record_json.encode("utf-8")
        if len(record_bytes) > MAX_PUBLIC_RECORD_JSON_BYTES:
            raise PublicDatasetError(
                "public record receipt.record_json exceeds its byte limit"
            )
        parsed = _strict_json_loads(
            record_bytes,
            MAX_PUBLIC_RECORD_JSON_BYTES,
            "public record receipt.record_json",
        )
        if canonical_json(parsed) != record_json:
            raise PublicDatasetError(
                "public record receipt.record_json must be canonical JSON"
            )
        if hashlib.sha256(record_bytes).hexdigest() != self.record_sha256:
            raise PublicDatasetError(
                "public record receipt.record_sha256 does not bind record_json"
            )
        _identifier(self.disposition, "public record receipt.disposition")
        _optional_string(
            self.reason,
            "public record receipt.reason",
            8_192,
        )

    @classmethod
    def from_record(
        cls,
        *,
        task_id: str,
        truth_id: Optional[str],
        source_role: str,
        record_pointer: str,
        upstream_id: Optional[str],
        record: Any,
        disposition: str,
        reason: Optional[str] = None,
    ) -> "PublicRecordReceipt":
        record_json = canonical_json(record)
        return cls(
            task_id=task_id,
            truth_id=truth_id,
            source_role=source_role,
            record_pointer=record_pointer,
            upstream_id=upstream_id,
            record_sha256=hashlib.sha256(record_json.encode("utf-8")).hexdigest(),
            record_json=record_json,
            disposition=disposition,
            reason=reason,
        )

    @classmethod
    def from_dict(cls, value: Any) -> "PublicRecordReceipt":
        payload = _object(value, "public record receipt")
        _exact_fields(
            payload,
            (
                "task_id",
                "truth_id",
                "source_role",
                "record_pointer",
                "upstream_id",
                "record_sha256",
                "record_json",
                "disposition",
                "reason",
            ),
            "public record receipt",
        )
        return cls(
            task_id=_identifier(
                payload["task_id"], "public record receipt.task_id"
            ),
            truth_id=(
                None
                if payload["truth_id"] is None
                else _identifier(
                    payload["truth_id"], "public record receipt.truth_id"
                )
            ),
            source_role=_identifier(
                payload["source_role"], "public record receipt.source_role"
            ),
            record_pointer=_string(
                payload["record_pointer"],
                "public record receipt.record_pointer",
                MAX_URL_CHARS,
            ),
            upstream_id=_optional_string(
                payload["upstream_id"],
                "public record receipt.upstream_id",
                MAX_URL_CHARS,
            ),
            record_sha256=_digest(
                payload["record_sha256"],
                "public record receipt.record_sha256",
            ),
            record_json=_string(
                payload["record_json"],
                "public record receipt.record_json",
                MAX_PUBLIC_RECORD_JSON_BYTES,
                allow_empty=False,
            ),
            disposition=_identifier(
                payload["disposition"],
                "public record receipt.disposition",
            ),
            reason=_optional_string(
                payload["reason"],
                "public record receipt.reason",
                8_192,
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "truth_id": self.truth_id,
            "source_role": self.source_role,
            "record_pointer": self.record_pointer,
            "upstream_id": self.upstream_id,
            "record_sha256": self.record_sha256,
            "record_json": self.record_json,
            "disposition": self.disposition,
            "reason": self.reason,
        }


def _ordered_record_receipts(
    values: Iterable[PublicRecordReceipt], context: str
) -> Tuple[PublicRecordReceipt, ...]:
    result = tuple(values)
    if len(result) > MAX_PUBLIC_RECORD_RECEIPTS:
        raise PublicDatasetError("%s exceeds the item limit" % context)
    if any(not isinstance(item, PublicRecordReceipt) for item in result):
        raise PublicDatasetError(
            "%s must contain PublicRecordReceipt values" % context
        )
    ordered = tuple(
        sorted(
            result,
            key=lambda item: (
                item.task_id,
                item.truth_id or "",
                item.source_role,
                item.record_pointer,
                item.disposition,
            ),
        )
    )
    keys = [
        (
            item.task_id,
            item.truth_id,
            item.source_role,
            item.record_pointer,
            item.disposition,
        )
        for item in ordered
    ]
    if len(keys) != len(set(keys)):
        raise PublicDatasetError(
            "%s contains duplicate provenance keys" % context
        )
    return ordered


def _records_digest(records: Tuple[PublicRecordReceipt, ...]) -> str:
    return canonical_sha256([item.to_dict() for item in records])


def _extra_files_digest(extra_files: Tuple[PublicExtraFile, ...]) -> str:
    return canonical_sha256([item.to_dict() for item in extra_files])


def _case_bindings_digest(cases: Iterable[SuiteCase]) -> str:
    return canonical_sha256([item.to_dict() for item in cases])


def _preparation_packet_digest(
    *,
    adapter_id: str,
    adapter_version: str,
    source_manifest_digest: str,
    filter_manifest_digest: str,
    case_bindings_digest: str,
    actual_statistics: Tuple[PublicStatistic, ...],
    records_digest: str,
    extra_files_digest: str,
) -> str:
    """Hash the complete acyclic preparation identity anchored by the Suite."""

    return canonical_sha256(
        {
            "schema_version": PUBLIC_PREPARATION_PACKET_SCHEMA_VERSION,
            "adapter_id": _identifier(adapter_id, "preparation packet.adapter_id"),
            "adapter_version": _identifier(
                adapter_version, "preparation packet.adapter_version"
            ),
            "source_manifest_digest": _digest(
                source_manifest_digest,
                "preparation packet.source_manifest_digest",
            ),
            "filter_manifest_digest": _digest(
                filter_manifest_digest,
                "preparation packet.filter_manifest_digest",
            ),
            "case_bindings_digest": _digest(
                case_bindings_digest,
                "preparation packet.case_bindings_digest",
            ),
            "actual_statistics": [
                item.to_dict() for item in actual_statistics
            ],
            "records_digest": _digest(
                records_digest, "preparation packet.records_digest"
            ),
            "extra_files_digest": _digest(
                extra_files_digest, "preparation packet.extra_files_digest"
            ),
        }
    )


@dataclass(frozen=True)
class PublicPreparedCase:
    case: EvalCase
    split: CaseSplit
    protocol_id: str
    dimensions: Tuple[CaseDimension, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.case, EvalCase):
            raise PublicDatasetError("public prepared case.case must be EvalCase")
        if not isinstance(self.split, CaseSplit):
            raise PublicDatasetError("public prepared case.split must be CaseSplit")
        _identifier(self.protocol_id, "public prepared case.protocol_id")
        if not isinstance(self.dimensions, tuple) or any(
            not isinstance(item, CaseDimension) for item in self.dimensions
        ):
            raise PublicDatasetError(
                "public prepared case.dimensions must contain CaseDimension values"
            )


@dataclass(frozen=True)
class PublicFrozenBundlePublication:
    bundle: "PreparedFrozenContextBundle"
    relative_root: str

    def __post_init__(self) -> None:
        from .swe_prbench import PreparedFrozenContextBundle

        if type(self.bundle) is not PreparedFrozenContextBundle:
            raise PublicPreparationError(
                "frozen publication.bundle must be PreparedFrozenContextBundle"
            )
        object.__setattr__(
            self,
            "relative_root",
            _portable_case_path(
                self.relative_root,
                "frozen publication.relative_root",
            ),
        )


@dataclass(frozen=True)
class PublicPreparationReceipt(_JsonModel):
    SCHEMA_VERSION: ClassVar[str] = PUBLIC_PREPARATION_RECEIPT_SCHEMA_VERSION

    schema_version: str
    adapter_id: str
    adapter_version: str
    source_manifest: PublicSourceManifest
    source_manifest_digest: str
    filter_manifest: PublicFilterManifest
    filter_manifest_digest: str
    case_bindings_digest: str
    suite_manifest_digest: str
    actual_statistics: Tuple[PublicStatistic, ...]
    records: Tuple[PublicRecordReceipt, ...]
    extra_files: Tuple[PublicExtraFile, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != self.SCHEMA_VERSION:
            raise PublicDatasetError(
                "public preparation receipt has an unknown schema_version"
            )
        _identifier(self.adapter_id, "public preparation receipt.adapter_id")
        _identifier(self.adapter_version, "public preparation receipt.adapter_version")
        if not isinstance(self.source_manifest, PublicSourceManifest):
            raise PublicDatasetError(
                "public preparation receipt.source_manifest is invalid"
            )
        if self.source_manifest.digest() != _digest(
            self.source_manifest_digest,
            "public preparation receipt.source_manifest_digest",
        ):
            raise PublicDatasetError(
                "public preparation receipt source manifest digest mismatch"
            )
        if not isinstance(self.filter_manifest, PublicFilterManifest):
            raise PublicDatasetError(
                "public preparation receipt.filter_manifest is invalid"
            )
        if self.filter_manifest.digest() != _digest(
            self.filter_manifest_digest,
            "public preparation receipt.filter_manifest_digest",
        ):
            raise PublicDatasetError(
                "public preparation receipt filter manifest digest mismatch"
            )
        if self.source_manifest.dataset_id != self.filter_manifest.dataset_id:
            raise PublicDatasetError(
                "public preparation receipt dataset manifests disagree"
            )
        _digest(
            self.case_bindings_digest,
            "public preparation receipt.case_bindings_digest",
        )
        _digest(
            self.suite_manifest_digest,
            "public preparation receipt.suite_manifest_digest",
        )
        if not isinstance(self.actual_statistics, tuple):
            raise PublicDatasetError(
                "public preparation receipt.actual_statistics must be a tuple"
            )
        object.__setattr__(
            self,
            "actual_statistics",
            _ordered_statistics(
                self.actual_statistics,
                "public preparation receipt.actual_statistics",
            ),
        )
        if not isinstance(self.records, tuple):
            raise PublicDatasetError(
                "public preparation receipt.records must be a tuple"
            )
        ordered_records = _ordered_record_receipts(
            self.records, "public preparation receipt.records"
        )
        source_roles = {item.role for item in self.source_manifest.files}
        unknown_roles = sorted(
            {item.source_role for item in ordered_records} - source_roles
        )
        if unknown_roles:
            raise PublicDatasetError(
                "public preparation receipt.records references unknown source role(s): %s"
                % ", ".join(unknown_roles)
            )
        object.__setattr__(self, "records", ordered_records)
        if not isinstance(self.extra_files, tuple):
            raise PublicDatasetError(
                "public preparation receipt.extra_files must be a tuple"
            )
        object.__setattr__(
            self,
            "extra_files",
            _ordered_extra_files(
                self.extra_files, "public preparation receipt.extra_files"
            ),
        )
        _check_model_size(
            self,
            MAX_PUBLIC_PREPARATION_RECEIPT_BYTES,
            "public preparation receipt",
        )

    @property
    def records_digest(self) -> str:
        return _records_digest(self.records)

    @property
    def extra_files_digest(self) -> str:
        return _extra_files_digest(self.extra_files)

    @property
    def preparation_packet_digest(self) -> str:
        return _preparation_packet_digest(
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            source_manifest_digest=self.source_manifest_digest,
            filter_manifest_digest=self.filter_manifest_digest,
            case_bindings_digest=self.case_bindings_digest,
            actual_statistics=self.actual_statistics,
            records_digest=self.records_digest,
            extra_files_digest=self.extra_files_digest,
        )

    @classmethod
    def from_dict(cls, value: Any) -> "PublicPreparationReceipt":
        payload = _object(value, "public preparation receipt")
        _exact_fields(
            payload,
            (
                "schema_version",
                "adapter_id",
                "adapter_version",
                "source_manifest",
                "source_manifest_digest",
                "filter_manifest",
                "filter_manifest_digest",
                "case_bindings_digest",
                "suite_manifest_digest",
                "actual_statistics",
                "records",
                "records_digest",
                "extra_files",
                "extra_files_digest",
            ),
            "public preparation receipt",
        )
        statistics = _array(
            payload["actual_statistics"],
            "public preparation receipt.actual_statistics",
            MAX_PUBLIC_STATISTICS,
        )
        records = _array(
            payload["records"],
            "public preparation receipt.records",
            MAX_PUBLIC_RECORD_RECEIPTS,
        )
        extra_files = _array(
            payload["extra_files"],
            "public preparation receipt.extra_files",
            MAX_PUBLIC_EXTRA_FILES,
        )
        receipt = cls(
            schema_version=payload["schema_version"],
            adapter_id=_identifier(
                payload["adapter_id"], "public preparation receipt.adapter_id"
            ),
            adapter_version=_identifier(
                payload["adapter_version"],
                "public preparation receipt.adapter_version",
            ),
            source_manifest=PublicSourceManifest.from_dict(
                payload["source_manifest"]
            ),
            source_manifest_digest=_digest(
                payload["source_manifest_digest"],
                "public preparation receipt.source_manifest_digest",
            ),
            filter_manifest=PublicFilterManifest.from_dict(
                payload["filter_manifest"]
            ),
            filter_manifest_digest=_digest(
                payload["filter_manifest_digest"],
                "public preparation receipt.filter_manifest_digest",
            ),
            case_bindings_digest=_digest(
                payload["case_bindings_digest"],
                "public preparation receipt.case_bindings_digest",
            ),
            suite_manifest_digest=_digest(
                payload["suite_manifest_digest"],
                "public preparation receipt.suite_manifest_digest",
            ),
            actual_statistics=tuple(
                PublicStatistic.from_dict(item) for item in statistics
            ),
            records=tuple(PublicRecordReceipt.from_dict(item) for item in records),
            extra_files=tuple(
                PublicExtraFile.from_dict(item) for item in extra_files
            ),
        )
        if receipt.records_digest != _digest(
            payload["records_digest"],
            "public preparation receipt.records_digest",
        ):
            raise PublicDatasetError(
                "public preparation receipt records digest mismatch"
            )
        if receipt.extra_files_digest != _digest(
            payload["extra_files_digest"],
            "public preparation receipt.extra_files_digest",
        ):
            raise PublicDatasetError(
                "public preparation receipt extra files digest mismatch"
            )
        return receipt

    @classmethod
    def from_json(cls, data: Any) -> "PublicPreparationReceipt":
        return cls.from_dict(
            _strict_json_loads(
                data,
                MAX_PUBLIC_PREPARATION_RECEIPT_BYTES,
                "public preparation receipt JSON",
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "source_manifest": self.source_manifest.to_dict(),
            "source_manifest_digest": self.source_manifest_digest,
            "filter_manifest": self.filter_manifest.to_dict(),
            "filter_manifest_digest": self.filter_manifest_digest,
            "case_bindings_digest": self.case_bindings_digest,
            "suite_manifest_digest": self.suite_manifest_digest,
            "actual_statistics": [
                item.to_dict() for item in self.actual_statistics
            ],
            "records": [item.to_dict() for item in self.records],
            "records_digest": self.records_digest,
            "extra_files": [item.to_dict() for item in self.extra_files],
            "extra_files_digest": self.extra_files_digest,
        }


@dataclass(frozen=True)
class PublicPreparationResult:
    suite_root: Path
    manifest: SuiteManifest
    receipt: PublicPreparationReceipt
    bundle_id: Optional[str] = None

    @property
    def preparation_packet_digest(self) -> str:
        return self.receipt.preparation_packet_digest

    @property
    def suite_manifest_digest(self) -> str:
        return self.manifest.digest()

    @property
    def case_bindings_digest(self) -> str:
        return self.receipt.case_bindings_digest


def _has_reparse_attribute(metadata: os.stat_result) -> bool:
    return bool(
        getattr(metadata, "st_file_attributes", 0) & _REPARSE_POINT_FLAG
    )


def _read_single_link_regular_file(
    root: Path,
    relative_path: str,
    maximum_bytes: int,
    context: str,
) -> bytes:
    """Reuse the secure reader while rejecting externally aliased bytes."""

    try:
        _path_before, before = _secure_regular_file(
            root, relative_path, context
        )
        if getattr(before, "st_nlink", None) != 1:
            raise PublicSourceIntegrityError(
                "%s must report exactly one link and may not have multiple hard links"
                % context
            )
        raw = _read_relative_regular_file(
            root, relative_path, maximum_bytes, context
        )
        _path_after, after = _secure_regular_file(
            root, relative_path, context
        )
        if getattr(after, "st_nlink", None) != 1:
            raise PublicSourceIntegrityError(
                "%s must report exactly one link and may not have multiple hard links"
                % context
            )
        if (
            before.st_ino
            and after.st_ino
            and (before.st_dev, before.st_ino)
            != (after.st_dev, after.st_ino)
        ):
            raise PublicSourceIntegrityError(
                "%s changed identity across its verified read" % context
            )
        return raw
    except PublicSourceIntegrityError:
        raise
    except SchemaError as exc:
        raise PublicSourceIntegrityError(str(exc)) from exc


def _assert_publication_parent(path: Path) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path)))
    try:
        metadata = os.lstat(str(lexical))
    except OSError as exc:
        raise PublicPreconditionError(
            "public Suite output parent must already exist"
        ) from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or _has_reparse_attribute(metadata)
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise PublicPreconditionError(
            "public Suite output parent must be a real directory"
        )
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise PublicPreconditionError(
            "public Suite output parent could not be resolved"
        ) from exc
    if os.path.normcase(str(lexical)) != os.path.normcase(str(resolved)):
        raise PublicPreconditionError(
            "public Suite output parent may not traverse a symlink or reparse point"
        )
    return resolved


def _validate_staging_root(
    root: Path,
    expected_identity: Optional[Tuple[int, int, int]],
) -> Tuple[int, int, int]:
    try:
        metadata = os.lstat(str(root))
    except OSError as exc:
        raise PublicPreparationError(
            "public Suite staging root is unavailable"
        ) from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or _has_reparse_attribute(metadata)
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise PublicPreparationError(
            "public Suite staging root is a symlink, reparse point, or non-directory"
        )
    identity = _file_identity(metadata)
    if expected_identity is not None and identity != expected_identity:
        raise PublicPreparationError("public Suite staging root identity changed")
    return identity


def _write_all(descriptor: int, data: bytes) -> None:
    offset = 0
    while offset < len(data):
        written = os.write(descriptor, data[offset:])
        if written <= 0:
            raise OSError("short public Suite write")
        offset += written


def _validate_open_output_file(
    root: Path,
    portable: str,
    descriptor: int,
    opened: os.stat_result,
    expected_size: int,
) -> None:
    after = os.fstat(descriptor)
    if (
        not stat.S_ISREG(after.st_mode)
        or _has_reparse_attribute(after)
        or getattr(after, "st_nlink", None) != 1
        or _file_identity(opened) != _file_identity(after)
        or after.st_size != expected_size
    ):
        raise PublicPreparationError(
            "public Suite output file changed or became unsafe while open"
        )
    if os.name != "nt" and (
        not getattr(opened, "st_ino", 0) or not getattr(after, "st_ino", 0)
    ):
        raise PublicPreparationError(
            "public Suite output file identity is unavailable"
        )
    opened_path = _windows_descriptor_path(descriptor)
    expected_path = root.joinpath(*portable.split("/"))
    if opened_path is not None and (
        _normalized_filesystem_path(opened_path)
        != _normalized_filesystem_path(expected_path)
        or not _path_is_within(root, opened_path)
    ):
        raise PublicPreparationError(
            "public Suite output file resolved outside its staging path"
        )
    try:
        _path, path_metadata = _secure_regular_file(
            root, portable, "public Suite output file"
        )
    except SchemaError as exc:
        raise PublicPreparationError(str(exc)) from exc
    if (
        getattr(path_metadata, "st_nlink", None) != 1
        or _file_identity(path_metadata) != _file_identity(after)
    ):
        raise PublicPreparationError(
            "public Suite output path changed identity after its write"
        )


def _write_new_posix(
    root: Path,
    portable: str,
    data: bytes,
    expected_root_identity: Tuple[int, int, int],
) -> None:
    if (
        not hasattr(os, "O_NOFOLLOW")
        or os.open not in getattr(os, "supports_dir_fd", set())
        or os.mkdir not in getattr(os, "supports_dir_fd", set())
    ):
        raise PublicPreparationError(
            "descriptor-relative no-follow public Suite writes are unavailable"
        )
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(str(root), directory_flags)
    file_descriptor: Optional[int] = None
    try:
        root_metadata = os.fstat(descriptor)
        if (
            _file_identity(root_metadata) != expected_root_identity
            or _has_reparse_attribute(root_metadata)
            or not stat.S_ISDIR(root_metadata.st_mode)
        ):
            raise PublicPreparationError(
                "public Suite staging descriptor identity changed"
            )
        components = portable.split("/")
        for component in components[:-1]:
            try:
                os.mkdir(component, 0o700, dir_fd=descriptor)
            except FileExistsError:
                pass
            next_descriptor = os.open(
                component, directory_flags, dir_fd=descriptor
            )
            next_metadata = os.fstat(next_descriptor)
            if (
                _has_reparse_attribute(next_metadata)
                or not stat.S_ISDIR(next_metadata.st_mode)
            ):
                os.close(next_descriptor)
                raise PublicPreparationError(
                    "public Suite output contains an unsafe directory component"
                )
            os.close(descriptor)
            descriptor = next_descriptor
        file_flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_BINARY", 0)
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        file_descriptor = os.open(
            components[-1],
            file_flags,
            0o600,
            dir_fd=descriptor,
        )
        opened = os.fstat(file_descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or _has_reparse_attribute(opened)
            or getattr(opened, "st_nlink", None) != 1
        ):
            raise PublicPreparationError(
                "public Suite output is not a single-link regular file"
            )
        _write_all(file_descriptor, data)
        os.fsync(file_descriptor)
        _validate_open_output_file(
            root, portable, file_descriptor, opened, len(data)
        )
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        os.close(descriptor)
    _validate_staging_root(root, expected_root_identity)


def _validate_windows_output_directory_handle(
    handle: int, expected_path: Path, root: Path
) -> None:
    actual = _windows_raw_handle_path(handle)
    attributes = _windows_raw_handle_attributes(handle)
    if (
        _normalized_filesystem_path(actual)
        != _normalized_filesystem_path(expected_path)
        or not _path_is_within(root, actual)
        or attributes & _REPARSE_POINT_FLAG
        or not attributes & 0x10
    ):
        raise PublicPreparationError(
            "public Suite output contains a reparse point or unsafe directory"
        )


def _write_new_windows(
    root: Path,
    portable: str,
    data: bytes,
    expected_root_identity: Tuple[int, int, int],
) -> None:
    handles: list[int] = []
    descriptor: Optional[int] = None
    current = root
    try:
        root_handle = _windows_open_directory_handle(root)
        handles.append(root_handle)
        _validate_windows_output_directory_handle(root_handle, root, root)
        if _file_identity(os.lstat(str(root))) != expected_root_identity:
            raise PublicPreparationError(
                "public Suite staging root identity changed"
            )
        components = portable.split("/")
        for component in components[:-1]:
            target = current / component
            try:
                os.mkdir(target, 0o700)
            except FileExistsError:
                pass
            handle = _windows_open_directory_handle(target)
            handles.append(handle)
            _validate_windows_output_directory_handle(handle, target, root)
            current = target
        target = current / components[-1]
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOINHERIT", 0)
        )
        descriptor = os.open(str(target), flags, 0o600)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or _has_reparse_attribute(opened)
            or getattr(opened, "st_nlink", None) != 1
        ):
            raise PublicPreparationError(
                "public Suite output is not a single-link regular file"
            )
        _write_all(descriptor, data)
        os.fsync(descriptor)
        _validate_open_output_file(root, portable, descriptor, opened, len(data))
    finally:
        if descriptor is not None:
            os.close(descriptor)
        for handle in reversed(handles):
            _windows_close_handle(handle)
    _validate_staging_root(root, expected_root_identity)


def _write_new(
    root: Path,
    relative: str,
    data: bytes,
    *,
    expected_root_identity: Optional[Tuple[int, int, int]] = None,
) -> None:
    portable = _portable_case_path(relative, "public Suite output path")
    if type(data) is not bytes:
        raise PublicPreparationError("public Suite output data must be bytes")
    root_identity = _validate_staging_root(root, expected_root_identity)
    try:
        if os.name == "nt":
            _write_new_windows(root, portable, data, root_identity)
        else:
            _write_new_posix(root, portable, data, root_identity)
    except FileExistsError as exc:
        raise PublicPreparationError(
            "public Suite output path already exists: %s" % portable
        ) from exc
    except PublicPreparationError:
        raise
    except (ArtifactError, SchemaError, OSError) as exc:
        raise PublicPreparationError(
            "public Suite output could not be written safely: %s" % portable
        ) from exc


def _validate_extra_files(extra_files: Mapping[str, bytes]) -> Dict[str, bytes]:
    if not isinstance(extra_files, Mapping):
        raise PublicPreparationError("public Suite extra_files must be a mapping")
    if len(extra_files) > MAX_PUBLIC_EXTRA_FILES:
        raise PublicPreparationError("public Suite extra_files exceeds the item limit")
    checked: Dict[str, bytes] = {}
    total = 0
    for raw_path, data in extra_files.items():
        path = _portable_case_path(raw_path, "public Suite extra file path")
        if type(data) is not bytes:
            raise PublicPreparationError("public Suite extra file data must be bytes")
        if len(data) > MAX_PUBLIC_EXTRA_FILE_BYTES:
            raise PublicPreparationError(
                "public Suite extra file exceeds the byte limit"
            )
        total += len(data)
        if total > MAX_PUBLIC_EXTRA_TOTAL_BYTES:
            raise PublicPreparationError(
                "public Suite extra files exceed the cumulative byte limit"
            )
        if path in checked:
            raise PublicPreparationError("public Suite extra file path is duplicated")
        checked[path] = data
    try:
        _assert_no_portable_path_collisions(
            checked,
            "public Suite extra files",
            reserved=(
                "suite_manifest.json",
                DEFAULT_PREPARATION_RECEIPT_PATH,
                "cases",
            ),
        )
    except PublicDatasetError as exc:
        raise PublicPreparationError(str(exc)) from exc
    return checked


def _file_identity(metadata: os.stat_result) -> Tuple[int, int, int]:
    return (
        int(getattr(metadata, "st_dev", 0)),
        int(getattr(metadata, "st_ino", 0)),
        stat.S_IFMT(metadata.st_mode),
    )


def _cleanup_owned_staging(
    staging: Path,
    parent: Path,
    identity: Tuple[int, int, int],
) -> bool:
    """Delete only the exact, link-free staging tree created by this call.

    Cleanup is best effort.  Any identity drift, symlink, reparse point, or
    unusual filesystem node causes the whole cleanup to be abandoned instead
    of recursively following an attacker-controlled tree.
    """

    if staging.parent != parent:
        return False
    try:
        root_metadata = os.lstat(str(staging))
    except OSError:
        return False
    if (
        _file_identity(root_metadata) != identity
        or stat.S_ISLNK(root_metadata.st_mode)
        or _has_reparse_attribute(root_metadata)
        or not stat.S_ISDIR(root_metadata.st_mode)
    ):
        return False

    snapshots: list[Tuple[Path, Tuple[int, int, int], bool]] = []
    pending = [staging]
    try:
        while pending:
            directory = pending.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    # On Windows, ``DirEntry.stat(follow_symlinks=False)`` may
                    # report zero ``st_dev``/``st_ino`` values while a later
                    # ``os.lstat`` returns the real file identity.  Mixing the
                    # two APIs would make safe cleanup mistake every owned
                    # child for a path-swap attack and leak staging trees.
                    # Capture and revalidate with the same no-follow API.
                    metadata = os.lstat(entry.path)
                    if stat.S_ISLNK(metadata.st_mode) or _has_reparse_attribute(
                        metadata
                    ):
                        return False
                    child = Path(entry.path)
                    if stat.S_ISDIR(metadata.st_mode):
                        snapshots.append((child, _file_identity(metadata), True))
                        pending.append(child)
                    elif stat.S_ISREG(metadata.st_mode):
                        snapshots.append((child, _file_identity(metadata), False))
                    else:
                        return False
    except OSError:
        return False

    try:
        for path, expected, is_directory in sorted(
            snapshots,
            key=lambda item: len(item[0].parts),
            reverse=True,
        ):
            current = os.lstat(str(path))
            if (
                _file_identity(current) != expected
                or stat.S_ISLNK(current.st_mode)
                or _has_reparse_attribute(current)
            ):
                return False
            if is_directory:
                os.rmdir(path)
            else:
                os.unlink(path)
        current_root = os.lstat(str(staging))
        if _file_identity(current_root) != identity:
            return False
        os.rmdir(staging)
        return True
    except OSError:
        return False


def _publish_directory_create_only(staging: Path, output: Path) -> None:
    """Atomically publish a directory without ever replacing a destination."""

    if os.name == "nt":
        try:
            os.rename(staging, output)
            return
        except FileExistsError as exc:
            raise PublicConflictError(
                "public Suite output already exists"
            ) from exc
        except OSError as exc:
            if getattr(exc, "winerror", None) in (80, 183) or os.path.lexists(output):
                raise PublicConflictError(
                    "public Suite output already exists"
                ) from exc
            raise PublicOperationalError(
                "public Suite could not be published without replacement"
            ) from exc

    # Linux exposes atomic RENAME_NOREPLACE through renameat2.  Refuse an
    # unsafe check-then-rename fallback on platforms without an equivalent.
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
    except (AttributeError, ImportError, OSError) as exc:
        raise PublicOperationalError(
            "platform lacks atomic create-only directory publication"
        ) from exc
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
        os.fsencode(staging),
        -100,
        os.fsencode(output),
        1,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in (errno.EEXIST, errno.ENOTEMPTY):
        raise PublicConflictError("public Suite output already exists")
    raise PublicOperationalError(
        "public Suite could not be published without replacement: %s"
        % os.strerror(error)
    )


@dataclass(frozen=True)
class _PreparedCaseFile:
    prepared: PublicPreparedCase
    path: str
    raw: bytes
    binding: SuiteCase


def _public_source_catalog_digest(source_manifest: PublicSourceManifest) -> str:
    """Bind the verified source manifest as the public source catalog root.

    Task 4A only has the already-verified source manifest/receipt boundary;
    acquisition and catalog artifacts are deliberately not implemented here.
    These domain-separated digests make that existing trust root explicit
    without inventing a downloader or a catalog parser.
    """

    return canonical_sha256(
        {
            "schema_version": "public-source-catalog-projection-v2",
            "source_manifest": source_manifest.to_dict(),
        }
    )


def _public_acquisition_receipt_digest(
    source_manifest: PublicSourceManifest,
) -> str:
    return canonical_sha256(
        {
            "schema_version": "verified-public-source-receipt-projection-v2",
            "source_manifest_digest": source_manifest.digest(),
            "files": [item.to_dict() for item in source_manifest.files],
        }
    )


def _repository_target_catalog_digest(
    cases: Tuple[SuiteCase, ...],
    prepared_cases: Tuple[_PreparedCaseFile, ...],
) -> str:
    targets = []
    for item, prepared in zip(cases, prepared_cases):
        target = prepared.prepared.case.input.review_target
        if target.kind is not ReviewTargetKind.REPOSITORY:
            raise PublicPreparationError(
                "Repository Suite contains a non-Repository Target"
            )
        targets.append(
            {
                "task_id": item.task_id,
                "target": target.to_dict(),
            }
        )
    return canonical_sha256(
        {
            "schema_version": "repository-target-catalog-projection-v2",
            "targets": targets,
        }
    )


def _validate_frozen_case_targets(
    bundle: "PreparedFrozenContextBundle",
    case_files: Tuple[_PreparedCaseFile, ...],
) -> None:
    from ..frozen_context import (
        frozen_context_record_id,
        frozen_context_source_binding_digest,
    )
    from ..materialization import MaterializationError

    for item in case_files:
        case = item.prepared.case
        target = case.input.review_target
        if not isinstance(target, FrozenContextReviewTarget):
            raise PublicPreparationError(
                "Frozen Suite contains a non-Frozen Case Target"
            )
        if target.bundle_id != bundle.manifest.bundle_id:
            raise PublicPreparationError(
                "Frozen Case Target bundle_id does not match the verified bundle"
            )
        matches = [
            binding
            for binding in bundle.manifest.records
            if frozen_context_record_id(binding) == target.record_id
        ]
        if len(matches) != 1:
            raise PublicPreparationError(
                "Frozen Case Target record_id does not uniquely match the verified bundle"
            )
        binding = matches[0]
        try:
            source_binding_digest = frozen_context_source_binding_digest(
                bundle, binding
            )
        except (PublicDatasetError, SchemaError, MaterializationError, TypeError) as exc:
            raise PublicPreparationError(
                "Frozen Case Target source binding could not be verified"
            ) from exc
        if (
            binding.task_id != case.task_id
            or target.context_format != "rendered_text"
            or target.rendered_sha256 != binding.rendered_sha256
            or target.rendered_utf8_bytes != binding.rendered_utf8_bytes
            or target.source_binding_digest != source_binding_digest
        ):
            raise PublicPreparationError(
                "Frozen Case Target binding does not match the verified bundle record"
            )


def _validate_frozen_publication_inputs(
    publication: PublicFrozenBundlePublication,
    *,
    source_manifest: PublicSourceManifest,
    filter_manifest: PublicFilterManifest,
    case_files: Tuple[_PreparedCaseFile, ...],
) -> None:
    bundle = publication.bundle
    if (
        bundle.manifest.source_manifest_digest != source_manifest.digest()
        or bundle.manifest.filter_manifest_digest != filter_manifest.digest()
    ):
        raise PublicPreparationError(
            "Frozen bundle source/filter binding does not match Suite inputs"
        )
    _validate_frozen_case_targets(bundle, case_files)


def _verify_staged_frozen_publication(
    staging: Path,
    publication: PublicFrozenBundlePublication,
    case_files: Tuple[_PreparedCaseFile, ...],
) -> None:
    from .swe_prbench import read_swe_prbench_frozen_bundle
    from ..materialization import MaterializationError

    try:
        staged = read_swe_prbench_frozen_bundle(
            staging / Path(*publication.relative_root.split("/")),
            expected_bundle_id=publication.bundle.manifest.bundle_id,
        )
    except (PublicDatasetError, SchemaError, MaterializationError, TypeError) as exc:
        raise PublicPreparationError(
            "staging Frozen bundle verifier rejected publication"
        ) from exc
    if staged.manifest != publication.bundle.manifest:
        raise PublicPreparationError(
            "staging Frozen bundle manifest differs from the verified source bundle"
        )
    _validate_frozen_case_targets(staged, case_files)


def _preparation_binding(
    *,
    source_manifest: PublicSourceManifest,
    filter_manifest: PublicFilterManifest,
    preparation_packet_digest: str,
    wire_contract: WireContractV2,
    manifest_cases: Tuple[SuiteCase, ...],
    case_files: Tuple[_PreparedCaseFile, ...],
    frozen_publication: Optional[PublicFrozenBundlePublication],
) -> PublicSuitePreparationBindingV2:
    source_digest = source_manifest.digest()
    filter_digest = filter_manifest.digest()
    repository_catalog_digest = None
    frozen_bundle_trust = None
    if wire_contract.review_target_kind is ReviewTargetKind.REPOSITORY:
        if frozen_publication is not None:
            raise PublicPreparationError(
                "Repository Suite may not carry a Frozen bundle binding"
            )
        repository_catalog_digest = _repository_target_catalog_digest(
            manifest_cases, case_files
        )
    elif wire_contract.review_target_kind is ReviewTargetKind.FROZEN_CONTEXT:
        if frozen_publication is None:
            raise PublicPreparationError(
                "Frozen Suite requires a verified Frozen bundle"
            )
        # Import lazily: frozen_context re-exports the SWE bundle verifier and
        # imports this module for its public-source errors.
        from ..frozen_context import frozen_bundle_trust_digest
        from ..materialization import MaterializationError

        provisional = PublicSuitePreparationBindingV2(
            schema_version=PUBLIC_SUITE_PREPARATION_BINDING_SCHEMA_VERSION,
            source_catalog_digest=_public_source_catalog_digest(source_manifest),
            acquisition_receipt_digest=_public_acquisition_receipt_digest(
                source_manifest
            ),
            source_manifest_digest=source_digest,
            filter_manifest_digest=filter_digest,
            preparation_packet_digest=preparation_packet_digest,
            repository_catalog_digest=None,
            frozen_bundle_trust_digest="0" * 64,
        )
        try:
            frozen_bundle_trust = frozen_bundle_trust_digest(
                frozen_publication.bundle, provisional
            )
        except (PublicDatasetError, SchemaError, MaterializationError, TypeError) as exc:
            raise PublicPreparationError(
                "Frozen Suite trust binding could not be derived"
            ) from exc
    else:  # pragma: no cover - WireContractV2 validates this enum
        raise PublicPreparationError("public Suite Target kind is unsupported")

    return PublicSuitePreparationBindingV2(
        schema_version=PUBLIC_SUITE_PREPARATION_BINDING_SCHEMA_VERSION,
        source_catalog_digest=_public_source_catalog_digest(source_manifest),
        acquisition_receipt_digest=_public_acquisition_receipt_digest(
            source_manifest
        ),
        source_manifest_digest=source_digest,
        filter_manifest_digest=filter_digest,
        preparation_packet_digest=preparation_packet_digest,
        repository_catalog_digest=repository_catalog_digest,
        frozen_bundle_trust_digest=frozen_bundle_trust,
    )


def _preflight_public_cases(
    cases: Iterable[PublicPreparedCase],
    *,
    suite_id: str,
    source_manifest: PublicSourceManifest,
    wire_contract: WireContractV2,
) -> Tuple[_PreparedCaseFile, ...]:
    bounded = []
    for item in cases:
        if len(bounded) >= MAX_SUITE_CASES:
            raise PublicPreparationError(
                "public Suite Case count exceeds the item limit of %d"
                % MAX_SUITE_CASES
            )
        if not isinstance(item, PublicPreparedCase):
            raise PublicPreparationError(
                "public Suite cases must contain PublicPreparedCase values"
            )
        bounded.append(item)
    if not bounded:
        raise PublicPreparationError("public Suite must contain at least one Case")
    task_ids = [item.case.task_id for item in bounded]
    if len(task_ids) != len(set(task_ids)):
        raise PublicPreparationError("public Suite contains duplicate task IDs")
    try:
        _assert_no_portable_path_collisions(
            ("cases/%s.json" % task_id for task_id in task_ids),
            "public Suite Case paths",
        )
    except PublicDatasetError as exc:
        raise PublicPreparationError(str(exc)) from exc

    total_case_bytes = 0
    result = []
    for prepared in sorted(bounded, key=lambda item: item.case.task_id):
        case = prepared.case
        if (
            case.source.suite != suite_id
            or case.source.source_version != source_manifest.source_revision
            or case.source.source_uri != source_manifest.source_uri
            or case.source.license != source_manifest.license
        ):
            raise PublicPreparationError(
                "public Case provenance does not match the requested Suite"
            )
        if case.input.review_target.kind is not wire_contract.review_target_kind:
            raise PublicPreparationError(
                "public Case Target kind does not match the Suite wire contract"
            )
        path = "cases/%s.json" % case.task_id
        raw = canonical_json_bytes(case.to_dict())
        if len(raw) > MAX_EVAL_CASE_BYTES:
            raise PublicPreparationError("public Case exceeds its byte limit")
        total_case_bytes += len(raw)
        if total_case_bytes > MAX_SUITE_TOTAL_CASE_BYTES:
            raise PublicPreparationError(
                "public Suite Case bytes exceed the cumulative byte limit of %d"
                % MAX_SUITE_TOTAL_CASE_BYTES
            )
        binding = SuiteCase(
            task_id=case.task_id,
            case_version=case.case_version,
            path=path,
            split=prepared.split,
            protocol_id=prepared.protocol_id,
            dimensions=prepared.dimensions,
            raw_file_size_bytes=len(raw),
            raw_file_sha256=hashlib.sha256(raw).hexdigest(),
            canonical_case_digest=case.digest(),
            eval_input_digest=case.eval_input().digest(),
            truth_completeness=case.review_truth.completeness,
        )
        result.append(_PreparedCaseFile(prepared, path, raw, binding))
    return tuple(result)


def write_public_suite(
    output_root: os.PathLike[str] | str,
    *,
    suite_id: str,
    suite_version: str,
    adapter_id: str,
    adapter_version: str,
    source_manifest: PublicSourceManifest,
    filter_manifest: PublicFilterManifest,
    wire_contract: WireContractV2,
    cases: Iterable[PublicPreparedCase],
    actual_statistics: Iterable[PublicStatistic],
    records: Iterable[PublicRecordReceipt],
    extra_files: Optional[Mapping[str, bytes]] = None,
    expected_source_manifest_digest: Optional[str] = None,
    frozen_publication: Optional[PublicFrozenBundlePublication] = None,
) -> PublicPreparationResult:
    """Create and atomically publish one immutable canonical public Suite."""

    suite_id = _identifier(suite_id, "public Suite suite_id")
    suite_version = _identifier(suite_version, "public Suite suite_version")
    adapter_id = _identifier(adapter_id, "public Suite adapter_id")
    adapter_version = _identifier(adapter_version, "public Suite adapter_version")
    if type(wire_contract) is not WireContractV2:
        raise PublicPreparationError(
            "public Suite wire_contract must be a WireContractV2"
        )
    if not isinstance(source_manifest, PublicSourceManifest):
        raise PublicPreparationError("public Suite source manifest is invalid")
    if expected_source_manifest_digest is not None:
        verify_public_source_manifest_digest(
            source_manifest, expected_source_manifest_digest
        )
    if not isinstance(filter_manifest, PublicFilterManifest):
        raise PublicPreparationError("public Suite filter manifest is invalid")
    if source_manifest.dataset_id != filter_manifest.dataset_id:
        raise PublicPreparationError("public Suite source and filter dataset disagree")
    case_files = _preflight_public_cases(
        cases,
        suite_id=suite_id,
        source_manifest=source_manifest,
        wire_contract=wire_contract,
    )
    if frozen_publication is not None:
        if type(frozen_publication) is not PublicFrozenBundlePublication:
            raise PublicPreparationError(
                "frozen_publication must be PublicFrozenBundlePublication"
            )
        _validate_frozen_publication_inputs(
            frozen_publication,
            source_manifest=source_manifest,
            filter_manifest=filter_manifest,
            case_files=case_files,
        )
    manifest_cases = tuple(item.binding for item in case_files)
    case_bindings_digest = _case_bindings_digest(manifest_cases)
    statistics = _ordered_statistics(
        tuple(actual_statistics), "public Suite actual statistics"
    )
    try:
        record_receipts = _ordered_record_receipts(
            tuple(records), "public Suite record receipts"
        )
    except PublicDatasetError as exc:
        raise PublicPreparationError(str(exc)) from exc
    extras = _validate_extra_files({} if extra_files is None else extra_files)
    try:
        extra_bindings = _ordered_extra_files(
            tuple(
                PublicExtraFile.from_bytes(path, data)
                for path, data in extras.items()
            ),
            "public Suite extra files",
        )
    except PublicDatasetError as exc:
        raise PublicPreparationError(str(exc)) from exc
    source_digest = source_manifest.digest()
    filter_digest = filter_manifest.digest()
    packet_digest = _preparation_packet_digest(
        adapter_id=adapter_id,
        adapter_version=adapter_version,
        source_manifest_digest=source_digest,
        filter_manifest_digest=filter_digest,
        case_bindings_digest=case_bindings_digest,
        actual_statistics=statistics,
        records_digest=_records_digest(record_receipts),
        extra_files_digest=_extra_files_digest(extra_bindings),
    )
    preparation_binding = _preparation_binding(
        source_manifest=source_manifest,
        filter_manifest=filter_manifest,
        preparation_packet_digest=packet_digest,
        wire_contract=wire_contract,
        manifest_cases=manifest_cases,
        case_files=tuple(case_files),
        frozen_publication=frozen_publication,
    )

    manifest = SuiteManifest(
        schema_version=SUITE_MANIFEST_SCHEMA_VERSION,
        suite_id=suite_id,
        suite_version=suite_version,
        wire_contract=wire_contract,
        source=SuiteSource(
            kind=SuiteKind.PUBLIC,
            source_id=source_manifest.dataset_id,
            source_version=source_manifest.source_revision,
            source_uri=source_manifest.source_uri,
            license=source_manifest.license,
            content_hash=packet_digest,
            preparation_binding=preparation_binding,
        ),
        cases=manifest_cases,
    )
    manifest_bytes = canonical_json_bytes(manifest.to_dict())
    if len(manifest_bytes) > MAX_SUITE_MANIFEST_BYTES:
        raise PublicPreparationError(
            "public Suite manifest exceeds the byte limit of %d"
            % MAX_SUITE_MANIFEST_BYTES
        )
    receipt = PublicPreparationReceipt(
        schema_version=PUBLIC_PREPARATION_RECEIPT_SCHEMA_VERSION,
        adapter_id=adapter_id,
        adapter_version=adapter_version,
        source_manifest=source_manifest,
        source_manifest_digest=source_digest,
        filter_manifest=filter_manifest,
        filter_manifest_digest=filter_digest,
        case_bindings_digest=case_bindings_digest,
        suite_manifest_digest=manifest.digest(),
        actual_statistics=statistics,
        records=record_receipts,
        extra_files=extra_bindings,
    )
    receipt_bytes = canonical_json_bytes(receipt.to_dict())

    lexical_output = Path(os.path.abspath(os.fspath(output_root)))
    parent = _assert_publication_parent(lexical_output.parent)
    output = parent / lexical_output.name
    if os.path.lexists(output):
        raise PublicConflictError("public Suite output already exists")
    prepared_result = PublicPreparationResult(
        output,
        manifest,
        receipt,
        bundle_id=(
            None
            if frozen_publication is None
            else frozen_publication.bundle.manifest.bundle_id
        ),
    )
    staging = Path(
        tempfile.mkdtemp(prefix=".%s." % output.name, suffix=".staging", dir=parent)
    )
    staging_identity = _file_identity(os.lstat(str(staging)))
    published = False
    try:
        for item in case_files:
            _write_new(
                staging,
                item.path,
                item.raw,
                expected_root_identity=staging_identity,
            )
        for path, data in sorted(extras.items()):
            _write_new(
                staging,
                path,
                data,
                expected_root_identity=staging_identity,
            )
        _write_new(
            staging,
            "suite_manifest.json",
            manifest_bytes,
            expected_root_identity=staging_identity,
        )
        _write_new(
            staging,
            DEFAULT_PREPARATION_RECEIPT_PATH,
            receipt_bytes,
            expected_root_identity=staging_identity,
        )
        if frozen_publication is not None:
            _verify_staged_frozen_publication(
                staging,
                frozen_publication,
                case_files,
            )
        CaseBank.open(staging)
        loaded_receipt = read_public_preparation_receipt(staging)
        if loaded_receipt != receipt:
            raise PublicPreparationError(
                "public preparation receipt changed during publication"
            )

        _publish_directory_create_only(staging, output)
        published = True
        return prepared_result
    finally:
        if not published and os.path.lexists(staging):
            _cleanup_owned_staging(staging, parent, staging_identity)


def _verify_public_case_files(root: Path, manifest: SuiteManifest) -> None:
    for binding in manifest.cases:
        context = "public Case %s" % binding.task_id
        raw = _read_single_link_regular_file(
            root,
            binding.path,
            MAX_EVAL_CASE_BYTES,
            context,
        )
        if len(raw) != binding.raw_file_size_bytes:
            raise PublicSourceIntegrityError(
                "%s size does not match the Suite manifest" % context
            )
        if hashlib.sha256(raw).hexdigest() != binding.raw_file_sha256:
            raise PublicSourceIntegrityError(
                "%s hash does not match the Suite manifest" % context
            )
        try:
            case = EvalCase.from_json(raw)
            if canonical_json_bytes(case.to_dict()) != raw:
                raise PublicSourceIntegrityError(
                    "%s bytes are not canonical" % context
                )
            validate_case_for_manifest(case, binding, manifest)
        except PublicSourceIntegrityError:
            raise
        except SchemaError as exc:
            raise PublicSourceIntegrityError(
                "%s is malformed or inconsistent with the Suite manifest"
                % context
            ) from exc


def read_public_preparation_receipt(
    suite_root: os.PathLike[str] | str,
    *,
    expected_source_manifest_digest: Optional[str] = None,
    expected_preparation_packet_digest: Optional[str] = None,
    expected_suite_manifest_digest: Optional[str] = None,
) -> PublicPreparationReceipt:
    try:
        expected_packet_digest = (
            None
            if expected_preparation_packet_digest is None
            else _digest(
                expected_preparation_packet_digest,
                "expected public preparation packet digest",
            )
        )
        expected_manifest_digest = (
            None
            if expected_suite_manifest_digest is None
            else _digest(
                expected_suite_manifest_digest,
                "expected public Suite manifest digest",
            )
        )
    except SchemaError as exc:
        raise PublicSourceIntegrityError(str(exc)) from exc
    root = _coerce_suite_root(suite_root)
    raw = _read_single_link_regular_file(
        root,
        DEFAULT_PREPARATION_RECEIPT_PATH,
        MAX_PUBLIC_PREPARATION_RECEIPT_BYTES,
        "public preparation receipt",
    )
    try:
        receipt = PublicPreparationReceipt.from_json(raw)
    except SchemaError as exc:
        raise PublicSourceIntegrityError(
            "public preparation receipt is malformed or internally inconsistent"
        ) from exc
    if canonical_json_bytes(receipt.to_dict()) != raw:
        raise PublicSourceIntegrityError(
            "public preparation receipt bytes are not canonical"
        )
    if expected_source_manifest_digest is not None:
        verify_public_source_manifest_digest(
            receipt.source_manifest, expected_source_manifest_digest
        )
    if (
        expected_packet_digest is not None
        and receipt.preparation_packet_digest != expected_packet_digest
    ):
        raise PublicSourceIntegrityError(
            "public preparation packet digest does not match the expected digest"
        )
    manifest_raw = _read_single_link_regular_file(
        root,
        "suite_manifest.json",
        MAX_SUITE_MANIFEST_BYTES,
        "public Suite manifest",
    )
    try:
        manifest = SuiteManifest.from_json(manifest_raw)
    except SchemaError as exc:
        raise PublicSourceIntegrityError(
            "public Suite manifest is malformed"
        ) from exc
    if canonical_json_bytes(manifest.to_dict()) != manifest_raw:
        raise PublicSourceIntegrityError(
            "public Suite manifest bytes are not canonical"
        )
    manifest_digest = manifest.digest()
    if manifest_digest != receipt.suite_manifest_digest:
        raise PublicSourceIntegrityError(
            "public preparation receipt does not bind the Suite manifest"
        )
    if (
        expected_manifest_digest is not None
        and manifest_digest != expected_manifest_digest
    ):
        raise PublicSourceIntegrityError(
            "public Suite manifest digest does not match the expected digest"
        )
    case_bindings_digest = _case_bindings_digest(manifest.cases)
    if case_bindings_digest != receipt.case_bindings_digest:
        raise PublicSourceIntegrityError(
            "public Suite Case bindings do not match the preparation receipt"
        )
    expected_source = receipt.source_manifest
    if (
        manifest.source.kind is not SuiteKind.PUBLIC
        or manifest.source.source_id != expected_source.dataset_id
        or manifest.source.source_version != expected_source.source_revision
        or manifest.source.source_uri != expected_source.source_uri
        or manifest.source.license != expected_source.license
    ):
        raise PublicSourceIntegrityError(
            "public Suite source metadata does not bind the source manifest"
        )
    preparation = manifest.source.preparation_binding
    if preparation is None:
        raise PublicSourceIntegrityError(
            "public Suite is missing its preparation binding"
        )
    if (
        preparation.source_manifest_digest != receipt.source_manifest_digest
        or preparation.filter_manifest_digest != receipt.filter_manifest_digest
        or preparation.preparation_packet_digest
        != receipt.preparation_packet_digest
    ):
        raise PublicSourceIntegrityError(
            "public Suite preparation binding does not match the receipt"
        )
    if manifest.source.content_hash != receipt.preparation_packet_digest:
        raise PublicSourceIntegrityError(
            "public Suite source hash does not bind the preparation packet"
        )
    _verify_public_case_files(root, manifest)
    for binding in receipt.extra_files:
        try:
            extra_raw = _read_single_link_regular_file(
                root,
                binding.path,
                MAX_PUBLIC_EXTRA_FILE_BYTES,
                "public extra file %s" % binding.path,
            )
        except SchemaError as exc:
            raise PublicSourceIntegrityError(str(exc)) from exc
        if len(extra_raw) != binding.size_bytes:
            raise PublicSourceIntegrityError(
                "public extra file %s size does not match its receipt"
                % binding.path
            )
        if hashlib.sha256(extra_raw).hexdigest() != binding.sha256:
            raise PublicSourceIntegrityError(
                "public extra file %s hash does not match its receipt"
                % binding.path
            )
    return receipt


def source_file_from_path(
    root: os.PathLike[str] | str, *, role: str, path: str
) -> PublicSourceFile:
    """Create a file binding for an explicitly acquired local source tree."""

    verified_root = _coerce_suite_root(root)
    portable = _portable_case_path(path, "public source file.path")
    raw = _read_single_link_regular_file(
        verified_root,
        portable,
        MAX_PUBLIC_SOURCE_FILE_BYTES,
        "public source file %s" % role,
    )
    if not raw:
        raise PublicSourceIntegrityError("public source files may not be empty")
    return PublicSourceFile(
        role=_identifier(role, "public source file.role"),
        path=portable,
        size_bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


__all__ = [
    "PUBLIC_SOURCE_MANIFEST_SCHEMA_VERSION",
    "PUBLIC_FILTER_MANIFEST_SCHEMA_VERSION",
    "PUBLIC_PREPARATION_RECEIPT_SCHEMA_VERSION",
    "PUBLIC_PREPARATION_PACKET_SCHEMA_VERSION",
    "DEFAULT_PREPARATION_RECEIPT_PATH",
    "MAX_PUBLIC_SOURCE_MANIFEST_BYTES",
    "MAX_PUBLIC_FILTER_MANIFEST_BYTES",
    "MAX_PUBLIC_PREPARATION_RECEIPT_BYTES",
    "PublicDatasetError",
    "PublicSourceIntegrityError",
    "PublicFormatError",
    "PublicPreparationError",
    "PublicConflictError",
    "PublicPreconditionError",
    "PublicOperationalError",
    "PublicOptionalDependencyError",
    "verify_public_source_manifest_digest",
    "verify_public_filter_manifest_digest",
    "read_public_source_manifest",
    "read_public_filter_manifest",
    "PublicSourceFile",
    "PublicStatistic",
    "PublicSourceManifest",
    "PublicSelector",
    "PublicFilterManifest",
    "VerifiedPublicSource",
    "PublicExtraFile",
    "PublicRecordReceipt",
    "PublicPreparedCase",
    "PublicFrozenBundlePublication",
    "PublicPreparationReceipt",
    "PublicPreparationResult",
    "write_public_suite",
    "read_public_preparation_receipt",
    "source_file_from_path",
]
