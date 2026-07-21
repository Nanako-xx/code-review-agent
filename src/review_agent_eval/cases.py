"""Immutable Suite/Case protocols and truth-safe Case projections.

This module deliberately contains no dataset-specific parsing.  Dataset
adapters emit the same Suite Manifest v2 and EvalCase v2 documents; the core
loader validates those documents without knowing which adapter produced them.
"""

from __future__ import annotations

from bisect import bisect_left
import json
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, Optional, Sequence, Tuple
from urllib.parse import urlsplit

from .models import (
    EVAL_CASE_SCHEMA_VERSION,
    EVAL_INPUT_SCHEMA_VERSION,
    EVAL_SUBMISSION_SCHEMA_VERSION,
    MAX_COUNTER,
    MAX_EVAL_CASE_BYTES,
    MAX_IDENTIFIER_CHARS,
    MAX_REPOSITORY_PATH_CHARS,
    MAX_URL_CHARS,
    CaseOrigin,
    CaseSource,
    ClarificationPolicy,
    EvalCase,
    EvalInput,
    IntentAuthority,
    ReviewTargetKind,
    SchemaError,
    TruthCompleteness,
    UnsupportedProtocolVersionError,
    _JsonModel,
    _array,
    _digest,
    _enum_value,
    _exact_fields,
    _identifier,
    _integer,
    _model_tuple,
    _object,
    _require_enum,
    _safe_repo_path,
    _strict_json_loads,
    _string,
    canonical_sha256,
    stable_id,
)


SUITE_MANIFEST_SCHEMA_VERSION = "suite_manifest_v2"
RUN_CASE_SNAPSHOT_SCHEMA_VERSION = "eval_run_case_snapshot_v2"
PUBLIC_SUITE_PREPARATION_BINDING_SCHEMA_VERSION = (
    "public_suite_preparation_binding_v2"
)
REPOSITORY_MATERIALIZER_PROTOCOL = "repository-materializer-v2"
FROZEN_CONTEXT_MATERIALIZER_PROTOCOL = "frozen-context-materializer-v2"

MAX_SUITE_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_RUN_CASE_SNAPSHOT_BYTES = 256 * 1024 * 1024
MAX_SUITE_CASES = 65_536
MAX_SUITE_TOTAL_CASE_BYTES = 512 * 1024 * 1024
MAX_CASE_DIMENSIONS = 64

_RUN_CASE_SNAPSHOT_ID_RE = re.compile(r"^run-case-snapshot-[0-9a-f]{64}$")


class CaseSplit(str, Enum):
    """A fixed, manifest-owned Case partition."""

    TRAIN = "train"
    DEV = "dev"
    CAPABILITY = "capability"
    REGRESSION = "regression"
    HELD_OUT = "held_out"


class SuiteKind(str, Enum):
    """Provenance/access category, independent of any particular dataset."""

    CORE = "core"
    PUBLIC = "public"
    PRIVATE = "private"


_WINDOWS_RESERVED_NAMES = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        "clock$",
        *("com%d" % index for index in range(1, 10)),
        *("lpt%d" % index for index in range(1, 10)),
    }
)
_WINDOWS_FORBIDDEN_PATH_CHARS = frozenset('<>:"|?*')
_LOCAL_CASE_ORIGINS = frozenset(
    {CaseOrigin.HAND_AUTHORED, CaseOrigin.PRIVATE}
)


def _portable_key(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value)
    return unicodedata.normalize("NFC", normalized.casefold())


def validate_run_case_snapshot_id(
    value: Any, context: str = "run Case snapshot.snapshot_id"
) -> str:
    result = _identifier(value, context)
    if _RUN_CASE_SNAPSHOT_ID_RE.fullmatch(result) is None:
        raise SchemaError("%s is not canonical" % context)
    return result


def _portable_case_path(value: Any, context: str) -> str:
    path = _safe_repo_path(value, context)
    for component in path.split("/"):
        if component.endswith((" ", ".")):
            raise SchemaError(
                "%s contains a component that is not portable across filesystems"
                % context
            )
        if any(character in _WINDOWS_FORBIDDEN_PATH_CHARS for character in component):
            raise SchemaError(
                "%s contains a character that is unsafe on Windows" % context
            )
        stem = unicodedata.normalize(
            "NFKC", component.split(".", 1)[0]
        ).casefold()
        if stem in _WINDOWS_RESERVED_NAMES:
            raise SchemaError(
                "%s contains a reserved portable path component" % context
            )
    return path


def _source_uri(value: Any, context: str) -> Optional[str]:
    if value is None:
        return None
    uri = _string(value, context, MAX_URL_CHARS)
    if any(character.isspace() or ord(character) < 32 for character in uri):
        raise SchemaError("%s may not contain whitespace or controls" % context)
    try:
        parsed = urlsplit(uri)
        username = parsed.username
        password = parsed.password
    except ValueError as exc:
        raise SchemaError("%s is not a valid absolute URI" % context) from exc
    if not parsed.scheme:
        raise SchemaError("%s must be an absolute URI" % context)
    if parsed.scheme.casefold() in {"http", "https"} and not parsed.netloc:
        raise SchemaError("%s must include a host" % context)
    if username is not None or password is not None:
        raise SchemaError("%s may not contain userinfo or credentials" % context)
    return uri


def _license_name(value: Any, context: str) -> Optional[str]:
    if value is None:
        return None
    result = _string(value, context, MAX_IDENTIFIER_CHARS)
    if result != result.strip() or any(
        ord(character) < 32 or ord(character) == 127 for character in result
    ):
        raise SchemaError(
            "%s may not contain leading/trailing whitespace or controls" % context
        )
    return result


def _require_protocol_version(actual: Any, expected: str, context: str) -> str:
    if type(actual) is not str:
        raise SchemaError("%s must be a string" % context)
    if actual != expected:
        raise UnsupportedProtocolVersionError(expected=expected, actual=actual)
    return actual


def _json_size_default(value: Any) -> Any:
    if isinstance(value, _JsonModel):
        return value.to_dict()
    if isinstance(value, Enum):
        return value.value
    raise TypeError("value is not JSON serializable")


def _check_raw_payload_size(value: Any, maximum: int, context: str) -> None:
    """Measure compact JSON bytes incrementally without sorting or hashing."""

    encoder = json.JSONEncoder(
        ensure_ascii=False,
        allow_nan=False,
        check_circular=True,
        separators=(",", ":"),
        sort_keys=False,
        default=_json_size_default,
    )
    size = 0
    try:
        for chunk in encoder.iterencode(value):
            size += len(chunk.encode("utf-8", "strict"))
            if size > maximum:
                raise SchemaError(
                    "%s exceeds the canonical byte limit of %d"
                    % (context, maximum)
                )
    except SchemaError:
        raise
    except (
        TypeError,
        ValueError,
        UnicodeError,
        OverflowError,
        RecursionError,
    ) as exc:
        raise SchemaError("%s is not a valid bounded JSON payload" % context) from exc


@dataclass(frozen=True)
class WireContractV2(_JsonModel):
    """The complete Case/Input/Submission/Target contract for one Suite."""

    case_schema_version: str
    input_schema_version: str
    submission_schema_version: str
    review_target_kind: ReviewTargetKind
    materializer_protocol: str

    def __post_init__(self) -> None:
        _require_protocol_version(
            self.case_schema_version,
            EVAL_CASE_SCHEMA_VERSION,
            "wire contract.case_schema_version",
        )
        _require_protocol_version(
            self.input_schema_version,
            EVAL_INPUT_SCHEMA_VERSION,
            "wire contract.input_schema_version",
        )
        _require_protocol_version(
            self.submission_schema_version,
            EVAL_SUBMISSION_SCHEMA_VERSION,
            "wire contract.submission_schema_version",
        )
        _require_enum(
            ReviewTargetKind,
            self.review_target_kind,
            "wire contract.review_target_kind",
        )
        protocol = _identifier(
            self.materializer_protocol,
            "wire contract.materializer_protocol",
        )
        expected = {
            ReviewTargetKind.REPOSITORY: REPOSITORY_MATERIALIZER_PROTOCOL,
            ReviewTargetKind.FROZEN_CONTEXT: FROZEN_CONTEXT_MATERIALIZER_PROTOCOL,
        }[self.review_target_kind]
        if protocol != expected:
            raise SchemaError(
                "wire contract target kind and materializer protocol mismatch"
            )

    @classmethod
    def from_dict(cls, value: Any) -> "WireContractV2":
        payload = _object(value, "wire contract")
        for field_name, expected in (
            ("case_schema_version", EVAL_CASE_SCHEMA_VERSION),
            ("input_schema_version", EVAL_INPUT_SCHEMA_VERSION),
            ("submission_schema_version", EVAL_SUBMISSION_SCHEMA_VERSION),
        ):
            if field_name in payload:
                _require_protocol_version(
                    payload[field_name], expected, "wire contract.%s" % field_name
                )
        _exact_fields(
            payload,
            (
                "case_schema_version",
                "input_schema_version",
                "submission_schema_version",
                "review_target_kind",
                "materializer_protocol",
            ),
            "wire contract",
        )
        return cls(
            case_schema_version=_require_protocol_version(
                payload["case_schema_version"],
                EVAL_CASE_SCHEMA_VERSION,
                "wire contract.case_schema_version",
            ),
            input_schema_version=_require_protocol_version(
                payload["input_schema_version"],
                EVAL_INPUT_SCHEMA_VERSION,
                "wire contract.input_schema_version",
            ),
            submission_schema_version=_require_protocol_version(
                payload["submission_schema_version"],
                EVAL_SUBMISSION_SCHEMA_VERSION,
                "wire contract.submission_schema_version",
            ),
            review_target_kind=_enum_value(
                ReviewTargetKind,
                payload["review_target_kind"],
                "wire contract.review_target_kind",
            ),
            materializer_protocol=_identifier(
                payload["materializer_protocol"],
                "wire contract.materializer_protocol",
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_schema_version": self.case_schema_version,
            "input_schema_version": self.input_schema_version,
            "submission_schema_version": self.submission_schema_version,
            "review_target_kind": self.review_target_kind.value,
            "materializer_protocol": self.materializer_protocol,
        }


@dataclass(frozen=True)
class PublicSuitePreparationBindingV2(_JsonModel):
    """Evaluator-private trust roots for one prepared public Suite."""

    schema_version: str
    source_catalog_digest: str
    acquisition_receipt_digest: str
    source_manifest_digest: str
    filter_manifest_digest: str
    preparation_packet_digest: str
    repository_catalog_digest: Optional[str]
    frozen_bundle_trust_digest: Optional[str]

    def __post_init__(self) -> None:
        _require_protocol_version(
            self.schema_version,
            PUBLIC_SUITE_PREPARATION_BINDING_SCHEMA_VERSION,
            "public preparation binding.schema_version",
        )
        for name in (
            "source_catalog_digest",
            "acquisition_receipt_digest",
            "source_manifest_digest",
            "filter_manifest_digest",
            "preparation_packet_digest",
        ):
            _digest(getattr(self, name), "public preparation binding.%s" % name)
        if self.repository_catalog_digest is not None:
            _digest(
                self.repository_catalog_digest,
                "public preparation binding.repository_catalog_digest",
            )
        if self.frozen_bundle_trust_digest is not None:
            _digest(
                self.frozen_bundle_trust_digest,
                "public preparation binding.frozen_bundle_trust_digest",
            )

    @classmethod
    def from_dict(cls, value: Any) -> "PublicSuitePreparationBindingV2":
        payload = _object(value, "public preparation binding")
        if "schema_version" in payload:
            _require_protocol_version(
                payload["schema_version"],
                PUBLIC_SUITE_PREPARATION_BINDING_SCHEMA_VERSION,
                "public preparation binding.schema_version",
            )
        fields = (
            "schema_version",
            "source_catalog_digest",
            "acquisition_receipt_digest",
            "source_manifest_digest",
            "filter_manifest_digest",
            "preparation_packet_digest",
            "repository_catalog_digest",
            "frozen_bundle_trust_digest",
        )
        _exact_fields(payload, fields, "public preparation binding")
        return cls(
            schema_version=_require_protocol_version(
                payload["schema_version"],
                PUBLIC_SUITE_PREPARATION_BINDING_SCHEMA_VERSION,
                "public preparation binding.schema_version",
            ),
            source_catalog_digest=_digest(
                payload["source_catalog_digest"],
                "public preparation binding.source_catalog_digest",
            ),
            acquisition_receipt_digest=_digest(
                payload["acquisition_receipt_digest"],
                "public preparation binding.acquisition_receipt_digest",
            ),
            source_manifest_digest=_digest(
                payload["source_manifest_digest"],
                "public preparation binding.source_manifest_digest",
            ),
            filter_manifest_digest=_digest(
                payload["filter_manifest_digest"],
                "public preparation binding.filter_manifest_digest",
            ),
            preparation_packet_digest=_digest(
                payload["preparation_packet_digest"],
                "public preparation binding.preparation_packet_digest",
            ),
            repository_catalog_digest=(
                None
                if payload["repository_catalog_digest"] is None
                else _digest(
                    payload["repository_catalog_digest"],
                    "public preparation binding.repository_catalog_digest",
                )
            ),
            frozen_bundle_trust_digest=(
                None
                if payload["frozen_bundle_trust_digest"] is None
                else _digest(
                    payload["frozen_bundle_trust_digest"],
                    "public preparation binding.frozen_bundle_trust_digest",
                )
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_catalog_digest": self.source_catalog_digest,
            "acquisition_receipt_digest": self.acquisition_receipt_digest,
            "source_manifest_digest": self.source_manifest_digest,
            "filter_manifest_digest": self.filter_manifest_digest,
            "preparation_packet_digest": self.preparation_packet_digest,
            "repository_catalog_digest": self.repository_catalog_digest,
            "frozen_bundle_trust_digest": self.frozen_bundle_trust_digest,
        }


@dataclass(frozen=True)
class SuiteSource(_JsonModel):
    """Versioned provenance shared by all Cases in one Suite manifest."""

    kind: SuiteKind
    source_id: str
    source_version: str
    source_uri: Optional[str]
    license: Optional[str]
    content_hash: str
    preparation_binding: Optional[PublicSuitePreparationBindingV2] = None

    def __post_init__(self) -> None:
        _require_enum(SuiteKind, self.kind, "suite source.kind")
        _identifier(self.source_id, "suite source.source_id")
        _identifier(self.source_version, "suite source.source_version")
        _source_uri(self.source_uri, "suite source.source_uri")
        _license_name(self.license, "suite source.license")
        _digest(self.content_hash, "suite source.content_hash")
        if self.kind is SuiteKind.PUBLIC:
            if self.source_uri is None:
                raise SchemaError("public suite source requires source_uri")
            if self.license is None:
                raise SchemaError("public suite source requires license")
            if not isinstance(
                self.preparation_binding, PublicSuitePreparationBindingV2
            ):
                raise SchemaError(
                    "public suite source requires preparation_binding"
                )
        elif self.preparation_binding is not None:
            raise SchemaError(
                "core/private suite source requires preparation_binding=null"
            )

    @classmethod
    def from_dict(cls, value: Any) -> "SuiteSource":
        payload = _object(value, "suite source")
        if payload.get("preparation_binding") is not None:
            binding_payload = _object(
                payload["preparation_binding"],
                "suite source.preparation_binding",
            )
            if "schema_version" in binding_payload:
                _require_protocol_version(
                    binding_payload["schema_version"],
                    PUBLIC_SUITE_PREPARATION_BINDING_SCHEMA_VERSION,
                    "public preparation binding.schema_version",
                )
        _exact_fields(
            payload,
            (
                "kind",
                "source_id",
                "source_version",
                "source_uri",
                "license",
                "content_hash",
                "preparation_binding",
            ),
            "suite source",
        )
        return cls(
            kind=_enum_value(SuiteKind, payload["kind"], "suite source.kind"),
            source_id=_identifier(
                payload["source_id"], "suite source.source_id"
            ),
            source_version=_identifier(
                payload["source_version"], "suite source.source_version"
            ),
            source_uri=_source_uri(
                payload["source_uri"], "suite source.source_uri"
            ),
            license=_license_name(payload["license"], "suite source.license"),
            content_hash=_digest(
                payload["content_hash"], "suite source.content_hash"
            ),
            preparation_binding=(
                None
                if payload["preparation_binding"] is None
                else PublicSuitePreparationBindingV2.from_dict(
                    payload["preparation_binding"]
                )
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "source_id": self.source_id,
            "source_version": self.source_version,
            "source_uri": self.source_uri,
            "license": self.license,
            "content_hash": self.content_hash,
            "preparation_binding": (
                None
                if self.preparation_binding is None
                else self.preparation_binding.to_dict()
            ),
        }


@dataclass(frozen=True)
class CaseDimension(_JsonModel):
    """One generic, evaluator-only grouping dimension for a Case."""

    name: str
    value: str

    def __post_init__(self) -> None:
        name = _identifier(self.name, "case dimension.name")
        if name != name.casefold() or not all(
            character.isascii()
            and (character.islower() or character.isdigit() or character in "_.-")
            for character in name
        ):
            raise SchemaError(
                "case dimension.name must be a lowercase ASCII grouping key"
            )
        if not name[0].isalpha():
            raise SchemaError("case dimension.name must start with a letter")
        dimension_value = _string(
            self.value, "case dimension.value", MAX_IDENTIFIER_CHARS
        )
        if dimension_value != dimension_value.strip() or any(
            ord(character) < 32 or ord(character) == 127
            for character in dimension_value
        ):
            raise SchemaError(
                "case dimension.value may not contain edge whitespace or controls"
            )

    @classmethod
    def from_dict(cls, value: Any) -> "CaseDimension":
        payload = _object(value, "case dimension")
        _exact_fields(payload, ("name", "value"), "case dimension")
        return cls(
            name=_identifier(payload["name"], "case dimension.name"),
            value=_string(
                payload["value"],
                "case dimension.value",
                MAX_IDENTIFIER_CHARS,
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "value": self.value}


@dataclass(frozen=True)
class SuiteCase(_JsonModel):
    """One immutable Case binding in Suite Manifest v2."""

    task_id: str
    case_version: int
    path: str
    split: CaseSplit
    protocol_id: str
    dimensions: Tuple[CaseDimension, ...]
    raw_file_size_bytes: int
    raw_file_sha256: str
    canonical_case_digest: str
    eval_input_digest: str
    truth_completeness: TruthCompleteness

    def __post_init__(self) -> None:
        _identifier(self.task_id, "suite case.task_id")
        _integer(
            self.case_version,
            "suite case.case_version",
            minimum=1,
            maximum=MAX_COUNTER,
        )
        _portable_case_path(self.path, "suite case.path")
        _require_enum(CaseSplit, self.split, "suite case.split")
        _identifier(self.protocol_id, "suite case.protocol_id")
        dimensions = _model_tuple(
            self.dimensions,
            CaseDimension,
            "suite case.dimensions",
            MAX_CASE_DIMENSIONS,
        )
        names = set()
        for item in dimensions:
            key = _portable_key(item.name)
            if key in names:
                raise SchemaError(
                    "suite case.dimensions contains a duplicate grouping key"
                )
            names.add(key)
        _integer(
            self.raw_file_size_bytes,
            "suite case.raw_file_size_bytes",
            minimum=1,
            maximum=MAX_EVAL_CASE_BYTES,
        )
        _digest(self.raw_file_sha256, "suite case.raw_file_sha256")
        _digest(
            self.canonical_case_digest,
            "suite case.canonical_case_digest",
        )
        _digest(self.eval_input_digest, "suite case.eval_input_digest")
        _require_enum(
            TruthCompleteness,
            self.truth_completeness,
            "suite case.truth_completeness",
        )
        object.__setattr__(
            self,
            "dimensions",
            tuple(sorted(dimensions, key=lambda item: item.name)),
        )

    @classmethod
    def from_dict(cls, value: Any) -> "SuiteCase":
        payload = _object(value, "suite case")
        _exact_fields(
            payload,
            (
                "task_id",
                "case_version",
                "path",
                "split",
                "protocol_id",
                "dimensions",
                "raw_file_size_bytes",
                "raw_file_sha256",
                "canonical_case_digest",
                "eval_input_digest",
                "truth_completeness",
            ),
            "suite case",
        )
        raw_dimensions = _array(
            payload["dimensions"],
            "suite case.dimensions",
            MAX_CASE_DIMENSIONS,
        )
        return cls(
            task_id=_identifier(payload["task_id"], "suite case.task_id"),
            case_version=_integer(
                payload["case_version"],
                "suite case.case_version",
                minimum=1,
                maximum=MAX_COUNTER,
            ),
            path=_portable_case_path(payload["path"], "suite case.path"),
            split=_enum_value(CaseSplit, payload["split"], "suite case.split"),
            protocol_id=_identifier(
                payload["protocol_id"], "suite case.protocol_id"
            ),
            dimensions=tuple(
                CaseDimension.from_dict(item) for item in raw_dimensions
            ),
            raw_file_size_bytes=_integer(
                payload["raw_file_size_bytes"],
                "suite case.raw_file_size_bytes",
                minimum=1,
                maximum=MAX_EVAL_CASE_BYTES,
            ),
            raw_file_sha256=_digest(
                payload["raw_file_sha256"], "suite case.raw_file_sha256"
            ),
            canonical_case_digest=_digest(
                payload["canonical_case_digest"],
                "suite case.canonical_case_digest",
            ),
            eval_input_digest=_digest(
                payload["eval_input_digest"], "suite case.eval_input_digest"
            ),
            truth_completeness=_enum_value(
                TruthCompleteness,
                payload["truth_completeness"],
                "suite case.truth_completeness",
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "case_version": self.case_version,
            "path": self.path,
            "split": self.split.value,
            "protocol_id": self.protocol_id,
            "dimensions": [item.to_dict() for item in self.dimensions],
            "raw_file_size_bytes": self.raw_file_size_bytes,
            "raw_file_sha256": self.raw_file_sha256,
            "canonical_case_digest": self.canonical_case_digest,
            "eval_input_digest": self.eval_input_digest,
            "truth_completeness": self.truth_completeness.value,
        }


@dataclass(frozen=True)
class SuiteManifest(_JsonModel):
    """Canonical, immutable inventory for one versioned Case Suite."""

    SCHEMA_VERSION: ClassVar[str] = SUITE_MANIFEST_SCHEMA_VERSION

    schema_version: str
    suite_id: str
    suite_version: str
    wire_contract: WireContractV2
    source: SuiteSource
    cases: Tuple[SuiteCase, ...]
    _task_ids: Tuple[str, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _require_protocol_version(
            self.schema_version,
            self.SCHEMA_VERSION,
            "SuiteManifest.schema_version",
        )
        _identifier(self.suite_id, "suite manifest.suite_id")
        _identifier(self.suite_version, "suite manifest.suite_version")
        if not isinstance(self.wire_contract, WireContractV2):
            raise SchemaError(
                "suite manifest.wire_contract must be a WireContractV2"
            )
        if not isinstance(self.source, SuiteSource):
            raise SchemaError("suite manifest.source must be a SuiteSource")
        binding = self.source.preparation_binding
        if self.source.kind is SuiteKind.PUBLIC:
            if not isinstance(binding, PublicSuitePreparationBindingV2):
                raise SchemaError(
                    "public Suite requires a complete preparation binding"
                )
            if self.wire_contract.review_target_kind is ReviewTargetKind.REPOSITORY:
                if (
                    binding.repository_catalog_digest is None
                    or binding.frozen_bundle_trust_digest is not None
                ):
                    raise SchemaError(
                        "Repository Suite preparation binding requires only repository_catalog_digest"
                    )
            elif (
                binding.frozen_bundle_trust_digest is None
                or binding.repository_catalog_digest is not None
            ):
                raise SchemaError(
                    "Frozen Suite preparation binding requires only frozen_bundle_trust_digest"
                )
        cases = _model_tuple(
            self.cases, SuiteCase, "suite manifest.cases", MAX_SUITE_CASES
        )
        if not cases:
            raise SchemaError("suite manifest.cases must contain at least one Case")
        _check_raw_payload_size(
            {
                "schema_version": self.schema_version,
                "suite_id": self.suite_id,
                "suite_version": self.suite_version,
                "wire_contract": self.wire_contract,
                "source": self.source,
                "cases": cases,
            },
            MAX_SUITE_MANIFEST_BYTES,
            "SuiteManifest",
        )

        task_ids = set()
        portable_task_ids = set()
        paths = set()
        portable_paths = set()
        total_case_bytes = 0
        for item in cases:
            if item.task_id in task_ids:
                raise SchemaError(
                    "suite manifest contains duplicate task_id %r" % item.task_id
                )
            task_ids.add(item.task_id)
            task_key = _portable_key(item.task_id)
            if task_key in portable_task_ids:
                raise SchemaError(
                    "suite manifest contains a portable task_id case collision"
                )
            portable_task_ids.add(task_key)

            if item.path in paths:
                raise SchemaError(
                    "suite manifest contains duplicate Case path %r" % item.path
                )
            paths.add(item.path)
            path_key = _portable_key(item.path)
            if path_key in portable_paths:
                raise SchemaError(
                    "suite manifest contains a portable Case path case collision"
                )
            portable_paths.add(path_key)
            total_case_bytes += item.raw_file_size_bytes
            if total_case_bytes > MAX_SUITE_TOTAL_CASE_BYTES:
                raise SchemaError(
                    "suite manifest Case bytes exceed the cumulative limit of %d"
                    % MAX_SUITE_TOTAL_CASE_BYTES
                )

        ordered = tuple(sorted(cases, key=lambda item: item.task_id))
        object.__setattr__(self, "cases", ordered)
        object.__setattr__(
            self,
            "_task_ids",
            tuple(item.task_id for item in ordered),
        )

    @classmethod
    def from_dict(cls, value: Any) -> "SuiteManifest":
        payload = _object(value, "SuiteManifest")
        if "schema_version" not in payload:
            raise SchemaError("SuiteManifest has missing field(s): schema_version")
        _require_protocol_version(
            payload["schema_version"],
            cls.SCHEMA_VERSION,
            "SuiteManifest.schema_version",
        )
        _exact_fields(
            payload,
            (
                "schema_version",
                "suite_id",
                "suite_version",
                "wire_contract",
                "source",
                "cases",
            ),
            "SuiteManifest",
        )
        wire_contract = WireContractV2.from_dict(payload["wire_contract"])
        suite_source = SuiteSource.from_dict(payload["source"])
        raw_cases = _array(
            payload["cases"], "suite manifest.cases", MAX_SUITE_CASES
        )
        _check_raw_payload_size(
            payload,
            MAX_SUITE_MANIFEST_BYTES,
            "SuiteManifest",
        )
        total_case_bytes = 0
        for index, raw_case in enumerate(raw_cases):
            case_payload = _object(
                raw_case, "suite manifest.cases[%d]" % index
            )
            if "raw_file_size_bytes" in case_payload:
                total_case_bytes += _integer(
                    case_payload["raw_file_size_bytes"],
                    "suite manifest.cases[%d].raw_file_size_bytes" % index,
                    minimum=1,
                    maximum=MAX_EVAL_CASE_BYTES,
                )
                if total_case_bytes > MAX_SUITE_TOTAL_CASE_BYTES:
                    raise SchemaError(
                        "suite manifest Case bytes exceed the cumulative limit of %d"
                        % MAX_SUITE_TOTAL_CASE_BYTES
                    )
        return cls(
            schema_version=payload["schema_version"],
            suite_id=_identifier(payload["suite_id"], "suite manifest.suite_id"),
            suite_version=_identifier(
                payload["suite_version"], "suite manifest.suite_version"
            ),
            wire_contract=wire_contract,
            source=suite_source,
            cases=tuple(SuiteCase.from_dict(item) for item in raw_cases),
        )

    @classmethod
    def from_json(cls, data: Any) -> "SuiteManifest":
        return cls.from_dict(
            _strict_json_loads(
                data, MAX_SUITE_MANIFEST_BYTES, "SuiteManifest JSON"
            )
        )

    def case_index(self, task_id: str) -> int:
        wanted = _identifier(task_id, "task_id")
        index = bisect_left(self._task_ids, wanted)
        if index < len(self._task_ids) and self._task_ids[index] == wanted:
            return index
        raise SchemaError("suite manifest has no task_id %r" % wanted)

    def case(self, task_id: str) -> SuiteCase:
        return self.cases[self.case_index(task_id)]

    @property
    def license(self) -> Optional[str]:
        return self.source.license

    @property
    def source_content_hash(self) -> str:
        return self.source.content_hash

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "suite_id": self.suite_id,
            "suite_version": self.suite_version,
            "wire_contract": self.wire_contract.to_dict(),
            "source": self.source.to_dict(),
            "cases": [item.to_dict() for item in self.cases],
        }


def _validate_intent_contract(case: EvalCase) -> None:
    truth = case.intent_truth
    if not truth.scorable:
        # EvalCase v2 already enforces the one canonical unscorable shape.
        return

    if (
        truth.authority is IntentAuthority.LINKED_REQUIREMENT
        and not getattr(
            case.input.review_target, "review_request", None
        )
    ):
        raise SchemaError(
            "linked_requirement intent authority requires a Repository review request"
        )
    if truth.authority is IntentAuthority.LINKED_REQUIREMENT:
        review_request = case.input.review_target.review_request
        if not review_request.linked_requirements:
            raise SchemaError(
                "linked_requirement intent authority requires linked_requirements input"
            )
    if (
        truth.clarification_policy is ClarificationPolicy.REQUIRED
        and not case.clarification_script.answers
    ):
        raise SchemaError(
            "required clarification policy requires at least one scripted answer"
        )


def validate_case_for_manifest(
    case: EvalCase,
    manifest_case: SuiteCase,
    manifest: SuiteManifest,
) -> None:
    """Validate all canonical and provenance bindings for one loaded Case."""

    if not isinstance(case, EvalCase):
        raise SchemaError("case must be an EvalCase")
    if not isinstance(manifest_case, SuiteCase):
        raise SchemaError("manifest_case must be a SuiteCase")
    if not isinstance(manifest, SuiteManifest):
        raise SchemaError("manifest must be a SuiteManifest")
    try:
        expected_entry = manifest.case(manifest_case.task_id)
    except SchemaError as exc:
        raise SchemaError("Case has no binding in the Suite manifest") from exc
    if expected_entry != manifest_case:
        raise SchemaError("Case manifest binding does not match the Suite manifest")
    if case.task_id != manifest_case.task_id:
        raise SchemaError("loaded Case task_id does not match its manifest task_id")
    if case.case_version != manifest_case.case_version:
        raise SchemaError("loaded Case version does not match its manifest version")
    if case.schema_version != manifest.wire_contract.case_schema_version:
        raise UnsupportedProtocolVersionError(
            expected=manifest.wire_contract.case_schema_version,
            actual=case.schema_version,
        )
    eval_input = case.eval_input()
    if eval_input.schema_version != manifest.wire_contract.input_schema_version:
        raise UnsupportedProtocolVersionError(
            expected=manifest.wire_contract.input_schema_version,
            actual=eval_input.schema_version,
        )
    if (
        eval_input.review_target.kind
        is not manifest.wire_contract.review_target_kind
    ):
        raise SchemaError(
            "loaded Case target kind does not match the Suite wire contract"
        )
    if case.review_truth.completeness is not manifest_case.truth_completeness:
        raise SchemaError(
            "loaded Case truth completeness does not match its manifest binding"
        )

    _validate_case_source_for_manifest(case.source, manifest)
    _validate_intent_contract(case)
    if canonical_sha256(case) != manifest_case.canonical_case_digest:
        raise SchemaError(
            "loaded Case canonical Case digest does not match its manifest binding"
        )
    if eval_input.digest() != manifest_case.eval_input_digest:
        raise SchemaError(
            "loaded Case EvalInput digest does not match its manifest binding"
        )


def _validate_case_source_for_manifest(
    source: CaseSource, manifest: SuiteManifest
) -> None:
    if not isinstance(source, CaseSource):
        raise SchemaError("Case source must be a CaseSource")
    if source.suite != manifest.suite_id:
        raise SchemaError("loaded Case source suite does not match the Suite manifest")
    if source.source_version != manifest.source.source_version:
        raise SchemaError(
            "loaded Case source version does not match the Suite source version"
        )
    if source.source_uri is not None:
        _source_uri(source.source_uri, "Case source.source_uri")
    if source.license is not None:
        _license_name(source.license, "Case source.license")

    # The policy is generic: all origins other than local hand-authored/private
    # provenance are public.  The core loader never branches on dataset names.
    origin_is_external = source.origin not in _LOCAL_CASE_ORIGINS
    if origin_is_external and manifest.source.kind is not SuiteKind.PUBLIC:
        raise SchemaError("externally sourced Case requires a public Suite source")
    if (
        source.origin is CaseOrigin.PRIVATE
        and manifest.source.kind is not SuiteKind.PRIVATE
    ):
        raise SchemaError("private Case source requires a private Suite source")
    if manifest.source.kind is SuiteKind.PUBLIC or origin_is_external:
        if source.source_uri is None:
            raise SchemaError("public Case source requires source_uri")
        if source.license is None:
            raise SchemaError("public Case source requires license")
        if source.license != manifest.source.license:
            raise SchemaError(
                "public Case license does not match the Suite source license"
            )


@dataclass(frozen=True)
class AgentCaseView(_JsonModel):
    """An Agent-safe type whose only payload is canonical EvalInput v2."""

    input: EvalInput

    def __post_init__(self) -> None:
        if not isinstance(self.input, EvalInput):
            raise SchemaError("AgentCaseView.input must be an EvalInput")

    @classmethod
    def from_case(cls, case: EvalCase) -> "AgentCaseView":
        if not isinstance(case, EvalCase):
            raise SchemaError("AgentCaseView requires an EvalCase")
        return cls(input=case.eval_input())

    @classmethod
    def from_dict(cls, value: Any) -> "AgentCaseView":
        return cls(input=EvalInput.from_dict(value))

    @classmethod
    def from_json(cls, data: Any) -> "AgentCaseView":
        return cls(input=EvalInput.from_json(data))

    @property
    def schema_version(self) -> str:
        return self.input.schema_version

    @property
    def task_id(self) -> str:
        return self.input.task_id

    @property
    def review_target(self):
        return self.input.review_target

    @property
    def repository(self):
        repository = getattr(self.input.review_target, "repository", None)
        if repository is None:
            raise SchemaError("Frozen EvalInput has no repository")
        return repository

    @property
    def review_request(self):
        review_request = getattr(
            self.input.review_target, "review_request", None
        )
        if review_request is None:
            raise SchemaError("Frozen EvalInput has no review_request")
        return review_request

    def to_dict(self) -> Dict[str, Any]:
        # Do not add a wrapper schema here: the serialized Agent boundary is
        # exactly EvalInput v2, and can never contain private Case fields.
        return self.input.to_dict()


@dataclass(frozen=True)
class RunCaseSnapshotEntry(_JsonModel):
    """Truth-free immutable binding captured after a verified Case read."""

    manifest_case: SuiteCase
    source: CaseSource
    input: EvalInput

    def __post_init__(self) -> None:
        if not isinstance(self.manifest_case, SuiteCase):
            raise SchemaError(
                "run Case snapshot manifest_case must be a SuiteCase"
            )
        if not isinstance(self.source, CaseSource):
            raise SchemaError("run Case snapshot source must be a CaseSource")
        if not isinstance(self.input, EvalInput):
            raise SchemaError("run Case snapshot input must be an EvalInput")
        if self.input.task_id != self.manifest_case.task_id:
            raise SchemaError("run Case snapshot task_id binding is inconsistent")
        if self.input.digest() != self.manifest_case.eval_input_digest:
            raise SchemaError(
                "run Case snapshot input digest does not match its manifest binding"
            )

    @classmethod
    def from_verified_case(
        cls,
        manifest_case: SuiteCase,
        case: EvalCase,
        manifest: SuiteManifest,
    ) -> "RunCaseSnapshotEntry":
        validate_case_for_manifest(case, manifest_case, manifest)
        return cls(
            manifest_case=manifest_case,
            source=case.source,
            input=case.eval_input(),
        )

    @classmethod
    def from_dict(cls, value: Any) -> "RunCaseSnapshotEntry":
        payload = _object(value, "run Case snapshot entry")
        _exact_fields(
            payload,
            ("manifest_case", "source", "input"),
            "run Case snapshot entry",
        )
        return cls(
            manifest_case=SuiteCase.from_dict(payload["manifest_case"]),
            source=CaseSource.from_dict(payload["source"]),
            input=EvalInput.from_dict(payload["input"]),
        )

    @property
    def task_id(self) -> str:
        return self.manifest_case.task_id

    @property
    def split(self) -> CaseSplit:
        return self.manifest_case.split

    @property
    def raw_file_sha256(self) -> str:
        return self.manifest_case.raw_file_sha256

    @property
    def canonical_case_digest(self) -> str:
        return self.manifest_case.canonical_case_digest

    @property
    def case_source_provenance_hash(self) -> str:
        return self.source.content_hash

    def agent_view(self) -> AgentCaseView:
        return AgentCaseView(input=self.input)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest_case": self.manifest_case.to_dict(),
            "source": self.source.to_dict(),
            "input": self.input.to_dict(),
        }


def _snapshot_id(
    manifest: SuiteManifest, entries: Sequence[RunCaseSnapshotEntry]
) -> str:
    return stable_id(
        "run-case-snapshot",
        RUN_CASE_SNAPSHOT_SCHEMA_VERSION,
        manifest.digest(),
        manifest.wire_contract.to_dict(),
        [item.to_dict() for item in entries],
    )


@dataclass(frozen=True)
class RunCaseSnapshot(_JsonModel):
    """The complete immutable Case selection used by one Eval Run."""

    SCHEMA_VERSION: ClassVar[str] = RUN_CASE_SNAPSHOT_SCHEMA_VERSION

    schema_version: str
    snapshot_id: str
    manifest: SuiteManifest
    wire_contract: WireContractV2
    cases: Tuple[RunCaseSnapshotEntry, ...]

    def __post_init__(self) -> None:
        _require_protocol_version(
            self.schema_version,
            self.SCHEMA_VERSION,
            "RunCaseSnapshot.schema_version",
        )
        validate_run_case_snapshot_id(self.snapshot_id)
        if not isinstance(self.manifest, SuiteManifest):
            raise SchemaError("run Case snapshot.manifest must be a SuiteManifest")
        if not isinstance(self.wire_contract, WireContractV2):
            raise SchemaError(
                "run Case snapshot.wire_contract must be a WireContractV2"
            )
        if self.wire_contract != self.manifest.wire_contract:
            raise SchemaError(
                "run Case snapshot wire contract does not match its Suite manifest"
            )
        entries = _model_tuple(
            self.cases,
            RunCaseSnapshotEntry,
            "run Case snapshot.cases",
            MAX_SUITE_CASES,
        )
        if not entries:
            raise SchemaError(
                "run Case snapshot.cases must contain at least one Case"
            )
        _check_raw_payload_size(
            {
                "schema_version": self.schema_version,
                "snapshot_id": self.snapshot_id,
                "manifest": self.manifest,
                "wire_contract": self.wire_contract,
                "cases": entries,
            },
            MAX_RUN_CASE_SNAPSHOT_BYTES,
            "RunCaseSnapshot",
        )
        task_ids = set()
        for item in entries:
            if item.task_id in task_ids:
                raise SchemaError(
                    "run Case snapshot contains duplicate task_id %r" % item.task_id
                )
            task_ids.add(item.task_id)
            try:
                manifest_case = self.manifest.case(item.task_id)
            except SchemaError as exc:
                raise SchemaError(
                    "run Case snapshot entry has no Suite manifest binding"
                ) from exc
            if manifest_case != item.manifest_case:
                raise SchemaError(
                    "run Case snapshot entry has an inconsistent manifest binding or split"
                )
            _validate_case_source_for_manifest(item.source, self.manifest)
            if item.input.task_id != item.task_id:
                raise SchemaError(
                    "run Case snapshot input does not match its task_id binding"
                )
            if (
                item.input.review_target.kind
                is not self.wire_contract.review_target_kind
            ):
                raise SchemaError(
                    "run Case snapshot Cases do not share a single wire contract"
                )

        ordered = tuple(sorted(entries, key=lambda item: item.task_id))
        object.__setattr__(self, "cases", ordered)
        expected_id = _snapshot_id(self.manifest, ordered)
        if self.snapshot_id != expected_id:
            raise SchemaError(
                "run Case snapshot ID does not match its canonical identity payload"
            )

    @classmethod
    def build(
        cls,
        manifest: SuiteManifest,
        cases: Iterable[Tuple[SuiteCase, EvalCase]],
    ) -> "RunCaseSnapshot":
        if not isinstance(manifest, SuiteManifest):
            raise SchemaError("RunCaseSnapshot.build requires a SuiteManifest")
        pairs = []
        target_kinds = set()
        for index, item in enumerate(cases):
            if index >= MAX_SUITE_CASES:
                raise SchemaError(
                    "run Case snapshot.cases exceeds the item limit of %d"
                    % MAX_SUITE_CASES
                )
            if type(item) not in (tuple, list) or len(item) != 2:
                raise SchemaError(
                    "run Case snapshot.cases[%d] must be a verified Case binding pair"
                    % index
                )
            if not isinstance(item[0], SuiteCase):
                raise SchemaError(
                    "run Case snapshot.cases[%d] must contain a SuiteCase" % index
                )
            if not isinstance(item[1], EvalCase):
                raise SchemaError(
                    "run Case snapshot.cases[%d] must contain an EvalCase" % index
                )
            eval_input = item[1].eval_input()
            pairs.append((item[0], item[1], eval_input))
            target_kinds.add(eval_input.review_target.kind)
        if not pairs:
            raise SchemaError(
                "run Case snapshot.cases must contain at least one Case"
            )
        if target_kinds != {manifest.wire_contract.review_target_kind}:
            raise SchemaError(
                "run Case snapshot Cases must share a single wire contract"
            )
        _check_raw_payload_size(
            {
                "schema_version": cls.SCHEMA_VERSION,
                "snapshot_id": "run-case-snapshot-" + "0" * 64,
                "manifest": manifest,
                "wire_contract": manifest.wire_contract,
                "cases": tuple(
                    {
                        "manifest_case": manifest_case,
                        "source": case.source,
                        "input": eval_input,
                    }
                    for manifest_case, case, eval_input in pairs
                ),
            },
            MAX_RUN_CASE_SNAPSHOT_BYTES,
            "RunCaseSnapshot",
        )
        entries = []
        for manifest_case, case, _eval_input in pairs:
            entries.append(
                RunCaseSnapshotEntry.from_verified_case(
                    manifest_case, case, manifest
                )
            )
        ordered = tuple(sorted(entries, key=lambda item: item.task_id))
        return cls(
            schema_version=cls.SCHEMA_VERSION,
            snapshot_id=_snapshot_id(manifest, ordered),
            manifest=manifest,
            wire_contract=manifest.wire_contract,
            cases=ordered,
        )

    @classmethod
    def from_dict(cls, value: Any) -> "RunCaseSnapshot":
        payload = _object(value, "RunCaseSnapshot")
        if "schema_version" not in payload:
            raise SchemaError(
                "RunCaseSnapshot has missing field(s): schema_version"
            )
        _require_protocol_version(
            payload["schema_version"],
            cls.SCHEMA_VERSION,
            "RunCaseSnapshot.schema_version",
        )
        _exact_fields(
            payload,
            (
                "schema_version",
                "snapshot_id",
                "manifest",
                "wire_contract",
                "cases",
            ),
            "RunCaseSnapshot",
        )
        wire_contract = WireContractV2.from_dict(payload["wire_contract"])
        manifest_payload = _object(
            payload["manifest"], "RunCaseSnapshot.manifest"
        )
        if "schema_version" in manifest_payload:
            _require_protocol_version(
                manifest_payload["schema_version"],
                SuiteManifest.SCHEMA_VERSION,
                "RunCaseSnapshot.manifest.schema_version",
            )
        if "wire_contract" in manifest_payload:
            WireContractV2.from_dict(manifest_payload["wire_contract"])
        raw_cases = _array(
            payload["cases"], "run Case snapshot.cases", MAX_SUITE_CASES
        )
        for index, raw_case in enumerate(raw_cases):
            entry_payload = _object(
                raw_case, "run Case snapshot.cases[%d]" % index
            )
            if "input" not in entry_payload:
                continue
            input_payload = _object(
                entry_payload["input"],
                "run Case snapshot.cases[%d].input" % index,
            )
            if "schema_version" in input_payload:
                _require_protocol_version(
                    input_payload["schema_version"],
                    EVAL_INPUT_SCHEMA_VERSION,
                    "run Case snapshot.cases[%d].input.schema_version" % index,
                )
        _check_raw_payload_size(
            payload,
            MAX_RUN_CASE_SNAPSHOT_BYTES,
            "RunCaseSnapshot",
        )
        return cls(
            schema_version=payload["schema_version"],
            snapshot_id=validate_run_case_snapshot_id(payload["snapshot_id"]),
            manifest=SuiteManifest.from_dict(payload["manifest"]),
            wire_contract=wire_contract,
            cases=tuple(RunCaseSnapshotEntry.from_dict(item) for item in raw_cases),
        )

    @classmethod
    def from_json(cls, data: Any) -> "RunCaseSnapshot":
        return cls.from_dict(
            _strict_json_loads(
                data,
                MAX_RUN_CASE_SNAPSHOT_BYTES,
                "RunCaseSnapshot JSON",
            )
        )

    def case(self, task_id: str) -> RunCaseSnapshotEntry:
        wanted = _identifier(task_id, "task_id")
        for item in self.cases:
            if item.task_id == wanted:
                return item
        raise SchemaError("run Case snapshot has no task_id %r" % wanted)

    def agent_view(self, task_id: str) -> AgentCaseView:
        return self.case(task_id).agent_view()

    def eval_input(self, task_id: str) -> EvalInput:
        return self.case(task_id).input

    def select(self, task_ids: Iterable[str]) -> "RunCaseSnapshot":
        """Return a new truth-free snapshot containing only ``task_ids``.

        Filtering is a Run-level operation.  It must produce a new snapshot
        identity rather than mutating the immutable snapshot that was used to
        make the original Run plan.  The entries already contain the complete
        Agent-facing projection, so this operation never needs to reopen or
        copy the private Case truth.
        """

        if type(task_ids) not in (tuple, list, set, frozenset):
            raise SchemaError("snapshot task_ids must be a bounded collection")
        normalized = tuple(
            _identifier(item, "snapshot.task_ids[%d]" % index)
            for index, item in enumerate(task_ids)
        )
        if not normalized:
            raise SchemaError("filtered Case snapshot may not be empty")
        if len(normalized) != len(set(normalized)):
            raise SchemaError("snapshot task_ids contains duplicates")
        wanted = set(normalized)
        entries = tuple(item for item in self.cases if item.task_id in wanted)
        if len(entries) != len(wanted):
            missing = sorted(wanted.difference(item.task_id for item in entries))
            raise SchemaError(
                "snapshot task_ids are not bound by the snapshot: %s" % missing
            )
        return RunCaseSnapshot(
            schema_version=self.SCHEMA_VERSION,
            snapshot_id=_snapshot_id(self.manifest, entries),
            manifest=self.manifest,
            wire_contract=self.wire_contract,
            cases=entries,
        )

    @property
    def snapshot_digest(self) -> str:
        """Canonical run snapshot SHA, distinct from every upstream digest."""

        return self.digest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "manifest": self.manifest.to_dict(),
            "wire_contract": self.wire_contract.to_dict(),
            "cases": [item.to_dict() for item in self.cases],
        }


@dataclass(frozen=True)
class CaseHandle(_JsonModel):
    """Immutable evaluator-side locator; it never caches private Case data."""

    _suite_root: Path
    _manifest_path: str
    manifest: SuiteManifest
    entry: SuiteCase

    def __post_init__(self) -> None:
        if not isinstance(self._suite_root, Path) or not self._suite_root.is_absolute():
            raise SchemaError("CaseHandle suite root must be an absolute Path")
        _portable_case_path(self._manifest_path, "CaseHandle manifest path")
        if not isinstance(self.manifest, SuiteManifest):
            raise SchemaError("CaseHandle.manifest must be a SuiteManifest")
        if not isinstance(self.entry, SuiteCase):
            raise SchemaError("CaseHandle.entry must be a SuiteCase")
        if self.manifest.case(self.entry.task_id) != self.entry:
            raise SchemaError("CaseHandle entry is not bound to its manifest")

    @property
    def task_id(self) -> str:
        return self.entry.task_id

    @property
    def case_version(self) -> int:
        return self.entry.case_version

    @property
    def split(self) -> CaseSplit:
        return self.entry.split

    @property
    def canonical_case_digest(self) -> str:
        return self.entry.canonical_case_digest

    @property
    def raw_file_sha256(self) -> str:
        return self.entry.raw_file_sha256

    def load(self) -> EvalCase:
        from .datasets import _load_case_from_handle

        return _load_case_from_handle(self)

    def snapshot(self) -> RunCaseSnapshot:
        case = self.load()
        return RunCaseSnapshot.build(self.manifest, ((self.entry, case),))

    def to_dict(self) -> Dict[str, Any]:
        # Root paths and private Case content are intentionally absent.
        return {
            "suite_id": self.manifest.suite_id,
            "suite_version": self.manifest.suite_version,
            "manifest_digest": self.manifest.digest(),
            "task_id": self.task_id,
            "case_version": self.case_version,
            "split": self.split.value,
            "raw_file_size_bytes": self.entry.raw_file_size_bytes,
            "raw_file_sha256": self.raw_file_sha256,
            "canonical_case_digest": self.canonical_case_digest,
            "eval_input_digest": self.entry.eval_input_digest,
        }


def load_suite_manifest(data: Any) -> SuiteManifest:
    return SuiteManifest.from_json(data)


def load_run_case_snapshot(data: Any) -> RunCaseSnapshot:
    return RunCaseSnapshot.from_json(data)


__all__ = [
    "SUITE_MANIFEST_SCHEMA_VERSION",
    "RUN_CASE_SNAPSHOT_SCHEMA_VERSION",
    "PUBLIC_SUITE_PREPARATION_BINDING_SCHEMA_VERSION",
    "REPOSITORY_MATERIALIZER_PROTOCOL",
    "FROZEN_CONTEXT_MATERIALIZER_PROTOCOL",
    "MAX_SUITE_MANIFEST_BYTES",
    "MAX_RUN_CASE_SNAPSHOT_BYTES",
    "MAX_SUITE_CASES",
    "MAX_SUITE_TOTAL_CASE_BYTES",
    "MAX_CASE_DIMENSIONS",
    "CaseSplit",
    "SuiteKind",
    "WireContractV2",
    "PublicSuitePreparationBindingV2",
    "SuiteSource",
    "CaseDimension",
    "SuiteCase",
    "SuiteManifest",
    "AgentCaseView",
    "RunCaseSnapshotEntry",
    "RunCaseSnapshot",
    "validate_run_case_snapshot_id",
    "CaseHandle",
    "validate_case_for_manifest",
    "load_suite_manifest",
    "load_run_case_snapshot",
]
