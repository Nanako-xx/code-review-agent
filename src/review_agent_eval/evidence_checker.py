"""Deterministic integrity verification for Agent-submitted Evidence.

The checker deliberately answers only whether cited material can be replayed
exactly from Harness-owned sources.  It does not decide whether that material
supports a Finding and it never reads a Trial workspace.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import re
from array import array
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Dict, Mapping, Optional, Tuple

from .frozen_context import FrozenContextReplay
from .materialization import MaterializationError
from .models import (
    CommandOutputEvidenceSource,
    EvalInput,
    EvidenceIntegrity,
    EvidenceKind,
    EvidenceStream,
    ExternalRecordEvidenceSource,
    FrozenContextEvidenceSource,
    FrozenContextReviewTarget,
    MAX_CLAIM_CHARS,
    MAX_COMMAND_ARGUMENTS,
    MAX_COUNTER,
    MAX_EVIDENCE_EXCERPT_BYTES,
    MAX_EVIDENCE_ITEMS,
    MAX_EVIDENCE_REFS,
    MAX_FINDINGS,
    MAX_IDENTIFIER_CHARS,
    RepositoryDiffEvidenceSource,
    RepositoryFileEvidenceSource,
    RepositoryReviewTarget,
    ReviewTargetKind,
    SchemaError,
    SubmissionEvidence,
    SubmissionFinding,
    canonical_json,
    canonical_sha256,
)
from .repository import (
    MAX_GIT_BLOB_BYTES,
    PreparedRepositoryReplay,
    RepositoryLimitError,
    RepositoryPolicyError,
    canonical_repository_path,
    repository_from_eval_input,
)


COMMAND_OUTPUT_ATTESTATION_SCHEMA_VERSION = "command_output_attestation_v1"
EVIDENCE_INTEGRITY_POLICY_VERSION = "evidence_integrity_v1"

# This bounds both work and bookkeeping for pathological blobs.  File bytes
# are separately bounded by the verified replay interface.
MAX_REPLAY_LINES = 1_000_000
MAX_COMMAND_OUTPUT_ATTESTATIONS = MAX_EVIDENCE_ITEMS
MAX_COMMAND_OUTPUT_ATTESTATION_BYTES = MAX_EVIDENCE_EXCERPT_BYTES
MAX_COMMAND_OUTPUT_ATTESTATION_JSON_BYTES = (
    2 * MAX_COMMAND_OUTPUT_ATTESTATION_BYTES + 128 * 1024
)
MAX_EVIDENCE_ITEM_DIAGNOSTICS = 64
MAX_EVIDENCE_RESULT_DIAGNOSTICS = MAX_EVIDENCE_REFS * 66 + 1
MAX_EVIDENCE_DIAGNOSTIC_JSON_BYTES = 8 * 1024
MAX_EVIDENCE_ITEM_RESULT_JSON_BYTES = 2 * 1024 * 1024
MAX_EVIDENCE_RESULT_JSON_BYTES = 64 * 1024 * 1024
_MAX_EVIDENCE_JSON_DEPTH = 128

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


def _schema_error(message: str) -> SchemaError:
    return SchemaError(message)


def _identifier(value: Any, context: str) -> str:
    if type(value) is not str or not value or len(value) > MAX_IDENTIFIER_CHARS:
        raise _schema_error("%s must be a non-empty bounded identifier" % context)
    try:
        value.encode("utf-8", "strict")
    except UnicodeEncodeError as exc:
        raise _schema_error("%s must contain valid Unicode" % context) from exc
    if value != value.strip() or any(
        character.isspace() or ord(character) < 32 or ord(character) == 127
        for character in value
    ):
        raise _schema_error("%s must not contain whitespace or controls" % context)
    return value


def _optional_identifier(value: Any, context: str) -> Optional[str]:
    if value is None:
        return None
    return _identifier(value, context)


def _exact_integer(
    value: Any, context: str, *, minimum: int, maximum: int
) -> int:
    if type(value) is not int or value < minimum or value > maximum:
        raise _schema_error("%s must be a bounded integer (bool is not accepted)" % context)
    return value


def _sha256(value: Any, context: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise _schema_error("%s must be a lowercase SHA-256 digest" % context)
    return value


def _git_object(value: Any, context: str) -> str:
    if type(value) is not str or _GIT_OBJECT_RE.fullmatch(value) is None:
        raise _schema_error("%s must be an exact lowercase Git object ID" % context)
    return value


def _strict_object(value: Any, fields: Tuple[str, ...], context: str) -> Dict[str, Any]:
    if type(value) is not dict:
        raise _schema_error("%s must be an object" % context)
    if set(value) != set(fields) or len(value) != len(fields):
        raise _schema_error("%s has unknown or missing fields" % context)
    return value


def _reject_json_constant(value: str) -> Any:
    raise _schema_error("command output attestation JSON contains %s" % value)


def _strict_json_object(pairs: Any) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise _schema_error("JSON has an invalid or duplicate key")
        result[key] = value
    return result


def _validate_json_tree(value: Any, context: str, depth: int = 0) -> None:
    if depth > _MAX_EVIDENCE_JSON_DEPTH:
        raise _schema_error("%s exceeds the maximum JSON depth" % context)
    if value is None or type(value) in (bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise _schema_error("%s contains a non-finite number" % context)
        return
    if type(value) is str:
        try:
            value.encode("utf-8", "strict")
        except UnicodeEncodeError as exc:
            raise _schema_error("%s contains invalid Unicode" % context) from exc
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_json_tree(item, "%s[%d]" % (context, index), depth + 1)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise _schema_error("%s contains a non-string key" % context)
            _validate_json_tree(item, "%s.%s" % (context, key), depth + 1)
        return
    raise _schema_error("%s contains a non-JSON value" % context)


def _strict_json_loads(data: Any, maximum_bytes: int, context: str) -> Any:
    if type(data) is bytes:
        if len(data) > maximum_bytes:
            raise _schema_error("%s exceeds its byte limit" % context)
        try:
            text = data.decode("utf-8", "strict")
        except UnicodeDecodeError as exc:
            raise _schema_error("%s must be strict UTF-8" % context) from exc
    elif type(data) is str:
        try:
            encoded = data.encode("utf-8", "strict")
        except UnicodeEncodeError as exc:
            raise _schema_error("%s contains invalid Unicode" % context) from exc
        if len(encoded) > maximum_bytes:
            raise _schema_error("%s exceeds its byte limit" % context)
        text = data
    else:
        raise _schema_error("%s must be text or bytes" % context)
    try:
        value = json.loads(
            text,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except SchemaError:
        raise
    except (
        json.JSONDecodeError,
        RecursionError,
        UnicodeError,
        OverflowError,
        ValueError,
    ) as exc:
        raise _schema_error("%s is invalid" % context) from exc
    _validate_json_tree(value, context)
    return value


@dataclass(frozen=True)
class CommandOutputAttestation:
    """Harness-private identity and byte attestation for one command stream."""

    SCHEMA_VERSION: ClassVar[str] = COMMAND_OUTPUT_ATTESTATION_SCHEMA_VERSION

    schema_version: str
    source_ref: str
    trial_id: str
    head_revision: str
    argv: Tuple[str, ...]
    exit_code: int
    stream: EvidenceStream
    output_bytes: bytes = field(repr=False)
    byte_size: int
    sha256: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not str or self.schema_version != self.SCHEMA_VERSION:
            raise _schema_error("CommandOutputAttestation has an unknown schema_version")
        _identifier(self.source_ref, "command attestation.source_ref")
        _identifier(self.trial_id, "command attestation.trial_id")
        _git_object(self.head_revision, "command attestation.head_revision")
        if type(self.argv) is not tuple or not self.argv:
            raise _schema_error("command attestation.argv must be a non-empty tuple")
        if len(self.argv) > MAX_COMMAND_ARGUMENTS:
            raise _schema_error("command attestation.argv exceeds its item limit")
        for index, argument in enumerate(self.argv):
            if type(argument) is not str or len(argument) > MAX_CLAIM_CHARS:
                raise _schema_error(
                    "command attestation.argv[%d] must be a bounded string" % index
                )
            try:
                argument.encode("utf-8", "strict")
            except UnicodeEncodeError as exc:
                raise _schema_error(
                    "command attestation.argv contains invalid Unicode"
                ) from exc
        _exact_integer(
            self.exit_code,
            "command attestation.exit_code",
            minimum=-MAX_COUNTER,
            maximum=MAX_COUNTER,
        )
        if type(self.stream) is not EvidenceStream:
            raise _schema_error("command attestation.stream must be an EvidenceStream")
        if type(self.output_bytes) is not bytes:
            raise _schema_error("command attestation.output_bytes must be immutable bytes")
        _exact_integer(
            self.byte_size,
            "command attestation.byte_size",
            minimum=0,
            maximum=MAX_COMMAND_OUTPUT_ATTESTATION_BYTES,
        )
        if len(self.output_bytes) != self.byte_size:
            raise _schema_error("command attestation.byte_size does not match its bytes")
        digest = _sha256(self.sha256, "command attestation.sha256")
        if hashlib.sha256(self.output_bytes).hexdigest() != digest:
            raise _schema_error("command attestation.sha256 does not match its bytes")

    @property
    def revision(self) -> str:
        """Alias emphasizing that the attested revision is the exact head."""

        return self.head_revision

    @property
    def output(self) -> bytes:
        return self.output_bytes

    @property
    def content_hash(self) -> str:
        return self.sha256

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_ref": self.source_ref,
            "trial_id": self.trial_id,
            "head_revision": self.head_revision,
            "argv": list(self.argv),
            "exit_code": self.exit_code,
            "stream": self.stream.value,
            "output_base64": base64.b64encode(self.output_bytes).decode("ascii"),
            "byte_size": self.byte_size,
            "sha256": self.sha256,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, value: Any) -> "CommandOutputAttestation":
        fields = (
            "schema_version",
            "source_ref",
            "trial_id",
            "head_revision",
            "argv",
            "exit_code",
            "stream",
            "output_base64",
            "byte_size",
            "sha256",
        )
        payload = _strict_object(value, fields, "CommandOutputAttestation")
        if type(payload["argv"]) is not list:
            raise _schema_error("command attestation.argv must be an array")
        encoded = payload["output_base64"]
        if type(encoded) is not str:
            raise _schema_error("command attestation.output_base64 must be a string")
        try:
            output = base64.b64decode(encoded.encode("ascii"), validate=True)
        except (UnicodeEncodeError, binascii.Error, ValueError) as exc:
            raise _schema_error(
                "command attestation.output_base64 is not strict base64"
            ) from exc
        if base64.b64encode(output).decode("ascii") != encoded:
            raise _schema_error("command attestation.output_base64 is not canonical")
        stream_value = payload["stream"]
        if type(stream_value) is not str:
            raise _schema_error("command attestation.stream must be a string enum")
        try:
            stream = EvidenceStream(stream_value)
        except ValueError as exc:
            raise _schema_error("command attestation.stream is unknown") from exc
        return cls(
            schema_version=payload["schema_version"],
            source_ref=payload["source_ref"],
            trial_id=payload["trial_id"],
            head_revision=payload["head_revision"],
            argv=tuple(payload["argv"]),
            exit_code=payload["exit_code"],
            stream=stream,
            output_bytes=output,
            byte_size=payload["byte_size"],
            sha256=payload["sha256"],
        )

    @classmethod
    def from_json(cls, data: Any) -> "CommandOutputAttestation":
        if type(data) is bytes:
            raw = data
            try:
                text = raw.decode("utf-8", "strict")
            except UnicodeDecodeError as exc:
                raise _schema_error(
                    "CommandOutputAttestation JSON must be UTF-8"
                ) from exc
        elif type(data) is str:
            text = data
            try:
                raw = text.encode("utf-8", "strict")
            except UnicodeEncodeError as exc:
                raise _schema_error(
                    "CommandOutputAttestation JSON contains invalid Unicode"
                ) from exc
        else:
            raise _schema_error("CommandOutputAttestation JSON must be str or bytes")
        if len(raw) > MAX_COMMAND_OUTPUT_ATTESTATION_JSON_BYTES:
            raise _schema_error("CommandOutputAttestation JSON exceeds its byte limit")
        try:
            value = json.loads(
                text,
                object_pairs_hook=_strict_json_object,
                parse_constant=_reject_json_constant,
            )
        except SchemaError:
            raise
        except (json.JSONDecodeError, RecursionError, ValueError) as exc:
            raise _schema_error("CommandOutputAttestation JSON is invalid") from exc
        return cls.from_dict(value)

    serialize = to_dict
    hydrate = from_dict


class EvidenceReasonCode(str, Enum):
    """Stable deterministic reasons; values are persisted evaluator vocabulary."""

    NO_EVIDENCE_REFS = "no_evidence_refs"
    DANGLING_REF = "dangling_ref"
    DUPLICATE_REF = "duplicate_ref"
    KIND_FIELD_MISMATCH = "kind_field_mismatch"
    TARGET_MATERIALIZATION_MISMATCH = "target_materialization_mismatch"
    CONTEXT_REF_MISMATCH = "context_ref_mismatch"
    REPLAY_BINDING_MISMATCH = "replay_binding_mismatch"
    REVISION_MISMATCH = "revision_mismatch"
    PATH_INVALID = "path_invalid"
    PATH_NOT_FOUND = "path_not_found"
    LINE_RANGE_PARTIAL = "line_range_partial"
    LINE_RANGE_REVERSED = "line_range_reversed"
    LINE_RANGE_OUT_OF_BOUNDS = "line_range_out_of_bounds"
    REPLAY_LINE_LIMIT_EXCEEDED = "replay_line_limit_exceeded"
    EXCERPT_TOO_LARGE = "excerpt_too_large"
    CONTENT_NOT_UTF8 = "content_not_utf8"
    EXCERPT_MISMATCH = "excerpt_mismatch"
    CONTENT_HASH_MISMATCH = "content_hash_mismatch"
    ATTESTATION_NOT_FOUND = "attestation_not_found"
    ATTESTATION_AMBIGUOUS = "attestation_ambiguous"
    ATTESTATION_TRIAL_MISMATCH = "attestation_trial_mismatch"
    ATTESTATION_REVISION_MISMATCH = "attestation_revision_mismatch"
    COMMAND_MISMATCH = "command_mismatch"
    EXIT_CODE_MISMATCH = "exit_code_mismatch"
    STREAM_MISMATCH = "stream_mismatch"
    EXTERNAL_RECORD_NOT_FOUND = "external_record_not_found"


def _bounded_json_array(value: Any, maximum: int, context: str) -> list[Any]:
    if type(value) is not list:
        raise _schema_error("%s must be an array" % context)
    if len(value) > maximum:
        raise _schema_error("%s exceeds its item limit" % context)
    return value


def _wire_enum(enum_type: Any, value: Any, context: str) -> Any:
    if type(value) is not str:
        raise _schema_error("%s must be a string enum" % context)
    try:
        return enum_type(value)
    except ValueError as exc:
        raise _schema_error("%s contains an unknown enum value" % context) from exc


@dataclass(frozen=True)
class EvidenceDiagnostic:
    """A redacted item/ref-level diagnostic; ref indexes are zero-based."""

    WIRE_FIELDS: ClassVar[Tuple[str, ...]] = (
        "reason_code",
        "evidence_id",
        "finding_id",
        "ref_index",
    )

    reason_code: EvidenceReasonCode
    evidence_id: Optional[str] = None
    finding_id: Optional[str] = None
    ref_index: Optional[int] = None

    def __post_init__(self) -> None:
        if type(self.reason_code) is not EvidenceReasonCode:
            raise TypeError("reason_code must be an EvidenceReasonCode")
        _optional_identifier(self.evidence_id, "diagnostic.evidence_id")
        _optional_identifier(self.finding_id, "diagnostic.finding_id")
        if self.ref_index is not None:
            _exact_integer(
                self.ref_index,
                "diagnostic.ref_index",
                minimum=0,
                maximum=MAX_EVIDENCE_REFS - 1,
            )
        if self.reason_code is EvidenceReasonCode.NO_EVIDENCE_REFS:
            if (
                self.evidence_id is not None
                or self.finding_id is None
                or self.ref_index is not None
            ):
                raise ValueError(
                    "no_evidence_refs diagnostic must identify only its Finding"
                )
        elif self.reason_code in (
            EvidenceReasonCode.DANGLING_REF,
            EvidenceReasonCode.DUPLICATE_REF,
        ):
            if (
                self.evidence_id is None
                or self.finding_id is None
                or self.ref_index is None
            ):
                raise ValueError(
                    "reference diagnostic requires evidence, Finding, and ref index"
                )
        else:
            if self.evidence_id is None:
                raise ValueError("item diagnostic requires an evidence ID")
            if (self.finding_id is None) != (self.ref_index is None):
                raise ValueError(
                    "propagated item diagnostic requires both Finding and ref index"
                )

    @property
    def code(self) -> EvidenceReasonCode:
        return self.reason_code

    @property
    def message(self) -> str:
        return self.reason_code.value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reason_code": self.reason_code.value,
            "evidence_id": self.evidence_id,
            "finding_id": self.finding_id,
            "ref_index": self.ref_index,
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    def digest(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: Any) -> "EvidenceDiagnostic":
        payload = _strict_object(value, cls.WIRE_FIELDS, "EvidenceDiagnostic")
        ref_index = payload["ref_index"]
        if ref_index is not None:
            ref_index = _exact_integer(
                ref_index,
                "diagnostic.ref_index",
                minimum=0,
                maximum=MAX_EVIDENCE_REFS - 1,
            )
        return cls(
            reason_code=_wire_enum(
                EvidenceReasonCode,
                payload["reason_code"],
                "diagnostic.reason_code",
            ),
            evidence_id=_optional_identifier(
                payload["evidence_id"], "diagnostic.evidence_id"
            ),
            finding_id=_optional_identifier(
                payload["finding_id"], "diagnostic.finding_id"
            ),
            ref_index=ref_index,
        )

    @classmethod
    def from_json(cls, data: Any) -> "EvidenceDiagnostic":
        return cls.from_dict(
            _strict_json_loads(
                data,
                MAX_EVIDENCE_DIAGNOSTIC_JSON_BYTES,
                "EvidenceDiagnostic JSON",
            )
        )

    serialize = to_dict
    hydrate = from_dict


@dataclass(frozen=True)
class EvidenceItemIntegrityResult:
    WIRE_FIELDS: ClassVar[Tuple[str, ...]] = (
        "policy_version",
        "evidence_id",
        "kind",
        "integrity",
        "diagnostics",
    )

    evidence_id: str
    kind: EvidenceKind
    integrity: EvidenceIntegrity
    diagnostics: Tuple[EvidenceDiagnostic, ...]
    policy_version: str = EVIDENCE_INTEGRITY_POLICY_VERSION

    def __post_init__(self) -> None:
        _identifier(self.evidence_id, "item result.evidence_id")
        if type(self.kind) is not EvidenceKind:
            raise TypeError("item result.kind must be an EvidenceKind")
        if type(self.integrity) is not EvidenceIntegrity or self.integrity is EvidenceIntegrity.MISSING:
            raise TypeError("item result.integrity must be valid or invalid")
        if (
            type(self.diagnostics) is not tuple
            or len(self.diagnostics) > MAX_EVIDENCE_ITEM_DIAGNOSTICS
        ):
            raise TypeError("item result.diagnostics must be a bounded tuple")
        if any(type(item) is not EvidenceDiagnostic for item in self.diagnostics):
            raise TypeError("item result diagnostics must be EvidenceDiagnostic items")
        if self.policy_version != EVIDENCE_INTEGRITY_POLICY_VERSION:
            raise ValueError("item result has an unknown policy_version")
        if self.integrity is EvidenceIntegrity.VALID and self.diagnostics:
            raise ValueError("valid item result cannot contain diagnostics")
        if self.integrity is EvidenceIntegrity.INVALID and not self.diagnostics:
            raise ValueError("invalid item result requires a diagnostic")
        if len(set(self.diagnostics)) != len(self.diagnostics):
            raise ValueError("item result diagnostics must be unique")
        canonical = tuple(
            sorted(self.diagnostics, key=lambda item: item.reason_code.value)
        )
        if self.diagnostics != canonical:
            raise ValueError("item result diagnostics must be in canonical order")
        for diagnostic in self.diagnostics:
            if (
                diagnostic.evidence_id != self.evidence_id
                or diagnostic.finding_id is not None
                or diagnostic.ref_index is not None
                or diagnostic.reason_code
                in (
                    EvidenceReasonCode.NO_EVIDENCE_REFS,
                    EvidenceReasonCode.DANGLING_REF,
                    EvidenceReasonCode.DUPLICATE_REF,
                )
            ):
                raise ValueError(
                    "item result contains a diagnostic outside its item scope"
                )

    @property
    def status(self) -> EvidenceIntegrity:
        return self.integrity

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "evidence_id": self.evidence_id,
            "kind": self.kind.value,
            "integrity": self.integrity.value,
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    def digest(self) -> str:
        return canonical_sha256(self.to_dict())

    def validate_against(
        self,
        evidence: SubmissionEvidence,
        checker: Optional["EvidenceIntegrityChecker"] = None,
    ) -> None:
        """Validate this receipt against one immutable Submission Evidence item."""

        if type(evidence) is not SubmissionEvidence:
            raise TypeError("evidence binding must be a SubmissionEvidence")
        if self.evidence_id != evidence.evidence_id:
            raise _schema_error("item result is bound to a different Evidence ID")
        if self.kind is not evidence.source.kind:
            raise _schema_error("item result kind does not match Submission Evidence")
        if checker is not None:
            if type(checker) is not EvidenceIntegrityChecker:
                raise TypeError("checker binding must be an EvidenceIntegrityChecker")
            expected = checker.check_item(evidence)
            if self != expected:
                raise _schema_error(
                    "item result does not match deterministic Evidence replay"
                )

    @classmethod
    def from_dict(
        cls,
        value: Any,
        evidence: Optional[SubmissionEvidence] = None,
        checker: Optional["EvidenceIntegrityChecker"] = None,
    ) -> "EvidenceItemIntegrityResult":
        payload = _strict_object(
            value, cls.WIRE_FIELDS, "EvidenceItemIntegrityResult"
        )
        diagnostics = _bounded_json_array(
            payload["diagnostics"],
            MAX_EVIDENCE_ITEM_DIAGNOSTICS,
            "item result.diagnostics",
        )
        result = cls(
            policy_version=payload["policy_version"],
            evidence_id=_identifier(
                payload["evidence_id"], "item result.evidence_id"
            ),
            kind=_wire_enum(EvidenceKind, payload["kind"], "item result.kind"),
            integrity=_wire_enum(
                EvidenceIntegrity,
                payload["integrity"],
                "item result.integrity",
            ),
            diagnostics=tuple(
                EvidenceDiagnostic.from_dict(item) for item in diagnostics
            ),
        )
        if evidence is not None or checker is not None:
            if evidence is None:
                raise _schema_error("strict source binding requires Evidence")
            result.validate_against(evidence, checker)
        return result

    @classmethod
    def from_json(
        cls,
        data: Any,
        evidence: Optional[SubmissionEvidence] = None,
        checker: Optional["EvidenceIntegrityChecker"] = None,
    ) -> "EvidenceItemIntegrityResult":
        return cls.from_dict(
            _strict_json_loads(
                data,
                MAX_EVIDENCE_ITEM_RESULT_JSON_BYTES,
                "EvidenceItemIntegrityResult JSON",
            ),
            evidence,
            checker,
        )

    serialize = to_dict
    hydrate = from_dict


def _derive_finding_integrity_state(
    finding_id: str,
    referenced_evidence_ids: Tuple[str, ...],
    item_results: Tuple[EvidenceItemIntegrityResult, ...],
) -> Tuple[EvidenceIntegrity, Tuple[EvidenceDiagnostic, ...]]:
    if not referenced_evidence_ids:
        if item_results:
            raise _schema_error(
                "finding result without refs cannot contain item_results"
            )
        return (
            EvidenceIntegrity.MISSING,
            (
                EvidenceDiagnostic(
                    reason_code=EvidenceReasonCode.NO_EVIDENCE_REFS,
                    finding_id=finding_id,
                ),
            ),
        )

    expected_diagnostics = []
    item_cursor = 0
    seen: Dict[str, Optional[EvidenceItemIntegrityResult]] = {}
    has_dangling = False
    has_invalid = False
    for ref_index, evidence_id in enumerate(referenced_evidence_ids):
        is_duplicate = evidence_id in seen
        if is_duplicate:
            expected_diagnostics.append(
                EvidenceDiagnostic(
                    reason_code=EvidenceReasonCode.DUPLICATE_REF,
                    evidence_id=evidence_id,
                    finding_id=finding_id,
                    ref_index=ref_index,
                )
            )

        if is_duplicate:
            item_result = seen[evidence_id]
            if item_result is not None:
                if (
                    item_cursor >= len(item_results)
                    or item_results[item_cursor].evidence_id != evidence_id
                ):
                    raise _schema_error(
                        "finding item_results do not preserve evidence_refs order"
                    )
                repeated = item_results[item_cursor]
                item_cursor += 1
                if repeated != item_result:
                    raise _schema_error(
                        "duplicate reference must repeat the same item result"
                    )
        elif (
            item_cursor < len(item_results)
            and item_results[item_cursor].evidence_id == evidence_id
        ):
            item_result = item_results[item_cursor]
            item_cursor += 1
            seen[evidence_id] = item_result
        else:
            item_result = None
            seen[evidence_id] = None

        if item_result is None:
            has_dangling = True
            expected_diagnostics.append(
                EvidenceDiagnostic(
                    reason_code=EvidenceReasonCode.DANGLING_REF,
                    evidence_id=evidence_id,
                    finding_id=finding_id,
                    ref_index=ref_index,
                )
            )
            continue

        if item_result.integrity is EvidenceIntegrity.INVALID:
            has_invalid = True
        expected_diagnostics.extend(
            EvidenceDiagnostic(
                reason_code=item_diagnostic.reason_code,
                evidence_id=evidence_id,
                finding_id=finding_id,
                ref_index=ref_index,
            )
            for item_diagnostic in item_result.diagnostics
        )

    if item_cursor != len(item_results):
        raise _schema_error(
            "finding result contains item_results outside evidence_refs order"
        )
    if has_dangling:
        integrity = EvidenceIntegrity.MISSING
    elif has_invalid:
        integrity = EvidenceIntegrity.INVALID
    else:
        integrity = EvidenceIntegrity.VALID
    return integrity, tuple(expected_diagnostics)


@dataclass(frozen=True)
class EvidenceIntegrityResult:
    WIRE_FIELDS: ClassVar[Tuple[str, ...]] = (
        "policy_version",
        "finding_id",
        "integrity",
        "referenced_evidence_ids",
        "item_results",
        "diagnostics",
    )

    finding_id: str
    integrity: EvidenceIntegrity
    referenced_evidence_ids: Tuple[str, ...]
    item_results: Tuple[EvidenceItemIntegrityResult, ...]
    diagnostics: Tuple[EvidenceDiagnostic, ...]
    policy_version: str = EVIDENCE_INTEGRITY_POLICY_VERSION

    def __post_init__(self) -> None:
        _identifier(self.finding_id, "finding result.finding_id")
        if type(self.integrity) is not EvidenceIntegrity:
            raise TypeError("finding result.integrity must be an EvidenceIntegrity")
        if type(self.referenced_evidence_ids) is not tuple or len(
            self.referenced_evidence_ids
        ) > MAX_EVIDENCE_REFS:
            raise TypeError("referenced_evidence_ids must be a bounded tuple")
        for value in self.referenced_evidence_ids:
            _identifier(value, "finding result referenced evidence ID")
        if type(self.item_results) is not tuple or len(self.item_results) > MAX_EVIDENCE_REFS:
            raise TypeError("item_results must be a bounded tuple")
        if any(type(item) is not EvidenceItemIntegrityResult for item in self.item_results):
            raise TypeError("item_results must contain EvidenceItemIntegrityResult")
        if (
            type(self.diagnostics) is not tuple
            or len(self.diagnostics) > MAX_EVIDENCE_RESULT_DIAGNOSTICS
        ):
            raise TypeError("finding result.diagnostics must be a bounded tuple")
        if any(type(item) is not EvidenceDiagnostic for item in self.diagnostics):
            raise TypeError("finding result diagnostics must be EvidenceDiagnostic items")
        if self.policy_version != EVIDENCE_INTEGRITY_POLICY_VERSION:
            raise ValueError("finding result has an unknown policy_version")
        if len(set(self.diagnostics)) != len(self.diagnostics):
            raise ValueError("finding result diagnostics must be unique")
        expected_integrity, expected_diagnostics = _derive_finding_integrity_state(
            self.finding_id,
            self.referenced_evidence_ids,
            self.item_results,
        )
        if self.integrity is not expected_integrity:
            raise ValueError("finding result.integrity is not canonically derived")
        if self.diagnostics != expected_diagnostics:
            raise ValueError(
                "finding result.diagnostics are not the canonical derived diagnostics"
            )

    @property
    def status(self) -> EvidenceIntegrity:
        return self.integrity

    @property
    def referenced_items(self) -> Tuple[EvidenceItemIntegrityResult, ...]:
        return self.item_results

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "finding_id": self.finding_id,
            "integrity": self.integrity.value,
            "referenced_evidence_ids": list(self.referenced_evidence_ids),
            "item_results": [item.to_dict() for item in self.item_results],
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    def digest(self) -> str:
        return canonical_sha256(self.to_dict())

    def validate_against(
        self,
        finding: SubmissionFinding,
        evidence_items: Tuple[SubmissionEvidence, ...],
        checker: Optional["EvidenceIntegrityChecker"] = None,
    ) -> None:
        """Validate this receipt against immutable Submission-side sources.

        Structural hydration derives the aggregate status and diagnostics from
        the persisted ref/item graph.  This binding additionally proves that
        the graph is the one belonging to the supplied Submission Finding.  A
        checker may be supplied to re-run exact repository/source replay and
        reject a coordinated forgery of the item-level outcome itself.
        """

        if type(finding) is not SubmissionFinding:
            raise TypeError("finding binding must be a SubmissionFinding")
        if self.finding_id != finding.finding_id:
            raise _schema_error("finding result is bound to a different Finding")
        if self.referenced_evidence_ids != finding.evidence_refs:
            raise _schema_error(
                "finding result does not preserve Submission evidence_refs order"
            )
        if type(evidence_items) is not tuple:
            raise TypeError("evidence binding must be an immutable tuple")
        if len(evidence_items) > MAX_EVIDENCE_ITEMS or any(
            type(item) is not SubmissionEvidence for item in evidence_items
        ):
            raise TypeError("evidence binding contains invalid items")
        evidence_by_id: Dict[str, SubmissionEvidence] = {}
        for item in evidence_items:
            if item.evidence_id in evidence_by_id:
                raise _schema_error("evidence binding contains a duplicate ID")
            evidence_by_id[item.evidence_id] = item
        expected_item_ids = tuple(
            evidence_id
            for evidence_id in finding.evidence_refs
            if evidence_id in evidence_by_id
        )
        actual_item_ids = tuple(item.evidence_id for item in self.item_results)
        if actual_item_ids != expected_item_ids:
            raise _schema_error(
                "finding result item_results do not match Submission Evidence"
            )
        for item_result in self.item_results:
            evidence_item = evidence_by_id[item_result.evidence_id]
            if item_result.kind is not evidence_item.source.kind:
                raise _schema_error(
                    "finding item result kind does not match Submission Evidence"
                )
        if checker is not None:
            if type(checker) is not EvidenceIntegrityChecker:
                raise TypeError("checker binding must be an EvidenceIntegrityChecker")
            expected = checker.check_finding(finding, evidence_items)
            if self != expected:
                raise _schema_error(
                    "finding result does not match deterministic Evidence replay"
                )

    @classmethod
    def from_dict(
        cls,
        value: Any,
        finding: Optional[SubmissionFinding] = None,
        evidence_items: Optional[Tuple[SubmissionEvidence, ...]] = None,
        checker: Optional["EvidenceIntegrityChecker"] = None,
    ) -> "EvidenceIntegrityResult":
        payload = _strict_object(value, cls.WIRE_FIELDS, "EvidenceIntegrityResult")
        references = _bounded_json_array(
            payload["referenced_evidence_ids"],
            MAX_EVIDENCE_REFS,
            "finding result.referenced_evidence_ids",
        )
        item_results = _bounded_json_array(
            payload["item_results"],
            MAX_EVIDENCE_REFS,
            "finding result.item_results",
        )
        diagnostics = _bounded_json_array(
            payload["diagnostics"],
            MAX_EVIDENCE_RESULT_DIAGNOSTICS,
            "finding result.diagnostics",
        )
        result = cls(
            policy_version=payload["policy_version"],
            finding_id=_identifier(
                payload["finding_id"], "finding result.finding_id"
            ),
            integrity=_wire_enum(
                EvidenceIntegrity,
                payload["integrity"],
                "finding result.integrity",
            ),
            referenced_evidence_ids=tuple(
                _identifier(
                    item,
                    "finding result.referenced_evidence_ids[%d]" % index,
                )
                for index, item in enumerate(references)
            ),
            item_results=tuple(
                EvidenceItemIntegrityResult.from_dict(item) for item in item_results
            ),
            diagnostics=tuple(
                EvidenceDiagnostic.from_dict(item) for item in diagnostics
            ),
        )
        if finding is not None or evidence_items is not None or checker is not None:
            if finding is None or evidence_items is None:
                raise _schema_error(
                    "strict source binding requires both Finding and Evidence"
                )
            result.validate_against(finding, evidence_items, checker)
        return result

    @classmethod
    def from_json(
        cls,
        data: Any,
        finding: Optional[SubmissionFinding] = None,
        evidence_items: Optional[Tuple[SubmissionEvidence, ...]] = None,
        checker: Optional["EvidenceIntegrityChecker"] = None,
    ) -> "EvidenceIntegrityResult":
        return cls.from_dict(
            _strict_json_loads(
                data,
                MAX_EVIDENCE_RESULT_JSON_BYTES,
                "EvidenceIntegrityResult JSON",
            ),
            finding,
            evidence_items,
            checker,
        )

    serialize = to_dict
    hydrate = from_dict


@dataclass(frozen=True)
class _CanonicalLines:
    excerpt: Optional[str]
    line_count: int
    too_many_lines: bool
    excerpt_too_large: bool


@dataclass(frozen=True)
class _CanonicalSource:
    text: Optional[str]
    content_hash: Optional[str]
    reasons: Tuple[EvidenceReasonCode, ...]

    def __post_init__(self) -> None:
        if self.reasons:
            if self.text is not None or self.content_hash is not None:
                raise ValueError("invalid canonical source cannot contain content")
            return
        if type(self.text) is not str or type(self.content_hash) is not str:
            raise ValueError("valid canonical source requires text and hash")


@dataclass(frozen=True)
class _FrozenReplayRecord:
    data: Optional[bytes]
    line_starts: Optional[array]
    reasons: Tuple[EvidenceReasonCode, ...]

    def __post_init__(self) -> None:
        if self.reasons:
            if self.data is not None or self.line_starts is not None:
                raise ValueError("invalid Frozen replay record cannot contain content")
            return
        if type(self.data) is not bytes or type(self.line_starts) is not array:
            raise ValueError("valid Frozen replay record requires bytes and line index")


_SourceCache = Dict[Tuple[Any, ...], Any]


_OTHER_SPLITLINE_BOUNDARIES = frozenset(
    ("\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029")
)


def _canonical_line_excerpt(text: str, from_line: int, to_line: int) -> _CanonicalLines:
    """Scan like ``splitlines(keepends=True)`` without allocating all lines.

    Every recognized logical boundary is emitted as LF (with CRLF consumed as
    one boundary).  Only the selected range is retained, and retention stops
    as soon as its UTF-8 representation exceeds the canonical Evidence limit.
    The scan itself stops at the logical-line cap.
    """

    selected: list[str] = []
    selected_bytes = 0
    excerpt_too_large = False
    line_count = 0
    index = 0
    length = len(text)
    ended_with_boundary = False

    def retain(value: str, line_number: int) -> None:
        nonlocal selected_bytes, excerpt_too_large
        if excerpt_too_large or not (from_line <= line_number <= to_line):
            return
        encoded_size = len(value.encode("utf-8"))
        if selected_bytes + encoded_size > MAX_EVIDENCE_EXCERPT_BYTES:
            excerpt_too_large = True
            selected.clear()
            return
        selected.append(value)
        selected_bytes += encoded_size

    while index < length:
        character = text[index]
        line_number = line_count + 1
        if character == "\r":
            if index + 1 < length and text[index + 1] == "\n":
                index += 1
            retain("\n", line_number)
            line_count += 1
            ended_with_boundary = True
        elif character == "\n":
            retain("\n", line_number)
            line_count += 1
            ended_with_boundary = True
        elif character in _OTHER_SPLITLINE_BOUNDARIES:
            retain("\n", line_number)
            line_count += 1
            ended_with_boundary = True
        else:
            retain(character, line_number)
            ended_with_boundary = False
        if line_count > MAX_REPLAY_LINES:
            return _CanonicalLines(None, line_count, True, excerpt_too_large)
        index += 1

    if length and not ended_with_boundary:
        line_count += 1
        if line_count > MAX_REPLAY_LINES:
            return _CanonicalLines(None, line_count, True, excerpt_too_large)
    return _CanonicalLines(
        None if excerpt_too_large else "".join(selected),
        line_count,
        False,
        excerpt_too_large,
    )


def _diagnostics(
    evidence_id: str, reasons: Tuple[EvidenceReasonCode, ...]
) -> Tuple[EvidenceDiagnostic, ...]:
    return tuple(
        EvidenceDiagnostic(reason_code=reason, evidence_id=evidence_id)
        for reason in sorted(set(reasons), key=lambda item: item.value)
    )


def _item_result(
    evidence: SubmissionEvidence, reasons: Tuple[EvidenceReasonCode, ...]
) -> EvidenceItemIntegrityResult:
    diagnostics = _diagnostics(evidence.evidence_id, reasons)
    return EvidenceItemIntegrityResult(
        evidence_id=evidence.evidence_id,
        kind=evidence.source.kind,
        integrity=(
            EvidenceIntegrity.INVALID if diagnostics else EvidenceIntegrity.VALID
        ),
        diagnostics=diagnostics,
    )


def _content_reasons(
    evidence: SubmissionEvidence, canonical_text: str, canonical_hash: str
) -> Tuple[EvidenceReasonCode, ...]:
    reasons = []
    if evidence.excerpt != canonical_text:
        reasons.append(EvidenceReasonCode.EXCERPT_MISMATCH)
    if evidence.content_hash != canonical_hash:
        reasons.append(EvidenceReasonCode.CONTENT_HASH_MISMATCH)
    return tuple(reasons)


@dataclass(frozen=True)
class EvidenceIntegrityChecker:
    """Verify Evidence against one exact EvalInput/replay/trial binding."""

    POLICY_VERSION: ClassVar[str] = EVIDENCE_INTEGRITY_POLICY_VERSION

    eval_input: EvalInput
    replay: PreparedRepositoryReplay | FrozenContextReplay
    trial_id: str
    target_materialization_id: str
    command_attestations: Tuple[CommandOutputAttestation, ...] = ()
    _attestations_by_source: Mapping[str, Tuple[CommandOutputAttestation, ...]] = field(
        init=False, repr=False, compare=False
    )
    _external_by_source: Mapping[str, Any] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if type(self.eval_input) is not EvalInput:
            raise TypeError("eval_input must be the canonical EvalInput")
        _identifier(self.trial_id, "evidence checker.trial_id")
        _identifier(
            self.target_materialization_id,
            "evidence checker.target_materialization_id",
        )
        if type(self.command_attestations) is not tuple:
            raise TypeError("command_attestations must be an immutable tuple")
        if len(self.command_attestations) > MAX_COMMAND_OUTPUT_ATTESTATIONS:
            raise ValueError("command_attestations exceeds its item limit")
        if any(type(item) is not CommandOutputAttestation for item in self.command_attestations):
            raise TypeError("command_attestations contains a non-attestation item")

        target = self.eval_input.review_target
        if target.kind is ReviewTargetKind.REPOSITORY:
            if type(target) is not RepositoryReviewTarget:
                raise TypeError("Repository target has a non-canonical target type")
            if type(self.replay) is not PreparedRepositoryReplay:
                raise TypeError(
                    "Repository review target requires PreparedRepositoryReplay"
                )
            repository = repository_from_eval_input(self.eval_input)
            if (
                self.replay.repository_descriptor_digest != repository.digest()
                or self.replay.base_revision != repository.base_revision
                or self.replay.head_revision != repository.head_revision
            ):
                raise ValueError("repository replay is not exactly bound to EvalInput")
        elif target.kind is ReviewTargetKind.FROZEN_CONTEXT:
            if type(target) is not FrozenContextReviewTarget:
                raise TypeError("Frozen target has a non-canonical target type")
            if type(self.replay) is not FrozenContextReplay:
                raise TypeError("Frozen review target requires FrozenContextReplay")
            if (
                self.replay.bundle_id != target.bundle_id
                or self.replay.record_id != target.record_id
                or self.replay.context_ref != target.record_id
                or self.replay.context_format != target.context_format
                or self.replay.rendered_sha256 != target.rendered_sha256
                or self.replay.rendered_utf8_bytes != target.rendered_utf8_bytes
                or self.replay.source_binding_digest
                != target.source_binding_digest
            ):
                raise ValueError("frozen replay is not exactly bound to EvalInput")
        else:
            raise TypeError("EvidenceIntegrityChecker review target kind is invalid")

        grouped: Dict[str, list[CommandOutputAttestation]] = {}
        for attestation in self.command_attestations:
            grouped.setdefault(attestation.source_ref, []).append(attestation)
        attestation_index = {
            source_ref: tuple(
                sorted(
                    items,
                    key=lambda item: (
                        item.trial_id,
                        item.head_revision,
                        item.argv,
                        item.exit_code,
                        item.stream.value,
                        item.sha256,
                    ),
                )
            )
            for source_ref, items in sorted(grouped.items())
        }
        external_index = (
            {
                item.source_id: item
                for item in target.review_request.existing_ci_evidence
            }
            if type(target) is RepositoryReviewTarget
            else {}
        )
        object.__setattr__(
            self, "_attestations_by_source", MappingProxyType(attestation_index)
        )
        object.__setattr__(
            self,
            "_external_by_source",
            MappingProxyType(dict(sorted(external_index.items()))),
        )

    def check_item(self, evidence: SubmissionEvidence) -> EvidenceItemIntegrityResult:
        return self._check_item(evidence, {})

    def _check_item(
        self,
        evidence: SubmissionEvidence,
        source_cache: _SourceCache,
    ) -> EvidenceItemIntegrityResult:
        if type(evidence) is not SubmissionEvidence:
            raise TypeError("evidence must be a SubmissionEvidence")
        if (
            evidence.source.target_materialization_id
            != self.target_materialization_id
        ):
            return _item_result(
                evidence,
                (EvidenceReasonCode.TARGET_MATERIALIZATION_MISMATCH,),
            )
        target_kind = self.eval_input.review_target.kind
        if target_kind is ReviewTargetKind.FROZEN_CONTEXT:
            if evidence.source.kind is not EvidenceKind.FROZEN_CONTEXT:
                return _item_result(
                    evidence,
                    (EvidenceReasonCode.REPLAY_BINDING_MISMATCH,),
                )
            return self._check_frozen_context(evidence, source_cache)
        if evidence.source.kind is EvidenceKind.FROZEN_CONTEXT:
            return _item_result(
                evidence,
                (EvidenceReasonCode.REPLAY_BINDING_MISMATCH,),
            )
        if evidence.source.kind is EvidenceKind.REPOSITORY_FILE:
            return self._check_repository_file(evidence, source_cache)
        if evidence.source.kind is EvidenceKind.REPOSITORY_DIFF:
            return self._check_repository_diff(evidence, source_cache)
        if evidence.source.kind is EvidenceKind.COMMAND_OUTPUT:
            return self._check_command_output(evidence)
        if evidence.source.kind is EvidenceKind.EXTERNAL_RECORD:
            return self._check_external_record(evidence)
        raise AssertionError("unreachable EvidenceKind")

    def _check_frozen_context(
        self,
        evidence: SubmissionEvidence,
        source_cache: _SourceCache,
    ) -> EvidenceItemIntegrityResult:
        source = evidence.source
        if type(source) is not FrozenContextEvidenceSource:
            return _item_result(
                evidence, (EvidenceReasonCode.KIND_FIELD_MISMATCH,)
            )
        if type(self.replay) is not FrozenContextReplay:
            return _item_result(
                evidence, (EvidenceReasonCode.REPLAY_BINDING_MISMATCH,)
            )
        if source.context_ref != self.replay.context_ref:
            return _item_result(
                evidence, (EvidenceReasonCode.CONTEXT_REF_MISMATCH,)
            )
        if source.from_line > source.to_line:
            return _item_result(
                evidence, (EvidenceReasonCode.LINE_RANGE_REVERSED,)
            )

        record_key = (
            "frozen_record",
            self.target_materialization_id,
            source.context_ref,
        )
        record = source_cache.get(record_key)
        if record is None:
            record_reasons: Tuple[EvidenceReasonCode, ...] = ()
            record_data: Optional[bytes] = None
            line_starts: Optional[array] = None
            try:
                verified = self.replay.read_exact()
            except MaterializationError:
                record_reasons = (EvidenceReasonCode.REPLAY_BINDING_MISMATCH,)
            else:
                starts = array("I", (0,))
                for index, value in enumerate(verified):
                    if value != 0x0A or index + 1 >= len(verified):
                        continue
                    if len(starts) >= MAX_REPLAY_LINES:
                        record_reasons = (
                            EvidenceReasonCode.REPLAY_LINE_LIMIT_EXCEEDED,
                        )
                        break
                    starts.append(index + 1)
                if not record_reasons:
                    record_data = verified
                    line_starts = starts
            record = _FrozenReplayRecord(
                data=record_data,
                line_starts=line_starts,
                reasons=record_reasons,
            )
            source_cache[record_key] = record
        if type(record) is not _FrozenReplayRecord:
            raise AssertionError("Frozen replay cache key collision")
        if record.reasons:
            return _item_result(evidence, record.reasons)
        assert record.data is not None
        assert record.line_starts is not None
        if source.to_line > len(record.line_starts):
            return _item_result(
                evidence, (EvidenceReasonCode.LINE_RANGE_OUT_OF_BOUNDS,)
            )

        source_key = (
            EvidenceKind.FROZEN_CONTEXT.value,
            source.context_ref,
            source.from_line,
            source.to_line,
        )
        canonical_source = source_cache.get(source_key)
        if canonical_source is None:
            source_reasons: Tuple[EvidenceReasonCode, ...] = ()
            canonical_text: Optional[str] = None
            canonical_hash: Optional[str] = None
            start = record.line_starts[source.from_line - 1]
            end = (
                record.line_starts[source.to_line]
                if source.to_line < len(record.line_starts)
                else len(record.data)
            )
            raw = record.data[start:end]
            if len(raw) > MAX_EVIDENCE_EXCERPT_BYTES:
                source_reasons = (EvidenceReasonCode.EXCERPT_TOO_LARGE,)
            else:
                try:
                    canonical_text = raw.decode("utf-8", "strict")
                except UnicodeDecodeError:
                    canonical_text = None
                    source_reasons = (EvidenceReasonCode.CONTENT_NOT_UTF8,)
                else:
                    canonical_hash = hashlib.sha256(raw).hexdigest()
            canonical_source = _CanonicalSource(
                text=canonical_text,
                content_hash=canonical_hash,
                reasons=source_reasons,
            )
            source_cache[source_key] = canonical_source
        if type(canonical_source) is not _CanonicalSource:
            raise AssertionError("Frozen excerpt cache key collision")
        if canonical_source.reasons:
            return _item_result(evidence, canonical_source.reasons)
        assert canonical_source.text is not None
        assert canonical_source.content_hash is not None
        return _item_result(
            evidence,
            _content_reasons(
                evidence,
                canonical_source.text,
                canonical_source.content_hash,
            ),
        )

    def _check_repository_file(
        self,
        evidence: SubmissionEvidence,
        source_cache: _SourceCache,
    ) -> EvidenceItemIntegrityResult:
        source = evidence.source
        if type(source) is not RepositoryFileEvidenceSource:
            return _item_result(evidence, (EvidenceReasonCode.KIND_FIELD_MISMATCH,))
        reasons = []
        repository = repository_from_eval_input(self.eval_input)
        base = repository.base_revision
        head = repository.head_revision
        if source.revision not in (base, head):
            reasons.append(EvidenceReasonCode.REVISION_MISMATCH)
        if source.from_line > source.to_line:
            reasons.append(EvidenceReasonCode.LINE_RANGE_REVERSED)
        if reasons:
            return _item_result(evidence, tuple(reasons))

        try:
            canonical_repository_path(source.path)
        except (RepositoryLimitError, RepositoryPolicyError):
            return _item_result(evidence, (EvidenceReasonCode.PATH_INVALID,))
        source_key = (
            EvidenceKind.REPOSITORY_FILE.value,
            source.revision,
            source.path,
            source.from_line,
            source.to_line,
        )
        canonical_source = source_cache.get(source_key)
        if canonical_source is None:
            source_reasons: Tuple[EvidenceReasonCode, ...] = ()
            canonical_text: Optional[str] = None
            canonical_hash: Optional[str] = None
            try:
                raw = self.replay.read_file(
                    source.revision,
                    source.path,
                    max_bytes=MAX_GIT_BLOB_BYTES,
                )
            except RepositoryLimitError:
                source_reasons = (EvidenceReasonCode.EXCERPT_TOO_LARGE,)
            except RepositoryPolicyError:
                source_reasons = (EvidenceReasonCode.PATH_INVALID,)
            else:
                if raw is None:
                    source_reasons = (EvidenceReasonCode.PATH_NOT_FOUND,)
                else:
                    try:
                        text = raw.decode("utf-8", "strict")
                    except UnicodeDecodeError:
                        source_reasons = (EvidenceReasonCode.CONTENT_NOT_UTF8,)
                    else:
                        canonical = _canonical_line_excerpt(
                            text, source.from_line, source.to_line
                        )
                        if canonical.too_many_lines:
                            source_reasons = (
                                EvidenceReasonCode.REPLAY_LINE_LIMIT_EXCEEDED,
                            )
                        elif source.to_line > canonical.line_count:
                            source_reasons = (
                                EvidenceReasonCode.LINE_RANGE_OUT_OF_BOUNDS,
                            )
                        elif canonical.excerpt_too_large:
                            source_reasons = (
                                EvidenceReasonCode.EXCERPT_TOO_LARGE,
                            )
                        else:
                            assert canonical.excerpt is not None
                            canonical_text = canonical.excerpt
                            canonical_hash = hashlib.sha256(
                                canonical_text.encode("utf-8")
                            ).hexdigest()
            canonical_source = _CanonicalSource(
                text=canonical_text,
                content_hash=canonical_hash,
                reasons=source_reasons,
            )
            source_cache[source_key] = canonical_source
        if canonical_source.reasons:
            return _item_result(evidence, canonical_source.reasons)
        assert canonical_source.text is not None
        assert canonical_source.content_hash is not None
        return _item_result(
            evidence,
            _content_reasons(
                evidence,
                canonical_source.text,
                canonical_source.content_hash,
            ),
        )

    def _check_repository_diff(
        self,
        evidence: SubmissionEvidence,
        source_cache: _SourceCache,
    ) -> EvidenceItemIntegrityResult:
        source = evidence.source
        if type(source) is not RepositoryDiffEvidenceSource:
            return _item_result(evidence, (EvidenceReasonCode.KIND_FIELD_MISMATCH,))
        reasons = []
        repository = repository_from_eval_input(self.eval_input)
        if (
            source.base_revision != repository.base_revision
            or source.head_revision != repository.head_revision
        ):
            reasons.append(EvidenceReasonCode.REVISION_MISMATCH)
        if reasons:
            return _item_result(evidence, tuple(reasons))

        source_key = (
            EvidenceKind.REPOSITORY_DIFF.value,
            source.base_revision,
            source.head_revision,
            source.path,
        )
        canonical_source = source_cache.get(source_key)
        if canonical_source is None:
            source_reasons = ()
            canonical_text = None
            canonical_hash = None
            try:
                exists = self.replay.contains_path(
                    repository.base_revision, source.path
                ) or self.replay.contains_path(
                    repository.head_revision, source.path
                )
            except (RepositoryLimitError, RepositoryPolicyError):
                source_reasons = (EvidenceReasonCode.PATH_INVALID,)
            else:
                if not exists:
                    source_reasons = (EvidenceReasonCode.PATH_NOT_FOUND,)
                else:
                    try:
                        raw = self.replay.diff(
                            source.path,
                            max_bytes=MAX_EVIDENCE_EXCERPT_BYTES,
                        )
                    except RepositoryLimitError:
                        source_reasons = (EvidenceReasonCode.EXCERPT_TOO_LARGE,)
                    except RepositoryPolicyError:
                        source_reasons = (EvidenceReasonCode.PATH_INVALID,)
                    else:
                        try:
                            canonical_text = raw.decode("utf-8", "strict")
                        except UnicodeDecodeError:
                            canonical_text = None
                            source_reasons = (
                                EvidenceReasonCode.CONTENT_NOT_UTF8,
                            )
                        else:
                            canonical_hash = hashlib.sha256(raw).hexdigest()
            canonical_source = _CanonicalSource(
                text=canonical_text,
                content_hash=canonical_hash,
                reasons=source_reasons,
            )
            source_cache[source_key] = canonical_source
        if canonical_source.reasons:
            return _item_result(evidence, canonical_source.reasons)
        assert canonical_source.text is not None
        assert canonical_source.content_hash is not None
        return _item_result(
            evidence,
            _content_reasons(
                evidence,
                canonical_source.text,
                canonical_source.content_hash,
            ),
        )

    def _check_command_output(
        self, evidence: SubmissionEvidence
    ) -> EvidenceItemIntegrityResult:
        source = evidence.source
        if type(source) is not CommandOutputEvidenceSource:
            return _item_result(evidence, (EvidenceReasonCode.KIND_FIELD_MISMATCH,))
        head = repository_from_eval_input(self.eval_input).head_revision
        candidates = self._attestations_by_source.get(source.artifact_ref, ())
        if not candidates:
            return _item_result(
                evidence, (EvidenceReasonCode.ATTESTATION_NOT_FOUND,)
            )
        trial_candidates = tuple(
            item for item in candidates if item.trial_id == self.trial_id
        )
        if not trial_candidates:
            return _item_result(
                evidence, (EvidenceReasonCode.ATTESTATION_TRIAL_MISMATCH,)
            )
        bound = tuple(
            item for item in trial_candidates if item.head_revision == head
        )
        if not bound:
            return _item_result(
                evidence, (EvidenceReasonCode.ATTESTATION_REVISION_MISMATCH,)
            )
        if len(bound) != 1:
            return _item_result(
                evidence, (EvidenceReasonCode.ATTESTATION_AMBIGUOUS,)
            )
        attestation = bound[0]
        comparison_reasons = []
        if source.command != attestation.argv:
            comparison_reasons.append(EvidenceReasonCode.COMMAND_MISMATCH)
        if source.exit_code != attestation.exit_code:
            comparison_reasons.append(EvidenceReasonCode.EXIT_CODE_MISMATCH)
        if source.stream is not attestation.stream:
            comparison_reasons.append(EvidenceReasonCode.STREAM_MISMATCH)
        try:
            canonical_text = attestation.output_bytes.decode("utf-8", "strict")
        except UnicodeDecodeError:
            comparison_reasons.append(EvidenceReasonCode.CONTENT_NOT_UTF8)
            return _item_result(evidence, tuple(comparison_reasons))
        comparison_reasons.extend(
            _content_reasons(evidence, canonical_text, attestation.sha256)
        )
        return _item_result(evidence, tuple(comparison_reasons))

    def _check_external_record(
        self, evidence: SubmissionEvidence
    ) -> EvidenceItemIntegrityResult:
        evidence_source = evidence.source
        if type(evidence_source) is not ExternalRecordEvidenceSource:
            return _item_result(evidence, (EvidenceReasonCode.KIND_FIELD_MISMATCH,))
        source = self._external_by_source.get(evidence_source.source_ref)
        if source is None:
            return _item_result(
                evidence, (EvidenceReasonCode.EXTERNAL_RECORD_NOT_FOUND,)
            )
        return _item_result(
            evidence,
            _content_reasons(evidence, source.text, source.content_hash),
        )

    def check_finding(
        self,
        finding: SubmissionFinding,
        evidence_items: Tuple[SubmissionEvidence, ...],
    ) -> EvidenceIntegrityResult:
        if type(finding) is not SubmissionFinding:
            raise TypeError("finding must be a SubmissionFinding")
        index = self._evidence_index(evidence_items)
        return self._check_finding_indexed(finding, index, {}, {})

    @staticmethod
    def _evidence_index(
        evidence_items: Tuple[SubmissionEvidence, ...]
    ) -> Mapping[str, SubmissionEvidence]:
        if type(evidence_items) is not tuple:
            raise TypeError("evidence_items must be an immutable tuple")
        if len(evidence_items) > MAX_EVIDENCE_ITEMS:
            raise ValueError("evidence_items exceeds its item limit")
        if any(type(item) is not SubmissionEvidence for item in evidence_items):
            raise TypeError("evidence_items contains a non-SubmissionEvidence item")
        index: Dict[str, SubmissionEvidence] = {}
        for item in evidence_items:
            if item.evidence_id in index:
                raise ValueError("evidence_items contains a duplicate evidence ID")
            index[item.evidence_id] = item
        return MappingProxyType(dict(sorted(index.items())))

    def _check_finding_indexed(
        self,
        finding: SubmissionFinding,
        index: Mapping[str, SubmissionEvidence],
        item_cache: Dict[str, EvidenceItemIntegrityResult],
        source_cache: _SourceCache,
    ) -> EvidenceIntegrityResult:
        if not finding.evidence_refs:
            diagnostic = EvidenceDiagnostic(
                reason_code=EvidenceReasonCode.NO_EVIDENCE_REFS,
                finding_id=finding.finding_id,
            )
            return EvidenceIntegrityResult(
                finding_id=finding.finding_id,
                integrity=EvidenceIntegrity.MISSING,
                referenced_evidence_ids=(),
                item_results=(),
                diagnostics=(diagnostic,),
            )

        item_results = []
        diagnostics = []
        seen_refs = set()
        dangling = False
        invalid = False
        for ref_index, evidence_id in enumerate(finding.evidence_refs):
            if evidence_id in seen_refs:
                diagnostics.append(
                    EvidenceDiagnostic(
                        reason_code=EvidenceReasonCode.DUPLICATE_REF,
                        evidence_id=evidence_id,
                        finding_id=finding.finding_id,
                        ref_index=ref_index,
                    )
                )
            else:
                seen_refs.add(evidence_id)
            evidence = index.get(evidence_id)
            if evidence is None:
                dangling = True
                diagnostics.append(
                    EvidenceDiagnostic(
                        reason_code=EvidenceReasonCode.DANGLING_REF,
                        evidence_id=evidence_id,
                        finding_id=finding.finding_id,
                        ref_index=ref_index,
                    )
                )
                continue
            item_result = item_cache.get(evidence_id)
            if item_result is None:
                item_result = self._check_item(evidence, source_cache)
                item_cache[evidence_id] = item_result
            item_results.append(item_result)
            if item_result.integrity is EvidenceIntegrity.INVALID:
                invalid = True
            diagnostics.extend(
                EvidenceDiagnostic(
                    reason_code=item_diagnostic.reason_code,
                    evidence_id=evidence_id,
                    finding_id=finding.finding_id,
                    ref_index=ref_index,
                )
                for item_diagnostic in item_result.diagnostics
            )

        if dangling:
            integrity = EvidenceIntegrity.MISSING
        elif invalid:
            integrity = EvidenceIntegrity.INVALID
        else:
            integrity = EvidenceIntegrity.VALID
        return EvidenceIntegrityResult(
            finding_id=finding.finding_id,
            integrity=integrity,
            referenced_evidence_ids=finding.evidence_refs,
            item_results=tuple(item_results),
            diagnostics=tuple(diagnostics),
        )

    def check(
        self,
        finding: SubmissionFinding,
        evidence_items: Tuple[SubmissionEvidence, ...],
    ) -> EvidenceIntegrityResult:
        return self.check_finding(finding, evidence_items)

    def check_all(
        self,
        findings: Tuple[SubmissionFinding, ...],
        evidence_items: Tuple[SubmissionEvidence, ...],
    ) -> Tuple[EvidenceIntegrityResult, ...]:
        if type(findings) is not tuple or len(findings) > MAX_FINDINGS:
            raise TypeError("findings must be a bounded immutable tuple")
        if any(type(item) is not SubmissionFinding for item in findings):
            raise TypeError("findings contains a non-SubmissionFinding item")
        if len({item.finding_id for item in findings}) != len(findings):
            raise ValueError("findings contains a duplicate finding ID")
        index = self._evidence_index(evidence_items)
        # Validate lazily but share the cache across every Finding.  Unreferenced
        # Submission material must not trigger Git work or alter a Finding-level
        # result, while one referenced item is still replayed at most once.
        item_cache: Dict[str, EvidenceItemIntegrityResult] = {}
        source_cache: _SourceCache = {}
        return tuple(
            self._check_finding_indexed(
                finding,
                index,
                item_cache,
                source_cache,
            )
            for finding in sorted(findings, key=lambda item: item.finding_id)
        )


__all__ = [
    "COMMAND_OUTPUT_ATTESTATION_SCHEMA_VERSION",
    "EVIDENCE_INTEGRITY_POLICY_VERSION",
    "MAX_COMMAND_OUTPUT_ATTESTATION_BYTES",
    "MAX_COMMAND_OUTPUT_ATTESTATIONS",
    "MAX_REPLAY_LINES",
    "CommandOutputAttestation",
    "EvidenceDiagnostic",
    "EvidenceIntegrityChecker",
    "EvidenceIntegrityResult",
    "EvidenceItemIntegrityResult",
    "EvidenceReasonCode",
]
