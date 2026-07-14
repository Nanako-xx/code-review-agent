"""Fail-closed validation for durable-memory source references.

The module is intentionally independent from the memory Store and lifecycle.  It
accepts only the closed ``SourceRef`` union from :mod:`memory_models`, re-reads
evidence from its authority, and returns a content-free validation report.  No
validated source excerpt or declaration body is retained by the report.

Repository evidence is read from an exact commit object rather than the working
tree.  Session evidence is hydrated through ``SessionStore`` and
``ObservationStore`` before its descriptor, revision binding, and digest are
accepted.  Human declarations require an in-memory, explicitly trusted request
record; the fields carried by a ``HumanDeclarationSourceRef`` are never treated as
authority by themselves.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import hmac
import io
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
import subprocess
import tokenize
import unicodedata
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from review_agent.artifacts import artifact_schema
from review_agent.memory_identity import repository_key as canonical_repository_key
from review_agent.memory_models import (
    CandidateAuthorityReceipt,
    CandidateStatus,
    GitCommitSourceRef,
    HumanDeclarationAuthority,
    HumanDeclarationOrigin,
    HumanDeclarationSourceRef,
    MemoryCandidate,
    ObservationSourceRef,
    ProducerType,
    RepositoryRangeSourceRef,
    RepositorySymbolSourceRef,
    Sensitivity,
    SessionArtifactSourceRef,
    SourceRef,
    SourceRefType,
    SymbolHashKind,
    canonical_json,
    canonical_sha256,
    validate_stable_id,
)
from review_agent.observations import Observation, ObservationStore
from review_agent.revision import (
    RevisionResolver,
    normalize_repository_identity_path,
    sanitized_git_environment,
)
from review_agent.session import (
    ArtifactDescriptor,
    PhaseStatus,
    SessionManifest,
    SupplementalTaskStatus,
)
from review_agent.session_store import SessionStore


SOURCE_VALIDATION_SCHEMA_VERSION = 1
DEFAULT_MAX_SOURCE_BYTES = 8 * 1024 * 1024
MAX_VALIDATION_SOURCE_REFS = 64

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT_ID_PATTERN = re.compile(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")
_SAFE_REVIEW_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_WINDOWS_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:[/\\]")
_OBSERVATION_SCHEMA = "observation_log_jsonl_v1"
_REGULAR_GIT_MODES = frozenset({"100644", "100755"})
_SOURCE_REF_TYPES = (
    RepositoryRangeSourceRef,
    RepositorySymbolSourceRef,
    GitCommitSourceRef,
    ObservationSourceRef,
    SessionArtifactSourceRef,
    HumanDeclarationSourceRef,
)
_SENSITIVE_PATH_COMPONENTS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".env",
        ".netrc",
        ".npmrc",
        ".pypirc",
        "credentials",
        "credentials.json",
        "credentials.yaml",
        "credentials.yml",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
        "secrets.json",
        "secrets.yaml",
        "secrets.yml",
    }
)
_WINDOWS_RESERVED_COMPONENTS = frozenset(
    {
        "aux",
        "clock$",
        "con",
        "nul",
        "prn",
        *("com%d" % index for index in range(1, 10)),
        *("lpt%d" % index for index in range(1, 10)),
        "com¹",
        "com²",
        "com³",
        "lpt¹",
        "lpt²",
        "lpt³",
    }
)
_WINDOWS_INVALID_PATH_CHARACTERS = frozenset('<>:"|?*')
_WINDOWS_SHORT_NAME_PATTERN = re.compile(
    r"^[^.]{1,6}~[1-9][0-9]*(?:\..*)?$",
    re.IGNORECASE,
)
_SOURCE_ELIGIBLE_SESSION_SCHEMAS = frozenset(
    {
        "change_summary_v1",
        "completion_check_v1",
        "evidence_reconciliation_v1",
        "final_risk_assessment_v1",
        "incremental_priority_map_v1",
        "intent_decision_v1",
        "intent_packet_v1",
        "intent_packet_v2",
        "multi_reviewer_result_v1",
        "observation_log_jsonl_v1",
        "planning_summary_v1",
        "portfolio_plan_v1",
        "quality_gate_results_v1",
        "reconciliation_analysis_summary_v1",
        "reconciliation_packet_v1",
        "reconciliation_prepass_v1",
        "repository_intelligence_v1",
        "review_brief_v1",
        "review_report_markdown_v1",
        "reviewer_assignments_v1",
        "risk_assessment_v1",
        "semantic_reconciliation_v1",
        "supplemental_plan_v1",
        "supplemental_summary_v1",
        "supplemental_task_spec_v1",
        "supplemental_wave_summary_v1",
    }
)
_REVIEWER_TASK_ARTIFACT_PATTERN = re.compile(r"^reviewer_([0-9]+)_")
_SUPPLEMENTAL_TASK_ARTIFACT_PREFIX = "supplemental_task_"
_SUPPLEMENTAL_WAVE_ARTIFACT_PREFIX = "supplemental_wave_"


def _canonical_sha256_digest(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError("%s must be a SHA-256 digest" % field_name)
    normalized = value.casefold()
    if _SHA256_PATTERN.fullmatch(normalized) is None:
        raise ValueError("%s must be a SHA-256 digest" % field_name)
    return normalized


def _canonical_authority_binding(
    locator_repository_key: str,
    authority_repository_key: str,
    binding_id: Optional[str],
) -> Optional[str]:
    direct = locator_repository_key == authority_repository_key
    if direct:
        if binding_id is not None:
            raise ValueError("direct authority requires binding_id to be None")
        return None
    if binding_id is None:
        raise ValueError("bound authority requires a canonical binding_id")
    return validate_stable_id(binding_id, "RB", "binding_id")


def candidate_authority_resolution_hash(
    locator_repository_key: str,
    authority_repository_key: str,
    *,
    binding_id: Optional[str] = None,
) -> str:
    """Hash one canonical direct or bound repository-authority resolution."""

    locator = _canonical_sha256_digest(
        locator_repository_key,
        "locator_repository_key",
    )
    authority = _canonical_sha256_digest(
        authority_repository_key,
        "authority_repository_key",
    )
    binding = _canonical_authority_binding(locator, authority, binding_id)
    return canonical_sha256(
        {
            "schema_version": SOURCE_VALIDATION_SCHEMA_VERSION,
            "locator_repository_key": locator,
            "authority_repository_key": authority,
            "binding_id": binding,
        }
    )


class SourceValidationCode(str, Enum):
    INVALID_INPUT = "invalid_input"
    INVALID_CONFIGURATION = "invalid_configuration"
    UNTYPED_SOURCE = "untyped_source"
    SOURCE_NOT_ALLOWLISTED = "source_not_allowlisted"
    RUNTIME_PROVENANCE_REQUIRED = "runtime_provenance_required"
    SENSITIVITY_BLOCKED = "sensitivity_blocked"
    SENSITIVE_CONTENT = "sensitive_content"
    VALIDATION_SKIPPED = "validation_skipped"
    UNSAFE_PATH = "unsafe_path"
    REPOSITORY_UNAVAILABLE = "repository_unavailable"
    REVISION_NOT_FOUND = "revision_not_found"
    REVISION_MISMATCH = "revision_mismatch"
    REVISION_LINEAGE_UNAUTHORIZED = "revision_lineage_unauthorized"
    SOURCE_NOT_FOUND = "source_not_found"
    SOURCE_NOT_REGULAR = "source_not_regular"
    SOURCE_TOO_LARGE = "source_too_large"
    SOURCE_ENCODING_INVALID = "source_encoding_invalid"
    RANGE_OUT_OF_BOUNDS = "range_out_of_bounds"
    HASH_MISMATCH = "hash_mismatch"
    SYMBOL_NOT_FOUND = "symbol_not_found"
    SYMBOL_AMBIGUOUS = "symbol_ambiguous"
    SYMBOL_UNSUPPORTED = "symbol_unsupported"
    SESSION_NOT_FOUND = "session_not_found"
    SESSION_UNTRUSTED = "session_untrusted"
    REVIEW_ID_MISMATCH = "review_id_mismatch"
    REPOSITORY_MISMATCH = "repository_mismatch"
    DESCRIPTOR_NOT_FOUND = "descriptor_not_found"
    DESCRIPTOR_SCHEMA_MISMATCH = "descriptor_schema_mismatch"
    SESSION_ARTIFACT_INELIGIBLE = "session_artifact_ineligible"
    REVISION_BINDING_MISMATCH = "revision_binding_mismatch"
    SESSION_ARTIFACT_INVALID = "session_artifact_invalid"
    OBSERVATION_NOT_FOUND = "observation_not_found"
    OBSERVATION_UNTRUSTED = "observation_untrusted"
    HUMAN_DECLARATION_UNAUTHORIZED = "human_declaration_unauthorized"
    HUMAN_DECLARATION_MISMATCH = "human_declaration_mismatch"
    CANDIDATE_AUTHORITY_MISMATCH = "candidate_authority_mismatch"
    VALIDATION_REPORT_MISMATCH = "validation_report_mismatch"
    AUTHORITY_RECEIPT_INVALID = "authority_receipt_invalid"
    PRODUCER_POLICY_UNAUTHORIZED = "producer_policy_unauthorized"
    INTERNAL_ERROR = "internal_error"


_ISSUE_MESSAGES = {
    SourceValidationCode.INVALID_INPUT: "validation input is not a supported typed value",
    SourceValidationCode.INVALID_CONFIGURATION: "source validator configuration is invalid",
    SourceValidationCode.UNTYPED_SOURCE: "source reference is not a supported typed SourceRef",
    SourceValidationCode.SOURCE_NOT_ALLOWLISTED: "source reference is absent from the Runtime allowlist",
    SourceValidationCode.RUNTIME_PROVENANCE_REQUIRED: "candidate validation requires trusted Runtime provenance",
    SourceValidationCode.SENSITIVITY_BLOCKED: "blocked content cannot be validated or persisted",
    SourceValidationCode.SENSITIVE_CONTENT: "sensitive content was detected and must not be retained",
    SourceValidationCode.VALIDATION_SKIPPED: "source validation was skipped after a fail-closed decision",
    SourceValidationCode.UNSAFE_PATH: "source path is not a safe canonical relative path",
    SourceValidationCode.REPOSITORY_UNAVAILABLE: "repository authority is unavailable",
    SourceValidationCode.REVISION_NOT_FOUND: "exact source revision does not exist",
    SourceValidationCode.REVISION_MISMATCH: "source revision did not resolve to the exact object ID",
    SourceValidationCode.REVISION_LINEAGE_UNAUTHORIZED: "candidate revision is outside the trusted Runtime lineage",
    SourceValidationCode.SOURCE_NOT_FOUND: "source does not exist at the exact revision",
    SourceValidationCode.SOURCE_NOT_REGULAR: "source is not a regular Git blob or regular file",
    SourceValidationCode.SOURCE_TOO_LARGE: "source exceeds the bounded validation size",
    SourceValidationCode.SOURCE_ENCODING_INVALID: "source is not valid bounded UTF-8 text",
    SourceValidationCode.RANGE_OUT_OF_BOUNDS: "source line range is outside the exact revision content",
    SourceValidationCode.HASH_MISMATCH: "source digest does not match the trusted content",
    SourceValidationCode.SYMBOL_NOT_FOUND: "symbol does not exist at the exact revision",
    SourceValidationCode.SYMBOL_AMBIGUOUS: "symbol reference is ambiguous at the exact revision",
    SourceValidationCode.SYMBOL_UNSUPPORTED: "symbol source format cannot be validated deterministically",
    SourceValidationCode.SESSION_NOT_FOUND: "referenced review Session does not exist",
    SourceValidationCode.SESSION_UNTRUSTED: "referenced review Session failed trusted hydration",
    SourceValidationCode.REVIEW_ID_MISMATCH: "Session review ID does not match the source reference",
    SourceValidationCode.REPOSITORY_MISMATCH: "Session belongs to a different repository authority",
    SourceValidationCode.DESCRIPTOR_NOT_FOUND: "Session artifact descriptor is not registered",
    SourceValidationCode.DESCRIPTOR_SCHEMA_MISMATCH: "Session artifact schema does not match the descriptor",
    SourceValidationCode.SESSION_ARTIFACT_INELIGIBLE: "Session artifact type is not eligible as durable evidence",
    SourceValidationCode.REVISION_BINDING_MISMATCH: "source revision binding does not match the trusted authority",
    SourceValidationCode.SESSION_ARTIFACT_INVALID: "Session artifact failed descriptor or digest validation",
    SourceValidationCode.OBSERVATION_NOT_FOUND: "Observation is not present in a registered Session authority",
    SourceValidationCode.OBSERVATION_UNTRUSTED: "Observation authority failed hydration or artifact validation",
    SourceValidationCode.HUMAN_DECLARATION_UNAUTHORIZED: "human declaration has no explicit trusted request",
    SourceValidationCode.HUMAN_DECLARATION_MISMATCH: "human declaration fields do not match the trusted request",
    SourceValidationCode.CANDIDATE_AUTHORITY_MISMATCH: "candidate authority does not match the trusted Runtime context",
    SourceValidationCode.VALIDATION_REPORT_MISMATCH: "source validation report does not match the exact candidate authority context",
    SourceValidationCode.AUTHORITY_RECEIPT_INVALID: "candidate authority receipt failed canonical validation",
    SourceValidationCode.PRODUCER_POLICY_UNAUTHORIZED: "candidate producer is outside the current Runtime source policy",
    SourceValidationCode.INTERNAL_ERROR: "source validation failed closed",
}


class SensitiveContentKind(str, Enum):
    PRIVATE_KEY = "private_key"
    KNOWN_TOKEN = "known_token"
    CREDENTIAL_FIELD = "credential_field"
    AUTHORIZATION_HEADER = "authorization_header"
    AUTHENTICATED_URL = "authenticated_url"
    DUPLICATE_JSON_KEY = "duplicate_json_key"


@dataclass(frozen=True)
class TrustedCandidateProvenance:
    """Executor-owned authority for validating one candidate proposal.

    ``MemoryCandidate.producer`` is persisted proposal metadata and is never an
    authority signal.  The Runtime supplies this record out of band so a model
    cannot opt itself out of source allowlisting or choose an unrelated commit
    lineage.
    """

    origin: ProducerType
    review_id: str
    target_head_sha: str
    locator_repository_key: str
    authority_repository_key: str
    authority_resolution_hash: str
    binding_id: Optional[str] = None
    allowed_source_refs: Tuple[SourceRef, ...] = field(
        default_factory=tuple,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.origin, ProducerType):
            raise ValueError("origin must be a ProducerType")
        if (
            not isinstance(self.review_id, str)
            or not _SAFE_REVIEW_ID_PATTERN.fullmatch(self.review_id)
            or not _is_safe_portable_path_component(self.review_id)
        ):
            raise ValueError("review_id must be a safe canonical Session ID")
        if (
            not isinstance(self.target_head_sha, str)
            or not _GIT_OBJECT_ID_PATTERN.fullmatch(self.target_head_sha)
        ):
            raise ValueError("target_head_sha must be a full Git object ID")
        locator_repository_key = _canonical_sha256_digest(
            self.locator_repository_key,
            "locator_repository_key",
        )
        authority_repository_key = _canonical_sha256_digest(
            self.authority_repository_key,
            "authority_repository_key",
        )
        binding_id = _canonical_authority_binding(
            locator_repository_key,
            authority_repository_key,
            self.binding_id,
        )
        resolution_hash = _canonical_sha256_digest(
            self.authority_resolution_hash,
            "authority_resolution_hash",
        )
        expected_resolution_hash = candidate_authority_resolution_hash(
            locator_repository_key,
            authority_repository_key,
            binding_id=binding_id,
        )
        if not hmac.compare_digest(resolution_hash, expected_resolution_hash):
            raise ValueError(
                "authority_resolution_hash does not match the canonical "
                "locator authority resolution"
            )
        try:
            values = tuple(self.allowed_source_refs)
        except TypeError as error:
            raise ValueError("allowed_source_refs must be typed SourceRef values") from error
        if any(type(item) not in _SOURCE_REF_TYPES for item in values):
            raise ValueError("allowed_source_refs must be typed SourceRef values")
        for item in values:
            try:
                hydrated = SourceRef.from_dict(item.to_dict())
            except (TypeError, ValueError):
                raise ValueError(
                    "allowed_source_refs must be canonical SourceRef values"
                ) from None
            if type(hydrated) is not type(item) or hydrated != item:
                raise ValueError(
                    "allowed_source_refs must be canonical SourceRef values"
                )
        if len(values) > MAX_VALIDATION_SOURCE_REFS:
            raise ValueError("allowed_source_refs exceeds the validation limit")
        keys = [_source_ref_key(item) for item in values]
        if len(keys) != len(set(keys)):
            raise ValueError("allowed_source_refs must not contain duplicates")
        object.__setattr__(self, "target_head_sha", self.target_head_sha.casefold())
        object.__setattr__(self, "locator_repository_key", locator_repository_key)
        object.__setattr__(self, "authority_repository_key", authority_repository_key)
        object.__setattr__(self, "authority_resolution_hash", resolution_hash)
        object.__setattr__(self, "binding_id", binding_id)
        object.__setattr__(
            self,
            "allowed_source_refs",
            tuple(sorted(values, key=lambda item: item.to_json())),
        )


@dataclass(frozen=True)
class SensitiveContentFinding:
    kind: SensitiveContentKind
    field_name: str

    def to_dict(self) -> Dict[str, str]:
        return {"kind": self.kind.value, "field": self.field_name}


@dataclass(frozen=True)
class SensitiveContentScan:
    findings: Tuple[SensitiveContentFinding, ...]

    @property
    def safe(self) -> bool:
        return not self.findings

    def to_dict(self) -> Dict[str, Any]:
        return {
            "safe": self.safe,
            "findings": [item.to_dict() for item in self.findings],
        }


@dataclass(frozen=True)
class ValidationIssue:
    code: SourceValidationCode
    message: str
    source_index: Optional[int] = None
    source_type: Optional[SourceRefType] = None
    field_name: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.code, SourceValidationCode):
            raise ValueError("validation issue code must be a SourceValidationCode")
        if self.message != _ISSUE_MESSAGES[self.code]:
            raise ValueError("validation issue message must use the stable catalog")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "source_index": self.source_index,
            "source_type": (
                None if self.source_type is None else self.source_type.value
            ),
            "field": self.field_name,
        }


@dataclass(frozen=True)
class SourceValidationResult:
    source_index: int
    source_type: Optional[SourceRefType]
    source_ref_hash: Optional[str]
    valid: bool
    verified_content_hash: Optional[str]
    revision_binding: Optional[str]
    content_size_bytes: int
    issues: Tuple[ValidationIssue, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_index": self.source_index,
            "source_type": (
                None if self.source_type is None else self.source_type.value
            ),
            "source_ref_hash": self.source_ref_hash,
            "valid": self.valid,
            "verified_content_hash": self.verified_content_hash,
            "revision_binding": self.revision_binding,
            "content_size_bytes": self.content_size_bytes,
            "issues": [item.to_dict() for item in self.issues],
        }


@dataclass(frozen=True)
class SensitivityDecision:
    declared: Sensitivity
    effective: Sensitivity
    content_persistable: bool
    remote_sendable: bool
    retain_content: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "declared": self.declared.value,
            "effective": self.effective.value,
            "content_persistable": self.content_persistable,
            "remote_sendable": self.remote_sendable,
            "retain_content": self.retain_content,
        }


@dataclass(frozen=True)
class SourceValidationReport:
    sensitivity: SensitivityDecision
    source_results: Tuple[SourceValidationResult, ...]
    issues: Tuple[ValidationIssue, ...]
    subject_id: Optional[str] = None
    candidate_validation_context_hash: Optional[str] = None
    authority_resolution_hash: Optional[str] = None
    schema_version: int = SOURCE_VALIDATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != SOURCE_VALIDATION_SCHEMA_VERSION:
            raise ValueError(
                "source validation report schema_version must be %d"
                % SOURCE_VALIDATION_SCHEMA_VERSION
            )
        if type(self.sensitivity) is not SensitivityDecision:
            raise ValueError("source validation report sensitivity is invalid")
        if not isinstance(self.source_results, tuple) or any(
            type(item) is not SourceValidationResult
            for item in self.source_results
        ):
            raise ValueError("source validation report results are invalid")
        if not isinstance(self.issues, tuple) or any(
            type(item) is not ValidationIssue for item in self.issues
        ):
            raise ValueError("source validation report issues are invalid")
        context_hash = self.candidate_validation_context_hash
        resolution_hash = self.authority_resolution_hash
        if (context_hash is None) != (resolution_hash is None):
            raise ValueError(
                "candidate validation context and authority resolution hashes "
                "must be present together"
            )
        if context_hash is not None:
            object.__setattr__(
                self,
                "candidate_validation_context_hash",
                _canonical_sha256_digest(
                    context_hash,
                    "candidate_validation_context_hash",
                ),
            )
            object.__setattr__(
                self,
                "authority_resolution_hash",
                _canonical_sha256_digest(
                    resolution_hash,
                    "authority_resolution_hash",
                ),
            )

    @property
    def valid(self) -> bool:
        return (
            not self.issues
            and all(item.valid for item in self.source_results)
            and self.sensitivity.effective is not Sensitivity.BLOCKED
        )

    @property
    def persistable(self) -> bool:
        return self.valid and self.sensitivity.content_persistable

    @property
    def remote_sendable(self) -> bool:
        return self.valid and self.sensitivity.remote_sendable

    @property
    def retain_content(self) -> bool:
        return self.sensitivity.retain_content

    @property
    def report_hash(self) -> str:
        return canonical_sha256(self._payload())

    def require_valid(self) -> "SourceValidationReport":
        if not self.valid:
            code = (
                self.issues[0].code
                if self.issues
                else SourceValidationCode.INTERNAL_ERROR
            )
            raise SourceValidationError(code, report=self)
        return self

    def _payload(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "subject_id": self.subject_id,
            "candidate_validation_context_hash": (
                self.candidate_validation_context_hash
            ),
            "authority_resolution_hash": self.authority_resolution_hash,
            "valid": self.valid,
            "persistable": self.persistable,
            "remote_sendable": self.remote_sendable,
            "retain_content": self.retain_content,
            "sensitivity": self.sensitivity.to_dict(),
            "source_results": [item.to_dict() for item in self.source_results],
            "issues": [item.to_dict() for item in self.issues],
        }

    def to_dict(self) -> Dict[str, Any]:
        payload = self._payload()
        payload["report_hash"] = self.report_hash
        return payload

    def to_json(self) -> str:
        return canonical_json(self.to_dict())


@dataclass(frozen=True)
class CandidateAuthorityRestoration:
    """Runtime-owned authority recovered only after strict receipt revalidation."""

    provenance: TrustedCandidateProvenance
    human_declarations: Tuple[HumanDeclarationAuthority, ...]

    def __post_init__(self) -> None:
        if type(self.provenance) is not TrustedCandidateProvenance:
            raise ValueError("provenance must be TrustedCandidateProvenance")
        values = tuple(self.human_declarations)
        if any(type(item) is not HumanDeclarationAuthority for item in values):
            raise ValueError(
                "human_declarations must contain exact "
                "HumanDeclarationAuthority values"
            )
        canonical = []
        for item in values:
            try:
                hydrated = HumanDeclarationAuthority.from_dict(item.to_dict())
            except (TypeError, ValueError):
                raise ValueError(
                    "human_declarations must contain canonical authority values"
                ) from None
            if hydrated != item:
                raise ValueError(
                    "human_declarations must contain canonical authority values"
                )
            canonical.append(item)
        if len({item.source_ref.to_json() for item in canonical}) != len(canonical):
            raise ValueError("human_declarations must not contain duplicates")
        object.__setattr__(
            self,
            "human_declarations",
            tuple(sorted(canonical, key=lambda item: item.to_json())),
        )


class SourceValidationError(ValueError):
    """A stable, content-free exception for callers requiring a valid report."""

    def __init__(
        self,
        code: SourceValidationCode,
        *,
        report: Optional[SourceValidationReport] = None,
    ) -> None:
        self.code = code
        self.report = report
        super().__init__("source validation failed: %s" % code.value)


@dataclass(frozen=True)
class TrustedHumanDeclaration:
    request_id: str
    actor: str
    created_at: str
    declaration: str = field(repr=False)
    origin: HumanDeclarationOrigin
    review_id: Optional[str] = None
    declaration_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.origin, HumanDeclarationOrigin):
            raise ValueError("origin must be a HumanDeclarationOrigin")
        if not isinstance(self.declaration, str) or not self.declaration.strip():
            raise ValueError("declaration must be a non-empty string")
        if "\x00" in self.declaration:
            raise ValueError("declaration must not contain NUL characters")
        digest = human_declaration_hash(self.declaration)
        # Reuse the canonical model's ID, actor, timestamp, review ID, and digest
        # checks rather than maintaining a second schema here.
        HumanDeclarationSourceRef(
            request_id=self.request_id,
            actor=self.actor,
            declaration_hash=digest,
            created_at=self.created_at,
            review_id=self.review_id,
        )
        if (
            self.origin is HumanDeclarationOrigin.USER_REQUEST
            and self.review_id is None
        ):
            raise ValueError("user request declarations must bind a review_id")
        if (
            self.review_id is not None
            and (
                not _SAFE_REVIEW_ID_PATTERN.fullmatch(self.review_id)
                or not _is_safe_portable_path_component(self.review_id)
            )
        ):
            raise ValueError("review_id must be a safe canonical Session ID")
        object.__setattr__(self, "declaration_hash", digest)

    def to_source_ref(self) -> HumanDeclarationSourceRef:
        return HumanDeclarationSourceRef(
            request_id=self.request_id,
            actor=self.actor,
            declaration_hash=self.declaration_hash,
            created_at=self.created_at,
            review_id=self.review_id,
        )

    def to_authority(self) -> HumanDeclarationAuthority:
        return HumanDeclarationAuthority(
            source_ref=self.to_source_ref(),
            origin=self.origin,
            declaration=self.declaration,
        )


@dataclass(frozen=True)
class _Failure(Exception):
    code: SourceValidationCode
    field_name: Optional[str] = None


@dataclass(frozen=True)
class _VerifiedEvidence:
    content_hash: Optional[str]
    revision_binding: Optional[str]
    size_bytes: int


@dataclass(frozen=True)
class _MaterializedText:
    text: str = field(repr=False)
    encoded: bytes = field(repr=False)


@dataclass(frozen=True)
class _PythonSymbol:
    qualified_name: str
    module_qualified_name: str
    body: str = field(repr=False)
    signature: str = field(repr=False)


def human_declaration_hash(declaration: str) -> str:
    """Hash the exact UTF-8 bytes of an explicit human declaration."""

    if not isinstance(declaration, str):
        raise SourceValidationError(SourceValidationCode.INVALID_INPUT)
    return hashlib.sha256(declaration.encode("utf-8")).hexdigest()


def repository_range_hash(
    repository: Path,
    revision: str,
    path: str,
    line_start: int,
    line_end: int,
    *,
    revision_resolver: Optional[RevisionResolver] = None,
    max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES,
) -> str:
    """Hash exact committed bytes selected by the inclusive line range.

    Git blob line endings are preserved.  A UTF-8 BOM is treated as encoding
    metadata and is not part of line-one content.
    """

    resolver = revision_resolver or RevisionResolver()
    try:
        root = _repository_root(repository, resolver)
        exact_revision = _exact_commit(root, revision, resolver)
        source = _read_repository_text(root, exact_revision, path, max_source_bytes)
        selected = _select_repository_range(source.text, line_start, line_end)
        return hashlib.sha256(selected.encoded).hexdigest()
    except _Failure as error:
        raise SourceValidationError(error.code) from None


def repository_symbol_hash(
    repository: Path,
    revision: str,
    path: str,
    qualified_name: str,
    hash_kind: SymbolHashKind,
    *,
    revision_resolver: Optional[RevisionResolver] = None,
    max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES,
) -> str:
    """Hash a deterministic Python symbol signature or body at one commit.

    Symbol text uses LF-normalized source, matching the existing repository
    intelligence body convention.  Signatures include decorators plus the
    declaration header through its terminating colon; bodies include the full
    declaration node but not decorators.
    """

    resolver = revision_resolver or RevisionResolver()
    try:
        root = _repository_root(repository, resolver)
        exact_revision = _exact_commit(root, revision, resolver)
        source = _read_repository_text(root, exact_revision, path, max_source_bytes)
        selected = _select_repository_symbol(
            source.text,
            path,
            qualified_name,
            hash_kind,
        )
        return hashlib.sha256(selected.encoded).hexdigest()
    except _Failure as error:
        raise SourceValidationError(error.code) from None


def git_commit_metadata_hash(
    repository: Path,
    commit_sha: str,
    *,
    revision_resolver: Optional[RevisionResolver] = None,
) -> str:
    """Hash the restricted ``git_commit_metadata_v1`` projection.

    The projection includes object IDs, parent IDs, and author/committer epoch
    times, but deliberately excludes names, email addresses, signatures, and
    commit-message text.
    """

    resolver = revision_resolver or RevisionResolver()
    try:
        root = _repository_root(repository, resolver)
        exact_revision = _exact_commit(root, commit_sha, resolver)
        return _restricted_commit_metadata_hash(root, exact_revision)
    except _Failure as error:
        raise SourceValidationError(error.code) from None


class SourceValidator:
    """Validate typed source refs against exact repository and Session authority."""

    def __init__(
        self,
        repository: Path,
        *,
        sessions_root: Optional[Path] = None,
        human_declarations: Iterable[
            TrustedHumanDeclaration | HumanDeclarationAuthority
        ] = (),
        allowed_source_refs: Optional[Iterable[SourceRef]] = None,
        revision_resolver: Optional[RevisionResolver] = None,
        max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES,
    ) -> None:
        try:
            self.repository = Path(repository)
            self._sessions_root_is_default = sessions_root is None
            self.sessions_root = (
                self.repository / ".review-agent" / "runs"
                if sessions_root is None
                else Path(sessions_root)
            )
        except (TypeError, ValueError, OSError):
            raise SourceValidationError(
                SourceValidationCode.INVALID_CONFIGURATION
            ) from None
        if (
            type(max_source_bytes) is not int
            or max_source_bytes < 1
            or max_source_bytes > 64 * 1024 * 1024
        ):
            raise SourceValidationError(SourceValidationCode.INVALID_CONFIGURATION)
        self.max_source_bytes = max_source_bytes
        self.revision_resolver = revision_resolver or RevisionResolver()
        self._repository_root_cache: Optional[Path] = None
        self._repository_common_dir_cache: Optional[str] = None
        self._repository_key_cache: Optional[str] = None

        declarations: Dict[str, HumanDeclarationAuthority] = {}
        try:
            declaration_values = tuple(human_declarations)
        except TypeError:
            raise SourceValidationError(
                SourceValidationCode.INVALID_CONFIGURATION
            ) from None
        for value in declaration_values:
            if type(value) is TrustedHumanDeclaration:
                try:
                    declaration = value.to_authority()
                except ValueError:
                    raise SourceValidationError(
                        SourceValidationCode.INVALID_CONFIGURATION
                    ) from None
            elif type(value) is HumanDeclarationAuthority:
                try:
                    declaration = HumanDeclarationAuthority.from_dict(
                        value.to_dict()
                    )
                except (TypeError, ValueError):
                    raise SourceValidationError(
                        SourceValidationCode.INVALID_CONFIGURATION
                    ) from None
                if declaration != value:
                    raise SourceValidationError(
                        SourceValidationCode.INVALID_CONFIGURATION
                    )
            else:
                raise SourceValidationError(
                    SourceValidationCode.INVALID_CONFIGURATION
                )
            request_id = declaration.source_ref.request_id
            existing = declarations.get(request_id)
            if existing is not None and existing != declaration:
                raise SourceValidationError(
                    SourceValidationCode.INVALID_CONFIGURATION
                )
            declarations[request_id] = declaration
        self._human_declarations = declarations

        self._allowed_source_ref_keys: Optional[frozenset] = None
        if allowed_source_refs is not None:
            try:
                values = tuple(allowed_source_refs)
            except TypeError:
                raise SourceValidationError(
                    SourceValidationCode.INVALID_CONFIGURATION
                ) from None
            keys = []
            for source_ref in values:
                if type(source_ref) not in _SOURCE_REF_TYPES:
                    raise SourceValidationError(
                        SourceValidationCode.INVALID_CONFIGURATION
                    )
                try:
                    hydrated = SourceRef.from_dict(source_ref.to_dict())
                except (TypeError, ValueError):
                    raise SourceValidationError(
                        SourceValidationCode.INVALID_CONFIGURATION
                    ) from None
                if type(hydrated) is not type(source_ref) or hydrated != source_ref:
                    raise SourceValidationError(
                        SourceValidationCode.INVALID_CONFIGURATION
                    )
                keys.append(_source_ref_key(source_ref))
            self._allowed_source_ref_keys = frozenset(keys)

    def validate_candidate(
        self,
        candidate: MemoryCandidate,
        *,
        runtime_provenance: Optional[TrustedCandidateProvenance] = None,
    ) -> SourceValidationReport:
        """Validate a canonical candidate and scan all persisted text fields."""

        if type(candidate) is not MemoryCandidate:
            return _invalid_input_report()
        if type(runtime_provenance) is not TrustedCandidateProvenance:
            return _single_issue_report(
                SourceValidationCode.RUNTIME_PROVENANCE_REQUIRED,
                declared=candidate.sensitivity,
                subject_id=candidate.candidate_id,
            )
        try:
            _require_canonical_provenance(runtime_provenance)
        except SourceValidationError:
            return _single_issue_report(
                SourceValidationCode.CANDIDATE_AUTHORITY_MISMATCH,
                declared=candidate.sensitivity,
                subject_id=candidate.candidate_id,
            )

        initial_issues = self._candidate_authority_issues(
            candidate,
            runtime_provenance,
        )
        for path in candidate.scope.paths:
            try:
                _canonical_scope_path(path)
            except _Failure:
                initial_issues.append(
                    _issue(SourceValidationCode.UNSAFE_PATH, field_name="scope.paths")
                )

        scan_targets = [
            (field_name, value, None)
            for field_name, value in _persisted_string_fields(
                candidate.to_dict(),
                "candidate",
            )
        ]
        return self._validate(
            candidate.source_refs,
            sensitivity=candidate.sensitivity,
            scan_targets=scan_targets,
            require_allowlisted=(
                runtime_provenance.origin is ProducerType.MODEL
            ),
            allowed_source_ref_keys=frozenset(
                _source_ref_key(item)
                for item in runtime_provenance.allowed_source_refs
            ),
            subject_id=candidate.candidate_id,
            candidate_validation_context_hash=(
                _candidate_validation_context_hash(
                    candidate,
                    runtime_provenance,
                )
            ),
            authority_resolution_hash=(
                runtime_provenance.authority_resolution_hash
            ),
            initial_issues=initial_issues,
        )

    def build_candidate_authority_receipt(
        self,
        candidate: MemoryCandidate,
        runtime_provenance: TrustedCandidateProvenance,
        report: SourceValidationReport,
        *,
        current_target_head_sha: str,
        created_at: str,
    ) -> CandidateAuthorityReceipt:
        """Issue a receipt only after repeating exact live source validation."""

        _require_canonical_candidate(candidate)
        _require_canonical_provenance(runtime_provenance)
        current_target = _canonical_git_object_id(
            current_target_head_sha,
            "current_target_head_sha",
        )
        if not hmac.compare_digest(
            current_target,
            runtime_provenance.target_head_sha,
        ):
            raise SourceValidationError(
                SourceValidationCode.CANDIDATE_AUTHORITY_MISMATCH
            )

        expected_context_hash = _candidate_validation_context_hash(
            candidate,
            runtime_provenance,
        )
        try:
            report_matches = (
                type(report) is SourceValidationReport
                and report.valid
                and report.subject_id == candidate.candidate_id
                and report.candidate_validation_context_hash is not None
                and report.authority_resolution_hash is not None
                and hmac.compare_digest(
                    report.candidate_validation_context_hash,
                    expected_context_hash,
                )
                and hmac.compare_digest(
                    report.authority_resolution_hash,
                    runtime_provenance.authority_resolution_hash,
                )
            )
        except (AttributeError, TypeError, ValueError):
            report_matches = False
        if not report_matches:
            raise SourceValidationError(
                SourceValidationCode.VALIDATION_REPORT_MISMATCH,
                report=(
                    report
                    if type(report) is SourceValidationReport
                    else None
                ),
            )

        fresh_report = self.validate_candidate(
            candidate,
            runtime_provenance=runtime_provenance,
        )
        fresh_report.require_valid()
        try:
            report_hash_matches = hmac.compare_digest(
                report.report_hash,
                fresh_report.report_hash,
            )
        except (AttributeError, TypeError, ValueError):
            report_hash_matches = False
        if not report_hash_matches:
            raise SourceValidationError(
                SourceValidationCode.VALIDATION_REPORT_MISMATCH,
                report=fresh_report,
            )

        trusted_declarations = self._trusted_candidate_declarations(
            candidate,
            runtime_provenance.review_id,
        )
        receipt_declarations: Tuple[HumanDeclarationAuthority, ...] = ()
        if runtime_provenance.origin is ProducerType.HUMAN:
            if not trusted_declarations:
                raise SourceValidationError(
                    SourceValidationCode.HUMAN_DECLARATION_UNAUTHORIZED
                )
            receipt_declarations = trusted_declarations

        return CandidateAuthorityReceipt(
            candidate_id=candidate.candidate_id,
            authority_repository_key=(
                runtime_provenance.authority_repository_key
            ),
            locator_repository_key=runtime_provenance.locator_repository_key,
            origin=runtime_provenance.origin,
            review_id=runtime_provenance.review_id,
            proposal_head_sha=current_target,
            authorized_source_refs=candidate.source_refs,
            human_declarations=receipt_declarations,
            initial_validation_report_hash=fresh_report.report_hash,
            authority_resolution_hash=(
                runtime_provenance.authority_resolution_hash
            ),
            binding_id=runtime_provenance.binding_id,
            created_at=created_at,
        )

    def restore_candidate_authority(
        self,
        receipt: CandidateAuthorityReceipt,
        candidate: MemoryCandidate,
        *,
        current_provenance: TrustedCandidateProvenance,
        current_target_head_sha: str,
    ) -> CandidateAuthorityRestoration:
        """Restore authority only from an independently trusted live context."""

        _require_canonical_receipt(receipt)
        _require_canonical_candidate(candidate)
        _require_canonical_provenance(current_provenance)
        current_target = _canonical_git_object_id(
            current_target_head_sha,
            "current_target_head_sha",
        )
        if not hmac.compare_digest(
            current_target,
            current_provenance.target_head_sha,
        ):
            raise SourceValidationError(
                SourceValidationCode.CANDIDATE_AUTHORITY_MISMATCH
            )

        exact_authority = (
            receipt.candidate_id == candidate.candidate_id
            and receipt.origin is current_provenance.origin
            and receipt.review_id == current_provenance.review_id
            and hmac.compare_digest(receipt.proposal_head_sha, current_target)
            and hmac.compare_digest(
                receipt.locator_repository_key,
                current_provenance.locator_repository_key,
            )
            and hmac.compare_digest(
                receipt.authority_repository_key,
                current_provenance.authority_repository_key,
            )
            and hmac.compare_digest(
                receipt.authority_resolution_hash,
                current_provenance.authority_resolution_hash,
            )
            and receipt.binding_id == current_provenance.binding_id
            and receipt.authorized_source_refs == candidate.source_refs
            and candidate.origin_review_id == current_provenance.review_id
            and hmac.compare_digest(
                candidate.repository_key,
                current_provenance.authority_repository_key,
            )
        )
        if not exact_authority:
            raise SourceValidationError(
                SourceValidationCode.CANDIDATE_AUTHORITY_MISMATCH
            )

        if current_provenance.origin is ProducerType.MODEL:
            allowed_keys = {
                _source_ref_key(item)
                for item in current_provenance.allowed_source_refs
            }
            if any(
                _source_ref_key(item) not in allowed_keys
                for item in candidate.source_refs
            ):
                raise SourceValidationError(
                    SourceValidationCode.PRODUCER_POLICY_UNAUTHORIZED
                )

        trusted_declarations = self._trusted_candidate_declarations(
            candidate,
            current_provenance.review_id,
        )
        if current_provenance.origin is ProducerType.HUMAN:
            if (
                not trusted_declarations
                or receipt.human_declarations != trusted_declarations
            ):
                raise SourceValidationError(
                    SourceValidationCode.HUMAN_DECLARATION_UNAUTHORIZED
                )
        elif receipt.human_declarations:
            raise SourceValidationError(
                SourceValidationCode.AUTHORITY_RECEIPT_INVALID
            )

        restored_provenance = current_provenance
        fresh_report = self.validate_candidate(
            candidate,
            runtime_provenance=restored_provenance,
        )
        fresh_report.require_valid()
        if not hmac.compare_digest(
            receipt.initial_validation_report_hash,
            fresh_report.report_hash,
        ):
            raise SourceValidationError(
                SourceValidationCode.VALIDATION_REPORT_MISMATCH,
                report=fresh_report,
            )
        return CandidateAuthorityRestoration(
            provenance=restored_provenance,
            human_declarations=trusted_declarations,
        )

    def validate_sources(
        self,
        source_refs: Sequence[SourceRef],
        *,
        sensitivity: Sensitivity,
        statement: Optional[str] = None,
        require_allowlisted: bool = False,
    ) -> SourceValidationReport:
        """Validate a non-empty typed SourceRef sequence and optional statement."""

        if not isinstance(source_refs, (list, tuple)):
            return _invalid_input_report()
        if type(sensitivity) is not Sensitivity:
            return _invalid_input_report()
        if type(require_allowlisted) is not bool:
            return _invalid_input_report(declared=sensitivity)
        if statement is not None and not isinstance(statement, str):
            return _invalid_input_report(declared=sensitivity)
        if not source_refs or len(source_refs) > MAX_VALIDATION_SOURCE_REFS:
            return _invalid_input_report(declared=sensitivity)
        scan_targets = []
        if statement is not None:
            scan_targets.append(("candidate.statement", statement, None))
        return self._validate(
            tuple(source_refs),
            sensitivity=sensitivity,
            scan_targets=scan_targets,
            require_allowlisted=require_allowlisted,
            allowed_source_ref_keys=self._allowed_source_ref_keys,
            subject_id=None,
            candidate_validation_context_hash=None,
            authority_resolution_hash=None,
            initial_issues=[],
        )

    def _validate(
        self,
        source_refs: Sequence[Any],
        *,
        sensitivity: Sensitivity,
        scan_targets: Sequence[Tuple[str, str, Optional[str]]],
        require_allowlisted: bool,
        allowed_source_ref_keys: Optional[frozenset],
        subject_id: Optional[str],
        candidate_validation_context_hash: Optional[str],
        authority_resolution_hash: Optional[str],
        initial_issues: Sequence[ValidationIssue],
    ) -> SourceValidationReport:
        issues: List[ValidationIssue] = list(initial_issues)
        results: List[SourceValidationResult] = []
        blocked = sensitivity is Sensitivity.BLOCKED
        validation_skipped = bool(initial_issues)

        all_scan_targets = list(scan_targets)
        for index, source_ref in enumerate(source_refs):
            if type(source_ref) in _SOURCE_REF_TYPES:
                all_scan_targets.extend(
                    (field_name, value, None)
                    for field_name, value in _persisted_string_fields(
                        source_ref.to_dict(),
                        "source_refs[%d]" % index,
                    )
                )

        if blocked:
            issues.append(_issue(SourceValidationCode.SENSITIVITY_BLOCKED))
        else:
            for field_name, text, schema in all_scan_targets:
                scan = scan_sensitive_text(
                    text,
                    schema=schema,
                    field_name=field_name,
                )
                if not scan.safe:
                    blocked = True
                    issues.append(
                        _issue(
                            SourceValidationCode.SENSITIVE_CONTENT,
                            field_name=field_name,
                        )
                    )
                    break

        for index, source_ref in enumerate(source_refs):
            if blocked or validation_skipped:
                result_issue = _issue(
                    SourceValidationCode.VALIDATION_SKIPPED,
                    source_index=index,
                    source_type=_source_type_or_none(source_ref),
                )
                results.append(
                    _failed_result(index, source_ref, result_issue)
                )
                continue
            if type(source_ref) not in _SOURCE_REF_TYPES:
                result_issue = _issue(
                    SourceValidationCode.UNTYPED_SOURCE,
                    source_index=index,
                )
                issues.append(result_issue)
                results.append(
                    _failed_result(index, source_ref, result_issue)
                )
                continue
            source_type = source_ref.source_type
            if require_allowlisted and not self._is_allowlisted(
                source_ref,
                allowed_source_ref_keys,
            ):
                result_issue = _issue(
                    SourceValidationCode.SOURCE_NOT_ALLOWLISTED,
                    source_index=index,
                    source_type=source_type,
                )
                issues.append(result_issue)
                results.append(
                    _failed_result(index, source_ref, result_issue)
                )
                continue
            try:
                verified = self._validate_one(source_ref)
            except _Failure as error:
                result_issue = _issue(
                    error.code,
                    source_index=index,
                    source_type=source_type,
                    field_name=error.field_name,
                )
                issues.append(result_issue)
                results.append(
                    _failed_result(index, source_ref, result_issue)
                )
                if error.code is SourceValidationCode.SENSITIVE_CONTENT:
                    blocked = True
                continue
            except Exception:
                result_issue = _issue(
                    SourceValidationCode.INTERNAL_ERROR,
                    source_index=index,
                    source_type=source_type,
                )
                issues.append(result_issue)
                results.append(
                    _failed_result(index, source_ref, result_issue)
                )
                continue
            results.append(
                SourceValidationResult(
                    source_index=index,
                    source_type=source_type,
                    source_ref_hash=_source_ref_key(source_ref),
                    valid=True,
                    verified_content_hash=verified.content_hash,
                    revision_binding=verified.revision_binding,
                    content_size_bytes=verified.size_bytes,
                    issues=(),
                )
            )

        effective = Sensitivity.BLOCKED if blocked else sensitivity
        decision = _sensitivity_decision(sensitivity, effective)
        return SourceValidationReport(
            sensitivity=decision,
            source_results=tuple(results),
            issues=tuple(issues),
            subject_id=subject_id,
            candidate_validation_context_hash=(
                candidate_validation_context_hash
            ),
            authority_resolution_hash=authority_resolution_hash,
        )

    def _is_allowlisted(
        self,
        source_ref: SourceRef,
        allowed_source_ref_keys: Optional[frozenset],
    ) -> bool:
        return (
            allowed_source_ref_keys is not None
            and _source_ref_key(source_ref) in allowed_source_ref_keys
        )

    def _trusted_candidate_declarations(
        self,
        candidate: MemoryCandidate,
        review_id: str,
    ) -> Tuple[HumanDeclarationAuthority, ...]:
        declarations: List[HumanDeclarationAuthority] = []
        for source_ref in candidate.source_refs:
            if type(source_ref) is not HumanDeclarationSourceRef:
                continue
            declaration = self._human_declarations.get(source_ref.request_id)
            if declaration is None:
                raise SourceValidationError(
                    SourceValidationCode.HUMAN_DECLARATION_UNAUTHORIZED
                )
            if declaration.source_ref != source_ref:
                raise SourceValidationError(
                    SourceValidationCode.HUMAN_DECLARATION_MISMATCH
                )
            declaration_review_id = declaration.source_ref.review_id
            if (
                declaration_review_id is not None
                and declaration_review_id != review_id
            ):
                raise SourceValidationError(
                    SourceValidationCode.REVIEW_ID_MISMATCH
                )
            declarations.append(declaration)
        return tuple(sorted(declarations, key=lambda item: item.to_json()))

    def _candidate_authority_issues(
        self,
        candidate: MemoryCandidate,
        runtime_provenance: TrustedCandidateProvenance,
    ) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        try:
            self._root()
        except _Failure as error:
            return [_issue(error.code)]

        if (
            self._repository_key_cache is None
            or not hmac.compare_digest(
                runtime_provenance.locator_repository_key,
                self._repository_key_cache,
            )
        ):
            issues.append(
                _issue(
                    SourceValidationCode.REPOSITORY_MISMATCH,
                    field_name="locator_repository_key",
                )
            )
        if not hmac.compare_digest(
            candidate.repository_key,
            runtime_provenance.authority_repository_key,
        ):
            issues.append(
                _issue(
                    SourceValidationCode.REPOSITORY_MISMATCH,
                    field_name="repository_key",
                )
            )
        if candidate.origin_review_id != runtime_provenance.review_id:
            issues.append(
                _issue(
                    SourceValidationCode.REVIEW_ID_MISMATCH,
                    field_name="origin_review_id",
                )
            )

        try:
            valid_from = self._exact_revision(candidate.valid_from_sha)
            target_head = self._exact_revision(runtime_provenance.target_head_sha)
            in_lineage = self.revision_resolver.is_ancestor(
                self._root(),
                valid_from,
                target_head,
            )
        except _Failure as error:
            issues.append(_issue(error.code, field_name="valid_from_sha"))
        except (OSError, RuntimeError, TypeError, ValueError):
            issues.append(
                _issue(
                    SourceValidationCode.REVISION_NOT_FOUND,
                    field_name="valid_from_sha",
                )
            )
        else:
            if not in_lineage:
                issues.append(
                    _issue(
                        SourceValidationCode.REVISION_LINEAGE_UNAUTHORIZED,
                        field_name="valid_from_sha",
                    )
                )
        return issues

    def _validate_one(self, source_ref: SourceRef) -> _VerifiedEvidence:
        if type(source_ref) is RepositoryRangeSourceRef:
            return self._validate_repository_range(source_ref)
        if type(source_ref) is RepositorySymbolSourceRef:
            return self._validate_repository_symbol(source_ref)
        if type(source_ref) is GitCommitSourceRef:
            return self._validate_git_commit(source_ref)
        if type(source_ref) is ObservationSourceRef:
            return self._validate_observation(source_ref)
        if type(source_ref) is SessionArtifactSourceRef:
            return self._validate_session_artifact(source_ref)
        if type(source_ref) is HumanDeclarationSourceRef:
            return self._validate_human_declaration(source_ref)
        raise _Failure(SourceValidationCode.UNTYPED_SOURCE)

    def _root(self) -> Path:
        if self._repository_root_cache is None:
            root = _repository_root(self.repository, self.revision_resolver)
            try:
                identity = self.revision_resolver.repository_identity(root)
                common_dir = normalize_repository_identity_path(
                    identity.git_common_dir
                )
                repository_key = canonical_repository_key(identity)
            except (OSError, RuntimeError, ValueError):
                raise _Failure(SourceValidationCode.REPOSITORY_UNAVAILABLE)
            self._repository_root_cache = root
            self._repository_common_dir_cache = common_dir
            self._repository_key_cache = repository_key
        return self._repository_root_cache

    def _exact_revision(self, revision: str) -> str:
        return _exact_commit(self._root(), revision, self.revision_resolver)

    def _validate_repository_range(
        self,
        source_ref: RepositoryRangeSourceRef,
    ) -> _VerifiedEvidence:
        revision = self._exact_revision(source_ref.revision)
        source = _read_repository_text(
            self._root(),
            revision,
            source_ref.path,
            self.max_source_bytes,
        )
        selected = _select_repository_range(
            source.text,
            source_ref.line_start,
            source_ref.line_end,
        )
        _require_safe_content(selected.text, field_name="repository_range.content")
        digest = hashlib.sha256(selected.encoded).hexdigest()
        if not hmac.compare_digest(digest, source_ref.content_hash):
            raise _Failure(SourceValidationCode.HASH_MISMATCH, "content_hash")
        return _VerifiedEvidence(digest, revision, len(selected.encoded))

    def _validate_repository_symbol(
        self,
        source_ref: RepositorySymbolSourceRef,
    ) -> _VerifiedEvidence:
        revision = self._exact_revision(source_ref.revision)
        source = _read_repository_text(
            self._root(),
            revision,
            source_ref.path,
            self.max_source_bytes,
        )
        selected = _select_repository_symbol(
            source.text,
            source_ref.path,
            source_ref.qualified_name,
            source_ref.hash_kind,
        )
        _require_safe_content(selected.text, field_name="repository_symbol.content")
        digest = hashlib.sha256(selected.encoded).hexdigest()
        if not hmac.compare_digest(digest, source_ref.content_hash):
            raise _Failure(SourceValidationCode.HASH_MISMATCH, "content_hash")
        return _VerifiedEvidence(digest, revision, len(selected.encoded))

    def _validate_git_commit(
        self,
        source_ref: GitCommitSourceRef,
    ) -> _VerifiedEvidence:
        revision = self._exact_revision(source_ref.commit_sha)
        metadata_hash = _restricted_commit_metadata_hash(self._root(), revision)
        if (
            source_ref.metadata_hash is not None
            and not hmac.compare_digest(metadata_hash, source_ref.metadata_hash)
        ):
            raise _Failure(SourceValidationCode.HASH_MISMATCH, "metadata_hash")
        return _VerifiedEvidence(metadata_hash, revision, 0)

    def _validate_session_artifact(
        self,
        source_ref: SessionArtifactSourceRef,
    ) -> _VerifiedEvidence:
        store, manifest, run_dir = self._load_session(source_ref.review_id)
        descriptor = manifest.artifacts.get(source_ref.artifact_name)
        if descriptor is None:
            raise _Failure(
                SourceValidationCode.DESCRIPTOR_NOT_FOUND,
                "artifact_name",
            )
        _require_canonical_descriptor_schema(descriptor)
        _require_source_eligible_session_artifact(manifest, descriptor)
        if descriptor.schema != source_ref.artifact_schema:
            raise _Failure(
                SourceValidationCode.DESCRIPTOR_SCHEMA_MISMATCH,
                "artifact_schema",
            )
        if (
            not isinstance(descriptor.revision_binding, str)
            or descriptor.revision_binding.casefold()
            != source_ref.revision_binding
        ):
            raise _Failure(
                SourceValidationCode.REVISION_BINDING_MISMATCH,
                "revision_binding",
            )
        if not hmac.compare_digest(descriptor.sha256, source_ref.artifact_hash):
            raise _Failure(SourceValidationCode.HASH_MISMATCH, "artifact_hash")
        content = _read_session_artifact(
            run_dir,
            descriptor,
            self.max_source_bytes,
        )
        if not store.validate_artifact(descriptor):
            raise _Failure(SourceValidationCode.SESSION_ARTIFACT_INVALID)
        self._require_descriptor_still_registered(store, descriptor)
        text = _decode_bounded_utf8(content)
        _require_safe_content(
            text,
            schema=descriptor.schema,
            field_name="session_artifact.content",
        )
        return _VerifiedEvidence(
            descriptor.sha256,
            source_ref.revision_binding,
            len(content),
        )

    def _validate_observation(
        self,
        source_ref: ObservationSourceRef,
    ) -> _VerifiedEvidence:
        store, manifest, run_dir = self._load_session(source_ref.review_id)
        allowed_revisions = _session_observation_bindings(manifest)
        if source_ref.revision_binding not in allowed_revisions:
            raise _Failure(
                SourceValidationCode.REVISION_BINDING_MISMATCH,
                "revision_binding",
            )

        observation: Optional[Observation] = None
        observation_root: Optional[Path] = None
        saw_untrusted_authority = False
        descriptors = sorted(
            (
                item
                for item in manifest.artifacts.values()
                if _registered_schema_or_none(item.name) == _OBSERVATION_SCHEMA
            ),
            key=lambda item: item.name,
        )
        for descriptor in descriptors:
            try:
                _require_canonical_descriptor_schema(descriptor)
                _require_source_eligible_session_artifact(
                    manifest,
                    descriptor,
                )
            except _Failure:
                saw_untrusted_authority = True
                continue
            try:
                _read_session_artifact(
                    run_dir,
                    descriptor,
                    self.max_source_bytes,
                )
                if not store.validate_artifact(descriptor):
                    raise _Failure(
                        SourceValidationCode.SESSION_ARTIFACT_INVALID
                    )
                root = run_dir.joinpath(
                    *PurePosixPath(descriptor.path).parent.parts
                )
                hydrated = ObservationStore.load(
                    root,
                    set(allowed_revisions),
                    max_log_bytes=self.max_source_bytes,
                    max_raw_artifact_bytes=self.max_source_bytes,
                    max_total_raw_bytes=self.max_source_bytes,
                )
                self._require_descriptor_still_registered(store, descriptor)
            except (OSError, UnicodeError, ValueError, _Failure):
                saw_untrusted_authority = True
                continue
            matches = [
                item
                for item in hydrated.list_observations()
                if item.observation_id == source_ref.observation_id
            ]
            if matches:
                observation = matches[0]
                observation_root = root
                break

        if observation is None or observation_root is None:
            code = (
                SourceValidationCode.OBSERVATION_UNTRUSTED
                if saw_untrusted_authority
                else SourceValidationCode.OBSERVATION_NOT_FOUND
            )
            raise _Failure(code, "observation_id")
        if observation.revision.casefold() != source_ref.revision_binding:
            raise _Failure(
                SourceValidationCode.REVISION_BINDING_MISMATCH,
                "revision_binding",
            )
        if not hmac.compare_digest(observation.content_hash, source_ref.content_hash):
            raise _Failure(SourceValidationCode.HASH_MISMATCH, "content_hash")
        if observation.path is not None:
            _canonical_source_path(observation.path)

        raw_relative = PurePosixPath(observation.raw_artifact_ref)
        raw_path = observation_root.joinpath(*raw_relative.parts)
        raw_content = _read_regular_file_under_root(
            run_dir,
            raw_path,
            self.max_source_bytes,
        )
        _require_observation_hash(raw_content, observation)
        raw_text = _decode_bounded_utf8(raw_content)
        _require_safe_content(
            raw_text,
            field_name="observation.content",
        )
        _require_safe_content(
            observation.context_view,
            field_name="observation.context_view",
        )
        return _VerifiedEvidence(
            observation.content_hash,
            observation.revision,
            len(raw_content),
        )

    def _validate_human_declaration(
        self,
        source_ref: HumanDeclarationSourceRef,
    ) -> _VerifiedEvidence:
        declaration = self._human_declarations.get(source_ref.request_id)
        if declaration is None:
            raise _Failure(
                SourceValidationCode.HUMAN_DECLARATION_UNAUTHORIZED,
                "request_id",
            )
        if declaration.source_ref != source_ref:
            raise _Failure(SourceValidationCode.HUMAN_DECLARATION_MISMATCH)
        _require_safe_content(
            declaration.declaration,
            field_name="human_declaration.content",
        )
        _require_safe_content(
            declaration.source_ref.actor,
            field_name="human_declaration.actor",
        )
        return _VerifiedEvidence(
            declaration.source_ref.declaration_hash,
            source_ref.review_id,
            len(declaration.declaration.encode("utf-8")),
        )

    def _load_session(
        self,
        review_id: str,
    ) -> Tuple[SessionStore, SessionManifest, Path]:
        if (
            not _SAFE_REVIEW_ID_PATTERN.fullmatch(review_id)
            or not _is_safe_portable_path_component(review_id)
        ):
            raise _Failure(SourceValidationCode.UNSAFE_PATH, "review_id")
        try:
            sessions_root_candidate = (
                self._root() / ".review-agent" / "runs"
                if self._sessions_root_is_default
                else self.sessions_root
            )
            sessions_root = _resolve_sessions_root(
                sessions_root_candidate,
                repository_root=(
                    self._root() if self._sessions_root_is_default else None
                ),
            )
            run_dir_candidate = sessions_root / review_id
            run_metadata = run_dir_candidate.lstat()
            if (
                not stat.S_ISDIR(run_metadata.st_mode)
                or run_dir_candidate.is_symlink()
            ):
                raise _Failure(SourceValidationCode.SESSION_UNTRUSTED)
            run_dir = run_dir_candidate.resolve(strict=True)
            run_dir.relative_to(sessions_root)
        except _Failure:
            raise
        except FileNotFoundError:
            raise _Failure(SourceValidationCode.SESSION_NOT_FOUND)
        except (OSError, ValueError):
            raise _Failure(SourceValidationCode.SESSION_UNTRUSTED)

        store = SessionStore(run_dir)
        try:
            manifest = store.load()
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
            raise _Failure(SourceValidationCode.SESSION_UNTRUSTED)
        if manifest.review_id != review_id:
            raise _Failure(SourceValidationCode.REVIEW_ID_MISMATCH, "review_id")
        try:
            session_common_dir = normalize_repository_identity_path(
                manifest.repository.git_common_dir
            )
            session_repository_key = canonical_repository_key(
                manifest.repository
            )
        except (OSError, ValueError):
            raise _Failure(SourceValidationCode.SESSION_UNTRUSTED)
        self._root()
        if (
            session_common_dir != self._repository_common_dir_cache
            or self._repository_key_cache is None
            or not hmac.compare_digest(
                session_repository_key,
                self._repository_key_cache,
            )
        ):
            raise _Failure(SourceValidationCode.REPOSITORY_MISMATCH)
        try:
            base = self._exact_revision(manifest.revisions.resolved_base_sha)
            head = self._exact_revision(manifest.revisions.resolved_head_sha)
        except _Failure:
            raise _Failure(SourceValidationCode.SESSION_UNTRUSTED)
        if (
            base != manifest.revisions.resolved_base_sha.casefold()
            or head != manifest.revisions.resolved_head_sha.casefold()
        ):
            raise _Failure(SourceValidationCode.SESSION_UNTRUSTED)
        return store, manifest, run_dir

    @staticmethod
    def _require_descriptor_still_registered(
        store: SessionStore,
        descriptor: ArtifactDescriptor,
    ) -> None:
        try:
            current = store.load()
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
            raise _Failure(SourceValidationCode.SESSION_UNTRUSTED)
        if current.artifacts.get(descriptor.name) != descriptor:
            raise _Failure(SourceValidationCode.SESSION_ARTIFACT_INVALID)
        _require_source_eligible_session_artifact(current, descriptor)


def scan_sensitive_text(
    text: str,
    *,
    schema: Optional[str] = None,
    field_name: str = "content",
) -> SensitiveContentScan:
    if not isinstance(text, str) or not isinstance(field_name, str):
        raise SourceValidationError(SourceValidationCode.INVALID_INPUT)
    findings: List[SensitiveContentFinding] = []

    def add(kind: SensitiveContentKind, field: str) -> None:
        finding = SensitiveContentFinding(kind, field)
        if finding not in findings:
            findings.append(finding)

    if _PRIVATE_KEY_PATTERN.search(text):
        add(SensitiveContentKind.PRIVATE_KEY, field_name)
    if _KNOWN_TOKEN_PATTERN.search(text):
        add(SensitiveContentKind.KNOWN_TOKEN, field_name)
    if _AUTHORIZATION_PATTERN.search(text):
        add(SensitiveContentKind.AUTHORIZATION_HEADER, field_name)
    if _AUTHENTICATED_URL_PATTERN.search(text):
        add(SensitiveContentKind.AUTHENTICATED_URL, field_name)

    structured_values, duplicate_json_key = _structured_values(text, schema)
    if duplicate_json_key:
        add(SensitiveContentKind.DUPLICATE_JSON_KEY, field_name)
    assignment_texts = [text]
    assignment_texts.extend(
        item
        for structured in structured_values
        for item in _structured_strings(structured)
    )
    for assignment_text in assignment_texts:
        for match in _CREDENTIAL_ASSIGNMENT_PATTERN.finditer(assignment_text):
            value = _strip_secret_quotes(match.group("value"))
            if _looks_like_secret_value(value):
                add(SensitiveContentKind.CREDENTIAL_FIELD, field_name)
        for match in _COMMAND_CREDENTIAL_PATTERN.finditer(assignment_text):
            value = _strip_secret_quotes(match.group("value"))
            if _looks_like_secret_value(value):
                add(SensitiveContentKind.CREDENTIAL_FIELD, field_name)
    for structured in structured_values:
        _scan_structured_credentials(structured, field_name, add)

    return SensitiveContentScan(
        tuple(sorted(findings, key=lambda item: (item.kind.value, item.field_name)))
    )


_PRIVATE_KEY_PATTERN = re.compile(
    r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----",
    re.IGNORECASE,
)
_KNOWN_TOKEN_PATTERN = re.compile(
    r"(?:"
    r"\bAKIA[0-9A-Z]{16}\b"
    r"|\bgh[pousr]_[A-Za-z0-9]{20,}\b"
    r"|\bgithub_pat_[A-Za-z0-9_]{20,}\b"
    r"|\bsk-[A-Za-z0-9_-]{16,}\b"
    r"|\bxox[baprs]-[A-Za-z0-9-]{12,}\b"
    r"|\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"
    r")"
)
_AUTHORIZATION_PATTERN = re.compile(
    r"\bauthorization\s*:\s*(?:bearer|basic)\s+[A-Za-z0-9+/=._~-]{8,}",
    re.IGNORECASE,
)
_AUTHENTICATED_URL_PATTERN = re.compile(
    r"\b(?:https?|ssh)://[^\s/@:]+:[^\s/@]+@[^\s/]+",
    re.IGNORECASE,
)
_CREDENTIAL_ASSIGNMENT_PATTERN = re.compile(
    r"(?im)(?:^|[,{;\s])"
    r"(?:[A-Za-z0-9]+[_-])*"
    r"(?:api[_-]?key|access[_-]?key|private[_-]?key|secret[_-]?key|"
    r"client[_-]?secret|password|passwd|secret|credential(?:s)?|"
    r"access[_-]?token|refresh[_-]?token|auth[_-]?token|token|cookie)"
    r"(?![A-Za-z0-9_])\s*(?:=|:)\s*"
    r"(?P<value>\"[^\"\r\n]{1,512}\"|'[^'\r\n]{1,512}'|[^\s,;#]{1,512})"
)
_COMMAND_CREDENTIAL_PATTERN = re.compile(
    r"(?im)(?:^|\s)--(?:api[_-]?key|password|secret|credential|token)"
    r"(?:=|\s+)"
    r"(?P<value>\"[^\"\r\n]{1,512}\"|'[^'\r\n]{1,512}'|[^\s,;#]{1,512})"
)
_SAFE_SECRET_VALUES = frozenset(
    {
        "",
        "***",
        "<redacted>",
        "changeme",
        "dummy",
        "example",
        "none",
        "not-set",
        "null",
        "placeholder",
        "redacted",
        "test",
    }
)
_ENV_REFERENCE_PATTERN = re.compile(
    r"^(?:\$[A-Z_][A-Z0-9_]*|\$\{[A-Z_][A-Z0-9_]*\}|%[A-Z_][A-Z0-9_]*%)$"
)
_SENSITIVE_FIELD_NAMES = frozenset(
    {
        "access_key",
        "access_token",
        "api_key",
        "auth_token",
        "authorization",
        "client_secret",
        "cookie",
        "credential",
        "credentials",
        "id_token",
        "passwd",
        "password",
        "private_key",
        "refresh_token",
        "secret",
        "secret_key",
        "session_token",
        "token",
    }
)


def _scan_structured_credentials(
    value: Any,
    field_name: str,
    add: Any,
) -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = _normalized_field_name(str(raw_key))
            child_field = "%s.%s" % (field_name, key or "field")
            if _is_sensitive_field_name(key):
                if _structured_secret_value(child):
                    add(SensitiveContentKind.CREDENTIAL_FIELD, child_field)
            else:
                _scan_structured_credentials(child, child_field, add)
    elif isinstance(value, list):
        for child in value:
            _scan_structured_credentials(child, field_name, add)


class _DuplicateJsonKey(ValueError):
    pass


def _reject_duplicate_json_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    value: Dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise _DuplicateJsonKey
        value[key] = child
    return value


def _strict_json_value(text: str) -> Any:
    return json.loads(text, object_pairs_hook=_reject_duplicate_json_keys)


def _structured_values(
    text: str,
    schema: Optional[str],
) -> Tuple[List[Any], bool]:
    stripped = text.strip()
    if not stripped:
        return [], False
    values: List[Any] = []
    if schema is not None and "jsonl" in schema.casefold():
        parsed_lines = []
        try:
            for line in stripped.splitlines():
                if line.strip():
                    parsed_lines.append(_strict_json_value(line))
        except _DuplicateJsonKey:
            return [], True
        except (json.JSONDecodeError, TypeError, ValueError):
            return [], False
        return parsed_lines, False
    if stripped[0] not in "[{":
        return [], False
    try:
        values.append(_strict_json_value(stripped))
    except _DuplicateJsonKey:
        return [], True
    except (json.JSONDecodeError, TypeError, ValueError):
        return [], False
    return values, False


def _structured_strings(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [
            text
            for child in value.values()
            for text in _structured_strings(child)
        ]
    if isinstance(value, list):
        return [
            text
            for child in value
            for text in _structured_strings(child)
        ]
    return []


def _normalized_field_name(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return normalized


def _is_sensitive_field_name(value: str) -> bool:
    if not value or value.endswith("_env"):
        return False
    if value.endswith(("_hash", "_sha", "_sha256", "_digest")):
        return False
    safe_token_fields = {
        "input_tokens",
        "max_tokens",
        "output_tokens",
        "token_budget",
        "token_count",
        "total_tokens",
    }
    if value in safe_token_fields:
        return False
    return value in _SENSITIVE_FIELD_NAMES or any(
        value.endswith("_" + item) for item in _SENSITIVE_FIELD_NAMES
    )


def _structured_secret_value(value: Any) -> bool:
    if isinstance(value, str):
        return _looks_like_secret_value(value)
    if isinstance(value, list):
        return any(_structured_secret_value(item) for item in value)
    if isinstance(value, Mapping):
        return any(_structured_secret_value(item) for item in value.values())
    return False


def _strip_secret_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _looks_like_secret_value(value: str) -> bool:
    normalized = value.strip()
    if normalized.casefold() in _SAFE_SECRET_VALUES:
        return False
    if _ENV_REFERENCE_PATTERN.fullmatch(normalized):
        return False
    if re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
        r"(?:\([^\r\n]*\)|\[[^\r\n\]]+\])",
        normalized,
    ):
        return False
    return bool(normalized)


def _issue(
    code: SourceValidationCode,
    *,
    source_index: Optional[int] = None,
    source_type: Optional[SourceRefType] = None,
    field_name: Optional[str] = None,
) -> ValidationIssue:
    return ValidationIssue(
        code=code,
        message=_ISSUE_MESSAGES[code],
        source_index=source_index,
        source_type=source_type,
        field_name=field_name,
    )


def _invalid_input_report(
    *,
    declared: Sensitivity = Sensitivity.BLOCKED,
) -> SourceValidationReport:
    effective = Sensitivity.BLOCKED
    issue = _issue(SourceValidationCode.INVALID_INPUT)
    return SourceValidationReport(
        sensitivity=_sensitivity_decision(declared, effective),
        source_results=(),
        issues=(issue,),
    )


def _single_issue_report(
    code: SourceValidationCode,
    *,
    declared: Sensitivity,
    subject_id: Optional[str] = None,
    field_name: Optional[str] = None,
) -> SourceValidationReport:
    effective = (
        Sensitivity.BLOCKED
        if declared is Sensitivity.BLOCKED
        else declared
    )
    issue = _issue(code, field_name=field_name)
    return SourceValidationReport(
        sensitivity=_sensitivity_decision(declared, effective),
        source_results=(),
        issues=(issue,),
        subject_id=subject_id,
    )


def _sensitivity_decision(
    declared: Sensitivity,
    effective: Sensitivity,
) -> SensitivityDecision:
    if effective is Sensitivity.BLOCKED:
        return SensitivityDecision(declared, effective, False, False, False)
    if effective is Sensitivity.LOCAL_ONLY:
        return SensitivityDecision(declared, effective, True, False, True)
    return SensitivityDecision(declared, effective, True, True, True)


def _source_type_or_none(value: Any) -> Optional[SourceRefType]:
    if type(value) in _SOURCE_REF_TYPES:
        return value.source_type
    return None


def _source_ref_key(source_ref: SourceRef) -> str:
    return canonical_sha256(source_ref.to_dict())


def _canonical_git_object_id(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError("%s must be a full Git object ID" % field_name)
    normalized = value.casefold()
    if _GIT_OBJECT_ID_PATTERN.fullmatch(normalized) is None:
        raise ValueError("%s must be a full Git object ID" % field_name)
    return normalized


def _candidate_validation_context_hash(
    candidate: MemoryCandidate,
    runtime_provenance: TrustedCandidateProvenance,
) -> str:
    proposal = replace(candidate, status=CandidateStatus.PROPOSED)
    return canonical_sha256(
        {
            "schema_version": SOURCE_VALIDATION_SCHEMA_VERSION,
            "candidate_id": proposal.candidate_id,
            "candidate_hash": canonical_sha256(proposal.to_dict()),
            "origin": runtime_provenance.origin.value,
            "review_id": runtime_provenance.review_id,
            "proposal_head_sha": runtime_provenance.target_head_sha,
            "locator_repository_key": (
                runtime_provenance.locator_repository_key
            ),
            "authority_repository_key": (
                runtime_provenance.authority_repository_key
            ),
            "authority_resolution_hash": (
                runtime_provenance.authority_resolution_hash
            ),
            "binding_id": runtime_provenance.binding_id,
            "authorized_source_refs_hash": canonical_sha256(
                [item.to_dict() for item in proposal.source_refs]
            ),
        }
    )


def _require_canonical_candidate(candidate: Any) -> MemoryCandidate:
    if type(candidate) is not MemoryCandidate:
        raise SourceValidationError(SourceValidationCode.AUTHORITY_RECEIPT_INVALID)
    try:
        hydrated = MemoryCandidate.from_dict(candidate.to_dict())
    except (TypeError, ValueError):
        raise SourceValidationError(
            SourceValidationCode.AUTHORITY_RECEIPT_INVALID
        ) from None
    if hydrated != candidate:
        raise SourceValidationError(
            SourceValidationCode.AUTHORITY_RECEIPT_INVALID
        )
    return candidate


def _require_canonical_provenance(
    provenance: Any,
) -> TrustedCandidateProvenance:
    if type(provenance) is not TrustedCandidateProvenance:
        raise SourceValidationError(
            SourceValidationCode.RUNTIME_PROVENANCE_REQUIRED
        )
    try:
        hydrated = TrustedCandidateProvenance(
            origin=provenance.origin,
            review_id=provenance.review_id,
            target_head_sha=provenance.target_head_sha,
            locator_repository_key=provenance.locator_repository_key,
            authority_repository_key=provenance.authority_repository_key,
            authority_resolution_hash=provenance.authority_resolution_hash,
            binding_id=provenance.binding_id,
            allowed_source_refs=provenance.allowed_source_refs,
        )
    except (TypeError, ValueError):
        raise SourceValidationError(
            SourceValidationCode.CANDIDATE_AUTHORITY_MISMATCH
        ) from None
    if hydrated != provenance:
        raise SourceValidationError(
            SourceValidationCode.CANDIDATE_AUTHORITY_MISMATCH
        )
    return provenance


def _require_canonical_receipt(receipt: Any) -> CandidateAuthorityReceipt:
    if type(receipt) is not CandidateAuthorityReceipt:
        raise SourceValidationError(SourceValidationCode.AUTHORITY_RECEIPT_INVALID)
    try:
        hydrated = CandidateAuthorityReceipt.from_dict(receipt.to_dict())
    except (TypeError, ValueError):
        raise SourceValidationError(
            SourceValidationCode.AUTHORITY_RECEIPT_INVALID
        ) from None
    if hydrated != receipt:
        raise SourceValidationError(
            SourceValidationCode.AUTHORITY_RECEIPT_INVALID
        )
    return receipt


def _persisted_string_fields(
    value: Any,
    prefix: str,
) -> List[Tuple[str, str]]:
    if isinstance(value, str):
        return [(prefix, value)]
    if isinstance(value, Mapping):
        fields: List[Tuple[str, str]] = []
        for key in sorted(value, key=lambda item: str(item)):
            child_prefix = "%s.%s" % (prefix, str(key))
            fields.extend(_persisted_string_fields(value[key], child_prefix))
        return fields
    if isinstance(value, (list, tuple)):
        fields = []
        for index, child in enumerate(value):
            fields.extend(
                _persisted_string_fields(
                    child,
                    "%s[%d]" % (prefix, index),
                )
            )
        return fields
    return []


def _failed_result(
    index: int,
    source_ref: Any,
    issue: ValidationIssue,
) -> SourceValidationResult:
    typed = type(source_ref) in _SOURCE_REF_TYPES
    return SourceValidationResult(
        source_index=index,
        source_type=source_ref.source_type if typed else None,
        source_ref_hash=_source_ref_key(source_ref) if typed else None,
        valid=False,
        verified_content_hash=None,
        revision_binding=None,
        content_size_bytes=0,
        issues=(issue,),
    )


def _repository_root(repository: Path, resolver: RevisionResolver) -> Path:
    try:
        candidate = Path(repository).resolve(strict=True)
        if not candidate.is_dir():
            raise _Failure(SourceValidationCode.REPOSITORY_UNAVAILABLE)
        identity = resolver.repository_identity(candidate)
        root = Path(identity.canonical_path).resolve(strict=True)
        if not root.is_dir():
            raise _Failure(SourceValidationCode.REPOSITORY_UNAVAILABLE)
        return root
    except _Failure:
        raise
    except (OSError, RuntimeError, TypeError, ValueError):
        raise _Failure(SourceValidationCode.REPOSITORY_UNAVAILABLE)


def _exact_commit(
    repository: Path,
    revision: str,
    resolver: RevisionResolver,
) -> str:
    try:
        resolved = resolver.resolve_commit(repository, revision).casefold()
        exists = resolver.commit_exists(repository, revision)
    except (OSError, RuntimeError, TypeError, ValueError):
        raise _Failure(SourceValidationCode.REVISION_NOT_FOUND)
    if not exists:
        raise _Failure(SourceValidationCode.REVISION_NOT_FOUND)
    if resolved != revision.casefold():
        raise _Failure(SourceValidationCode.REVISION_MISMATCH)
    return resolved


def _canonical_scope_path(path: str) -> str:
    return _canonical_portable_path(path, allow_glob=True)


def _canonical_source_path(path: str) -> str:
    return _canonical_portable_path(path, allow_glob=False)


def _canonical_portable_path(path: str, *, allow_glob: bool) -> str:
    if not isinstance(path, str) or not path or path != path.strip():
        raise _Failure(SourceValidationCode.UNSAFE_PATH, "path")
    if "\\" in path or "\x00" in path:
        raise _Failure(SourceValidationCode.UNSAFE_PATH, "path")
    posix = PurePosixPath(path)
    windows = PureWindowsPath(path)
    parts = path.split("/")
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or _WINDOWS_DRIVE_PATTERN.match(path)
        or posix.as_posix() != path
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise _Failure(SourceValidationCode.UNSAFE_PATH, "path")
    for part in parts:
        if not _is_safe_portable_path_component(part, allow_glob=allow_glob):
            raise _Failure(SourceValidationCode.UNSAFE_PATH, "path")
        lowered = part.casefold()
        if lowered in _SENSITIVE_PATH_COMPONENTS or lowered.startswith(".env."):
            raise _Failure(SourceValidationCode.UNSAFE_PATH, "path")
    if not scan_sensitive_text(path, field_name="path").safe:
        raise _Failure(SourceValidationCode.UNSAFE_PATH, "path")
    return path


def _is_safe_portable_path_component(
    component: str,
    *,
    allow_glob: bool = False,
) -> bool:
    if (
        not isinstance(component, str)
        or not component
        or component in {".", ".."}
        or component != unicodedata.normalize("NFC", component)
        or component[-1] in {".", " "}
        or any(ord(character) < 32 or ord(character) == 127 for character in component)
        or ":" in component
    ):
        return False
    invalid_characters = _WINDOWS_INVALID_PATH_CHARACTERS
    if allow_glob:
        invalid_characters = invalid_characters - frozenset("?*")
    if any(character in invalid_characters for character in component):
        return False

    # Windows resolves device basenames even when an extension is present.
    basename = component.split(".", 1)[0].casefold()
    if basename in _WINDOWS_RESERVED_COMPONENTS:
        return False
    if _WINDOWS_SHORT_NAME_PATTERN.fullmatch(component):
        return False
    return True


def _read_repository_text(
    repository: Path,
    revision: str,
    path: str,
    max_source_bytes: int,
) -> _MaterializedText:
    canonical_path = _canonical_source_path(path)
    tree_output = _run_git_bytes(
        repository,
        ["ls-tree", "-z", "--full-tree", revision, "--", canonical_path],
        missing_code=SourceValidationCode.SOURCE_NOT_FOUND,
    )
    entries = [item for item in tree_output.split(b"\x00") if item]
    if len(entries) != 1:
        raise _Failure(SourceValidationCode.SOURCE_NOT_FOUND)
    try:
        metadata, raw_path = entries[0].split(b"\t", 1)
        mode, object_type, _object_id = metadata.decode("ascii").split(" ", 2)
        decoded_path = raw_path.decode("utf-8")
    except (UnicodeError, ValueError):
        raise _Failure(SourceValidationCode.SOURCE_NOT_REGULAR)
    if decoded_path != canonical_path:
        raise _Failure(SourceValidationCode.SOURCE_NOT_FOUND)
    if mode not in _REGULAR_GIT_MODES or object_type != "blob":
        raise _Failure(SourceValidationCode.SOURCE_NOT_REGULAR)

    object_spec = "%s:%s" % (revision, canonical_path)
    size_output = _run_git_bytes(
        repository,
        ["cat-file", "-s", object_spec],
        missing_code=SourceValidationCode.SOURCE_NOT_FOUND,
    )
    try:
        size = int(size_output.strip())
    except ValueError:
        raise _Failure(SourceValidationCode.SOURCE_NOT_REGULAR)
    if size < 0 or size > max_source_bytes:
        raise _Failure(SourceValidationCode.SOURCE_TOO_LARGE)
    content = _run_git_bytes(
        repository,
        ["cat-file", "blob", object_spec],
        missing_code=SourceValidationCode.SOURCE_NOT_FOUND,
    )
    if len(content) != size:
        raise _Failure(SourceValidationCode.SOURCE_NOT_REGULAR)
    text = _decode_bounded_utf8(content)
    return _MaterializedText(text, text.encode("utf-8"))


def _select_repository_range(
    text: str,
    line_start: int,
    line_end: int,
) -> _MaterializedText:
    if (
        type(line_start) is not int
        or type(line_end) is not int
        or line_start < 1
        or line_end < line_start
    ):
        raise _Failure(SourceValidationCode.RANGE_OUT_OF_BOUNDS)
    lines = text.splitlines(keepends=True)
    if line_start > len(lines) or line_end > len(lines):
        raise _Failure(SourceValidationCode.RANGE_OUT_OF_BOUNDS)
    selected = "".join(lines[line_start - 1 : line_end])
    return _MaterializedText(selected, selected.encode("utf-8"))


def _select_repository_symbol(
    text: str,
    path: str,
    qualified_name: str,
    hash_kind: SymbolHashKind,
) -> _MaterializedText:
    if not isinstance(qualified_name, str) or not qualified_name:
        raise _Failure(SourceValidationCode.SYMBOL_NOT_FOUND)
    if type(hash_kind) is not SymbolHashKind:
        raise _Failure(SourceValidationCode.SYMBOL_UNSUPPORTED)
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    symbols = _python_symbols(normalized, path)
    matches = [
        symbol
        for symbol in symbols
        if qualified_name
        in {symbol.qualified_name, symbol.module_qualified_name}
    ]
    if not matches:
        raise _Failure(SourceValidationCode.SYMBOL_NOT_FOUND)
    if len(matches) != 1:
        raise _Failure(SourceValidationCode.SYMBOL_AMBIGUOUS)
    selected = (
        matches[0].signature
        if hash_kind is SymbolHashKind.SIGNATURE
        else matches[0].body
    )
    return _MaterializedText(selected, selected.encode("utf-8"))


def _python_symbols(text: str, path: str) -> List[_PythonSymbol]:
    if not path.casefold().endswith(".py"):
        raise _Failure(SourceValidationCode.SYMBOL_UNSUPPORTED)
    try:
        tree = ast.parse(text, filename=path, type_comments=True)
    except (SyntaxError, TypeError, ValueError):
        raise _Failure(SourceValidationCode.SYMBOL_UNSUPPORTED)
    module_name = path[:-3].replace("/", ".")
    if module_name.endswith(".__init__"):
        module_name = module_name[: -len(".__init__")]
    elif module_name == "__init__":
        module_name = ""
    lines = text.splitlines()
    symbols: List[_PythonSymbol] = []

    class Collector(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: List[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self._add(node)
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._add(node)
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._add(node)
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def _add(self, node: Any) -> None:
            lexical = ".".join(self.stack + [node.name])
            module_qualified = (
                "%s.%s" % (module_name, lexical) if module_name else lexical
            )
            line_start = int(node.lineno)
            line_end = int(getattr(node, "end_lineno", node.lineno))
            body = "\n".join(lines[line_start - 1 : line_end])
            signature = _python_signature_text(text, node)
            symbols.append(
                _PythonSymbol(lexical, module_qualified, body, signature)
            )

    try:
        Collector().visit(tree)
    except (tokenize.TokenError, IndentationError, ValueError):
        raise _Failure(SourceValidationCode.SYMBOL_UNSUPPORTED)
    return symbols


def _python_signature_text(text: str, node: Any) -> str:
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (IndentationError, tokenize.TokenError):
        raise _Failure(SourceValidationCode.SYMBOL_UNSUPPORTED)
    keyword_index: Optional[int] = None
    expected_keyword = "class" if isinstance(node, ast.ClassDef) else "def"
    for index, token in enumerate(tokens):
        if token.start[0] < node.lineno:
            continue
        if token.start[0] > node.lineno + 1:
            break
        if token.type == tokenize.NAME and token.string == expected_keyword:
            keyword_index = index
            break
    if keyword_index is None:
        raise _Failure(SourceValidationCode.SYMBOL_UNSUPPORTED)
    depth = 0
    colon = None
    for token in tokens[keyword_index + 1 :]:
        if token.type == tokenize.OP:
            if token.string in "([{":
                depth += 1
            elif token.string in ")]}" and depth:
                depth -= 1
            elif token.string == ":" and depth == 0:
                colon = token
                break
    if colon is None:
        raise _Failure(SourceValidationCode.SYMBOL_UNSUPPORTED)
    start_column = tokens[keyword_index].start[1]
    if (
        isinstance(node, ast.AsyncFunctionDef)
        and keyword_index > 0
        and tokens[keyword_index - 1].string == "async"
    ):
        start_column = tokens[keyword_index - 1].start[1]
    header = _slice_text(
        text,
        (node.lineno, start_column),
        colon.end,
    )
    decorators = []
    for decorator in node.decorator_list:
        segment = ast.get_source_segment(text, decorator)
        if segment is None:
            raise _Failure(SourceValidationCode.SYMBOL_UNSUPPORTED)
        decorators.append("@" + segment)
    return "\n".join(decorators + [header])


def _slice_text(
    text: str,
    start: Tuple[int, int],
    end: Tuple[int, int],
) -> str:
    lines = text.splitlines(keepends=True)
    start_line, start_column = start
    end_line, end_column = end
    if start_line == end_line:
        return lines[start_line - 1][start_column:end_column]
    chunks = [lines[start_line - 1][start_column:]]
    chunks.extend(lines[start_line:end_line - 1])
    chunks.append(lines[end_line - 1][:end_column])
    return "".join(chunks)


def _restricted_commit_metadata_hash(repository: Path, revision: str) -> str:
    output = _run_git_bytes(
        repository,
        [
            "show",
            "--no-show-signature",
            "-s",
            "--format=%T%x00%P%x00%at%x00%ct",
            revision,
        ],
        missing_code=SourceValidationCode.REVISION_NOT_FOUND,
    )
    try:
        fields = output.rstrip(b"\r\n").decode("ascii").split("\x00")
    except UnicodeError:
        raise _Failure(SourceValidationCode.REVISION_NOT_FOUND)
    if len(fields) != 4:
        raise _Failure(SourceValidationCode.REVISION_NOT_FOUND)
    tree_sha, parents, author_time, committer_time = fields
    if not tree_sha or not author_time.isdigit() or not committer_time.isdigit():
        raise _Failure(SourceValidationCode.REVISION_NOT_FOUND)
    payload = {
        "schema": "git_commit_metadata_v1",
        "commit_sha": revision,
        "tree_sha": tree_sha.casefold(),
        "parent_shas": [item.casefold() for item in parents.split() if item],
        "author_time": int(author_time),
        "committer_time": int(committer_time),
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _run_git_bytes(
    repository: Path,
    arguments: Sequence[str],
    *,
    missing_code: SourceValidationCode,
) -> bytes:
    try:
        result = subprocess.run(
            ["git", "--no-replace-objects"] + list(arguments),
            cwd=repository,
            env=sanitized_git_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        raise _Failure(SourceValidationCode.REPOSITORY_UNAVAILABLE)
    if result.returncode != 0:
        raise _Failure(missing_code)
    return result.stdout


def _decode_bounded_utf8(content: bytes) -> str:
    if b"\x00" in content:
        raise _Failure(SourceValidationCode.SOURCE_ENCODING_INVALID)
    try:
        return content.decode("utf-8-sig")
    except UnicodeError:
        raise _Failure(SourceValidationCode.SOURCE_ENCODING_INVALID)


def _require_safe_content(
    text: str,
    *,
    schema: Optional[str] = None,
    field_name: str,
) -> None:
    scan = scan_sensitive_text(text, schema=schema, field_name=field_name)
    if not scan.safe:
        raise _Failure(SourceValidationCode.SENSITIVE_CONTENT, field_name)


def _session_observation_bindings(manifest: SessionManifest) -> frozenset:
    raw_base = manifest.revisions.resolved_base_sha
    raw_head = manifest.revisions.resolved_head_sha
    base = raw_base.casefold()
    head = raw_head.casefold()
    return frozenset(
        {
            "base@%s" % base,
            "head@%s" % head,
            "%s..%s" % (base, head),
            "base@%s" % raw_base,
            "head@%s" % raw_head,
            "%s..%s" % (raw_base, raw_head),
        }
    )


def _read_session_artifact(
    run_dir: Path,
    descriptor: ArtifactDescriptor,
    max_source_bytes: int,
) -> bytes:
    try:
        relative = _canonical_source_path(descriptor.path)
    except _Failure:
        raise _Failure(SourceValidationCode.SESSION_ARTIFACT_INVALID)
    path = run_dir.joinpath(*PurePosixPath(relative).parts)
    content = _read_regular_file_under_root(run_dir, path, max_source_bytes)
    digest = hashlib.sha256(content).hexdigest()
    if not hmac.compare_digest(digest, descriptor.sha256):
        raise _Failure(SourceValidationCode.SESSION_ARTIFACT_INVALID)
    return content


def _resolve_sessions_root(
    candidate: Path,
    *,
    repository_root: Optional[Path],
) -> Path:
    try:
        path = Path(candidate)
        metadata = path.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink():
            raise _Failure(SourceValidationCode.SESSION_UNTRUSTED)
        if repository_root is not None:
            canonical_repository = Path(repository_root).resolve(strict=True)
            relative = path.relative_to(canonical_repository)
            current = canonical_repository
            for part in relative.parts:
                current = current / part
                component_metadata = current.lstat()
                if current.is_symlink() or not stat.S_ISDIR(
                    component_metadata.st_mode
                ):
                    raise _Failure(SourceValidationCode.SESSION_UNTRUSTED)
        resolved = path.resolve(strict=True)
        if repository_root is not None:
            resolved.relative_to(Path(repository_root).resolve(strict=True))
        return resolved
    except _Failure:
        raise
    except FileNotFoundError:
        raise _Failure(SourceValidationCode.SESSION_NOT_FOUND)
    except (OSError, ValueError):
        raise _Failure(SourceValidationCode.SESSION_UNTRUSTED)


def _require_canonical_descriptor_schema(
    descriptor: ArtifactDescriptor,
) -> None:
    expected = _registered_schema_or_none(descriptor.name)
    if expected is None:
        raise _Failure(
            SourceValidationCode.DESCRIPTOR_SCHEMA_MISMATCH,
            "artifact_schema",
        )
    if descriptor.schema != expected:
        raise _Failure(
            SourceValidationCode.DESCRIPTOR_SCHEMA_MISMATCH,
            "artifact_schema",
        )


def _require_source_eligible_session_artifact(
    manifest: SessionManifest,
    descriptor: ArtifactDescriptor,
) -> None:
    if descriptor.schema not in _SOURCE_ELIGIBLE_SESSION_SCHEMAS:
        raise _Failure(
            SourceValidationCode.SESSION_ARTIFACT_INELIGIBLE,
            "artifact_schema",
        )

    checkpoint = manifest.phases.get(descriptor.phase.value)
    if (
        checkpoint is None
        or checkpoint.status is not PhaseStatus.COMPLETED
        or descriptor.name not in checkpoint.artifacts
    ):
        raise _Failure(SourceValidationCode.SESSION_ARTIFACT_INVALID)

    reviewer_match = _REVIEWER_TASK_ARTIFACT_PATTERN.match(descriptor.name)
    if reviewer_match is not None:
        task = checkpoint.tasks.get("reviewer-%s" % reviewer_match.group(1))
        if (
            task is None
            or task.status is not PhaseStatus.COMPLETED
            or descriptor.name not in task.artifacts
        ):
            raise _Failure(SourceValidationCode.SESSION_ARTIFACT_INVALID)

    if descriptor.name.startswith(_SUPPLEMENTAL_TASK_ARTIFACT_PREFIX):
        task_authoritative = any(
            wave.status is PhaseStatus.COMPLETED
            and descriptor.name in wave.artifacts
            and any(
                descriptor.name in task.artifacts
                and task.status
                in {
                    SupplementalTaskStatus.COMPLETED,
                    SupplementalTaskStatus.PARTIAL,
                }
                for task in wave.tasks.values()
            )
            for wave in manifest.supplemental_waves.values()
        )
        if not task_authoritative:
            raise _Failure(SourceValidationCode.SESSION_ARTIFACT_INVALID)

    if descriptor.name.startswith(_SUPPLEMENTAL_WAVE_ARTIFACT_PREFIX):
        wave_authoritative = any(
            wave.status is PhaseStatus.COMPLETED
            and descriptor.name in wave.artifacts
            for wave in manifest.supplemental_waves.values()
        )
        if not wave_authoritative:
            raise _Failure(SourceValidationCode.SESSION_ARTIFACT_INVALID)


def _registered_schema_or_none(name: str) -> Optional[str]:
    try:
        return artifact_schema(name)
    except (TypeError, ValueError):
        return None


def _read_regular_file_under_root(
    root: Path,
    path: Path,
    max_source_bytes: int,
) -> bytes:
    try:
        resolved_root = Path(root).resolve(strict=True)
        candidate = Path(path)
        relative = candidate.relative_to(resolved_root)
        current = resolved_root
        parts = relative.parts
        for index, part in enumerate(parts):
            current = current / part
            metadata = current.lstat()
            if current.is_symlink():
                raise _Failure(SourceValidationCode.SESSION_ARTIFACT_INVALID)
            if index < len(parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
                raise _Failure(SourceValidationCode.SESSION_ARTIFACT_INVALID)
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(resolved_root)
        metadata = resolved.lstat()
        if not stat.S_ISREG(metadata.st_mode) or resolved.is_symlink():
            raise _Failure(SourceValidationCode.SOURCE_NOT_REGULAR)
        if metadata.st_size > max_source_bytes:
            raise _Failure(SourceValidationCode.SOURCE_TOO_LARGE)
    except _Failure:
        raise
    except (OSError, ValueError):
        raise _Failure(SourceValidationCode.SESSION_ARTIFACT_INVALID)

    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(resolved, flags)
    except OSError:
        raise _Failure(SourceValidationCode.SESSION_ARTIFACT_INVALID)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or not os.path.samestat(metadata, opened)
        ):
            raise _Failure(SourceValidationCode.SESSION_ARTIFACT_INVALID)
        chunks = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_source_bytes + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > max_source_bytes:
                raise _Failure(SourceValidationCode.SOURCE_TOO_LARGE)
            chunks.append(chunk)
        final = resolved.lstat()
        if not os.path.samestat(opened, final):
            raise _Failure(SourceValidationCode.SESSION_ARTIFACT_INVALID)
        return b"".join(chunks)
    except OSError:
        raise _Failure(SourceValidationCode.SESSION_ARTIFACT_INVALID)
    finally:
        os.close(descriptor)


def _require_observation_hash(content: bytes, observation: Observation) -> None:
    digest = hashlib.sha256(content).hexdigest()
    if hmac.compare_digest(digest, observation.content_hash):
        return
    # ObservationStore retains a narrowly scoped compatibility rule for legacy
    # 12-hex IDs whose Windows text write translated LF to CRLF.  Match that
    # authority exactly instead of inventing a broader normalization rule.
    identifier_digest = observation.observation_id.partition("-")[2]
    if len(identifier_digest) == 12:
        try:
            normalized = content.decode("utf-8").replace("\r\n", "\n")
        except UnicodeError:
            raise _Failure(SourceValidationCode.OBSERVATION_UNTRUSTED)
        translated = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if hmac.compare_digest(translated, observation.content_hash):
            return
    raise _Failure(SourceValidationCode.OBSERVATION_UNTRUSTED)


def build_candidate_authority_receipt(
    candidate: MemoryCandidate,
    runtime_provenance: TrustedCandidateProvenance,
    report: SourceValidationReport,
    *,
    validator: SourceValidator,
    current_target_head_sha: str,
    created_at: str,
) -> CandidateAuthorityReceipt:
    """Build a canonical receipt through the live SourceValidator boundary."""

    if type(validator) is not SourceValidator:
        raise SourceValidationError(SourceValidationCode.INVALID_CONFIGURATION)
    return validator.build_candidate_authority_receipt(
        candidate,
        runtime_provenance,
        report,
        current_target_head_sha=current_target_head_sha,
        created_at=created_at,
    )


def restore_candidate_authority(
    receipt: CandidateAuthorityReceipt,
    candidate: MemoryCandidate,
    *,
    validator: SourceValidator,
    current_provenance: TrustedCandidateProvenance,
    current_target_head_sha: str,
) -> CandidateAuthorityRestoration:
    """Restore and revalidate a receipt against independent current authority."""

    if type(validator) is not SourceValidator:
        raise SourceValidationError(SourceValidationCode.INVALID_CONFIGURATION)
    return validator.restore_candidate_authority(
        receipt,
        candidate,
        current_provenance=current_provenance,
        current_target_head_sha=current_target_head_sha,
    )


def hydrate_candidate_authority_receipt(
    receipt: CandidateAuthorityReceipt,
    candidate: MemoryCandidate,
    *,
    validator: SourceValidator,
    current_provenance: TrustedCandidateProvenance,
    current_target_head_sha: str,
) -> CandidateAuthorityRestoration:
    """Compatibility name for strict candidate-authority restoration."""

    return restore_candidate_authority(
        receipt,
        candidate,
        validator=validator,
        current_provenance=current_provenance,
        current_target_head_sha=current_target_head_sha,
    )


__all__ = [
    "CandidateAuthorityRestoration",
    "DEFAULT_MAX_SOURCE_BYTES",
    "HumanDeclarationOrigin",
    "SOURCE_VALIDATION_SCHEMA_VERSION",
    "SensitiveContentFinding",
    "SensitiveContentKind",
    "SensitiveContentScan",
    "SensitivityDecision",
    "SourceValidationCode",
    "SourceValidationError",
    "SourceValidationReport",
    "SourceValidationResult",
    "SourceValidator",
    "TrustedCandidateProvenance",
    "TrustedHumanDeclaration",
    "ValidationIssue",
    "build_candidate_authority_receipt",
    "candidate_authority_resolution_hash",
    "git_commit_metadata_hash",
    "hydrate_candidate_authority_receipt",
    "human_declaration_hash",
    "repository_range_hash",
    "repository_symbol_hash",
    "restore_candidate_authority",
    "scan_sensitive_text",
]
