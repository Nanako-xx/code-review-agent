from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import re
from typing import Any, ClassVar, Mapping, TypeVar


class WireProtocolError(ValueError):
    """Raised when a v6 public wire value is not exact and canonical."""


class IntentSource(str, Enum):
    EXPLICIT = "explicit"
    INFERRED = "inferred"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReviewerRoleKind(str, Enum):
    CORE = "core"
    ADVERSARIAL = "adversarial"
    DYNAMIC = "dynamic"


class FindingSeverity(str, Enum):
    BLOCKER = "blocker"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ReviewResultStatus(str, Enum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class ConversationSpeaker(str, Enum):
    USER = "user"
    ORCHESTRATOR = "orchestrator"


EnumT = TypeVar("EnumT", bound=Enum)

_PR_ID_PATTERN = re.compile(r"\APR-[0-9a-f]{64}\Z")
_SNAPSHOT_ID_PATTERN = re.compile(r"\AS-[0-9a-f]{64}\Z")
_FINDING_ID_PATTERN = re.compile(r"\AF-[0-9a-f]{64}\Z")
_INTENT_ANALYSIS_ID_PATTERN = re.compile(r"\AIA-[0-9a-f]{64}\Z")
_ASSIGNMENT_ID_PATTERN = re.compile(r"\AASG-[0-9a-f]{64}\Z")
_WINDOWS_DRIVE_PATTERN = re.compile(r"\A[A-Za-z]:")


def _strict_json_object(raw: str | bytes | bytearray) -> dict[str, Any]:
    if isinstance(raw, (bytes, bytearray)):
        try:
            text = bytes(raw).decode("utf-8", "strict")
        except UnicodeError as error:
            raise WireProtocolError("wire JSON must be valid UTF-8") from error
    elif type(raw) is str:
        text = raw
    else:
        raise WireProtocolError("wire JSON must be text or UTF-8 bytes")

    def reject_duplicate_keys(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise WireProtocolError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_non_finite(token: str) -> None:
        raise WireProtocolError(f"wire JSON contains a non-finite number: {token}")

    try:
        payload = json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_non_finite,
        )
    except WireProtocolError:
        raise
    except (json.JSONDecodeError, RecursionError, TypeError, ValueError) as error:
        raise WireProtocolError("wire JSON is invalid") from error
    if type(payload) is not dict:
        raise WireProtocolError("wire JSON root must be an object")
    return payload


def _exact_object(
    payload: Mapping[str, Any],
    expected: tuple[str, ...],
    context: str,
) -> dict[str, Any]:
    if type(payload) is not dict:
        raise WireProtocolError(f"{context} must be an object")
    expected_set = set(expected)
    actual_set = set(payload)
    unknown = sorted(actual_set - expected_set)
    if unknown:
        raise WireProtocolError(
            f"{context} has unknown field(s): {', '.join(unknown)}"
        )
    missing = sorted(expected_set - actual_set)
    if missing:
        raise WireProtocolError(
            f"{context} has missing field(s): {', '.join(missing)}"
        )
    return dict(payload)


def _text(value: Any, field_name: str) -> str:
    if type(value) is not str:
        raise WireProtocolError(f"{field_name} must be a string")
    if not value.strip():
        raise WireProtocolError(f"{field_name} must not be empty")
    if "\x00" in value:
        raise WireProtocolError(f"{field_name} contains an unsafe control character")
    return value


def _optional_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _text(value, field_name)


def _text_tuple(value: Any, field_name: str, *, wire: bool = False) -> tuple[str, ...]:
    expected_type = list if wire else tuple
    if type(value) is not expected_type:
        shape = "array" if wire else "tuple"
        raise WireProtocolError(f"{field_name} must be a {shape}")
    return tuple(
        _text(item, f"{field_name}[{index}]") for index, item in enumerate(value)
    )


def _enum_member(enum_type: type[EnumT], value: Any, field_name: str) -> EnumT:
    if type(value) is not str:
        raise WireProtocolError(f"{field_name} must be a string")
    try:
        return enum_type(value)
    except ValueError as error:
        choices = ", ".join(member.value for member in enum_type)
        raise WireProtocolError(f"{field_name} must be one of: {choices}") from error


def _require_enum_instance(
    enum_type: type[EnumT], value: Any, field_name: str
) -> EnumT:
    if not isinstance(value, enum_type):
        choices = ", ".join(member.value for member in enum_type)
        raise WireProtocolError(f"{field_name} must be one of: {choices}")
    return value


def _stable_id(value: Any, pattern: re.Pattern[str], field_name: str) -> str:
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise WireProtocolError(
            f"{field_name} must use its prefix and a full lowercase SHA-256 digest"
        )
    return value


def _repository_path(value: Any) -> str:
    path = _text(value, "path")
    if (
        path.startswith("/")
        or path.endswith("/")
        or "\\" in path
        or ":" in path
        or _WINDOWS_DRIVE_PATTERN.match(path) is not None
    ):
        raise WireProtocolError("path must be a safe repository-relative path")
    parts = path.split("/")
    if any(
        part in {"", ".", ".."} or part != part.strip()
        for part in parts
    ):
        raise WireProtocolError("path must be a canonical repository-relative path")
    return path


def _positive_line(value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise WireProtocolError("line must be a positive integer")
    return value


class WireModel:
    _WIRE_FIELDS: ClassVar[tuple[str, ...]]

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> WireModel:
        raise NotImplementedError

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self)

    def to_json(self) -> str:
        return self.to_json_bytes().decode("utf-8")

    @classmethod
    def from_json(cls, raw: str | bytes | bytearray) -> WireModel:
        return cls.from_dict(_strict_json_object(raw))


def canonical_json_bytes(value: WireModel) -> bytes:
    """Serialize a validated wire model with its declared, fixed field order."""

    if not isinstance(value, WireModel):
        raise WireProtocolError("canonical wire JSON requires a WireModel")
    try:
        return json.dumps(
            value.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise WireProtocolError("wire model is not canonical JSON data") from error


@dataclass(frozen=True)
class ConversationMessage(WireModel):
    speaker: ConversationSpeaker
    content: str

    _WIRE_FIELDS = ("speaker", "content")

    def __post_init__(self) -> None:
        _require_enum_instance(ConversationSpeaker, self.speaker, "speaker")
        _text(self.content, "content")

    def to_dict(self) -> dict[str, Any]:
        return {"speaker": self.speaker.value, "content": self.content}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ConversationMessage:
        value = _exact_object(payload, cls._WIRE_FIELDS, "ConversationMessage")
        return cls(
            speaker=_enum_member(
                ConversationSpeaker, value["speaker"], "speaker"
            ),
            content=_text(value["content"], "content"),
        )


@dataclass(frozen=True)
class ReviewRequest(WireModel):
    conversation: tuple[ConversationMessage, ...]

    _WIRE_FIELDS = ("conversation",)

    def __post_init__(self) -> None:
        if type(self.conversation) is not tuple or not self.conversation:
            raise WireProtocolError("conversation must be a non-empty tuple")
        if any(type(message) is not ConversationMessage for message in self.conversation):
            raise WireProtocolError(
                "conversation must contain only ConversationMessage values"
            )

    def to_dict(self) -> dict[str, Any]:
        return {"conversation": [message.to_dict() for message in self.conversation]}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ReviewRequest:
        value = _exact_object(payload, cls._WIRE_FIELDS, "ReviewRequest")
        conversation = value["conversation"]
        if type(conversation) is not list or not conversation:
            raise WireProtocolError("conversation must be a non-empty array")
        return cls(
            conversation=tuple(
                ConversationMessage.from_dict(message) for message in conversation
            )
        )


@dataclass(frozen=True)
class IntentPacket(WireModel):
    goal: str | None
    source: IntentSource | None
    uncertainties: tuple[str, ...]

    _WIRE_FIELDS = ("goal", "source", "uncertainties")

    def __post_init__(self) -> None:
        _optional_text(self.goal, "goal")
        if self.source is not None:
            _require_enum_instance(IntentSource, self.source, "source")
        _text_tuple(self.uncertainties, "uncertainties")
        if self.goal is None:
            if self.source is not None:
                raise WireProtocolError("source must be null when goal is null")
            if not self.uncertainties:
                raise WireProtocolError(
                    "uncertainties must be non-empty when goal is null"
                )
        elif self.source is None:
            raise WireProtocolError("source is required when goal is present")

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "source": self.source.value if self.source is not None else None,
            "uncertainties": list(self.uncertainties),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> IntentPacket:
        value = _exact_object(payload, cls._WIRE_FIELDS, "IntentPacket")
        source = value["source"]
        return cls(
            goal=_optional_text(value["goal"], "goal"),
            source=(
                None
                if source is None
                else _enum_member(IntentSource, source, "source")
            ),
            uncertainties=_text_tuple(
                value["uncertainties"], "uncertainties", wire=True
            ),
        )


@dataclass(frozen=True)
class IntentVersionEnvelope(WireModel):
    version: int
    source_snapshot_id: str
    packet: IntentPacket
    analysis_record_ref: str

    _WIRE_FIELDS = (
        "version",
        "source_snapshot_id",
        "packet",
        "analysis_record_ref",
    )

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version < 1:
            raise WireProtocolError("version must be a positive integer")
        _stable_id(
            self.source_snapshot_id,
            _SNAPSHOT_ID_PATTERN,
            "source_snapshot_id",
        )
        if type(self.packet) is not IntentPacket:
            raise WireProtocolError("packet must be an IntentPacket")
        _stable_id(
            self.analysis_record_ref,
            _INTENT_ANALYSIS_ID_PATTERN,
            "analysis_record_ref",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source_snapshot_id": self.source_snapshot_id,
            "packet": self.packet.to_dict(),
            "analysis_record_ref": self.analysis_record_ref,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "IntentVersionEnvelope":
        value = _exact_object(payload, cls._WIRE_FIELDS, "IntentVersionEnvelope")
        version = value["version"]
        if type(version) is not int or version < 1:
            raise WireProtocolError("version must be a positive integer")
        return cls(
            version=version,
            source_snapshot_id=_stable_id(
                value["source_snapshot_id"],
                _SNAPSHOT_ID_PATTERN,
                "source_snapshot_id",
            ),
            packet=IntentPacket.from_dict(value["packet"]),
            analysis_record_ref=_stable_id(
                value["analysis_record_ref"],
                _INTENT_ANALYSIS_ID_PATTERN,
                "analysis_record_ref",
            ),
        )


@dataclass(frozen=True)
class RiskDecision(WireModel):
    level: RiskLevel

    _WIRE_FIELDS = ("level",)

    def __post_init__(self) -> None:
        _require_enum_instance(RiskLevel, self.level, "level")

    def to_dict(self) -> dict[str, Any]:
        return {"level": self.level.value}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RiskDecision:
        value = _exact_object(payload, cls._WIRE_FIELDS, "RiskDecision")
        return cls(level=_enum_member(RiskLevel, value["level"], "level"))


@dataclass(frozen=True)
class AssignmentTargets(WireModel):
    files: tuple[str, ...]
    symbols: tuple[str, ...]
    hunks: tuple[str, ...]

    _WIRE_FIELDS = ("files", "symbols", "hunks")

    def __post_init__(self) -> None:
        files = _text_tuple(self.files, "files")
        for file_path in files:
            _repository_path(file_path)
        _text_tuple(self.symbols, "symbols")
        _text_tuple(self.hunks, "hunks")
        for name, values in (
            ("files", self.files),
            ("symbols", self.symbols),
            ("hunks", self.hunks),
        ):
            if len(values) != len(set(values)):
                raise WireProtocolError(f"{name} must not contain duplicates")

    def to_dict(self) -> dict[str, Any]:
        return {
            "files": list(self.files),
            "symbols": list(self.symbols),
            "hunks": list(self.hunks),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AssignmentTargets":
        value = _exact_object(payload, cls._WIRE_FIELDS, "AssignmentTargets")
        return cls(
            files=_text_tuple(value["files"], "files", wire=True),
            symbols=_text_tuple(value["symbols"], "symbols", wire=True),
            hunks=_text_tuple(value["hunks"], "hunks", wire=True),
        )


@dataclass(frozen=True)
class ReviewerAssignment(WireModel):
    assignment_id: str
    snapshot_id: str
    role: str
    role_kind: ReviewerRoleKind
    perspective: str | None
    mission: str
    targets: AssignmentTargets
    checks: tuple[str, ...]
    permissions: tuple[str, ...]

    _WIRE_FIELDS = (
        "assignment_id",
        "snapshot_id",
        "role",
        "role_kind",
        "perspective",
        "mission",
        "targets",
        "checks",
        "permissions",
    )

    def __post_init__(self) -> None:
        _stable_id(self.assignment_id, _ASSIGNMENT_ID_PATTERN, "assignment_id")
        _stable_id(self.snapshot_id, _SNAPSHOT_ID_PATTERN, "snapshot_id")
        _text(self.role, "role")
        _require_enum_instance(ReviewerRoleKind, self.role_kind, "role_kind")
        _optional_text(self.perspective, "perspective")
        if self.role_kind is ReviewerRoleKind.DYNAMIC and self.perspective is None:
            raise WireProtocolError("dynamic Assignment requires a perspective")
        _text(self.mission, "mission")
        if type(self.targets) is not AssignmentTargets:
            raise WireProtocolError("targets must be AssignmentTargets")
        _text_tuple(self.checks, "checks")
        permissions = _text_tuple(self.permissions, "permissions")
        if not permissions:
            raise WireProtocolError("permissions must not be empty")
        if len(permissions) != len(set(permissions)):
            raise WireProtocolError("permissions must not contain duplicates")

    def to_dict(self) -> dict[str, Any]:
        return {
            "assignment_id": self.assignment_id,
            "snapshot_id": self.snapshot_id,
            "role": self.role,
            "role_kind": self.role_kind.value,
            "perspective": self.perspective,
            "mission": self.mission,
            "targets": self.targets.to_dict(),
            "checks": list(self.checks),
            "permissions": list(self.permissions),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReviewerAssignment":
        value = _exact_object(payload, cls._WIRE_FIELDS, "ReviewerAssignment")
        return cls(
            assignment_id=_stable_id(
                value["assignment_id"],
                _ASSIGNMENT_ID_PATTERN,
                "assignment_id",
            ),
            snapshot_id=_stable_id(
                value["snapshot_id"], _SNAPSHOT_ID_PATTERN, "snapshot_id"
            ),
            role=_text(value["role"], "role"),
            role_kind=_enum_member(
                ReviewerRoleKind, value["role_kind"], "role_kind"
            ),
            perspective=_optional_text(value["perspective"], "perspective"),
            mission=_text(value["mission"], "mission"),
            targets=AssignmentTargets.from_dict(value["targets"]),
            checks=_text_tuple(value["checks"], "checks", wire=True),
            permissions=_text_tuple(
                value["permissions"], "permissions", wire=True
            ),
        )


@dataclass(frozen=True)
class ReviewPlan(WireModel):
    snapshot_id: str
    risk_level: RiskLevel
    assignments: tuple[ReviewerAssignment, ...]

    _WIRE_FIELDS = ("snapshot_id", "risk_level", "assignments")

    def __post_init__(self) -> None:
        _stable_id(self.snapshot_id, _SNAPSHOT_ID_PATTERN, "snapshot_id")
        _require_enum_instance(RiskLevel, self.risk_level, "risk_level")
        if type(self.assignments) is not tuple or not self.assignments:
            raise WireProtocolError("assignments must be a non-empty tuple")
        if any(
            type(assignment) is not ReviewerAssignment
            for assignment in self.assignments
        ):
            raise WireProtocolError(
                "assignments must contain only ReviewerAssignment values"
            )
        if any(
            assignment.snapshot_id != self.snapshot_id
            for assignment in self.assignments
        ):
            raise WireProtocolError(
                "all Assignments must be bound to the ReviewPlan Snapshot"
            )
        assignment_ids = [
            assignment.assignment_id for assignment in self.assignments
        ]
        if len(assignment_ids) != len(set(assignment_ids)):
            raise WireProtocolError("assignments must have unique assignment_id values")

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "risk_level": self.risk_level.value,
            "assignments": [
                assignment.to_dict() for assignment in self.assignments
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReviewPlan":
        value = _exact_object(payload, cls._WIRE_FIELDS, "ReviewPlan")
        assignments = value["assignments"]
        if type(assignments) is not list or not assignments:
            raise WireProtocolError("assignments must be a non-empty array")
        return cls(
            snapshot_id=_stable_id(
                value["snapshot_id"], _SNAPSHOT_ID_PATTERN, "snapshot_id"
            ),
            risk_level=_enum_member(
                RiskLevel, value["risk_level"], "risk_level"
            ),
            assignments=tuple(
                ReviewerAssignment.from_dict(item) for item in assignments
            ),
        )


def _validate_finding(
    *,
    claim: Any,
    severity: Any,
    path: Any,
    line: Any,
    suggestion: Any,
) -> None:
    _text(claim, "claim")
    _require_enum_instance(FindingSeverity, severity, "severity")
    _repository_path(path)
    _positive_line(line)
    _text(suggestion, "suggestion")


@dataclass(frozen=True)
class ReviewerFinding(WireModel):
    claim: str
    severity: FindingSeverity
    path: str
    line: int
    suggestion: str

    _WIRE_FIELDS = ("claim", "severity", "path", "line", "suggestion")

    def __post_init__(self) -> None:
        _validate_finding(
            claim=self.claim,
            severity=self.severity,
            path=self.path,
            line=self.line,
            suggestion=self.suggestion,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "severity": self.severity.value,
            "path": self.path,
            "line": self.line,
            "suggestion": self.suggestion,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ReviewerFinding:
        value = _exact_object(payload, cls._WIRE_FIELDS, "ReviewerFinding")
        return cls(
            claim=_text(value["claim"], "claim"),
            severity=_enum_member(
                FindingSeverity, value["severity"], "severity"
            ),
            path=_repository_path(value["path"]),
            line=_positive_line(value["line"]),
            suggestion=_text(value["suggestion"], "suggestion"),
        )


@dataclass(frozen=True)
class ReviewerOutput(WireModel):
    findings: tuple[ReviewerFinding, ...]
    uncertainties: tuple[str, ...]

    _WIRE_FIELDS = ("findings", "uncertainties")

    def __post_init__(self) -> None:
        if type(self.findings) is not tuple or any(
            type(finding) is not ReviewerFinding for finding in self.findings
        ):
            raise WireProtocolError(
                "findings must be a tuple of ReviewerFinding values"
            )
        _text_tuple(self.uncertainties, "uncertainties")

    def to_dict(self) -> dict[str, Any]:
        return {
            "findings": [finding.to_dict() for finding in self.findings],
            "uncertainties": list(self.uncertainties),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ReviewerOutput:
        value = _exact_object(payload, cls._WIRE_FIELDS, "ReviewerOutput")
        findings = value["findings"]
        if type(findings) is not list:
            raise WireProtocolError("findings must be an array")
        return cls(
            findings=tuple(ReviewerFinding.from_dict(item) for item in findings),
            uncertainties=_text_tuple(
                value["uncertainties"], "uncertainties", wire=True
            ),
        )


@dataclass(frozen=True)
class FinalFinding(WireModel):
    finding_id: str
    claim: str
    severity: FindingSeverity
    path: str
    line: int
    suggestion: str

    _WIRE_FIELDS = (
        "finding_id",
        "claim",
        "severity",
        "path",
        "line",
        "suggestion",
    )

    def __post_init__(self) -> None:
        _stable_id(self.finding_id, _FINDING_ID_PATTERN, "finding_id")
        _validate_finding(
            claim=self.claim,
            severity=self.severity,
            path=self.path,
            line=self.line,
            suggestion=self.suggestion,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "claim": self.claim,
            "severity": self.severity.value,
            "path": self.path,
            "line": self.line,
            "suggestion": self.suggestion,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FinalFinding:
        value = _exact_object(payload, cls._WIRE_FIELDS, "FinalFinding")
        return cls(
            finding_id=_stable_id(
                value["finding_id"], _FINDING_ID_PATTERN, "finding_id"
            ),
            claim=_text(value["claim"], "claim"),
            severity=_enum_member(
                FindingSeverity, value["severity"], "severity"
            ),
            path=_repository_path(value["path"]),
            line=_positive_line(value["line"]),
            suggestion=_text(value["suggestion"], "suggestion"),
        )


@dataclass(frozen=True)
class ReviewResult(WireModel):
    pr_id: str
    snapshot_id: str
    status: ReviewResultStatus
    risk_level: RiskLevel
    findings: tuple[FinalFinding, ...]
    uncertainties: tuple[str, ...]

    _WIRE_FIELDS = (
        "pr_id",
        "snapshot_id",
        "status",
        "risk_level",
        "findings",
        "uncertainties",
    )

    def __post_init__(self) -> None:
        _stable_id(self.pr_id, _PR_ID_PATTERN, "pr_id")
        _stable_id(self.snapshot_id, _SNAPSHOT_ID_PATTERN, "snapshot_id")
        _require_enum_instance(ReviewResultStatus, self.status, "status")
        _require_enum_instance(RiskLevel, self.risk_level, "risk_level")
        if type(self.findings) is not tuple or any(
            type(finding) is not FinalFinding for finding in self.findings
        ):
            raise WireProtocolError("findings must be a tuple of FinalFinding values")
        _text_tuple(self.uncertainties, "uncertainties")

    def to_dict(self) -> dict[str, Any]:
        return {
            "pr_id": self.pr_id,
            "snapshot_id": self.snapshot_id,
            "status": self.status.value,
            "risk_level": self.risk_level.value,
            "findings": [finding.to_dict() for finding in self.findings],
            "uncertainties": list(self.uncertainties),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ReviewResult:
        value = _exact_object(payload, cls._WIRE_FIELDS, "ReviewResult")
        findings = value["findings"]
        if type(findings) is not list:
            raise WireProtocolError("findings must be an array")
        return cls(
            pr_id=_stable_id(value["pr_id"], _PR_ID_PATTERN, "pr_id"),
            snapshot_id=_stable_id(
                value["snapshot_id"], _SNAPSHOT_ID_PATTERN, "snapshot_id"
            ),
            status=_enum_member(
                ReviewResultStatus, value["status"], "status"
            ),
            risk_level=_enum_member(
                RiskLevel, value["risk_level"], "risk_level"
            ),
            findings=tuple(FinalFinding.from_dict(item) for item in findings),
            uncertainties=_text_tuple(
                value["uncertainties"], "uncertainties", wire=True
            ),
        )


__all__ = [
    "AssignmentTargets",
    "ConversationMessage",
    "ConversationSpeaker",
    "FinalFinding",
    "FindingSeverity",
    "IntentPacket",
    "IntentSource",
    "IntentVersionEnvelope",
    "ReviewRequest",
    "ReviewerAssignment",
    "ReviewerFinding",
    "ReviewerOutput",
    "ReviewerRoleKind",
    "ReviewPlan",
    "ReviewResult",
    "ReviewResultStatus",
    "RiskDecision",
    "RiskLevel",
    "WireModel",
    "WireProtocolError",
    "canonical_json_bytes",
]
