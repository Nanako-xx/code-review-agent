"""Compile approved Durable Memory policy effects into safe Runtime actions.

This module is a deliberately narrow authority boundary.  It accepts canonical,
active :class:`DurableMemoryRecord` values, validates their already-approved
``policy_effect`` against caller-owned immutable registries, and emits immutable
declarative actions.  It does not resolve verification templates to commands and
has no representation for filesystem, network, tool, shell, model-budget, or
external-side-effect grants.

Malformed, unknown, inactive, conflicting, or registry-missing hard policies are
reported as blocking diagnostics.  Ordinary records without a ``policy_effect``
remain informational and never become Runtime policy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from review_agent.memory_models import (
    MAX_SNAPSHOT_RECORDS,
    MODEL_SCHEMA_VERSION,
    DurableMemoryRecord,
    PolicyEffect,
    PolicyEffectKind,
    RecordStatus,
    Sensitivity,
    validate_stable_id,
)
from review_agent.models import RiskLevel


MEMORY_POLICY_COMPILER_VERSION = "memory_policy_v1"
MAX_POLICY_REGISTRY_ITEMS = 10_000

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+#/@-]{0,511}$")
_RISK_LEVELS = (
    RiskLevel.LOW,
    RiskLevel.MEDIUM,
    RiskLevel.HIGH,
    RiskLevel.CRITICAL,
)


class RuntimeActionKind(str, Enum):
    """The complete allowlist of effects this compiler can grant."""

    RAISE_RISK_FLOOR = "raise_risk_floor"
    REQUIRE_CONTRACT = "require_contract"
    REQUIRE_CHECK = "require_check"
    VERIFICATION_HINT = "verification_hint"


class PolicyDiagnosticSeverity(str, Enum):
    INFO = "info"
    BLOCKING = "blocking"


class PolicyDiagnosticCode(str, Enum):
    INPUT_LIMIT_EXCEEDED = "input_limit_exceeded"
    INVALID_RECORD_TYPE = "invalid_record_type"
    INVALID_MEMORY_ID = "invalid_memory_id"
    NON_CANONICAL_RECORD = "non_canonical_record"
    CONFLICTING_MEMORY_RECORD = "conflicting_memory_record"
    INACTIVE_RECORD = "inactive_record"
    BLOCKED_SENSITIVITY = "blocked_sensitivity"
    UNTYPED_POLICY_EFFECT = "untyped_policy_effect"
    UNKNOWN_POLICY_EFFECT = "unknown_policy_effect"
    INVALID_POLICY_EFFECT = "invalid_policy_effect"
    UNKNOWN_CONTRACT = "unknown_contract"
    UNKNOWN_CHECK = "unknown_check"
    UNKNOWN_COMMAND_TEMPLATE = "unknown_command_template"
    RISK_FLOOR_NOT_RAISED = "risk_floor_not_raised"


class PolicyDisposition(str, Enum):
    APPLIED = "applied"
    INFORMATIONAL = "informational"
    NOT_APPLIED = "not_applied"
    REJECTED = "rejected"


def _validated_identifier(value: Any, field_name: str) -> str:
    if type(value) is not str or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise ValueError("%s must be a bounded policy identifier" % field_name)
    return value


def _canonical_registry_ids(values: Any, field_name: str) -> Tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("%s must be an iterable of identifiers" % field_name)
    try:
        items = tuple(values)
    except TypeError as error:
        raise ValueError("%s must be an iterable of identifiers" % field_name) from error
    if len(items) > MAX_POLICY_REGISTRY_ITEMS:
        raise ValueError(
            "%s exceeds the maximum size of %d"
            % (field_name, MAX_POLICY_REGISTRY_ITEMS)
        )
    return tuple(
        sorted({_validated_identifier(item, "%s item" % field_name) for item in items})
    )


def _canonical_memory_ids(values: Any) -> Tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("memory_ids must be an iterable of Memory IDs")
    try:
        items = tuple(values)
    except TypeError as error:
        raise ValueError("memory_ids must be an iterable of Memory IDs") from error
    if not items:
        raise ValueError("Runtime actions require at least one source memory ID")
    validated = {
        validate_stable_id(item, "MEM", "memory_ids item") for item in items
    }
    return tuple(sorted(validated))


def _require_risk_level(value: Any, field_name: str) -> RiskLevel:
    if type(value) is not RiskLevel:
        raise ValueError("%s must be a RiskLevel" % field_name)
    return value


def _risk_rank(value: RiskLevel) -> int:
    return _RISK_LEVELS.index(value)


@dataclass(frozen=True)
class RuntimePolicyRegistry:
    """Caller-owned catalogs of IDs that approved effects may reference.

    The command-template catalog contains identifiers only.  It intentionally
    cannot carry a command line, executable, environment, or callable.
    """

    contract_ids: Tuple[str, ...] = ()
    check_ids: Tuple[str, ...] = ()
    command_template_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "contract_ids",
            _canonical_registry_ids(self.contract_ids, "contract_ids"),
        )
        object.__setattr__(
            self,
            "check_ids",
            _canonical_registry_ids(self.check_ids, "check_ids"),
        )
        object.__setattr__(
            self,
            "command_template_ids",
            _canonical_registry_ids(
                self.command_template_ids,
                "command_template_ids",
            ),
        )


@dataclass(frozen=True)
class RaiseRiskFloorAction:
    minimum_level: RiskLevel
    memory_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        _require_risk_level(self.minimum_level, "minimum_level")
        object.__setattr__(self, "memory_ids", _canonical_memory_ids(self.memory_ids))

    @property
    def kind(self) -> RuntimeActionKind:
        return RuntimeActionKind.RAISE_RISK_FLOOR

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.kind.value,
            "minimum_level": self.minimum_level.value,
            "memory_ids": list(self.memory_ids),
        }


@dataclass(frozen=True)
class RequireContractAction:
    contract_id: str
    memory_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "contract_id",
            _validated_identifier(self.contract_id, "contract_id"),
        )
        object.__setattr__(self, "memory_ids", _canonical_memory_ids(self.memory_ids))

    @property
    def kind(self) -> RuntimeActionKind:
        return RuntimeActionKind.REQUIRE_CONTRACT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.kind.value,
            "contract_id": self.contract_id,
            "memory_ids": list(self.memory_ids),
        }


@dataclass(frozen=True)
class RequireCheckAction:
    check_id: str
    memory_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "check_id",
            _validated_identifier(self.check_id, "check_id"),
        )
        object.__setattr__(self, "memory_ids", _canonical_memory_ids(self.memory_ids))

    @property
    def kind(self) -> RuntimeActionKind:
        return RuntimeActionKind.REQUIRE_CHECK

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.kind.value,
            "check_id": self.check_id,
            "memory_ids": list(self.memory_ids),
        }


@dataclass(frozen=True)
class VerificationHintAction:
    command_template_id: str
    memory_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "command_template_id",
            _validated_identifier(
                self.command_template_id,
                "command_template_id",
            ),
        )
        object.__setattr__(self, "memory_ids", _canonical_memory_ids(self.memory_ids))

    @property
    def kind(self) -> RuntimeActionKind:
        return RuntimeActionKind.VERIFICATION_HINT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.kind.value,
            "command_template_id": self.command_template_id,
            "memory_ids": list(self.memory_ids),
        }


RuntimePolicyAction = Union[
    RaiseRiskFloorAction,
    RequireContractAction,
    RequireCheckAction,
    VerificationHintAction,
]
_RUNTIME_ACTION_TYPES = (
    RaiseRiskFloorAction,
    RequireContractAction,
    RequireCheckAction,
    VerificationHintAction,
)


@dataclass(frozen=True)
class PolicyDiagnostic:
    code: PolicyDiagnosticCode
    severity: PolicyDiagnosticSeverity
    message: str
    memory_id: Optional[str] = None

    def __post_init__(self) -> None:
        if type(self.code) is not PolicyDiagnosticCode:
            raise ValueError("diagnostic.code must be a PolicyDiagnosticCode")
        if type(self.severity) is not PolicyDiagnosticSeverity:
            raise ValueError("diagnostic.severity must be a PolicyDiagnosticSeverity")
        if type(self.message) is not str or not self.message:
            raise ValueError("diagnostic.message must be a non-empty string")
        if self.memory_id is not None:
            validate_stable_id(self.memory_id, "MEM", "diagnostic.memory_id")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code.value,
            "severity": self.severity.value,
            "message": self.message,
            "memory_id": self.memory_id,
        }


@dataclass(frozen=True)
class PolicyProvenance:
    """One deterministic audit row per distinct Memory ID seen by the compiler."""

    memory_id: str
    disposition: PolicyDisposition
    candidate_id: Optional[str] = None
    approved_by: Optional[str] = None
    approval_event_id: Optional[str] = None
    effect_kind: Optional[PolicyEffectKind] = None
    effect_value: Optional[str] = None
    runtime_action_kind: Optional[RuntimeActionKind] = None
    diagnostic_codes: Tuple[PolicyDiagnosticCode, ...] = ()

    def __post_init__(self) -> None:
        validate_stable_id(self.memory_id, "MEM", "provenance.memory_id")
        if type(self.disposition) is not PolicyDisposition:
            raise ValueError("provenance.disposition must be a PolicyDisposition")
        if self.candidate_id is not None:
            validate_stable_id(
                self.candidate_id,
                "MC",
                "provenance.candidate_id",
            )
        if self.approval_event_id is not None:
            validate_stable_id(
                self.approval_event_id,
                "EVT",
                "provenance.approval_event_id",
            )
        if self.approved_by is not None and (
            type(self.approved_by) is not str or not self.approved_by
        ):
            raise ValueError("provenance.approved_by must be a non-empty string or None")
        if self.effect_kind is None:
            if self.effect_value is not None:
                raise ValueError("effect_value requires an effect_kind")
        else:
            if type(self.effect_kind) is not PolicyEffectKind:
                raise ValueError("effect_kind must be a PolicyEffectKind or None")
            if self.effect_kind is PolicyEffectKind.RISK_FLOOR:
                if type(self.effect_value) is not str:
                    raise ValueError("risk-floor provenance requires a string value")
                try:
                    RiskLevel(self.effect_value)
                except ValueError as error:
                    raise ValueError("risk-floor provenance has an invalid value") from error
            else:
                _validated_identifier(self.effect_value, "provenance.effect_value")
        if self.runtime_action_kind is not None and (
            type(self.runtime_action_kind) is not RuntimeActionKind
        ):
            raise ValueError("runtime_action_kind must be a RuntimeActionKind or None")
        if self.disposition is PolicyDisposition.APPLIED:
            if self.effect_kind is None or self.runtime_action_kind is None:
                raise ValueError("applied provenance must identify its effect and action")
        elif self.runtime_action_kind is not None:
            raise ValueError("only applied provenance may identify a Runtime action")
        if self.disposition is PolicyDisposition.INFORMATIONAL and self.effect_kind is not None:
            raise ValueError("informational provenance cannot contain a policy effect")
        codes = tuple(self.diagnostic_codes)
        if any(type(item) is not PolicyDiagnosticCode for item in codes):
            raise ValueError("diagnostic_codes must contain PolicyDiagnosticCode values")
        object.__setattr__(
            self,
            "diagnostic_codes",
            tuple(sorted(set(codes), key=lambda item: item.value)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "candidate_id": self.candidate_id,
            "approved_by": self.approved_by,
            "approval_event_id": self.approval_event_id,
            "effect_kind": None if self.effect_kind is None else self.effect_kind.value,
            "effect_value": self.effect_value,
            "disposition": self.disposition.value,
            "runtime_action_kind": (
                None
                if self.runtime_action_kind is None
                else self.runtime_action_kind.value
            ),
            "diagnostic_codes": [item.value for item in self.diagnostic_codes],
        }


def _action_value(action: RuntimePolicyAction) -> str:
    if type(action) is RaiseRiskFloorAction:
        return action.minimum_level.value
    if type(action) is RequireContractAction:
        return action.contract_id
    if type(action) is RequireCheckAction:
        return action.check_id
    if type(action) is VerificationHintAction:
        return action.command_template_id
    raise ValueError("unsupported Runtime policy action type")


def _effect_kind_for_action(action: RuntimePolicyAction) -> PolicyEffectKind:
    if type(action) is RaiseRiskFloorAction:
        return PolicyEffectKind.RISK_FLOOR
    if type(action) is RequireContractAction:
        return PolicyEffectKind.REQUIRE_CONTRACT
    if type(action) is RequireCheckAction:
        return PolicyEffectKind.REQUIRE_CHECK
    if type(action) is VerificationHintAction:
        return PolicyEffectKind.VERIFICATION_HINT
    raise ValueError("unsupported Runtime policy action type")


def _action_sort_key(action: RuntimePolicyAction) -> Tuple[int, int, str]:
    if type(action) is RaiseRiskFloorAction:
        return (0, _risk_rank(action.minimum_level), action.minimum_level.value)
    if type(action) is RequireContractAction:
        return (1, 0, action.contract_id)
    if type(action) is RequireCheckAction:
        return (2, 0, action.check_id)
    if type(action) is VerificationHintAction:
        return (3, 0, action.command_template_id)
    raise ValueError("unsupported Runtime policy action type")


def _diagnostic_sort_key(
    diagnostic: PolicyDiagnostic,
) -> Tuple[str, str, str, str]:
    return (
        diagnostic.memory_id or "",
        diagnostic.code.value,
        diagnostic.severity.value,
        diagnostic.message,
    )


@dataclass(frozen=True)
class PolicyCompilation:
    initial_risk_floor: RiskLevel
    effective_risk_floor: RiskLevel
    actions: Tuple[RuntimePolicyAction, ...]
    diagnostics: Tuple[PolicyDiagnostic, ...]
    provenance: Tuple[PolicyProvenance, ...]
    policy_version: str = MEMORY_POLICY_COMPILER_VERSION

    def __post_init__(self) -> None:
        _require_risk_level(self.initial_risk_floor, "initial_risk_floor")
        _require_risk_level(self.effective_risk_floor, "effective_risk_floor")
        if self.policy_version != MEMORY_POLICY_COMPILER_VERSION:
            raise ValueError("unsupported memory policy compiler version")

        actions = tuple(self.actions)
        if any(type(item) not in _RUNTIME_ACTION_TYPES for item in actions):
            raise ValueError("actions contain an unsupported Runtime action")
        actions = tuple(sorted(actions, key=_action_sort_key))
        action_keys = tuple((item.kind, _action_value(item)) for item in actions)
        if len(action_keys) != len(set(action_keys)):
            raise ValueError("actions must be deduplicated by kind and value")

        diagnostics = tuple(self.diagnostics)
        if any(type(item) is not PolicyDiagnostic for item in diagnostics):
            raise ValueError("diagnostics must contain PolicyDiagnostic values")
        diagnostics = tuple(sorted(set(diagnostics), key=_diagnostic_sort_key))

        provenance = tuple(self.provenance)
        if any(type(item) is not PolicyProvenance for item in provenance):
            raise ValueError("provenance must contain PolicyProvenance values")
        provenance = tuple(sorted(provenance, key=lambda item: item.memory_id))
        provenance_ids = tuple(item.memory_id for item in provenance)
        if len(provenance_ids) != len(set(provenance_ids)):
            raise ValueError("provenance must contain at most one row per memory ID")

        highest = self.initial_risk_floor
        action_sources: Dict[Tuple[PolicyEffectKind, str], Set[str]] = {}
        for action in actions:
            if type(action) is RaiseRiskFloorAction:
                if _risk_rank(action.minimum_level) <= _risk_rank(
                    self.initial_risk_floor
                ):
                    raise ValueError("risk-floor actions must strictly raise the initial floor")
                if _risk_rank(action.minimum_level) > _risk_rank(highest):
                    highest = action.minimum_level
            key = (_effect_kind_for_action(action), _action_value(action))
            action_sources[key] = set(action.memory_ids)

        if highest is not self.effective_risk_floor:
            raise ValueError("effective_risk_floor does not match compiled actions")

        provenance_sources: Dict[Tuple[PolicyEffectKind, str], Set[str]] = {}
        for item in provenance:
            if item.disposition is PolicyDisposition.APPLIED:
                key = (item.effect_kind, item.effect_value)
                provenance_sources.setdefault(key, set()).add(item.memory_id)
                expected_action_kind = _runtime_kind_for_effect(item.effect_kind)
                if item.runtime_action_kind is not expected_action_kind:
                    raise ValueError("applied provenance names the wrong Runtime action")
        if action_sources != provenance_sources:
            raise ValueError("Runtime actions and applied provenance do not agree")

        diagnostic_codes_by_memory: Dict[str, Set[PolicyDiagnosticCode]] = {}
        for diagnostic in diagnostics:
            if diagnostic.memory_id is not None:
                diagnostic_codes_by_memory.setdefault(
                    diagnostic.memory_id,
                    set(),
                ).add(diagnostic.code)
        for item in provenance:
            if not set(item.diagnostic_codes).issubset(
                diagnostic_codes_by_memory.get(item.memory_id, set())
            ):
                raise ValueError("provenance references a missing diagnostic")

        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "diagnostics", diagnostics)
        object.__setattr__(self, "provenance", provenance)

    @property
    def blocked(self) -> bool:
        return any(
            item.severity is PolicyDiagnosticSeverity.BLOCKING
            for item in self.diagnostics
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "initial_risk_floor": self.initial_risk_floor.value,
            "effective_risk_floor": self.effective_risk_floor.value,
            "blocked": self.blocked,
            "actions": [item.to_dict() for item in self.actions],
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "provenance": [item.to_dict() for item in self.provenance],
        }


def _runtime_kind_for_effect(effect_kind: PolicyEffectKind) -> RuntimeActionKind:
    mapping = {
        PolicyEffectKind.RISK_FLOOR: RuntimeActionKind.RAISE_RISK_FLOOR,
        PolicyEffectKind.REQUIRE_CONTRACT: RuntimeActionKind.REQUIRE_CONTRACT,
        PolicyEffectKind.REQUIRE_CHECK: RuntimeActionKind.REQUIRE_CHECK,
        PolicyEffectKind.VERIFICATION_HINT: RuntimeActionKind.VERIFICATION_HINT,
    }
    try:
        return mapping[effect_kind]
    except KeyError as error:
        raise ValueError("unsupported policy effect kind") from error


def _diagnostic_message(code: PolicyDiagnosticCode) -> str:
    messages = {
        PolicyDiagnosticCode.INPUT_LIMIT_EXCEEDED: (
            "policy input exceeded the bounded snapshot record limit"
        ),
        PolicyDiagnosticCode.INVALID_RECORD_TYPE: (
            "policy input must contain canonical DurableMemoryRecord values"
        ),
        PolicyDiagnosticCode.INVALID_MEMORY_ID: (
            "policy input contained an invalid Memory ID"
        ),
        PolicyDiagnosticCode.NON_CANONICAL_RECORD: (
            "Durable Memory record failed canonical validation"
        ),
        PolicyDiagnosticCode.CONFLICTING_MEMORY_RECORD: (
            "the same Memory ID resolved to conflicting canonical records"
        ),
        PolicyDiagnosticCode.INACTIVE_RECORD: (
            "only active Durable Memory records may compile policy"
        ),
        PolicyDiagnosticCode.BLOCKED_SENSITIVITY: (
            "blocked-sensitivity Memory cannot compile Runtime policy"
        ),
        PolicyDiagnosticCode.UNTYPED_POLICY_EFFECT: (
            "policy_effect must be the canonical typed PolicyEffect model"
        ),
        PolicyDiagnosticCode.UNKNOWN_POLICY_EFFECT: (
            "policy_effect kind is not in the Runtime allowlist"
        ),
        PolicyDiagnosticCode.INVALID_POLICY_EFFECT: (
            "policy_effect parameters failed canonical validation"
        ),
        PolicyDiagnosticCode.UNKNOWN_CONTRACT: (
            "required contract is absent from the caller's local registry"
        ),
        PolicyDiagnosticCode.UNKNOWN_CHECK: (
            "required check is absent from the caller's local registry"
        ),
        PolicyDiagnosticCode.UNKNOWN_COMMAND_TEMPLATE: (
            "verification template ID is absent from the caller's local registry"
        ),
        PolicyDiagnosticCode.RISK_FLOOR_NOT_RAISED: (
            "requested risk floor does not exceed the caller's current floor"
        ),
    }
    return messages[code]


def _diagnostic(
    code: PolicyDiagnosticCode,
    memory_id: Optional[str] = None,
) -> PolicyDiagnostic:
    severity = (
        PolicyDiagnosticSeverity.INFO
        if code is PolicyDiagnosticCode.RISK_FLOOR_NOT_RAISED
        else PolicyDiagnosticSeverity.BLOCKING
    )
    return PolicyDiagnostic(
        code=code,
        severity=severity,
        message=_diagnostic_message(code),
        memory_id=memory_id,
    )


def _canonical_effect_code(effect: Any) -> Optional[PolicyDiagnosticCode]:
    if type(effect) is not PolicyEffect:
        return PolicyDiagnosticCode.UNTYPED_POLICY_EFFECT
    if type(effect.effect_kind) is not PolicyEffectKind:
        return PolicyDiagnosticCode.UNKNOWN_POLICY_EFFECT
    if type(effect.value) is not str or type(effect.schema_version) is not int:
        return PolicyDiagnosticCode.INVALID_POLICY_EFFECT
    if effect.schema_version != MODEL_SCHEMA_VERSION:
        return PolicyDiagnosticCode.INVALID_POLICY_EFFECT
    if effect.effect_kind is PolicyEffectKind.RISK_FLOOR:
        try:
            RiskLevel(effect.value)
        except ValueError:
            return PolicyDiagnosticCode.INVALID_POLICY_EFFECT
    elif _IDENTIFIER_PATTERN.fullmatch(effect.value) is None:
        return PolicyDiagnosticCode.INVALID_POLICY_EFFECT
    try:
        canonical = PolicyEffect(
            effect_kind=effect.effect_kind,
            value=effect.value,
            schema_version=effect.schema_version,
        )
    except (TypeError, ValueError):
        return PolicyDiagnosticCode.INVALID_POLICY_EFFECT
    if canonical != effect:
        return PolicyDiagnosticCode.INVALID_POLICY_EFFECT
    return None


def _canonical_record(
    record: DurableMemoryRecord,
) -> Tuple[Optional[DurableMemoryRecord], Optional[PolicyDiagnosticCode]]:
    effect = record.policy_effect
    if effect is not None:
        effect_code = _canonical_effect_code(effect)
        if effect_code is not None:
            return None, effect_code
    try:
        canonical = DurableMemoryRecord.from_dict(record.to_dict())
    except (AttributeError, KeyError, TypeError, ValueError):
        return None, PolicyDiagnosticCode.NON_CANONICAL_RECORD
    if canonical != record:
        return None, PolicyDiagnosticCode.NON_CANONICAL_RECORD
    return canonical, None


def _full_provenance(
    record: DurableMemoryRecord,
    disposition: PolicyDisposition,
    *,
    runtime_action_kind: Optional[RuntimeActionKind] = None,
    diagnostic_codes: Sequence[PolicyDiagnosticCode] = (),
) -> PolicyProvenance:
    effect = record.policy_effect
    return PolicyProvenance(
        memory_id=record.memory_id,
        candidate_id=record.candidate_id,
        approved_by=record.approved_by,
        approval_event_id=record.approval_event_id,
        effect_kind=None if effect is None else effect.effect_kind,
        effect_value=None if effect is None else effect.value,
        disposition=disposition,
        runtime_action_kind=runtime_action_kind,
        diagnostic_codes=tuple(diagnostic_codes),
    )


def _minimal_rejected_provenance(
    memory_id: str,
    codes: Sequence[PolicyDiagnosticCode],
) -> PolicyProvenance:
    return PolicyProvenance(
        memory_id=memory_id,
        disposition=PolicyDisposition.REJECTED,
        diagnostic_codes=tuple(codes),
    )


def _effect_sort_key(
    item: Tuple[PolicyEffectKind, str],
) -> Tuple[int, int, str]:
    kind, value = item
    if kind is PolicyEffectKind.RISK_FLOOR:
        return (0, _risk_rank(RiskLevel(value)), value)
    order = {
        PolicyEffectKind.REQUIRE_CONTRACT: 1,
        PolicyEffectKind.REQUIRE_CHECK: 2,
        PolicyEffectKind.VERIFICATION_HINT: 3,
    }
    return (order[kind], 0, value)


def _action_for_effect(
    kind: PolicyEffectKind,
    value: str,
    memory_ids: Sequence[str],
) -> RuntimePolicyAction:
    ids = tuple(memory_ids)
    if kind is PolicyEffectKind.RISK_FLOOR:
        return RaiseRiskFloorAction(
            minimum_level=RiskLevel(value),
            memory_ids=ids,
        )
    if kind is PolicyEffectKind.REQUIRE_CONTRACT:
        return RequireContractAction(contract_id=value, memory_ids=ids)
    if kind is PolicyEffectKind.REQUIRE_CHECK:
        return RequireCheckAction(check_id=value, memory_ids=ids)
    if kind is PolicyEffectKind.VERIFICATION_HINT:
        return VerificationHintAction(command_template_id=value, memory_ids=ids)
    raise ValueError("unsupported policy effect kind")


def _bounded_input(records: Any) -> Optional[Tuple[Any, ...]]:
    if isinstance(records, (str, bytes)):
        raise ValueError("records must be an iterable of DurableMemoryRecord values")
    try:
        iterator = iter(records)
    except TypeError as error:
        raise ValueError(
            "records must be an iterable of DurableMemoryRecord values"
        ) from error
    items: List[Any] = []
    for item in iterator:
        if len(items) >= MAX_SNAPSHOT_RECORDS:
            return None
        items.append(item)
    return tuple(items)


def _compile_memory_policy(
    records: Iterable[DurableMemoryRecord],
    *,
    current_risk_floor: RiskLevel,
    registry: RuntimePolicyRegistry,
) -> PolicyCompilation:
    _require_risk_level(current_risk_floor, "current_risk_floor")
    if type(registry) is not RuntimePolicyRegistry:
        raise ValueError("registry must be a RuntimePolicyRegistry")

    bounded = _bounded_input(records)
    if bounded is None:
        return PolicyCompilation(
            initial_risk_floor=current_risk_floor,
            effective_risk_floor=current_risk_floor,
            actions=(),
            diagnostics=(_diagnostic(PolicyDiagnosticCode.INPUT_LIMIT_EXCEEDED),),
            provenance=(),
        )

    diagnostics: List[PolicyDiagnostic] = []
    provenance_by_memory: Dict[str, PolicyProvenance] = {}
    records_by_memory: Dict[str, List[DurableMemoryRecord]] = {}

    for item in bounded:
        if type(item) is not DurableMemoryRecord:
            diagnostics.append(_diagnostic(PolicyDiagnosticCode.INVALID_RECORD_TYPE))
            continue
        try:
            memory_id = validate_stable_id(
                item.memory_id,
                "MEM",
                "record.memory_id",
            )
        except (AttributeError, TypeError, ValueError):
            diagnostics.append(_diagnostic(PolicyDiagnosticCode.INVALID_MEMORY_ID))
            continue
        records_by_memory.setdefault(memory_id, []).append(item)

    canonical_records: Dict[str, DurableMemoryRecord] = {}
    for memory_id in sorted(records_by_memory):
        canonical_group: List[DurableMemoryRecord] = []
        rejection_codes: Set[PolicyDiagnosticCode] = set()
        for record in records_by_memory[memory_id]:
            canonical, code = _canonical_record(record)
            if code is not None:
                rejection_codes.add(code)
            elif canonical is not None:
                canonical_group.append(canonical)
        if rejection_codes:
            ordered_codes = tuple(sorted(rejection_codes, key=lambda item: item.value))
            diagnostics.extend(_diagnostic(code, memory_id) for code in ordered_codes)
            provenance_by_memory[memory_id] = _minimal_rejected_provenance(
                memory_id,
                ordered_codes,
            )
            continue
        signatures = {record.to_json() for record in canonical_group}
        if len(signatures) != 1:
            code = PolicyDiagnosticCode.CONFLICTING_MEMORY_RECORD
            diagnostics.append(_diagnostic(code, memory_id))
            provenance_by_memory[memory_id] = _minimal_rejected_provenance(
                memory_id,
                (code,),
            )
            continue
        canonical_records[memory_id] = canonical_group[0]

    accepted_effects: Dict[Tuple[PolicyEffectKind, str], Set[str]] = {}
    accepted_records: Dict[str, DurableMemoryRecord] = {}
    contract_ids = frozenset(registry.contract_ids)
    check_ids = frozenset(registry.check_ids)
    command_template_ids = frozenset(registry.command_template_ids)

    for memory_id in sorted(canonical_records):
        record = canonical_records[memory_id]
        if record.status is not RecordStatus.ACTIVE:
            code = PolicyDiagnosticCode.INACTIVE_RECORD
            diagnostics.append(_diagnostic(code, memory_id))
            provenance_by_memory[memory_id] = _full_provenance(
                record,
                PolicyDisposition.REJECTED,
                diagnostic_codes=(code,),
            )
            continue
        if record.sensitivity is Sensitivity.BLOCKED:
            code = PolicyDiagnosticCode.BLOCKED_SENSITIVITY
            diagnostics.append(_diagnostic(code, memory_id))
            provenance_by_memory[memory_id] = _full_provenance(
                record,
                PolicyDisposition.REJECTED,
                diagnostic_codes=(code,),
            )
            continue
        effect = record.policy_effect
        if effect is None:
            provenance_by_memory[memory_id] = _full_provenance(
                record,
                PolicyDisposition.INFORMATIONAL,
            )
            continue

        rejection_code: Optional[PolicyDiagnosticCode] = None
        if effect.effect_kind is PolicyEffectKind.RISK_FLOOR:
            requested_floor = RiskLevel(effect.value)
            if _risk_rank(requested_floor) <= _risk_rank(current_risk_floor):
                code = PolicyDiagnosticCode.RISK_FLOOR_NOT_RAISED
                diagnostics.append(_diagnostic(code, memory_id))
                provenance_by_memory[memory_id] = _full_provenance(
                    record,
                    PolicyDisposition.NOT_APPLIED,
                    diagnostic_codes=(code,),
                )
                continue
        elif effect.effect_kind is PolicyEffectKind.REQUIRE_CONTRACT:
            if effect.value not in contract_ids:
                rejection_code = PolicyDiagnosticCode.UNKNOWN_CONTRACT
        elif effect.effect_kind is PolicyEffectKind.REQUIRE_CHECK:
            if effect.value not in check_ids:
                rejection_code = PolicyDiagnosticCode.UNKNOWN_CHECK
        elif effect.effect_kind is PolicyEffectKind.VERIFICATION_HINT:
            if effect.value not in command_template_ids:
                rejection_code = PolicyDiagnosticCode.UNKNOWN_COMMAND_TEMPLATE
        else:
            rejection_code = PolicyDiagnosticCode.UNKNOWN_POLICY_EFFECT

        if rejection_code is not None:
            diagnostics.append(_diagnostic(rejection_code, memory_id))
            provenance_by_memory[memory_id] = _full_provenance(
                record,
                PolicyDisposition.REJECTED,
                diagnostic_codes=(rejection_code,),
            )
            continue

        key = (effect.effect_kind, effect.value)
        accepted_effects.setdefault(key, set()).add(memory_id)
        accepted_records[memory_id] = record

    actions: List[RuntimePolicyAction] = []
    effective_risk_floor = current_risk_floor
    for key in sorted(accepted_effects, key=_effect_sort_key):
        kind, value = key
        memory_ids = tuple(sorted(accepted_effects[key]))
        action = _action_for_effect(kind, value, memory_ids)
        actions.append(action)
        if type(action) is RaiseRiskFloorAction and _risk_rank(
            action.minimum_level
        ) > _risk_rank(effective_risk_floor):
            effective_risk_floor = action.minimum_level
        for memory_id in memory_ids:
            provenance_by_memory[memory_id] = _full_provenance(
                accepted_records[memory_id],
                PolicyDisposition.APPLIED,
                runtime_action_kind=action.kind,
            )

    return PolicyCompilation(
        initial_risk_floor=current_risk_floor,
        effective_risk_floor=effective_risk_floor,
        actions=tuple(actions),
        diagnostics=tuple(diagnostics),
        provenance=tuple(provenance_by_memory.values()),
    )


@dataclass(frozen=True)
class TypedPolicyCompiler:
    registry: RuntimePolicyRegistry

    def __post_init__(self) -> None:
        if type(self.registry) is not RuntimePolicyRegistry:
            raise ValueError("registry must be a RuntimePolicyRegistry")

    def compile(
        self,
        records: Iterable[DurableMemoryRecord],
        *,
        current_risk_floor: RiskLevel,
    ) -> PolicyCompilation:
        return _compile_memory_policy(
            records,
            current_risk_floor=current_risk_floor,
            registry=self.registry,
        )


def compile_memory_policy(
    records: Iterable[DurableMemoryRecord],
    *,
    current_risk_floor: RiskLevel,
    registry: RuntimePolicyRegistry,
) -> PolicyCompilation:
    """Compile records without performing I/O or executing verification hints."""

    return _compile_memory_policy(
        records,
        current_risk_floor=current_risk_floor,
        registry=registry,
    )


__all__ = [
    "MEMORY_POLICY_COMPILER_VERSION",
    "PolicyCompilation",
    "PolicyDiagnostic",
    "PolicyDiagnosticCode",
    "PolicyDiagnosticSeverity",
    "PolicyDisposition",
    "PolicyProvenance",
    "RaiseRiskFloorAction",
    "RequireCheckAction",
    "RequireContractAction",
    "RuntimeActionKind",
    "RuntimePolicyAction",
    "RuntimePolicyRegistry",
    "TypedPolicyCompiler",
    "VerificationHintAction",
    "compile_memory_policy",
]
