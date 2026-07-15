import json

import pytest

from review_agent.model_adapter import FakeToolCallingAdapter
from review_agent.model_protocol import ModelResponseKind, ModelTurnResponse
from review_agent.memory_models import PolicyEffectKind
from review_agent.memory_policy import (
    PolicyCompilation,
    PolicyDisposition,
    PolicyProvenance,
    RequireContractAction,
    RuntimeActionKind,
    RuntimePolicyRegistry,
)
from review_agent.models import (
    CompiledMemoryRequirement,
    MemoryDiagnosticCode,
    MemoryReference,
    PlannerMemoryProjection,
    PlannerPerspectiveHint,
    RiskAssessment,
    RiskLevel,
    ReviewProfile,
    VerificationTemplateHint,
)
from review_agent.portfolio import (
    CORE_REVIEW_CONTRACT,
    PortfolioProposalParseError,
    build_portfolio_packet,
    build_planner_memory_projection,
    deterministic_fallback_proposal,
    parse_portfolio_proposal,
    portfolio_packet_to_model_input,
    portfolio_size_bounds,
    run_portfolio_planner,
)
from review_agent.runtime import compile_portfolio


def risk(level=RiskLevel.HIGH):
    return RiskAssessment(
        level=level,
        dimensions={
            "impact": "public behavior may change",
            "blast_radius": "several callers may be affected",
            "reversibility": "rollback is available",
            "uncertainty": "one domain invariant is unknown",
            "verification_strength": "targeted tests are present",
        },
        reasons=["public behavior changed"],
        signal_refs=["signal:public-api"],
        uncertainties=["domain invariant is not documented"],
        suggested_focus=["caller compatibility"],
    )


def packet(level=RiskLevel.HIGH):
    return build_portfolio_packet(
        risk(level),
        change_map={"changed_files": ["src/app.py"]},
        changed_symbols=[{"path": "src/app.py", "symbol": "run"}],
        intent_summary={"goal": "preserve public behavior"},
        ref_catalog={"context:run": "changed run symbol"},
    )


def candidate(
    candidate_id,
    role_kind,
    perspective_key,
    *,
    priority=50,
    reason_refs=None,
    context_refs=None,
    extra_contract=None,
):
    return {
        "candidate_id": candidate_id,
        "role_kind": role_kind,
        "role_name": f"{candidate_id} Reviewer",
        "perspective_key": perspective_key,
        "mission": f"Investigate {perspective_key}",
        "reason_refs": list(reason_refs or ["signal:public-api"]),
        "context_refs": list(context_refs or ["context:run"]),
        "extra_contract": list(extra_contract or []),
        "required_checks": [f"check {perspective_key}"],
        "priority": priority,
    }


def proposal_payload(*candidates):
    return {
        "candidates": list(candidates),
        "summary": "Target the highest-risk behavior",
        "uncertainties": ["one planner uncertainty"],
    }


def parse(payload, target=None):
    return parse_portfolio_proposal(
        json.dumps(payload),
        target or packet(),
    )


def test_portfolio_packet_exposes_only_runtime_owned_policy():
    target = packet(RiskLevel.HIGH)
    payload = target.to_dict()

    assert payload["reviewer_count_bounds"] == {"minimum": 3, "maximum": 4}
    assert payload["budget_policy"]["per_reviewer"] == {
        "max_turns": 16,
        "max_tool_calls": 40,
        "max_output_tokens": 8192,
        "max_total_tokens": 131072,
        "max_elapsed_seconds": 600.0,
        "max_provider_attempts": 3,
    }
    assert "permissions" not in payload
    assert set(payload["ref_allowlist"]) == {"signal:public-api", "context:run"}

    indented_catalog = build_portfolio_packet(
        risk(RiskLevel.LOW),
        ref_catalog={"diff:line": "  def changed():"},
    )
    assert indented_catalog.ref_catalog == {"diff:line": "def changed():"}


def test_strict_parser_accepts_only_authorized_candidate_fields():
    target = packet()
    parsed = parse(
        proposal_payload(
            candidate(
                "async",
                "specialist",
                "async-lifecycle",
                extra_contract=["unresolved_uncertainties"],
            )
        ),
        target,
    )

    assert parsed.candidates[0].perspective_key == "async-lifecycle"
    assert parsed.candidates[0].reason_refs == ["signal:public-api"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update({"budget": 999}), "unknown fields"),
        (
            lambda value: value["candidates"][0].update({"command": "pytest"}),
            "unknown fields",
        ),
        (
            lambda value: value["candidates"][0].update(
                {"reason_refs": ["unknown:signal"]}
            ),
            "unknown refs",
        ),
        (
            lambda value: value["candidates"][0].update(
                {"extra_contract": ["caller_compatibility"]}
            ),
            "unknown Contract",
        ),
        (
            lambda value: value["candidates"][0].update({"priority": True}),
            "integer from 0 through 100",
        ),
        (
            lambda value: value["candidates"][0].update({"mission": " "}),
            "non-empty string",
        ),
    ],
)
def test_strict_parser_rejects_schema_drift_and_unauthorized_values(mutate, message):
    payload = proposal_payload(candidate("core", "core", "core"))
    mutate(payload)

    with pytest.raises(PortfolioProposalParseError, match=message):
        parse(payload)


def test_strict_parser_rejects_duplicate_ids_and_excess_candidates():
    duplicate = proposal_payload(
        candidate("same", "core", "core"),
        candidate("same", "adversarial", "adversarial"),
    )
    with pytest.raises(PortfolioProposalParseError, match="duplicate candidate_id"):
        parse(duplicate)

    too_many = proposal_payload(
        *(
            candidate(f"specialist-{index}", "specialist", f"focus-{index}")
            for index in range(5)
        )
    )
    with pytest.raises(PortfolioProposalParseError, match="maximum of 4"):
        parse(too_many)


def test_strict_parser_rejects_duplicate_json_fields():
    content = (
        '{"candidates":[],"summary":"first","summary":"second",'
        '"uncertainties":[]}'
    )

    with pytest.raises(PortfolioProposalParseError, match="duplicate field: summary"):
        parse_portfolio_proposal(content, packet())


@pytest.mark.parametrize(
    ("level", "expected_bounds", "expected_roles"),
    [
        (RiskLevel.LOW, (1, 1), ["core"]),
        (RiskLevel.MEDIUM, (2, 2), ["core", "adversarial"]),
        (RiskLevel.HIGH, (3, 4), ["core", "adversarial", "specialist"]),
        (
            RiskLevel.CRITICAL,
            (4, 6),
            ["core", "adversarial", "specialist", "specialist"],
        ),
    ],
)
def test_deterministic_compiler_enforces_count_roles_permissions_and_budget(
    level,
    expected_bounds,
    expected_roles,
):
    target = packet(level)
    first = compile_portfolio(target)
    second = compile_portfolio(target)
    profile = ReviewProfile.for_risk(level)

    assert portfolio_size_bounds(level) == expected_bounds
    assert [item.role_kind for item in first.assignments] == expected_roles
    assert [item.assignment_id for item in first.assignments] == [
        item.assignment_id for item in second.assignments
    ]
    assert len({item.assignment_id for item in first.assignments}) == len(first.assignments)
    for assignment in first.assignments:
        assert assignment.repository_permission == "read_only"
        assert assignment.command_permission == "safe_checks_only"
        assert assignment.max_turns == profile.max_turns_per_reviewer
        assert assignment.max_tool_calls == profile.max_tool_calls_per_reviewer
        assert assignment.max_total_tokens == profile.max_total_tokens
        assert assignment.max_elapsed_seconds == profile.max_elapsed_seconds
        assert assignment.planner_source == "local"
        assert assignment.initial_context.observation_refs == []
        assert "signal:public-api" in assignment.initial_context.signal_refs


def test_core_contract_is_exact_full_five_and_callers_remain_a_required_check():
    core = compile_portfolio(packet(RiskLevel.LOW)).assignments[0]

    assert CORE_REVIEW_CONTRACT == (
        "intent_alignment",
        "behavioral_correctness",
        "regression_safety",
        "test_adequacy",
        "unresolved_uncertainties",
    )
    assert core.assigned_contract == list(CORE_REVIEW_CONTRACT)
    assert "caller_compatibility" not in core.assigned_contract
    assert any("caller" in check for check in core.required_checks)

    model_core = parse(
        proposal_payload(candidate("model-core", "core", "core")),
        packet(RiskLevel.LOW),
    )
    compiled_model_core = compile_portfolio(
        packet(RiskLevel.LOW),
        model_core,
    ).assignments[0]
    assert any("caller" in check for check in compiled_model_core.required_checks)


def test_compiler_injects_required_roles_deduplicates_perspectives_and_caps_slots():
    target = packet(RiskLevel.HIGH)
    model_proposal = parse(
        proposal_payload(
            candidate("async-high", "specialist", "async", priority=100),
            candidate("async-low", "specialist", "ASYNC", priority=10),
            candidate("domain", "specialist", "domain", priority=90),
            candidate("storage", "specialist", "storage", priority=80),
        ),
        target,
    )

    plan = compile_portfolio(target, model_proposal)

    assert len(plan.assignments) == 4
    assert [item.role_kind for item in plan.assignments[:2]] == ["core", "adversarial"]
    assert [item.planner_source for item in plan.assignments[:2]] == [
        "runtime_injected",
        "runtime_injected",
    ]
    assert {item.perspective_key for item in plan.assignments[2:]} == {"async", "domain"}
    assert "async-low" in plan.rejected_candidate_ids
    assert "storage" in plan.rejected_candidate_ids
    assert any(action.startswith("deduplicated_perspective:async-low") for action in plan.policy_actions)
    assert any(action == "rejected_maximum_slots:storage" for action in plan.policy_actions)


def test_compiler_separates_observation_ids_from_planning_signal_refs():
    observation_id = "O-0123456789abcdef0123456789abcdef"
    target = build_portfolio_packet(
        risk(RiskLevel.LOW),
        ref_allowlist=[observation_id, "context:run"],
    )
    model_proposal = parse_portfolio_proposal(
        json.dumps(
            proposal_payload(
                candidate(
                    "core",
                    "core",
                    "core",
                    context_refs=[observation_id, "context:run"],
                )
            )
        ),
        target,
    )

    initial_context = compile_portfolio(target, model_proposal).assignments[0].initial_context

    assert initial_context.observation_refs == [observation_id]
    assert initial_context.signal_refs == ["signal:public-api", "context:run"]


def test_planner_uses_single_turn_requests_and_retries_only_within_bounds():
    target = packet(RiskLevel.LOW)
    valid = proposal_payload(candidate("core", "core", "core"))
    adapter = FakeToolCallingAdapter(
        [
            ModelTurnResponse(kind=ModelResponseKind.FINAL, final_text="not-json"),
            ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                final_text=json.dumps(valid),
                raw={"id": "response-2"},
            ),
        ]
    )

    run = run_portfolio_planner(
        adapter,
        target,
        invocation_id="review-1-portfolio-deadbeef",
        model="fake-planner",
        max_provider_attempts=2,
    )

    assert run.status == "accepted"
    assert [attempt.status for attempt in run.attempts] == ["parse_error", "accepted"]
    assert len(adapter.requests) == 2
    assert all(request.tools == [] and request.tool_results == [] for request in adapter.requests)
    assert [request.parameters["attempt_index"] for request in adapter.requests] == [1, 2]
    assert {
        request.parameters["invocation_id"] for request in adapter.requests
    } == {"review-1-portfolio-deadbeef"}


def test_model_failure_uses_the_same_compiler_deterministic_fallback():
    target = packet(RiskLevel.MEDIUM)
    adapter = FakeToolCallingAdapter(
        [ModelTurnResponse(kind=ModelResponseKind.INVALID, error="provider unavailable")]
    )
    run = run_portfolio_planner(
        adapter,
        target,
        invocation_id="review-1-portfolio-fallback",
        max_provider_attempts=1,
    )

    fallback_plan = compile_portfolio(target, planner_run=run)
    local_plan = compile_portfolio(target)

    assert run.status == "fallback"
    assert fallback_plan.planner_status == "fallback"
    assert fallback_plan.fallback_reason == "provider unavailable"
    assert [item.assignment_id for item in fallback_plan.assignments] == [
        item.assignment_id for item in local_plan.assignments
    ]
    assert all(item.planner_source == "local" for item in fallback_plan.assignments)
    assert any("planner_fallback" in action for action in fallback_plan.policy_actions)


def test_malformed_adapter_response_is_bounded_and_falls_back():
    class MalformedAdapter:
        provider_name = "malformed"

        def __init__(self):
            self.calls = 0

        def complete_turn(self, request):
            self.calls += 1
            return None

    adapter = MalformedAdapter()

    run = run_portfolio_planner(
        adapter,
        packet(RiskLevel.LOW),
        invocation_id="review-1-malformed",
        max_provider_attempts=2,
    )

    assert run.status == "fallback"
    assert adapter.calls == 2
    assert [attempt.status for attempt in run.attempts] == [
        "invalid_response",
        "invalid_response",
    ]


def test_typed_memory_planner_projection_only_adds_registered_requirements_and_hints() -> None:
    memory_id = "MEM-" + "7" * 64
    projection = PlannerMemoryProjection(
        required_contracts=(
            CompiledMemoryRequirement(
                requirement_id="api_compatibility",
                memory_ids=(memory_id,),
            ),
        ),
        required_checks=(
            CompiledMemoryRequirement(
                requirement_id="schema_check",
                memory_ids=(memory_id,),
            ),
        ),
        verification_hints=(
            VerificationTemplateHint(
                command_template_id="python_schema_check",
                memory_ids=(memory_id,),
            ),
        ),
        perspective_hints=(
            PlannerPerspectiveHint(
                perspective_id="domain-invariants",
                source_feedback_ids=("FB-" + "8" * 64,),
            ),
        ),
        selected_memory=(
            MemoryReference(
                memory_id=memory_id,
                kind="review_rule",
                source_refs=("memory-source:" + "9" * 64,),
            ),
        ),
    )

    target = build_portfolio_packet(
        risk(RiskLevel.LOW),
        contract_allowlist=[*CORE_REVIEW_CONTRACT, "api_compatibility"],
        check_allowlist=["schema_check"],
        command_template_allowlist=["python_schema_check"],
        perspective_allowlist=["domain-invariants"],
        memory_projection=projection,
    )
    payload = target.to_dict()
    fallback = deterministic_fallback_proposal(target)

    assert target.reviewer_count_bounds == {"minimum": 1, "maximum": 1}
    assert target.budget_policy == {
        "risk_level": "low",
        "reviewer_count": {"minimum": 1, "maximum": 1},
        "per_reviewer": {
            "max_turns": 6,
            "max_tool_calls": 12,
            "max_output_tokens": 4096,
            "max_total_tokens": 32768,
            "max_elapsed_seconds": 120.0,
            "max_provider_attempts": 2,
        },
    }
    assert payload["memory_policy"]["required_contracts"][0]["requirement_id"] == (
        "api_compatibility"
    )
    assert payload["memory_policy"]["verification_hints"] == [
        {
            "command_template_id": "python_schema_check",
            "memory_ids": [memory_id],
        }
    ]
    assert "command" not in payload["memory_policy"]["verification_hints"][0]
    assert "api_compatibility" in fallback.candidates[0].extra_contract
    assert "schema_check" in fallback.candidates[0].required_checks

    missing_requirements = proposal_payload(
        candidate(
            "core",
            "core",
            "core",
            context_refs=["signal:public-api"],
        )
    )
    with pytest.raises(PortfolioProposalParseError, match="memory-required"):
        parse_portfolio_proposal(json.dumps(missing_requirements), target)


def test_memory_projection_cannot_expand_runtime_registries() -> None:
    memory_id = "MEM-" + "a" * 64
    reference = MemoryReference(
        memory_id=memory_id,
        kind="review_rule",
        source_refs=("memory-source:" + "b" * 64,),
    )
    projection = PlannerMemoryProjection(
        required_contracts=(
            CompiledMemoryRequirement("api_compatibility", (memory_id,)),
        ),
        required_checks=(
            CompiledMemoryRequirement("schema_check", (memory_id,)),
        ),
        verification_hints=(
            VerificationTemplateHint("python_schema_check", (memory_id,)),
        ),
        perspective_hints=(
            PlannerPerspectiveHint(
                "custom-domain",
                ("FB-" + "c" * 64,),
            ),
        ),
        selected_memory=(reference,),
    )

    with pytest.raises(ValueError, match="contract_allowlist"):
        build_portfolio_packet(risk(RiskLevel.LOW), memory_projection=projection)
    with pytest.raises(ValueError, match="check_allowlist"):
        build_portfolio_packet(
            risk(RiskLevel.LOW),
            contract_allowlist=[*CORE_REVIEW_CONTRACT, "api_compatibility"],
            memory_projection=projection,
        )
    with pytest.raises(ValueError, match="command_template_allowlist"):
        build_portfolio_packet(
            risk(RiskLevel.LOW),
            contract_allowlist=[*CORE_REVIEW_CONTRACT, "api_compatibility"],
            check_allowlist=["schema_check"],
            memory_projection=projection,
        )
    with pytest.raises(ValueError, match="perspective_allowlist"):
        build_portfolio_packet(
            risk(RiskLevel.LOW),
            contract_allowlist=[*CORE_REVIEW_CONTRACT, "api_compatibility"],
            check_allowlist=["schema_check"],
            command_template_allowlist=["python_schema_check"],
            memory_projection=projection,
        )


def test_planner_model_payload_transitively_removes_local_only_memory() -> None:
    memory_id = "MEM-" + "d" * 64
    statement = "secret local incident detail"
    projection = PlannerMemoryProjection(
        required_contracts=(
            CompiledMemoryRequirement("api_compatibility", (memory_id,)),
        ),
        selected_memory=(
            MemoryReference(
                memory_id=memory_id,
                kind="review_rule",
                source_refs=("memory-source:" + "e" * 64,),
                local_only=True,
            ),
        ),
    )
    assessment = risk(RiskLevel.HIGH)
    assessment = RiskAssessment(
        level=assessment.level,
        dimensions=assessment.dimensions,
        reasons=[f"approved memory risk signal: {statement}"],
        signal_refs=[f"memory:{memory_id}", f"memory_floor:{memory_id}"],
        uncertainties=[],
        suggested_focus=["approved incident lessons"],
    )
    target = build_portfolio_packet(
        assessment,
        contract_allowlist=[*CORE_REVIEW_CONTRACT, "api_compatibility"],
        ref_catalog={f"memory:{memory_id}": statement},
        memory_projection=projection,
    )

    encoded = json.dumps(portfolio_packet_to_model_input(target))

    assert "api_compatibility" in encoded
    assert memory_id not in encoded
    assert statement not in encoded
    assert "local_only" not in encoded
    assert "memory_floor" not in encoded


def test_planner_projection_revalidates_registry_and_blocks_its_own_overflow() -> None:
    memory_id = "MEM-" + "2" * 64
    compilation = PolicyCompilation(
        initial_risk_floor=RiskLevel.LOW,
        effective_risk_floor=RiskLevel.LOW,
        actions=(RequireContractAction("api_compatibility", (memory_id,)),),
        diagnostics=(),
        provenance=(
            PolicyProvenance(
                memory_id=memory_id,
                disposition=PolicyDisposition.APPLIED,
                effect_kind=PolicyEffectKind.REQUIRE_CONTRACT,
                effect_value="api_compatibility",
                runtime_action_kind=RuntimeActionKind.REQUIRE_CONTRACT,
            ),
        ),
    )
    reference = MemoryReference(
        memory_id=memory_id,
        kind="review_rule",
        source_refs=("memory-source:" + "3" * 64,),
    )

    with pytest.raises(ValueError, match="Runtime registry"):
        build_planner_memory_projection(
            compilation,
            registry=RuntimePolicyRegistry(contract_ids=("other",)),
            selected_memory=(reference,),
        )

    projection = build_planner_memory_projection(
        compilation,
        registry=RuntimePolicyRegistry(contract_ids=("api_compatibility",)),
        selected_memory=(reference,),
        max_hard_policy_items=0,
    )

    assert projection.required_contracts[0].requirement_id == "api_compatibility"
    assert any(
        item.code is MemoryDiagnosticCode.HARD_POLICY_OVERFLOW
        and item.blocking
        for item in projection.diagnostics
    )
