from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from review_agent.memory_curator import (
    CuratorAuthority,
    CuratorDecisionOutcome,
    CuratorWarningCode,
    ExistingFingerprint,
    FinalVerifiedContext,
    LocalCuratorRule,
    MemoryCuratorInput,
    MemoryCuratorParseError,
    ValidatedCuratorSource,
    build_memory_curator_envelope,
    parse_memory_curator_response,
    run_local_memory_curator,
    run_model_memory_curator,
)
from review_agent.memory_models import (
    CandidateStatus,
    GitCommitSourceRef,
    HumanDeclarationAuthority,
    HumanDeclarationOrigin,
    HumanDeclarationSourceRef,
    MemoryConfidence,
    MemoryKind,
    MemoryScope,
    PolicyEffect,
    PolicyEffectKind,
    ProducerType,
    Sensitivity,
    ValidityPolicy,
)
from review_agent.model_adapter import FakeToolCallingAdapter
from review_agent.model_adapter_factory import (
    ModelAdapterConfig,
    build_model_adapter_factory_from_config,
)
from review_agent.model_protocol import (
    ModelResponseKind,
    ModelTurnResponse,
)


REPOSITORY_KEY = "a" * 64
HEAD_SHA = "b" * 40
CREATED_AT = "2026-07-15T01:02:03Z"
REVIEW_ID = "review-curator-1"


class _Factory:
    def __init__(self, script):
        self.script = list(script)
        self.adapters: list[FakeToolCallingAdapter] = []

    def create(self):
        adapter = FakeToolCallingAdapter(list(self.script))
        self.adapters.append(adapter)
        return adapter


class _RaisingAdapter:
    provider_name = "raising"

    def complete_turn(self, request):
        raise RuntimeError("provider-specific text must not reach a warning")


class _RaisingFactory:
    def create(self):
        return _RaisingAdapter()


class _StepClock:
    def __init__(self, values: list[float]):
        self._values = iter(values)
        self._last = values[-1]

    def __call__(self) -> float:
        self._last = next(self._values, self._last)
        return self._last


def _validated_source(*, remote_sendable: bool = True) -> ValidatedCuratorSource:
    return ValidatedCuratorSource(
        source_ref=GitCommitSourceRef(commit_sha=HEAD_SHA),
        excerpt="The compatibility contract is explicit at this revision.",
        validation_report_hash="c" * 64,
        remote_sendable=remote_sendable,
    )


def _input(
    *,
    sources=(),
    rules=(),
    declarations=(),
    fingerprints=(),
    policy_effects=(),
    created_at: str = CREATED_AT,
    allowed_kinds=tuple(MemoryKind),
    context: FinalVerifiedContext | None = None,
) -> MemoryCuratorInput:
    return MemoryCuratorInput(
        repository_key=REPOSITORY_KEY,
        origin_review_id=REVIEW_ID,
        head_sha=HEAD_SHA,
        created_at=created_at,
        final_verified_context=(
            context
            if context is not None
            else FinalVerifiedContext(
                verified_findings=("Verified behavior is revision-bound.",),
                uncertainties=("One optional integration was not exercised.",),
                contract_coverage=("regression_safety:covered",),
                final_risk="medium",
            )
        ),
        validated_sources=tuple(sources),
        explicit_project_rules=tuple(rules),
        trusted_human_declarations=tuple(declarations),
        existing_fingerprints=tuple(fingerprints),
        policy_effect_catalog=tuple(policy_effects),
        allowed_kinds=tuple(allowed_kinds),
    )


def _rule(source: ValidatedCuratorSource) -> LocalCuratorRule:
    return LocalCuratorRule(
        rule_id="explicit-rule-1",
        authority=CuratorAuthority.EXPLICIT_PROJECT_RULE,
        kind=MemoryKind.REVIEW_RULE,
        statement="Preserve the public compatibility contract.",
        scope=MemoryScope(),
        source_ref_ids=(source.source_ref_id,),
        validity_policies=(ValidityPolicy.SOURCE_CONTENT_HASH,),
        confidence=MemoryConfidence.HIGH,
        sensitivity=Sensitivity.NORMAL,
    )


def _human_declaration() -> HumanDeclarationAuthority:
    declaration = "Treat changes to the public wire format as compatibility-sensitive."
    return HumanDeclarationAuthority(
        source_ref=HumanDeclarationSourceRef(
            request_id="REQ-" + "d" * 64,
            actor="maintainer@example.test",
            declaration_hash=hashlib.sha256(declaration.encode("utf-8")).hexdigest(),
            created_at=CREATED_AT,
            review_id=REVIEW_ID,
        ),
        origin=HumanDeclarationOrigin.USER_REQUEST,
        declaration=declaration,
    )


def _proposal_payload(
    source_id: str,
    *,
    candidate_id: str = "proposal-1",
    statement: str = "Preserve the public compatibility contract.",
    policy_effect_id: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "candidates": [
            {
                "candidate_id": candidate_id,
                "kind": "review_rule",
                "statement": statement,
                "scope": {
                    "schema_version": 1,
                    "paths": [],
                    "symbols": [],
                    "contracts": [],
                    "languages": [],
                },
                "source_ref_ids": [source_id],
                "validity_policies": ["source_content_hash"],
                "confidence": "high",
                "sensitivity": "normal",
                "policy_effect_id": policy_effect_id,
            }
        ],
    }


def _final_response(payload: dict[str, object] | str) -> ModelTurnResponse:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return ModelTurnResponse(
        kind=ModelResponseKind.FINAL,
        final_text=text,
        provider_name="test-provider",
        model="test-curator",
        raw={"request_id": "provider-request-1"},
    )


def test_local_curator_returns_empty_batch_without_explicit_authority() -> None:
    result = run_local_memory_curator(_input())

    assert result.batch.candidates == ()
    assert result.decision.outcome is CuratorDecisionOutcome.EMPTY
    assert result.decision.warning_codes == ()
    assert result.decision.review_conclusion_impact == "none"
    assert result.envelope is None
    assert result.raw_response is None


def test_local_curator_compiles_only_explicit_rules_as_proposed_candidates() -> None:
    source = _validated_source()
    result = run_local_memory_curator(_input(sources=(source,), rules=(_rule(source),)))

    assert len(result.batch.candidates) == 1
    candidate = result.batch.candidates[0]
    assert candidate.status is CandidateStatus.PROPOSED
    assert candidate.producer.producer_type is ProducerType.LOCAL
    assert candidate.source_refs == (source.source_ref,)
    assert result.decision.outcome is CuratorDecisionOutcome.PROPOSED
    assert result.decision.review_conclusion_impact == "none"
    assert not hasattr(result.decision, "actor")


def test_local_curator_accepts_runtime_trusted_human_declaration() -> None:
    declaration = _human_declaration()

    result = run_local_memory_curator(_input(declarations=(declaration,)))

    assert len(result.batch.candidates) == 1
    candidate = result.batch.candidates[0]
    assert candidate.statement == declaration.declaration
    assert candidate.source_refs == (declaration.source_ref,)
    assert candidate.validity_policies == (ValidityPolicy.MANUAL_UNTIL_REVOKED,)
    assert candidate.status is CandidateStatus.PROPOSED


def test_local_curator_rejects_rule_whose_source_is_not_in_runtime_allowlist() -> None:
    source = _validated_source()

    result = run_local_memory_curator(_input(rules=(_rule(source),)))

    assert result.batch.candidates == ()
    assert result.decision.outcome is CuratorDecisionOutcome.REJECTED
    assert result.decision.warning_codes == (
        CuratorWarningCode.UNAUTHORIZED_LOCAL_PROPOSAL,
    )


def test_model_envelope_is_minimal_revision_bound_and_contains_runtime_catalogs() -> None:
    source = _validated_source()
    fingerprint = ExistingFingerprint("e" * 64, "active")
    policy = PolicyEffect(PolicyEffectKind.REQUIRE_CONTRACT, "regression_safety")

    envelope = build_memory_curator_envelope(
        _input(
            sources=(source,),
            fingerprints=(fingerprint,),
            policy_effects=(policy,),
        )
    )
    payload = envelope.to_dict()

    assert set(payload) == {
        "schema",
        "request_digest",
        "invocation_id",
        "repository_key",
        "origin_review_id",
        "head_sha",
        "created_at",
        "final_verified_context",
        "source_ref_allowlist",
        "existing_fingerprint_catalog",
        "proposal_whitelist",
        "candidate_schema",
    }
    assert set(payload["final_verified_context"]) == {
        "verified_findings",
        "uncertainties",
        "contract_coverage",
        "final_risk",
    }
    assert payload["source_ref_allowlist"][0]["source_ref_id"] == source.source_ref_id
    assert payload["created_at"] == CREATED_AT
    assert payload["source_ref_allowlist"][0]["source_ref"] == source.source_ref.to_dict()
    assert payload["existing_fingerprint_catalog"] == [fingerprint.to_dict()]
    assert payload["proposal_whitelist"]["policy_effect_catalog"][0][
        "policy_effect"
    ] == policy.to_dict()
    serialized = envelope.to_json()
    for forbidden in ("api_key", "root_path", "review_conclusion", "tool_results"):
        assert forbidden not in serialized


def test_model_envelope_enforces_a_total_input_byte_budget() -> None:
    oversized = FinalVerifiedContext(
        verified_findings=tuple(
            f"finding-{index:03d}-" + "x" * 4084 for index in range(128)
        ),
        uncertainties=tuple(
            f"uncertainty-{index:03d}-" + "y" * 4080 for index in range(128)
        ),
        final_risk="high",
    )

    with pytest.raises(ValueError, match="total input byte budget"):
        build_memory_curator_envelope(_input(context=oversized))


def test_model_envelope_excludes_local_only_source_content() -> None:
    local_only = _validated_source(remote_sendable=False)

    envelope = build_memory_curator_envelope(_input(sources=(local_only,)))

    assert envelope.source_ref_allowlist == ()
    assert local_only.excerpt not in envelope.to_json()


def test_model_run_uses_no_tools_one_message_one_turn_and_runtime_allowlists() -> None:
    source = _validated_source()
    factory = _Factory([_final_response(_proposal_payload(source.source_ref_id))])

    result = run_model_memory_curator(factory, _input(sources=(source,)))

    request = factory.adapters[0].requests[0]
    assert request.tools == []
    assert request.tool_results == []
    assert len(request.messages) == 1
    assert request.parameters["tool_choice"] == "none"
    assert request.parameters["temperature"] == 0
    assert request.parameters["response_schema"] == "memory_curator_proposal_v1"
    assert request.parameters["request_digest"] == result.envelope.request_digest
    assert request.parameters["invocation_id"] == result.envelope.invocation_id
    assert len(result.batch.candidates) == 1
    assert result.batch.candidates[0].status is CandidateStatus.PROPOSED
    assert result.batch.candidates[0].producer.producer_type is ProducerType.MODEL


def test_registered_fake_model_curator_runs_end_to_end() -> None:
    source = _validated_source()
    factory = build_model_adapter_factory_from_config(
        ModelAdapterConfig(
            provider_name="fake",
            model=None,
            base_url=None,
            api_key_env="REVIEW_AGENT_API_KEY",
            stage_label="memory_curator",
        )
    )

    result = run_model_memory_curator(factory, _input(sources=(source,)))

    assert result.decision.outcome is CuratorDecisionOutcome.PROPOSED
    assert len(result.batch.candidates) == 1
    assert result.raw_response.attempts[0].model == "fake-memory-curator"


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda payload: payload.update({"unknown": True}), "unknown field"),
        (lambda payload: payload.pop("schema_version"), "missing field"),
        (
            lambda payload: payload["candidates"][0].update({"kind": "new_policy_kind"}),
            "kind",
        ),
        (
            lambda payload: payload["candidates"][0].update({"statement": "x" * 8193}),
            "statement",
        ),
        (
            lambda payload: payload["candidates"][0].update(
                {"source_ref_ids": ["SRC-" + "f" * 64]}
            ),
            "unauthorized source",
        ),
    ],
)
def test_strict_parser_rejects_unknown_missing_invalid_long_or_unauthorized_values(
    mutate,
    match: str,
) -> None:
    source = _validated_source()
    envelope = build_memory_curator_envelope(_input(sources=(source,)))
    payload = _proposal_payload(source.source_ref_id)
    mutate(payload)

    with pytest.raises(MemoryCuratorParseError, match=match):
        parse_memory_curator_response(json.dumps(payload), envelope)


def test_strict_parser_rejects_duplicate_json_keys() -> None:
    source = _validated_source()
    envelope = build_memory_curator_envelope(_input(sources=(source,)))
    text = '{"schema_version":1,"schema_version":1,"candidates":[]}'

    with pytest.raises(MemoryCuratorParseError, match="duplicate key"):
        parse_memory_curator_response(text, envelope)


def test_strict_parser_wraps_excessive_json_nesting() -> None:
    source = _validated_source()
    envelope = build_memory_curator_envelope(_input(sources=(source,)))
    deeply_nested = "[" * 2000 + "0" + "]" * 2000

    with pytest.raises(MemoryCuratorParseError, match="invalid JSON"):
        parse_memory_curator_response(deeply_nested, envelope)


def test_strict_parser_rejects_duplicate_wire_and_canonical_candidate_ids() -> None:
    source = _validated_source()
    envelope = build_memory_curator_envelope(_input(sources=(source,)))
    duplicate_wire = _proposal_payload(source.source_ref_id)
    duplicate_wire["candidates"].append(
        dict(duplicate_wire["candidates"][0])
    )

    with pytest.raises(MemoryCuratorParseError, match="duplicate candidate_id"):
        parse_memory_curator_response(json.dumps(duplicate_wire), envelope)

    duplicate_canonical = _proposal_payload(source.source_ref_id)
    second = dict(duplicate_canonical["candidates"][0])
    second["candidate_id"] = "proposal-2"
    duplicate_canonical["candidates"].append(second)
    with pytest.raises(MemoryCuratorParseError, match="same canonical candidate"):
        parse_memory_curator_response(json.dumps(duplicate_canonical), envelope)


@pytest.mark.parametrize(
    "forbidden",
    [
        {"status": "approved"},
        {"record_status": "active"},
        {"actor": "model"},
        {"decision": "approve"},
        {"policy_effect": {"type": "run_shell", "value": "rm -rf"}},
    ],
)
def test_model_cannot_return_status_actor_decision_or_arbitrary_policy(forbidden) -> None:
    source = _validated_source()
    envelope = build_memory_curator_envelope(_input(sources=(source,)))
    payload = _proposal_payload(source.source_ref_id)
    payload["candidates"][0].update(forbidden)

    with pytest.raises(MemoryCuratorParseError, match="unknown field"):
        parse_memory_curator_response(json.dumps(payload), envelope)


def test_model_policy_effect_must_reference_runtime_catalog() -> None:
    source = _validated_source()
    policy = PolicyEffect(PolicyEffectKind.REQUIRE_CHECK, "python-tests")
    allowed_input = _input(sources=(source,), policy_effects=(policy,))
    allowed_envelope = build_memory_curator_envelope(allowed_input)
    allowed_payload = _proposal_payload(
        source.source_ref_id,
        policy_effect_id=allowed_envelope.policy_effect_catalog[0].policy_effect_id,
    )

    drafts = parse_memory_curator_response(
        json.dumps(allowed_payload),
        allowed_envelope,
    )
    assert drafts[0].policy_effect == policy

    allowed_payload["candidates"][0]["policy_effect_id"] = "POL-" + "f" * 64
    with pytest.raises(MemoryCuratorParseError, match="unauthorized policy effect"):
        parse_memory_curator_response(json.dumps(allowed_payload), allowed_envelope)


@pytest.mark.parametrize(
    "factory,expected_code",
    [
        (_RaisingFactory(), CuratorWarningCode.PROVIDER_FAILURE),
        (
            _Factory([_final_response("not-json"), _final_response("still-not-json")]),
            CuratorWarningCode.PARSE_FAILURE,
        ),
    ],
)
def test_provider_and_parse_exhaustion_return_deterministic_empty_decision(
    factory,
    expected_code: CuratorWarningCode,
) -> None:
    source = _validated_source()

    first = run_model_memory_curator(
        factory,
        _input(sources=(source,)),
        max_provider_attempts=2,
    )
    second_factory = (
        _RaisingFactory()
        if isinstance(factory, _RaisingFactory)
        else _Factory([_final_response("not-json"), _final_response("still-not-json")])
    )
    second = run_model_memory_curator(
        second_factory,
        _input(sources=(source,)),
        max_provider_attempts=2,
    )

    assert first.batch.candidates == ()
    assert first.decision.outcome is CuratorDecisionOutcome.REJECTED
    assert expected_code in first.decision.warning_codes
    assert CuratorWarningCode.ATTEMPTS_EXHAUSTED in first.decision.warning_codes
    assert first.decision.warnings == second.decision.warnings
    assert first.decision.review_conclusion_impact == "none"


def test_elapsed_timeout_returns_warning_and_empty_decision() -> None:
    source = _validated_source()
    factory = _Factory([_final_response(_proposal_payload(source.source_ref_id))])

    result = run_model_memory_curator(
        factory,
        _input(sources=(source,)),
        max_elapsed_seconds=1,
        clock=_StepClock([0.0, 0.0, 2.0, 2.0]),
    )

    assert result.batch.candidates == ()
    assert result.decision.outcome is CuratorDecisionOutcome.REJECTED
    assert CuratorWarningCode.TIMEOUT in result.decision.warning_codes
    assert result.decision.review_conclusion_impact == "none"


def test_safe_secret_redaction_happens_before_parse_and_persistence() -> None:
    source = _validated_source()
    secret = "sk-ABCDEFGHIJKLMNOPQRSTUVWX"
    payload = _proposal_payload(
        source.source_ref_id,
        statement=f"Never persist provider token {secret}.",
    )
    factory = _Factory([_final_response(payload)])

    result = run_model_memory_curator(factory, _input(sources=(source,)))

    attempt = result.raw_response.attempts[0]
    assert attempt.retained_content is True
    assert attempt.redactions
    assert secret not in attempt.final_text
    assert secret not in json.dumps(attempt.raw_response)
    assert secret not in result.batch.candidates[0].statement
    assert "<redacted" in result.batch.candidates[0].statement


def test_unsafe_redaction_retains_only_hash_and_metadata_and_rejects_batch() -> None:
    source = _validated_source()
    secret = "sk-ABCDEFGHIJKLMNOPQRSTUVWX"
    unsafe = (
        '{"schema_version":1,"schema_version":1,"candidates":[],"api_key":"'
        + secret
        + '"}'
    )
    factory = _Factory([_final_response(unsafe)])

    result = run_model_memory_curator(factory, _input(sources=(source,)))

    attempt = result.raw_response.attempts[0]
    assert attempt.retained_content is False
    assert attempt.final_text is None
    assert attempt.raw_response is None
    assert len(attempt.response_hash) == 64
    assert secret not in json.dumps(result.raw_response.to_dict())
    assert result.batch.candidates == ()
    assert result.decision.outcome is CuratorDecisionOutcome.REJECTED
    assert result.decision.warning_codes == (CuratorWarningCode.UNSAFE_RESPONSE,)


def test_sanitized_raw_response_does_not_retain_hidden_reasoning() -> None:
    source = _validated_source()
    response = _final_response(_proposal_payload(source.source_ref_id))
    response = ModelTurnResponse(
        kind=response.kind,
        final_text=response.final_text,
        provider_name=response.provider_name,
        model=response.model,
        raw={
            "request_id": "provider-request-1",
            "reasoning_content": "private chain of thought",
        },
    )
    factory = _Factory([response])

    result = run_model_memory_curator(factory, _input(sources=(source,)))

    attempt = result.raw_response.attempts[0]
    assert attempt.raw_response["reasoning_content"] == "<redacted:hidden_reasoning>"
    assert "private chain of thought" not in json.dumps(attempt.to_dict())
    assert "hidden_reasoning" in attempt.redactions


def test_sanitized_final_text_does_not_retain_hidden_reasoning_fields() -> None:
    source = _validated_source()
    payload = _proposal_payload(source.source_ref_id)
    payload["reasoning_content"] = "private chain of thought"
    factory = _Factory([_final_response(json.dumps(payload))])

    result = run_model_memory_curator(factory, _input(sources=(source,)))

    attempt = result.raw_response.attempts[0]
    assert "private chain of thought" not in (attempt.final_text or "")
    assert "<redacted:hidden_reasoning>" in (attempt.final_text or "")
    assert "hidden_reasoning" in attempt.redactions
    assert result.batch.candidates == ()
    assert result.decision.outcome is CuratorDecisionOutcome.REJECTED


def test_request_digest_invocation_and_retry_envelope_are_stable() -> None:
    source = _validated_source()
    curator_input = _input(sources=(source,))
    first_envelope = build_memory_curator_envelope(curator_input)
    second_envelope = build_memory_curator_envelope(curator_input)
    factory = _Factory(
        [
            _final_response("not-json"),
            _final_response(_proposal_payload(source.source_ref_id)),
        ]
    )

    result = run_model_memory_curator(
        factory,
        curator_input,
        max_provider_attempts=2,
    )
    requests = factory.adapters[0].requests

    assert first_envelope.request_digest == second_envelope.request_digest
    assert first_envelope.invocation_id == second_envelope.invocation_id
    assert len(requests) == 2
    assert requests[0].messages == requests[1].messages
    assert requests[0].system == requests[1].system
    assert requests[0].parameters["request_digest"] == requests[1].parameters[
        "request_digest"
    ]
    assert requests[0].parameters["invocation_id"] == requests[1].parameters[
        "invocation_id"
    ]
    assert result.envelope.request_digest == first_envelope.request_digest
    assert result.decision.attempt_count == 2


def test_invocation_identity_covers_every_output_affecting_input() -> None:
    source = _validated_source()
    rule = _rule(source)
    baseline_input = _input(sources=(source,), rules=(rule,))
    later_input = _input(
        sources=(source,),
        rules=(rule,),
        created_at="2026-07-15T01:02:04Z",
    )
    restricted_input = _input(
        sources=(source,),
        rules=(rule,),
        allowed_kinds=(MemoryKind.BUSINESS_INVARIANT,),
    )
    effect_input = _input(
        sources=(source,),
        rules=(rule,),
        policy_effects=(
            PolicyEffect(PolicyEffectKind.REQUIRE_CHECK, "python-tests"),
        ),
    )

    baseline_local = run_local_memory_curator(baseline_input)
    later_local = run_local_memory_curator(later_input)
    restricted_local = run_local_memory_curator(restricted_input)
    effect_local = run_local_memory_curator(effect_input)
    baseline_model = build_memory_curator_envelope(baseline_input)
    later_model = build_memory_curator_envelope(later_input)

    assert baseline_local.decision.request_digest != later_local.decision.request_digest
    assert baseline_local.decision.invocation_id != later_local.decision.invocation_id
    assert baseline_local.decision.request_digest != restricted_local.decision.request_digest
    assert baseline_local.decision.request_digest != effect_local.decision.request_digest
    assert baseline_model.request_digest != later_model.request_digest
    assert baseline_model.invocation_id != later_model.invocation_id


def test_existing_fingerprint_catalog_defers_source_aware_dedupe_to_lifecycle() -> None:
    source = _validated_source()
    first = run_local_memory_curator(_input(sources=(source,), rules=(_rule(source),)))
    fingerprint = ExistingFingerprint(
        first.batch.candidates[0].content_fingerprint,
        "pending_approval",
    )

    duplicate = run_local_memory_curator(
        _input(
            sources=(source,),
            rules=(_rule(source),),
            fingerprints=(fingerprint,),
        )
    )

    assert len(duplicate.batch.candidates) == 1
    assert duplicate.decision.outcome is CuratorDecisionOutcome.PROPOSED
    assert duplicate.decision.duplicate_fingerprints == (
        fingerprint.content_fingerprint,
    )


def test_curator_has_no_provider_store_or_approval_dependency() -> None:
    module = Path("src/review_agent/memory_curator.py")
    source = module.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(module))
    review_agent_imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            review_agent_imports.update(
                alias.name
                for alias in node.names
                if alias.name.startswith("review_agent.")
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("review_agent."):
                review_agent_imports.add(node.module)

    assert review_agent_imports <= {
        "review_agent.memory_models",
        "review_agent.memory_sources",
        "review_agent.model_adapter_factory",
        "review_agent.model_protocol",
    }
    assert "memory_store" not in source
    assert "approve" not in source.casefold()
    assert "review_agent.provider" not in source
    assert "from review_agent.model_adapter import" not in source
