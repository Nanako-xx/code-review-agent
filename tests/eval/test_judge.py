import json
import pytest

from review_agent.model_adapter import ModelAdapterCapabilities
from review_agent.model_protocol import ModelResponseKind, ModelToolCall, ModelTurnResponse
from review_agent_eval.config import (
    EvaluatorExecutionConfig,
    EvaluatorRunConfig,
    JudgeExecutionBudgets,
    JudgeKind,
    JudgeProfileSnapshot,
)
from review_agent_eval.intent_evaluator import (
    IntentDimension,
    IntentJudgeRelation,
    IntentSemanticJudgeRequest,
    IntentTruthKind,
)
from review_agent_eval.judge import (
    DEFAULT_JUDGE_RUBRICS,
    JUDGE_CONTEXT_BUILDER_VERSION,
    JUDGE_PARSER_VERSION,
    JUDGE_SYSTEM_PROMPT_VERSION,
    JudgeAttemptStatus,
    JudgeContextKind,
    JudgeExecutionSource,
    JudgeExecutionResult,
    JudgeFailureCode,
    JudgeInputArtifact,
    JudgeOutputArtifact,
    JudgeProtocolError,
    JudgeRubric,
    JudgeRubricCatalog,
    JudgeRunStatus,
    JudgeTask,
    JudgeUngradedReason,
    InMemoryJudgeResultCache,
    SemanticJudge,
    build_intent_judge_input,
    build_novel_factuality_judge_input,
    intent_resolution_from_judge_result,
    repository_context,
)
from review_agent_eval.models import (
    DiffSide,
    FindingSeverity,
    SubmissionFinding,
    canonical_sha256,
)


def _request():
    return build_intent_judge_input(
        IntentSemanticJudgeRequest(
            request_id="intent-request-1",
            generated_id="generated-1",
            truth_id="truth-1",
            dimension=IntentDimension.GOAL,
            generated_text="Preserve existing callers.",
            truth_text="Keep all existing callers working.",
            truth_kind=IntentTruthKind.EXPECTED,
        )
    )


def _profiles(
    provider="scripted-provider",
    model="scripted-model",
    rubrics=DEFAULT_JUDGE_RUBRICS,
):
    profiles = []
    for task in JudgeTask:
        rubric = rubrics.for_task(task)
        # Build the exact task input so the persisted prompt digest is the
        # digest of the prompt actually sent to the model.
        if task is JudgeTask.INTENT_EQUIVALENCE:
            source = IntentSemanticJudgeRequest(
                request_id="intent-request-1",
                generated_id="generated-1",
                truth_id="truth-1",
                dimension=IntentDimension.GOAL,
                generated_text="Preserve existing callers.",
                truth_text="Keep all existing callers working.",
                truth_kind=IntentTruthKind.EXPECTED,
            )
            request = build_intent_judge_input(source, rubrics=rubrics)
        else:
            from tests.eval.test_judge_rubrics import _blind_request

            base_request = _blind_request(task)
            request = base_request
            if rubrics is not DEFAULT_JUDGE_RUBRICS:
                from review_agent_eval.judge import BlindJudgeInput

                request = BlindJudgeInput.create(
                    source_request_id=base_request.source_request_id,
                    source_request_digest=base_request.source_request_digest,
                    task=task,
                    rubric=rubric,
                    items=base_request.items,
                    contexts=base_request.contexts,
                    reference_bindings=base_request.reference_bindings,
                )
        profiles.append(
            JudgeProfileSnapshot(
                schema_version="eval_judge_profile_v1",
                kind=JudgeKind(task.value),
                judge_id=f"{task.value}-judge",
                judge_version="judge-v1",
                adapter_id="scripted-adapter",
                adapter_version="adapter-v1",
                adapter_config_digest=canonical_sha256(
                    {"provider": provider, "model": model}
                ),
                provider=provider,
                model=model,
                model_artifact_digest=None,
                parameters={"temperature": 0},
                system_prompt_version=JUDGE_SYSTEM_PROMPT_VERSION,
                system_prompt_digest=canonical_sha256(request.system_prompt),
                rubric_id=rubric.rubric_id,
                rubric_version=rubric.rubric_version,
                rubric_digest=rubric.rubric_digest,
                response_schema_version=rubric.response_schema,
                response_schema_digest=canonical_sha256(rubric.response_schema),
                context_builder_version=JUDGE_CONTEXT_BUILDER_VERSION,
                parser_version=JUDGE_PARSER_VERSION,
            )
        )
    return tuple(profiles)


def _execution(
    *,
    provider="scripted-provider",
    model="scripted-model",
    budgets=None,
    rubrics=DEFAULT_JUDGE_RUBRICS,
    artifact_file_bytes=8 * 1024 * 1024,
    artifact_total_bytes=32 * 1024 * 1024,
):
    evaluator = EvaluatorRunConfig(
        evaluator_id="evaluator-1",
        evaluator_version="evaluator-v1",
        grader_version="grader-v1",
        judge_profiles=_profiles(provider, model, rubrics),
    )
    if budgets is None:
        budgets = JudgeExecutionBudgets.defaults(
            evaluator_timeout_seconds=300,
            max_execution_artifact_file_bytes=artifact_file_bytes,
            max_execution_artifact_total_bytes=artifact_total_bytes,
        )
    return EvaluatorExecutionConfig.create(
        evaluator=evaluator,
        evaluator_timeout_seconds=300,
        max_execution_artifact_file_bytes=artifact_file_bytes,
        max_execution_artifact_total_bytes=artifact_total_bytes,
        judge_budgets=budgets,
    )


def _valid_output(request, *, relation="equivalent", score=900_000):
    return json.dumps(
        {
            "schema_version": request.rubric.response_schema,
            "request_id": request.request_id,
            "relation": relation,
            "score_ppm": score,
            "reason_refs": ["item-a"],
        }
    )


class _Factory:
    def __init__(self, script, *, provider="scripted-provider", model="scripted-model", capabilities=None):
        self.script = list(script)
        self.provider = provider
        self.model = model
        self.capabilities = capabilities or ModelAdapterCapabilities(
            supports_tool_choice_none=True,
            enforces_request_timeout=True,
            max_response_bytes=1 * 1024 * 1024,
        )
        self.created = 0
        self.adapters = []

    def create(self):
        self.created += 1
        factory = self

        class Adapter:
            provider_name = factory.provider
            capabilities = factory.capabilities

            def complete_turn(self, request):
                factory.adapters.append(request)
                item = factory.script.pop(0)
                if isinstance(item, BaseException):
                    raise item
                if callable(item):
                    item = item(request)
                return ModelTurnResponse(
                    kind=item.kind,
                    tool_calls=item.tool_calls,
                    final_text=item.final_text,
                    error=item.error,
                    raw=item.raw,
                    provider_name=factory.provider,
                    model=factory.model,
                )

        return Adapter()


def test_semantic_judge_uses_blind_turn_and_returns_typed_decision():
    request = _request()
    factory = _Factory(
        [
            ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                final_text=_valid_output(request),
            )
        ],
    )

    result = SemanticJudge(
        adapter_factory=factory,
        evaluator_execution=_execution(),
    ).execute(request)

    assert result.status is JudgeRunStatus.GRADED
    assert result.source is JudgeExecutionSource.LIVE
    assert result.decision.relation is IntentJudgeRelation.EQUIVALENT
    assert result.attempts[-1].status is JudgeAttemptStatus.ACCEPTED
    decision, failure, ungraded = intent_resolution_from_judge_result(result)
    assert decision == result.decision
    assert failure is None
    assert ungraded is None
    sent = factory.adapters[0]
    assert sent.tools == []
    assert sent.tool_results == []
    assert sent.parameters["tool_choice"] == "none"
    assert "scripted-provider" not in sent.messages[0]["content"]


def test_invalid_output_is_retried_and_accepted_attempt_stops_the_loop():
    request = _request()
    factory = _Factory(
        [
            ModelTurnResponse(kind=ModelResponseKind.FINAL, final_text="{}"),
            ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                final_text=_valid_output(request),
            ),
            ModelTurnResponse(kind=ModelResponseKind.FINAL, final_text="{}"),
        ]
    )

    result = SemanticJudge(
        adapter_factory=factory,
        evaluator_execution=_execution(),
    ).run(request)

    assert result.status is JudgeRunStatus.GRADED
    assert [item.status for item in result.attempts] == [
        JudgeAttemptStatus.INVALID_OUTPUT,
        JudgeAttemptStatus.ACCEPTED,
    ]
    assert factory.created == 2


def test_semantic_unknown_is_a_graded_decision_not_judge_failed():
    request = _request()
    factory = _Factory(
        [
            ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                final_text=_valid_output(request, relation="unknown", score=0),
            )
        ]
    )

    result = SemanticJudge(
        adapter_factory=factory,
        evaluator_execution=_execution(),
    ).execute(request)

    assert result.status is JudgeRunStatus.GRADED
    assert result.decision.relation is IntentJudgeRelation.UNKNOWN
    assert result.failure is None


def test_provider_failures_are_judge_failed_and_not_cached():
    request = _request()
    factory = _Factory([TimeoutError("secret endpoint"), TimeoutError("again")])
    cache = InMemoryJudgeResultCache()
    judge = SemanticJudge(
        adapter_factory=factory,
        evaluator_execution=_execution(),
        cache=cache,
    )

    result = judge.execute(request)

    assert result.status is JudgeRunStatus.JUDGE_FAILED
    assert result.failure.code is JudgeFailureCode.ATTEMPTS_EXHAUSTED
    assert len(result.attempts) == 2
    assert cache.get(result.cache_key) is None
    decision, failure, ungraded = intent_resolution_from_judge_result(result)
    assert decision is None
    assert failure.request_id == request.source_request_id
    assert failure.failure_code == "attempts_exhausted"
    assert failure.evaluator_execution_digest == result.evaluator_execution_digest
    assert failure.judge_result_digest == result.digest()
    assert ungraded is None


def test_failed_result_hydration_requires_live_canonical_terminal_failure():
    request = _request()
    result = SemanticJudge(
        adapter_factory=_Factory([TimeoutError("one"), TimeoutError("two")]),
        evaluator_execution=_execution(),
    ).execute(request)

    non_live = result.to_dict()
    non_live["source"] = JudgeExecutionSource.NOT_RUN.value
    with pytest.raises(JudgeProtocolError, match="inconsistent terminal fields"):
        JudgeExecutionResult.from_dict(non_live)

    retryable_terminal = result.to_dict()
    retryable_terminal["failure"]["retryable"] = True
    with pytest.raises(JudgeProtocolError, match="terminal failure must be non-retryable"):
        JudgeExecutionResult.from_dict(retryable_terminal)

    truncated = result.to_dict()
    truncated["attempts"] = truncated["attempts"][:1]
    truncated["failure"] = dict(truncated["attempts"][0]["failure"])
    truncated["failure"]["retryable"] = False
    with pytest.raises(JudgeProtocolError, match="canonical attempts_exhausted"):
        JudgeExecutionResult.from_dict(truncated)


def test_request_deadline_reserves_a_full_timeout_before_starting_each_retry():
    request = _request()

    class Clock:
        value = 0.0

        def __call__(self):
            return self.value

    clock = Clock()

    def slow_response(_request):
        clock.value = 61.0
        return ModelTurnResponse(
            kind=ModelResponseKind.FINAL,
            final_text=_valid_output(request),
        )

    factory = _Factory([slow_response])
    result = SemanticJudge(
        adapter_factory=factory,
        evaluator_execution=_execution(),
        clock=clock,
    ).execute(request)

    assert result.status is JudgeRunStatus.JUDGE_FAILED
    assert [item.status for item in result.attempts] == [
        JudgeAttemptStatus.TIMEOUT,
        JudgeAttemptStatus.DEADLINE_EXCEEDED,
    ]
    assert result.failure.code is JudgeFailureCode.DEADLINE_EXCEEDED
    assert factory.created == 1


def test_cache_reuses_only_graded_result_and_binds_execution_digest():
    request = _request()
    cache = InMemoryJudgeResultCache()
    first_factory = _Factory(
        [ModelTurnResponse(kind=ModelResponseKind.FINAL, final_text=_valid_output(request))]
    )
    first = SemanticJudge(
        adapter_factory=first_factory,
        evaluator_execution=_execution(),
        cache=cache,
    ).execute(request)
    second_factory = _Factory([])
    second = SemanticJudge(
        adapter_factory=second_factory,
        evaluator_execution=_execution(),
        cache=cache,
    ).execute(request)

    assert first.status is JudgeRunStatus.GRADED
    assert second.source is JudgeExecutionSource.CACHE
    assert second.attempts == first.attempts
    assert second.accepted_attempt_index == first.accepted_attempt_index
    assert second.decision == first.decision
    assert second_factory.created == 0


def test_hydration_replays_accepted_output_and_rejects_forged_decision_or_identity():
    request = _request()
    result = SemanticJudge(
        adapter_factory=_Factory(
            [
                ModelTurnResponse(
                    kind=ModelResponseKind.FINAL,
                    final_text=_valid_output(request),
                )
            ]
        ),
        evaluator_execution=_execution(),
    ).execute(request)

    forged = result.to_dict()
    forged["attempts"][0]["decision"]["relation"] = "different"
    with pytest.raises(JudgeProtocolError, match="differs from parsed output"):
        JudgeExecutionResult.from_dict(forged)

    wrong_kind = result.to_dict()
    wrong_kind["attempts"][0]["response_kind"] = "invalid"
    with pytest.raises(JudgeProtocolError, match="invalid response identity"):
        JudgeExecutionResult.from_dict(wrong_kind)

    wrong_terminal = result.to_dict()
    wrong_terminal["decision"]["relation"] = "different"
    with pytest.raises(JudgeProtocolError, match="terminal decision"):
        JudgeExecutionResult.from_dict(wrong_terminal)

    cached = result.as_cache_hit().to_dict()
    cached["cache_entry_digest"] = "0" * 64
    with pytest.raises(JudgeProtocolError, match="live cache entry"):
        JudgeExecutionResult.from_dict(cached)


def test_concurrent_cache_return_is_revalidated_before_reuse():
    request = _request()
    poison_request = build_intent_judge_input(
        IntentSemanticJudgeRequest(
            request_id="intent-request-poison",
            generated_id="generated-poison",
            truth_id="truth-poison",
            dimension=IntentDimension.GOAL,
            generated_text="Poison candidate",
            truth_text="Poison truth",
            truth_kind=IntentTruthKind.EXPECTED,
        )
    )
    poison = SemanticJudge(
        adapter_factory=_Factory(
            [
                ModelTurnResponse(
                    kind=ModelResponseKind.FINAL,
                    final_text=_valid_output(poison_request),
                )
            ]
        ),
        evaluator_execution=_execution(),
    ).execute(poison_request)

    class RacingCache:
        def get(self, cache_key):
            return None

        def put_if_absent(self, cache_key, result):
            return poison

    with pytest.raises(JudgeProtocolError, match="cache entry binding"):
        SemanticJudge(
            adapter_factory=_Factory(
                [
                    ModelTurnResponse(
                        kind=ModelResponseKind.FINAL,
                        final_text=_valid_output(request),
                    )
                ]
            ),
            evaluator_execution=_execution(),
            cache=RacingCache(),
        ).execute(request)


def test_model_rubric_budget_and_context_changes_all_miss_the_cache_key():
    request = _request()
    base_execution = _execution()
    base = SemanticJudge(
        adapter_factory=_Factory([]),
        evaluator_execution=base_execution,
    ).execute(request, ungraded_reason=JudgeUngradedReason.POLICY_SKIPPED)

    changed_model = SemanticJudge(
        adapter_factory=_Factory([], model="model-v2"),
        evaluator_execution=_execution(model="model-v2"),
    ).execute(request, ungraded_reason=JudgeUngradedReason.POLICY_SKIPPED)
    assert changed_model.cache_key != base.cache_key

    budget_payload = base_execution.judge_budgets.to_dict()
    budget_payload["attempt_timeout_seconds"] = 59
    changed_budget = SemanticJudge(
        adapter_factory=_Factory([]),
        evaluator_execution=_execution(
            budgets=JudgeExecutionBudgets.from_dict(budget_payload)
        ),
    ).execute(request, ungraded_reason=JudgeUngradedReason.POLICY_SKIPPED)
    assert changed_budget.cache_key != base.cache_key

    custom_rubrics = list(DEFAULT_JUDGE_RUBRICS.rubrics)
    intent_index = next(
        index
        for index, rubric in enumerate(custom_rubrics)
        if rubric.task is JudgeTask.INTENT_EQUIVALENCE
    )
    intent_rubric = custom_rubrics[intent_index]
    custom_rubrics[intent_index] = JudgeRubric.create(
        task=intent_rubric.task,
        rubric_id=intent_rubric.rubric_id,
        rubric_version="intent-equivalence-v2",
        response_schema=intent_rubric.response_schema,
        instruction=intent_rubric.instruction + " Apply the same-meaning test.",
    )
    custom_catalog = JudgeRubricCatalog.create(
        "core-code-review-judge-rubrics-v2",
        custom_rubrics,
    )
    custom_source = IntentSemanticJudgeRequest(
        request_id="intent-request-1",
        generated_id="generated-1",
        truth_id="truth-1",
        dimension=IntentDimension.GOAL,
        generated_text="Preserve existing callers.",
        truth_text="Keep all existing callers working.",
        truth_kind=IntentTruthKind.EXPECTED,
    )
    custom_request = build_intent_judge_input(
        custom_source,
        rubrics=custom_catalog,
    )
    changed_rubric = SemanticJudge(
        adapter_factory=_Factory([]),
        evaluator_execution=_execution(rubrics=custom_catalog),
    ).execute(
        custom_request,
        ungraded_reason=JudgeUngradedReason.POLICY_SKIPPED,
    )
    assert changed_rubric.cache_key != base.cache_key

    finding = SubmissionFinding(
        finding_id="novel-1",
        claim="The handler ignores parse errors.",
        severity=FindingSeverity.HIGH,
        path="src/handler.py",
        side=DiffSide.RIGHT,
        from_line=1,
        to_line=2,
        evidence_refs=(),
        suggested_action=None,
    )
    context_a = build_novel_factuality_judge_input(
        "novel-request-1",
        finding,
        context_sources=(
            repository_context(
                source_id="code-1",
                kind=JudgeContextKind.CODE,
                content="err is ignored",
                revision="head",
                path="src/handler.py",
            ),
        ),
    )
    context_b = build_novel_factuality_judge_input(
        "novel-request-1",
        finding,
        context_sources=(
            repository_context(
                source_id="code-1",
                kind=JudgeContextKind.CODE,
                content="err is returned to the caller",
                revision="head",
                path="src/handler.py",
            ),
        ),
    )
    context_a_result = SemanticJudge(
        adapter_factory=_Factory([]),
        evaluator_execution=_execution(),
    ).execute(context_a, ungraded_reason=JudgeUngradedReason.POLICY_SKIPPED)
    context_b_result = SemanticJudge(
        adapter_factory=_Factory([]),
        evaluator_execution=_execution(),
    ).execute(context_b, ungraded_reason=JudgeUngradedReason.POLICY_SKIPPED)
    assert context_a_result.cache_key != context_b_result.cache_key


def test_capability_preflight_rejects_unbounded_third_party_adapter():
    request = _request()
    factory = _Factory(
        [],
        capabilities=ModelAdapterCapabilities(
            supports_tool_choice_none=True,
            enforces_request_timeout=False,
            max_response_bytes=None,
        ),
    )

    result = SemanticJudge(
        adapter_factory=factory,
        evaluator_execution=_execution(),
    ).execute(request)

    assert result.status is JudgeRunStatus.JUDGE_FAILED
    assert result.failure.code is JudgeFailureCode.ADAPTER_CAPABILITY_MISSING
    assert result.attempts[0].status is JudgeAttemptStatus.PREFLIGHT_FAILED


def test_identity_drift_is_fail_closed_without_retry():
    request = _request()
    factory = _Factory(
        [ModelTurnResponse(kind=ModelResponseKind.FINAL, final_text=_valid_output(request))],
        model="unexpected-model",
    )
    result = SemanticJudge(
        adapter_factory=factory,
        evaluator_execution=_execution(),
    ).execute(request)

    assert result.status is JudgeRunStatus.JUDGE_FAILED
    assert result.failure.code is JudgeFailureCode.ADAPTER_IDENTITY_MISMATCH
    assert len(result.attempts) == 1


@pytest.mark.parametrize(
    ("response", "failure_code", "attempt_status"),
    [
        (
            ModelTurnResponse(
                kind=ModelResponseKind.TOOL_CALLS,
                tool_calls=[ModelToolCall("call-1", "shell", {})],
            ),
            JudgeFailureCode.UNSAFE_OUTPUT,
            JudgeAttemptStatus.UNSAFE_OUTPUT,
        ),
        (
            ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                final_text="{}",
                raw={"choices": [{"finish_reason": "length"}]},
            ),
            JudgeFailureCode.OUTPUT_TRUNCATED,
            JudgeAttemptStatus.OUTPUT_TRUNCATED,
        ),
    ],
)
def test_tool_calls_and_truncated_output_are_rejected(response, failure_code, attempt_status):
    result = SemanticJudge(
        adapter_factory=_Factory([response]),
        evaluator_execution=_execution(),
    ).execute(_request())

    assert result.status is JudgeRunStatus.JUDGE_FAILED
    assert result.failure.code is failure_code
    assert result.attempts[0].status is attempt_status


def test_output_byte_overflow_is_not_retained_or_cached():
    request = _request()
    payload = JudgeExecutionBudgets.defaults(
        evaluator_timeout_seconds=300,
        max_execution_artifact_file_bytes=8 * 1024 * 1024,
        max_execution_artifact_total_bytes=32 * 1024 * 1024,
    ).to_dict()
    payload["max_model_response_bytes"] = 64
    payload["max_total_judge_response_bytes"] = 1024
    budgets = JudgeExecutionBudgets.from_dict(payload)
    factory = _Factory(
        [
            ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                final_text="x" * 1000,
            )
        ],
        capabilities=ModelAdapterCapabilities(
            supports_tool_choice_none=True,
            enforces_request_timeout=True,
            max_response_bytes=64,
        ),
    )
    result = SemanticJudge(
        adapter_factory=factory,
        evaluator_execution=_execution(budgets=budgets),
    ).execute(request)

    assert result.status is JudgeRunStatus.JUDGE_FAILED
    assert result.failure.code is JudgeFailureCode.OUTPUT_LIMIT_EXCEEDED
    assert result.attempts[0].output_text is None
    assert result.attempts[0].output_size_bytes == 1000


def test_input_context_budget_failure_is_typed_and_does_not_call_adapter():
    request = _request()
    payload = JudgeExecutionBudgets.defaults(
        evaluator_timeout_seconds=300,
        max_execution_artifact_file_bytes=8 * 1024 * 1024,
        max_execution_artifact_total_bytes=32 * 1024 * 1024,
    ).to_dict()
    payload["max_context_block_bytes"] = 10
    payload["max_context_bytes_per_request"] = 10
    payload["max_model_request_bytes"] = 10
    budgets = JudgeExecutionBudgets.from_dict(payload)
    factory = _Factory([])

    result = SemanticJudge(
        adapter_factory=factory,
        evaluator_execution=_execution(budgets=budgets),
    ).execute(request)

    assert result.status is JudgeRunStatus.JUDGE_FAILED
    assert result.failure.code is JudgeFailureCode.CONTEXT_BUDGET_EXCEEDED
    assert result.attempts == ()
    assert factory.created == 0


def test_policy_skip_is_ungraded_and_does_not_call_adapter():
    request = _request()
    factory = _Factory([])
    result = SemanticJudge(
        adapter_factory=factory,
        evaluator_execution=_execution(),
    ).execute(request, ungraded_reason=JudgeUngradedReason.POLICY_SKIPPED)

    assert result.status is JudgeRunStatus.UNGRADED
    assert result.source is JudgeExecutionSource.NOT_RUN
    assert result.ungraded_reason is JudgeUngradedReason.POLICY_SKIPPED
    assert factory.created == 0
    decision, failure, ungraded = intent_resolution_from_judge_result(result)
    assert decision is None
    assert failure is None
    assert ungraded.request_id == request.source_request_id
    assert ungraded.ungraded_reason == "policy_skipped"


def test_aggregate_judge_artifacts_round_trip_and_bind_every_request():
    from tests.eval.test_intent_evaluator import (
        evaluate as evaluate_intent,
        semantic_fixture,
    )

    intent, truth, pending = semantic_fixture()
    request = build_intent_judge_input(pending.judge_requests[0])
    execution = _execution()
    result = SemanticJudge(
        adapter_factory=_Factory(
            [
                ModelTurnResponse(
                    kind=ModelResponseKind.FINAL,
                    final_text=_valid_output(request),
                )
            ]
        ),
        evaluator_execution=execution,
    ).execute(request)
    decision, failure, ungraded = intent_resolution_from_judge_result(result)
    assert failure is None and ungraded is None
    intent_evaluation = evaluate_intent(
        intent,
        truth,
        decisions=(decision,),
    )
    inputs = JudgeInputArtifact.create(execution, (request,))
    outputs = JudgeOutputArtifact.create(
        inputs,
        execution,
        (result,),
        intent_evaluation=intent_evaluation,
    )

    assert JudgeInputArtifact.from_json(
        inputs.to_json(),
        evaluator_execution=execution,
    ) == inputs
    hydrated = JudgeOutputArtifact.from_json(
        outputs.to_json(),
        input_artifact=inputs,
        evaluator_execution=execution,
        intent_evaluation=intent_evaluation,
    )
    hydrated.validate_against(inputs)
    hydrated.validate_against_execution(execution)
    hydrated.validate_intent_evaluation(intent_evaluation)
    assert hydrated == outputs

    with pytest.raises(JudgeProtocolError, match="requires its bound Intent"):
        JudgeOutputArtifact.from_json(
            outputs.to_json(),
            input_artifact=inputs,
            evaluator_execution=execution,
        )

    different_inputs = JudgeInputArtifact.create(execution, ())
    with pytest.raises(JudgeProtocolError, match="does not bind"):
        hydrated.validate_against(different_inputs)


def test_intent_failure_provenance_must_match_the_actual_judge_output_artifact():
    from review_agent_eval.intent_evaluator import IntentSemanticJudgeFailure
    from tests.eval.test_intent_evaluator import (
        evaluate as evaluate_intent,
        semantic_fixture,
    )

    intent, truth, pending = semantic_fixture()
    request = build_intent_judge_input(pending.judge_requests[0])
    execution = _execution()
    result = SemanticJudge(
        adapter_factory=_Factory([TimeoutError("one"), TimeoutError("two")]),
        evaluator_execution=execution,
    ).execute(request)
    decision, failure, ungraded = intent_resolution_from_judge_result(result)
    assert decision is None and ungraded is None
    evaluation = evaluate_intent(intent, truth, failures=(failure,))
    inputs = JudgeInputArtifact.create(execution, (request,))
    JudgeOutputArtifact.create(
        inputs,
        execution,
        (result,),
        intent_evaluation=evaluation,
    )

    forged_failure = IntentSemanticJudgeFailure(
        request_id=failure.request_id,
        failure_code=failure.failure_code,
        evaluator_execution_digest=failure.evaluator_execution_digest,
        judge_result_digest="0" * 64,
    )
    forged_evaluation = evaluate_intent(
        intent,
        truth,
        failures=(forged_failure,),
    )
    with pytest.raises(JudgeProtocolError, match="failures differ"):
        JudgeOutputArtifact.create(
            inputs,
            execution,
            (result,),
            intent_evaluation=forged_evaluation,
        )


def test_aggregate_artifact_budget_counts_full_serialized_result_not_only_model_text():
    from tests.eval.test_intent_evaluator import (
        evaluate as evaluate_intent,
        semantic_fixture,
    )

    intent, truth, pending = semantic_fixture()
    request = build_intent_judge_input(pending.judge_requests[0])
    execution = _execution(
        artifact_file_bytes=6_000,
        artifact_total_bytes=12_000,
    )
    result = SemanticJudge(
        adapter_factory=_Factory(
            [
                ModelTurnResponse(
                    kind=ModelResponseKind.FINAL,
                    final_text=_valid_output(request),
                )
            ],
            capabilities=ModelAdapterCapabilities(
                supports_tool_choice_none=True,
                enforces_request_timeout=True,
                max_response_bytes=6_000,
            ),
        ),
        evaluator_execution=execution,
    ).execute(request)
    decision, failure, ungraded = intent_resolution_from_judge_result(result)
    assert failure is None and ungraded is None
    evaluation = evaluate_intent(intent, truth, decisions=(decision,))
    inputs = JudgeInputArtifact.create(execution, (request,))

    with pytest.raises(JudgeProtocolError, match="artifact file budget"):
        JudgeOutputArtifact.create(
            inputs,
            execution,
            (result,),
            intent_evaluation=evaluation,
        )


def test_output_hydration_rechecks_combined_input_output_artifact_budget():
    from tests.eval.test_intent_evaluator import (
        evaluate as evaluate_intent,
        semantic_fixture,
    )

    intent, truth, pending = semantic_fixture()
    request = build_intent_judge_input(pending.judge_requests[0])
    execution = _execution(
        artifact_file_bytes=13_000,
        artifact_total_bytes=13_000,
    )
    result = SemanticJudge(
        adapter_factory=_Factory(
            [
                ModelTurnResponse(
                    kind=ModelResponseKind.FINAL,
                    final_text=_valid_output(request),
                )
            ],
            capabilities=ModelAdapterCapabilities(
                supports_tool_choice_none=True,
                enforces_request_timeout=True,
                max_response_bytes=13_000,
            ),
        ),
        evaluator_execution=execution,
    ).execute(request)
    decision, failure, ungraded = intent_resolution_from_judge_result(result)
    assert failure is None and ungraded is None
    evaluation = evaluate_intent(intent, truth, decisions=(decision,))
    inputs = JudgeInputArtifact.create(execution, (request,))
    raw_output = JudgeOutputArtifact(
        schema_version="eval_judge_output_artifact_v1",
        evaluator_execution_digest=execution.digest(),
        input_artifact_digest=inputs.digest(),
        intent_evaluation_digest=evaluation.digest(),
        results=(result,),
    )
    input_bytes = len(inputs.to_json().encode("utf-8"))
    output_bytes = len(raw_output.to_json().encode("utf-8"))
    assert input_bytes <= execution.max_execution_artifact_file_bytes
    assert output_bytes <= execution.max_execution_artifact_file_bytes
    assert input_bytes + output_bytes > execution.max_execution_artifact_total_bytes

    with pytest.raises(JudgeProtocolError, match="input/output artifacts"):
        JudgeOutputArtifact.create(
            inputs,
            execution,
            (result,),
            intent_evaluation=evaluation,
        )
    with pytest.raises(JudgeProtocolError, match="input/output artifacts"):
        JudgeOutputArtifact.from_json(
            raw_output.to_json(),
            input_artifact=inputs,
            evaluator_execution=execution,
            intent_evaluation=evaluation,
        )
