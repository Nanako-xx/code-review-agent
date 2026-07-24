from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from typing import Any, Iterator

import pytest

import review_agent_eval.evidence_checker as evidence_checker_module
from review_agent_eval.evidence_checker import (
    EvidenceIntegrityChecker,
    EvidenceItemIntegrityResult,
    EvidenceReasonCode,
)
from review_agent_eval.frozen_context import (
    FrozenContextReplay,
    FrozenContextTargetMaterializer,
)
from review_agent_eval.judge import JudgeContextKind, JudgeTask
from review_agent_eval.models import (
    EVAL_SUBMISSION_SCHEMA_VERSION,
    EvalInput,
    EvalSubmission,
    EvidenceIntegrity,
    EvidenceKind,
    ExpectedFinding,
    FindingSeverity,
    FrozenContextEvidenceSource,
    IntentResult,
    MetricAuthority,
    NovelFindingPolicy,
    RepositoryFileEvidenceSource,
    RequiredContextLevel,
    ReviewTruth,
    SchemaError,
    SubmissionEvidence,
    SubmissionFinding,
    SubmissionIntent,
    SubmissionReview,
    SubmissionStatus,
    SubmissionUsage,
    TrialStatus,
    TruthCompleteness,
)
from review_agent_eval.review_evaluator import (
    EvidenceSupportResolution,
    ReviewEvaluationPhase,
    ReviewEvaluator,
)

from tests.eval.test_evidence_checker import (
    TARGET_MATERIALIZATION_ID,
    TRIAL_ID,
    _build_harness,
    _file_evidence,
)
from tests.eval.test_frozen_context import _eval_input, _prepared_bundle, _request
from tests.eval.test_judge import _execution


@dataclass(frozen=True)
class FrozenHarness:
    prepared: Any
    eval_input: EvalInput
    replay: FrozenContextReplay
    target_materialization_id: str


@pytest.fixture
def frozen_harness(tmp_path: Path) -> Iterator[FrozenHarness]:
    prepared = _prepared_bundle(tmp_path)
    eval_input = _eval_input(prepared)
    materialized = FrozenContextTargetMaterializer(
        bundle_root=prepared.root,
        workspace_root=tmp_path / "w",
    ).materialize(_request(eval_input, prepared))
    with materialized:
        yield FrozenHarness(
            prepared=prepared,
            eval_input=eval_input,
            replay=materialized.replay,
            target_materialization_id=materialized.materialization_id,
        )


@pytest.fixture
def repository_harness(tmp_path: Path) -> Iterator[Any]:
    value = _build_harness(tmp_path, heavy=False)
    try:
        yield value
    finally:
        value.preparer.__exit__(None, None, None)


def _checker(harness: FrozenHarness) -> EvidenceIntegrityChecker:
    return EvidenceIntegrityChecker(
        eval_input=harness.eval_input,
        replay=harness.replay,
        trial_id=TRIAL_ID,
        target_materialization_id=harness.target_materialization_id,
    )


def _evidence(
    harness: FrozenHarness,
    *,
    evidence_id: str = "evidence-frozen",
    target_materialization_id: str | None = None,
    context_ref: str | None = None,
    from_line: int = 1,
    to_line: int = 1,
    excerpt: str | None = None,
    content_hash: str | None = None,
) -> SubmissionEvidence:
    raw = harness.replay.read_lines(from_line, to_line) if excerpt is None else None
    canonical = raw.decode("utf-8", "strict") if raw is not None else excerpt
    assert canonical is not None
    return SubmissionEvidence(
        evidence_id=evidence_id,
        source=FrozenContextEvidenceSource(
            kind=EvidenceKind.FROZEN_CONTEXT,
            target_materialization_id=(
                harness.target_materialization_id
                if target_materialization_id is None
                else target_materialization_id
            ),
            context_ref=(
                harness.replay.context_ref if context_ref is None else context_ref
            ),
            from_line=from_line,
            to_line=to_line,
        ),
        content_hash=(
            hashlib.sha256(canonical.encode("utf-8", "strict")).hexdigest()
            if content_hash is None
            else content_hash
        ),
        excerpt=canonical,
    )


def _reasons(result: EvidenceItemIntegrityResult) -> set[EvidenceReasonCode]:
    return {item.reason_code for item in result.diagnostics}


def test_exact_frozen_lines_replay_as_valid_canonical_evidence(
    frozen_harness: FrozenHarness,
) -> None:
    evidence = _evidence(frozen_harness, from_line=1, to_line=2)

    result = _checker(frozen_harness).check_item(evidence)

    assert result.integrity is EvidenceIntegrity.VALID
    assert result.diagnostics == ()


def test_check_all_reads_and_indexes_one_frozen_record_once(
    frozen_harness: FrozenHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _evidence(
        frozen_harness,
        evidence_id="evidence-frozen-first",
        from_line=1,
        to_line=1,
    )
    second = _evidence(
        frozen_harness,
        evidence_id="evidence-frozen-second",
        from_line=2,
        to_line=2,
    )
    original = FrozenContextReplay.read_exact
    calls = 0

    def counted(replay: FrozenContextReplay) -> bytes:
        nonlocal calls
        calls += 1
        return original(replay)

    monkeypatch.setattr(FrozenContextReplay, "read_exact", counted)
    findings = (
        replace(_finding(first.evidence_id), finding_id="finding-frozen-first"),
        replace(_finding(second.evidence_id), finding_id="finding-frozen-second"),
    )

    results = _checker(frozen_harness).check_all(findings, (first, second))

    assert calls == 1
    assert all(item.integrity is EvidenceIntegrity.VALID for item in results)


def test_frozen_record_line_limit_is_global_and_deterministic(
    frozen_harness: FrozenHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _evidence(frozen_harness)
    monkeypatch.setattr(evidence_checker_module, "MAX_REPLAY_LINES", 1)

    result = _checker(frozen_harness).check_item(evidence)

    assert result.integrity is EvidenceIntegrity.INVALID
    assert _reasons(result) == {EvidenceReasonCode.REPLAY_LINE_LIMIT_EXCEEDED}


def test_frozen_line_index_preserves_crlf_and_final_line_exactly(
    frozen_harness: FrozenHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _evidence(
        frozen_harness,
        evidence_id="evidence-frozen-crlf",
        from_line=1,
        to_line=1,
        excerpt="a\r\n",
    )
    last = _evidence(
        frozen_harness,
        evidence_id="evidence-frozen-final",
        from_line=3,
        to_line=3,
        excerpt="last",
    )
    monkeypatch.setattr(
        FrozenContextReplay,
        "read_exact",
        lambda _replay: b"a\r\nb\nlast",
    )

    results = _checker(frozen_harness).check_all(
        (
            replace(_finding(first.evidence_id), finding_id="finding-frozen-crlf"),
            replace(_finding(last.evidence_id), finding_id="finding-frozen-final"),
        ),
        (first, last),
    )

    assert all(item.integrity is EvidenceIntegrity.VALID for item in results)


def test_frozen_evidence_rejects_wrong_target_materialization_id(
    frozen_harness: FrozenHarness,
) -> None:
    evidence = _evidence(
        frozen_harness,
        target_materialization_id=TARGET_MATERIALIZATION_ID,
    )

    result = _checker(frozen_harness).check_item(evidence)

    assert result.integrity is EvidenceIntegrity.INVALID
    assert _reasons(result) == {EvidenceReasonCode.TARGET_MATERIALIZATION_MISMATCH}


def test_frozen_evidence_rejects_wrong_context_ref(
    frozen_harness: FrozenHarness,
) -> None:
    result = _checker(frozen_harness).check_item(
        _evidence(frozen_harness, context_ref="wrong-context-ref")
    )

    assert result.integrity is EvidenceIntegrity.INVALID
    assert _reasons(result) == {EvidenceReasonCode.CONTEXT_REF_MISMATCH}


@pytest.mark.parametrize(
    ("from_line", "to_line", "reason"),
    [
        (2, 1, EvidenceReasonCode.LINE_RANGE_REVERSED),
        (1, 1_000_000, EvidenceReasonCode.LINE_RANGE_OUT_OF_BOUNDS),
    ],
)
def test_frozen_evidence_rejects_invalid_line_ranges(
    frozen_harness: FrozenHarness,
    from_line: int,
    to_line: int,
    reason: EvidenceReasonCode,
) -> None:
    evidence = _evidence(
        frozen_harness,
        from_line=1,
        to_line=1,
    )
    evidence = replace(
        evidence,
        source=replace(
            evidence.source,
            from_line=from_line,
            to_line=to_line,
        ),
    )

    result = _checker(frozen_harness).check_item(evidence)

    assert result.integrity is EvidenceIntegrity.INVALID
    assert _reasons(result) == {reason}


def test_frozen_evidence_turns_bundle_or_rendered_drift_into_invalid_result(
    frozen_harness: FrozenHarness,
) -> None:
    checker = _checker(frozen_harness)
    evidence = _evidence(frozen_harness)
    binding = frozen_harness.prepared.manifest.records[0]
    record_path = frozen_harness.prepared.root / binding.path
    original = record_path.read_bytes()
    record_path.chmod(0o600)
    try:
        record_path.write_bytes(original + b"drift")
        result = checker.check_item(evidence)
    finally:
        record_path.write_bytes(original)

    assert result.integrity is EvidenceIntegrity.INVALID
    assert _reasons(result) == {EvidenceReasonCode.REPLAY_BINDING_MISMATCH}


def test_checker_rejects_frozen_replay_metadata_not_bound_to_eval_input(
    frozen_harness: FrozenHarness,
) -> None:
    replay = replace(frozen_harness.replay, rendered_sha256="0" * 64)

    with pytest.raises(ValueError, match="exactly bound"):
        EvidenceIntegrityChecker(
            frozen_harness.eval_input,
            replay,
            TRIAL_ID,
            frozen_harness.target_materialization_id,
        )


def test_repository_and_frozen_replays_cannot_be_cross_bound(
    frozen_harness: FrozenHarness,
    repository_harness: Any,
) -> None:
    with pytest.raises(TypeError, match="FrozenContextReplay"):
        EvidenceIntegrityChecker(
            frozen_harness.eval_input,
            repository_harness.replay,
            TRIAL_ID,
            frozen_harness.target_materialization_id,
        )
    with pytest.raises(TypeError, match="PreparedRepositoryReplay"):
        EvidenceIntegrityChecker(
            repository_harness.eval_input,
            frozen_harness.replay,
            TRIAL_ID,
            TARGET_MATERIALIZATION_ID,
        )


def test_repository_and_frozen_source_kinds_do_not_fallback_across_replays(
    frozen_harness: FrozenHarness,
    repository_harness: Any,
) -> None:
    repository_checker = EvidenceIntegrityChecker(
        repository_harness.eval_input,
        repository_harness.replay,
        TRIAL_ID,
        TARGET_MATERIALIZATION_ID,
    )
    frozen_for_repository = _evidence(
        frozen_harness,
        target_materialization_id=TARGET_MATERIALIZATION_ID,
    )
    repository_for_frozen = _file_evidence(repository_harness)
    repository_for_frozen = replace(
        repository_for_frozen,
        source=replace(
            repository_for_frozen.source,
            target_materialization_id=frozen_harness.target_materialization_id,
        ),
    )

    first = repository_checker.check_item(frozen_for_repository)
    second = _checker(frozen_harness).check_item(repository_for_frozen)

    assert _reasons(first) == {EvidenceReasonCode.REPLAY_BINDING_MISMATCH}
    assert _reasons(second) == {EvidenceReasonCode.REPLAY_BINDING_MISMATCH}


def test_source_bound_frozen_receipt_hydration_cannot_forge_validity(
    frozen_harness: FrozenHarness,
) -> None:
    checker = _checker(frozen_harness)
    evidence = _evidence(frozen_harness, context_ref="wrong-context-ref")
    forged = checker.check_item(evidence).to_dict()
    forged["integrity"] = EvidenceIntegrity.VALID.value
    forged["diagnostics"] = []

    with pytest.raises(SchemaError, match="deterministic Evidence replay"):
        EvidenceItemIntegrityResult.from_dict(
            forged,
            evidence=evidence,
            checker=checker,
        )


def _finding(evidence_id: str) -> SubmissionFinding:
    return SubmissionFinding(
        finding_id="finding-frozen",
        claim="The frozen context contains the exact reported defect.",
        severity=FindingSeverity.HIGH,
        path=None,
        side=None,
        from_line=None,
        to_line=None,
        evidence_refs=(evidence_id,),
        suggested_action="Fix the defect.",
    )


def _truth(claim: str) -> ReviewTruth:
    return ReviewTruth(
        completeness=TruthCompleteness.CLOSED_WORLD,
        novel_finding_policy=NovelFindingPolicy.FORBID,
        expected_findings=(
            ExpectedFinding(
                truth_id="truth-frozen",
                claim=claim,
                severity=None,
                category="correctness",
                required=True,
                metric_authority=MetricAuthority(
                    severity_scorable=False,
                    severity_authority=None,
                    location_scorable=False,
                    location_authority=None,
                ),
                locations=(),
                evidence_anchors=(),
                required_context_level=RequiredContextLevel.DIFF,
                rationale="The exact claim is part of the frozen truth fixture.",
            ),
        ),
        known_invalid_findings=(),
    )


def _submission(
    harness: FrozenHarness,
    finding: SubmissionFinding,
    evidence: SubmissionEvidence,
) -> EvalSubmission:
    return EvalSubmission(
        schema_version=EVAL_SUBMISSION_SCHEMA_VERSION,
        task_id=harness.eval_input.task_id,
        agent_id="agent-frozen-evidence",
        trial_id=TRIAL_ID,
        eval_input_digest=harness.eval_input.digest(),
        target_materialization_id=harness.target_materialization_id,
        status=SubmissionStatus.COMPLETED,
        intent=SubmissionIntent(
            status=IntentResult.SUFFICIENT,
            goal="Review the frozen context.",
            acceptance_criteria=(),
            scope=(),
            constraints=(),
            claims=(),
            clarification_questions=(),
            uncertainties=(),
        ),
        review=SubmissionReview(findings=(finding,), uncertainties=()),
        evidence=(evidence,),
        usage=SubmissionUsage(
            elapsed_seconds=0,
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            tool_calls=None,
            cost_amount=None,
            cost_currency=None,
        ),
        trace_ref=None,
        failure=None,
    )


def _evaluator(harness: FrozenHarness) -> ReviewEvaluator:
    return ReviewEvaluator(
        eval_input=harness.eval_input,
        replay=harness.replay,
        trial_id=TRIAL_ID,
        target_materialization_id=harness.target_materialization_id,
        evaluator_execution=_execution(),
    )


def test_valid_frozen_evidence_is_projected_to_support_judge_metadata(
    frozen_harness: FrozenHarness,
) -> None:
    evidence = _evidence(frozen_harness)
    finding = _finding(evidence.evidence_id)

    result = _evaluator(frozen_harness).evaluate(
        _submission(frozen_harness, finding, evidence),
        _truth(finding.claim),
    )

    assert result.phase is ReviewEvaluationPhase.EVIDENCE_SUPPORT
    assert len(result.judge_requests) == 1
    request = result.judge_requests[0].request
    assert request.task is JudgeTask.EVIDENCE_SUPPORT
    context = next(item for item in request.contexts if item.kind is JudgeContextKind.EVIDENCE)
    assert context.metadata["kind"] == EvidenceKind.FROZEN_CONTEXT.value
    assert context.metadata["source_ref"] == frozen_harness.replay.context_ref


def test_wrong_frozen_ref_never_creates_support_or_judge_repair_path(
    frozen_harness: FrozenHarness,
) -> None:
    evidence = _evidence(frozen_harness, context_ref="wrong-context-ref")
    finding = _finding(evidence.evidence_id)

    result = _evaluator(frozen_harness).evaluate(
        _submission(frozen_harness, finding, evidence),
        _truth(finding.claim),
    )

    outcome = result.finding_outcomes[0]
    assert outcome.evidence_integrity is EvidenceIntegrity.INVALID
    assert outcome.evidence_support_resolution is EvidenceSupportResolution.NOT_REQUESTED
    assert result.judge_requests == ()
