from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Iterator

import pytest

from tests.eval.test_evidence_checker import (
    TARGET_MATERIALIZATION_ID,
    _build_harness,
    _file_evidence,
)
from tests.eval.test_judge import _execution
from tests.eval.test_review_evaluator import (
    _finding,
    _run_scripted_judge,
    _submission,
    _truth,
)
from review_agent_eval.judge import (
    JudgeContextKind,
    JudgeContextTrust,
    repository_context,
)
from review_agent_eval.models import (
    EvaluatorContextProvenance,
    EvaluatorContextSource,
    EvaluatorContextSourceKind,
    EvaluatorContextTask,
    NovelFindingPolicy,
    ReviewEvaluatorContext,
    TruthCompleteness,
    TruthEvaluatorContext,
    UnsupportedProtocolVersionError,
    stable_id,
)
from review_agent_eval.review_evaluator import (
    ReviewContextBundle,
    ReviewEvaluationPhase,
    ReviewEvaluationResult,
    ReviewEvaluator,
    ReviewPairContextEntry,
    ReviewTruthKind,
)


SOURCE_KIND = "swe_truth_diff_hunk_v1"
SOURCE_ID_KIND = "swe-truth-diff-hunk-v1"
HUNK_A = "@@ -10,2 +10,2 @@\n-old_a()\n+new_a()\n"
HUNK_B = "@@ -20,2 +20,2 @@\n-old_b()\n+new_b()\n"


@pytest.fixture
def harness(tmp_path: Path) -> Iterator[object]:
    value = _build_harness(tmp_path, heavy=False)
    try:
        yield value
    finally:
        value.preparer.__exit__(None, None, None)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _provenance(record: str) -> EvaluatorContextProvenance:
    return EvaluatorContextProvenance(
        source_role="swe_truth_diff",
        source_file_sha256=_sha256_text("swe-source-file"),
        record_pointer=f"/records/{record}",
        record_sha256=_sha256_text(record),
    )


def _diff_source(
    content: str,
    *,
    provenance: EvaluatorContextProvenance,
) -> EvaluatorContextSource:
    return EvaluatorContextSource(
        kind=EvaluatorContextSourceKind.DIFF_HUNK,
        content=content,
        content_sha256=_sha256_text(content),
        provenance=provenance,
    )


def _truth_context(
    truth_id: str,
    content: str,
    *,
    provenance: EvaluatorContextProvenance,
) -> TruthEvaluatorContext:
    return TruthEvaluatorContext(
        truth_id=truth_id,
        allowed_tasks=(EvaluatorContextTask.FINDING_EQUIVALENCE,),
        sources=(_diff_source(content, provenance=provenance),),
    )


def _review_context(*items: TruthEvaluatorContext) -> ReviewEvaluatorContext:
    return ReviewEvaluatorContext(truth_contexts=tuple(items))


def _evaluator(
    harness: object,
    *,
    review_context: ReviewEvaluatorContext | None = None,
    context_bundle: ReviewContextBundle | None = None,
    execution=None,
) -> ReviewEvaluator:
    kwargs = {}
    if review_context is not None:
        kwargs["review_evaluator_context"] = review_context
    return ReviewEvaluator(
        eval_input=harness.eval_input,
        replay=harness.replay,
        trial_id="trial-review-evaluator",
        target_materialization_id=TARGET_MATERIALIZATION_ID,
        evaluator_execution=_execution() if execution is None else execution,
        context_bundle=(
            ReviewContextBundle()
            if context_bundle is None
            else context_bundle
        ),
        **kwargs,
    )


def _two_truths():
    truth = _truth("truth-a", "Truth A describes the changed branch defect.")
    truth_b = replace(
        truth.expected_findings[0],
        truth_id="truth-b",
        claim="Truth B describes a different changed branch defect.",
        rationale="Truth B rationale.",
    )
    return replace(
        truth,
        expected_findings=(truth.expected_findings[0], truth_b),
    )


def _context_binding(request):
    return next(
        item
        for item in request.reference_bindings
        if item.source_kind == SOURCE_KIND
    )


def test_diff_hunks_are_scoped_to_the_exact_finding_truth_pair(
    harness: object,
) -> None:
    source_a = _diff_source(HUNK_A, provenance=_provenance("a"))
    source_b = _diff_source(HUNK_B, provenance=_provenance("b"))
    context = _review_context(
        TruthEvaluatorContext(
            truth_id="truth-a",
            allowed_tasks=(EvaluatorContextTask.FINDING_EQUIVALENCE,),
            sources=(source_a,),
        ),
        TruthEvaluatorContext(
            truth_id="truth-b",
            allowed_tasks=(EvaluatorContextTask.FINDING_EQUIVALENCE,),
            sources=(source_b,),
        ),
    )
    evaluator = _evaluator(harness, review_context=context)
    finding = _finding(
        "finding-shared",
        "The generated Finding is semantically compared with both truths.",
    )

    result = evaluator.evaluate(_submission(harness, finding), _two_truths())

    requests = {
        item.truth_id: item.request
        for item in result.judge_requests
        if item.phase is ReviewEvaluationPhase.EXPECTED_ASSIGNMENT
    }
    assert set(requests) == {"truth-a", "truth-b"}
    for truth_id, own_hunk, other_hunk, source in (
        ("truth-a", HUNK_A, HUNK_B, source_a),
        ("truth-b", HUNK_B, HUNK_A, source_b),
    ):
        request = requests[truth_id]
        assert [item.content for item in request.contexts] == [own_hunk]
        block = request.contexts[0]
        assert block.kind is JudgeContextKind.DIFF
        assert block.trust is JudgeContextTrust.UNTRUSTED_REPOSITORY_DATA
        assert block.to_model_dict()["data_boundary"] == (
            JudgeContextTrust.UNTRUSTED_REPOSITORY_DATA.value
        )
        assert block.metadata == {
            "revision": None,
            "path": None,
            "side": None,
            "from_line": None,
            "to_line": None,
        }
        binding = _context_binding(request)
        assert binding.source_id == stable_id(
            SOURCE_ID_KIND,
            truth_id,
            source.content_sha256,
            source.provenance.digest(),
        )
        assert other_hunk not in request.to_json()
        assert own_hunk not in request.system_prompt
        assert all(own_hunk not in item.text for item in request.items)


def test_duplicate_typed_source_is_canonically_deduplicated(
    harness: object,
) -> None:
    source = _diff_source(HUNK_A, provenance=_provenance("duplicate"))
    context = _review_context(
        TruthEvaluatorContext(
            truth_id="truth-duplicate",
            allowed_tasks=(EvaluatorContextTask.FINDING_EQUIVALENCE,),
            sources=(source, source),
        )
    )
    evaluator = _evaluator(harness, review_context=context)
    finding = _finding("finding-duplicate", "A semantic defect claim.")
    submission = _submission(harness, finding)
    truth = _truth("truth-duplicate", "The canonical truth claim.")

    first = evaluator.evaluate(submission, truth).judge_requests[0]
    second = evaluator.evaluate(submission, truth).judge_requests[0]

    bindings = [
        item
        for item in first.request.reference_bindings
        if item.source_kind == SOURCE_KIND
    ]
    assert len(first.request.contexts) == 1
    assert len(bindings) == 1
    assert second.request_id == first.request_id
    assert second.request_digest == first.request_digest


def test_provenance_context_and_policy_bind_source_and_request_identity(
    harness: object,
) -> None:
    finding = _finding("finding-binding", "A semantic version of the defect.")
    truth = _truth("truth-binding", "The canonical truth defect.")
    submission = _submission(harness, finding)
    source_one = _diff_source(HUNK_A, provenance=_provenance("one"))
    source_two = _diff_source(HUNK_A, provenance=_provenance("two"))
    context_one = _review_context(
        TruthEvaluatorContext(
            truth_id="truth-binding",
            allowed_tasks=(EvaluatorContextTask.FINDING_EQUIVALENCE,),
            sources=(source_one,),
        )
    )
    context_two = _review_context(
        TruthEvaluatorContext(
            truth_id="truth-binding",
            allowed_tasks=(EvaluatorContextTask.FINDING_EQUIVALENCE,),
            sources=(source_two,),
        )
    )
    evaluator_one = _evaluator(harness, review_context=context_one)
    evaluator_two = _evaluator(harness, review_context=context_two)
    request_one = evaluator_one.evaluate(submission, truth).judge_requests[0]
    request_two = evaluator_two.evaluate(submission, truth).judge_requests[0]
    binding_one = _context_binding(request_one.request)
    binding_two = _context_binding(request_two.request)

    assert context_one.digest() != context_two.digest()
    assert binding_one.source_id != binding_two.source_id
    assert binding_one.source_digest != binding_two.source_digest
    assert evaluator_one.deterministic_context_digest != (
        evaluator_two.deterministic_context_digest
    )
    assert request_one.request_id != request_two.request_id
    assert request_one.request_digest != request_two.request_digest

    changed_policy = replace(
        _execution(),
        review_evaluator_context_policy_version="truth-scoped-context-v3",
    )
    with pytest.raises(UnsupportedProtocolVersionError):
        _evaluator(
            harness,
            review_context=context_one,
            execution=changed_policy,
        )


def test_evaluator_context_is_absent_from_agent_input_and_submission_evidence(
    harness: object,
) -> None:
    context = _review_context(
        _truth_context(
            "truth-isolation",
            HUNK_A,
            provenance=_provenance("isolation"),
        )
    )
    finding = _finding("finding-isolation", "A semantic defect claim.")
    submission = _submission(harness, finding)

    assert "review_evaluator_context" not in harness.eval_input.to_dict()
    assert HUNK_A not in harness.eval_input.to_json()
    assert HUNK_A not in submission.to_json()
    assert submission.evidence == ()

    evaluator = _evaluator(harness, review_context=context)
    assert evaluator.review_evaluator_context is context


def test_evaluator_diff_hunk_never_enters_novel_or_support_requests(
    harness: object,
) -> None:
    context = _review_context(
        _truth_context(
            "truth-stage",
            HUNK_A,
            provenance=_provenance("stages"),
        )
    )
    evaluator = _evaluator(harness, review_context=context)

    novel_finding = _finding("finding-novel-stage", "A novel semantic claim.")
    novel_truth = replace(
        _truth("truth-stage", "The known truth claim."),
        completeness=TruthCompleteness.EXPERT_AUGMENTED,
        novel_finding_policy=NovelFindingPolicy.VERIFY,
    )
    novel_submission = _submission(harness, novel_finding)
    equivalence_stage = evaluator.evaluate(novel_submission, novel_truth)
    equivalence_request = next(
        item
        for item in equivalence_stage.judge_requests
        if item.phase is ReviewEvaluationPhase.EXPECTED_ASSIGNMENT
    )
    equivalence_result = _run_scripted_judge(
        equivalence_request,
        evaluator.evaluator_execution,
        relation="different",
        score_ppm=900_000,
        severity_assessment="consistent",
        actionability="actionable",
    )
    novel_stage = evaluator.evaluate(
        novel_submission,
        novel_truth,
        judge_results=(equivalence_result,),
    )
    novel_request = next(
        item.request
        for item in novel_stage.judge_requests
        if item.phase is ReviewEvaluationPhase.NOVEL_FACTUALITY
    )
    assert HUNK_A not in novel_request.to_json()
    assert all(
        item.source_kind != SOURCE_KIND
        for item in novel_request.reference_bindings
    )

    exact_finding = _finding(
        "finding-support-stage",
        "The exact supported truth claim.",
        evidence_refs=("evidence-support-stage",),
    )
    support_truth = _truth("truth-stage", exact_finding.claim)
    evidence = _file_evidence(
        harness,
        evidence_id="evidence-support-stage",
        from_line=1,
        to_line=1,
        excerpt="alpha\n",
    )
    support_stage = evaluator.evaluate(
        _submission(harness, exact_finding, evidence=(evidence,)),
        support_truth,
    )
    support_request = next(
        item.request
        for item in support_stage.judge_requests
        if item.phase is ReviewEvaluationPhase.EVIDENCE_SUPPORT
    )
    assert HUNK_A not in support_request.to_json()
    assert all(
        item.source_kind != SOURCE_KIND
        for item in support_request.reference_bindings
    )


def test_context_truth_selector_outside_runtime_truth_fails_closed(
    harness: object,
) -> None:
    evaluator = _evaluator(
        harness,
        review_context=_review_context(
            _truth_context(
                "truth-outside",
                HUNK_A,
                provenance=_provenance("outside"),
            )
        ),
    )
    finding = _finding("finding-selector", "A semantic defect claim.")

    with pytest.raises(ValueError, match="evaluator context.*truth"):
        evaluator.evaluate(
            _submission(harness, finding),
            _truth("truth-inside", "The in-scope truth claim."),
        )


def test_truth_context_public_schema_rejects_empty_allowed_tasks() -> None:
    with pytest.raises(ValueError, match="allowed_tasks must be non-empty"):
        TruthEvaluatorContext(
            truth_id="truth-task",
            allowed_tasks=(),
            sources=(),
        )


def test_hydration_rejects_a_different_evaluator_context(
    harness: object,
) -> None:
    finding = _finding("finding-replay-context", "The exact truth claim.")
    truth = _truth("truth-replay-context", finding.claim)
    submission = _submission(harness, finding)
    context_one = _review_context(
        _truth_context(
            "truth-replay-context",
            HUNK_A,
            provenance=_provenance("replay-one"),
        )
    )
    context_two = _review_context(
        _truth_context(
            "truth-replay-context",
            HUNK_A,
            provenance=_provenance("replay-two"),
        )
    )
    evaluator_one = _evaluator(harness, review_context=context_one)
    evaluator_two = _evaluator(harness, review_context=context_two)
    result = evaluator_one.evaluate(submission, truth)

    assert HUNK_A not in result.to_json()
    with pytest.raises(ValueError, match="persisted.*replay"):
        ReviewEvaluationResult.from_dict(
            result.to_dict(),
            submission=submission,
            review_truth=truth,
            evaluator=evaluator_two,
            judge_results=(),
        )


def test_evaluator_context_merges_with_manual_pair_context_canonically(
    harness: object,
) -> None:
    source = _diff_source(HUNK_A, provenance=_provenance("manual-merge"))
    context = _review_context(
        TruthEvaluatorContext(
            truth_id="truth-merge",
            allowed_tasks=(EvaluatorContextTask.FINDING_EQUIVALENCE,),
            sources=(source,),
        )
    )
    manual = repository_context(
        source_id="manual-scoped-code",
        kind=JudgeContextKind.CODE,
        content="return value",
        revision="head",
        path="src/app.py",
    )
    bundle = ReviewContextBundle.create(
        pair_entries=(
            ReviewPairContextEntry.create(
                "finding-merge",
                ReviewTruthKind.EXPECTED,
                "truth-merge",
                (manual,),
            ),
        )
    )
    evaluator = _evaluator(
        harness,
        review_context=context,
        context_bundle=bundle,
    )
    finding = _finding("finding-merge", "A semantic defect claim.")
    request = evaluator.evaluate(
        _submission(harness, finding),
        _truth("truth-merge", "The truth claim."),
    ).judge_requests[0].request

    context_bindings = {
        item.model_ref: item
        for item in request.reference_bindings
        if item.model_ref.startswith("ctx-")
    }
    compiled_order = [
        (
            block.kind.value,
            context_bindings[block.ref_id].source_kind,
            context_bindings[block.ref_id].source_id,
            context_bindings[block.ref_id].source_digest,
        )
        for block in request.contexts
    ]
    assert compiled_order == sorted(compiled_order)
    assert {item.source_kind for item in context_bindings.values()} == {
        "repository_context",
        SOURCE_KIND,
    }

    generated_source_id = stable_id(
        SOURCE_ID_KIND,
        "truth-merge",
        source.content_sha256,
        source.provenance.digest(),
    )
    conflicting_manual = repository_context(
        source_id=generated_source_id,
        kind=JudgeContextKind.CODE,
        content="conflicting manual source",
    )
    conflicting_bundle = ReviewContextBundle.create(
        pair_entries=(
            ReviewPairContextEntry.create(
                "finding-merge",
                ReviewTruthKind.EXPECTED,
                "truth-merge",
                (conflicting_manual,),
            ),
        )
    )
    conflicting_evaluator = _evaluator(
        harness,
        review_context=context,
        context_bundle=conflicting_bundle,
    )
    with pytest.raises(ValueError, match="conflicting source identity"):
        conflicting_evaluator.evaluate(
            _submission(harness, finding),
            _truth("truth-merge", "The truth claim."),
        )


def test_review_evaluator_context_defaults_to_empty(harness: object) -> None:
    evaluator = _evaluator(harness)

    assert evaluator.review_evaluator_context == ReviewEvaluatorContext(
        truth_contexts=()
    )
