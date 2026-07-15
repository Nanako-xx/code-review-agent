from pathlib import Path
import errno
import json

import pytest

import review_agent.checkpoint as checkpoint_module
from review_agent.brief import build_memory_audit_projection
from review_agent.checkpoint import CheckpointStore
from review_agent.models import IntentStatus, ReviewerResult, ReviewerResultStatus, RiskAssessment, RiskLevel
from review_agent.reporting import render_markdown_report
from review_agent.run_state import initial_run_state


def test_checkpoint_store_writes_json_and_jsonl(tmp_path: Path):
    store = CheckpointStore(tmp_path, review_id="review-1")

    store.write_json("request.json", {"base": "main", "head": "HEAD"})
    store.append_jsonl("observations.jsonl", {"observation_id": "O-1", "summary": "changed app.py"})

    assert json.loads((tmp_path / ".review-agent" / "runs" / "review-1" / "request.json").read_text(encoding="utf-8")) == {
        "base": "main",
        "head": "HEAD",
    }
    assert "O-1" in (tmp_path / ".review-agent" / "runs" / "review-1" / "observations.jsonl").read_text(
        encoding="utf-8"
    )


def test_checkpoint_store_serializes_enum_values(tmp_path: Path):
    store = CheckpointStore(tmp_path, review_id="review-1")

    store.write_json("intent.json", {"status": IntentStatus.PARTIAL})

    assert json.loads((tmp_path / ".review-agent" / "runs" / "review-1" / "intent.json").read_text(encoding="utf-8")) == {
        "status": "partial",
    }


def test_checkpoint_store_writes_and_reads_run_state(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, "review-1")
    state = initial_run_state(
        review_id="review-1",
        repository_path=str(tmp_path),
        base_revision="main",
        head_revision="HEAD",
    )

    store.write_state(state)

    assert store.read_state() == state


def test_checkpoint_store_json_write_uses_unique_same_directory_temps_and_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = CheckpointStore(tmp_path, "review-atomic")
    temporary_paths: list[Path] = []
    fsync_calls: list[int] = []
    real_replace = checkpoint_module.os.replace
    real_fsync = checkpoint_module.os.fsync

    def recording_replace(source: object, destination: object) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        assert source_path.parent == destination_path.parent
        assert source_path.exists()
        temporary_paths.append(source_path)
        real_replace(source, destination)

    def recording_fsync(file_descriptor: int) -> None:
        fsync_calls.append(file_descriptor)
        real_fsync(file_descriptor)

    monkeypatch.setattr(checkpoint_module.os, "replace", recording_replace)
    monkeypatch.setattr(checkpoint_module.os, "fsync", recording_fsync)

    store.write_json("request.json", {"head": "HEAD"})
    store.write_json("request.json", {"head": "feature"})

    assert len(temporary_paths) == 2
    assert temporary_paths[0] != temporary_paths[1]
    assert all(path.name.endswith(".tmp") for path in temporary_paths)
    assert all(not path.exists() for path in temporary_paths)
    assert len(fsync_calls) >= 2
    assert json.loads((store.run_dir / "request.json").read_text(encoding="utf-8")) == {
        "head": "feature"
    }


def test_checkpoint_store_cleans_temp_and_preserves_destination_when_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = CheckpointStore(tmp_path, "review-atomic-failure")
    destination = store.run_dir / "request.json"
    destination.write_text('{"head":"old"}', encoding="utf-8")
    temporary_paths: list[Path] = []

    def failing_replace(source: object, destination_path: object) -> None:
        temporary_paths.append(Path(source))
        raise OSError("replace failed")

    monkeypatch.setattr(checkpoint_module.os, "replace", failing_replace)

    with pytest.raises(OSError, match="replace failed"):
        store.write_json("request.json", {"head": "new"})

    assert json.loads(destination.read_text(encoding="utf-8")) == {"head": "old"}
    assert len(temporary_paths) == 1
    assert not temporary_paths[0].exists()
    assert [path for path in store.run_dir.iterdir() if path.name.endswith(".tmp")] == []


def test_checkpoint_store_fsyncs_parent_directory_after_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = CheckpointStore(tmp_path, "review-directory-fsync")
    directory_descriptor = 987654
    opened_directories: list[Path] = []
    fsync_calls: list[int] = []
    closed_descriptors: list[int] = []
    real_fsync = checkpoint_module.os.fsync

    def opening_directory(path: object, flags: int) -> int:
        opened_directories.append(Path(path))
        return directory_descriptor

    def recording_fsync(file_descriptor: int) -> None:
        fsync_calls.append(file_descriptor)
        if file_descriptor != directory_descriptor:
            real_fsync(file_descriptor)

    def recording_close(file_descriptor: int) -> None:
        closed_descriptors.append(file_descriptor)

    monkeypatch.setattr(checkpoint_module.os, "open", opening_directory)
    monkeypatch.setattr(checkpoint_module.os, "fsync", recording_fsync)
    monkeypatch.setattr(checkpoint_module.os, "close", recording_close)

    store.write_json("request.json", {"head": "HEAD"})

    assert opened_directories == [store.run_dir]
    assert fsync_calls[-1] == directory_descriptor
    assert closed_descriptors == [directory_descriptor]


def test_checkpoint_store_safely_degrades_when_directory_fsync_is_unsupported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = CheckpointStore(tmp_path, "review-directory-fsync-unsupported")

    def unsupported_directory_open(path: object, flags: int) -> int:
        raise OSError(errno.EACCES, "directory handles are unsupported")

    monkeypatch.setattr(checkpoint_module.os, "open", unsupported_directory_open)

    store.write_json("request.json", {"head": "HEAD"})

    assert json.loads((store.run_dir / "request.json").read_text(encoding="utf-8")) == {
        "head": "HEAD"
    }


def test_markdown_report_contains_risk_signals_and_uncertainties():
    assessment = RiskAssessment(
        level=RiskLevel.HIGH,
        dimensions={"impact": "sensitive path"},
        reasons=["sensitive path changed: auth.py"],
        signal_refs=["changed_file:auth.py"],
        uncertainties=["user did not provide explicit intent"],
        suggested_focus=["regression safety"],
    )

    report = render_markdown_report(
        review_id="review-1",
        base_revision="base",
        head_revision="head",
        risk_assessment=assessment,
        changed_files=["auth.py"],
    )

    assert "# Review Brief" in report
    assert "Risk level: high" in report
    assert "## Initial And Final Risk Assessment" in report
    assert "- changed_file:auth.py" in report
    assert "## Uncertainties" in report
    assert "- user did not provide explicit intent" in report


def test_markdown_report_includes_single_reviewer_result():
    assessment = RiskAssessment(
        level=RiskLevel.LOW,
        dimensions={"impact": "local"},
        reasons=["small change"],
        signal_refs=[],
        uncertainties=[],
        suggested_focus=["intent alignment"],
    )
    reviewer_result = ReviewerResult(
        investigation_summary="Fake reviewer executed.",
        status=ReviewerResultStatus.PARTIAL,
        uncertainties=["Fake provider does not perform semantic review."],
    )

    report = render_markdown_report(
        review_id="review-1",
        base_revision="base",
        head_revision="head",
        risk_assessment=assessment,
        changed_files=["auth.py"],
        reviewer_result=reviewer_result,
    )

    assert "## Uncertainties" in report
    assert "- Fake provider does not perform semantic review." in report


def test_markdown_report_includes_observation_summaries():
    assessment = RiskAssessment(
        level=RiskLevel.LOW,
        dimensions={"impact": "local"},
        reasons=[],
        signal_refs=[],
        uncertainties=[],
        suggested_focus=[],
    )

    report = render_markdown_report(
        review_id="review-1",
        base_revision="base",
        head_revision="head",
        risk_assessment=assessment,
        changed_files=["auth.py"],
        observation_summaries={"O-abc": "auth.py changed between base and head"},
    )

    assert "## Verification Evidence" in report
    assert "- observation:O-abc - auth.py changed between base and head" in report


def test_markdown_report_includes_repository_intelligence_summary():
    assessment = RiskAssessment(
        level=RiskLevel.LOW,
        dimensions={},
        reasons=[],
        signal_refs=[],
        uncertainties=[],
        suggested_focus=[],
    )

    report = render_markdown_report(
        review_id="review-1",
        base_revision="base",
        head_revision="head",
        risk_assessment=assessment,
        changed_files=["app.py"],
        repository_intelligence_summary="Repository Intelligence\n- modified function add app.py:1-2",
    )

    assert "## Change Map And Repository Impact" in report
    assert "Repository intelligence:" in report
    assert "modified function add app.py:1-2" in report


def test_markdown_report_includes_multi_reviewer_summary():
    assessment = RiskAssessment(
        level=RiskLevel.MEDIUM,
        dimensions={},
        reasons=[],
        signal_refs=[],
        uncertainties=[],
        suggested_focus=[],
    )

    report = render_markdown_report(
        review_id="review-1",
        base_revision="base",
        head_revision="head",
        risk_assessment=assessment,
        changed_files=["auth.py"],
        multi_reviewer_summary={
            "reviewer_count": 2,
            "status_counts": {"partial": 2},
            "roles": ["Core Reviewer", "Adversarial Reviewer"],
        },
    )

    assert "## Change Map And Repository Impact" in report
    assert "reviewer_count: 2" in report
    assert "roles: Core Reviewer, Adversarial Reviewer" in report


def test_markdown_report_includes_reconciliation_and_completion_sections():
    assessment = RiskAssessment(
        level=RiskLevel.HIGH,
        dimensions={},
        reasons=[],
        signal_refs=[],
        uncertainties=[],
        suggested_focus=[],
    )

    report = render_markdown_report(
        review_id="review-1",
        base_revision="base",
        head_revision="head",
        risk_assessment=assessment,
        changed_files=["auth.py"],
        reconciliation_summary={
            "canonical_count": 1,
            "rejected_count": 0,
            "evidence_quality": "verified",
        },
        completion_summary={
            "status": "completed",
            "recommendation": "needs_work",
            "blockers": [],
            "uncertainties": [],
            "missing_perspectives": [],
        },
    )

    assert "## Verified Findings" in report
    assert "## Rejected Hypotheses" in report
    assert "## Non-Binding Recommendation" in report
    assert "Needs work before merge." in report


def test_markdown_report_accepts_structured_memory_audit_summary() -> None:
    memory_id = "MEM-" + "a" * 64
    report = render_markdown_report(
        review_id="review-memory-report",
        base_revision="base",
        head_revision="head",
        risk_assessment=RiskAssessment(
            level=RiskLevel.MEDIUM,
            dimensions={},
            reasons=[],
            signal_refs=[],
            uncertainties=[],
            suggested_focus=[],
        ),
        changed_files=["auth.py"],
        memory_audit_summary={
            "applied_memory": [
                {
                    "memory_id": memory_id,
                    "kind": "business_invariant",
                    "statement": "Authentication remains mandatory.",
                    "scope": {"paths": ["auth.py"]},
                    "authority": "human_approved_context",
                    "source_refs": [
                        {"type": "git_commit", "commit_sha": "b" * 40}
                    ],
                    "validity": {
                        "applicability": "selected",
                        "valid_from_sha": "a" * 40,
                        "policies": ["manual_until_revoked"],
                    },
                }
            ],
            "status": {
                "mode": "read",
                "available": True,
                "curator": {"status": "disabled", "mode": "disabled"},
            },
        },
    )

    assert "## Memory Audit" in report
    assert memory_id in report
    assert "human_approved_context" in report
    assert "curator status: disabled" in report
    assert "### Memory Records Not Applied" in report
    assert "reason=record_status_missing" in report


def test_memory_json_and_markdown_keep_pending_cache_and_curator_semantics() -> None:
    candidate_id = "MC-" + "a" * 64
    entry_id = "RKE-" + "b" * 64
    memory = {
        "cache_provenance": [
            {
                "status": "rebuild",
                "entry_id": entry_id,
                "corruption_reason": "hash_mismatch",
            }
        ],
        "pending_candidates": [
            {
                "candidate_id": candidate_id,
                "kind": "business_invariant",
                "statement": "Authentication remains mandatory.",
                "scope": {
                    "paths": ["auth/**"],
                    "symbols": ["Auth.check"],
                    "contracts": ["behavioral-correctness"],
                    "languages": ["python"],
                },
                "status": "pending_approval",
            }
        ],
        "status": {
            "mode": "read-write",
            "available": True,
            "curator": {
                "mode": "model",
                "status": "proposed",
                "outcome": "proposed",
                "attempt_count": 2,
                "candidate_ids": [candidate_id],
                "review_conclusion_impact": "none",
            },
        },
    }
    audit = build_memory_audit_projection(memory)
    report = render_markdown_report(
        review_id="review-memory-parity",
        base_revision="base",
        head_revision="head",
        risk_assessment=RiskAssessment(
            level=RiskLevel.MEDIUM,
            dimensions={},
            reasons=[],
            signal_refs=[],
            uncertainties=[],
            suggested_focus=[],
        ),
        changed_files=["auth/check.py"],
        memory_audit_summary=memory,
    )

    assert audit["pending_candidates"][0]["scope"] == {
        "paths": ["auth/**"],
        "symbols": ["Auth.check"],
        "contracts": ["behavioral-correctness"],
        "languages": ["python"],
    }
    assert audit["cache_provenance"][0]["corruption_reason"] == "hash_mismatch"
    assert audit["status"]["curator"] == {
        "mode": "model",
        "outcome": "proposed",
        "status": "proposed",
        "attempt_count": 2,
        "review_conclusion_impact": "none",
        "candidate_ids": [candidate_id],
    }
    for expected in (
        "Scope: paths=['auth/**']",
        "symbols=['Auth.check']",
        "corruption_reason=hash_mismatch",
        "curator outcome: proposed",
        "curator attempt_count: 2",
        f"curator candidate_ids: {candidate_id}",
        "curator review_conclusion_impact: none",
    ):
        assert expected in report


def test_pipeline_import_smoke_has_no_reporting_cycle() -> None:
    from review_agent.pipeline import ReviewPipeline

    assert ReviewPipeline.__name__ == "ReviewPipeline"
