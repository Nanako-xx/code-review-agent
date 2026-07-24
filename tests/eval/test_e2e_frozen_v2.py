from __future__ import annotations

import hashlib
import json
import shutil
import socket
import sys
from pathlib import Path
from typing import Any, Iterable

import pytest

import review_agent_eval.cli as cli_module
from review_agent_eval.adapters._public import (
    PUBLIC_FILTER_MANIFEST_SCHEMA_VERSION,
    PublicFilterManifest,
    PublicSelector,
    PublicSourceManifest,
    read_public_preparation_receipt,
)
from review_agent_eval.adapters.current_agent import CurrentAgentAdapter
from review_agent_eval.adapters.subprocess_agent import (
    SUBPROCESS_JSON_ADAPTER_KIND,
    SubprocessAgentAdapter,
    subprocess_adapter_capabilities,
)
from review_agent_eval.adapters.swe_prbench import (
    SWE_PRBENCH_DATASET_ID,
    SWE_PRBENCH_FROZEN_PROTOCOL_ID,
    SWE_PRBENCH_FROZEN_SUITE_RELATIVE_ROOT,
    SWE_PRBENCH_HARNESS_LICENSE,
    SWE_PRBENCH_HARNESS_REVISION,
    SWE_PRBENCH_PIPELINE_VERSION,
    SWE_PRBENCH_PROTOCOL_FROZEN,
    SWE_PRBENCH_SOURCE_PROFILE_FIXTURE,
    read_swe_prbench_frozen_bundle,
)
from review_agent_eval.artifacts import ArtifactStore
from review_agent_eval.cli import EXIT_OK, EXIT_PRECONDITION, main
from review_agent_eval.config import AgentConfigSnapshot
from review_agent_eval.datasets import CaseBank
from review_agent_eval.frozen_context import (
    FROZEN_CONTEXT_TARGET_PATH,
    frozen_bundle_trust_digest,
)
from review_agent_eval.models import (
    EvidenceIntegrity,
    FrozenContextReviewTarget,
    ReviewTargetKind,
    canonical_json_bytes,
    canonical_sha256,
)
from review_agent_eval.target_replay import FrozenContextReplayResolver


FIXTURE_ROOT = (
    Path(__file__).parent / "fixtures" / "public_datasets" / "swe_prbench"
)

_FROZEN_AGENT_PROGRAM = r'''
import hashlib
import json
from pathlib import Path
import sys


raw = sys.stdin.buffer.read()
invocation = json.loads(raw.decode("utf-8"))
target = invocation["eval_input"]["review_target"]
binding = invocation["trial_binding"]
access = invocation["target_access"]
materialization_id = invocation["materialization_id"]

assert target["kind"] == "frozen_context"
assert access["target_materialization_id"] == materialization_id
assert access["readable_relative_paths"] == ["target/context.txt"]
context = Path("target/context.txt").read_bytes()
first_line = context.splitlines(keepends=True)[0]
evidence_id = "evidence-frozen-e2e"

submission = {
    "schema_version": "eval_submission_v2",
    "task_id": binding["task_id"],
    "agent_id": sys.argv[1],
    "trial_id": binding["trial_id"],
    "eval_input_digest": binding["eval_input_digest"],
    "target_materialization_id": materialization_id,
    "status": "completed",
    "intent": {
        "status": "sufficient",
        "goal": "Review the supplied frozen context.",
        "acceptance_criteria": [],
        "scope": [],
        "constraints": [],
        "claims": [],
        "clarification_questions": [],
        "uncertainties": [],
    },
    "review": {
        "findings": [
            {
                "finding_id": "finding-frozen-e2e",
                "claim": "The frozen context contains a review-relevant defect.",
                "severity": "high",
                "path": None,
                "side": None,
                "from_line": None,
                "to_line": None,
                "evidence_refs": [evidence_id],
                "suggested_action": "Correct the behavior described by the context.",
            }
        ],
        "uncertainties": [],
    },
    "evidence": [
        {
            "evidence_id": evidence_id,
            "source": {
                "kind": "frozen_context",
                "target_materialization_id": materialization_id,
                "context_ref": target["record_id"],
                "from_line": 1,
                "to_line": 1,
            },
            "content_hash": hashlib.sha256(first_line).hexdigest(),
            "excerpt": first_line.decode("utf-8"),
        }
    ],
    "usage": {
        "elapsed_seconds": 0.01,
        "input_tokens": 2,
        "output_tokens": 3,
        "total_tokens": 5,
        "tool_calls": 1,
        "cost_amount": None,
        "cost_currency": None,
    },
    "trace_ref": None,
    "failure": None,
}
sys.stdout.write(
    json.dumps(
        submission,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
)
'''


def _copy_source(tmp_path: Path) -> tuple[Path, PublicSourceManifest]:
    source = tmp_path / "source"
    shutil.copytree(FIXTURE_ROOT, source)
    manifest_path = source / "source_manifest.json"
    manifest = PublicSourceManifest.from_json(manifest_path.read_bytes())
    manifest_path.write_bytes(canonical_json_bytes(manifest.to_dict()))
    return source, manifest


def _write_filter(
    path: Path, selectors: Iterable[PublicSelector]
) -> PublicFilterManifest:
    manifest = PublicFilterManifest(
        schema_version=PUBLIC_FILTER_MANIFEST_SCHEMA_VERSION,
        dataset_id=SWE_PRBENCH_DATASET_ID,
        selectors=tuple(selectors),
    )
    path.write_bytes(canonical_json_bytes(manifest.to_dict()))
    return manifest


def _frozen_filter(path: Path) -> PublicFilterManifest:
    return _write_filter(
        path,
        (
            PublicSelector("source_scope", ("fixture",)),
            PublicSelector(
                "source_profile", (SWE_PRBENCH_SOURCE_PROFILE_FIXTURE,)
            ),
            PublicSelector("source_format", ("raw_jsonl",)),
            PublicSelector("protocol", (SWE_PRBENCH_PROTOCOL_FROZEN,)),
            PublicSelector("context_config", ("config_A",)),
            PublicSelector(
                "harness_revision", (SWE_PRBENCH_HARNESS_REVISION,)
            ),
            PublicSelector("harness_license", (SWE_PRBENCH_HARNESS_LICENSE,)),
            PublicSelector("pipeline_version", (SWE_PRBENCH_PIPELINE_VERSION,)),
        ),
    )


def _invoke_json(
    capsys: pytest.CaptureFixture[str], arguments: list[str]
) -> tuple[int, dict[str, Any]]:
    code = main(arguments)
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out
    return code, json.loads(captured.out)


def _root_arguments(suite_root: Path, root: Path) -> list[str]:
    root.mkdir(parents=True, exist_ok=True)
    return [
        "--suite-root",
        str(suite_root),
        "--runs-root",
        str(root / ".eval-runs"),
        "--data-root",
        str(root / ".eval-data"),
        "--workspace-root",
        str(root / ".eval-workspaces"),
    ]


def _write_frozen_agent_config(tmp_path: Path) -> Path:
    program = tmp_path / "frozen-subprocess-agent.py"
    program.write_text(_FROZEN_AGENT_PROGRAM, encoding="utf-8")
    capabilities = subprocess_adapter_capabilities()
    adapter = {
        "kind": SUBPROCESS_JSON_ADAPTER_KIND,
        "command": [
            str(Path(sys.executable).resolve()),
            str(program.resolve()),
            "{agent_id}",
            "{task_id}",
            "{trial_id}",
        ],
        "environment_allowlist": [],
        "capabilities": capabilities.to_dict(),
    }
    parameters = {"adapter": adapter}
    snapshot = AgentConfigSnapshot(
        agent_id="task-5b-frozen-agent",
        agent_name="Task 5B frozen subprocess fixture",
        agent_version="1",
        commit="b" * 40,
        model="fixture",
        provider="subprocess",
        parameters=parameters,
        prompt_config_digest=canonical_sha256(parameters),
    )
    path = tmp_path / "frozen-agent-config.json"
    path.write_bytes(canonical_json_bytes(snapshot.to_dict()))
    return path


def test_frozen_public_suite_cli_lifecycle_binds_one_canonical_identity_chain(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, source_manifest = _copy_source(tmp_path)
    filter_path = tmp_path / "frozen-filter.json"
    filter_manifest = _frozen_filter(filter_path)
    suite_root = tmp_path / "frozen-suite"

    code, public_result = _invoke_json(
        capsys,
        [
            "prepare-public",
            "--mode",
            "local-import",
            "--dataset",
            "swe-prbench",
            "--source-root",
            str(source),
            "--source-manifest",
            str(source / "source_manifest.json"),
            "--expected-source-manifest-digest",
            source_manifest.digest(),
            "--filter-manifest",
            str(filter_path),
            "--expected-profile-digest",
            filter_manifest.digest(),
            "--output-root",
            str(suite_root),
            "--json",
        ],
    )
    assert code == EXIT_OK
    assert public_result["protocol"] == SWE_PRBENCH_PROTOCOL_FROZEN
    assert public_result["protocol_id"] == SWE_PRBENCH_FROZEN_PROTOCOL_ID
    assert public_result["target_kind"] == ReviewTargetKind.FROZEN_CONTEXT.value

    receipt = read_public_preparation_receipt(
        suite_root,
        expected_source_manifest_digest=source_manifest.digest(),
        expected_preparation_packet_digest=public_result[
            "preparation_packet_digest"
        ],
        expected_suite_manifest_digest=public_result["suite_manifest_digest"],
    )
    bank = CaseBank.open(
        suite_root,
        expected_manifest_digest=public_result["suite_manifest_digest"],
    )
    bank.verify()
    bundle_root = suite_root / SWE_PRBENCH_FROZEN_SUITE_RELATIVE_ROOT
    bundle = read_swe_prbench_frozen_bundle(
        bundle_root,
        expected_bundle_id=public_result["bundle_id"],
    )
    assert len(bank.manifest.cases) == len(bundle.manifest.records) == 1
    suite_case = bank.manifest.cases[0]
    eval_case = bank.evaluator_case(suite_case.task_id)
    eval_input = eval_case.eval_input()
    target = eval_input.review_target
    assert isinstance(target, FrozenContextReviewTarget)
    record = bundle.manifest.records[0]
    preparation = bank.manifest.source.preparation_binding
    assert preparation is not None

    assert public_result["source_manifest_digest"] == source_manifest.digest()
    assert (
        source_manifest.digest()
        == receipt.source_manifest_digest
        == bundle.manifest.source_manifest_digest
        == preparation.source_manifest_digest
    )
    assert public_result["filter_manifest_digest"] == filter_manifest.digest()
    assert (
        filter_manifest.digest()
        == receipt.filter_manifest_digest
        == bundle.manifest.filter_manifest_digest
        == preparation.filter_manifest_digest
    )
    assert receipt.source_manifest == bundle.manifest.source_manifest
    assert receipt.filter_manifest == bundle.manifest.filter_manifest
    assert public_result["suite_manifest_digest"] == receipt.suite_manifest_digest
    assert receipt.suite_manifest_digest == bank.manifest_digest
    assert bank.manifest.source.content_hash == public_result[
        "preparation_packet_digest"
    ]
    assert public_result["preparation_packet_digest"] == (
        preparation.preparation_packet_digest
    )
    assert preparation.repository_catalog_digest is None
    assert preparation.frozen_bundle_trust_digest == frozen_bundle_trust_digest(
        bundle, preparation
    )
    assert target.bundle_id == bundle.manifest.bundle_id == public_result["bundle_id"]
    assert target.record_id == record.task_id == eval_input.task_id
    assert target.rendered_sha256 == record.rendered_sha256
    assert target.rendered_utf8_bytes == record.rendered_utf8_bytes
    assert suite_case.protocol_id == SWE_PRBENCH_FROZEN_PROTOCOL_ID
    assert suite_case.eval_input_digest == eval_input.digest()
    assert suite_case.canonical_case_digest == eval_case.digest()

    def no_network(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("Task 5B E2E attempted network access")

    def no_repository_preparer(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("Frozen E2E constructed a Repository preparer/Catalog")

    current_agent_calls = 0

    def forbidden_current_agent(
        _self: CurrentAgentAdapter,
        _eval_input: Any,
        _workspace: Path,
        _config: Any,
        _clarification_channel: Any,
        *,
        target_access: Any,
        target_materialization_id: str,
        cancel_event: Any = None,
    ) -> object:
        del target_access, target_materialization_id, cancel_event
        nonlocal current_agent_calls
        current_agent_calls += 1
        raise AssertionError("current-agent was invoked for a Frozen Target")

    monkeypatch.setattr(socket, "socket", no_network)
    monkeypatch.setattr(cli_module, "_repository_preparer", no_repository_preparer)
    monkeypatch.setattr(CurrentAgentAdapter, "run", forbidden_current_agent)

    current_root = tmp_path / "current"
    current_arguments = [
        "prepare",
        *_root_arguments(suite_root, current_root),
        "--run-instance-key",
        "task-5b-current-frozen-incompatible",
        "--agent-adapter",
        "current",
        "--json",
    ]
    current_results = [
        _invoke_json(capsys, current_arguments) for _index in range(2)
    ]
    assert current_results[0] == current_results[1]
    current_code, current_payload = current_results[0]
    assert current_code == EXIT_PRECONDITION, current_payload
    assert current_payload == {
        "schema_version": "review_agent_eval_cli_v1",
        "command": "cli",
        "status": "error",
        "error_code": "precondition",
        "message": "RunIncompatibilityError",
    }
    assert current_agent_calls == 0
    current_runs_root = current_root / ".eval-runs"
    assert not current_runs_root.exists() or not any(
        item.name.startswith("run-") for item in current_runs_root.iterdir()
    )

    agent_config = _write_frozen_agent_config(tmp_path)
    capable_root = tmp_path / "capable"
    common = _root_arguments(suite_root, capable_root)
    subprocess_calls: list[dict[str, Any]] = []
    original_subprocess_run = SubprocessAgentAdapter.run

    def recording_subprocess_run(
        self: SubprocessAgentAdapter,
        current_eval_input: Any,
        workspace: Path,
        config: Any,
        clarification_channel: Any,
        *,
        target_access: Any,
        target_materialization_id: str,
        cancel_event: Any = None,
    ) -> Any:
        subprocess_calls.append(
            {
                "target_kind": current_eval_input.review_target.kind,
                "context_sha256": hashlib.sha256(
                    (workspace / FROZEN_CONTEXT_TARGET_PATH).read_bytes()
                ).hexdigest(),
                "target_access": target_access,
                "target_materialization_id": target_materialization_id,
            }
        )
        return original_subprocess_run(
            self,
            current_eval_input,
            workspace,
            config,
            clarification_channel,
            target_access=target_access,
            target_materialization_id=target_materialization_id,
            cancel_event=cancel_event,
        )

    judge_calls = 0
    original_judge_for = cli_module._judge_for

    def recording_judge(args: Any, execution: Any) -> Any:
        nonlocal judge_calls
        judge_calls += 1
        assert args.judge_provider == "fake"
        assert args.judge_model == "task-5b-fake-judge"
        return original_judge_for(args, execution)

    monkeypatch.setattr(SubprocessAgentAdapter, "run", recording_subprocess_run)
    monkeypatch.setattr(cli_module, "_judge_for", recording_judge)

    capable_prepare_arguments = [
        "prepare",
        *common,
        "--agent-config",
        str(agent_config),
        "--run-instance-key",
        "task-5b-frozen-capable",
        "--json",
    ]
    code, prepared = _invoke_json(capsys, capable_prepare_arguments)
    assert code == EXIT_OK
    assert prepared["preflight"]["compatible_task_ids"] == [
        eval_input.task_id
    ]
    assert prepared["preflight"]["incompatible_task_ids"] == []
    assert prepared["preflight"]["coverage"] == {
        "checked_trials": 1,
        "compatible_cases": 1,
        "incompatible_cases": 0,
        "issues": 0,
    }
    code, resumed = _invoke_json(
        capsys, [*capable_prepare_arguments, "--resume"]
    )
    assert code == EXIT_OK
    assert resumed["run_id"] == prepared["run_id"]
    assert resumed["resumed"] is True
    run_id = prepared["run_id"]
    runs_root = capable_root / ".eval-runs"
    store = ArtifactStore(runs_root, create_root=False)
    run_config = store.load_run_config(run_id)
    run_manifest = store.load_run_manifest(run_id)
    plan = run_manifest.trials[0]
    trial_manifest = store.load_trial_manifest(
        run_id, eval_input.task_id, plan.trial_id
    )
    assert ReviewTargetKind.FROZEN_CONTEXT in (
        run_config.adapter_capabilities.target_kinds
    )
    assert plan.task_id == trial_manifest.task_id == eval_input.task_id
    assert plan.eval_input_digest == trial_manifest.eval_input_digest == eval_input.digest()
    assert (
        plan.canonical_case_digest
        == trial_manifest.canonical_case_digest
        == eval_case.digest()
    )
    assert trial_manifest.target_kind is ReviewTargetKind.FROZEN_CONTEXT

    code, agent_result = _invoke_json(
        capsys,
        ["run-agent", run_id, *common, "--json"],
    )
    assert code == EXIT_OK, agent_result
    assert agent_result["run_status"] == "completed"
    assert agent_result["trials"] == [
        {
            "task_id": eval_input.task_id,
            "trial_id": plan.trial_id,
            "trial_index": 1,
            "status": "completed",
            "submission_status": "completed",
            "skipped": False,
        }
    ]
    assert len(subprocess_calls) == 1

    materialization = store.load_trial_materialization(
        run_id, eval_input.task_id, plan.trial_id
    )
    submission = store.load_existing_submission(
        run_id, eval_input.task_id, plan.trial_id
    )
    materialization_id = materialization.manifest.materialization_id
    assert materialization.eval_input == eval_input
    assert materialization.suite_preparation_binding == preparation
    assert materialization.manifest.suite_preparation_binding_digest == (
        preparation.digest()
    )
    assert materialization.manifest.prepared_source_id == target.bundle_id
    assert materialization.manifest.review_target_digest == target.digest()
    assert materialization.manifest.eval_input_digest == eval_input.digest()
    assert materialization.manifest.files[0].relative_path == (
        FROZEN_CONTEXT_TARGET_PATH
    )
    assert materialization.manifest.files[0].sha256 == target.rendered_sha256
    assert subprocess_calls[0]["target_kind"] is ReviewTargetKind.FROZEN_CONTEXT
    assert subprocess_calls[0]["context_sha256"] == target.rendered_sha256
    assert subprocess_calls[0]["target_materialization_id"] == materialization_id
    assert subprocess_calls[0]["target_access"] == (
        materialization.manifest.target_access
    )
    assert submission.eval_input_digest == eval_input.digest()
    assert submission.target_materialization_id == materialization_id
    assert submission.failure is None
    assert len(submission.evidence) == 1
    evidence = submission.evidence[0]
    assert evidence.source.target_materialization_id == materialization_id
    assert evidence.source.context_ref == target.record_id
    replay = FrozenContextReplayResolver(bundle_root=bundle_root).resolve(
        materialization
    )
    first_line = replay.read_lines(1, 1)
    assert evidence.excerpt.encode("utf-8") == first_line
    assert evidence.content_hash == hashlib.sha256(first_line).hexdigest()

    code, evaluated = _invoke_json(
        capsys,
        [
            "evaluate",
            run_id,
            *common,
            "--revision",
            "task-5b-frozen-v2",
            "--judge-provider",
            "fake",
            "--judge-model",
            "task-5b-fake-judge",
            "--json",
        ],
    )
    assert code == EXIT_OK, evaluated
    assert evaluated["trial_count"] == 1
    assert judge_calls == 1
    evaluation_id = evaluated["evaluation_id"]
    evaluated_trial = evaluated["trials"][0]
    evaluation = store.load_evaluation_bundle(
        run_id, eval_input.task_id, plan.trial_id, evaluation_id
    )
    assert evaluation.submission_digest == submission.digest()
    assert evaluation.canonical_case_digest == eval_case.digest()
    assert evaluation.trial_manifest_digest == trial_manifest.digest()
    assert evaluation.score["task_id"] == eval_input.task_id
    assert evaluation.score["trial_id"] == plan.trial_id
    assert evaluation.score["eval_input_digest"] == eval_input.digest()
    assert evaluation.score["canonical_case_digest"] == eval_case.digest()
    assert evaluation.score["submission_digest"] == submission.digest()
    assert canonical_sha256(evaluation.score) == evaluated_trial["score_digest"]
    assert hashlib.sha256(evaluation.report.encode("utf-8")).hexdigest() == (
        evaluated_trial["report_digest"]
    )
    evidence_results = evaluation.review_matches[
        "evidence_integrity_results"
    ]
    assert len(evidence_results) == 1
    finding = submission.review.findings[0]
    evidence_result = evidence_results[0]
    assert evidence_result["finding_id"] == finding.finding_id
    assert evidence_result["integrity"] == EvidenceIntegrity.VALID.value
    assert evidence_result["referenced_evidence_ids"] == list(
        finding.evidence_refs
    )
    assert len(evidence_result["item_results"]) == 1
    item_result = evidence_result["item_results"][0]
    assert item_result["evidence_id"] == evidence.evidence_id
    assert item_result["kind"] == evidence.source.kind.value
    assert item_result["integrity"] == EvidenceIntegrity.VALID.value
    assert evaluation.judge_output["results"]
    assert {
        item["source"] for item in evaluation.judge_output["results"]
    } == {"live"}
    judge_profiles = (
        evaluation.evaluator_execution.evaluator.judge_profiles
    )
    assert judge_profiles
    assert {profile.provider for profile in judge_profiles} == {"fake"}
    assert {profile.model for profile in judge_profiles} == {
        "task-5b-fake-judge"
    }

    code, inspected = _invoke_json(
        capsys,
        [
            "inspect",
            run_id,
            *common,
            "--task-id",
            eval_input.task_id,
            "--trial-id",
            plan.trial_id,
            "--evaluation-id",
            evaluation_id,
            "--format",
            "json",
        ],
    )
    assert code == EXIT_OK
    inspection = inspected["inspection"]
    assert inspection["source_bindings"] == {
        "run_id": run_id,
        "task_id": eval_input.task_id,
        "trial_id": plan.trial_id,
        "evaluation_id": evaluation_id,
        "evaluation_revision": "task-5b-frozen-v2",
        "evaluator_execution_digest": evaluation.evaluator_execution.digest(),
        "submission_digest": submission.digest(),
        "canonical_case_digest": eval_case.digest(),
        "eval_input_digest": eval_input.digest(),
        "trial_manifest_digest": trial_manifest.digest(),
    }
    assert inspection["submission"]["status"] == "completed"
    assert inspection["submission"]["finding_count"] == 1
    assert inspection["submission"]["evidence_count"] == 1
    assert inspection["review_evaluation"][
        "evidence_integrity_result_count"
    ] == 1
    assert inspection["score"]["artifact_digest"] == evaluated_trial[
        "score_digest"
    ]
    assert inspection["report"] == {
        "available": True,
        "sha256": hashlib.sha256(evaluation.report.encode("utf-8")).hexdigest(),
    }
