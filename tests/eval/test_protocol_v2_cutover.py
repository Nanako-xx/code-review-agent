from __future__ import annotations

import json

import pytest

from review_agent_eval.evidence_checker import EvidenceIntegrity, EvidenceReasonCode
from review_agent_eval.models import EvalInput, SchemaError, SubmissionEvidence

from .test_frozen_evidence import (
    FrozenHarness,
    _checker,
    _evidence,
    frozen_harness,
)


BASE = "a" * 40
HEAD = "b" * 40
DIGEST = "c" * 64


def repository() -> dict:
    return {
        "source": "fixture",
        "path": "fixtures/repo",
        "url": None,
        "base_revision": BASE,
        "head_revision": HEAD,
    }


def review_request() -> dict:
    return {
        "title": "Review",
        "description": None,
        "user_intent": None,
        "review_focus": None,
        "linked_requirements": [],
        "project_rules": [],
        "existing_ci_evidence": [],
    }


def test_v1_eval_input_root_is_rejected_before_nested_hydration() -> None:
    raw = json.dumps({"schema_version": "eval_input_v1"})

    with pytest.raises(SchemaError) as exc_info:
        EvalInput.from_json(raw)

    error = exc_info.value
    assert type(error).__name__ == "UnsupportedProtocolVersionError"
    assert error.code == "unsupported_protocol_version"
    assert error.expected == "eval_input_v2"
    assert error.actual == "eval_input_v1"


def test_repository_target_rejects_frozen_only_field_as_exact_key_violation() -> None:
    payload = {
        "schema_version": "eval_input_v2",
        "task_id": "task-001",
        "review_target": {
            "kind": "repository",
            "repository": repository(),
            "review_request": review_request(),
            "bundle_id": "bundle-001",
        },
    }

    with pytest.raises(
        SchemaError, match=r"review_target.*unknown field\(s\): bundle_id"
    ):
        EvalInput.from_dict(payload)


def test_frozen_target_rejects_repository_branch_fields() -> None:
    payload = {
        "schema_version": "eval_input_v2",
        "task_id": "task-001",
        "review_target": {
            "kind": "frozen_context",
            "bundle_id": "bundle-001",
            "record_id": "record-001",
            "context_format": "rendered_text",
            "rendered_sha256": DIGEST,
            "rendered_utf8_bytes": 12,
            "source_binding_digest": "d" * 64,
            "repository": repository(),
            "review_request": review_request(),
        },
    }

    with pytest.raises(
        SchemaError,
        match=r"review_target.*unknown field\(s\): repository, review_request",
    ):
        EvalInput.from_dict(payload)


def test_evidence_source_kind_shape_mixing_is_rejected() -> None:
    payload = {
        "evidence_id": "evidence-001",
        "source": {
            "kind": "repository_file",
            "target_materialization_id": "materialization-001",
            "revision": HEAD,
            "path": "src/app.py",
            "from_line": 1,
            "to_line": 2,
            "command": ["pytest"],
        },
        "content_hash": DIGEST,
        "excerpt": "evidence",
    }

    with pytest.raises(
        SchemaError, match=r"evidence.source.*unknown field\(s\): command"
    ):
        SubmissionEvidence.from_dict(payload)


def test_frozen_evidence_rejects_content_hash_drift(
    frozen_harness: FrozenHarness,
) -> None:
    result = _checker(frozen_harness).check_item(
        _evidence(frozen_harness, content_hash="0" * 64)
    )

    assert result.integrity is EvidenceIntegrity.INVALID
    assert {item.reason_code for item in result.diagnostics} == {
        EvidenceReasonCode.CONTENT_HASH_MISMATCH
    }
