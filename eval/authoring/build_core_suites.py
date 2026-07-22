from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import stat
import sys
import tempfile
import unicodedata
from types import MappingProxyType
from typing import Any, Iterable, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
AUTHORING_ROOT = Path(__file__).resolve().parent
if str(AUTHORING_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTHORING_ROOT))

from review_agent_eval.cases import (  # noqa: E402
    REPOSITORY_MATERIALIZER_PROTOCOL,
    SUITE_MANIFEST_SCHEMA_VERSION,
    SuiteManifest,
)
from review_agent_eval.models import (  # noqa: E402
    EVAL_CASE_SCHEMA_VERSION,
    EVAL_INPUT_SCHEMA_VERSION,
    EVAL_SUBMISSION_SCHEMA_VERSION,
    EvalCase,
    EvalSubmission,
    canonical_json,
    canonical_sha256,
    stable_id,
)
from review_agent_eval.repository import FixtureRepositoryBuilder  # noqa: E402
from core_human_review import (  # noqa: E402
    ANNOTATION_PROTOCOL_VERSION,
    _is_dos_83_alias_component,
    _is_windows_reserved_component,
    annotation_protocol_binding,
    fixture_manifest_from_mappings,
    load_source_bound_ledger_record,
    make_packet,
    project_ledger_record,
)


CORE_SOURCE_VERSION = "core-2026-07-21-v3"
CASE_VERSION = 3
PROTOCOL_ID = "native_repository"
CASE_PROVENANCE_SCHEMA_VERSION = "core_case_provenance_v2"
GOLDEN_INDEX_SCHEMA_VERSION = "core_golden_index_v2"
GOLDEN_ENTRY_SCHEMA_VERSION = "core_golden_entry_v2"
SUITE_SOURCE_PACKET_SCHEMA_VERSION = "core_suite_source_v3"
GOLDEN_RUN_BINDING_SCHEMA_VERSION = "core_golden_run_binding_v2"
GOLDEN_REPLAY_BINDING_SCHEMA_VERSION = "core_golden_repository_replay_v2"
GOLDEN_MATERIALIZATION_BINDING_SCHEMA_VERSION = (
    "core_golden_materialization_binding_v2"
)
GOLDEN_RUN_INSTANCE_KEY = "core-golden-authoring-v2"
GOLDEN_ATTEMPT = 1
ANNOTATION_RECORD_SCHEMA_VERSION = "core_annotation_record_v2"
HUMAN_REVIEW_RECORD_SCHEMA_VERSION = "core_human_review_record_v2"
GOLDEN_SCENARIOS = (
    "perfect",
    "empty",
    "duplicate",
    "fabricated",
    "unsupported-evidence",
    "compound",
    "judge-unknown",
    "unsupported-intent",
    "contradicted-intent",
    "bad-evidence",
    "bad-evidence-path",
    "bad-evidence-line",
)
_GOLDEN_TASK_BY_SCENARIO = MappingProxyType(
    {
        "perfect": "core-py-001",
        "empty": "core-py-011",
        "duplicate": "core-py-001",
        "fabricated": "core-py-015",
        "unsupported-evidence": "core-py-001",
        "compound": "core-py-012",
        "judge-unknown": "core-py-015",
        "unsupported-intent": "core-py-004",
        "contradicted-intent": "core-py-011",
        "bad-evidence": "core-py-014",
        "bad-evidence-path": "core-py-014",
        "bad-evidence-line": "core-py-014",
    }
)
REPOSITORY_WIRE_CONTRACT = {
    "case_schema_version": EVAL_CASE_SCHEMA_VERSION,
    "input_schema_version": EVAL_INPUT_SCHEMA_VERSION,
    "submission_schema_version": EVAL_SUBMISSION_SCHEMA_VERSION,
    "review_target_kind": "repository",
    "materializer_protocol": REPOSITORY_MATERIALIZER_PROTOCOL,
}

# This is intentionally pending until a person independently annotates and
# signs each frozen blind-review packet.  A model/sub-agent review must never
# be recorded as the external human audit gate.
HUMAN_REVIEW_STATUS = "requires_independent_re_review"


@dataclass(frozen=True)
class CoreCaseSpec:
    task_id: str
    split: str
    request: Mapping[str, Any]
    clarification_answers: tuple[Mapping[str, Any], ...]
    intent_truth: Mapping[str, Any]
    base_files: Mapping[str, str]
    head_files: Mapping[str, str]
    expected_findings: tuple[Mapping[str, Any], ...]
    known_invalid_findings: tuple[Mapping[str, Any], ...]
    dimensions: Mapping[str, str]
    intent_expectation: Mapping[str, Any]
    coverage: tuple[str, ...]
    annotation_rationale: str

    @property
    def suite_id(self) -> str:
        return "core-regression" if self.split == "regression" else "core-capability"


@dataclass(frozen=True)
class CoreBuildPlan:
    writable_outputs: Mapping[str, bytes]
    check_only_fixtures: Mapping[str, bytes]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "writable_outputs",
            MappingProxyType(dict(self.writable_outputs)),
        )
        object.__setattr__(
            self,
            "check_only_fixtures",
            MappingProxyType(dict(self.check_only_fixtures)),
        )


@dataclass(frozen=True)
class _ValidatedBuildPlanOwnership:
    writable_by_portable_key: Mapping[str, str]
    fixtures_by_portable_key: Mapping[str, str]
    known_by_portable_key: Mapping[str, str]


def _claim(
    truth_id: str,
    dimension: str,
    text: str,
    *,
    required: bool = True,
) -> dict[str, Any]:
    return {
        "truth_id": truth_id,
        "dimension": dimension,
        "text": text,
        "required": required,
    }


def _forbidden(
    truth_id: str,
    dimension: str,
    text: str,
    rationale: str,
) -> dict[str, Any]:
    return {
        "truth_id": truth_id,
        "dimension": dimension,
        "text": text,
        "rationale": rationale,
    }


def _location(
    path: str,
    from_line: int | None,
    to_line: int | None = None,
    *,
    side: str | None = "right",
) -> dict[str, Any]:
    return {
        "path": path,
        "side": side,
        "from_line": from_line,
        "to_line": from_line if from_line is not None and to_line is None else to_line,
    }


def _anchor(fact: str, *locations: Mapping[str, Any]) -> dict[str, Any]:
    return {"fact": fact, "locations": list(locations)}


def _finding(
    truth_id: str,
    claim: str,
    severity: str,
    category: str,
    context: str,
    rationale: str,
    *locations: Mapping[str, Any],
    required: bool = True,
    anchors: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    return {
        "truth_id": truth_id,
        "claim": claim,
        "severity": severity,
        "category": category,
        "required": required,
        "metric_authority": {
            "severity_scorable": True,
            "severity_authority": "expert_annotation",
            "location_scorable": True,
            "location_authority": "expert_annotation",
        },
        "locations": list(locations),
        "evidence_anchors": list(anchors),
        "required_context_level": context,
        "rationale": rationale,
    }


def _invalid(
    truth_id: str,
    claim: str,
    rationale: str,
    *locations: Mapping[str, Any],
    category: str | None = None,
) -> dict[str, Any]:
    return {
        "truth_id": truth_id,
        "claim": claim,
        "category": category,
        "locations": list(locations),
        "rationale": rationale,
    }


def _answer(
    answer_id: str,
    dimension: str,
    material_claim: str,
    action: str,
    response: str | None,
    corrected_values: Iterable[str] = (),
) -> dict[str, Any]:
    return {
        "answer_id": answer_id,
        "dimension": dimension,
        "material_claim": material_claim,
        "action": action,
        "response": response,
        "corrected_values": list(corrected_values),
    }


def _intent(
    authority: str,
    expected: Iterable[Mapping[str, Any]],
    *,
    forbidden: Iterable[Mapping[str, Any]] = (),
    clarification_policy: str = "not_required",
) -> dict[str, Any]:
    return {
        "scorable": True,
        "authority": authority,
        "expected_claims": list(expected),
        "forbidden_claims": list(forbidden),
        "clarification_policy": clarification_policy,
    }


def _request(
    title: str | None,
    *,
    description: str | None = None,
    user_intent: str | None = None,
    review_focus: str | None = None,
    linked_requirements: Iterable[str] = (),
    project_rules: Iterable[str] = (),
    existing_ci_evidence: Iterable[tuple[str, str]] = (),
) -> dict[str, Any]:
    ci = []
    for source_id, text in existing_ci_evidence:
        ci.append(
            {
                "source_id": source_id,
                "text": text,
                "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )
    return {
        "title": title,
        "description": description,
        "user_intent": user_intent,
        "review_focus": review_focus,
        "linked_requirements": list(linked_requirements),
        "project_rules": list(project_rules),
        "existing_ci_evidence": ci,
    }


def _expectation(
    *,
    initial_source: str,
    final_source: str,
    initial_status: str,
    final_status: str = "sufficient",
    clarification: str,
) -> dict[str, Any]:
    return {
        "initial_source": initial_source,
        "final_source": final_source,
        "initial_status": initial_status,
        "final_status": final_status,
        "clarification_decision": clarification,
    }


CASES: tuple[CoreCaseSpec, ...] = (
    CoreCaseSpec(
        task_id="core-py-001",
        split="regression",
        request=_request(
            "Keep timeout validation strict",
            user_intent="Reject both zero and negative timeout values while simplifying the validation.",
            review_focus="Validation behavior at the zero boundary.",
        ),
        clarification_answers=(),
        intent_truth=_intent(
            "explicit_author_metadata",
            (
                _claim("intent-0001", "goal", "Simplify timeout validation."),
                _claim(
                    "intent-0002",
                    "acceptance_criterion",
                    "A timeout of zero must be rejected.",
                ),
                _claim(
                    "intent-0003",
                    "constraint",
                    "A negative timeout must be rejected.",
                ),
            ),
        ),
        base_files={
            "src/timeout.py": "def normalize_timeout(value: int) -> int:\n    if value <= 0:\n        raise ValueError(\"timeout must be positive\")\n    return value\n"
        },
        head_files={
            "src/timeout.py": "def normalize_timeout(value: int) -> int:\n    if value < 0:\n        raise ValueError(\"timeout must be positive\")\n    return value\n"
        },
        expected_findings=(
            _finding(
                "issue-0001",
                "The new comparison accepts a zero timeout even though zero must remain invalid.",
                "medium",
                "regression",
                "diff",
                "Base rejects zero and negative values with <= 0, while head changes the boundary to < 0 and therefore accepts zero. A zero timeout now reaches callers despite the explicit acceptance criterion. This is a required medium regression because one supported validation boundary is lost while negative handling remains intact; the complete cause, trigger, and impact are visible in the diff.",
                _location("src/timeout.py", 2),
            ),
        ),
        known_invalid_findings=(),
        dimensions={
            "language": "python",
            "intent.behavior": "explicit_no_clarification",
            "review.category": "regression",
            "review.context": "diff",
            "risk.level": "medium",
        },
        intent_expectation=_expectation(
            initial_source="explicit",
            final_source="explicit",
            initial_status="sufficient",
            clarification="must_not_ask",
        ),
        coverage=("explicit_intent", "must_not_clarify", "boundary_bug", "diff_context"),
        annotation_rationale="The zero and negative Intent criteria are atomic; only the zero boundary regresses, producing one medium diff-level Finding.",
    ),
    CoreCaseSpec(
        task_id="core-py-002",
        split="regression",
        request=_request(
            "Refactor cache key construction",
            description="Centralize the cache key helper.",
        ),
        clarification_answers=(
            _answer(
                "answer-0001",
                "constraint",
                "Equal resource IDs in different tenants must continue to produce distinct cache keys.",
                "confirm",
                "Yes. Tenant isolation is required.",
            ),
        ),
        intent_truth=_intent(
            "synthetic",
            (
                _claim("intent-0004", "goal", "Refactor cache key construction."),
                _claim(
                    "intent-0005",
                    "constraint",
                    "Equal resource IDs in different tenants must continue to produce distinct cache keys.",
                ),
            ),
            clarification_policy="required",
        ),
        base_files={
            "src/cache.py": "def cache_key(tenant_id: str, resource_id: str) -> str:\n    return f\"{tenant_id}:{resource_id}\"\n",
            "tests/test_cache.py": "from src.cache import cache_key\n\n\ndef test_tenants_do_not_share_keys():\n    assert cache_key(\"tenant-a\", \"42\") != cache_key(\"tenant-b\", \"42\")\n",
        },
        head_files={
            "src/cache.py": "def cache_key(tenant_id: str, resource_id: str) -> str:\n    \"\"\"Build a cache key for one resource.\"\"\"\n    return resource_id\n",
            "tests/test_cache.py": "from src.cache import cache_key\n\n\ndef test_tenants_do_not_share_keys():\n    assert cache_key(\"tenant-a\", \"42\") != cache_key(\"tenant-b\", \"42\")\n",
        },
        expected_findings=(
            _finding(
                "issue-0002",
                "The refactored key omits tenant_id, so equal resource IDs collide across tenants.",
                "high",
                "regression",
                "diff",
                "Base incorporates tenant_id and resource_id, while head returns only resource_id. Two tenants using the same resource ID now receive the same cache key, violating the user-confirmed isolation constraint and risking cross-tenant cache contamination. This is a required high regression, and the return-value change alone proves the full claim from the diff.",
                _location("src/cache.py", 3),
            ),
        ),
        known_invalid_findings=(),
        dimensions={
            "language": "python",
            "intent.behavior": "inferred_then_confirmed",
            "review.category": "regression",
            "review.context": "diff",
            "risk.level": "high",
        },
        intent_expectation=_expectation(
            initial_source="inferred",
            final_source="explicit",
            initial_status="partial",
            clarification="must_ask_and_confirm",
        ),
        coverage=("inferred_intent", "confirmation_required", "diff_context", "regression"),
        annotation_rationale="The title supplies the goal, while tenant isolation is inferred from base/test context and becomes explicit only after the scripted user confirmation.",
    ),
    CoreCaseSpec(
        task_id="core-py-003",
        split="regression",
        request=_request(
            "Adjust the retention default",
            description="Update the default for the new retention policy.",
        ),
        clarification_answers=(
            _answer(
                "answer-0002",
                "acceptance_criterion",
                "Set the default retention period to 21 days.",
                "correct",
                "No. The new default is 14 days.",
                ("Set the default retention period to 14 days.",),
            ),
        ),
        intent_truth=_intent(
            "synthetic",
            (
                _claim("intent-0006", "goal", "Change the default retention period."),
                _claim(
                    "intent-0007",
                    "acceptance_criterion",
                    "The default retention period is 14 days.",
                ),
            ),
            clarification_policy="required",
        ),
        base_files={"src/settings.py": "DEFAULT_RETENTION_DAYS = 30\n"},
        head_files={"src/settings.py": "DEFAULT_RETENTION_DAYS = 21\n"},
        expected_findings=(
            _finding(
                "issue-0003",
                "The head sets the retention default to 21 days instead of the user-corrected 14-day target.",
                "medium",
                "correctness",
                "diff",
                "Base uses 30 days, head changes the constant to 21, and the required clarification corrects the inferred value to 14. The implementation therefore fails the newly supplied acceptance criterion rather than regressing a previously correct value. This is a required medium correctness issue whose value mismatch is fully visible in the diff after clarification.",
                _location("src/settings.py", 1),
            ),
        ),
        known_invalid_findings=(),
        dimensions={
            "language": "python",
            "intent.behavior": "insufficient_requires_clarification",
            "review.category": "correctness",
            "review.context": "diff",
            "risk.level": "medium",
        },
        intent_expectation=_expectation(
            initial_source="inferred",
            final_source="explicit",
            initial_status="insufficient",
            clarification="must_ask_and_correct",
        ),
        coverage=("insufficient_intent", "must_clarify", "user_correction", "correctness"),
        annotation_rationale="The target value is unknowable from the initial request; the model may infer 21 from head, but the user corrects it to 14, making the head value a concrete correctness mismatch.",
    ),
    CoreCaseSpec(
        task_id="core-py-004",
        split="regression",
        request=_request(
            "Make JSON output deterministic",
            description="The serializer accepts mappings whose keys are strings.",
            user_intent="Serialize string-keyed mappings deterministically without changing their data.",
            review_focus="Stable byte output.",
        ),
        clarification_answers=(),
        intent_truth=_intent(
            "explicit_author_metadata",
            (
                _claim("intent-0008", "goal", "Make JSON serialization deterministic."),
                _claim(
                    "intent-0009",
                    "constraint",
                    "Serialization of string-keyed mappings must preserve the input data.",
                ),
            ),
        ),
        base_files={
            "src/serializer.py": "import json\n\n\ndef serialize(payload: dict[str, object]):\n    return json.dumps(payload)\n"
        },
        head_files={
            "src/serializer.py": "import json\n\n\ndef serialize(payload: dict[str, object]):\n    return json.dumps(payload, sort_keys=True, separators=(\",\", \":\"))\n"
        },
        expected_findings=(),
        known_invalid_findings=(),
        dimensions={
            "language": "python",
            "intent.behavior": "explicit_no_clarification",
            "review.category": "clean",
            "review.context": "diff",
            "risk.level": "low",
        },
        intent_expectation=_expectation(
            initial_source="explicit",
            final_source="explicit",
            initial_status="sufficient",
            clarification="must_not_ask",
        ),
        coverage=(
            "explicit_intent",
            "must_not_clarify",
            "unsupported_intent",
            "clean_pr",
        ),
        annotation_rationale="The declared input contract is limited to string-keyed mappings, for which sorting is defined; compact separators change only insignificant JSON whitespace, so the closed-world review has no material Finding.",
    ),
    CoreCaseSpec(
        task_id="core-py-005",
        split="regression",
        request=_request(
            "Simplify the delete-user endpoint",
            user_intent="Simplify the handler while preserving administrator-only authorization.",
            linked_requirements=("SEC-7: deleting users requires an authenticated administrator",),
        ),
        clarification_answers=(),
        intent_truth=_intent(
            "explicit_author_metadata",
            (
                _claim("intent-0010", "goal", "Simplify the delete-user handler."),
                _claim(
                    "intent-0011",
                    "constraint",
                    "Unauthenticated actors must not be able to delete users.",
                ),
                _claim(
                    "intent-0012",
                    "constraint",
                    "Authenticated non-administrators must not be able to delete users.",
                ),
            ),
        ),
        base_files={
            "src/auth.py": "def is_admin(user):\n    return user is not None and \"admin\" in user.roles\n",
            "src/api.py": "from .auth import is_admin\n\n\ndef delete_user(actor, target_id):\n    if not is_admin(actor):\n        raise PermissionError(\"administrator required\")\n    return {\"deleted\": target_id}\n",
        },
        head_files={
            "src/auth.py": "def is_admin(user):\n    return user is not None\n",
            "src/api.py": "from .auth import is_admin\n\n\ndef delete_user(actor, target_id):\n    if not is_admin(actor):\n        raise PermissionError(\"administrator required\")\n    return {\"deleted\": target_id}\n",
        },
        expected_findings=(
            _finding(
                "issue-0004",
                "Weakening is_admin to a non-null check lets authenticated non-administrators pass the unchanged delete-user authorization guard.",
                "high",
                "security",
                "repo",
                "Base requires both a user and admin role in is_admin, while head removes the role test. The unchanged cross-file delete_user endpoint still trusts that helper as its authorization boundary, so a logged-in non-admin can delete users. This is a required high security issue; repository context is necessary to connect the changed generic helper to the protected operation and exclude an endpoint-level compensating check.",
                _location("src/auth.py", 2, side="left"),
                _location("src/auth.py", 2),
                anchors=(
                    _anchor(
                        "The unchanged delete-user endpoint relies on is_admin before deleting the target.",
                        _location("src/api.py", 1),
                        _location("src/api.py", 4, 7),
                    ),
                ),
            ),
        ),
        known_invalid_findings=(),
        dimensions={
            "language": "python",
            "intent.behavior": "explicit_no_clarification",
            "review.category": "security",
            "review.context": "repo",
            "risk.level": "high",
        },
        intent_expectation=_expectation(
            initial_source="explicit",
            final_source="explicit",
            initial_status="sufficient",
            clarification="must_not_ask",
        ),
        coverage=("security", "high_miss", "repo_context", "explicit_intent"),
        annotation_rationale="The changed predicate introduces one authorization regression; the cross-file helper definitions are needed to prove that authenticated and administrator access are different.",
    ),
    CoreCaseSpec(
        task_id="core-py-006",
        split="regression",
        request=_request(
            "Refactor service URL loading",
            user_intent="Use the primary environment variable while preserving the documented legacy alias.",
            linked_requirements=("CFG-2: LEGACY_SERVICE_URL remains supported through version 2",),
            project_rules=("Backward-compatible configuration aliases cannot be removed before version 3.",),
        ),
        clarification_answers=(),
        intent_truth=_intent(
            "explicit_author_metadata",
            (
                _claim(
                    "intent-0013",
                    "goal",
                    "Use SERVICE_URL as the primary service URL source.",
                ),
                _claim(
                    "intent-0014",
                    "constraint",
                    "When SERVICE_URL is absent or empty, continue accepting LEGACY_SERVICE_URL through version 2.",
                ),
            ),
        ),
        base_files={
            "src/config.py": "import os\n\nSUPPORTED_ENV_COUNT = 2\n\n\ndef _env_names():\n    return (\"SERVICE_URL\", \"LEGACY_SERVICE_URL\")\n\n\ndef service_url():\n    for name in _env_names()[:SUPPORTED_ENV_COUNT]:\n        value = os.environ.get(name)\n        if value:\n            return value\n    raise KeyError(\"service URL is missing\")\n"
        },
        head_files={
            "src/config.py": "import os\n\nSUPPORTED_ENV_COUNT = 1\n\n\ndef _env_names():\n    return (\"SERVICE_URL\", \"LEGACY_SERVICE_URL\")\n\n\ndef service_url():\n    for name in _env_names()[:SUPPORTED_ENV_COUNT]:\n        value = os.environ.get(name)\n        if value:\n            return value\n    raise KeyError(\"service URL is missing\")\n"
        },
        expected_findings=(
            _finding(
                "issue-0005",
                "Reducing SUPPORTED_ENV_COUNT to one prevents service_url from trying the required LEGACY_SERVICE_URL fallback, so legacy-only deployments now fail to load a URL.",
                "high",
                "regression",
                "file",
                "Only the count changes in the diff; the unchanged slice later in the file limits iteration to that count, and the unchanged name tuple puts LEGACY_SERVICE_URL second. With only the legacy variable set, head reaches the KeyError. This violates the explicit compatibility contract and is a required high-severity regression for existing deployments; the same-file non-diff control flow is necessary to prove the complete claim.",
                _location("src/config.py", 3),
                anchors=(
                    _anchor(
                        "The unchanged environment-name tuple places LEGACY_SERVICE_URL after SERVICE_URL.",
                        _location("src/config.py", 6, 7),
                    ),
                    _anchor(
                        "The unchanged service_url loop slices the environment-name tuple by SUPPORTED_ENV_COUNT before reading values.",
                        _location("src/config.py", 10, 14),
                    ),
                ),
            ),
        ),
        known_invalid_findings=(),
        dimensions={
            "language": "python",
            "intent.behavior": "linked_requirement",
            "review.category": "regression",
            "review.context": "file",
            "risk.level": "high",
        },
        intent_expectation=_expectation(
            initial_source="explicit",
            final_source="explicit",
            initial_status="sufficient",
            clarification="must_not_ask",
        ),
        coverage=("regression", "file_context", "project_rule", "backward_compatibility"),
        annotation_rationale="The one-line count change is meaningful only through unchanged same-file slicing and iteration, making this the Core file-context regression Case.",
    ),
    CoreCaseSpec(
        task_id="core-py-007",
        split="regression",
        request=_request(
            "Handle empty batches",
            user_intent="Return 0.0 when averaging an empty batch and preserve non-empty behavior.",
        ),
        clarification_answers=(),
        intent_truth=_intent(
            "explicit_author_metadata",
            (
                _claim("intent-0015", "goal", "Handle empty batches without an exception."),
                _claim(
                    "intent-0016",
                    "acceptance_criterion",
                    "An empty batch returns 0.0.",
                ),
                _claim(
                    "intent-0017",
                    "constraint",
                    "Averaging non-empty inputs must preserve existing behavior.",
                ),
            ),
        ),
        base_files={
            "src/stats.py": "def average(values):\n    return sum(values) / len(values)\n"
        },
        head_files={
            "src/stats.py": "def average(values):\n    if len(values) == 0:\n        return 0.0\n    return sum(values) / len(values)\n"
        },
        expected_findings=(),
        known_invalid_findings=(),
        dimensions={
            "language": "python",
            "intent.behavior": "explicit_no_clarification",
            "review.category": "clean",
            "review.context": "diff",
            "risk.level": "low",
        },
        intent_expectation=_expectation(
            initial_source="explicit",
            final_source="explicit",
            initial_status="sufficient",
            clarification="must_not_ask",
        ),
        coverage=("clean_pr", "must_not_clarify", "diff_context"),
        annotation_rationale="The explicit length check handles only empty inputs and leaves the existing non-empty formula unchanged, including containers whose truth value is ambiguous or customized.",
    ),
    CoreCaseSpec(
        task_id="core-py-008",
        split="regression",
        request=_request(
            "Normalize display names",
            user_intent="Make display-name normalization case-insensitive.",
            review_focus="Only regressions introduced by this change.",
        ),
        clarification_answers=(),
        intent_truth=_intent(
            "explicit_author_metadata",
            (_claim("intent-0018", "goal", "Normalize display names case-insensitively."),),
        ),
        base_files={
            "src/formatting.py": "def normalize_name(name):\n    return name.strip()\n",
            "src/legacy.py": "def evaluate_legacy(expression):\n    return eval(expression)\n",
        },
        head_files={
            "src/formatting.py": "def normalize_name(name):\n    return name.strip().casefold()\n",
            "src/legacy.py": "def evaluate_legacy(expression):\n    return eval(expression)\n",
        },
        expected_findings=(),
        known_invalid_findings=(
            _invalid(
                "invalid-0001",
                "This change introduces unsafe eval usage in the legacy evaluator.",
                "The eval call is identical in base and head and is therefore pre-existing, not introduced by this PR.",
                _location("src/legacy.py", 2, side="left"),
                _location("src/legacy.py", 2),
                category="security",
            ),
        ),
        dimensions={
            "language": "python",
            "intent.behavior": "explicit_no_clarification",
            "review.category": "clean_with_trap",
            "review.context": "repo",
            "risk.level": "low",
        },
        intent_expectation=_expectation(
            initial_source="explicit",
            final_source="explicit",
            initial_status="sufficient",
            clarification="must_not_ask",
        ),
        coverage=("preexisting_trap", "known_invalid", "clean_pr", "fabrication_guard"),
        annotation_rationale="The legacy eval is intentionally unchanged so reporting it as introduced is a known-invalid finding.",
    ),
    CoreCaseSpec(
        task_id="core-py-009",
        split="regression",
        request=_request(
            "Simplify header parsing",
            user_intent="Keep parsing headers whose payload contains additional colons.",
            existing_ci_evidence=(
                ("ci-header-colon", "test_parse_header_with_colon failed: too many values to unpack"),
            ),
        ),
        clarification_answers=(),
        intent_truth=_intent(
            "explicit_author_metadata",
            (
                _claim("intent-0019", "goal", "Simplify header parsing."),
                _claim(
                    "intent-0020",
                    "acceptance_criterion",
                    "Payloads containing additional colons must continue to parse.",
                ),
            ),
        ),
        base_files={
            "src/parser.py": "def parse_header(value):\n    name, payload = value.split(\":\", 1)\n    return name, payload\n"
        },
        head_files={
            "src/parser.py": "def parse_header(value):\n    name, payload = value.split(\":\")\n    return name, payload\n"
        },
        expected_findings=(
            _finding(
                "issue-0006",
                "Using an unbounded split raises when the payload itself contains a colon.",
                "medium",
                "regression",
                "diff",
                "Base bounds split to one separator, while head removes maxsplit. A payload containing another colon therefore produces more than two values and raises during tuple unpacking, matching the supplied CI signal. This is a required medium correctness regression on a specified input and is fully provable from the diff; no alternative parser path is involved.",
                _location("src/parser.py", 2),
            ),
        ),
        known_invalid_findings=(),
        dimensions={
            "language": "python",
            "intent.behavior": "explicit_no_clarification",
            "review.category": "regression",
            "review.context": "diff",
            "risk.level": "medium",
        },
        intent_expectation=_expectation(
            initial_source="explicit",
            final_source="explicit",
            initial_status="sufficient",
            clarification="must_not_ask",
        ),
        coverage=("existing_ci", "command_output_signal", "regression", "diff_context"),
        annotation_rationale="The explicit payload contract and deterministic split semantics establish one atomic parsing failure; CI is corroborating Agent-visible evidence, not the hidden oracle.",
    ),
    CoreCaseSpec(
        task_id="core-py-010",
        split="regression",
        request=_request(
            "Register the requested admin route",
            description="Add the route discussed with the platform team.",
        ),
        clarification_answers=(
            _answer(
                "answer-0003",
                "scope",
                "Expose /admin on the public router.",
                "correct",
                "No. Register /admin only on the internal router.",
                ("Register /admin only in the internal router.",),
            ),
        ),
        intent_truth=_intent(
            "synthetic",
            (
                _claim("intent-0021", "goal", "Register the /admin route."),
                _claim(
                    "intent-0022",
                    "scope",
                    "Register /admin only in the internal router.",
                ),
            ),
            forbidden=(
                _forbidden(
                    "forbidden-intent-0001",
                    "scope",
                    "Expose /admin on the public router.",
                    "The user correction explicitly limits the route to the internal router.",
                ),
            ),
            clarification_policy="required",
        ),
        base_files={
            "src/admin.py": "def delete_user(request):\n    request.user_store.delete(request.target_id)\n    return {\"deleted\": request.target_id}\n",
            "src/router.py": "from .routes import INTERNAL_ROUTES, PUBLIC_ROUTES\n\n\ndef dispatch_public(path, request):\n    return PUBLIC_ROUTES[path](request)\n\n\ndef dispatch_internal(path, request):\n    if not request.actor.is_admin:\n        raise PermissionError(\"administrator required\")\n    return INTERNAL_ROUTES[path](request)\n",
            "src/routes.py": "from .admin import delete_user\n\nPUBLIC_ROUTES = {}\nINTERNAL_ROUTES = {}\n",
        },
        head_files={
            "src/admin.py": "def delete_user(request):\n    request.user_store.delete(request.target_id)\n    return {\"deleted\": request.target_id}\n",
            "src/router.py": "from .routes import INTERNAL_ROUTES, PUBLIC_ROUTES\n\n\ndef dispatch_public(path, request):\n    return PUBLIC_ROUTES[path](request)\n\n\ndef dispatch_internal(path, request):\n    if not request.actor.is_admin:\n        raise PermissionError(\"administrator required\")\n    return INTERNAL_ROUTES[path](request)\n",
            "src/routes.py": "from .admin import delete_user\n\nPUBLIC_ROUTES = {\"/admin\": delete_user}\nINTERNAL_ROUTES = {}\n",
        },
        expected_findings=(
            _finding(
                "issue-0007",
                "Registering /admin in PUBLIC_ROUTES lets the unauthenticated public dispatcher invoke the user-deletion handler instead of applying the internal router's administrator check.",
                "high",
                "security",
                "repo",
                "The scripted correction requires /admin to use the internal router. Head instead adds delete_user to PUBLIC_ROUTES. The unchanged public dispatcher calls mapped handlers without authentication, while the unchanged internal dispatcher rejects actors whose is_admin flag is false; the mapped handler deletes the requested user. An unauthenticated public request can therefore perform an administrative deletion, which is a required high-severity authorization failure. Repository context is necessary to connect the changed registration to both dispatcher policies and the handler's capability.",
                _location("src/routes.py", 3),
                anchors=(
                    _anchor(
                        "The public dispatcher invokes a registered handler without authenticating the request.",
                        _location("src/router.py", 4, 5),
                    ),
                    _anchor(
                        "The internal dispatcher requires request.actor.is_admin before invoking a registered handler.",
                        _location("src/router.py", 8, 11),
                    ),
                    _anchor(
                        "The /admin handler deletes the target user from the supplied user store.",
                        _location("src/admin.py", 1, 3),
                    ),
                ),
            ),
        ),
        known_invalid_findings=(),
        dimensions={
            "language": "python",
            "intent.behavior": "user_corrected",
            "review.category": "security",
            "review.context": "repo",
            "risk.level": "high",
        },
        intent_expectation=_expectation(
            initial_source="inferred",
            final_source="explicit",
            initial_status="insufficient",
            clarification="must_ask_and_correct",
        ),
        coverage=("user_correction", "forbidden_intent", "high_miss", "security"),
        annotation_rationale="The user correction replaces the inferred public scope with an explicit internal-only scope; head implements the rejected value, yielding one atomic security Finding.",
    ),
    CoreCaseSpec(
        task_id="core-py-011",
        split="capability",
        request=_request(
            "Remove API authentication checks",
            user_intent="Remove authentication so local clients can call the endpoint directly.",
            linked_requirements=("SEC-1: every API request must be authenticated",),
        ),
        clarification_answers=(
            _answer(
                "answer-0004",
                "goal",
                "Remove authentication from API requests.",
                "reject",
                "No. Authentication remains mandatory for every API request.",
            ),
        ),
        intent_truth=_intent(
            "linked_requirement",
            (
                _claim(
                    "intent-0023",
                    "goal",
                    "Preserve authentication for every API request.",
                ),
            ),
            forbidden=(
                _forbidden(
                    "forbidden-intent-0002",
                    "goal",
                    "Remove authentication from API requests.",
                    "The requested removal contradicts the authoritative linked security requirement.",
                ),
            ),
            clarification_policy="required",
        ),
        base_files={
            "src/api.py": "from .database import export_all_customers\n\n\ndef export_customers(request):\n    if request.token is None:\n        raise PermissionError(\"token required\")\n    return export_all_customers()\n",
            "src/database.py": "CUSTOMER_RECORDS = [\n    {\"email\": \"first@example.test\", \"password_reset_token\": \"reset-example-1\"},\n    {\"email\": \"second@example.test\", \"password_reset_token\": \"reset-example-2\"},\n]\n\n\ndef export_all_customers():\n    return CUSTOMER_RECORDS\n",
            "src/routes.py": "PUBLIC_ROUTES = {\"/customers/export\": \"api.export_customers\"}\n",
        },
        head_files={
            "src/api.py": "from .database import export_all_customers\n\n\ndef export_customers(request):\n    return export_all_customers()\n",
            "src/database.py": "CUSTOMER_RECORDS = [\n    {\"email\": \"first@example.test\", \"password_reset_token\": \"reset-example-1\"},\n    {\"email\": \"second@example.test\", \"password_reset_token\": \"reset-example-2\"},\n]\n\n\ndef export_all_customers():\n    return CUSTOMER_RECORDS\n",
            "src/routes.py": "PUBLIC_ROUTES = {\"/customers/export\": \"api.export_customers\"}\n",
        },
        expected_findings=(
            _finding(
                "issue-0008",
                "The public customer-export endpoint returns every stored customer record, including password-reset tokens, without authenticating the request.",
                "critical",
                "security",
                "repo",
                "Base rejects requests without a token, while head removes that only guard and immediately calls export_all_customers. The unchanged public route exposes this function, and the unchanged database export returns the complete CUSTOMER_RECORDS collection containing password-reset tokens. An unauthenticated caller can therefore obtain reset credentials for every stored account, directly violating SEC-1 and enabling broad account takeover. This is a required critical security Finding; repository context proves both public reachability and the sensitive full-dataset impact, while the rejected author request is not authoritative.",
                _location("src/api.py", 5, 6, side="left"),
                _location("src/api.py", 5),
                anchors=(
                    _anchor(
                        "The customer export function is registered on the public route table.",
                        _location("src/routes.py", 1),
                    ),
                    _anchor(
                        "The export returns the complete customer collection, whose records include password-reset tokens.",
                        _location("src/database.py", 1, 8),
                    ),
                ),
            ),
        ),
        known_invalid_findings=(),
        dimensions={
            "language": "python",
            "intent.behavior": "contradicted_then_rejected",
            "review.category": "security",
            "review.context": "repo",
            "risk.level": "critical",
        },
        intent_expectation=_expectation(
            initial_source="explicit",
            final_source="explicit",
            initial_status="insufficient",
            clarification="must_ask_and_reject",
        ),
        coverage=(
            "contradicted_intent",
            "critical",
            "critical_miss",
            "security",
        ),
        annotation_rationale="The explicit author request conflicts with the linked mandatory-authentication requirement, so reliable review requires clarification; the rejection preserves authentication and makes the unauthenticated customer export a critical mismatch.",
    ),
    CoreCaseSpec(
        task_id="core-py-012",
        split="capability",
        request=_request(
            "Streamline order updates",
            user_intent="Simplify order updates without accepting negative totals or dropping modification timestamps.",
        ),
        clarification_answers=(),
        intent_truth=_intent(
            "explicit_author_metadata",
            (
                _claim("intent-0024", "goal", "Simplify order updates."),
                _claim(
                    "intent-0025",
                    "constraint",
                    "Negative order totals must remain invalid.",
                ),
                _claim(
                    "intent-0026",
                    "acceptance_criterion",
                    "Successful updates must refresh updated_at.",
                ),
            ),
        ),
        base_files={
            "src/order.py": "from datetime import datetime, timezone\n\n\ndef update_order(order, total):\n    if total < 0:\n        raise ValueError(\"negative total\")\n    order.total = total\n    order.updated_at = datetime.now(timezone.utc)\n    return order\n"
        },
        head_files={
            "src/order.py": "from datetime import datetime, timezone\n\n\ndef update_order(order, total):\n    order.total = total\n    return order\n"
        },
        expected_findings=(
            _finding(
                "issue-0009",
                "The streamlined update accepts negative totals because validation was removed.",
                "high",
                "regression",
                "diff",
                "Base rejects negative totals before mutation, while head deletes that guard and assigns any supplied total. Negative values now enter persisted order state, violating the explicit constraint. This is a required high-severity correctness issue because it permits invalid data, and the deletion plus unguarded assignment are fully visible in the diff.",
                _location("src/order.py", 5, 6, side="left"),
                _location("src/order.py", 5),
            ),
            _finding(
                "issue-0010",
                "The update no longer refreshes updated_at after changing the total.",
                "medium",
                "regression",
                "diff",
                "Base refreshes updated_at after assigning the total, but head deletes that assignment and returns immediately. Successful updates therefore retain stale modification metadata, violating the explicit acceptance criterion. This independently fixable, required medium regression is distinct from negative-value validation and is completely shown by the diff.",
                _location("src/order.py", 8, side="left"),
                _location("src/order.py", 6),
            ),
        ),
        known_invalid_findings=(),
        dimensions={
            "language": "python",
            "intent.behavior": "explicit_no_clarification",
            "review.category": "multiple_atomic",
            "review.context": "diff",
            "risk.level": "high",
        },
        intent_expectation=_expectation(
            initial_source="explicit",
            final_source="explicit",
            initial_status="sufficient",
            clarification="must_not_ask",
        ),
        coverage=("compound_finding_trap", "atomic_findings", "high_miss", "diff_context"),
        annotation_rationale="Validation and timestamp refresh have different acceptance tests and fixes, so the two deletions are separate atomic truth items even when an Agent combines them in one Finding.",
    ),
    CoreCaseSpec(
        task_id="core-py-013",
        split="capability",
        request=_request(
            "Handle update failures",
            description="Return None when the locked update raises RuntimeError without changing lock lifecycle behavior.",
        ),
        clarification_answers=(),
        intent_truth=_intent(
            "explicit_author_metadata",
            (
                _claim("intent-0027", "goal", "Return None for RuntimeError during an update."),
                _claim(
                    "intent-0028",
                    "constraint",
                    "Release the lock on every exit path, including success, handled RuntimeError, and other exceptions.",
                ),
            ),
        ),
        base_files={
            "src/locked.py": "def update_with_lock(lock, update):\n    lock.acquire()\n    try:\n        return update()\n    finally:\n        lock.release()\n"
        },
        head_files={
            "src/locked.py": "def update_with_lock(lock, update):\n    lock.acquire()\n    try:\n        return update()\n    except RuntimeError:\n        return None\n"
        },
        expected_findings=(
            _finding(
                "issue-0011",
                "Removing the finally block leaves the acquired lock unreleased on every exit path: success, handled RuntimeError, and other exceptions.",
                "high",
                "regression",
                "diff",
                "Base releases the acquired lock from a finally block after a successful update, a handled RuntimeError, or any other exception. Head removes that single cleanup site and contains no release before either return or exceptional propagation. Every exit can therefore retain the lock and block subsequent callers indefinitely, violating the explicit lifecycle constraint. All exit-path consequences share the deleted finally block and the same fix, so this remains one required high-severity atomic Finding fully visible in the diff.",
                _location("src/locked.py", 5, 6, side="left"),
            ),
        ),
        known_invalid_findings=(),
        dimensions={
            "language": "python",
            "intent.behavior": "explicit_no_clarification",
            "review.category": "duplicate_trap",
            "review.context": "diff",
            "risk.level": "high",
        },
        intent_expectation=_expectation(
            initial_source="explicit",
            final_source="explicit",
            initial_status="sufficient",
            clarification="must_not_ask",
        ),
        coverage=("duplicate_finding_trap", "multi_location_one_issue", "must_not_clarify"),
        annotation_rationale="The request explicitly preserves lock lifecycle; deleting one finally block affects successful return, handled RuntimeError, and other exceptional exits but remains one root cause and one independently fixable Finding.",
    ),
    CoreCaseSpec(
        task_id="core-py-014",
        split="capability",
        request=_request(
            "Simplify token validity checks",
            user_intent="Keep tokens valid only until their expiration timestamp.",
        ),
        clarification_answers=(),
        intent_truth=_intent(
            "explicit_author_metadata",
            (
                _claim("intent-0029", "goal", "Simplify token validity checks."),
                _claim(
                    "intent-0030",
                    "acceptance_criterion",
                    "A token is valid only while expires_at is in the future.",
                ),
            ),
        ),
        base_files={
            "src/token.py": "import time\n\n\ndef is_valid(expires_at):\n    return expires_at > time.time()\n",
            "src/cache.py": "import time\n\n\ndef is_fresh(cached_at, ttl):\n    return cached_at + ttl >= time.time()\n",
        },
        head_files={
            "src/token.py": "import time\n\n\ndef is_valid(expires_at):\n    return expires_at <= time.time()\n",
            "src/cache.py": "import time\n\n\ndef is_fresh(cached_at, ttl):\n    return cached_at + ttl >= time.time()\n",
        },
        expected_findings=(
            _finding(
                "issue-0012",
                "The comparison is reversed, making expired tokens valid and future tokens invalid.",
                "high",
                "security",
                "diff",
                "Base accepts only expires_at values later than now; head reverses the comparison and accepts expired tokens while rejecting future ones. That permits use of expired authentication material and violates the explicit validity contract. This is a required high-severity security issue, entirely demonstrated by the changed expression; the nearby cache comparison is an unchanged alternative explanation and not the cause.",
                _location("src/token.py", 5),
            ),
        ),
        known_invalid_findings=(
            _invalid(
                "invalid-0002",
                "The cache freshness comparison is reversed by this change.",
                "The cache file is byte-identical in base and head, and cached_at + ttl >= now is the correct freshness predicate; it cannot be attributed to this change.",
                _location("src/cache.py", 5),
                category="regression",
            ),
        ),
        dimensions={
            "language": "python",
            "intent.behavior": "explicit_no_clarification",
            "review.category": "security",
            "review.context": "diff",
            "risk.level": "high",
        },
        intent_expectation=_expectation(
            initial_source="explicit",
            final_source="explicit",
            initial_status="sufficient",
            clarification="must_not_ask",
        ),
        coverage=(
            "wrong_path_trap",
            "wrong_line_trap",
            "high_miss",
        ),
        annotation_rationale="A nearby unchanged comparison is a location decoy; only src/token.py contains the introduced defect.",
    ),
    CoreCaseSpec(
        task_id="core-py-015",
        split="capability",
        request=_request(
            "Escape quoted display names",
            user_intent="Ensure HTML attributes escape both text and quote characters.",
        ),
        clarification_answers=(),
        intent_truth=_intent(
            "explicit_author_metadata",
            (
                _claim(
                    "intent-0031",
                    "goal",
                    "Safely escape display names rendered in HTML attributes.",
                ),
                _claim(
                    "intent-0032",
                    "acceptance_criterion",
                    "Continue escaping HTML text metacharacters such as ampersands and angle brackets.",
                ),
                _claim(
                    "intent-0033",
                    "acceptance_criterion",
                    "Escape single and double quote characters.",
                ),
            ),
        ),
        base_files={
            "src/renderer.py": "import html\n\n\ndef render_name(value):\n    return html.escape(value, quote=False)\n"
        },
        head_files={
            "src/renderer.py": "import html\n\n\ndef render_name(value):\n    return html.escape(value, quote=True)\n"
        },
        expected_findings=(),
        known_invalid_findings=(
            _invalid(
                "invalid-0003",
                "The new quote=True option disables escaping and introduces HTML injection.",
                "Base quote=False already escapes ampersands and angle brackets; head quote=True preserves those escapes and additionally escapes quotes. For example, html.escape('\"&', quote=True) yields '&quot;&amp;', so the claimed disabling of escaping is directly false.",
                _location("src/renderer.py", 5),
                category="security",
            ),
        ),
        dimensions={
            "language": "python",
            "intent.behavior": "explicit_no_clarification",
            "review.category": "clean_with_trap",
            "review.context": "diff",
            "risk.level": "low",
        },
        intent_expectation=_expectation(
            initial_source="explicit",
            final_source="explicit",
            initial_status="sufficient",
            clarification="must_not_ask",
        ),
        coverage=(
            "clean_pr",
            "fabricated_finding_trap",
            "judge_unknown_carrier",
            "known_invalid",
        ),
        annotation_rationale="The change strengthens escaping; the inverse security claim is deliberately known-invalid.",
    ),
    CoreCaseSpec(
        task_id="core-py-016",
        split="capability",
        request=_request(
            "Prepare the alpha plugin for metadata cleanup",
            description="Clean up plugin metadata without changing how the alpha plugin is discovered.",
        ),
        clarification_answers=(),
        intent_truth=_intent(
            "explicit_author_metadata",
            (
                _claim("intent-0034", "goal", "Clean up alpha plugin metadata."),
                _claim(
                    "intent-0035",
                    "constraint",
                    "The alpha registry entry must continue to discover the alpha plugin.",
                ),
            ),
        ),
        base_files={
            "src/registry.py": "from importlib import import_module\n\nPLUGIN_MODULES = {\"alpha\": \"plugins.alpha\"}\n\n\ndef discover(name):\n    plugin = import_module(PLUGIN_MODULES[name])\n    if plugin.PLUGIN_NAME != name:\n        raise LookupError(\"plugin identity mismatch\")\n    return plugin\n",
            "plugins/alpha.py": "PLUGIN_NAME = \"alpha\"\n",
        },
        head_files={
            "src/registry.py": "from importlib import import_module\n\nPLUGIN_MODULES = {\"alpha\": \"plugins.alpha\"}\n\n\ndef discover(name):\n    plugin = import_module(PLUGIN_MODULES[name])\n    if plugin.PLUGIN_NAME != name:\n        raise LookupError(\"plugin identity mismatch\")\n    return plugin\n",
            "plugins/alpha.py": "PLUGIN_NAME = \"beta\"\n",
        },
        expected_findings=(
            _finding(
                "issue-0013",
                "discover('alpha') now deterministically raises LookupError because the imported alpha module declares the name beta.",
                "medium",
                "regression",
                "repo",
                "Head changes PLUGIN_NAME from alpha to beta while the unchanged registry maps the alpha key to plugins.alpha and rejects any imported module whose declared name differs from the requested key. Calling discover('alpha') therefore deterministically raises LookupError before returning, making the alpha plugin unavailable and violating the explicit discovery-preservation contract. The fixture proves loss of one plugin but not system-wide outage, so this is a required medium regression; cross-file repository context is necessary to prove the failure path.",
                _location("plugins/alpha.py", 1),
                anchors=(
                    _anchor(
                        "The registry still maps the alpha key to plugins.alpha.",
                        _location("src/registry.py", 3),
                    ),
                    _anchor(
                        "discover() raises LookupError when the imported plugin's declared name differs from the requested registry key.",
                        _location("src/registry.py", 6, 9),
                    ),
                ),
            ),
        ),
        known_invalid_findings=(),
        dimensions={
            "language": "python",
            "intent.behavior": "explicit_no_clarification",
            "review.category": "regression",
            "review.context": "repo",
            "risk.level": "medium",
        },
        intent_expectation=_expectation(
            initial_source="explicit",
            final_source="explicit",
            initial_status="sufficient",
            clarification="must_not_ask",
        ),
        coverage=("repo_context", "must_not_clarify", "cross_file_invariant"),
        annotation_rationale="The request explicitly preserves alpha discovery, while proving the introduced identity mismatch requires relating the changed module metadata to the unchanged registry mapping.",
    ),
    CoreCaseSpec(
        task_id="core-py-017",
        split="capability",
        request=_request(
            "Simplify configuration reads",
            user_intent="Read configuration text with UTF-8 while preserving reliable file-handle cleanup.",
        ),
        clarification_answers=(),
        intent_truth=_intent(
            "explicit_author_metadata",
            (
                _claim("intent-0036", "goal", "Simplify UTF-8 configuration reads."),
                _claim(
                    "intent-0037",
                    "constraint",
                    "The opened file handle must always be closed.",
                ),
            ),
        ),
        base_files={
            "src/config_reader.py": "def read_config(path):\n    with open(path, encoding=\"utf-8\") as handle:\n        return handle.read()\n"
        },
        head_files={
            "src/config_reader.py": "def read_config(path):\n    handle = open(path, encoding=\"utf-8\")\n    return handle.read()\n"
        },
        expected_findings=(
            _finding(
                "issue-0014",
                "The simplified function returns without closing the opened configuration file.",
                "medium",
                "regression",
                "diff",
                "Base uses a context manager that closes the handle on return and exceptions; head opens the file directly and returns without close. Repeated reads can leak descriptors and retain file locks, violating the explicit cleanup constraint. This is a required medium correctness issue, and the entire lifecycle change is visible in the diff.",
                _location("src/config_reader.py", 2, 3),
            ),
        ),
        known_invalid_findings=(),
        dimensions={
            "language": "python",
            "intent.behavior": "explicit_no_clarification",
            "review.category": "correctness",
            "review.context": "diff",
            "risk.level": "medium",
        },
        intent_expectation=_expectation(
            initial_source="explicit",
            final_source="explicit",
            initial_status="sufficient",
            clarification="must_not_ask",
        ),
        coverage=("must_not_clarify", "diff_context", "resource_leak"),
        annotation_rationale="The cleanup requirement is explicit and the context-manager deletion directly proves one file-handle lifecycle regression, so clarification is unnecessary.",
    ),
    CoreCaseSpec(
        task_id="core-py-018",
        split="capability",
        request=_request(
            "Reduce request boilerplate",
            user_intent="Simplify request and error handling without disabling TLS verification or removing exception details from failure messages.",
        ),
        clarification_answers=(),
        intent_truth=_intent(
            "explicit_author_metadata",
            (
                _claim("intent-0038", "goal", "Reduce request boilerplate."),
                _claim(
                    "intent-0039",
                    "constraint",
                    "TLS certificate verification must remain enabled.",
                ),
                _claim(
                    "intent-0040",
                    "acceptance_criterion",
                    "Failure messages should retain exception details.",
                ),
            ),
        ),
        base_files={
            "src/client.py": "def fetch(session, url):\n    return session.get(url, verify=True, timeout=5)\n\n\ndef failure_message(exc):\n    return f\"request failed: {exc}\"\n"
        },
        head_files={
            "src/client.py": "def fetch(session, url):\n    return session.get(url, verify=False, timeout=5)\n\n\ndef failure_message(exc):\n    return \"request failed\"\n"
        },
        expected_findings=(
            _finding(
                "issue-0015",
                "The request explicitly disables TLS certificate verification.",
                "high",
                "security",
                "diff",
                "Base passes verify=True and head changes it to verify=False. HTTPS requests will accept untrusted certificates, enabling man-in-the-middle interception and violating the explicit TLS constraint. This is a required high-severity security issue; the changed call argument and impact are fully established by the diff.",
                _location("src/client.py", 2),
            ),
            _finding(
                "issue-0016",
                "The failure message drops the exception detail that previously aided diagnosis.",
                "low",
                "regression",
                "diff",
                "Base interpolates the exception into the failure message, while head returns only a generic string. Failures therefore lose actionable diagnostics, contrary to the explicit acceptance criterion. This Finding remains optional because the impact is low and recoverable, has a separate fix from TLS verification, and is fully proved by line 6 of the diff.",
                _location("src/client.py", 6),
                required=False,
            ),
        ),
        known_invalid_findings=(),
        dimensions={
            "language": "python",
            "intent.behavior": "explicit_no_clarification",
            "review.category": "severity_priority",
            "review.context": "diff",
            "risk.level": "high",
        },
        intent_expectation=_expectation(
            initial_source="explicit",
            final_source="explicit",
            initial_status="sufficient",
            clarification="must_not_ask",
        ),
        coverage=("high_miss", "severity_calibration", "optional_low_finding", "security"),
        annotation_rationale="The high-severity TLS defect is required; the separate low diagnostic regression is optional and must not hide it.",
    ),
)


def _is_link_or_reparse(metadata: os.stat_result) -> bool:
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse_flag)


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _safe_relative_parts(relative: str, *, context: str) -> tuple[str, ...]:
    if type(relative) is not str or not relative:
        raise ValueError("%s must be a non-empty relative POSIX path" % context)
    if "\\" in relative or "\x00" in relative:
        raise ValueError("%s contains an unsafe path separator or NUL" % context)
    raw_parts = relative.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError("%s contains an empty or dot path component" % context)
    if any(":" in part for part in raw_parts):
        raise ValueError("%s contains a Windows drive or stream component" % context)
    if any(part.endswith((".", " ")) for part in raw_parts):
        raise ValueError(
            "%s contains a Windows trailing dot or space component" % context
        )
    if any(_is_windows_reserved_component(part) for part in raw_parts):
        raise ValueError(
            "%s contains a Windows reserved device name component" % context
        )
    if any(_is_dos_83_alias_component(part) for part in raw_parts):
        raise ValueError(
            "%s contains a possible Windows 8.3 short-name alias component"
            % context
        )
    pure = PurePosixPath(relative)
    if pure.is_absolute() or tuple(pure.parts) != tuple(raw_parts):
        raise ValueError("%s is not a canonical relative POSIX path" % context)
    return tuple(raw_parts)


def _assert_safe_existing_directory(path: Path, *, context: str) -> None:
    metadata = path.lstat()
    if _is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError("%s is a link, reparse point, or non-directory" % context)


def _assert_safe_root(root: Path, *, context: str, create: bool = False) -> Path:
    absolute = _absolute_lexical(root)
    existing = absolute
    missing: list[Path] = []
    while not os.path.lexists(existing):
        missing.append(existing)
        parent = existing.parent
        if parent == existing:
            raise RuntimeError("%s has no existing directory ancestor" % context)
        existing = parent
    _assert_safe_existing_directory(existing, context=context + " ancestor")
    for ancestor in reversed(existing.parents):
        if ancestor == ancestor.parent:
            continue
        if os.path.lexists(ancestor):
            _assert_safe_existing_directory(ancestor, context=context + " ancestor")
    if not create:
        return absolute
    for directory in reversed(missing):
        directory.mkdir()
        _assert_safe_existing_directory(directory, context=context)
    _assert_safe_existing_directory(absolute, context=context)
    return absolute


def _safe_directory(
    root: Path,
    parts: tuple[str, ...],
    *,
    context: str,
    create: bool,
) -> Path:
    current = _assert_safe_root(root, context=context + " root", create=create)
    for component in parts:
        current = current / component
        if not os.path.lexists(current):
            if not create:
                return current
            current.mkdir()
        _assert_safe_existing_directory(current, context=context)
    return current


def _safe_target(
    root: Path,
    relative: str,
    *,
    context: str,
    create_parents: bool,
) -> Path:
    parts = _safe_relative_parts(relative, context=context)
    parent = _safe_directory(
        root,
        parts[:-1],
        context=context + " parent",
        create=create_parents,
    )
    if not os.path.lexists(parent):
        return _absolute_lexical(root).joinpath(*parts)
    return parent / parts[-1]


def _write_bytes_safely(root: Path, relative: str, data: bytes) -> None:
    if type(data) is not bytes:
        raise TypeError("generated output values must be bytes")
    target = _safe_target(
        root,
        relative,
        context="generated output %r" % relative,
        create_parents=True,
    )
    if os.path.lexists(target):
        metadata = target.lstat()
        if _is_link_or_reparse(metadata) or not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(
                "generated output target is a link, reparse point, or special file: %s"
                % relative
            )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=".core-authoring-",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _write_tree(root: Path, files: Mapping[str, str]) -> None:
    validated: list[tuple[str, str]] = []
    for relative, text in sorted(files.items()):
        _safe_relative_parts(relative, context="fixture source path")
        if type(text) is not str:
            raise TypeError("fixture source values must be strings")
        validated.append((relative, text))
    for relative, text in validated:
        _write_bytes_safely(root, relative, text.encode("utf-8"))


def _json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    if pretty:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    else:
        rendered = canonical_json(value)
    if pretty:
        rendered += "\n"
    return rendered.encode("utf-8")


def _case_source_content_hash(spec: CoreCaseSpec, built: Any) -> str:
    return canonical_sha256(
        {
            "schema_version": CASE_PROVENANCE_SCHEMA_VERSION,
            "task_id": spec.task_id,
            "case_version": CASE_VERSION,
            "source_version": CORE_SOURCE_VERSION,
            "annotation_protocol_version": ANNOTATION_PROTOCOL_VERSION,
            "review_target_kind": "repository",
            "review_request": dict(spec.request),
            "fixture_source_digest": built.source_digest,
            "base_tree": built.base_tree,
            "head_tree": built.head_tree,
        }
    )


def _repository_binding(built: Any) -> dict[str, str]:
    return {
        "base_revision": built.base_revision,
        "head_revision": built.head_revision,
        "base_tree": built.base_tree,
        "head_tree": built.head_tree,
        "source_digest": built.source_digest,
    }


def _blind_review_packet(
    case: EvalCase,
    built: Any,
    fixture_manifest: Mapping[str, Any],
    protocol_binding: Mapping[str, str],
) -> dict[str, Any]:
    return make_packet(
        case,
        _repository_binding(built),
        fixture_manifest,
        protocol_binding,
    )


def _pending_human_review(
    case: EvalCase,
    built: Any,
    fixture_manifest: Mapping[str, Any],
    protocol_binding: Mapping[str, str],
) -> dict[str, Any]:
    packet = _blind_review_packet(case, built, fixture_manifest, protocol_binding)
    return {
        "schema_version": HUMAN_REVIEW_RECORD_SCHEMA_VERSION,
        "status": HUMAN_REVIEW_STATUS,
        "approval_identity_status": "requires_re_review",
        "prior_approval_carried_forward": False,
        "final_decision": None,
        "task_id": case.task_id,
        "case_version": case.case_version,
        "canonical_case_digest": canonical_sha256(case),
        "eval_input_digest": case.eval_input().digest(),
        "base_revision": built.base_revision,
        "head_revision": built.head_revision,
        "base_tree": built.base_tree,
        "head_tree": built.head_tree,
        "annotation_protocol_version": ANNOTATION_PROTOCOL_VERSION,
        "blind_review_packet_digest": packet["packet_digest"],
        "author_id": None,
        "reviewer_id": None,
        "adjudicator_id": None,
        "review_batch_id": None,
        "blind_review_started_at": None,
        "blind_review_completed_at": None,
        "independent_annotation_digest": None,
        "adjudication_digest": None,
        "leakage_review_completed": False,
    }


def _annotation_record(
    spec: CoreCaseSpec,
    built: Any,
    case: EvalCase,
    fixture_manifest: Mapping[str, Any],
    protocol_binding: Mapping[str, str],
    ledger_root: Path,
) -> dict[str, Any]:
    human_review = _pending_human_review(
        case, built, fixture_manifest, protocol_binding
    )
    checklist = {
        "agent_input_contains_no_truth": True,
        "atomic_findings_reviewed": False,
        "base_head_binding_reproduced": True,
        "evidence_anchors_are_non_exclusive": False,
        "fixture_contains_no_vcs_metadata": True,
        "known_invalid_traps_reviewed": False,
        "semantic_truth_leakage_reviewed": False,
        "severity_category_context_reviewed": False,
        "truth_completeness_reviewed": False,
        "human_review_completed": False,
    }
    annotation = {
        "schema_version": ANNOTATION_RECORD_SCHEMA_VERSION,
        "task_id": spec.task_id,
        "case_version": CASE_VERSION,
        "suite_id": spec.suite_id,
        "authoring": {
            "kind": "ai_assisted",
            "source_version": CORE_SOURCE_VERSION,
            "author_id": None,
            "truth_frozen_at": None,
            "status": "draft_requires_independent_re_review",
        },
        "intent_expectation": dict(spec.intent_expectation),
        "case_binding": {
            "canonical_case_digest": canonical_sha256(case),
            "eval_input_digest": case.eval_input().digest(),
            "case_source_content_hash": case.source.content_hash,
        },
        "provenance_binding": {
            "schema_version": CASE_PROVENANCE_SCHEMA_VERSION,
            "annotation_protocol_version": ANNOTATION_PROTOCOL_VERSION,
            "content_hash": case.source.content_hash,
        },
        "repository_binding": {
            "base_revision": built.base_revision,
            "head_revision": built.head_revision,
            "base_tree": built.base_tree,
            "head_tree": built.head_tree,
            "source_digest": built.source_digest,
        },
        "truth_summary": {
            "expected_intent_ids": [
                item.truth_id for item in case.intent_truth.expected_claims
            ],
            "forbidden_intent_ids": [
                item.truth_id for item in case.intent_truth.forbidden_claims
            ],
            "expected_finding_ids": [
                item.truth_id for item in case.review_truth.expected_findings
            ],
            "known_invalid_finding_ids": [
                item.truth_id for item in case.review_truth.known_invalid_findings
            ],
            "truth_completeness": case.review_truth.completeness.value,
        },
        "coverage": sorted(spec.coverage),
        "suite_assignment": {
            "split": spec.split,
            "status": (
                "pending_current_agent_baseline"
                if spec.split == "regression"
                else "capability"
            ),
            "promotion_evidence": None,
        },
        "annotation_rationale": spec.annotation_rationale,
        "checklist": checklist,
        "human_review": human_review,
        "disagreements": {
            "status": "requires_independent_re_review",
            "items": [],
            "adjudicator_id": None,
            "adjudication_digest": None,
        },
    }
    ledger_record = load_source_bound_ledger_record(
        ledger_root,
        case,
        _repository_binding(built),
        fixture_manifest,
        protocol_binding,
    )
    return project_ledger_record(annotation, ledger_record)


def _golden_intent(case: EvalCase) -> dict[str, Any]:
    return {
        "status": "sufficient",
        "goal": None,
        "acceptance_criteria": [],
        "scope": [],
        "constraints": [],
        "claims": [
            {
                "claim_id": "golden-" + item.truth_id,
                "dimension": item.dimension.value,
                "text": item.text,
                "source": "explicit",
            }
            for item in case.intent_truth.expected_claims
        ],
        "clarification_questions": [],
        "uncertainties": [],
    }


def _submission_finding(
    finding_id: str,
    truth: Any,
    *,
    evidence_refs: Iterable[str] = (),
    claim: str | None = None,
    severity: str | None = None,
    path: str | None = None,
    side: str | None = None,
    from_line: int | None = None,
    to_line: int | None = None,
) -> dict[str, Any]:
    locations = tuple(truth.locations)
    primary = None if not locations else locations[0]
    return {
        "finding_id": finding_id,
        "claim": truth.claim if claim is None else claim,
        "severity": (
            getattr(truth, "severity", None).value
            if severity is None and getattr(truth, "severity", None) is not None
            else ("medium" if severity is None else severity)
        ),
        "path": (
            (None if primary is None else primary.path) if path is None else path
        ),
        "side": (
            (None if primary is None or primary.side is None else primary.side.value)
            if side is None
            else side
        ),
        "from_line": (
            (None if primary is None else primary.from_line)
            if from_line is None
            else from_line
        ),
        "to_line": (
            (None if primary is None else primary.to_line)
            if to_line is None
            else to_line
        ),
        "evidence_refs": list(evidence_refs),
        "suggested_action": "Correct the behavior while preserving the stated contract.",
    }


def _golden_run_id() -> str:
    return stable_id(
        "run",
        {
            "schema_version": GOLDEN_RUN_BINDING_SCHEMA_VERSION,
            "run_instance_key": GOLDEN_RUN_INSTANCE_KEY,
            "source_version": CORE_SOURCE_VERSION,
            "scenario_order": list(GOLDEN_SCENARIOS),
        },
    )


def _golden_trial_id(case: EvalCase, scenario: str) -> str:
    try:
        trial_index = GOLDEN_SCENARIOS.index(scenario) + 1
    except ValueError as exc:
        raise ValueError("Golden scenario is not registered: %s" % scenario) from exc
    return stable_id("trial", _golden_run_id(), case.task_id, trial_index)


def _golden_replay_binding_digest(case: EvalCase) -> str:
    target = case.input.review_target
    if target.kind.value != "repository":
        raise ValueError("Core Golden replay requires a Repository review target")
    return canonical_sha256(
        {
            "schema_version": GOLDEN_REPLAY_BINDING_SCHEMA_VERSION,
            "task_id": case.task_id,
            "repository": target.repository.to_dict(),
        }
    )


def _golden_materialization_id(case: EvalCase, scenario: str) -> str:
    trial_id = _golden_trial_id(case, scenario)
    return stable_id(
        "materialization",
        {
            "schema_version": GOLDEN_MATERIALIZATION_BINDING_SCHEMA_VERSION,
            "run_id": _golden_run_id(),
            "task_id": case.task_id,
            "trial_id": trial_id,
            "attempt": GOLDEN_ATTEMPT,
            "eval_input_digest": case.eval_input().digest(),
            "review_target_digest": canonical_sha256(case.input.review_target),
            "wire_contract": dict(REPOSITORY_WIRE_CONTRACT),
            "suite_preparation_binding_digest": None,
            "replay_binding_digest": _golden_replay_binding_digest(case),
        },
    )


def _bind_golden_evidence(
    values: Iterable[Mapping[str, Any]], target_materialization_id: str
) -> list[dict[str, Any]]:
    result = []
    for value in values:
        item = dict(value)
        source = dict(item["source"])
        if "target_materialization_id" in source:
            raise ValueError("unbound Golden Evidence unexpectedly carries an identity")
        source["target_materialization_id"] = target_materialization_id
        item["source"] = source
        result.append(item)
    return result


def _golden_submission(
    case: EvalCase,
    scenario: str,
    *,
    findings: Iterable[Mapping[str, Any]],
    evidence: Iterable[Mapping[str, Any]] = (),
    intent: Mapping[str, Any] | None = None,
) -> EvalSubmission:
    trial_id = _golden_trial_id(case, scenario)
    materialization_id = _golden_materialization_id(case, scenario)
    return EvalSubmission.from_dict(
        {
            "schema_version": EVAL_SUBMISSION_SCHEMA_VERSION,
            "task_id": case.task_id,
            "agent_id": "core-golden-agent-v2",
            "trial_id": trial_id,
            "eval_input_digest": case.eval_input().digest(),
            "target_materialization_id": materialization_id,
            "status": "completed",
            "intent": _golden_intent(case) if intent is None else dict(intent),
            "review": {
                "findings": list(findings),
                "uncertainties": [],
            },
            "evidence": _bind_golden_evidence(evidence, materialization_id),
            "usage": {
                "elapsed_seconds": 0,
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "tool_calls": None,
                "cost_amount": None,
                "cost_currency": None,
            },
            "trace_ref": None,
            "failure": None,
        }
    )


def _snapshot_excerpt(
    spec: CoreCaseSpec,
    snapshot: str,
    path: str,
    from_line: int,
    to_line: int,
) -> str:
    if snapshot == "base":
        files = spec.base_files
    elif snapshot == "head":
        files = spec.head_files
    else:
        raise ValueError("snapshot must be base or head")
    lines = files[path].splitlines(keepends=True)
    return "".join(lines[from_line - 1 : to_line])


def _file_evidence(
    evidence_id: str,
    case: EvalCase,
    *,
    path: str,
    from_line: int,
    to_line: int,
    excerpt: str,
    revision: str | None = None,
    content_hash: str | None = None,
) -> dict[str, Any]:
    return {
        "evidence_id": evidence_id,
        "source": {
            "kind": "repository_file",
            "revision": (
                case.input.review_target.repository.head_revision
                if revision is None
                else revision
            ),
            "path": path,
            "from_line": from_line,
            "to_line": to_line,
        },
        "content_hash": (
            hashlib.sha256(excerpt.encode("utf-8")).hexdigest()
            if content_hash is None
            else content_hash
        ),
        "excerpt": excerpt,
    }


def _full_file_evidence(
    evidence_id: str,
    case: EvalCase,
    spec: CoreCaseSpec,
    *,
    snapshot: str,
    path: str,
) -> dict[str, Any]:
    files = spec.base_files if snapshot == "base" else spec.head_files
    line_count = len(files[path].splitlines())
    revision = (
        case.input.review_target.repository.base_revision
        if snapshot == "base"
        else case.input.review_target.repository.head_revision
    )
    return _file_evidence(
        evidence_id,
        case,
        path=path,
        from_line=1,
        to_line=line_count,
        excerpt=_snapshot_excerpt(spec, snapshot, path, 1, line_count),
        revision=revision,
    )


def _base_head_file_evidence(
    prefix: str,
    case: EvalCase,
    spec: CoreCaseSpec,
    path: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        _full_file_evidence(
            prefix + "-base", case, spec, snapshot="base", path=path
        ),
        _full_file_evidence(
            prefix + "-head", case, spec, snapshot="head", path=path
        ),
    )


def _golden_outputs(
    cases: Mapping[str, EvalCase],
    specs: Mapping[str, CoreCaseSpec],
) -> dict[str, bytes]:
    result: dict[str, bytes] = {}

    perfect_case = cases["core-py-001"]
    perfect_spec = specs[perfect_case.task_id]
    perfect_truth = perfect_case.review_truth.expected_findings[0]
    perfect_evidence = _base_head_file_evidence(
        "evidence-perfect", perfect_case, perfect_spec, "src/timeout.py"
    )
    perfect_refs = tuple(item["evidence_id"] for item in perfect_evidence)
    perfect = _golden_submission(
        perfect_case,
        "perfect",
        findings=(
            _submission_finding(
                "finding-perfect",
                perfect_truth,
                evidence_refs=perfect_refs,
            ),
        ),
        evidence=perfect_evidence,
    )

    empty_case = cases["core-py-011"]
    empty = _golden_submission(empty_case, "empty", findings=())

    duplicate_case = cases["core-py-001"]
    duplicate_truth = duplicate_case.review_truth.expected_findings[0]
    duplicate = _golden_submission(
        duplicate_case,
        "duplicate",
        findings=(
            _submission_finding(
                "finding-perfect", duplicate_truth, evidence_refs=perfect_refs
            ),
            _submission_finding(
                "finding-duplicate", duplicate_truth, evidence_refs=perfect_refs
            ),
        ),
        evidence=perfect_evidence,
    )

    fabricated_case = cases["core-py-015"]
    fabricated_spec = specs[fabricated_case.task_id]
    fabricated_truth = fabricated_case.review_truth.known_invalid_findings[0]
    fabricated_evidence = _base_head_file_evidence(
        "evidence-fabricated",
        fabricated_case,
        fabricated_spec,
        "src/renderer.py",
    )
    fabricated_refs = tuple(item["evidence_id"] for item in fabricated_evidence)
    fabricated = _golden_submission(
        fabricated_case,
        "fabricated",
        findings=(
            _submission_finding(
                "finding-fabricated",
                fabricated_truth,
                evidence_refs=fabricated_refs,
            ),
        ),
        evidence=fabricated_evidence,
    )

    bad_case = cases["core-py-014"]
    bad_spec = specs[bad_case.task_id]
    bad_truth = bad_case.review_truth.expected_findings[0]
    valid_bad_evidence = _base_head_file_evidence(
        "evidence-bad", bad_case, bad_spec, "src/token.py"
    )
    valid_bad_base, valid_bad_head = valid_bad_evidence
    bad_variants = (
        (
            "bad-evidence",
            "finding-bad-evidence-hash",
            {**valid_bad_head, "content_hash": "0" * 64},
        ),
        (
            "bad-evidence-path",
            "finding-bad-evidence-path",
            {
                **valid_bad_head,
                "source": {
                    **valid_bad_head["source"],
                    "path": "src/tokens.py",
                },
            },
        ),
        (
            "bad-evidence-line",
            "finding-bad-evidence-line",
            {
                **valid_bad_head,
                "source": {
                    **valid_bad_head["source"],
                    "from_line": 99,
                    "to_line": 99,
                },
            },
        ),
    )
    bad_submissions = []
    for scenario, finding_id, invalid_head in bad_variants:
        evidence = (valid_bad_base, invalid_head)
        refs = tuple(item["evidence_id"] for item in evidence)
        bad_submissions.append(
            (
                scenario,
                _golden_submission(
                    bad_case,
                    scenario,
                    findings=(
                        _submission_finding(
                            finding_id,
                            bad_truth,
                            evidence_refs=refs,
                        ),
                    ),
                    evidence=evidence,
                ),
            )
        )

    unsupported_evidence_case = cases["core-py-001"]
    unsupported_evidence_spec = specs[unsupported_evidence_case.task_id]
    unsupported_evidence_truth = (
        unsupported_evidence_case.review_truth.expected_findings[0]
    )
    unrelated_excerpt = _snapshot_excerpt(
        unsupported_evidence_spec,
        "head",
        "src/timeout.py",
        1,
        1,
    )
    unrelated_evidence = _file_evidence(
        "evidence-valid-unrelated",
        unsupported_evidence_case,
        path="src/timeout.py",
        from_line=1,
        to_line=1,
        excerpt=unrelated_excerpt,
    )
    unsupported_evidence = _golden_submission(
        unsupported_evidence_case,
        "unsupported-evidence",
        findings=(
            _submission_finding(
                "finding-unsupported-evidence",
                unsupported_evidence_truth,
                evidence_refs=("evidence-valid-unrelated",),
            ),
        ),
        evidence=(unrelated_evidence,),
    )

    compound_case = cases["core-py-012"]
    compound_spec = specs[compound_case.task_id]
    compound_truths = compound_case.review_truth.expected_findings
    assert len(compound_truths) == 2
    compound_evidence = _base_head_file_evidence(
        "evidence-compound", compound_case, compound_spec, "src/order.py"
    )
    compound_refs = tuple(item["evidence_id"] for item in compound_evidence)
    compound_claim = "%s In addition, %s" % (
        compound_truths[0].claim,
        compound_truths[1].claim,
    )
    compound = _golden_submission(
        compound_case,
        "compound",
        findings=(
            _submission_finding(
                "finding-compound",
                compound_truths[0],
                claim=compound_claim,
                evidence_refs=compound_refs,
            ),
        ),
        evidence=compound_evidence,
    )

    unknown_case = cases["core-py-015"]
    unknown_spec = specs[unknown_case.task_id]
    unknown_truth = unknown_case.review_truth.known_invalid_findings[0]
    unknown_evidence = _base_head_file_evidence(
        "evidence-judge-unknown",
        unknown_case,
        unknown_spec,
        "src/renderer.py",
    )
    unknown_refs = tuple(item["evidence_id"] for item in unknown_evidence)
    unknown = _golden_submission(
        unknown_case,
        "judge-unknown",
        findings=(
            _submission_finding(
                "finding-judge-unknown",
                unknown_truth,
                claim=(
                    "The changed encoded output may violate an exact literal-quote "
                    "contract required by repository-external consumers."
                ),
                severity="medium",
                evidence_refs=unknown_refs,
            ),
        ),
        evidence=unknown_evidence,
    )

    unsupported_intent_case = cases["core-py-004"]
    unsupported_intent_payload = _golden_intent(unsupported_intent_case)
    unsupported_intent_payload["claims"].append(
        {
            "claim_id": "golden-extra-0001",
            "dimension": "scope",
            "text": "Migrate persistence to SQLite as part of this change.",
            "source": "inferred",
        }
    )
    unsupported_intent = _golden_submission(
        unsupported_intent_case,
        "unsupported-intent",
        findings=(),
        intent=unsupported_intent_payload,
    )

    contradicted_intent_case = cases["core-py-011"]
    contradicted_intent_spec = specs[contradicted_intent_case.task_id]
    contradicted_intent_payload = _golden_intent(contradicted_intent_case)
    contradicted_intent_payload["status"] = "partial"
    forbidden_claim = contradicted_intent_case.intent_truth.forbidden_claims[0]
    contradicted_intent_payload["claims"].append(
        {
            "claim_id": "golden-extra-0002",
            "dimension": forbidden_claim.dimension.value,
            "text": forbidden_claim.text,
            "source": "explicit",
        }
    )
    contradicted_truth = contradicted_intent_case.review_truth.expected_findings[0]
    contradicted_evidence = (
        *_base_head_file_evidence(
            "evidence-contradicted-api",
            contradicted_intent_case,
            contradicted_intent_spec,
            "src/api.py",
        ),
        _full_file_evidence(
            "evidence-contradicted-database-head",
            contradicted_intent_case,
            contradicted_intent_spec,
            snapshot="head",
            path="src/database.py",
        ),
        _full_file_evidence(
            "evidence-contradicted-routes-head",
            contradicted_intent_case,
            contradicted_intent_spec,
            snapshot="head",
            path="src/routes.py",
        ),
    )
    contradicted_refs = tuple(item["evidence_id"] for item in contradicted_evidence)
    contradicted_intent = _golden_submission(
        contradicted_intent_case,
        "contradicted-intent",
        findings=(
            _submission_finding(
                "finding-contradicted-intent-baseline",
                contradicted_truth,
                evidence_refs=contradicted_refs,
            ),
        ),
        evidence=contradicted_evidence,
        intent=contradicted_intent_payload,
    )

    submissions = [
        ("perfect", perfect),
        ("empty", empty),
        ("duplicate", duplicate),
        ("fabricated", fabricated),
        ("unsupported-evidence", unsupported_evidence),
        ("compound", compound),
        ("judge-unknown", unknown),
        ("unsupported-intent", unsupported_intent),
        ("contradicted-intent", contradicted_intent),
        *bad_submissions,
    ]
    scenario_names = [scenario for scenario, _submission in submissions]
    if len(scenario_names) != len(set(scenario_names)) or set(
        scenario_names
    ) != set(GOLDEN_SCENARIOS):
        raise ValueError("Golden outputs do not match the fixed scenario registry")
    for scenario, submission in submissions:
        result[
            "cases/core/%s/golden/%s.json" % (submission.task_id, scenario)
        ] = submission.to_json().encode("utf-8")
    return result


def _golden_index(
    golden_outputs: Mapping[str, bytes],
    cases: Mapping[str, EvalCase],
) -> dict[str, Any]:
    entries = []
    for path, raw in sorted(golden_outputs.items()):
        parts = PurePosixPath(path).parts
        if (
            len(parts) != 5
            or parts[:2] != ("cases", "core")
            or parts[3] != "golden"
            or not parts[4].endswith(".json")
        ):
            raise ValueError("Golden output path is not canonical: %s" % path)
        task_id = parts[2]
        case = cases[task_id]
        submission = EvalSubmission.from_json(raw)
        case.validate_submission(submission)
        core = {
            "path": path,
            "task_id": task_id,
            "suite_id": case.source.suite,
            "scenario": parts[4][:-5],
            "raw_file_size_bytes": len(raw),
            "raw_file_sha256": hashlib.sha256(raw).hexdigest(),
            "canonical_submission_digest": submission.digest(),
            "eval_input_digest": submission.eval_input_digest,
            "trial_id": submission.trial_id,
            "target_materialization_id": submission.target_materialization_id,
        }
        entry_digest = canonical_sha256(
            {"schema_version": GOLDEN_ENTRY_SCHEMA_VERSION, **core}
        )
        entries.append({**core, "entry_digest": entry_digest})
    return {
        "schema_version": GOLDEN_INDEX_SCHEMA_VERSION,
        "source_version": CORE_SOURCE_VERSION,
        "run_binding": {
            "schema_version": GOLDEN_RUN_BINDING_SCHEMA_VERSION,
            "run_instance_key": GOLDEN_RUN_INSTANCE_KEY,
            "run_id": _golden_run_id(),
            "attempt": GOLDEN_ATTEMPT,
            "scenario_order": list(GOLDEN_SCENARIOS),
        },
        "entries": entries,
    }


def build_plan(
    temporary_root: Path | None = None,
    approval_ledger_root: Path | None = None,
) -> CoreBuildPlan:
    writable_outputs: dict[str, bytes] = {}
    check_only_fixtures: dict[str, bytes] = {}
    ledger_root = (
        REPOSITORY_ROOT / "eval" / "human-reviews"
        if approval_ledger_root is None
        else _absolute_lexical(approval_ledger_root)
    )
    protocol_binding = annotation_protocol_binding(REPOSITORY_ROOT / "eval")
    suite_cases: dict[str, list[tuple[CoreCaseSpec, EvalCase, bytes]]] = {
        "core-regression": [],
        "core-capability": [],
    }
    suite_source_bindings: dict[str, list[dict[str, str]]] = {
        "core-regression": [],
        "core-capability": [],
    }
    cases_by_id: dict[str, EvalCase] = {}
    specs_by_id = {spec.task_id: spec for spec in CASES}

    temporary_parent = (
        None
        if temporary_root is None
        else str(
            _assert_safe_root(
                temporary_root,
                context="authoring temporary root",
                create=True,
            )
        )
    )
    with tempfile.TemporaryDirectory(
        prefix="review-agent-core-suite-",
        dir=temporary_parent,
    ) as temporary:
        temporary_root = Path(temporary)
        for spec in CASES:
            fixture = temporary_root / "fixtures" / spec.task_id
            _write_tree(fixture / "base", spec.base_files)
            _write_tree(fixture / "head", spec.head_files)
            built = FixtureRepositoryBuilder().build(
                fixture,
                temporary_root / "repositories" / (spec.task_id + ".git"),
            )
            relative_root = f"cases/core/{spec.task_id}"
            for snapshot_name, files in (
                ("base", spec.base_files),
                ("head", spec.head_files),
            ):
                for relative, text in sorted(files.items()):
                    check_only_fixtures[
                        f"{relative_root}/repository/{snapshot_name}/{relative}"
                    ] = text.encode("utf-8")

            payload = {
                "schema_version": EVAL_CASE_SCHEMA_VERSION,
                "task_id": spec.task_id,
                "case_version": CASE_VERSION,
                "source": {
                    "suite": spec.suite_id,
                    "origin": "hand_authored",
                    "source_id": spec.task_id + "-source",
                    "source_version": CORE_SOURCE_VERSION,
                    "source_uri": None,
                    "license": None,
                    "content_hash": _case_source_content_hash(spec, built),
                },
                "input": {
                    "review_target": {
                        "kind": "repository",
                        "repository": {
                            "source": "fixture",
                            "path": f"{relative_root}/repository",
                            "url": None,
                            "base_revision": built.base_revision,
                            "head_revision": built.head_revision,
                        },
                        "review_request": dict(spec.request),
                    },
                },
                "clarification_script": {
                    "max_rounds": max(1, len(spec.clarification_answers)),
                    "answers": list(spec.clarification_answers),
                },
                "intent_truth": dict(spec.intent_truth),
                "review_truth": {
                    "completeness": "closed_world",
                    "novel_finding_policy": (
                        "verify"
                        if "judge_unknown_carrier" in spec.coverage
                        else "forbid"
                    ),
                    "expected_findings": list(spec.expected_findings),
                    "known_invalid_findings": list(spec.known_invalid_findings),
                },
                "review_evaluator_context": {"truth_contexts": []},
            }
            case = EvalCase.from_dict(payload)
            fixture_manifest = fixture_manifest_from_mappings(
                spec.base_files, spec.head_files
            )
            cases_by_id[case.task_id] = case
            case_bytes = _json_bytes(case.to_dict())
            writable_outputs[f"{relative_root}/case.json"] = case_bytes
            writable_outputs[f"{relative_root}/annotation.json"] = _json_bytes(
                _annotation_record(
                    spec,
                    built,
                    case,
                    fixture_manifest,
                    protocol_binding,
                    ledger_root,
                ),
                pretty=True,
            )
            suite_cases[spec.suite_id].append((spec, case, case_bytes))
            suite_source_bindings[spec.suite_id].append(
                {
                    "task_id": spec.task_id,
                    "case_source_content_hash": case.source.content_hash,
                }
            )

    golden_outputs = _golden_outputs(cases_by_id, specs_by_id)
    golden_index = _golden_index(golden_outputs, cases_by_id)
    writable_outputs.update(golden_outputs)
    writable_outputs["cases/core/golden-index.json"] = _json_bytes(golden_index)

    for suite_id, bound_cases in suite_cases.items():
        suite_golden_digests = sorted(
            entry["entry_digest"]
            for entry in golden_index["entries"]
            if entry["suite_id"] == suite_id
        )
        source_content_hash = canonical_sha256(
            {
                "schema_version": SUITE_SOURCE_PACKET_SCHEMA_VERSION,
                "suite_id": suite_id,
                "source_version": CORE_SOURCE_VERSION,
                "cases": sorted(
                    suite_source_bindings[suite_id],
                    key=lambda item: item["task_id"],
                ),
                "golden_entry_digests": suite_golden_digests,
            }
        )
        manifest_payload = {
            "schema_version": SUITE_MANIFEST_SCHEMA_VERSION,
            "suite_id": suite_id,
            "suite_version": CORE_SOURCE_VERSION,
            "wire_contract": dict(REPOSITORY_WIRE_CONTRACT),
            "source": {
                "kind": "core",
                "source_id": suite_id + "-source",
                "source_version": CORE_SOURCE_VERSION,
                "source_uri": None,
                "license": None,
                "content_hash": source_content_hash,
                "preparation_binding": None,
            },
            "cases": [],
        }
        for spec, case, case_bytes in bound_cases:
            dimensions = [
                {"name": name, "value": value}
                for name, value in sorted(spec.dimensions.items())
            ]
            manifest_payload["cases"].append(
                {
                    "task_id": spec.task_id,
                    "case_version": CASE_VERSION,
                    "path": f"cases/core/{spec.task_id}/case.json",
                    "split": spec.split,
                    "protocol_id": PROTOCOL_ID,
                    "dimensions": dimensions,
                    "raw_file_size_bytes": len(case_bytes),
                    "raw_file_sha256": hashlib.sha256(case_bytes).hexdigest(),
                    "canonical_case_digest": case.digest(),
                    "eval_input_digest": case.eval_input().digest(),
                    "truth_completeness": case.review_truth.completeness.value,
                }
            )
        manifest = SuiteManifest.from_dict(manifest_payload)
        writable_outputs[f"suites/{suite_id}/manifest.json"] = _json_bytes(
            manifest.to_dict()
        )

    writable_outputs["cases/core/README.md"] = (
        "# Core code-review Cases\n\n"
        "Derived Core artifacts are generated by "
        "`eval/authoring/build_core_suites.py`. The sole active projection "
        "contains 18 `eval_case_v2` Cases, 12 "
        "`eval_submission_v2` Golden submissions, and two `suite_manifest_v2` "
        "Repository Suites at source version `core-2026-07-21-v3`. Each Case keeps "
        "private truth and its audit record beside deterministic `repository/base` "
        "and `repository/head` fixture bytes. The Agent receives only the canonical "
        "EvalInput v2 Repository review target and a per-attempt materialization; "
        "truth and audit sidecars remain evaluator-private. Repository fixture bytes "
        "are immutable, check-only trust inputs: `--write` validates them before any "
        "derived output write and never creates, repairs, or overwrites them.\n\n"
        "Golden Submission identities use the fixed `core-golden-authoring-v2` run "
        "binding, the scenario order recorded in `golden-index.json`, attempt 1, "
        "and a replay digest over the exact Repository descriptor. These deterministic "
        "fixtures exercise protocol semantics; they are not real Agent trials and "
        "cannot satisfy promotion. Run the authoring script with `--check` after any "
        "edit.\n\n"
        "These manifests are pending candidate snapshots, not release-ready Suites. "
        "The v3 Case/Input digests supersede every earlier approval identity, so each "
        "Case explicitly requires a fresh independent human Reviewer B review. A "
        "Regression candidate must additionally pass a real current-Agent model "
        "baseline with at least three trials for every Regression Case. AI, scripted, "
        "or internally consistent fake evidence cannot satisfy either external gate.\n"
    ).encode("utf-8")
    return CoreBuildPlan(
        writable_outputs=writable_outputs,
        check_only_fixtures=check_only_fixtures,
    )


def _walk_generated_files(root: Path, eval_root: Path) -> Iterable[str]:
    with os.scandir(root) as entries:
        ordered = sorted(entries, key=lambda item: item.name)
    for entry in ordered:
        path = Path(entry.path)
        metadata = entry.stat(follow_symlinks=False)
        if _is_link_or_reparse(metadata):
            raise RuntimeError(
                "generated artifact tree contains a link or reparse point: %s"
                % path.relative_to(eval_root).as_posix()
            )
        if stat.S_ISDIR(metadata.st_mode):
            yield from _walk_generated_files(path, eval_root)
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(
                "generated artifact tree contains a special file: %s"
                % path.relative_to(eval_root).as_posix()
            )
        if "__pycache__" not in path.parts:
            yield path.relative_to(eval_root).as_posix()


def _portable_path_identity(parts: tuple[str, ...]) -> str:
    return "/".join(
        unicodedata.normalize(
            "NFC",
            unicodedata.normalize("NFC", part).casefold(),
        )
        for part in parts
    )


def _existing_generated_files(eval_root: Path) -> dict[str, str]:
    eval_root = _assert_safe_root(eval_root, context="Eval authoring root")
    roots = (
        ("cases", "core"),
        ("suites", "core-regression"),
        ("suites", "core-capability"),
    )
    result: dict[str, str] = {}
    for parts in roots:
        root = _safe_directory(
            eval_root,
            parts,
            context="generated artifact root",
            create=False,
        )
        if not os.path.lexists(root):
            continue
        for relative in _walk_generated_files(root, eval_root):
            relative_parts = _safe_relative_parts(
                relative,
                context="existing Core file path",
            )
            portable_key = _portable_path_identity(relative_parts)
            previous = result.get(portable_key)
            if previous is not None and previous != relative:
                raise RuntimeError(
                    "existing Core files collide portably: %s and %s"
                    % (previous, relative)
                )
            result[portable_key] = relative
    return result


def _is_repository_path(portable_key: str) -> bool:
    parts = tuple(portable_key.split("/"))
    return (
        len(parts) >= 4
        and parts[:2] == ("cases", "core")
        and parts[3] == "repository"
    )


def _registered_case_ids() -> frozenset[str]:
    return frozenset(spec.task_id for spec in CASES)


def _is_allowlisted_writable_shape(relative: str) -> bool:
    parts = tuple(relative.split("/"))
    registered = _registered_case_ids()
    if parts == ("cases", "core", "README.md"):
        return True
    if parts == ("cases", "core", "golden-index.json"):
        return True
    if parts == ("suites", "core-regression", "manifest.json"):
        return True
    if parts == ("suites", "core-capability", "manifest.json"):
        return True
    if len(parts) == 4 and parts[:2] == ("cases", "core"):
        return parts[2] in registered and parts[3] in {"case.json", "annotation.json"}
    return (
        len(parts) == 5
        and parts[:2] == ("cases", "core")
        and parts[2] in registered
        and parts[3] == "golden"
        and parts[4].endswith(".json")
        and _GOLDEN_TASK_BY_SCENARIO.get(parts[4][:-5]) == parts[2]
    )


def _is_repository_fixture_path(relative: str) -> bool:
    parts = tuple(relative.split("/"))
    return (
        len(parts) >= 6
        and parts[:2] == ("cases", "core")
        and parts[2] in _registered_case_ids()
        and parts[3] == "repository"
        and parts[4] in {"base", "head"}
    )


def _portable_output_paths(
    outputs: Mapping[str, bytes],
    *,
    label: str,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative, data in outputs.items():
        parts = _safe_relative_parts(relative, context=label + " path")
        if type(data) is not bytes:
            raise TypeError(label + " values must be bytes")
        portable_key = _portable_path_identity(parts)
        previous = result.get(portable_key)
        if previous is not None:
            raise ValueError(
                "%s paths collide portably: %s and %s"
                % (label, previous, relative)
            )
        result[portable_key] = relative
    return result


def _validated_plan_paths(plan: CoreBuildPlan) -> _ValidatedBuildPlanOwnership:
    if type(plan) is not CoreBuildPlan:
        raise TypeError("Core authoring requires a CoreBuildPlan")
    writable_paths = _portable_output_paths(
        plan.writable_outputs,
        label="writable output",
    )
    fixture_paths = _portable_output_paths(
        plan.check_only_fixtures,
        label="check-only fixture",
    )
    overlap = set(writable_paths) & set(fixture_paths)
    if overlap:
        aliases = [
            "%s and %s" % (writable_paths[key], fixture_paths[key])
            for key in sorted(overlap)
        ]
        raise ValueError(
            "build-plan ownership sets overlap portably: " + ", ".join(aliases)
        )
    for portable_key, relative in writable_paths.items():
        if not _is_allowlisted_writable_shape(relative):
            raise ValueError(
                "writable output is not an allowlisted registered derived output: %s"
                % relative
            )
    for _portable_key, relative in fixture_paths.items():
        if not _is_repository_fixture_path(relative):
            raise ValueError(
                "check-only fixture must be an exact registered repository/base or repository/head path: %s"
                % relative
            )
    known_paths = dict(writable_paths)
    known_paths.update(fixture_paths)
    return _ValidatedBuildPlanOwnership(
        writable_by_portable_key=MappingProxyType(writable_paths),
        fixtures_by_portable_key=MappingProxyType(fixture_paths),
        known_by_portable_key=MappingProxyType(known_paths),
    )


def _check_expected_files(
    eval_root: Path,
    expected_files: Mapping[str, bytes],
    *,
    label: str,
) -> list[str]:
    errors: list[str] = []
    for relative, expected in sorted(expected_files.items()):
        path = _safe_target(
            eval_root,
            relative,
            context="%s %r" % (label, relative),
            create_parents=False,
        )
        if not os.path.lexists(path):
            errors.append("missing %s: %s" % (label, relative))
            continue
        metadata = path.lstat()
        if _is_link_or_reparse(metadata) or not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(
                "%s is a link, reparse point, or special file: %s"
                % (label, relative)
            )
        if path.read_bytes() != expected:
            errors.append("drifted %s: %s" % (label, relative))
    return errors


def _inventory_errors(
    existing_by_portable_key: Mapping[str, str],
    known_by_portable_key: Mapping[str, str],
) -> list[str]:
    errors: list[str] = []
    for portable_key, relative in sorted(
        existing_by_portable_key.items(),
        key=lambda item: item[1],
    ):
        expected = known_by_portable_key.get(portable_key)
        if expected is None:
            errors.append("unexpected: " + relative)
        elif relative != expected:
            errors.append(
                "unexpected portable alias: %s (expected %s)"
                % (relative, expected)
            )
    return errors


def check_outputs(eval_root: Path, plan: CoreBuildPlan) -> list[str]:
    ownership = _validated_plan_paths(plan)
    errors = _check_expected_files(
        eval_root,
        plan.writable_outputs,
        label="writable output",
    )
    errors.extend(
        _check_expected_files(
            eval_root,
            plan.check_only_fixtures,
            label="check-only fixture",
        )
    )
    errors.extend(
        _inventory_errors(
            _existing_generated_files(eval_root),
            ownership.known_by_portable_key,
        )
    )
    return errors


def write_outputs(eval_root: Path, plan: CoreBuildPlan) -> None:
    ownership = _validated_plan_paths(plan)
    fixture_errors = _check_expected_files(
        eval_root,
        plan.check_only_fixtures,
        label="check-only fixture",
    )
    if fixture_errors:
        raise RuntimeError(
            "check-only fixture validation failed: " + "; ".join(fixture_errors)
        )
    inventory_errors = _inventory_errors(
        _existing_generated_files(eval_root),
        ownership.known_by_portable_key,
    )
    if inventory_errors:
        raise RuntimeError(
            "refusing to write with unexpected generated files: "
            + "; ".join(inventory_errors)
        )
    for relative, data in sorted(plan.writable_outputs.items()):
        _write_bytes_safely(eval_root, relative, data)


def _summary(plan: CoreBuildPlan) -> dict[str, Any]:
    return {
        "schema_version": "core_suite_authoring_summary_v2",
        "case_count": len(CASES),
        "golden_count": len(GOLDEN_SCENARIOS),
        "suite_count": 2,
        "regression_count": sum(item.split == "regression" for item in CASES),
        "capability_count": sum(item.split == "capability" for item in CASES),
        "writable_generated_file_count": len(plan.writable_outputs),
        "checked_fixture_file_count": len(plan.check_only_fixtures),
        "human_review_status": HUMAN_REVIEW_STATUS,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or audit the Core Eval suites")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--write",
        action="store_true",
        help="validate fixtures, then write canonical derived files",
    )
    mode.add_argument("--check", action="store_true", help="verify committed files")
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=REPOSITORY_ROOT / "eval",
        help="Eval root containing cases/ and suites/",
    )
    parser.add_argument(
        "--temporary-root",
        type=Path,
        default=None,
        help="optional parent for deterministic fixture build scratch data",
    )
    args = parser.parse_args(argv)
    eval_root = _absolute_lexical(args.eval_root)
    temporary_root = (
        None
        if args.temporary_root is None
        else _absolute_lexical(args.temporary_root)
    )
    plan = build_plan(temporary_root)
    if args.write:
        write_outputs(eval_root, plan)
        print(json.dumps(_summary(plan), sort_keys=True))
        return 0
    errors = check_outputs(eval_root, plan)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(json.dumps(_summary(plan), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
