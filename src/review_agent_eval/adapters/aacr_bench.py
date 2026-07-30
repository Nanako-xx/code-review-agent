"""Strict offline adapter for the pinned AACR-Bench GitHub split files.

The adapter deliberately does not download data or implement scoring.  It
verifies locally acquired source bytes through ``PublicSourceManifest``, maps
the upstream records into canonical ``EvalCase`` objects, and delegates
immutable Suite publication to :mod:`review_agent_eval.adapters._public`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
import re
import unicodedata
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

from ._public import (
    PublicDatasetError,
    PublicFilterManifest,
    PublicFormatError,
    PublicPreparedCase,
    PublicPreparationResult,
    PublicRecordReceipt,
    PublicSourceIntegrityError,
    PublicSourceManifest,
    PublicStatistic,
    VerifiedPublicSource,
    write_public_suite,
)
from ..cases import (
    REPOSITORY_MATERIALIZER_PROTOCOL,
    CaseDimension,
    CaseSplit,
    WireContractV2,
)
from ..models import (
    EVAL_CASE_SCHEMA_VERSION,
    EVAL_INPUT_SCHEMA_VERSION,
    EVAL_SUBMISSION_SCHEMA_VERSION,
    CaseOrigin,
    CaseSource,
    ClarificationScript,
    DiffSide,
    EvalCase,
    EvalCaseInput,
    ExpectedFinding,
    IntentTruth,
    KnownInvalidFinding,
    MetricAuthority,
    MetricAuthoritySource,
    NovelFindingPolicy,
    Repository,
    RepositorySource,
    RepositoryReviewTarget,
    RequiredContextLevel,
    ReviewRequest,
    ReviewTruth,
    ReviewEvaluatorContext,
    ReviewTargetKind,
    SchemaError,
    TruthCompleteness,
    TruthLocation,
    _strict_json_loads,
    canonical_sha256,
    stable_id,
)


AACR_DATASET_ID = "aacr-bench"
AACR_DATASET_VERSION = "v1.0"
AACR_SOURCE_URI = "https://github.com/alibaba/aacr-bench"
AACR_SOURCE_REVISION = "082b1bc1e45377fc24ca8bdcd033d4a5a260e61c"
AACR_LICENSE = "Apache-2.0"

AACR_OFFICIAL_PROFILE = "official"
AACR_FIXTURE_PROFILE = "fixture"
AACR_FIXTURE_DATASET_ID = "aacr-bench-fixture"
AACR_FIXTURE_DATASET_VERSION = "fixture-v1"
AACR_FIXTURE_SOURCE_REVISION = "fixture-freecad-records-v1"

AACR_POSITIVE_ROLE = "positive_samples"
AACR_NEGATIVE_ROLE = "negative_samples"
AACR_REJECTION_SELECTOR = "reject_record"
AACR_LANGUAGE_SELECTOR = "language"

AACR_SUITE_ID = "aacr-bench"
AACR_FIXTURE_SUITE_ID = "aacr-bench-fixture"
AACR_PROTOCOL_ID = "aacr-repository-v2"
AACR_ADAPTER_ID = "aacr-bench"
AACR_ADAPTER_VERSION = "github-split-v2"

AACR_WIRE_CONTRACT = WireContractV2(
    case_schema_version=EVAL_CASE_SCHEMA_VERSION,
    input_schema_version=EVAL_INPUT_SCHEMA_VERSION,
    submission_schema_version=EVAL_SUBMISSION_SCHEMA_VERSION,
    review_target_kind=ReviewTargetKind.REPOSITORY,
    materializer_protocol=REPOSITORY_MATERIALIZER_PROTOCOL,
)

AACR_FIXTURE_SOURCE_MANIFEST_DIGEST = (
    "b7d211701d1a5060962625e0f0d362e82f6732c2dc11f6984cb75ae697fe8813"
)
AACR_OFFICIAL_REQUIRED_REJECTION_BINDINGS = (
    "negative_samples#/0/comments/11@43434cbc8c3a3efe6e58c078374aa68bbdb77118a95dbd98f0c65e654290774b",
    "positive_samples#/0/comments/11@2b047dafbac53c0361172b9dac0e1f2e6c8fda67618d303254a34072152568e2",
    "positive_samples#/0/comments/12@48f9fa9d84149e4e365d4fac164c47016f538b345a0708932fecf47bca17cf40",
    "positive_samples#/0/comments/2@0a5425b9d619c5f5fbd6e50747806958381c35d168ae2563fcba7e4f0603c498",
)

_POSITIVE_PATH = "dataset/positive_samples.json"
_NEGATIVE_PATH = "dataset/negative_samples.json"

_OFFICIAL_FILES = {
    AACR_POSITIVE_ROLE: (
        _POSITIVE_PATH,
        1_100_995,
        "d8683cb240249bc4e0aff6428802bdffa7b7573ace600552cab1cd0cb7e905c9",
    ),
    AACR_NEGATIVE_ROLE: (
        _NEGATIVE_PATH,
        496_162,
        "c0601008ec5f444317143b0ee59d7f99a0bc2b45735710d25c2f1a305ee519d0",
    ),
}
_OFFICIAL_STATISTICS = {
    "positive_prs": 196,
    "negative_prs": 155,
    "positive_comments": 1_505,
    "negative_comments": 640,
    "unique_prs": 200,
    "overlap_prs": 151,
}

_MAX_SOURCE_FILE_BYTES = 64 * 1024 * 1024
_MAX_PRS_PER_SPLIT = 100_000
_MAX_COMMENTS_PER_PR = 2_048
_MAX_TOTAL_COMMENTS = 262_144
_MAX_CHANGE_LINE_COUNT = 2_147_483_647

_PR_FIELDS = frozenset(
    {
        "change_line_count",
        "project_main_language",
        "source_commit",
        "target_commit",
        "githubPrUrl",
        "comments",
        "category",
    }
)
_COMMENT_FIELDS = frozenset(
    {
        "is_ai_comment",
        "note",
        "path",
        "side",
        "source_model",
        "from_line",
        "to_line",
        "category",
        "context",
    }
)
_EXPECTED_STATISTICS = frozenset(
    {
        "positive_prs",
        "negative_prs",
        "positive_comments",
        "negative_comments",
        "unique_prs",
        "overlap_prs",
    }
)

AACR_LANGUAGES = frozenset(
    {"C", "C#", "C++", "Go", "Java", "JavaScript", "PHP", "Python", "Rust", "TypeScript"}
)
AACR_PR_CATEGORIES = frozenset(
    {
        "Bug Fix",
        "Code Refactoring / Architectural Improvement",
        "Code Style, Linting, Formatting Fixes",
        "Dependency Updates & Environment Compatibility",
        "Documentation Update",
        "New Feature Additions",
        "Performance Optimizations",
        "Security Patches / Vulnerability Fixes",
        "Test Suite / CI Enhancements",
    }
)
AACR_FINDING_CATEGORIES = frozenset(
    {
        "Code Defect",
        "Maintainability and Readability",
        "Performance",
        "Security Vulnerability",
    }
)
AACR_CONTEXT_LEVELS = frozenset({"Diff Level", "File Level", "Repo Level"})
AACR_SIDES = frozenset({"left", "right"})

_CONTEXT_MAP = {
    "Diff Level": RequiredContextLevel.DIFF,
    "File Level": RequiredContextLevel.FILE,
    "Repo Level": RequiredContextLevel.REPO,
}
_SIDE_MAP = {"left": DiffSide.LEFT, "right": DiffSide.RIGHT}

_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_PR_URL_RE = re.compile(
    r"^https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/pull/([1-9][0-9]*)$"
)
_POINTER_RE = re.compile(r"^/[0-9]+/comments/[0-9]+$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")

_ISOLATION_REVERSED_LINE_RANGE = "reversed_line_range"
_ISOLATION_POLARITY_CONFLICT = "polarity_conflict"

@dataclass(frozen=True)
class _PrMetadata:
    change_line_count: int
    language: str
    source_commit: str
    target_commit: str
    pr_url: str
    pr_category: str
    repository_url: str

    def source_record(self) -> Dict[str, Any]:
        return {
            "change_line_count": self.change_line_count,
            "project_main_language": self.language,
            "source_commit": self.source_commit,
            "target_commit": self.target_commit,
            "githubPrUrl": self.pr_url,
            "category": self.pr_category,
        }


@dataclass(frozen=True)
class _CommentRecord:
    source_role: str
    pointer: str
    record: Mapping[str, Any]
    record_digest: str
    truth_id: str
    isolation_reasons: Tuple[str, ...]

    @property
    def context(self) -> str:
        return self.record["context"]  # type: ignore[return-value]

    @property
    def isolated(self) -> bool:
        return bool(self.isolation_reasons)


@dataclass(frozen=True)
class _PrSourceRecord:
    source_role: str
    pointer: str
    record: Mapping[str, Any]


@dataclass(frozen=True)
class _ParsedPr:
    metadata: _PrMetadata
    comments: Tuple[_CommentRecord, ...]
    source_record: _PrSourceRecord


@dataclass
class _PrAggregate:
    metadata: _PrMetadata
    positive_present: bool
    negative_present: bool
    comments: List[_CommentRecord]
    source_records: List[_PrSourceRecord]


def _format(message: str) -> PublicFormatError:
    return PublicFormatError("AACR-Bench %s" % message)


def _exact_fields(value: Any, expected: frozenset, context: str) -> Dict[str, Any]:
    if type(value) is not dict:
        raise _format("%s must be an object" % context)
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise _format(
            "%s fields do not match pinned schema (missing=%r, extra=%r)"
            % (context, missing, extra)
        )
    return value


def _string(
    value: Any,
    context: str,
    *,
    allow_empty: bool = False,
) -> str:
    if type(value) is not str:
        raise _format("%s must be a string" % context)
    if not allow_empty and not value.strip():
        raise _format("%s must be non-empty" % context)
    return value


def _integer(
    value: Any,
    context: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if type(value) is not int:
        raise _format("%s must be an integer (bool is not accepted)" % context)
    if value < minimum or value > maximum:
        raise _format(
            "%s must be between %d and %d" % (context, minimum, maximum)
        )
    return value


def _boolean(value: Any, context: str) -> bool:
    if type(value) is not bool:
        raise _format("%s must be a boolean" % context)
    return value


def _enum(value: Any, allowed: frozenset, context: str) -> str:
    result = _string(value, context)
    if result not in allowed:
        raise _format("%s has unknown value %r" % (context, result))
    return result


def _git_sha(value: Any, context: str) -> str:
    if type(value) is not str or _GIT_SHA_RE.fullmatch(value) is None:
        raise _format("%s must be a 40-character lowercase Git SHA" % context)
    return value


def _pr_url(value: Any, context: str) -> Tuple[str, str]:
    url = _string(value, context)
    match = _PR_URL_RE.fullmatch(url)
    if match is None:
        raise _format(
            "%s must match https://github.com/{owner}/{repo}/pull/{integer}"
            % context
        )
    owner, repository, _number = match.groups()
    return url, "https://github.com/%s/%s.git" % (owner, repository)


def aacr_rejection_binding(
    source_role: str, record_pointer: str, record: Any
) -> str:
    """Bind one explicitly rejected upstream record by role, pointer, and digest."""

    if source_role not in {AACR_POSITIVE_ROLE, AACR_NEGATIVE_ROLE}:
        raise _format("rejection source role is unknown")
    if type(record_pointer) is not str or _POINTER_RE.fullmatch(record_pointer) is None:
        raise _format("rejection record pointer is not canonical")
    return "%s#%s@%s" % (
        source_role,
        record_pointer,
        canonical_sha256(record),
    )


def _parse_rejection_binding(value: str) -> Tuple[str, str, str]:
    if type(value) is not str:
        raise _format("rejection selector values must be strings")
    try:
        role, remainder = value.split("#", 1)
        pointer, digest = remainder.rsplit("@", 1)
    except ValueError as exc:
        raise _format(
            "rejection selector must use role#/pr/comments/comment@sha256"
        ) from exc
    if role not in {AACR_POSITIVE_ROLE, AACR_NEGATIVE_ROLE}:
        raise _format("rejection selector has an unknown source role")
    if _POINTER_RE.fullmatch(pointer) is None:
        raise _format("rejection selector has a non-canonical record pointer")
    if _DIGEST_RE.fullmatch(digest) is None:
        raise _format("rejection selector must contain a full lowercase SHA-256")
    if value != "%s#%s@%s" % (role, pointer, digest):
        raise _format("rejection selector is not canonical")
    return role, pointer, digest


def _source_profile(source_manifest: PublicSourceManifest) -> str:
    identity = (
        source_manifest.dataset_id,
        source_manifest.dataset_version,
        source_manifest.source_uri,
        source_manifest.source_revision,
        source_manifest.license,
    )
    official = (
        AACR_DATASET_ID,
        AACR_DATASET_VERSION,
        AACR_SOURCE_URI,
        AACR_SOURCE_REVISION,
        AACR_LICENSE,
    )
    fixture = (
        AACR_FIXTURE_DATASET_ID,
        AACR_FIXTURE_DATASET_VERSION,
        AACR_SOURCE_URI,
        AACR_FIXTURE_SOURCE_REVISION,
        AACR_LICENSE,
    )
    if identity == official:
        return AACR_OFFICIAL_PROFILE
    if identity == fixture:
        return AACR_FIXTURE_PROFILE
    raise _format(
        "source manifest does not match a pinned official or fixture profile"
    )


def _receipt_marker(
    profile: str, source_manifest: PublicSourceManifest
) -> str:
    return (
        "benchmark=AACR-Bench; profile=%s; source-revision=%s; "
        "completeness=expert_augmented; protocol=%s; split=capability; "
        "release-gate=not-unique; dataset-license-scope=aacr-dataset-only; "
        "underlying_repository_license=not_normalized_by_upstream; "
        % (profile, source_manifest.source_revision, AACR_PROTOCOL_ID)
    )


def _validate_manifests(
    source_manifest: PublicSourceManifest,
    filter_manifest: PublicFilterManifest,
) -> Tuple[str, Optional[Set[str]], Set[str]]:
    if not isinstance(source_manifest, PublicSourceManifest):
        raise _format("requires a PublicSourceManifest")
    profile = _source_profile(source_manifest)

    files = {item.role: item.path for item in source_manifest.files}
    expected_files = {
        AACR_POSITIVE_ROLE: _POSITIVE_PATH,
        AACR_NEGATIVE_ROLE: _NEGATIVE_PATH,
    }
    if files != expected_files:
        raise _format("source manifest files do not match pinned split paths")
    statistic_names = {
        item.name for item in source_manifest.expected_statistics
    }
    if statistic_names != _EXPECTED_STATISTICS:
        raise _format(
            "source manifest expected statistics must be exactly %r"
            % sorted(_EXPECTED_STATISTICS)
        )
    if profile == AACR_OFFICIAL_PROFILE:
        bindings = {
            item.role: (item.path, item.size_bytes, item.sha256)
            for item in source_manifest.files
        }
        if bindings != _OFFICIAL_FILES:
            raise _format(
                "official profile requires the pinned raw file size and SHA-256 bindings"
            )
        official_statistics = {
            item.name: item.value
            for item in source_manifest.expected_statistics
        }
        if official_statistics != _OFFICIAL_STATISTICS:
            raise _format(
                "official profile requires the pinned 196/155/1505/640/200/151 statistics"
            )
    elif source_manifest.digest() != AACR_FIXTURE_SOURCE_MANIFEST_DIGEST:
        raise PublicSourceIntegrityError(
            "AACR-Bench fixture source manifest does not match its independently "
            "pinned digest"
        )

    if not isinstance(filter_manifest, PublicFilterManifest):
        raise _format("requires a PublicFilterManifest")
    if filter_manifest.dataset_id != source_manifest.dataset_id:
        raise _format("filter manifest dataset_id must match the source profile")
    selectors = {item.name: item.values for item in filter_manifest.selectors}
    unknown = set(selectors) - {AACR_LANGUAGE_SELECTOR, AACR_REJECTION_SELECTOR}
    if unknown:
        raise _format("filter manifest has unknown selector(s) %r" % sorted(unknown))

    selected_languages: Optional[Set[str]] = None
    if AACR_LANGUAGE_SELECTOR in selectors:
        raw_languages = selectors[AACR_LANGUAGE_SELECTOR]
        if not raw_languages:
            raise _format("language selector must contain at least one language")
        selected_languages = set(raw_languages)
        unknown_languages = selected_languages - AACR_LANGUAGES
        if unknown_languages:
            raise _format(
                "language selector contains unknown language(s) %r"
                % sorted(unknown_languages)
            )

    rejections = set(selectors.get(AACR_REJECTION_SELECTOR, ()))
    for item in rejections:
        _parse_rejection_binding(item)
    return profile, selected_languages, rejections


def _parse_comment(
    value: Any,
    *,
    source_role: str,
    pr_index: int,
    comment_index: int,
    rejection_bindings: Set[str],
    consumed_rejections: Set[str],
) -> _CommentRecord:
    pointer = "/%d/comments/%d" % (pr_index, comment_index)
    context = "%s%s" % (source_role, pointer)
    record = _exact_fields(value, _COMMENT_FIELDS, "comment %s" % context)
    is_ai_comment = _boolean(
        record["is_ai_comment"], "comment %s.is_ai_comment" % context
    )
    _string(record["note"], "comment %s.note" % context)
    _string(record["path"], "comment %s.path" % context)
    side = _enum(record["side"], AACR_SIDES, "comment %s.side" % context)
    source_model = _string(
        record["source_model"],
        "comment %s.source_model" % context,
        allow_empty=True,
    )
    if is_ai_comment != bool(source_model):
        raise _format(
            "comment %s source_model must be non-empty exactly for AI comments"
            % context
        )
    from_line = _integer(
        record["from_line"],
        "comment %s.from_line" % context,
        minimum=1,
        maximum=2_147_483_647,
    )
    to_line = _integer(
        record["to_line"],
        "comment %s.to_line" % context,
        minimum=1,
        maximum=2_147_483_647,
    )
    _enum(
        record["category"],
        AACR_FINDING_CATEGORIES,
        "comment %s finding category" % context,
    )
    _enum(
        record["context"],
        AACR_CONTEXT_LEVELS,
        "comment %s context" % context,
    )
    # Keep the reads above explicit even though the canonical constructors
    # validate them again.  They are pinned upstream-format checks, not score
    # normalization.
    assert side in AACR_SIDES

    digest = canonical_sha256(record)
    truth_id = stable_id("aacr-truth", source_role, pointer, digest)
    isolation_reasons: Tuple[str, ...] = ()
    if to_line < from_line:
        binding = aacr_rejection_binding(source_role, pointer, record)
        if binding not in rejection_bindings:
            raise _format(
                "comment %s has a reversed line range; an exact pointer+digest "
                "rejection is required and line swapping is forbidden" % context
            )
        consumed_rejections.add(binding)
        isolation_reasons = (_ISOLATION_REVERSED_LINE_RANGE,)
    return _CommentRecord(
        source_role=source_role,
        pointer=pointer,
        record=record,
        record_digest=digest,
        truth_id=truth_id,
        isolation_reasons=isolation_reasons,
    )


def _parse_split(
    raw: bytes,
    *,
    source_role: str,
    rejection_bindings: Set[str],
    consumed_rejections: Set[str],
) -> Tuple[_ParsedPr, ...]:
    payload = _strict_json_loads(
        raw,
        _MAX_SOURCE_FILE_BYTES,
        "AACR-Bench %s JSON" % source_role,
    )
    if type(payload) is not list:
        raise _format("%s top level must be an array" % source_role)
    if len(payload) > _MAX_PRS_PER_SPLIT:
        raise _format("%s exceeds the PR record limit" % source_role)

    seen_prs: Set[str] = set()
    total_comments = 0
    parsed: List[_ParsedPr] = []
    for pr_index, value in enumerate(payload):
        context = "%s/%d" % (source_role, pr_index)
        record = _exact_fields(value, _PR_FIELDS, "PR %s" % context)
        change_line_count = _integer(
            record["change_line_count"],
            "PR %s.change_line_count" % context,
            minimum=0,
            maximum=_MAX_CHANGE_LINE_COUNT,
        )
        language = _enum(
            record["project_main_language"],
            AACR_LANGUAGES,
            "PR %s language" % context,
        )
        source_commit = _git_sha(
            record["source_commit"], "PR %s.source_commit" % context
        )
        target_commit = _git_sha(
            record["target_commit"], "PR %s.target_commit" % context
        )
        if source_commit == target_commit:
            raise _format("PR %s source and target commits must differ" % context)
        pr_url, repository_url = _pr_url(
            record["githubPrUrl"], "PR %s.githubPrUrl" % context
        )
        if pr_url in seen_prs:
            raise _format(
                "%s contains duplicate PR %r" % (source_role, pr_url)
            )
        seen_prs.add(pr_url)
        pr_category = _enum(
            record["category"],
            AACR_PR_CATEGORIES,
            "PR %s PR category" % context,
        )
        raw_comments = record["comments"]
        if type(raw_comments) is not list:
            raise _format("PR %s.comments must be an array" % context)
        if len(raw_comments) > _MAX_COMMENTS_PER_PR:
            raise _format("PR %s.comments exceeds the item limit" % context)
        total_comments += len(raw_comments)
        if total_comments > _MAX_TOTAL_COMMENTS:
            raise _format("%s exceeds the total comment limit" % source_role)
        comments = tuple(
            _parse_comment(
                comment,
                source_role=source_role,
                pr_index=pr_index,
                comment_index=comment_index,
                rejection_bindings=rejection_bindings,
                consumed_rejections=consumed_rejections,
            )
            for comment_index, comment in enumerate(raw_comments)
        )
        parsed.append(
            _ParsedPr(
                metadata=_PrMetadata(
                    change_line_count=change_line_count,
                    language=language,
                    source_commit=source_commit,
                    target_commit=target_commit,
                    pr_url=pr_url,
                    pr_category=pr_category,
                    repository_url=repository_url,
                ),
                comments=comments,
                source_record=_PrSourceRecord(
                    source_role=source_role,
                    pointer="/%d" % pr_index,
                    record=record,
                ),
            )
        )
    return tuple(parsed)


def _aggregate_splits(
    positive: Tuple[_ParsedPr, ...], negative: Tuple[_ParsedPr, ...]
) -> Dict[str, _PrAggregate]:
    aggregates: Dict[str, _PrAggregate] = {}
    seen_comments: Dict[Tuple[str, str], Set[str]] = {}
    for source_role, records in (
        (AACR_POSITIVE_ROLE, positive),
        (AACR_NEGATIVE_ROLE, negative),
    ):
        for item in records:
            pr_url = item.metadata.pr_url
            aggregate = aggregates.get(pr_url)
            if aggregate is None:
                aggregate = _PrAggregate(
                    metadata=item.metadata,
                    positive_present=False,
                    negative_present=False,
                    comments=[],
                    source_records=[],
                )
                aggregates[pr_url] = aggregate
            elif aggregate.metadata != item.metadata:
                raise _format(
                    "overlapping PR %r has a metadata conflict across splits"
                    % pr_url
                )
            if source_role == AACR_POSITIVE_ROLE:
                aggregate.positive_present = True
            else:
                aggregate.negative_present = True
            aggregate.source_records.append(item.source_record)
            for comment in item.comments:
                identity = (pr_url, comment.record_digest)
                seen_roles = seen_comments.setdefault(identity, set())
                if source_role in seen_roles:
                    raise _format(
                        "PR %r contains a duplicate comment record" % pr_url
                    )
                seen_roles.add(source_role)
                aggregate.comments.append(comment)
    return aggregates


def _isolate_polarity_conflicts(
    aggregates: Mapping[str, _PrAggregate],
    rejection_bindings: Set[str],
    consumed_rejections: Set[str],
) -> None:
    """Require exact isolation for every cross-polarity NFC claim conflict."""

    for pr_url in sorted(aggregates):
        aggregate = aggregates[pr_url]
        grouped: Dict[str, List[int]] = {}
        for index, comment in enumerate(aggregate.comments):
            canonical_claim = unicodedata.normalize(
                "NFC", comment.record["note"]  # type: ignore[arg-type]
            )
            grouped.setdefault(canonical_claim, []).append(index)

        for indices in grouped.values():
            roles = {aggregate.comments[index].source_role for index in indices}
            if roles != {AACR_POSITIVE_ROLE, AACR_NEGATIVE_ROLE}:
                continue

            missing: List[str] = []
            bindings: Dict[int, str] = {}
            for index in indices:
                comment = aggregate.comments[index]
                binding = aacr_rejection_binding(
                    comment.source_role, comment.pointer, comment.record
                )
                bindings[index] = binding
                if binding not in rejection_bindings:
                    missing.append(binding)
            if missing:
                raise _format(
                    "PR %r contains an NFC-equivalent positive/negative claim "
                    "polarity conflict; every conflicting record requires an "
                    "exact pointer+digest rejection binding (missing=%r)"
                    % (pr_url, sorted(missing))
                )

            for index in indices:
                comment = aggregate.comments[index]
                consumed_rejections.add(bindings[index])
                aggregate.comments[index] = replace(
                    comment,
                    isolation_reasons=tuple(
                        sorted(
                            set(comment.isolation_reasons)
                            | {_ISOLATION_POLARITY_CONFLICT}
                        )
                    ),
                )


def _verify_source_statistics(
    source_manifest: PublicSourceManifest,
    positive: Tuple[_ParsedPr, ...],
    negative: Tuple[_ParsedPr, ...],
    aggregates: Mapping[str, _PrAggregate],
) -> Dict[str, int]:
    positive_urls = {item.metadata.pr_url for item in positive}
    negative_urls = {item.metadata.pr_url for item in negative}
    actual = {
        "positive_prs": len(positive),
        "negative_prs": len(negative),
        "positive_comments": sum(len(item.comments) for item in positive),
        "negative_comments": sum(len(item.comments) for item in negative),
        "unique_prs": len(aggregates),
        "overlap_prs": len(positive_urls & negative_urls),
    }
    expected = {
        item.name: item.value for item in source_manifest.expected_statistics
    }
    for name in sorted(_EXPECTED_STATISTICS):
        if actual[name] != expected[name]:
            raise _format(
                "source statistic %s=%d does not match manifest value %d"
                % (name, actual[name], expected[name])
            )
    return actual


def _truth_profile(expected_count: int, invalid_count: int) -> str:
    if expected_count and invalid_count:
        return "positive_and_negative"
    if expected_count:
        return "positive_only"
    if invalid_count:
        return "negative_only_not_clean"
    return "zero_truth_not_clean"


def _case_content_hash(aggregate: _PrAggregate) -> str:
    return canonical_sha256(
        {
            "pr_metadata": aggregate.metadata.source_record(),
            "records": [
                {
                    "source_role": item.source_role,
                    "record_pointer": item.pointer,
                    "record_sha256": item.record_digest,
                }
                for item in sorted(
                    aggregate.comments,
                    key=lambda value: (value.source_role, value.pointer),
                )
            ],
        }
    )


def _location(comment: _CommentRecord) -> TruthLocation:
    return TruthLocation(
        path=comment.record["path"],  # type: ignore[arg-type]
        side=_SIDE_MAP[comment.record["side"]],
        from_line=comment.record["from_line"],  # type: ignore[arg-type]
        to_line=comment.record["to_line"],  # type: ignore[arg-type]
    )


def _isolation_disposition(comment: _CommentRecord) -> str:
    reasons = set(comment.isolation_reasons)
    if reasons == {_ISOLATION_REVERSED_LINE_RANGE}:
        return "isolated_reversed_line_range"
    if reasons == {_ISOLATION_POLARITY_CONFLICT}:
        return "isolated_polarity_conflict"
    return "isolated_multiple_reasons"


def _isolation_receipt_reason(
    comment: _CommentRecord,
    receipt_marker: str,
    *,
    filtered_language: Optional[str] = None,
) -> str:
    details = [
        receipt_marker,
        "isolated-upstream-truth; exact-pointer-digest-binding; ",
        "claim-unchanged; ",
        "isolation-reasons=%s"
        % "|".join(comment.isolation_reasons),
    ]
    if _ISOLATION_REVERSED_LINE_RANGE in comment.isolation_reasons:
        details.append("; no-line-swapping")
    if _ISOLATION_POLARITY_CONFLICT in comment.isolation_reasons:
        details.append("; no-polarity-selection")
    if filtered_language is not None:
        details.append("; filtered-language=%s" % filtered_language)
    return "".join(details)


def _build_case(
    aggregate: _PrAggregate,
    *,
    source_manifest: PublicSourceManifest,
    source_profile: str,
    suite_id: str,
    receipt_marker: str,
) -> Tuple[PublicPreparedCase, List[Tuple[_CommentRecord, str, str]]]:
    metadata = aggregate.metadata
    task_id = stable_id("aacr-pr", metadata.pr_url)
    expected: List[ExpectedFinding] = []
    invalid: List[KnownInvalidFinding] = []
    dispositions: List[Tuple[_CommentRecord, str, str]] = []
    context_levels: Set[str] = set()
    for comment in aggregate.comments:
        if comment.isolated:
            dispositions.append(
                (
                    comment,
                    _isolation_disposition(comment),
                    _isolation_receipt_reason(comment, receipt_marker),
                )
            )
            continue
        context_levels.add(comment.context)
        if comment.source_role == AACR_POSITIVE_ROLE:
            expected.append(
                ExpectedFinding(
                    truth_id=comment.truth_id,
                    claim=comment.record["note"],  # type: ignore[arg-type]
                    severity=None,
                    category=comment.record["category"],  # type: ignore[arg-type]
                    required=True,
                    metric_authority=MetricAuthority(
                        severity_scorable=False,
                        severity_authority=None,
                        location_scorable=True,
                        location_authority=MetricAuthoritySource.UPSTREAM_ANNOTATION,
                    ),
                    locations=(_location(comment),),
                    evidence_anchors=(),
                    required_context_level=_CONTEXT_MAP[comment.context],
                    rationale=(
                        "AACR-Bench positive sample. The upstream record supplies "
                        "no severity authority; its path and line range remain "
                        "the upstream annotation authority for location."
                    ),
                )
            )
            dispositions.append(
                (
                    comment,
                    "expected_finding",
                    receipt_marker
                    + "positive-sample; severity=not-scorable; "
                    "location=upstream-annotation",
                )
            )
        else:
            invalid.append(
                KnownInvalidFinding(
                    truth_id=comment.truth_id,
                    claim=comment.record["note"],  # type: ignore[arg-type]
                    category=comment.record["category"],  # type: ignore[arg-type]
                    locations=(_location(comment),),
                    rationale=(
                        "AACR-Bench negative sample (known invalid); upstream "
                        "context=%s and the exact source record is retained in "
                        "the preparation receipt." % comment.context
                    ),
                )
            )
            dispositions.append(
                (
                    comment,
                    "known_invalid",
                    receipt_marker
                    + "negative-sample; upstream-has-no-severity; "
                    "severity=not-scorable",
                )
            )

    case = EvalCase(
        schema_version=EVAL_CASE_SCHEMA_VERSION,
        task_id=task_id,
        case_version=1,
        source=CaseSource(
            suite=suite_id,
            origin=CaseOrigin.AACR_BENCH,
            source_id=metadata.pr_url,
            source_version=source_manifest.source_revision,
            source_uri=source_manifest.source_uri,
            license=source_manifest.license,
            content_hash=_case_content_hash(aggregate),
        ),
        input=EvalCaseInput(
            review_target=RepositoryReviewTarget(
                kind=ReviewTargetKind.REPOSITORY,
                repository=Repository(
                    source=RepositorySource.GIT,
                    path=None,
                    url=metadata.repository_url,
                    base_revision=metadata.target_commit,
                    head_revision=metadata.source_commit,
                ),
                review_request=ReviewRequest(
                    title=None,
                    description=None,
                    user_intent=None,
                    review_focus=None,
                    linked_requirements=(),
                    project_rules=(),
                    existing_ci_evidence=(),
                ),
            ),
        ),
        clarification_script=ClarificationScript(max_rounds=4, answers=()),
        intent_truth=IntentTruth(
            scorable=False,
            authority=None,
            expected_claims=(),
            forbidden_claims=(),
            clarification_policy=None,
        ),
        review_truth=ReviewTruth(
            completeness=TruthCompleteness.EXPERT_AUGMENTED,
            novel_finding_policy=NovelFindingPolicy.VERIFY,
            expected_findings=tuple(expected),
            known_invalid_findings=tuple(invalid),
        ),
        review_evaluator_context=ReviewEvaluatorContext(truth_contexts=()),
    )
    dimensions = (
        CaseDimension(name="benchmark", value="AACR-Bench"),
        CaseDimension(name="benchmark_profile", value=source_profile),
        CaseDimension(
            name="source_revision", value=source_manifest.source_revision
        ),
        CaseDimension(name="completeness", value="expert_augmented"),
        CaseDimension(name="protocol", value=AACR_PROTOCOL_ID),
        CaseDimension(name="language", value=metadata.language),
        CaseDimension(name="pr_category", value=metadata.pr_category),
        CaseDimension(
            name="context_levels",
            value=("|".join(sorted(context_levels)) if context_levels else "none"),
        ),
        CaseDimension(
            name="truth_profile", value=_truth_profile(len(expected), len(invalid))
        ),
        CaseDimension(name="positive_truth_count", value=str(len(expected))),
        CaseDimension(name="known_invalid_count", value=str(len(invalid))),
        CaseDimension(
            name="isolated_truth_count",
            value=str(sum(item.isolated for item in aggregate.comments)),
        ),
        CaseDimension(
            name="polarity_conflict_isolated_truth_count",
            value=str(
                sum(
                    _ISOLATION_POLARITY_CONFLICT in item.isolation_reasons
                    for item in aggregate.comments
                )
            ),
        ),
        CaseDimension(
            name="reversed_line_range_isolated_truth_count",
            value=str(
                sum(
                    _ISOLATION_REVERSED_LINE_RANGE in item.isolation_reasons
                    for item in aggregate.comments
                )
            ),
        ),
        CaseDimension(name="severity_source", value="upstream_unavailable"),
        CaseDimension(name="severity_metric_scope", value="not_scorable"),
        CaseDimension(name="location_metric_scope", value="upstream_annotation"),
        CaseDimension(name="dataset_license_scope", value="aacr_dataset_only"),
        CaseDimension(
            name="underlying_repository_license",
            value="not_normalized_by_upstream",
        ),
        CaseDimension(name="release_gate_role", value="not_unique_gate"),
    )
    return (
        PublicPreparedCase(
            case=case,
            split=CaseSplit.CAPABILITY,
            protocol_id=AACR_PROTOCOL_ID,
            dimensions=dimensions,
        ),
        dispositions,
    )


def _record_receipt(
    *,
    task_id: str,
    comment: _CommentRecord,
    disposition: str,
    reason: str,
) -> PublicRecordReceipt:
    return PublicRecordReceipt.from_record(
        task_id=task_id,
        truth_id=comment.truth_id,
        source_role=comment.source_role,
        record_pointer=comment.pointer,
        upstream_id=None,
        record=comment.record,
        disposition=disposition,
        reason=reason,
    )


def _pr_record_receipt(
    *,
    task_id: str,
    pr_url: str,
    source_record: _PrSourceRecord,
    selected: bool,
    receipt_marker: str,
) -> PublicRecordReceipt:
    return PublicRecordReceipt.from_record(
        task_id=task_id,
        truth_id=None,
        source_role=source_record.source_role,
        record_pointer=source_record.pointer,
        upstream_id=pr_url,
        record=source_record.record,
        disposition=("selected_pr_record" if selected else "filtered_pr_record"),
        reason=(
            receipt_marker
            + "pr-level-source-record; selection=%s"
            % ("selected" if selected else "filtered")
        ),
    )


def prepare_aacr_bench(
    source_root: os.PathLike[str] | str,
    source_manifest: PublicSourceManifest,
    filter_manifest: PublicFilterManifest,
    output_root: os.PathLike[str] | str,
) -> PublicPreparationResult:
    """Prepare a canonical, create-only AACR-Bench Suite from local bytes.

    This function performs no network access.  Source acquisition must happen
    before this boundary, and Trials consume only the published CaseBank.
    """

    source_profile, selected_languages, rejection_bindings = _validate_manifests(
        source_manifest, filter_manifest
    )
    suite_id = (
        AACR_SUITE_ID
        if source_profile == AACR_OFFICIAL_PROFILE
        else AACR_FIXTURE_SUITE_ID
    )
    receipt_marker = _receipt_marker(source_profile, source_manifest)
    verified = VerifiedPublicSource.open(source_root, source_manifest)
    consumed_rejections: Set[str] = set()
    try:
        positive = _parse_split(
            verified.read(AACR_POSITIVE_ROLE),
            source_role=AACR_POSITIVE_ROLE,
            rejection_bindings=rejection_bindings,
            consumed_rejections=consumed_rejections,
        )
        negative = _parse_split(
            verified.read(AACR_NEGATIVE_ROLE),
            source_role=AACR_NEGATIVE_ROLE,
            rejection_bindings=rejection_bindings,
            consumed_rejections=consumed_rejections,
        )
        aggregates = _aggregate_splits(positive, negative)
        _isolate_polarity_conflicts(
            aggregates, rejection_bindings, consumed_rejections
        )
        raw_statistics = _verify_source_statistics(
            source_manifest, positive, negative, aggregates
        )
        unused_rejections = rejection_bindings - consumed_rejections
        if unused_rejections:
            raise _format(
                "filter manifest has unused rejection binding(s); only exact "
                "reversed-range or polarity-conflict records may be isolated: %r"
                % sorted(unused_rejections)
            )

        source_comments = tuple(
            comment
            for aggregate in aggregates.values()
            for comment in aggregate.comments
        )
        source_isolated_comments = sum(item.isolated for item in source_comments)
        source_reversed_isolated_comments = sum(
            _ISOLATION_REVERSED_LINE_RANGE in item.isolation_reasons
            for item in source_comments
        )
        source_polarity_isolated_comments = sum(
            _ISOLATION_POLARITY_CONFLICT in item.isolation_reasons
            for item in source_comments
        )
        source_multi_reason_isolated_comments = sum(
            len(item.isolation_reasons) > 1 for item in source_comments
        )
        source_scorable_comments = len(source_comments) - source_isolated_comments
        source_scorable_positive_comments = sum(
            not item.isolated and item.source_role == AACR_POSITIVE_ROLE
            for item in source_comments
        )
        source_scorable_negative_comments = sum(
            not item.isolated and item.source_role == AACR_NEGATIVE_ROLE
            for item in source_comments
        )

        prepared_cases: List[PublicPreparedCase] = []
        receipts: List[PublicRecordReceipt] = []
        selected_prs = 0
        filtered_prs = 0
        selected_source_comments = 0
        selected_scorable_comments = 0
        selected_isolated_comments = 0
        selected_reversed_isolated_comments = 0
        selected_polarity_isolated_comments = 0
        filtered_source_comments = 0
        filtered_scorable_comments = 0
        filtered_isolated_comments = 0
        filtered_reversed_isolated_comments = 0
        filtered_polarity_isolated_comments = 0
        selected_positive_comments = 0
        selected_negative_comments = 0
        expected_findings = 0
        known_invalid_findings = 0
        zero_positive_prs = 0
        negative_only_prs = 0
        zero_truth_prs = 0
        pr_record_receipts = 0

        for pr_url in sorted(aggregates):
            aggregate = aggregates[pr_url]
            task_id = stable_id("aacr-pr", pr_url)
            selected = (
                selected_languages is None
                or aggregate.metadata.language in selected_languages
            )
            receipts.extend(
                _pr_record_receipt(
                    task_id=task_id,
                    pr_url=pr_url,
                    source_record=source_record,
                    selected=selected,
                    receipt_marker=receipt_marker,
                )
                for source_record in aggregate.source_records
            )
            pr_record_receipts += len(aggregate.source_records)
            if not selected:
                filtered_prs += 1
                filtered_source_comments += len(aggregate.comments)
                filtered_isolated_comments += sum(
                    item.isolated for item in aggregate.comments
                )
                filtered_scorable_comments += sum(
                    not item.isolated for item in aggregate.comments
                )
                filtered_reversed_isolated_comments += sum(
                    _ISOLATION_REVERSED_LINE_RANGE in item.isolation_reasons
                    for item in aggregate.comments
                )
                filtered_polarity_isolated_comments += sum(
                    _ISOLATION_POLARITY_CONFLICT in item.isolation_reasons
                    for item in aggregate.comments
                )
                for comment in aggregate.comments:
                    if comment.isolated:
                        disposition = _isolation_disposition(comment)
                        reason = _isolation_receipt_reason(
                            comment,
                            receipt_marker,
                            filtered_language=aggregate.metadata.language,
                        )
                    else:
                        disposition = "filtered_out"
                        reason = (
                            receipt_marker
                            + "language selector excluded %s"
                            % aggregate.metadata.language
                        )
                    receipts.append(
                        _record_receipt(
                            task_id=task_id,
                            comment=comment,
                            disposition=disposition,
                            reason=reason,
                        )
                    )
                continue

            selected_prs += 1
            selected_source_comments += len(aggregate.comments)
            selected_isolated_comments += sum(
                item.isolated for item in aggregate.comments
            )
            selected_scorable_comments += sum(
                not item.isolated for item in aggregate.comments
            )
            selected_reversed_isolated_comments += sum(
                _ISOLATION_REVERSED_LINE_RANGE in item.isolation_reasons
                for item in aggregate.comments
            )
            selected_polarity_isolated_comments += sum(
                _ISOLATION_POLARITY_CONFLICT in item.isolation_reasons
                for item in aggregate.comments
            )
            selected_positive_comments += sum(
                item.source_role == AACR_POSITIVE_ROLE
                for item in aggregate.comments
            )
            selected_negative_comments += sum(
                item.source_role == AACR_NEGATIVE_ROLE
                for item in aggregate.comments
            )
            prepared, dispositions = _build_case(
                aggregate,
                source_manifest=source_manifest,
                source_profile=source_profile,
                suite_id=suite_id,
                receipt_marker=receipt_marker,
            )
            prepared_cases.append(prepared)
            expected_count = len(prepared.case.review_truth.expected_findings)
            invalid_count = len(prepared.case.review_truth.known_invalid_findings)
            expected_findings += expected_count
            known_invalid_findings += invalid_count
            if expected_count == 0:
                zero_positive_prs += 1
            if expected_count == 0 and invalid_count > 0:
                negative_only_prs += 1
            if expected_count == 0 and invalid_count == 0:
                zero_truth_prs += 1
            receipts.extend(
                _record_receipt(
                    task_id=task_id,
                    comment=comment,
                    disposition=disposition,
                    reason=reason,
                )
                for comment, disposition, reason in dispositions
            )
    except PublicDatasetError:
        raise
    except SchemaError as exc:
        raise _format("record cannot map to canonical EvalCase: %s" % exc) from exc

    if (
        len(source_comments)
        != source_scorable_comments + source_isolated_comments
        or len(source_comments)
        != selected_source_comments + filtered_source_comments
        or selected_source_comments
        != selected_scorable_comments + selected_isolated_comments
        or filtered_source_comments
        != filtered_scorable_comments + filtered_isolated_comments
        or source_scorable_comments
        != selected_scorable_comments + filtered_scorable_comments
        or source_isolated_comments
        != selected_isolated_comments + filtered_isolated_comments
        or source_reversed_isolated_comments
        != selected_reversed_isolated_comments
        + filtered_reversed_isolated_comments
        or source_polarity_isolated_comments
        != selected_polarity_isolated_comments
        + filtered_polarity_isolated_comments
        or selected_scorable_comments
        != expected_findings + known_invalid_findings
        or source_scorable_comments
        != source_scorable_positive_comments + source_scorable_negative_comments
    ):
        raise _format("internal source/scorable/isolation statistics disagree")

    actual_statistics = dict(raw_statistics)
    actual_statistics.update(
        {
            "selected_prs": selected_prs,
            "filtered_prs": filtered_prs,
            "source_comments": len(source_comments),
            "source_scorable_comments": source_scorable_comments,
            "source_scorable_positive_comments": source_scorable_positive_comments,
            "source_scorable_negative_comments": source_scorable_negative_comments,
            "source_isolated_comments": source_isolated_comments,
            "source_reversed_line_range_isolated_comments": source_reversed_isolated_comments,
            "source_polarity_conflict_isolated_comments": source_polarity_isolated_comments,
            "source_multi_reason_isolated_comments": source_multi_reason_isolated_comments,
            "selected_source_comments": selected_source_comments,
            "selected_scorable_comments": selected_scorable_comments,
            "selected_isolated_comments": selected_isolated_comments,
            "selected_reversed_line_range_isolated_comments": selected_reversed_isolated_comments,
            "selected_polarity_conflict_isolated_comments": selected_polarity_isolated_comments,
            "filtered_source_comments": filtered_source_comments,
            "filtered_scorable_comments": filtered_scorable_comments,
            "filtered_isolated_comments": filtered_isolated_comments,
            "filtered_reversed_line_range_isolated_comments": filtered_reversed_isolated_comments,
            "filtered_polarity_conflict_isolated_comments": filtered_polarity_isolated_comments,
            "filtered_comments": filtered_source_comments,
            "selected_positive_comments": selected_positive_comments,
            "selected_negative_comments": selected_negative_comments,
            "expected_findings": expected_findings,
            "known_invalid_findings": known_invalid_findings,
            "scorable_comments": selected_scorable_comments,
            "isolated_comments": source_isolated_comments,
            "reversed_line_range_isolated_comments": source_reversed_isolated_comments,
            "polarity_conflict_isolated_comments": source_polarity_isolated_comments,
            "rejected_comments": source_isolated_comments,
            "severity_unscorable_findings": expected_findings,
            "zero_positive_prs": zero_positive_prs,
            "negative_only_prs": negative_only_prs,
            "zero_truth_prs": zero_truth_prs,
            "pr_record_receipts": pr_record_receipts,
        }
    )
    suite_version = "%s-%s" % (
        source_manifest.dataset_version,
        filter_manifest.digest(),
    )
    return write_public_suite(
        output_root,
        suite_id=suite_id,
        suite_version=suite_version,
        adapter_id=AACR_ADAPTER_ID,
        adapter_version=AACR_ADAPTER_VERSION,
        source_manifest=source_manifest,
        filter_manifest=filter_manifest,
        wire_contract=AACR_WIRE_CONTRACT,
        cases=prepared_cases,
        actual_statistics=tuple(
            PublicStatistic(name=name, value=value)
            for name, value in sorted(actual_statistics.items())
        ),
        records=receipts,
    )


__all__ = [
    "AACR_DATASET_ID",
    "AACR_DATASET_VERSION",
    "AACR_SOURCE_URI",
    "AACR_SOURCE_REVISION",
    "AACR_LICENSE",
    "AACR_OFFICIAL_PROFILE",
    "AACR_FIXTURE_PROFILE",
    "AACR_FIXTURE_DATASET_ID",
    "AACR_FIXTURE_DATASET_VERSION",
    "AACR_FIXTURE_SOURCE_REVISION",
    "AACR_FIXTURE_SOURCE_MANIFEST_DIGEST",
    "AACR_OFFICIAL_REQUIRED_REJECTION_BINDINGS",
    "AACR_POSITIVE_ROLE",
    "AACR_NEGATIVE_ROLE",
    "AACR_REJECTION_SELECTOR",
    "AACR_LANGUAGE_SELECTOR",
    "AACR_SUITE_ID",
    "AACR_FIXTURE_SUITE_ID",
    "AACR_PROTOCOL_ID",
    "AACR_WIRE_CONTRACT",
    "AACR_ADAPTER_ID",
    "AACR_ADAPTER_VERSION",
    "AACR_LANGUAGES",
    "AACR_PR_CATEGORIES",
    "AACR_FINDING_CATEGORIES",
    "AACR_CONTEXT_LEVELS",
    "AACR_SIDES",
    "aacr_rejection_binding",
    "prepare_aacr_bench",
]
