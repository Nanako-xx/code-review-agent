"""Immutable, source-bound artifacts for the Eval v2 analysis layer.

This module deliberately depends only on Eval protocol/storage modules.  It
does not import or construct the product Runtime, an Agent adapter, a Judge,
or any acquisition service.
"""

from __future__ import annotations

import hashlib
import os
import stat
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from .artifacts import (
    DEFAULT_MAX_FILE_BYTES,
    DEFAULT_MAX_TOTAL_READ_BYTES,
    ArtifactConflictError,
    ArtifactIntegrityError,
    ArtifactSecurityError,
    ArtifactStore,
    _ReadBudget,
    _absolute_storage_path,
    _hardlinked_file,
    _strict_json_loads,
    _unsafe_node,
    _windows_close_handle,
    _windows_open_directory_handle,
    _windows_raw_handle_attributes,
    _windows_raw_handle_path,
)
from .cases import RunCaseSnapshot
from .config import (
    EvalRunConfig,
    EvaluatorExecutionConfig,
    derive_evaluation_id,
    validate_safe_json,
    validate_evaluation_id_shape,
    validate_path_segment,
    validate_run_id,
)
from .models import (
    EvalCase,
    EvalSubmission,
    SchemaError,
    _JsonModel,
    canonical_json_bytes,
    stable_id,
)


ANALYSIS_RECEIPT_SCHEMA_VERSION = "analysis_receipt_v1"
ANALYSIS_ARTIFACT_KINDS = frozenset(
    {
        "statistics",
        "comparison",
        "calibration-package",
        "calibration-result",
        "gate-policy",
        "gate-result",
    }
)
MAX_ANALYSIS_ARTIFACTS = 64
_RECEIPT_NAME = "receipt.json"
_HEX_DIGITS = frozenset("0123456789abcdef")
_GLOBAL_REPORT_SOURCE_BINDING_FIELDS = frozenset(
    {
        "run_id",
        "run_config_digest",
        "run_manifest_digest",
        "case_snapshot_id",
        "case_snapshot_digest",
        "evaluation_id",
        "evaluation_revision",
        "evaluator_execution_digest",
        "metrics_policy",
    }
)
_TRIAL_REPORT_SOURCE_BINDING_FIELDS = (
    _GLOBAL_REPORT_SOURCE_BINDING_FIELDS
    | {
        "task_id",
        "trial_id",
        "trial_index",
        "canonical_case_digest",
        "eval_input_digest",
        "submission_digest",
        "intent_result_digest",
        "review_result_digest",
        "trial_score_digest",
    }
)


def _digest(value: Any, context: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise SchemaError("%s must be a lowercase SHA-256 digest" % context)
    return value


def _kind(value: Any) -> str:
    kind = validate_path_segment(value, "analysis artifact kind")
    if kind not in ANALYSIS_ARTIFACT_KINDS:
        raise SchemaError("analysis artifact kind is unsupported")
    return kind


def _artifact_id(value: Any) -> str:
    return validate_path_segment(value, "analysis artifact_id")


def _json_artifact_name(value: Any) -> str:
    name = validate_path_segment(value, "analysis JSON artifact name")
    if (
        _portable_artifact_name_key(name)
        == _portable_artifact_name_key(_RECEIPT_NAME)
        or not name.endswith(".json")
    ):
        raise SchemaError(
            "analysis JSON artifact name must end in .json and not collide "
            "with reserved receipt.json"
        )
    return name


def _portable_artifact_name_key(value: str) -> str:
    """Return the Windows-portable collision key for one validated name."""

    return unicodedata.normalize("NFKC", value).casefold()


def _calibration_publication_digest(role: str, payload: Mapping[str, Any]) -> str:
    """Bind a calibration Analysis namespace to its exact nested payload."""

    return hashlib.sha256(
        canonical_json_bytes(
            {
                "protocol": "analysis-calibration-publication-v1",
                "role": role,
                "payload": dict(payload),
            }
        )
    ).hexdigest()


def _human_provenance_digest(labels: Any) -> str:
    """Digest only typed reviewer/adjudicator provenance for receipt binding."""

    return hashlib.sha256(
        canonical_json_bytes(
            [
                {
                    "calibration_item_id": item.calibration_item_id,
                    "reviewer_provenance": item.reviewer_provenance.to_dict(),
                    "adjudication": (
                        None
                        if item.adjudication is None
                        else item.adjudication.to_dict()
                    ),
                }
                for item in labels.labels
            ]
        )
    ).hexdigest()


def _windows_portable_path_segment_key(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold().rstrip(" .")


def _path_contains_portable_run_root(path: os.PathLike[str] | str) -> bool:
    portable_run_root = _windows_portable_path_segment_key(".eval-runs")
    normalized = os.fspath(path).replace("\\", "/")
    return any(
        _windows_portable_path_segment_key(segment) == portable_run_root
        for segment in normalized.split("/")
        if segment
    )


def _nearest_existing_directory(path: Path) -> Path:
    current = path
    while True:
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            parent = current.parent
            if parent == current:
                raise ArtifactSecurityError(
                    "could not locate an existing Analysis root ancestor"
                )
            current = parent
            continue
        except OSError as exc:
            raise ArtifactSecurityError(
                "could not inspect the Analysis root ancestor"
            ) from exc
        if _unsafe_node(metadata) or not stat.S_ISDIR(metadata.st_mode):
            raise ArtifactSecurityError(
                "Analysis root ancestor is a link, reparse point, or non-directory"
            )
        return current


def _validate_windows_final_root_boundary(root: Path) -> None:
    if os.name != "nt":
        return
    ancestor = _nearest_existing_directory(root)
    handle = _windows_open_directory_handle(ancestor)
    try:
        final_ancestor = _windows_raw_handle_path(handle)
        attributes = _windows_raw_handle_attributes(handle)
    finally:
        _windows_close_handle(handle)
    if attributes & 0x400 or not attributes & 0x10:
        raise ArtifactSecurityError(
            "Analysis root ancestor handle is a reparse point or non-directory"
        )
    if _path_contains_portable_run_root(final_ancestor):
        raise ArtifactSecurityError(
            "Analysis root final path may not be the Run Store or descend from .eval-runs"
        )


def _validate_analysis_root_boundary(root: os.PathLike[str] | str) -> None:
    try:
        absolute = _absolute_storage_path(root)
    except (TypeError, ValueError) as exc:
        raise ValueError("Analysis root must be a filesystem path") from exc
    if _path_contains_portable_run_root(absolute):
        raise ValueError(
            "Analysis root may not be the Run Store or descend from .eval-runs"
        )
    _validate_windows_final_root_boundary(absolute)


def _exact_mapping(value: Any, fields: set[str], context: str) -> Mapping[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise SchemaError("%s has invalid exact keys" % context)
    return value


def _source_sort_key(value: "AnalysisSourceBinding") -> tuple[Any, ...]:
    """Sort by the complete canonical source identity, never a prefix."""

    return (canonical_json_bytes(value.to_dict()),)


@dataclass(frozen=True)
class AnalysisSourceBinding(_JsonModel):
    """Trusted Evaluation roots consumed by one analysis calculation."""

    run_id: str
    evaluation_id: str
    summary_id: str
    summary_digest: str
    run_config_digest: str
    case_snapshot_digest: str
    trial_score_digests: Tuple[str, ...]

    def __post_init__(self) -> None:
        validate_run_id(self.run_id)
        validate_evaluation_id_shape(self.evaluation_id)
        validate_path_segment(self.summary_id, "analysis summary_id")
        _digest(self.summary_digest, "analysis summary_digest")
        _digest(self.run_config_digest, "analysis run_config_digest")
        _digest(self.case_snapshot_digest, "analysis case_snapshot_digest")
        if type(self.trial_score_digests) not in (tuple, list):
            raise SchemaError("analysis trial_score_digests must be a list")
        scores = tuple(
            _digest(item, "analysis trial_score_digest")
            for item in self.trial_score_digests
        )
        if not scores:
            raise SchemaError("analysis source must bind at least one Trial score")
        if len(scores) != len(set(scores)):
            raise SchemaError("analysis source contains duplicate Trial score digests")
        if scores != tuple(sorted(scores)):
            raise SchemaError("analysis Trial score digests are not canonical")
        object.__setattr__(self, "trial_score_digests", scores)

    @classmethod
    def from_dict(cls, value: Any) -> "AnalysisSourceBinding":
        payload = _exact_mapping(
            value,
            {
                "run_id",
                "evaluation_id",
                "summary_id",
                "summary_digest",
                "run_config_digest",
                "case_snapshot_digest",
                "trial_score_digests",
            },
            "AnalysisSourceBinding",
        )
        digests = payload["trial_score_digests"]
        if type(digests) is not list:
            raise SchemaError("AnalysisSourceBinding trial_score_digests must be a list")
        return cls(
            run_id=payload["run_id"],
            evaluation_id=payload["evaluation_id"],
            summary_id=payload["summary_id"],
            summary_digest=payload["summary_digest"],
            run_config_digest=payload["run_config_digest"],
            case_snapshot_digest=payload["case_snapshot_digest"],
            trial_score_digests=tuple(digests),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "evaluation_id": self.evaluation_id,
            "summary_id": self.summary_id,
            "summary_digest": self.summary_digest,
            "run_config_digest": self.run_config_digest,
            "case_snapshot_digest": self.case_snapshot_digest,
            "trial_score_digests": list(self.trial_score_digests),
        }


def _canonical_source_bindings(
    values: Iterable[AnalysisSourceBinding],
    context: str,
) -> Tuple[AnalysisSourceBinding, ...]:
    try:
        sources = tuple(values)
    except TypeError as exc:
        raise SchemaError("%s must be an iterable" % context) from exc
    if not sources or any(type(item) is not AnalysisSourceBinding for item in sources):
        raise SchemaError("%s requires typed source bindings" % context)
    keyed = tuple((_source_sort_key(item), item) for item in sources)
    identities = tuple(key for key, _item in keyed)
    if len(identities) != len(set(identities)):
        raise SchemaError("%s contains a duplicate source binding" % context)
    return tuple(item for _key, item in sorted(keyed, key=lambda pair: pair[0]))


@dataclass(frozen=True)
class AnalysisArtifactRef(_JsonModel):
    """Digest and path descriptor for one committed analysis child file."""

    kind: str
    artifact_id: str
    relative_path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        kind = _kind(self.kind)
        artifact_id = _artifact_id(self.artifact_id)
        if type(self.relative_path) is not str or "\\" in self.relative_path:
            raise SchemaError("analysis artifact path must be a portable relative path")
        parts = self.relative_path.split("/")
        if len(parts) != 3:
            raise SchemaError("analysis artifact path must have exactly three segments")
        if parts[0] != kind or parts[1] != artifact_id:
            raise SchemaError("analysis artifact path differs from kind/artifact_id")
        _json_artifact_name(parts[2])
        _digest(self.sha256, "analysis artifact sha256")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise SchemaError("analysis artifact size_bytes must be a non-negative integer")

    @classmethod
    def create(
        cls,
        *,
        kind: str,
        artifact_id: str,
        name: str,
        data: bytes,
    ) -> "AnalysisArtifactRef":
        if type(data) is not bytes:
            raise TypeError("analysis artifact data must be bytes")
        name = _json_artifact_name(name)
        return cls(
            kind=kind,
            artifact_id=artifact_id,
            relative_path="%s/%s/%s" % (kind, artifact_id, name),
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
        )

    @classmethod
    def from_dict(cls, value: Any) -> "AnalysisArtifactRef":
        payload = _exact_mapping(
            value,
            {"kind", "artifact_id", "relative_path", "sha256", "size_bytes"},
            "AnalysisArtifactRef",
        )
        return cls(**dict(payload))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "artifact_id": self.artifact_id,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


def derive_analysis_artifact_id(
    kind: str,
    source_bindings: Iterable[AnalysisSourceBinding],
    algorithm_digest: str,
) -> str:
    """Derive the stable namespace ID from every receipt identity input."""

    canonical_kind = _kind(kind)
    algorithm = _digest(algorithm_digest, "analysis algorithm_digest")
    sources = _canonical_source_bindings(
        source_bindings,
        "analysis artifact",
    )
    return stable_id(
        "analysis-artifact-v1",
        {
            "kind": canonical_kind,
            "source_bindings": [item.to_dict() for item in sources],
            "algorithm_digest": algorithm,
        },
    )


def _canonical_file_payloads(files: Any) -> Dict[str, bytes]:
    if type(files) is not dict or not files:
        raise SchemaError("analysis files must be a non-empty JSON mapping")
    if len(files) > MAX_ANALYSIS_ARTIFACTS:
        raise SchemaError("analysis bundle contains too many artifacts")
    result: Dict[str, bytes] = {}
    portable_names: Dict[str, str] = {}
    for raw_name, value in files.items():
        name = _json_artifact_name(raw_name)
        if name in result:
            raise SchemaError("analysis bundle contains duplicate artifact names")
        portable_name = _portable_artifact_name_key(name)
        previous = portable_names.get(portable_name)
        if previous is not None:
            raise SchemaError(
                "analysis bundle contains a portable filename collision: %s / %s"
                % (previous, name)
            )
        portable_names[portable_name] = name
        validate_safe_json(value, "analysis artifact %s" % name)
        result[name] = canonical_json_bytes(value)
    return {name: result[name] for name in sorted(result)}


@dataclass(frozen=True)
class AnalysisReceipt(_JsonModel):
    """The authoritative commit marker for one analysis JSON bundle."""

    schema_version: str
    kind: str
    artifact_id: str
    source_bindings: Tuple[AnalysisSourceBinding, ...]
    artifacts: Tuple[AnalysisArtifactRef, ...]
    algorithm_digest: str

    def __post_init__(self) -> None:
        if self.schema_version != ANALYSIS_RECEIPT_SCHEMA_VERSION:
            raise SchemaError("AnalysisReceipt has an unsupported schema version")
        kind = _kind(self.kind)
        artifact_id = _artifact_id(self.artifact_id)
        _digest(self.algorithm_digest, "analysis algorithm_digest")
        if type(self.source_bindings) not in (tuple, list):
            raise SchemaError("AnalysisReceipt source_bindings must be a list")
        sources = tuple(self.source_bindings)
        canonical_sources = _canonical_source_bindings(
            sources,
            "AnalysisReceipt",
        )
        if sources != canonical_sources:
            raise SchemaError("AnalysisReceipt source bindings are not canonical")
        if type(self.artifacts) not in (tuple, list):
            raise SchemaError("AnalysisReceipt artifacts must be a list")
        artifacts = tuple(self.artifacts)
        if not artifacts or len(artifacts) > MAX_ANALYSIS_ARTIFACTS:
            raise SchemaError("AnalysisReceipt artifacts are empty or excessive")
        if any(type(item) is not AnalysisArtifactRef for item in artifacts):
            raise SchemaError("AnalysisReceipt artifacts are invalid")
        if artifacts != tuple(sorted(artifacts, key=lambda item: item.relative_path)):
            raise SchemaError("AnalysisReceipt artifacts are not canonical")
        if len({item.relative_path for item in artifacts}) != len(artifacts):
            raise SchemaError("AnalysisReceipt contains duplicate artifact paths")
        portable_names = tuple(
            _portable_artifact_name_key(item.relative_path.rsplit("/", 1)[-1])
            for item in artifacts
        )
        if len(portable_names) != len(set(portable_names)):
            raise SchemaError(
                "AnalysisReceipt contains a portable artifact path collision"
            )
        if any(item.kind != kind or item.artifact_id != artifact_id for item in artifacts):
            raise SchemaError("AnalysisReceipt artifact refs leave their namespace")
        expected_id = derive_analysis_artifact_id(
            kind,
            sources,
            self.algorithm_digest,
        )
        if artifact_id != expected_id:
            raise SchemaError("AnalysisReceipt artifact ID is not canonical")
        object.__setattr__(self, "source_bindings", sources)
        object.__setattr__(self, "artifacts", artifacts)

    @classmethod
    def create(
        cls,
        *,
        kind: str,
        source_bindings: Iterable[AnalysisSourceBinding],
        algorithm_digest: str,
        files: Mapping[str, Any],
    ) -> "AnalysisReceipt":
        canonical_kind = _kind(kind)
        sources = _canonical_source_bindings(
            source_bindings,
            "AnalysisReceipt",
        )
        artifact_id = derive_analysis_artifact_id(
            canonical_kind,
            sources,
            algorithm_digest,
        )
        payloads = _canonical_file_payloads(files)
        artifacts = tuple(
            AnalysisArtifactRef.create(
                kind=canonical_kind,
                artifact_id=artifact_id,
                name=name,
                data=data,
            )
            for name, data in payloads.items()
        )
        return cls(
            schema_version=ANALYSIS_RECEIPT_SCHEMA_VERSION,
            kind=canonical_kind,
            artifact_id=artifact_id,
            source_bindings=sources,
            artifacts=artifacts,
            algorithm_digest=algorithm_digest,
        )

    @classmethod
    def from_dict(cls, value: Any) -> "AnalysisReceipt":
        payload = _exact_mapping(
            value,
            {
                "schema_version",
                "kind",
                "artifact_id",
                "source_bindings",
                "artifacts",
                "algorithm_digest",
            },
            "AnalysisReceipt",
        )
        if type(payload["source_bindings"]) is not list:
            raise SchemaError("AnalysisReceipt source_bindings must be a list")
        if type(payload["artifacts"]) is not list:
            raise SchemaError("AnalysisReceipt artifacts must be a list")
        return cls(
            schema_version=payload["schema_version"],
            kind=payload["kind"],
            artifact_id=payload["artifact_id"],
            source_bindings=tuple(
                AnalysisSourceBinding.from_dict(item)
                for item in payload["source_bindings"]
            ),
            artifacts=tuple(
                AnalysisArtifactRef.from_dict(item) for item in payload["artifacts"]
            ),
            algorithm_digest=payload["algorithm_digest"],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "artifact_id": self.artifact_id,
            "source_bindings": [item.to_dict() for item in self.source_bindings],
            "artifacts": [item.to_dict() for item in self.artifacts],
            "algorithm_digest": self.algorithm_digest,
        }


class _AnalysisSafeStorage(ArtifactStore):
    """Reuse ArtifactStore's hardened low-level I/O without its Run layout."""

    def __init__(
        self,
        root: os.PathLike[str] | str,
        *,
        create_root: bool,
        max_file_bytes: int,
        max_total_read_bytes: int,
    ) -> None:
        self._initialize_storage_root(
            root,
            max_file_bytes=max_file_bytes,
            max_total_read_bytes=max_total_read_bytes,
            create_root=create_root,
            required_root_name=None,
            reject_hardlinks=True,
        )


class AnalysisArtifactStore:
    """Create-only store for canonical JSON analysis bundles."""

    def __init__(
        self,
        root: os.PathLike[str] | str,
        *,
        create_root: bool = True,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        max_total_read_bytes: int = DEFAULT_MAX_TOTAL_READ_BYTES,
    ) -> None:
        _validate_analysis_root_boundary(root)
        self._storage = _AnalysisSafeStorage(
            root,
            create_root=create_root,
            max_file_bytes=max_file_bytes,
            max_total_read_bytes=max_total_read_bytes,
        )
        self.root = self._storage.root
        self.max_file_bytes = self._storage.max_file_bytes
        self.max_total_read_bytes = self._storage.max_total_read_bytes

    def _directory(self, kind: str, artifact_id: str) -> Path:
        canonical_kind = _kind(kind)
        canonical_id = _artifact_id(artifact_id)
        return self._storage._within_root(
            self.root / canonical_kind / canonical_id
        )

    def _entries(self, directory: Path) -> frozenset[str]:
        if not os.path.lexists(directory):
            return frozenset()
        self._storage._assert_directory(directory)
        names = set()
        try:
            with os.scandir(directory) as entries:
                for entry_count, entry in enumerate(entries, start=1):
                    if entry_count > MAX_ANALYSIS_ARTIFACTS + 1:
                        raise ArtifactIntegrityError(
                            "analysis namespace exceeds its child entry limit"
                        )
                    try:
                        # Windows DirEntry.stat may report st_nlink=0 even
                        # though direct lstat reports the authoritative count.
                        metadata = os.lstat(entry.path)
                    except OSError as exc:
                        raise ArtifactSecurityError(
                            "could not inspect analysis artifact"
                        ) from exc
                    if _unsafe_node(metadata) or not stat.S_ISREG(
                        metadata.st_mode
                    ):
                        raise ArtifactSecurityError(
                            "analysis namespace contains a symlink, reparse point, or unsafe entry"
                        )
                    if _hardlinked_file(metadata):
                        raise ArtifactSecurityError(
                            "analysis artifact has an unsafe hardlink count"
                        )
                    names.add(entry.name)
        except ArtifactIntegrityError:
            raise
        except OSError as exc:
            raise ArtifactSecurityError(
                "could not inspect analysis artifact namespace"
            ) from exc
        return frozenset(names)

    def _read_receipt(
        self,
        directory: Path,
        *,
        budget: _ReadBudget,
    ) -> AnalysisReceipt:
        try:
            payload = self._storage._read_json(
                directory / _RECEIPT_NAME,
                budget=budget,
            )
            return AnalysisReceipt.from_dict(payload)
        except ArtifactIntegrityError:
            raise
        except (SchemaError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "analysis receipt violates its canonical schema"
            ) from exc

    def _read_ref(
        self,
        ref: AnalysisArtifactRef,
        *,
        budget: _ReadBudget,
    ) -> Any:
        path = self.root / Path(*ref.relative_path.split("/"))
        data = self._storage._read_bytes(
            path,
            expected_sha256=ref.sha256,
            expected_size=ref.size_bytes,
            budget=budget,
        )
        value = _strict_json_loads(data, self.max_file_bytes, "analysis artifact JSON")
        if canonical_json_bytes(value) != data:
            raise ArtifactIntegrityError(
                "analysis JSON artifact is not canonical UTF-8 JSON"
            )
        try:
            validate_safe_json(value, "analysis artifact")
        except (SchemaError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "analysis JSON artifact violates the safe JSON boundary"
            ) from exc
        return value

    def _load_with_receipt(
        self,
        kind: str,
        artifact_id: str,
    ) -> tuple[AnalysisReceipt, Dict[str, Any]]:
        directory = self._directory(kind, artifact_id)
        names = self._entries(directory)
        if not names:
            raise ArtifactIntegrityError("analysis artifact is unknown or uncommitted")
        if _RECEIPT_NAME not in names:
            raise ArtifactIntegrityError("analysis artifact has no receipt commit marker")
        budget = _ReadBudget(self.max_total_read_bytes)
        receipt = self._read_receipt(directory, budget=budget)
        if receipt.kind != kind or receipt.artifact_id != artifact_id:
            raise ArtifactIntegrityError(
                "analysis receipt belongs to another artifact namespace"
            )
        expected_names = {
            Path(ref.relative_path).name for ref in receipt.artifacts
        } | {_RECEIPT_NAME}
        unknown = names - expected_names
        if unknown:
            raise ArtifactIntegrityError(
                "analysis namespace contains an unknown artifact"
            )
        if names != expected_names:
            raise ArtifactIntegrityError("analysis namespace is incomplete")
        decoded = {}
        for ref in receipt.artifacts:
            decoded[Path(ref.relative_path).name] = self._read_ref(
                ref,
                budget=budget,
            )
        return receipt, decoded

    def publish_json_bundle(
        self,
        kind: str,
        artifact_id: str,
        files: Mapping[str, Any],
        receipt: AnalysisReceipt,
    ) -> AnalysisReceipt:
        canonical_kind = _kind(kind)
        canonical_id = _artifact_id(artifact_id)
        if type(receipt) is not AnalysisReceipt:
            raise TypeError("receipt must be an AnalysisReceipt")
        try:
            receipt_payload = receipt.to_dict()
            replayed_receipt = AnalysisReceipt.from_dict(receipt_payload)
            receipt_data = canonical_json_bytes(receipt_payload)
            replayed_receipt_data = canonical_json_bytes(
                replayed_receipt.to_dict()
            )
        except (AttributeError, SchemaError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "analysis receipt fails canonical source-bound hydration"
            ) from exc
        if (
            replayed_receipt != receipt
            or replayed_receipt_data != receipt_data
        ):
            raise ArtifactIntegrityError(
                "analysis receipt differs from canonical hydration"
            )
        receipt = replayed_receipt
        if receipt.kind != canonical_kind or receipt.artifact_id != canonical_id:
            raise ArtifactIntegrityError(
                "analysis receipt differs from the requested namespace"
            )
        payloads = _canonical_file_payloads(files)
        if any(len(data) > self.max_file_bytes for data in payloads.values()):
            raise ArtifactIntegrityError(
                "analysis JSON artifact exceeds the single-file byte limit"
            )
        expected_refs = tuple(
            AnalysisArtifactRef.create(
                kind=canonical_kind,
                artifact_id=canonical_id,
                name=name,
                data=data,
            )
            for name, data in payloads.items()
        )
        if receipt.artifacts != expected_refs:
            raise ArtifactIntegrityError(
                "analysis receipt artifact digests differ from planned bytes"
            )
        validate_safe_json(receipt.to_dict(), "analysis receipt")
        if len(receipt_data) > self.max_file_bytes:
            raise ArtifactIntegrityError(
                "analysis receipt exceeds the single-file byte limit"
            )
        if sum(len(item) for item in payloads.values()) + len(receipt_data) > (
            self.max_total_read_bytes
        ):
            raise ArtifactIntegrityError(
                "analysis bundle exceeds the cumulative byte limit"
            )

        directory = self._directory(canonical_kind, canonical_id)
        lock_name = stable_id(
            "analysis-lock-v1",
            {"kind": canonical_kind, "artifact_id": canonical_id},
        ) + ".lock"
        with self._storage._lock(self.root / ".locks" / lock_name):
            names = self._entries(directory)
            if _RECEIPT_NAME in names:
                stored_receipt, stored = self._load_with_receipt(
                    canonical_kind,
                    canonical_id,
                )
                if (
                    stored_receipt != receipt
                    or canonical_json_bytes(stored) != canonical_json_bytes(dict(files))
                ):
                    raise ArtifactConflictError(
                        "existing analysis bundle differs from requested bytes"
                    )
                return stored_receipt
            expected_names = set(payloads)
            unknown = names - expected_names
            if unknown:
                raise ArtifactIntegrityError(
                    "analysis namespace contains an unknown orphan artifact"
                )
            self._storage._ensure_directory(directory)
            for name, data in payloads.items():
                path = directory / name
                if name in names:
                    ref = next(
                        item
                        for item in expected_refs
                        if Path(item.relative_path).name == name
                    )
                    try:
                        self._storage._read_bytes(
                            path,
                            expected_sha256=ref.sha256,
                            expected_size=ref.size_bytes,
                            budget=_ReadBudget(self.max_total_read_bytes),
                        )
                    except ArtifactSecurityError:
                        raise
                    except ArtifactIntegrityError as exc:
                        raise ArtifactConflictError(
                            "existing analysis orphan differs from requested bytes"
                        ) from exc
                    continue
                self._storage._write_bytes_exclusive(path, data)
            # receipt.json is the sole commit marker and is always published last.
            self._storage._write_bytes_exclusive(
                directory / _RECEIPT_NAME,
                receipt_data,
            )
            stored_receipt, stored = self._load_with_receipt(
                canonical_kind,
                canonical_id,
            )
            if (
                stored_receipt != receipt
                or canonical_json_bytes(stored) != canonical_json_bytes(dict(files))
            ):
                raise ArtifactIntegrityError(
                    "published analysis bundle differs from its planned identity"
                )
            return stored_receipt

    def load_json_bundle(self, kind: str, artifact_id: str) -> Dict[str, Any]:
        _receipt, decoded = self._load_with_receipt(
            _kind(kind),
            _artifact_id(artifact_id),
        )
        return decoded

    def publish_comparison(
        self,
        comparison: Any,
        *,
        policy: Any,
    ) -> AnalysisReceipt:
        """Publish one canonical comparison result through the receipt-last seam."""

        from .comparison import ComparisonPolicyV1, RunComparisonV1

        if type(comparison) is not RunComparisonV1:
            raise TypeError("comparison must be a RunComparisonV1")
        if type(policy) is not ComparisonPolicyV1:
            raise TypeError("policy must be a ComparisonPolicyV1")
        try:
            canonical_policy = ComparisonPolicyV1.from_dict(policy.to_dict())
            replayed = RunComparisonV1.from_dict(comparison.to_dict())
        except (SchemaError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "comparison fails strict canonical hydration"
            ) from exc
        if replayed != comparison:
            raise ArtifactIntegrityError(
                "comparison differs from strict canonical hydration"
            )
        if (
            replayed.algorithm_digest != canonical_policy.algorithm_digest
            or replayed.compatibility.policy_digest
            != canonical_policy.policy_digest
        ):
            raise ArtifactIntegrityError(
                "comparison algorithm/policy digest differs from publication policy"
            )
        files = {"comparison_result.json": replayed.to_dict()}
        receipt = AnalysisReceipt.create(
            kind="comparison",
            source_bindings=(
                replayed.baseline_binding,
                replayed.candidate_binding,
            ),
            algorithm_digest=replayed.algorithm_digest,
            files=files,
        )
        return self.publish_json_bundle(
            receipt.kind,
            receipt.artifact_id,
            files,
            receipt,
        )

    def load_comparison(self, artifact_id: str) -> Any:
        """Strictly hydrate a stored comparison without claiming source replay."""

        from .comparison import RunComparisonV1

        receipt, files = self._load_with_receipt(
            "comparison",
            _artifact_id(artifact_id),
        )
        if set(files) != {"comparison_result.json"}:
            raise ArtifactIntegrityError(
                "comparison bundle has an invalid exact artifact set"
            )
        try:
            result = RunComparisonV1.from_dict(
                files["comparison_result.json"]
            )
            expected_sources = _canonical_source_bindings(
                (result.baseline_binding, result.candidate_binding),
                "comparison receipt",
            )
        except (SchemaError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "comparison result violates its strict canonical schema"
            ) from exc
        if (
            receipt.source_bindings != expected_sources
            or receipt.algorithm_digest != result.algorithm_digest
        ):
            raise ArtifactIntegrityError(
                "comparison receipt differs from nested bindings or algorithm"
            )
        return result

    def load_verified_comparison(
        self,
        artifact_id: str,
        *,
        baseline: Any,
        candidate: Any,
        policy: Any,
    ) -> Any:
        """Replay a comparison from caller-supplied verified source Evaluations."""

        from .comparison import (
            ComparisonPolicyV1,
            VerifiedRunEvaluation,
            compare_runs,
        )

        if type(baseline) is not VerifiedRunEvaluation:
            raise TypeError("baseline must be a VerifiedRunEvaluation")
        if type(candidate) is not VerifiedRunEvaluation:
            raise TypeError("candidate must be a VerifiedRunEvaluation")
        if type(policy) is not ComparisonPolicyV1:
            raise TypeError("policy must be a ComparisonPolicyV1")
        stored = self.load_comparison(artifact_id)
        if (
            stored.baseline_binding != baseline.source_binding
            or stored.candidate_binding != candidate.source_binding
        ):
            raise ArtifactIntegrityError(
                "caller-supplied comparison sources differ from stored provenance"
            )
        try:
            replayed = compare_runs(baseline, candidate, policy)
        except (SchemaError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "comparison source replay failed"
            ) from exc
        if canonical_json_bytes(replayed.to_dict()) != canonical_json_bytes(
            stored.to_dict()
        ):
            raise ArtifactIntegrityError(
                "stored comparison differs from exact source replay"
            )
        return stored

    def publish_calibration_package(
        self,
        package: Any,
        *,
        evaluation: Any,
        policy: Any,
    ) -> AnalysisReceipt:
        """Publish only the source-bound manifest for an external blind package."""

        from .calibration import (
            CALIBRATION_ALGORITHM_VERSION,
            CalibrationPackageManifestV1,
            CalibrationPackageV1,
            CalibrationSelectionPolicyV1,
            _build_package,
        )
        from .comparison import VerifiedRunEvaluation

        if type(package) is not CalibrationPackageV1:
            raise TypeError("package must be a CalibrationPackageV1")
        if type(evaluation) is not VerifiedRunEvaluation:
            raise TypeError("evaluation must be a VerifiedRunEvaluation")
        if type(policy) is not CalibrationSelectionPolicyV1:
            raise TypeError("policy must be a CalibrationSelectionPolicyV1")
        try:
            canonical_policy = CalibrationSelectionPolicyV1.from_dict(
                policy.to_dict()
            )
            canonical_package = CalibrationPackageV1.from_dict(package.to_dict())
            source_binding = evaluation.verify()
            replayed = _build_package(
                evaluation,
                profile=canonical_package.profile,
                policy=canonical_policy,
            )
            if canonical_json_bytes(replayed.to_dict()) != canonical_json_bytes(
                canonical_package.to_dict()
            ):
                raise ValueError("package differs from exact Evaluation replay")
            manifest = CalibrationPackageManifestV1.from_package(replayed)
        except ArtifactIntegrityError:
            raise
        except (AttributeError, SchemaError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "calibration package fails strict source-bound replay"
            ) from exc
        files = {"calibration_package_manifest.json": manifest.to_dict()}
        algorithm_digest = _calibration_publication_digest(
            "calibration-package-manifest",
            {
                "algorithm_version": CALIBRATION_ALGORITHM_VERSION,
                "manifest": manifest.to_dict(),
            },
        )
        receipt = AnalysisReceipt.create(
            kind="calibration-package",
            source_bindings=(source_binding,),
            algorithm_digest=algorithm_digest,
            files=files,
        )
        return self.publish_json_bundle(
            receipt.kind,
            receipt.artifact_id,
            files,
            receipt,
        )

    def _load_calibration_package_manifest_with_receipt(
        self,
        artifact_id: str,
    ) -> tuple[AnalysisReceipt, Any]:
        from .calibration import (
            CALIBRATION_ALGORITHM_VERSION,
            CalibrationPackageManifestV1,
        )

        receipt, files = self._load_with_receipt(
            "calibration-package",
            _artifact_id(artifact_id),
        )
        if set(files) != {"calibration_package_manifest.json"}:
            raise ArtifactIntegrityError(
                "calibration package bundle has an invalid exact artifact set"
            )
        try:
            manifest = CalibrationPackageManifestV1.from_dict(
                files["calibration_package_manifest.json"]
            )
            expected_algorithm = _calibration_publication_digest(
                "calibration-package-manifest",
                {
                    "algorithm_version": CALIBRATION_ALGORITHM_VERSION,
                    "manifest": manifest.to_dict(),
                },
            )
        except (SchemaError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "calibration package manifest violates its strict schema"
            ) from exc
        if (
            len(receipt.source_bindings) != 1
            or receipt.algorithm_digest != expected_algorithm
        ):
            raise ArtifactIntegrityError(
                "calibration package receipt differs from nested manifest bindings"
            )
        return receipt, manifest

    def load_calibration_package_manifest(self, artifact_id: str) -> Any:
        """Hydrate a manifest without claiming Evaluation source replay."""

        _receipt, manifest = self._load_calibration_package_manifest_with_receipt(
            artifact_id
        )
        return manifest

    def load_verified_calibration_package_manifest(
        self,
        artifact_id: str,
        *,
        evaluation: Any,
        policy: Any,
        package: Any,
    ) -> Any:
        """Replay a package manifest from caller-supplied verified sources."""

        from .calibration import (
            CalibrationPackageManifestV1,
            CalibrationPackageV1,
            CalibrationSelectionPolicyV1,
            _build_package,
        )
        from .comparison import VerifiedRunEvaluation

        if type(evaluation) is not VerifiedRunEvaluation:
            raise TypeError("evaluation must be a VerifiedRunEvaluation")
        if type(policy) is not CalibrationSelectionPolicyV1:
            raise TypeError("policy must be a CalibrationSelectionPolicyV1")
        if type(package) is not CalibrationPackageV1:
            raise TypeError("package must be a CalibrationPackageV1")
        receipt, stored = self._load_calibration_package_manifest_with_receipt(
            artifact_id
        )
        try:
            source_binding = evaluation.verify()
            canonical_policy = CalibrationSelectionPolicyV1.from_dict(
                policy.to_dict()
            )
            canonical_package = CalibrationPackageV1.from_dict(package.to_dict())
            replayed_package = _build_package(
                evaluation,
                profile=canonical_package.profile,
                policy=canonical_policy,
            )
            if canonical_json_bytes(replayed_package.to_dict()) != canonical_json_bytes(
                canonical_package.to_dict()
            ):
                raise ValueError("caller package differs from Evaluation replay")
            replayed_manifest = CalibrationPackageManifestV1.from_package(
                replayed_package
            )
        except ArtifactIntegrityError:
            raise
        except (AttributeError, SchemaError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "calibration package source replay failed"
            ) from exc
        if receipt.source_bindings != (source_binding,):
            raise ArtifactIntegrityError(
                "caller Evaluation differs from calibration package provenance"
            )
        if canonical_json_bytes(replayed_manifest.to_dict()) != canonical_json_bytes(
            stored.to_dict()
        ):
            raise ArtifactIntegrityError(
                "stored calibration package manifest differs from exact source replay"
            )
        return stored

    def publish_human_label_set(
        self,
        labels: Any,
        *,
        evaluation: Any,
        policy: Any,
        package: Any,
    ) -> AnalysisReceipt:
        """Publish human provenance under a separate create-only result namespace."""

        from .calibration import (
            CALIBRATION_ALGORITHM_VERSION,
            CalibrationPackageV1,
            CalibrationSelectionPolicyV1,
            HumanLabelSetV1,
            _build_package,
        )
        from .comparison import VerifiedRunEvaluation

        if type(labels) is not HumanLabelSetV1:
            raise TypeError("labels must be a HumanLabelSetV1")
        if type(evaluation) is not VerifiedRunEvaluation:
            raise TypeError("evaluation must be a VerifiedRunEvaluation")
        if type(policy) is not CalibrationSelectionPolicyV1:
            raise TypeError("policy must be a CalibrationSelectionPolicyV1")
        if type(package) is not CalibrationPackageV1:
            raise TypeError("package must be a CalibrationPackageV1")
        try:
            canonical_policy = CalibrationSelectionPolicyV1.from_dict(
                policy.to_dict()
            )
            canonical_package = CalibrationPackageV1.from_dict(package.to_dict())
            source_binding = evaluation.verify()
            replayed_package = _build_package(
                evaluation,
                profile=canonical_package.profile,
                policy=canonical_policy,
            )
            if canonical_json_bytes(replayed_package.to_dict()) != canonical_json_bytes(
                canonical_package.to_dict()
            ):
                raise ValueError("package differs from exact Evaluation replay")
            canonical_labels = HumanLabelSetV1.from_dict(
                labels.to_dict(),
                package=replayed_package,
            )
            if canonical_json_bytes(canonical_labels.to_dict()) != canonical_json_bytes(
                labels.to_dict()
            ):
                raise ValueError("label set differs from canonical package replay")
        except ArtifactIntegrityError:
            raise
        except (AttributeError, SchemaError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "human label set fails strict source/package replay"
            ) from exc
        files = {"human_label_set.json": canonical_labels.to_dict()}
        algorithm_digest = _calibration_publication_digest(
            "human-label-set",
            {
                "algorithm_version": CALIBRATION_ALGORITHM_VERSION,
                "source_digest": replayed_package.source_digest,
                "policy_digest": canonical_policy.digest(),
                "package_id": replayed_package.package_id,
                "package_digest": replayed_package.digest(),
                "payload_digest": replayed_package.payload_digest,
                "selection_digest": replayed_package.selection_digest,
                "label_set_id": canonical_labels.label_set_id,
                "label_set_digest": canonical_labels.digest(),
                "human_provenance_digest": _human_provenance_digest(
                    canonical_labels
                ),
            },
        )
        receipt = AnalysisReceipt.create(
            kind="calibration-result",
            source_bindings=(source_binding,),
            algorithm_digest=algorithm_digest,
            files=files,
        )
        return self.publish_json_bundle(
            receipt.kind,
            receipt.artifact_id,
            files,
            receipt,
        )

    def _load_human_label_set_with_receipt(
        self,
        artifact_id: str,
        *,
        package: Any,
    ) -> tuple[AnalysisReceipt, Any]:
        from .calibration import (
            CALIBRATION_ALGORITHM_VERSION,
            CalibrationPackageV1,
            HumanLabelSetV1,
        )

        if type(package) is not CalibrationPackageV1:
            raise TypeError("package must be a CalibrationPackageV1")
        receipt, files = self._load_with_receipt(
            "calibration-result",
            _artifact_id(artifact_id),
        )
        if set(files) != {"human_label_set.json"}:
            raise ArtifactIntegrityError(
                "human label bundle has an invalid exact artifact set"
            )
        try:
            canonical_package = CalibrationPackageV1.from_dict(package.to_dict())
            labels = HumanLabelSetV1.from_dict(
                files["human_label_set.json"],
                package=canonical_package,
            )
            expected_algorithm = _calibration_publication_digest(
                "human-label-set",
                {
                    "algorithm_version": CALIBRATION_ALGORITHM_VERSION,
                    "source_digest": canonical_package.source_digest,
                    "policy_digest": canonical_package.policy.digest(),
                    "package_id": canonical_package.package_id,
                    "package_digest": canonical_package.digest(),
                    "payload_digest": canonical_package.payload_digest,
                    "selection_digest": canonical_package.selection_digest,
                    "label_set_id": labels.label_set_id,
                    "label_set_digest": labels.digest(),
                    "human_provenance_digest": _human_provenance_digest(labels),
                },
            )
        except (AttributeError, SchemaError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "human label set violates strict package binding"
            ) from exc
        if (
            len(receipt.source_bindings) != 1
            or receipt.algorithm_digest != expected_algorithm
        ):
            raise ArtifactIntegrityError(
                "human label receipt differs from nested package/label digests"
            )
        return receipt, labels

    def load_human_label_set(self, artifact_id: str, *, package: Any) -> Any:
        """Hydrate labels against a package without claiming Evaluation replay."""

        _receipt, labels = self._load_human_label_set_with_receipt(
            artifact_id,
            package=package,
        )
        return labels

    def load_verified_human_label_set(
        self,
        artifact_id: str,
        *,
        evaluation: Any,
        policy: Any,
        package: Any,
        labels: Any,
    ) -> Any:
        """Replay stored labels against every caller-supplied calibration source."""

        from .calibration import (
            CalibrationPackageV1,
            CalibrationSelectionPolicyV1,
            HumanLabelSetV1,
            _build_package,
        )
        from .comparison import VerifiedRunEvaluation

        if type(evaluation) is not VerifiedRunEvaluation:
            raise TypeError("evaluation must be a VerifiedRunEvaluation")
        if type(policy) is not CalibrationSelectionPolicyV1:
            raise TypeError("policy must be a CalibrationSelectionPolicyV1")
        if type(package) is not CalibrationPackageV1:
            raise TypeError("package must be a CalibrationPackageV1")
        if type(labels) is not HumanLabelSetV1:
            raise TypeError("labels must be a HumanLabelSetV1")
        receipt, stored = self._load_human_label_set_with_receipt(
            artifact_id,
            package=package,
        )
        try:
            source_binding = evaluation.verify()
            canonical_policy = CalibrationSelectionPolicyV1.from_dict(
                policy.to_dict()
            )
            canonical_package = CalibrationPackageV1.from_dict(package.to_dict())
            replayed_package = _build_package(
                evaluation,
                profile=canonical_package.profile,
                policy=canonical_policy,
            )
            if canonical_json_bytes(replayed_package.to_dict()) != canonical_json_bytes(
                canonical_package.to_dict()
            ):
                raise ValueError("caller package differs from Evaluation replay")
            replayed_labels = HumanLabelSetV1.from_dict(
                labels.to_dict(),
                package=replayed_package,
            )
        except ArtifactIntegrityError:
            raise
        except (AttributeError, SchemaError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "human label source replay failed"
            ) from exc
        if receipt.source_bindings != (source_binding,):
            raise ArtifactIntegrityError(
                "caller Evaluation differs from human label provenance"
            )
        if canonical_json_bytes(replayed_labels.to_dict()) != canonical_json_bytes(
            stored.to_dict()
        ):
            raise ArtifactIntegrityError(
                "stored human label set differs from exact source replay"
            )
        return stored

    def publish_calibration_result(
        self,
        result: Any,
        *,
        evaluation: Any,
        policy: Any,
        package: Any,
        labels: Any,
    ) -> AnalysisReceipt:
        """Publish a scored profile only after exact offline source replay."""

        from .calibration import (
            CALIBRATION_ALGORITHM_VERSION,
            CalibrationPackageV1,
            CalibrationResultV1,
            CalibrationSelectionPolicyV1,
            HumanLabelSetV1,
            _build_package,
            score_calibration,
        )
        from .comparison import VerifiedRunEvaluation

        if type(result) is not CalibrationResultV1:
            raise TypeError("result must be a CalibrationResultV1")
        if type(evaluation) is not VerifiedRunEvaluation:
            raise TypeError("evaluation must be a VerifiedRunEvaluation")
        if type(policy) is not CalibrationSelectionPolicyV1:
            raise TypeError("policy must be a CalibrationSelectionPolicyV1")
        if type(package) is not CalibrationPackageV1:
            raise TypeError("package must be a CalibrationPackageV1")
        if type(labels) is not HumanLabelSetV1:
            raise TypeError("labels must be a HumanLabelSetV1")
        try:
            canonical_policy = CalibrationSelectionPolicyV1.from_dict(
                policy.to_dict()
            )
            canonical_package = CalibrationPackageV1.from_dict(package.to_dict())
            source_binding = evaluation.verify()
            replayed_package = _build_package(
                evaluation,
                profile=canonical_package.profile,
                policy=canonical_policy,
            )
            if canonical_json_bytes(replayed_package.to_dict()) != canonical_json_bytes(
                canonical_package.to_dict()
            ):
                raise ValueError("package differs from exact Evaluation replay")
            canonical_labels = HumanLabelSetV1.from_dict(
                labels.to_dict(),
                package=replayed_package,
            )
            canonical_result = CalibrationResultV1.from_dict(result.to_dict())
            replayed_result = score_calibration(
                evaluation,
                package=replayed_package,
                labels=canonical_labels,
            )
            if canonical_json_bytes(replayed_result.to_dict()) != canonical_json_bytes(
                canonical_result.to_dict()
            ):
                raise ValueError("calibration result differs from exact score replay")
        except ArtifactIntegrityError:
            raise
        except (AttributeError, SchemaError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "calibration result fails strict source/package/label replay"
            ) from exc
        files = {"calibration_result.json": replayed_result.to_dict()}
        algorithm_digest = _calibration_publication_digest(
            "calibration-result",
            {
                "algorithm_version": CALIBRATION_ALGORITHM_VERSION,
                "source_digest": replayed_result.source_digest,
                "policy_digest": replayed_result.policy.digest(),
                "package_id": replayed_result.package_id,
                "package_digest": replayed_result.package_digest,
                "payload_digest": replayed_result.payload_digest,
                "label_set_id": replayed_result.label_set_id,
                "label_set_digest": replayed_result.label_set_digest,
                "calibration_result_id": replayed_result.calibration_result_id,
                "calibration_result_digest": replayed_result.digest(),
            },
        )
        receipt = AnalysisReceipt.create(
            kind="calibration-result",
            source_bindings=(source_binding,),
            algorithm_digest=algorithm_digest,
            files=files,
        )
        return self.publish_json_bundle(
            receipt.kind,
            receipt.artifact_id,
            files,
            receipt,
        )

    def _load_calibration_result_with_receipt(
        self,
        artifact_id: str,
    ) -> tuple[AnalysisReceipt, Any]:
        from .calibration import CALIBRATION_ALGORITHM_VERSION, CalibrationResultV1

        receipt, files = self._load_with_receipt(
            "calibration-result",
            _artifact_id(artifact_id),
        )
        if set(files) != {"calibration_result.json"}:
            raise ArtifactIntegrityError(
                "calibration result bundle has an invalid exact artifact set"
            )
        try:
            result = CalibrationResultV1.from_dict(
                files["calibration_result.json"]
            )
            expected_algorithm = _calibration_publication_digest(
                "calibration-result",
                {
                    "algorithm_version": CALIBRATION_ALGORITHM_VERSION,
                    "source_digest": result.source_digest,
                    "policy_digest": result.policy.digest(),
                    "package_id": result.package_id,
                    "package_digest": result.package_digest,
                    "payload_digest": result.payload_digest,
                    "label_set_id": result.label_set_id,
                    "label_set_digest": result.label_set_digest,
                    "calibration_result_id": result.calibration_result_id,
                    "calibration_result_digest": result.digest(),
                },
            )
        except (SchemaError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "calibration result violates its strict canonical schema"
            ) from exc
        if (
            len(receipt.source_bindings) != 1
            or receipt.algorithm_digest != expected_algorithm
        ):
            raise ArtifactIntegrityError(
                "calibration result receipt differs from nested source/policy/package/label digests"
            )
        return receipt, result

    def load_calibration_result(self, artifact_id: str) -> Any:
        """Hydrate a result without claiming Evaluation or human-label replay."""

        _receipt, result = self._load_calibration_result_with_receipt(artifact_id)
        return result

    def load_verified_calibration_result(
        self,
        artifact_id: str,
        *,
        evaluation: Any,
        policy: Any,
        package: Any,
        labels: Any,
    ) -> Any:
        """Recompute and byte-compare a stored result from all trusted inputs."""

        from .calibration import (
            CalibrationPackageV1,
            CalibrationSelectionPolicyV1,
            HumanLabelSetV1,
            _build_package,
            score_calibration,
        )
        from .comparison import VerifiedRunEvaluation

        if type(evaluation) is not VerifiedRunEvaluation:
            raise TypeError("evaluation must be a VerifiedRunEvaluation")
        if type(policy) is not CalibrationSelectionPolicyV1:
            raise TypeError("policy must be a CalibrationSelectionPolicyV1")
        if type(package) is not CalibrationPackageV1:
            raise TypeError("package must be a CalibrationPackageV1")
        if type(labels) is not HumanLabelSetV1:
            raise TypeError("labels must be a HumanLabelSetV1")
        receipt, stored = self._load_calibration_result_with_receipt(artifact_id)
        try:
            source_binding = evaluation.verify()
            canonical_policy = CalibrationSelectionPolicyV1.from_dict(
                policy.to_dict()
            )
            canonical_package = CalibrationPackageV1.from_dict(package.to_dict())
            replayed_package = _build_package(
                evaluation,
                profile=canonical_package.profile,
                policy=canonical_policy,
            )
            if canonical_json_bytes(replayed_package.to_dict()) != canonical_json_bytes(
                canonical_package.to_dict()
            ):
                raise ValueError("caller package differs from Evaluation replay")
            canonical_labels = HumanLabelSetV1.from_dict(
                labels.to_dict(),
                package=replayed_package,
            )
            replayed = score_calibration(
                evaluation,
                package=replayed_package,
                labels=canonical_labels,
            )
        except ArtifactIntegrityError:
            raise
        except (AttributeError, SchemaError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "calibration result source replay failed"
            ) from exc
        if receipt.source_bindings != (source_binding,):
            raise ArtifactIntegrityError(
                "caller Evaluation differs from calibration result provenance"
            )
        if canonical_json_bytes(replayed.to_dict()) != canonical_json_bytes(
            stored.to_dict()
        ):
            raise ArtifactIntegrityError(
                "stored calibration result differs from exact source replay"
            )
        return stored

    def publish_gate_policy(
        self,
        policy: Any,
        *,
        baseline: Any,
        candidate_run_config: Any,
    ) -> AnalysisReceipt:
        """Commit a prepared Gate Policy; this is its create-only freeze point."""

        from .comparison import VerifiedRunEvaluation
        from .gates import GatePolicyV1, prepare_gate_policy

        if type(policy) is not GatePolicyV1:
            raise TypeError("policy must be a GatePolicyV1")
        if type(baseline) is not VerifiedRunEvaluation:
            raise TypeError("baseline must be a VerifiedRunEvaluation")
        if type(candidate_run_config) is not EvalRunConfig:
            raise TypeError("candidate_run_config must be an EvalRunConfig")
        try:
            canonical = GatePolicyV1.from_dict(policy.to_dict())
            replayed = prepare_gate_policy(
                baseline,
                candidate_run_config,
                policy=canonical,
            )
        except ArtifactIntegrityError:
            raise
        except (AttributeError, SchemaError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "gate policy fails strict baseline/RunConfig replay"
            ) from exc
        if (
            canonical != policy
            or canonical_json_bytes(replayed.to_dict())
            != canonical_json_bytes(canonical.to_dict())
        ):
            raise ArtifactIntegrityError(
                "gate policy differs from its prepared canonical identity"
            )
        files = {"gate_policy.json": replayed.to_dict()}
        receipt = AnalysisReceipt.create(
            kind="gate-policy",
            source_bindings=(replayed.baseline_binding,),
            algorithm_digest=replayed.algorithm_digest,
            files=files,
        )
        return self.publish_json_bundle(
            receipt.kind,
            receipt.artifact_id,
            files,
            receipt,
        )

    def _load_gate_policy_with_receipt(
        self,
        artifact_id: str,
    ) -> tuple[AnalysisReceipt, Any]:
        from .gates import GatePolicyV1

        receipt, files = self._load_with_receipt(
            "gate-policy",
            _artifact_id(artifact_id),
        )
        if set(files) != {"gate_policy.json"}:
            raise ArtifactIntegrityError(
                "gate policy bundle has an invalid exact artifact set"
            )
        try:
            policy = GatePolicyV1.from_dict(files["gate_policy.json"])
        except (SchemaError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "gate policy violates its strict canonical schema"
            ) from exc
        if (
            receipt.source_bindings != (policy.baseline_binding,)
            or receipt.algorithm_digest != policy.algorithm_digest
        ):
            raise ArtifactIntegrityError(
                "gate policy receipt differs from baseline/Run plan bindings"
            )
        return receipt, policy

    def load_gate_policy(self, artifact_id: str) -> Any:
        """Hydrate a frozen policy without claiming baseline source replay."""

        _receipt, policy = self._load_gate_policy_with_receipt(artifact_id)
        return policy

    def load_verified_gate_policy(
        self,
        artifact_id: str,
        *,
        baseline: Any,
        candidate_run_config: Any,
    ) -> Any:
        """Re-prepare and byte-compare a stored policy from trusted inputs."""

        from .comparison import VerifiedRunEvaluation
        from .gates import prepare_gate_policy

        if type(baseline) is not VerifiedRunEvaluation:
            raise TypeError("baseline must be a VerifiedRunEvaluation")
        if type(candidate_run_config) is not EvalRunConfig:
            raise TypeError("candidate_run_config must be an EvalRunConfig")
        receipt, stored = self._load_gate_policy_with_receipt(artifact_id)
        try:
            replayed = prepare_gate_policy(
                baseline,
                candidate_run_config,
                policy=stored,
            )
            source_binding = baseline.verify()
        except ArtifactIntegrityError:
            raise
        except (SchemaError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError("gate policy source replay failed") from exc
        if receipt.source_bindings != (source_binding,):
            raise ArtifactIntegrityError(
                "caller baseline differs from gate policy provenance"
            )
        if canonical_json_bytes(replayed.to_dict()) != canonical_json_bytes(
            stored.to_dict()
        ):
            raise ArtifactIntegrityError(
                "stored gate policy differs from exact source replay"
            )
        return stored

    @staticmethod
    def _gate_result_publication_digest(policy: Any, result: Any) -> str:
        from .gates import GATE_ALGORITHM_VERSION

        return hashlib.sha256(
            canonical_json_bytes(
                {
                    "algorithm_version": GATE_ALGORITHM_VERSION,
                    "policy_digest": result.policy_digest,
                    "comparison_id": result.comparison_id,
                    "calibration_result_digests": list(
                        policy.calibration_result_digests
                    ),
                }
            )
        ).hexdigest()

    def publish_gate_result(
        self,
        result: Any,
        *,
        policy: Any,
        comparison: Any,
        calibrations: Mapping[Any, Any],
    ) -> AnalysisReceipt:
        """Publish a result only after exact policy/comparison/calibration replay."""

        from .comparison import RunComparisonV1
        from .gates import GatePolicyV1, GateResultV1, evaluate_gate

        if type(result) is not GateResultV1:
            raise TypeError("result must be a GateResultV1")
        if type(policy) is not GatePolicyV1:
            raise TypeError("policy must be a GatePolicyV1")
        if type(comparison) is not RunComparisonV1:
            raise TypeError("comparison must be a RunComparisonV1")
        try:
            canonical_result = GateResultV1.from_dict(result.to_dict())
            replayed = evaluate_gate(policy, comparison, calibrations)
        except ArtifactIntegrityError:
            raise
        except (AttributeError, SchemaError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "gate result fails strict policy/comparison/calibration replay"
            ) from exc
        if (
            canonical_result != result
            or canonical_json_bytes(replayed.to_dict())
            != canonical_json_bytes(canonical_result.to_dict())
        ):
            raise ArtifactIntegrityError(
                "gate result differs from exact source replay"
            )
        files = {"gate_result.json": replayed.to_dict()}
        receipt = AnalysisReceipt.create(
            kind="gate-result",
            source_bindings=(
                comparison.baseline_binding,
                comparison.candidate_binding,
            ),
            algorithm_digest=self._gate_result_publication_digest(
                policy,
                replayed,
            ),
            files=files,
        )
        return self.publish_json_bundle(
            receipt.kind,
            receipt.artifact_id,
            files,
            receipt,
        )

    def _load_gate_result_with_receipt(
        self,
        artifact_id: str,
    ) -> tuple[AnalysisReceipt, Any]:
        from .gates import GateResultV1

        receipt, files = self._load_with_receipt(
            "gate-result",
            _artifact_id(artifact_id),
        )
        if set(files) != {"gate_result.json"}:
            raise ArtifactIntegrityError(
                "gate result bundle has an invalid exact artifact set"
            )
        try:
            result = GateResultV1.from_dict(files["gate_result.json"])
        except (SchemaError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "gate result violates its strict canonical schema"
            ) from exc
        if (
            len(receipt.source_bindings) != 2
            or receipt.algorithm_digest != result.algorithm_digest
        ):
            raise ArtifactIntegrityError(
                "gate result receipt differs from nested policy/comparison/calibrations"
            )
        return receipt, result

    def load_gate_result(self, artifact_id: str) -> Any:
        """Hydrate a result without claiming comparison/calibration replay."""

        _receipt, result = self._load_gate_result_with_receipt(artifact_id)
        return result

    def load_verified_gate_result(
        self,
        artifact_id: str,
        *,
        policy: Any,
        comparison: Any,
        calibrations: Mapping[Any, Any],
    ) -> Any:
        """Re-evaluate and byte-compare a stored result from all trusted inputs."""

        from .comparison import RunComparisonV1
        from .gates import GatePolicyV1, evaluate_gate

        if type(policy) is not GatePolicyV1:
            raise TypeError("policy must be a GatePolicyV1")
        if type(comparison) is not RunComparisonV1:
            raise TypeError("comparison must be a RunComparisonV1")
        receipt, stored = self._load_gate_result_with_receipt(artifact_id)
        try:
            replayed = evaluate_gate(policy, comparison, calibrations)
            expected_sources = _canonical_source_bindings(
                (
                    comparison.baseline_binding,
                    comparison.candidate_binding,
                ),
                "gate result receipt",
            )
            expected_algorithm = self._gate_result_publication_digest(
                policy,
                replayed,
            )
        except ArtifactIntegrityError:
            raise
        except (SchemaError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError("gate result source replay failed") from exc
        if (
            receipt.source_bindings != expected_sources
            or receipt.algorithm_digest != expected_algorithm
        ):
            raise ArtifactIntegrityError(
                "gate result receipt differs from trusted source bindings"
            )
        if canonical_json_bytes(replayed.to_dict()) != canonical_json_bytes(
            stored.to_dict()
        ):
            raise ArtifactIntegrityError(
                "stored gate result differs from exact source replay"
            )
        return stored


def _validate_review_judge_cross_binding(
    review_result: Any,
    judge_output: Any,
    *,
    evaluator_execution_digest: str,
) -> None:
    """Cross-bind terminal Review receipts to persisted Judge executions."""

    from .judge import JudgeOutputArtifact, JudgeRunStatus, JudgeTask
    from .review_evaluator import (
        ReviewEvaluationResult,
        ReviewEvaluationStatus,
        ReviewJudgeDecisionReceipt,
        ReviewJudgeFailureReceipt,
        ReviewJudgeRequestRecord,
        ReviewJudgeUngradedReceipt,
    )

    if type(judge_output) is not JudgeOutputArtifact:
        raise ArtifactIntegrityError(
            "Review Judge cross-binding requires a concrete JudgeOutputArtifact"
        )
    review_results = tuple(
        item
        for item in judge_output.results
        if item.request.task is not JudgeTask.INTENT_EQUIVALENCE
    )
    result_by_request: Dict[str, Any] = {}
    blind_request_ids = set()
    result_digests = set()
    for result in review_results:
        request_id = result.request.source_request_id
        result_digest = result.digest()
        if (
            request_id in result_by_request
            or result.request.request_id in blind_request_ids
            or result_digest in result_digests
        ):
            raise ArtifactIntegrityError(
                "Review JudgeOutput contains duplicate request or result identities"
            )
        if result.evaluator_execution_digest != evaluator_execution_digest:
            raise ArtifactIntegrityError(
                "Review Judge result belongs to another evaluator execution"
            )
        result_by_request[request_id] = result
        blind_request_ids.add(result.request.request_id)
        result_digests.add(result_digest)

    if review_result is None:
        if result_by_request:
            raise ArtifactIntegrityError(
                "Review JudgeOutput has results without a ReviewEvaluationResult"
            )
        return
    if type(review_result) is not ReviewEvaluationResult:
        raise ArtifactIntegrityError(
            "Review Judge cross-binding requires ReviewEvaluationResult"
        )
    if review_result.status is ReviewEvaluationStatus.PENDING_JUDGE:
        raise ArtifactIntegrityError(
            "terminal Trial contains a pending ReviewEvaluationResult"
        )
    requests = tuple(review_result.judge_requests)
    if any(type(item) is not ReviewJudgeRequestRecord for item in requests):
        raise ArtifactIntegrityError(
            "Review evaluation contains a non-concrete Judge request"
        )
    request_by_id = {item.request_id: item for item in requests}
    if len(request_by_id) != len(requests):
        raise ArtifactIntegrityError(
            "Review evaluation contains duplicate Judge request IDs"
        )
    if set(result_by_request) != set(request_by_id):
        raise ArtifactIntegrityError(
            "Review JudgeOutput does not exactly cover terminal Review requests"
        )

    derived_decisions = []
    derived_failures = []
    derived_ungraded = []
    for request_id in sorted(request_by_id):
        record = request_by_id[request_id]
        result = result_by_request[request_id]
        if (
            result.request != record.request
            or result.request.source_request_id != record.request_id
            or result.request.source_request_digest
            != record.request.source_request_digest
            or result.request.digest() != record.request_digest
            or result.request.request_id != record.blind_request_id
            or result.request.task is not record.task
            or result.evaluator_execution_digest
            != review_result.evaluator_execution_digest
        ):
            raise ArtifactIntegrityError(
                "Review Judge result differs from its canonical request binding"
            )
        common = {
            "request_id": record.request_id,
            "task": record.task,
            "request_digest": record.request_digest,
            "evaluator_execution_digest": result.evaluator_execution_digest,
            "judge_result_digest": result.digest(),
            "blind_request_id": record.blind_request_id,
        }
        try:
            if result.status is JudgeRunStatus.GRADED:
                if result.decision is None:
                    raise ValueError("graded Judge result lacks decision")
                derived_decisions.append(
                    ReviewJudgeDecisionReceipt(
                        decision=result.decision,
                        **common,
                    )
                )
            elif result.status is JudgeRunStatus.JUDGE_FAILED:
                if result.failure is None:
                    raise ValueError("failed Judge result lacks failure")
                derived_failures.append(
                    ReviewJudgeFailureReceipt(
                        failure=result.failure,
                        **common,
                    )
                )
            elif result.status is JudgeRunStatus.UNGRADED:
                if result.ungraded_reason is None:
                    raise ValueError("ungraded Judge result lacks reason")
                derived_ungraded.append(
                    ReviewJudgeUngradedReceipt(
                        ungraded_reason=result.ungraded_reason,
                        **common,
                    )
                )
            else:
                raise ValueError("Judge result has unsupported terminal status")
        except (SchemaError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "Review Judge result cannot form a canonical receipt"
            ) from exc

    expected_groups = (
        (
            tuple(sorted(derived_decisions, key=lambda item: item.request_id)),
            tuple(review_result.judge_decisions),
            ReviewJudgeDecisionReceipt,
        ),
        (
            tuple(sorted(derived_failures, key=lambda item: item.request_id)),
            tuple(review_result.judge_failures),
            ReviewJudgeFailureReceipt,
        ),
        (
            tuple(sorted(derived_ungraded, key=lambda item: item.request_id)),
            tuple(review_result.judge_ungraded),
            ReviewJudgeUngradedReceipt,
        ),
    )
    actual_receipt_ids = []
    for derived, actual, receipt_type in expected_groups:
        if (
            any(type(item) is not receipt_type for item in actual)
            or actual != tuple(sorted(actual, key=lambda item: item.request_id))
            or actual != derived
        ):
            raise ArtifactIntegrityError(
                "Review Judge receipts differ from JudgeOutput terminal results"
            )
        actual_receipt_ids.extend(item.request_id for item in actual)
    if (
        len(actual_receipt_ids) != len(set(actual_receipt_ids))
        or set(actual_receipt_ids) != set(request_by_id)
    ):
        raise ArtifactIntegrityError(
            "Review Judge requests do not have exactly one terminal receipt"
        )


def bind_analysis_source(
    bundle: Any,
    *,
    run_config: EvalRunConfig,
    case_snapshot: RunCaseSnapshot,
) -> AnalysisSourceBinding:
    """Verify one hydrated orchestrator Run Evaluation and seal its roots."""

    # Local imports avoid making protocol-only module import pull in the
    # orchestrator/report graph while still requiring concrete hydrated types.
    from .intent_evaluator import IntentEvaluationResult, IntentJudgeRelation
    from .judge import JudgeInputArtifact, JudgeOutputArtifact
    from .metrics import (
        IntentScoreBinding,
        ReviewScoreBinding,
        TrialScore,
        TrialScorer,
    )
    from .orchestrator import RunEvaluationBundle, TrialEvaluationBundle
    from .report import (
        RunReportSummary,
        TrialInspection,
        render_run_markdown,
        render_trial_markdown,
    )
    from .review_evaluator import ReviewEvaluationResult

    if type(bundle) is not RunEvaluationBundle:
        raise TypeError("bundle must be a concrete hydrated RunEvaluationBundle")
    if type(run_config) is not EvalRunConfig:
        raise TypeError("run_config must be an EvalRunConfig")
    if type(case_snapshot) is not RunCaseSnapshot:
        raise TypeError("case_snapshot must be a RunCaseSnapshot")
    try:
        run_config_digest = run_config.digest()
        case_snapshot_digest = case_snapshot.digest()
    except (AttributeError, SchemaError, TypeError, ValueError) as exc:
        raise ArtifactIntegrityError(
            "RunConfig or Case Snapshot digest is invalid"
        ) from exc
    if bundle.run_id != run_config.run_id:
        raise ArtifactIntegrityError("hydrated Evaluation belongs to another RunConfig")
    try:
        if type(bundle.evaluator_execution) is not EvaluatorExecutionConfig:
            raise TypeError("evaluator execution is not concrete")
        validate_evaluation_id_shape(bundle.evaluation_id)
        execution_digest = bundle.evaluator_execution.digest()
        expected_evaluation_id = derive_evaluation_id(
            run_config.run_id,
            execution_digest,
            bundle.evaluation_revision,
        )
    except (AttributeError, SchemaError, TypeError, ValueError) as exc:
        raise ArtifactIntegrityError(
            "hydrated Evaluation identity is invalid"
        ) from exc
    if bundle.evaluation_id != expected_evaluation_id:
        raise ArtifactIntegrityError("hydrated Evaluation ID differs from its sources")
    if (
        case_snapshot.snapshot_id != run_config.suite.case_snapshot_id
        or case_snapshot_digest != run_config.suite.case_snapshot_digest
    ):
        raise ArtifactIntegrityError("Case Snapshot differs from RunConfig sources")

    summary = bundle.summary
    try:
        if type(summary) is not RunReportSummary:
            raise TypeError("summary is not a sealed RunReportSummary")
        summary.__post_init__()
        summary_bindings = summary.source_bindings
        summary_id = summary.summary_id
        summary_digest = summary.digest()
        summary_cases = summary.cases
        summary_identity = summary.identity
    except (AttributeError, SchemaError, TypeError, ValueError) as exc:
        raise ArtifactIntegrityError(
            "hydrated Evaluation summary is not a sealed RunReportSummary"
        ) from exc
    try:
        rendered_report = render_run_markdown(summary)
    except (SchemaError, TypeError, ValueError) as exc:
        raise ArtifactIntegrityError(
            "hydrated Evaluation summary cannot be canonically rendered"
        ) from exc
    if type(bundle.report) is not str or bundle.report != rendered_report:
        raise ArtifactIntegrityError(
            "hydrated Evaluation report differs from canonical summary rendering"
        )
    if (
        type(summary_bindings) is not dict
        or set(summary_bindings) != _GLOBAL_REPORT_SOURCE_BINDING_FIELDS
    ):
        raise ArtifactIntegrityError(
            "hydrated Evaluation summary source bindings have an invalid exact schema"
        )
    try:
        summary_manifest_digest = _digest(
            summary_bindings["run_manifest_digest"],
            "summary RunManifest digest",
        )
    except SchemaError as exc:
        raise ArtifactIntegrityError(
            "hydrated Evaluation summary RunManifest digest is invalid"
        ) from exc
    expected_summary_bindings = {
        "run_id": run_config.run_id,
        "run_config_digest": run_config_digest,
        "case_snapshot_id": case_snapshot.snapshot_id,
        "case_snapshot_digest": case_snapshot_digest,
        "evaluation_id": bundle.evaluation_id,
        "evaluation_revision": bundle.evaluation_revision,
        "evaluator_execution_digest": execution_digest,
    }
    if any(
        summary_bindings.get(name) != value
        for name, value in expected_summary_bindings.items()
    ):
        raise ArtifactIntegrityError(
            "hydrated Evaluation summary differs from verified source digests"
        )
    expected_summary_suite = {
        "suite_id": run_config.suite.suite_id,
        "suite_version": run_config.suite.suite_version,
        "manifest_digest": run_config.suite.manifest_digest,
        "case_snapshot_id": case_snapshot.snapshot_id,
        "case_snapshot_digest": case_snapshot_digest,
    }
    evaluator_identity = (
        None
        if type(summary_identity) is not dict
        else summary_identity.get("evaluator")
    )
    if (
        type(summary_identity) is not dict
        or summary_identity.get("suite") != expected_summary_suite
        or summary_identity.get("agent") != run_config.agent.to_dict()
        or type(evaluator_identity) is not dict
        or evaluator_identity.get("configuration")
        != bundle.evaluator_execution.evaluator.to_dict()
        or evaluator_identity.get("evaluator_config_digest")
        != bundle.evaluator_execution.evaluator_config_digest
        or evaluator_identity.get("execution_config_digest") != execution_digest
        or evaluator_identity.get("evaluation_id") != bundle.evaluation_id
        or evaluator_identity.get("evaluation_revision")
        != bundle.evaluation_revision
    ):
        raise ArtifactIntegrityError(
            "hydrated Evaluation summary identity differs from available sources"
        )

    if type(bundle.trials) is not tuple:
        raise ArtifactIntegrityError(
            "hydrated Evaluation Trial collection is not canonical"
        )
    trials = bundle.trials
    planned_count = len(run_config.suite.cases) * run_config.trial_count
    if len(trials) != planned_count:
        raise ArtifactIntegrityError(
            "hydrated Evaluation Trial coverage differs from RunConfig"
        )
    keyed_scores = []
    score_metrics_policies = []
    for trial in trials:
        if type(trial) is not TrialEvaluationBundle:
            raise ArtifactIntegrityError(
                "hydrated Evaluation contains a non-concrete TrialEvaluationBundle"
            )
        if type(trial.trial_score) is not TrialScore:
            raise ArtifactIntegrityError(
                "hydrated Evaluation Trial score is not a concrete TrialScore"
            )
        if type(trial.judge_input) is not JudgeInputArtifact:
            raise ArtifactIntegrityError(
                "hydrated Trial Judge input is not a concrete JudgeInputArtifact"
            )
        if type(trial.judge_output) is not JudgeOutputArtifact:
            raise ArtifactIntegrityError(
                "hydrated Trial Judge output is not a concrete JudgeOutputArtifact"
            )
        if type(trial.inspection) is not TrialInspection:
            raise ArtifactIntegrityError(
                "hydrated Trial inspection is not a concrete TrialInspection"
            )
        try:
            task_id = trial.task_id
            trial_index = trial.trial_index
            trial_id = trial.trial_id
            suite_case = run_config.suite.case(task_id)
            snapshot_entry = case_snapshot.case(task_id)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "hydrated Evaluation contains an invalid Trial source"
            ) from exc
        try:
            expected_trial_id = run_config.trial_id(task_id, trial_index)
        except (SchemaError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "hydrated Evaluation contains an invalid Trial slot"
            ) from exc
        if (
            trial.run_id != run_config.run_id
            or trial.evaluation_id != bundle.evaluation_id
            or trial.evaluation_revision != bundle.evaluation_revision
            or type(trial.evaluator_execution) is not EvaluatorExecutionConfig
            or trial.evaluator_execution.digest() != execution_digest
            or trial_id != expected_trial_id
        ):
            raise ArtifactIntegrityError(
                "hydrated Evaluation Trial differs from its immutable Run slot"
            )
        eval_case = trial.eval_case
        if type(eval_case) is not EvalCase:
            raise ArtifactIntegrityError(
                "hydrated Evaluation Trial does not contain a concrete EvalCase"
            )
        case_digest = eval_case.digest()
        input_digest = eval_case.eval_input().digest()
        if (
            eval_case.task_id != task_id
            or eval_case.case_version != suite_case.case_version
            or eval_case.case_version != snapshot_entry.manifest_case.case_version
            or case_digest != suite_case.canonical_case_digest
            or case_digest != snapshot_entry.canonical_case_digest
            or input_digest != suite_case.eval_input_digest
            or input_digest != snapshot_entry.input.digest()
        ):
            raise ArtifactIntegrityError(
                "hydrated Evaluation Case differs from RunConfig/Case Snapshot"
            )
        score = trial.trial_score
        compatibility = score.compatibility
        try:
            source_scorer = TrialScorer(
                compatibility.metrics_policy,
                intent_evaluator_revision=(
                    compatibility.intent_evaluator_revision
                ),
                review_evaluator_revision=(
                    compatibility.review_evaluator_revision
                ),
                intent_policy_version=compatibility.intent_policy_version,
                intent_normalization_version=(
                    compatibility.intent_normalization_version
                ),
                review_policy_version=compatibility.review_policy_version,
                assignment_policy_version=(
                    compatibility.assignment_policy_version
                ),
                location_policy_version=compatibility.location_policy_version,
                evidence_policy_version=compatibility.evidence_policy_version,
            )
        except (AttributeError, SchemaError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "hydrated TrialScore has an invalid scorer configuration"
            ) from exc
        if (
            score.task_id != task_id
            or score.case_version != eval_case.case_version
            or score.trial_index != trial_index
            or score.trial_id != trial_id
            or score.canonical_case_digest != case_digest
            or score.eval_input_digest != input_digest
            or score.evaluation_revision != bundle.evaluation_revision
            or compatibility.run_id != run_config.run_id
            or compatibility.evaluation_id != bundle.evaluation_id
            or compatibility.case_snapshot_id != case_snapshot.snapshot_id
            or compatibility.case_snapshot_digest != case_snapshot_digest
            or compatibility.trial_count != run_config.trial_count
            or compatibility.evaluator_execution_digest != execution_digest
        ):
            raise ArtifactIntegrityError(
                "hydrated TrialScore differs from Trial/Run/Case bindings"
            )
        submission = trial.submission
        if type(submission) is not EvalSubmission:
            raise ArtifactIntegrityError(
                "hydrated Trial source does not contain a concrete Submission"
            )
        submission_digest = submission.digest()
        expected_failure_code = (
            None if submission.failure is None else submission.failure.code
        )
        expected_failure_retryable = (
            None if submission.failure is None else submission.failure.retryable
        )
        if (
            submission.task_id != task_id
            or submission.trial_id != trial_id
            or submission.agent_id != run_config.agent.agent_id
            or submission.eval_input_digest != input_digest
            or submission_digest != score.submission_digest
            or submission.status is not score.submission_status
            or expected_failure_code is not score.failure_code
            or expected_failure_retryable != score.failure_retryable
            or submission.usage != score.usage
            or submission.trace_ref != score.trace_ref
        ):
            raise ArtifactIntegrityError(
                "hydrated Submission differs from its TrialScore source binding"
            )

        intent_result = trial.intent_result
        intent_binding = score.intent_binding
        if intent_result is None:
            if intent_binding is not None:
                raise ArtifactIntegrityError(
                    "hydrated Intent result is missing for its TrialScore binding"
                )
        else:
            if type(intent_result) is not IntentEvaluationResult:
                raise ArtifactIntegrityError(
                    "hydrated Intent result is not a concrete IntentEvaluationResult"
                )
            if submission.intent is None:
                raise ArtifactIntegrityError(
                    "hydrated Intent result has no Submission Intent source"
                )
            expected_submission_intent_digest = canonical_json_bytes(
                submission.intent.to_dict()
            )
            expected_submission_intent_digest = hashlib.sha256(
                expected_submission_intent_digest
            ).hexdigest()
            try:
                expected_intent_binding = IntentScoreBinding(
                    result_digest=intent_result.digest(),
                    evaluator_revision=intent_result.evaluator_revision,
                    policy_version=intent_result.policy_version,
                    normalization_version=intent_result.normalization_version,
                    status=intent_result.status,
                    judge_request_count=len(intent_result.judge_requests),
                    judge_graded_count=len(intent_result.judge_decisions),
                    judge_failed_count=len(intent_result.judge_failures),
                    judge_ungraded_count=len(intent_result.judge_ungraded),
                    semantic_unknown_count=sum(
                        item.relation is IntentJudgeRelation.UNKNOWN
                        for item in intent_result.judge_decisions
                    ),
                )
            except (SchemaError, TypeError, ValueError) as exc:
                raise ArtifactIntegrityError(
                    "hydrated Intent result cannot form a terminal score binding"
                ) from exc
            if (
                intent_result.submission_intent_digest
                != expected_submission_intent_digest
                or intent_result.intent_truth_digest
                != eval_case.intent_truth.digest()
                or intent_result.clarification_script_digest
                != eval_case.clarification_script.digest()
                or intent_result.evaluator_revision
                != compatibility.intent_evaluator_revision
                or intent_result.policy_version
                != compatibility.intent_policy_version
                or intent_result.normalization_version
                != compatibility.intent_normalization_version
                or any(
                    item.evaluator_execution_digest != execution_digest
                    for item in (
                        *intent_result.judge_failures,
                        *intent_result.judge_ungraded,
                    )
                )
                or intent_binding != expected_intent_binding
            ):
                raise ArtifactIntegrityError(
                    "hydrated Intent result differs from Submission/TrialScore bindings"
                )

        review_result = trial.review_result
        review_binding = score.review_binding
        if review_result is None:
            if review_binding is not None:
                raise ArtifactIntegrityError(
                    "hydrated Review result is missing for its TrialScore binding"
                )
        else:
            if type(review_result) is not ReviewEvaluationResult:
                raise ArtifactIntegrityError(
                    "hydrated Review result is not a concrete ReviewEvaluationResult"
                )
            if submission.review is None:
                raise ArtifactIntegrityError(
                    "hydrated Review result has no Submission Review source"
                )
            submission_review_digest = hashlib.sha256(
                canonical_json_bytes(submission.review.to_dict())
            ).hexdigest()
            submission_evidence_digest = hashlib.sha256(
                canonical_json_bytes(
                    [item.to_dict() for item in submission.evidence]
                )
            ).hexdigest()
            coverage = review_result.coverage
            try:
                expected_review_binding = ReviewScoreBinding(
                    result_digest=review_result.digest(),
                    evaluator_revision=review_result.evaluator_revision,
                    review_policy_version=review_result.review_policy_version,
                    assignment_policy_version=(
                        review_result.assignment_policy_version
                    ),
                    location_policy_version=review_result.location_policy_version,
                    evidence_policy_version=(
                        review_result.evidence_integrity_policy_version
                    ),
                    status=review_result.status,
                    phase=review_result.phase,
                    judge_request_count=coverage.judge_request_count,
                    judge_graded_count=coverage.judge_graded_count,
                    judge_failed_count=coverage.judge_failed_count,
                    judge_ungraded_count=coverage.judge_ungraded_count,
                    judge_pending_count=coverage.judge_pending_count,
                    semantic_unknown_count=coverage.semantic_unknown_count,
                    finding_count=coverage.finding_count,
                    finding_resolved_count=coverage.finding_resolved_count,
                )
            except (SchemaError, TypeError, ValueError) as exc:
                raise ArtifactIntegrityError(
                    "hydrated Review result cannot form a terminal score binding"
                ) from exc
            if (
                review_result.submission_digest != submission_digest
                or review_result.submission_review_digest
                != submission_review_digest
                or review_result.submission_evidence_digest
                != submission_evidence_digest
                or review_result.eval_input_digest != input_digest
                or review_result.review_truth_digest
                != eval_case.review_truth.digest()
                or review_result.evaluator_execution_digest != execution_digest
                or review_result.evaluator_revision
                != compatibility.review_evaluator_revision
                or review_result.review_policy_version
                != compatibility.review_policy_version
                or review_result.assignment_policy_version
                != compatibility.assignment_policy_version
                or review_result.location_policy_version
                != compatibility.location_policy_version
                or review_result.evidence_integrity_policy_version
                != compatibility.evidence_policy_version
                or review_binding != expected_review_binding
            ):
                raise ArtifactIntegrityError(
                    "hydrated Review result differs from Submission/TrialScore bindings"
                )
        try:
            replayed_score = TrialScore.from_dict(
                score.to_dict(),
                scorer=source_scorer,
                run_config=run_config,
                evaluator_execution=bundle.evaluator_execution,
                evaluation_revision=bundle.evaluation_revision,
                eval_case=eval_case,
                submission=submission,
                trial_index=trial_index,
                intent_result=intent_result,
                review_result=review_result,
            )
            if canonical_json_bytes(replayed_score.to_dict()) != canonical_json_bytes(
                score.to_dict()
            ):
                raise ValueError("TrialScore canonical replay differs")
        except (SchemaError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "hydrated TrialScore differs from complete source-bound replay"
            ) from exc
        score = replayed_score
        compatibility = replayed_score.compatibility
        score_digest = replayed_score.digest()
        try:
            replayed_judge_input = JudgeInputArtifact.from_dict(
                trial.judge_input.to_dict(),
                evaluator_execution=bundle.evaluator_execution,
            )
            bound_intent_evaluation = (
                intent_result
                if trial.judge_output.intent_evaluation_digest is not None
                else None
            )
            replayed_judge_output = JudgeOutputArtifact.from_dict(
                trial.judge_output.to_dict(),
                input_artifact=replayed_judge_input,
                evaluator_execution=bundle.evaluator_execution,
                intent_evaluation=bound_intent_evaluation,
            )
            if bound_intent_evaluation is not None:
                replayed_judge_output.validate_intent_evaluation(
                    bound_intent_evaluation
                )
        except (SchemaError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "hydrated Trial Judge artifacts fail source-bound hydration"
            ) from exc
        if (
            replayed_judge_input.to_dict() != trial.judge_input.to_dict()
            or replayed_judge_output.to_dict() != trial.judge_output.to_dict()
        ):
            raise ArtifactIntegrityError(
                "hydrated Trial Judge artifacts differ from canonical hydration"
            )
        _validate_review_judge_cross_binding(
            review_result,
            replayed_judge_output,
            evaluator_execution_digest=execution_digest,
        )

        inspection = trial.inspection
        try:
            inspection.__post_init__()
            inspection_bindings = inspection.source_bindings
        except (AttributeError, SchemaError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "hydrated Trial inspection is not canonically sealed"
            ) from exc
        expected_inspection_bindings = {
            "run_id": run_config.run_id,
            "run_config_digest": run_config_digest,
            "run_manifest_digest": summary_manifest_digest,
            "case_snapshot_id": case_snapshot.snapshot_id,
            "case_snapshot_digest": case_snapshot_digest,
            "evaluation_id": bundle.evaluation_id,
            "evaluation_revision": bundle.evaluation_revision,
            "evaluator_execution_digest": execution_digest,
            "task_id": task_id,
            "trial_id": trial_id,
            "trial_index": trial_index,
            "canonical_case_digest": case_digest,
            "eval_input_digest": input_digest,
            "submission_digest": submission_digest,
            "intent_result_digest": (
                None if intent_result is None else intent_result.digest()
            ),
            "review_result_digest": (
                None if review_result is None else review_result.digest()
            ),
            "trial_score_digest": score_digest,
            "metrics_policy": compatibility.metrics_policy.to_dict(),
        }
        if (
            type(inspection_bindings) is not dict
            or set(inspection_bindings) != _TRIAL_REPORT_SOURCE_BINDING_FIELDS
            or inspection_bindings != expected_inspection_bindings
        ):
            raise ArtifactIntegrityError(
                "hydrated Trial inspection source bindings differ from Trial sources"
            )
        try:
            rendered_trial_report = render_trial_markdown(inspection)
        except (SchemaError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "hydrated Trial inspection cannot be canonically rendered"
            ) from exc
        if type(trial.report) is not str or trial.report != rendered_trial_report:
            raise ArtifactIntegrityError(
                "hydrated Trial report differs from canonical inspection rendering"
            )
        score_metrics_policies.append(compatibility.metrics_policy.to_dict())
        keyed_scores.append(((task_id, trial_index), _digest(score_digest, "Trial score")))
    if len({key for key, _digest_value in keyed_scores}) != len(keyed_scores):
        raise ArtifactIntegrityError("hydrated Evaluation contains duplicate Trial slots")
    canonical_metrics_policies = {
        canonical_json_bytes(item) for item in score_metrics_policies
    }
    if (
        len(canonical_metrics_policies) != 1
        or summary_bindings.get("metrics_policy") != score_metrics_policies[0]
    ):
        raise ArtifactIntegrityError(
            "hydrated Evaluation summary MetricsPolicy differs from Trial sources"
        )
    # RunReportSummary.from_dict additionally requires the immutable
    # RunManifest/TrialEvaluationSource graph, which this confirmed Task 1
    # binder signature intentionally does not accept.  The concrete bundle is
    # therefore trusted only for that already-completed orchestrator replay;
    # every source available here is checked above.

    projected = []
    if type(summary_cases) is not list:
        raise ArtifactIntegrityError("hydrated Evaluation summary Cases are invalid")
    for case in summary_cases:
        if type(case) is not dict or type(case.get("trials")) is not list:
            raise ArtifactIntegrityError("hydrated Evaluation Trial projection is invalid")
        try:
            summary_entry = case_snapshot.case(case.get("task_id"))
        except (SchemaError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "hydrated Evaluation summary contains an unknown Case"
            ) from exc
        if (
            case.get("case_version") != summary_entry.manifest_case.case_version
            or case.get("canonical_case_digest")
            != summary_entry.canonical_case_digest
            or case.get("eval_input_digest") != summary_entry.input.digest()
        ):
            raise ArtifactIntegrityError(
                "hydrated Evaluation summary Case differs from Case Snapshot"
            )
        for trial in case["trials"]:
            if type(trial) is not dict or type(trial.get("score_ref")) is not dict:
                raise ArtifactIntegrityError(
                    "hydrated Evaluation summary is missing a Trial score binding"
                )
            score_ref = trial["score_ref"]
            if (
                trial.get("task_id") != case.get("task_id")
                or score_ref.get("task_id") != trial.get("task_id")
                or score_ref.get("trial_id") != trial.get("trial_id")
            ):
                raise ArtifactIntegrityError(
                    "hydrated Evaluation summary Trial ref is misbound"
                )
            projected.append(
                (
                    (trial.get("task_id"), trial.get("trial_index")),
                    _digest(score_ref.get("score_digest"), "summary Trial score"),
                )
            )
    if sorted(projected) != sorted(keyed_scores):
        raise ArtifactIntegrityError(
            "hydrated Trial score digests differ from the Run summary"
        )

    try:
        return AnalysisSourceBinding(
            run_id=run_config.run_id,
            evaluation_id=bundle.evaluation_id,
            summary_id=summary_id,
            summary_digest=summary_digest,
            run_config_digest=run_config_digest,
            case_snapshot_digest=case_snapshot_digest,
            trial_score_digests=tuple(sorted(value for _key, value in keyed_scores)),
        )
    except (SchemaError, TypeError, ValueError) as exc:
        raise ArtifactIntegrityError(
            "hydrated Evaluation source binding is invalid"
        ) from exc


__all__ = [
    "ANALYSIS_ARTIFACT_KINDS",
    "ANALYSIS_RECEIPT_SCHEMA_VERSION",
    "MAX_ANALYSIS_ARTIFACTS",
    "AnalysisSourceBinding",
    "AnalysisArtifactRef",
    "AnalysisReceipt",
    "AnalysisArtifactStore",
    "derive_analysis_artifact_id",
    "bind_analysis_source",
]
