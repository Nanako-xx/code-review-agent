"""Immutable, source-bound artifacts for the Eval v2 analysis layer.

This module deliberately depends only on Eval protocol/storage modules.  It
does not import or construct the product Runtime, an Agent adapter, a Judge,
or any acquisition service.
"""

from __future__ import annotations

import hashlib
import os
import stat
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
    _hardlinked_file,
    _strict_json_loads,
    _unsafe_node,
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
    SchemaError,
    _JsonModel,
    canonical_json_bytes,
    stable_id,
)


ANALYSIS_RECEIPT_SCHEMA_VERSION = "analysis_receipt_v1"
MAX_ANALYSIS_ARTIFACTS = 64
_RECEIPT_NAME = "receipt.json"
_HEX_DIGITS = frozenset("0123456789abcdef")


def _digest(value: Any, context: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise SchemaError("%s must be a lowercase SHA-256 digest" % context)
    return value


def _kind(value: Any) -> str:
    return validate_path_segment(value, "analysis artifact kind")


def _artifact_id(value: Any) -> str:
    return validate_path_segment(value, "analysis artifact_id")


def _json_artifact_name(value: Any) -> str:
    name = validate_path_segment(value, "analysis JSON artifact name")
    if name == _RECEIPT_NAME or not name.endswith(".json"):
        raise SchemaError(
            "analysis JSON artifact name must end in .json and not be receipt.json"
        )
    return name


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
    for raw_name, value in files.items():
        name = _json_artifact_name(raw_name)
        if name in result:
            raise SchemaError("analysis bundle contains duplicate artifact names")
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
        try:
            entries = list(os.scandir(directory))
        except OSError as exc:
            raise ArtifactSecurityError(
                "could not inspect analysis artifact namespace"
            ) from exc
        names = set()
        for entry in entries:
            try:
                # Windows DirEntry.stat may report st_nlink=0 even though a
                # direct lstat of the same file reports the authoritative
                # link count.  Use the path-based no-follow metadata here.
                metadata = os.lstat(entry.path)
            except OSError as exc:
                raise ArtifactSecurityError(
                    "could not inspect analysis artifact"
                ) from exc
            if _unsafe_node(metadata) or not stat.S_ISREG(metadata.st_mode):
                raise ArtifactSecurityError(
                    "analysis namespace contains a symlink, reparse point, or unsafe entry"
                )
            if _hardlinked_file(metadata):
                raise ArtifactSecurityError(
                    "analysis artifact has an unsafe hardlink count"
                )
            names.add(entry.name)
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
        receipt_data = canonical_json_bytes(receipt.to_dict())
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


def bind_analysis_source(
    bundle: Any,
    *,
    run_config: EvalRunConfig,
    case_snapshot: RunCaseSnapshot,
) -> AnalysisSourceBinding:
    """Verify one hydrated orchestrator Run Evaluation and seal its roots."""

    # Local imports avoid making protocol-only module import pull in the
    # orchestrator/report graph while still requiring concrete hydrated types.
    from .metrics import TrialScore
    from .orchestrator import RunEvaluationBundle, TrialEvaluationBundle
    from .report import RunReportSummary, render_run_markdown

    if type(bundle) is not RunEvaluationBundle:
        raise TypeError("bundle must be a concrete hydrated RunEvaluationBundle")
    if type(run_config) is not EvalRunConfig:
        raise TypeError("run_config must be an EvalRunConfig")
    if type(case_snapshot) is not RunCaseSnapshot:
        raise TypeError("case_snapshot must be a RunCaseSnapshot")
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
        or case_snapshot.digest() != run_config.suite.case_snapshot_digest
    ):
        raise ArtifactIntegrityError("Case Snapshot differs from RunConfig sources")

    summary = bundle.summary
    try:
        if type(summary) is not RunReportSummary:
            raise TypeError("summary is not a sealed RunReportSummary")
        summary_bindings = summary.source_bindings
        summary_id = summary.summary_id
        summary_digest = summary.digest()
        summary_cases = summary.cases
    except (AttributeError, TypeError, ValueError) as exc:
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
    expected_summary_bindings = {
        "run_id": run_config.run_id,
        "run_config_digest": run_config.digest(),
        "case_snapshot_id": case_snapshot.snapshot_id,
        "case_snapshot_digest": case_snapshot.digest(),
        "evaluation_id": bundle.evaluation_id,
        "evaluation_revision": bundle.evaluation_revision,
        "evaluator_execution_digest": execution_digest,
    }
    if type(summary_bindings) is not dict or any(
        summary_bindings.get(name) != value
        for name, value in expected_summary_bindings.items()
    ):
        raise ArtifactIntegrityError(
            "hydrated Evaluation summary differs from verified source digests"
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
    for trial in trials:
        if type(trial) is not TrialEvaluationBundle:
            raise ArtifactIntegrityError(
                "hydrated Evaluation contains a non-concrete TrialEvaluationBundle"
            )
        if type(trial.trial_score) is not TrialScore:
            raise ArtifactIntegrityError(
                "hydrated Evaluation Trial score is not a concrete TrialScore"
            )
        try:
            task_id = trial.task_id
            trial_index = trial.trial_index
            trial_id = trial.trial_id
            score_digest = trial.trial_score.digest()
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
            or compatibility.case_snapshot_digest != case_snapshot.digest()
            or compatibility.trial_count != run_config.trial_count
            or compatibility.evaluator_execution_digest != execution_digest
        ):
            raise ArtifactIntegrityError(
                "hydrated TrialScore differs from Trial/Run/Case bindings"
            )
        keyed_scores.append(((task_id, trial_index), _digest(score_digest, "Trial score")))
    if len({key for key, _digest_value in keyed_scores}) != len(keyed_scores):
        raise ArtifactIntegrityError("hydrated Evaluation contains duplicate Trial slots")

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
            run_config_digest=run_config.digest(),
            case_snapshot_digest=case_snapshot.digest(),
            trial_score_digests=tuple(sorted(value for _key, value in keyed_scores)),
        )
    except (SchemaError, TypeError, ValueError) as exc:
        raise ArtifactIntegrityError(
            "hydrated Evaluation source binding is invalid"
        ) from exc


__all__ = [
    "ANALYSIS_RECEIPT_SCHEMA_VERSION",
    "MAX_ANALYSIS_ARTIFACTS",
    "AnalysisSourceBinding",
    "AnalysisArtifactRef",
    "AnalysisReceipt",
    "AnalysisArtifactStore",
    "derive_analysis_artifact_id",
    "bind_analysis_source",
]
