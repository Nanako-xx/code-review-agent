from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import secrets
from typing import Any, Mapping
import unicodedata

from review_agent.revision import (
    CanonicalRepositoryIdentity,
    RepositoryIdentity,
    canonical_repository_identity,
)
from review_agent.safe_io import (
    SafeIOError,
    atomic_replace_bytes,
    canonical_json_bytes,
    canonical_relative_path,
    cleanup_staging_files,
    ensure_secure_directory,
    publish_create_only_bytes,
    read_strict_json,
    read_verified_bytes,
    resolve_managed_path,
    sha256_hex,
    strict_json_loads,
)


class PRWorkspaceError(ValueError):
    """A PRWorkspace operation could not be completed."""


class PRWorkspaceSecurityError(PRWorkspaceError):
    """A persisted binding, path, or content failed closed."""


WORKSPACE_SCHEMA = "pr_workspace_manifest_v1"
PR_METADATA_SCHEMA = "pr_metadata_v1"
SNAPSHOT_SCHEMA = "snapshot_manifest_v1"
SESSION_BINDING_SCHEMA = "session_binding_v1"
CONTEXT_MANIFEST_SCHEMA = "context_manifest_v1"
ARTIFACT_RECEIPT_SCHEMA = "snapshot_artifact_receipt_v1"

_PR_ID = re.compile(r"\APR-[0-9a-f]{64}\Z")
_SNAPSHOT_ID = re.compile(r"\AS-[0-9a-f]{64}\Z")
_SESSION_ID = re.compile(r"\ASESSION-[0-9a-f]{64}\Z")
_ARTIFACT_ID = re.compile(r"\AA-[0-9a-f]{64}\Z")
_GIT_OBJECT_ID = re.compile(r"\A(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_PROVIDER = re.compile(r"\A[a-z0-9][a-z0-9._-]{0,63}\Z")
_EXTERNAL_REVIEW_ID = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._:+#@-]{0,255}\Z")


def _optional_metadata_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise PRWorkspaceError(f"{field_name} must be non-empty text or null")
    return value


@dataclass(frozen=True)
class PRMetadata:
    title: str | None = None
    description: str | None = None
    base_ref: str | None = None
    head_ref: str | None = None
    author: str | None = None
    status: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "title",
            "description",
            "base_ref",
            "head_ref",
            "author",
            "status",
        ):
            _optional_metadata_text(getattr(self, name), name)


@dataclass(frozen=True)
class ResolvedPR:
    repository: CanonicalRepositoryIdentity
    provider: str
    external_review_id: str
    pr_id: str


@dataclass(frozen=True)
class PRWorkspace:
    store_root: Path
    path: Path
    resolved_pr: ResolvedPR

    @property
    def pr_id(self) -> str:
        return self.resolved_pr.pr_id


@dataclass(frozen=True)
class SnapshotWorkspace:
    workspace: PRWorkspace
    path: Path
    snapshot_id: str
    base_sha: str
    head_sha: str


@dataclass(frozen=True)
class SessionWorkspace:
    workspace: PRWorkspace
    snapshot: SnapshotWorkspace
    path: Path
    session_id: str


@dataclass(frozen=True)
class ArtifactDescriptor:
    artifact_id: str
    pr_id: str
    snapshot_id: str
    relative_path: str
    sha256: str
    size_bytes: int
    path: Path

    def to_receipt(self) -> dict[str, Any]:
        return {
            "schema_version": ARTIFACT_RECEIPT_SCHEMA,
            "artifact_id": self.artifact_id,
            "pr_id": self.pr_id,
            "snapshot_id": self.snapshot_id,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class ReviewResultArtifactBundle:
    aggregation_descriptor: ArtifactDescriptor
    review_result_descriptor: ArtifactDescriptor
    aggregation_bytes: bytes
    review_result_bytes: bytes


def _stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    return f"{prefix}-{hashlib.sha256(canonical_json_bytes(payload)).hexdigest()}"


def _validate_id(value: Any, pattern: re.Pattern[str], field_name: str) -> str:
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise PRWorkspaceSecurityError(f"{field_name} is not a canonical stable ID")
    return value


def _git_sha(value: Any, field_name: str) -> str:
    if type(value) is not str or _GIT_OBJECT_ID.fullmatch(value) is None:
        raise PRWorkspaceError(
            f"{field_name} must be a full lowercase SHA-1 or SHA-256 object ID"
        )
    return value


def _strict_object(
    payload: Any,
    fields: tuple[str, ...],
    context: str,
) -> dict[str, Any]:
    if type(payload) is not dict:
        raise PRWorkspaceSecurityError(f"{context} must be a JSON object")
    expected = set(fields)
    actual = set(payload)
    if actual != expected:
        raise PRWorkspaceSecurityError(f"{context} has an invalid exact schema")
    return dict(payload)


def _path_exists(path: Path) -> bool:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError as error:
        raise PRWorkspaceSecurityError("managed path is unavailable") from error
    return True


class PRWorkspaceStore:
    def __init__(self, root: Path) -> None:
        try:
            self.root = ensure_secure_directory(Path(root))
            self._pr_root = ensure_secure_directory(self.root / "pr")
            cleanup_staging_files(self.root)
            cleanup_staging_files(self._pr_root)
        except SafeIOError as error:
            raise PRWorkspaceSecurityError(str(error)) from error

    def resolve_pr(
        self,
        repository: RepositoryIdentity,
        provider: str,
        pr_number_or_external_review_id: str,
    ) -> ResolvedPR:
        try:
            canonical_repository = canonical_repository_identity(repository)
        except ValueError as error:
            raise PRWorkspaceError("repository identity is invalid") from error
        if type(provider) is not str:
            raise PRWorkspaceError("provider must be a canonical token")
        canonical_provider = provider.strip().casefold()
        if _PROVIDER.fullmatch(canonical_provider) is None:
            raise PRWorkspaceError("provider must be a canonical token")
        if type(pr_number_or_external_review_id) is not str:
            raise PRWorkspaceError("external review ID must be a canonical token")
        external_id = unicodedata.normalize(
            "NFKC", pr_number_or_external_review_id.strip()
        )
        if (
            external_id != pr_number_or_external_review_id.strip()
            or _EXTERNAL_REVIEW_ID.fullmatch(external_id) is None
        ):
            raise PRWorkspaceError("external review ID must be a canonical token")
        identity = {
            "repository_key": canonical_repository.repository_key,
            "provider": canonical_provider,
            "external_review_id": external_id,
        }
        return ResolvedPR(
            repository=canonical_repository,
            provider=canonical_provider,
            external_review_id=external_id,
            pr_id=_stable_id("PR", identity),
        )

    def create_or_load_workspace(
        self,
        resolved_pr: ResolvedPR,
        metadata: PRMetadata | None = None,
    ) -> PRWorkspace:
        self._validate_resolved_pr(resolved_pr)
        if metadata is None:
            metadata = PRMetadata()
        if not isinstance(metadata, PRMetadata):
            raise PRWorkspaceError("PR metadata is invalid")
        path = self._workspace_path(resolved_pr.pr_id)
        try:
            ensure_secure_directory(path)
            for relative in (
                "PR",
                "Intent",
                "Intent/history",
                "Snapshots",
                "Sessions",
            ):
                ensure_secure_directory(resolve_managed_path(path, relative))
            cleanup_staging_files(path)
        except SafeIOError as error:
            raise PRWorkspaceSecurityError(str(error)) from error

        workspace = PRWorkspace(
            store_root=self.root,
            path=path,
            resolved_pr=resolved_pr,
        )
        manifest_path = path / "manifest.json"
        initial_manifest = {
            "workspace_schema_version": WORKSPACE_SCHEMA,
            "pr_id": resolved_pr.pr_id,
            "current_snapshot_id": None,
            "current_intent_version": None,
        }
        if not _path_exists(manifest_path):
            self._publish_json_create_only(manifest_path, initial_manifest)
        self._validate_workspace_manifest(workspace)

        pr_path = path / "PR" / "pr.json"
        pr_payload = {
            "schema_version": PR_METADATA_SCHEMA,
            "pr_id": resolved_pr.pr_id,
            "repository_identity": resolved_pr.repository.to_dict(),
            "provider": resolved_pr.provider,
            "pr_number_or_external_review_id": resolved_pr.external_review_id,
            "title": metadata.title,
            "description": metadata.description,
            "base_ref": metadata.base_ref,
            "head_ref": metadata.head_ref,
            "author": metadata.author,
            "status": metadata.status,
        }
        if not _path_exists(pr_path):
            self._publish_json_create_only(pr_path, pr_payload)
        self._validate_pr_metadata(workspace)
        return workspace

    def create_or_load_snapshot(
        self,
        workspace: PRWorkspace,
        base_sha: str,
        head_sha: str,
    ) -> SnapshotWorkspace:
        self._assert_workspace_authority(workspace)
        base = _git_sha(base_sha, "base_sha")
        head = _git_sha(head_sha, "head_sha")
        snapshot_id = self._snapshot_id(workspace, base, head)
        path = workspace.path / "Snapshots" / self._physical_snapshot_id(snapshot_id)
        try:
            ensure_secure_directory(path)
            for relative in (
                ".artifacts",
                "DiffArtifact",
                "QualityGate",
                "ChangedSymbols",
                "Risk",
                "ReviewPlan",
                "ReviewPlan/Assignments",
                "ToolResults",
                "ToolResults/artifacts",
                "Results",
            ):
                ensure_secure_directory(resolve_managed_path(path, relative))
            cleanup_staging_files(path)
        except SafeIOError as error:
            raise PRWorkspaceSecurityError(str(error)) from error

        snapshot = SnapshotWorkspace(
            workspace=workspace,
            path=path,
            snapshot_id=snapshot_id,
            base_sha=base,
            head_sha=head,
        )
        snapshot_payload = {
            "schema_version": SNAPSHOT_SCHEMA,
            "repository_identity": workspace.resolved_pr.repository.to_dict(),
            "pr_id": workspace.pr_id,
            "snapshot_id": snapshot_id,
            "base_sha": base,
            "head_sha": head,
        }
        manifest_path = path / "snapshot.json"
        if not _path_exists(manifest_path):
            self._publish_json_create_only(manifest_path, snapshot_payload)
        self._validate_snapshot_manifest(snapshot)
        self._set_current_snapshot(workspace, snapshot_id)
        return snapshot

    def create_session(
        self,
        workspace: PRWorkspace,
        snapshot: SnapshotWorkspace,
        *,
        session_id: str | None = None,
    ) -> SessionWorkspace:
        self._assert_snapshot_authority(snapshot)
        if snapshot.workspace != workspace:
            raise PRWorkspaceSecurityError("Snapshot is not bound to this PR workspace")
        if session_id is None:
            session_id = "SESSION-" + secrets.token_hex(32)
        _validate_id(session_id, _SESSION_ID, "session_id")
        path = workspace.path / "Sessions" / ("u-" + session_id[8:40])
        if _path_exists(path):
            raise PRWorkspaceError("Session already exists")
        try:
            ensure_secure_directory(path)
        except SafeIOError as error:
            raise PRWorkspaceSecurityError(str(error)) from error

        state = {
            "schema_version": SESSION_BINDING_SCHEMA,
            "session_id": session_id,
            "pr_id": workspace.pr_id,
            "snapshot_id": snapshot.snapshot_id,
            "status": "created",
        }
        context = {
            "schema_version": CONTEXT_MANIFEST_SCHEMA,
            "session_id": session_id,
            "pr_id": workspace.pr_id,
            "snapshot_id": snapshot.snapshot_id,
            "last_api_request_at": None,
            "compaction_generation": 0,
            "compacted_through_turn": 0,
            "compaction_trigger": None,
            "compaction_summary_hash": None,
        }
        self._publish_json_create_only(path / "state.json", state)
        self._publish_bytes_create_only(path / "execution-log.jsonl", b"")
        self._publish_json_create_only(path / "context-manifest.json", context)
        return SessionWorkspace(
            workspace=workspace,
            snapshot=snapshot,
            path=path,
            session_id=session_id,
        )

    def publish_create_only(
        self,
        snapshot: SnapshotWorkspace,
        relative_path: str,
        content: bytes,
    ) -> ArtifactDescriptor:
        descriptor = self.describe_artifact(snapshot, relative_path, content)
        destination = descriptor.path
        digest = descriptor.sha256

        if _path_exists(destination):
            try:
                existing = read_verified_bytes(destination, digest)
            except SafeIOError as error:
                raise PRWorkspaceError(
                    "Artifact path already exists with different content"
                ) from error
            if existing != content:
                raise PRWorkspaceError(
                    "Artifact path already exists with different content"
                )
        else:
            self._publish_bytes_create_only(destination, content)

        receipt_path = self._artifact_receipt_path(snapshot, descriptor.artifact_id)
        if not _path_exists(receipt_path):
            self._publish_json_create_only(receipt_path, descriptor.to_receipt())
        stored = self._load_artifact_descriptor(snapshot, descriptor.artifact_id)
        if stored.to_receipt() != descriptor.to_receipt():
            raise PRWorkspaceSecurityError("Artifact physical ID collision detected")
        return descriptor

    def verify_snapshot(self, snapshot: SnapshotWorkspace) -> None:
        """Fail closed unless a Snapshot handle belongs to this Store."""

        self._assert_snapshot_authority(snapshot)

    def verify_workspace(self, workspace: PRWorkspace) -> None:
        self._assert_workspace_authority(workspace)

    def verify_session(self, session: SessionWorkspace) -> None:
        self._assert_session_authority(session)

    def publish_review_result_bundle(
        self,
        snapshot: SnapshotWorkspace,
        *,
        aggregation_bytes: bytes,
        review_result_bytes: bytes,
    ) -> ReviewResultArtifactBundle:
        """Create the internal record first, then the authoritative result."""

        self._assert_snapshot_authority(snapshot)
        if type(aggregation_bytes) is not bytes or type(review_result_bytes) is not bytes:
            raise PRWorkspaceError("Review Result bundle content must be bytes")
        aggregation = self.publish_create_only(
            snapshot,
            "Results/aggregation.json",
            aggregation_bytes,
        )
        review_result = self.publish_create_only(
            snapshot,
            "Results/review-result.json",
            review_result_bytes,
        )
        return ReviewResultArtifactBundle(
            aggregation_descriptor=aggregation,
            review_result_descriptor=review_result,
            aggregation_bytes=aggregation_bytes,
            review_result_bytes=review_result_bytes,
        )

    def load_review_result_bundle(
        self,
        snapshot: SnapshotWorkspace,
    ) -> ReviewResultArtifactBundle | None:
        """Load both create-only result files or fail closed on a partial bundle."""

        self._assert_snapshot_authority(snapshot)
        aggregation_path = snapshot.path / "Results" / "aggregation.json"
        review_result_path = snapshot.path / "Results" / "review-result.json"
        aggregation_exists = _path_exists(aggregation_path)
        review_result_exists = _path_exists(review_result_path)
        if not aggregation_exists and not review_result_exists:
            return None
        if aggregation_exists != review_result_exists:
            raise PRWorkspaceSecurityError(
                "Review Result bundle is only partially published"
            )
        aggregation = self.find_snapshot_artifact(
            snapshot, "Results/aggregation.json"
        )
        review_result = self.find_snapshot_artifact(
            snapshot, "Results/review-result.json"
        )
        return ReviewResultArtifactBundle(
            aggregation_descriptor=aggregation,
            review_result_descriptor=review_result,
            aggregation_bytes=self.read_verified_artifact(
                snapshot, aggregation.artifact_id
            ),
            review_result_bytes=self.read_verified_artifact(
                snapshot, review_result.artifact_id
            ),
        )

    def review_result_bundle_state(self, snapshot: SnapshotWorkspace) -> str:
        """Return absent, aggregation_only, or complete after authority checks."""

        self._assert_snapshot_authority(snapshot)
        aggregation_exists = _path_exists(
            snapshot.path / "Results" / "aggregation.json"
        )
        review_result_exists = _path_exists(
            snapshot.path / "Results" / "review-result.json"
        )
        if review_result_exists and not aggregation_exists:
            raise PRWorkspaceSecurityError(
                "Review Result exists without its Aggregation Record"
            )
        if not aggregation_exists:
            return "absent"
        if not review_result_exists:
            return "aggregation_only"
        aggregation = self.find_snapshot_artifact(
            snapshot, "Results/aggregation.json"
        )
        self.read_verified_artifact(snapshot, aggregation.artifact_id)
        return "complete"

    def publish_intent_history(
        self,
        workspace: PRWorkspace,
        filename: str,
        content: bytes,
    ) -> Path:
        self._assert_workspace_authority(workspace)
        try:
            canonical_relative_path(filename)
        except SafeIOError as error:
            raise PRWorkspaceSecurityError(str(error)) from error
        if "/" in filename:
            raise PRWorkspaceSecurityError("Intent history filename must be flat")
        path = workspace.path / "Intent" / "history" / filename
        if _path_exists(path):
            try:
                existing = path.read_bytes()
            except OSError as error:
                raise PRWorkspaceSecurityError(
                    "Intent history file is unavailable"
                ) from error
            if existing != content:
                raise PRWorkspaceSecurityError(
                    "Intent history is create-only and already exists"
                )
            return path
        self._publish_bytes_create_only(path, content)
        return path

    def replace_current_intent(
        self,
        workspace: PRWorkspace,
        content: bytes,
    ) -> Path:
        self._assert_workspace_authority(workspace)
        path = workspace.path / "Intent" / "current.json"
        try:
            atomic_replace_bytes(path, content)
        except SafeIOError as error:
            raise PRWorkspaceSecurityError(str(error)) from error
        return path

    def intent_current_exists(self, workspace: PRWorkspace) -> bool:
        self._assert_workspace_authority(workspace)
        return _path_exists(workspace.path / "Intent" / "current.json")

    def read_intent_json(
        self,
        workspace: PRWorkspace,
        relative_path: str,
    ) -> dict[str, Any]:
        self._assert_workspace_authority(workspace)
        try:
            relative = canonical_relative_path(relative_path)
            path = resolve_managed_path(workspace.path / "Intent", relative)
        except SafeIOError as error:
            raise PRWorkspaceSecurityError(str(error)) from error
        value = self._read_json(path, "Intent record")
        if type(value) is not dict:
            raise PRWorkspaceSecurityError("Intent record must be an object")
        return value

    def set_current_intent_version(
        self,
        workspace: PRWorkspace,
        version: int,
    ) -> None:
        self._assert_workspace_authority(workspace)
        if type(version) is not int or version < 1:
            raise PRWorkspaceError("Intent version must be a positive integer")
        manifest = self._validate_workspace_manifest(workspace)
        if manifest["current_intent_version"] == version:
            return
        manifest["current_intent_version"] = version
        try:
            atomic_replace_bytes(
                workspace.path / "manifest.json",
                canonical_json_bytes(manifest),
            )
        except SafeIOError as error:
            raise PRWorkspaceSecurityError(str(error)) from error

    def current_intent_version(self, workspace: PRWorkspace) -> int | None:
        self._assert_workspace_authority(workspace)
        return self._validate_workspace_manifest(workspace)["current_intent_version"]

    def describe_artifact(
        self,
        snapshot: SnapshotWorkspace,
        relative_path: str,
        content: bytes,
    ) -> ArtifactDescriptor:
        """Build the deterministic descriptor without publishing any bytes."""

        self._assert_snapshot_authority(snapshot)
        if type(content) is not bytes:
            raise PRWorkspaceError("Artifact content must be bytes")
        try:
            relative = canonical_relative_path(relative_path)
        except SafeIOError as error:
            raise PRWorkspaceSecurityError(str(error)) from error
        if relative == "snapshot.json" or relative.startswith(".artifacts/"):
            raise PRWorkspaceSecurityError("Artifact path is runtime-managed")
        try:
            destination = resolve_managed_path(snapshot.path, relative)
        except SafeIOError as error:
            raise PRWorkspaceSecurityError(str(error)) from error

        digest = sha256_hex(content)
        identity = {
            "pr_id": snapshot.workspace.pr_id,
            "snapshot_id": snapshot.snapshot_id,
            "relative_path": relative,
            "sha256": digest,
            "size_bytes": len(content),
        }
        artifact_id = _stable_id("A", identity)
        descriptor = ArtifactDescriptor(
            artifact_id=artifact_id,
            pr_id=snapshot.workspace.pr_id,
            snapshot_id=snapshot.snapshot_id,
            relative_path=relative,
            sha256=digest,
            size_bytes=len(content),
            path=destination,
        )
        return descriptor

    def resolve_snapshot_artifact(
        self,
        snapshot: SnapshotWorkspace,
        artifact_id: str,
    ) -> Path:
        descriptor, _content = self._resolve_artifact(snapshot, artifact_id)
        return descriptor.path

    def find_snapshot_artifact(
        self,
        snapshot: SnapshotWorkspace,
        relative_path: str,
    ) -> ArtifactDescriptor:
        """Resolve a trusted runtime path to its opaque Snapshot Artifact."""

        self._assert_snapshot_authority(snapshot)
        try:
            relative = canonical_relative_path(relative_path)
            receipt_paths = sorted(
                (snapshot.path / ".artifacts").iterdir(),
                key=lambda item: item.name,
            )
        except (OSError, SafeIOError) as error:
            raise PRWorkspaceSecurityError(
                "Snapshot Artifact catalog is unavailable"
            ) from error
        matches: list[ArtifactDescriptor] = []
        for receipt_path in receipt_paths:
            if re.fullmatch(r"a-[0-9a-f]{32}\.json", receipt_path.name) is None:
                raise PRWorkspaceSecurityError(
                    "Snapshot Artifact catalog contains an invalid entry"
                )
            payload = self._read_json(receipt_path, "Artifact receipt")
            if type(payload) is not dict or "artifact_id" not in payload:
                raise PRWorkspaceSecurityError("Artifact receipt is invalid")
            descriptor = self._load_artifact_descriptor(
                snapshot,
                payload["artifact_id"],
            )
            if descriptor.relative_path == relative:
                matches.append(descriptor)
        if not matches:
            raise PRWorkspaceSecurityError(
                "Artifact path is not authorized for this PR Snapshot"
            )
        if len(matches) != 1:
            raise PRWorkspaceSecurityError(
                "Artifact path has duplicate Snapshot authorizations"
            )
        descriptor, _content = self._resolve_artifact(
            snapshot,
            matches[0].artifact_id,
        )
        return descriptor

    def read_verified_json(
        self,
        snapshot: SnapshotWorkspace,
        artifact_id: str,
    ) -> dict[str, Any]:
        _descriptor, content = self._resolve_artifact(snapshot, artifact_id)
        try:
            value = strict_json_loads(content)
        except SafeIOError as error:
            raise PRWorkspaceSecurityError(str(error)) from error
        if type(value) is not dict:
            raise PRWorkspaceSecurityError("Artifact JSON root must be an object")
        return value

    def read_verified_artifact(
        self,
        snapshot: SnapshotWorkspace,
        artifact_id: str,
    ) -> bytes:
        """Read one Snapshot-authorized Artifact with its receipt binding verified."""

        _descriptor, content = self._resolve_artifact(snapshot, artifact_id)
        return content

    def _resolve_artifact(
        self,
        snapshot: SnapshotWorkspace,
        artifact_id: str,
    ) -> tuple[ArtifactDescriptor, bytes]:
        self._assert_snapshot_authority(snapshot)
        _validate_id(artifact_id, _ARTIFACT_ID, "artifact_id")
        descriptor = self._load_artifact_descriptor(snapshot, artifact_id)
        try:
            content = read_verified_bytes(
                descriptor.path,
                descriptor.sha256,
                max_bytes=descriptor.size_bytes,
            )
        except SafeIOError as error:
            raise PRWorkspaceSecurityError(str(error)) from error
        if len(content) != descriptor.size_bytes:
            raise PRWorkspaceSecurityError("Artifact size binding does not match")
        return descriptor, content

    def _load_artifact_descriptor(
        self,
        snapshot: SnapshotWorkspace,
        artifact_id: str,
    ) -> ArtifactDescriptor:
        receipt_path = self._artifact_receipt_path(snapshot, artifact_id)
        if not _path_exists(receipt_path):
            raise PRWorkspaceSecurityError(
                "Artifact ID is not authorized for this PR Snapshot"
            )
        payload = self._read_json(receipt_path, "Artifact receipt")
        value = _strict_object(
            payload,
            (
                "schema_version",
                "artifact_id",
                "pr_id",
                "snapshot_id",
                "relative_path",
                "sha256",
                "size_bytes",
            ),
            "Artifact receipt",
        )
        if value["schema_version"] != ARTIFACT_RECEIPT_SCHEMA:
            raise PRWorkspaceSecurityError("Artifact receipt schema is unsupported")
        _validate_id(value["artifact_id"], _ARTIFACT_ID, "artifact_id")
        if value["artifact_id"] != artifact_id:
            raise PRWorkspaceSecurityError("Artifact receipt binding does not match")
        if value["pr_id"] != snapshot.workspace.pr_id or value[
            "snapshot_id"
        ] != snapshot.snapshot_id:
            raise PRWorkspaceSecurityError("Artifact receipt PR binding does not match")
        try:
            relative = canonical_relative_path(value["relative_path"])
        except (KeyError, SafeIOError, TypeError) as error:
            raise PRWorkspaceSecurityError("Artifact receipt path is invalid") from error
        digest = value["sha256"]
        size = value["size_bytes"]
        if (
            type(digest) is not str
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or type(size) is not int
            or size < 0
        ):
            raise PRWorkspaceSecurityError("Artifact receipt content binding is invalid")
        identity = {
            "pr_id": snapshot.workspace.pr_id,
            "snapshot_id": snapshot.snapshot_id,
            "relative_path": relative,
            "sha256": digest,
            "size_bytes": size,
        }
        if _stable_id("A", identity) != artifact_id:
            raise PRWorkspaceSecurityError("Artifact ID does not match its receipt")
        try:
            path = resolve_managed_path(snapshot.path, relative)
        except SafeIOError as error:
            raise PRWorkspaceSecurityError(str(error)) from error
        return ArtifactDescriptor(
            artifact_id=artifact_id,
            pr_id=snapshot.workspace.pr_id,
            snapshot_id=snapshot.snapshot_id,
            relative_path=relative,
            sha256=digest,
            size_bytes=size,
            path=path,
        )

    def _workspace_path(self, pr_id: str) -> Path:
        _validate_id(pr_id, _PR_ID, "pr_id")
        return self._pr_root / ("p-" + pr_id[3:35])

    @staticmethod
    def _physical_snapshot_id(snapshot_id: str) -> str:
        _validate_id(snapshot_id, _SNAPSHOT_ID, "snapshot_id")
        return "s-" + snapshot_id[2:34]

    def _snapshot_id(
        self,
        workspace: PRWorkspace,
        base_sha: str,
        head_sha: str,
    ) -> str:
        return _stable_id(
            "S",
            {
                "repository_key": workspace.resolved_pr.repository.repository_key,
                "pr_id": workspace.pr_id,
                "base_sha": base_sha,
                "head_sha": head_sha,
            },
        )

    def _artifact_receipt_path(
        self,
        snapshot: SnapshotWorkspace,
        artifact_id: str,
    ) -> Path:
        _validate_id(artifact_id, _ARTIFACT_ID, "artifact_id")
        return snapshot.path / ".artifacts" / ("a-" + artifact_id[2:34] + ".json")

    def _validate_resolved_pr(self, value: ResolvedPR) -> None:
        if not isinstance(value, ResolvedPR) or not isinstance(
            value.repository, CanonicalRepositoryIdentity
        ):
            raise PRWorkspaceError("resolved PR identity is invalid")
        _validate_id(value.pr_id, _PR_ID, "pr_id")
        expected = _stable_id(
            "PR",
            {
                "repository_key": value.repository.repository_key,
                "provider": value.provider,
                "external_review_id": value.external_review_id,
            },
        )
        if expected != value.pr_id:
            raise PRWorkspaceSecurityError("resolved PR binding does not match")

    def _assert_workspace_authority(self, workspace: PRWorkspace) -> None:
        if not isinstance(workspace, PRWorkspace) or workspace.store_root != self.root:
            raise PRWorkspaceSecurityError("PR workspace belongs to another store")
        self._validate_resolved_pr(workspace.resolved_pr)
        if workspace.path != self._workspace_path(workspace.pr_id):
            raise PRWorkspaceSecurityError("PR workspace physical binding does not match")
        self._validate_workspace_manifest(workspace)
        self._validate_pr_metadata(workspace)

    def _assert_snapshot_authority(self, snapshot: SnapshotWorkspace) -> None:
        if not isinstance(snapshot, SnapshotWorkspace):
            raise PRWorkspaceSecurityError("Snapshot handle is invalid")
        self._assert_workspace_authority(snapshot.workspace)
        expected_id = self._snapshot_id(
            snapshot.workspace, snapshot.base_sha, snapshot.head_sha
        )
        expected_path = (
            snapshot.workspace.path
            / "Snapshots"
            / self._physical_snapshot_id(expected_id)
        )
        if snapshot.snapshot_id != expected_id or snapshot.path != expected_path:
            raise PRWorkspaceSecurityError("Snapshot handle binding does not match")
        self._validate_snapshot_manifest(snapshot)

    def _assert_session_authority(self, session: SessionWorkspace) -> None:
        if not isinstance(session, SessionWorkspace):
            raise PRWorkspaceSecurityError("Session handle is invalid")
        self._assert_snapshot_authority(session.snapshot)
        if (
            session.workspace != session.snapshot.workspace
            or session.workspace.store_root != self.root
        ):
            raise PRWorkspaceSecurityError("Session PR binding does not match")
        _validate_id(session.session_id, _SESSION_ID, "session_id")
        expected_path = session.workspace.path / "Sessions" / (
            "u-" + session.session_id[8:40]
        )
        if session.path != expected_path:
            raise PRWorkspaceSecurityError("Session physical binding does not match")
        payload = self._read_json(session.path / "state.json", "Session state")
        value = _strict_object(
            payload,
            (
                "schema_version",
                "session_id",
                "pr_id",
                "snapshot_id",
                "status",
            ),
            "Session state",
        )
        if (
            value["schema_version"] != SESSION_BINDING_SCHEMA
            or value["session_id"] != session.session_id
            or value["pr_id"] != session.workspace.pr_id
            or value["snapshot_id"] != session.snapshot.snapshot_id
        ):
            raise PRWorkspaceSecurityError("Session state binding does not match")
        if type(value["status"]) is not str or not value["status"]:
            raise PRWorkspaceSecurityError("Session status is invalid")

    def _validate_workspace_manifest(self, workspace: PRWorkspace) -> dict[str, Any]:
        payload = self._read_json(workspace.path / "manifest.json", "Workspace manifest")
        value = _strict_object(
            payload,
            (
                "workspace_schema_version",
                "pr_id",
                "current_snapshot_id",
                "current_intent_version",
            ),
            "Workspace manifest",
        )
        if (
            value["workspace_schema_version"] != WORKSPACE_SCHEMA
            or value["pr_id"] != workspace.pr_id
        ):
            raise PRWorkspaceSecurityError(
                "Workspace manifest binding or physical ID collision does not match"
            )
        current = value["current_snapshot_id"]
        if current is not None:
            _validate_id(current, _SNAPSHOT_ID, "current_snapshot_id")
        intent_version = value["current_intent_version"]
        if intent_version is not None and (
            type(intent_version) is not int or intent_version < 1
        ):
            raise PRWorkspaceSecurityError("Workspace Intent pointer is invalid")
        return value

    def _validate_pr_metadata(self, workspace: PRWorkspace) -> dict[str, Any]:
        payload = self._read_json(workspace.path / "PR" / "pr.json", "PR metadata")
        value = _strict_object(
            payload,
            (
                "schema_version",
                "pr_id",
                "repository_identity",
                "provider",
                "pr_number_or_external_review_id",
                "title",
                "description",
                "base_ref",
                "head_ref",
                "author",
                "status",
            ),
            "PR metadata",
        )
        if (
            value["schema_version"] != PR_METADATA_SCHEMA
            or value["pr_id"] != workspace.pr_id
            or value["repository_identity"]
            != workspace.resolved_pr.repository.to_dict()
            or value["provider"] != workspace.resolved_pr.provider
            or value["pr_number_or_external_review_id"]
            != workspace.resolved_pr.external_review_id
        ):
            raise PRWorkspaceSecurityError("PR metadata binding does not match")
        for field_name in (
            "title",
            "description",
            "base_ref",
            "head_ref",
            "author",
            "status",
        ):
            try:
                _optional_metadata_text(value[field_name], field_name)
            except PRWorkspaceError as error:
                raise PRWorkspaceSecurityError(str(error)) from error
        return value

    def _validate_snapshot_manifest(self, snapshot: SnapshotWorkspace) -> None:
        payload = self._read_json(snapshot.path / "snapshot.json", "Snapshot manifest")
        value = _strict_object(
            payload,
            (
                "schema_version",
                "repository_identity",
                "pr_id",
                "snapshot_id",
                "base_sha",
                "head_sha",
            ),
            "Snapshot manifest",
        )
        expected = {
            "schema_version": SNAPSHOT_SCHEMA,
            "repository_identity": snapshot.workspace.resolved_pr.repository.to_dict(),
            "pr_id": snapshot.workspace.pr_id,
            "snapshot_id": snapshot.snapshot_id,
            "base_sha": snapshot.base_sha,
            "head_sha": snapshot.head_sha,
        }
        if value != expected:
            raise PRWorkspaceSecurityError(
                "Snapshot manifest binding or physical ID collision does not match"
            )

    def _set_current_snapshot(self, workspace: PRWorkspace, snapshot_id: str) -> None:
        manifest = self._validate_workspace_manifest(workspace)
        if manifest["current_snapshot_id"] == snapshot_id:
            return
        manifest["current_snapshot_id"] = snapshot_id
        try:
            atomic_replace_bytes(
                workspace.path / "manifest.json",
                canonical_json_bytes(manifest),
            )
        except SafeIOError as error:
            raise PRWorkspaceSecurityError(str(error)) from error

    def _publish_json_create_only(
        self,
        path: Path,
        payload: Mapping[str, Any],
    ) -> None:
        self._publish_bytes_create_only(path, canonical_json_bytes(payload))

    @staticmethod
    def _publish_bytes_create_only(path: Path, content: bytes) -> None:
        try:
            publish_create_only_bytes(path, content)
        except SafeIOError as error:
            raise PRWorkspaceError(str(error)) from error

    @staticmethod
    def _read_json(path: Path, context: str) -> Any:
        try:
            return read_strict_json(path)
        except SafeIOError as error:
            raise PRWorkspaceSecurityError(f"{context}: {error}") from error


__all__ = [
    "ArtifactDescriptor",
    "PRMetadata",
    "PRWorkspace",
    "PRWorkspaceError",
    "PRWorkspaceSecurityError",
    "PRWorkspaceStore",
    "ResolvedPR",
    "ReviewResultArtifactBundle",
    "SessionWorkspace",
    "SnapshotWorkspace",
]
