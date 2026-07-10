# Session Foundation And Revision Binding Implementation Plan

> **For agentic workers:** Execute this plan task-by-task with TDD. Use a fresh implementation context for each task when subagents are available, and perform spec-compliance plus code-quality review before advancing. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the permanent Session foundation for resumable reviews by binding every new run to resolved Base/Head commit SHAs, persisting non-secret execution configuration, atomically writing `session.json`, and registering verifiable run artifacts.

**Architecture:** Add a Git-only `RevisionResolver`, immutable Session schema dataclasses, and a `SessionStore` that atomically persists manifests and hashes existing artifacts. Integrate them into the existing CLI without yet refactoring the monolithic pipeline; Batch B will consume these exact models and stores to implement phase hydration and true continuation.

**Tech Stack:** Python dataclasses and enums, `subprocess` Git commands, `hashlib`, atomic `os.replace`, pytest, existing local CLI fixtures.

---

## Scope

This is Batch A from `docs/superpowers/specs/2026-07-10-review-session-memory-resume-design.md`.

It includes:

- Stable repository identity based on Git common directory.
- Resolution of requested Base/Head expressions to immutable commit SHAs.
- `session.json` schema version 1.
- Non-secret reviewer execution configuration persistence.
- Atomic manifest and checkpoint JSON writes.
- Artifact descriptors containing path, SHA-256, schema, phase, and revision binding.
- Session phase/status updates alongside the existing `state.json` summary.
- Requested and resolved revisions in Preflight output and `state.json`.
- CLI-created Review runs producing a completed or failed Session Manifest.

It does not include:

- Real phase continuation from `resume`.
- Typed hydration of reviewer/evidence artifacts.
- ObservationStore loading and validation.
- Reviewer sub-checkpoints or attempt directories.
- Revision drift child Session creation.
- Eval or platform integration.

Those capabilities use the same schema and store in Batches B and C; this batch does not introduce an alternate temporary checkpoint format.

## File structure

- Create `src/review_agent/revision.py`
  - Repository identity and commit resolution.
- Create `src/review_agent/session.py`
  - Session enums/dataclasses and strict serialization.
- Create `src/review_agent/session_store.py`
  - Atomic manifest writes, artifact hashing, registration, and state capture.
- Modify `src/review_agent/checkpoint.py`
  - Make JSON/state writes atomic through a shared safe write helper.
- Modify `src/review_agent/run_state.py`
  - Add optional resolved Base/Head SHA fields while preserving old payload compatibility.
- Modify `src/review_agent/cli.py`
  - Resolve revisions before execution, create Session Manifest, use SHAs internally, capture artifacts/status, and print requested/resolved revisions.
- Create `tests/test_revision.py`
- Create `tests/test_session.py`
- Create `tests/test_session_store.py`
- Modify `tests/test_checkpoint_reporting.py`
- Modify `tests/test_run_state.py`
- Modify `tests/test_cli_smoke.py`

---

## Task 1: Add repository identity and immutable revision resolution

**Files:**
- Create: `src/review_agent/revision.py`
- Create: `tests/test_revision.py`

- [ ] **Step 1: Write failing RevisionResolver tests**

Create `tests/test_revision.py`:

```python
from pathlib import Path

from conftest import run_git
from review_agent.revision import RevisionResolver


def test_revision_resolver_resolves_symbolic_revisions_to_commit_shas(git_repo: Path) -> None:
    base_sha = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "app.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    run_git(git_repo, "add", "app.py")
    run_git(git_repo, "commit", "-m", "change app")
    head_sha = run_git(git_repo, "rev-parse", "HEAD")

    resolved = RevisionResolver().resolve_pair(git_repo, "HEAD~1", "HEAD")

    assert resolved.requested_base == "HEAD~1"
    assert resolved.requested_head == "HEAD"
    assert resolved.resolved_base_sha == base_sha
    assert resolved.resolved_head_sha == head_sha


def test_repository_identity_uses_git_common_directory_without_requiring_origin(git_repo: Path) -> None:
    identity = RevisionResolver().repository_identity(git_repo)

    assert identity.canonical_path == str(git_repo.resolve())
    assert Path(identity.git_common_dir).resolve() == (git_repo / ".git").resolve()
    assert identity.origin_url is None


def test_revision_resolver_reports_invalid_revision(git_repo: Path) -> None:
    try:
        RevisionResolver().resolve_commit(git_repo, "missing-revision")
    except ValueError as error:
        assert "missing-revision" in str(error)
    else:
        raise AssertionError("expected invalid revision to raise ValueError")


def test_revision_resolver_checks_commit_existence(git_repo: Path) -> None:
    head_sha = run_git(git_repo, "rev-parse", "HEAD")

    resolver = RevisionResolver()

    assert resolver.commit_exists(git_repo, head_sha) is True
    assert resolver.commit_exists(git_repo, "0" * 40) is False
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_revision.py -q -p no:cacheprovider
```

Expected: collection fails with `ModuleNotFoundError: No module named 'review_agent.revision'`.

- [ ] **Step 3: Implement `src/review_agent/revision.py`**

Create:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess


@dataclass(frozen=True)
class RepositoryIdentity:
    canonical_path: str
    git_common_dir: str
    origin_url: str | None


@dataclass(frozen=True)
class ResolvedRevisions:
    requested_base: str
    requested_head: str
    resolved_base_sha: str
    resolved_head_sha: str


class RevisionResolver:
    def repository_identity(self, repo: Path) -> RepositoryIdentity:
        repository = Path(repo).resolve()
        top_level = Path(self._git(repository, ["rev-parse", "--show-toplevel"])).resolve()
        common_raw = self._git(repository, ["rev-parse", "--git-common-dir"])
        common_path = Path(common_raw)
        if not common_path.is_absolute():
            common_path = repository / common_path
        origin_url = self._optional_git(repository, ["remote", "get-url", "origin"])
        return RepositoryIdentity(
            canonical_path=str(top_level),
            git_common_dir=str(common_path.resolve()),
            origin_url=origin_url,
        )

    def resolve_pair(self, repo: Path, base_revision: str, head_revision: str) -> ResolvedRevisions:
        return ResolvedRevisions(
            requested_base=base_revision,
            requested_head=head_revision,
            resolved_base_sha=self.resolve_commit(repo, base_revision),
            resolved_head_sha=self.resolve_commit(repo, head_revision),
        )

    def resolve_commit(self, repo: Path, revision: str) -> str:
        result = self._run_git(Path(repo).resolve(), ["rev-parse", "--verify", f"{revision}^{{commit}}"])
        if result.returncode != 0:
            raise ValueError(f"revision does not resolve to a commit: {revision}")
        return result.stdout.strip()

    def commit_exists(self, repo: Path, sha: str) -> bool:
        result = self._run_git(Path(repo).resolve(), ["cat-file", "-e", f"{sha}^{{commit}}"])
        return result.returncode == 0

    def _git(self, repo: Path, args: list[str]) -> str:
        result = self._run_git(repo, args)
        if result.returncode != 0:
            message = result.stderr.strip() or f"git {' '.join(args)} failed"
            raise ValueError(message)
        return result.stdout.strip()

    def _optional_git(self, repo: Path, args: list[str]) -> str | None:
        result = self._run_git(repo, args)
        if result.returncode != 0:
            return None
        value = result.stdout.strip()
        return value or None

    def _run_git(self, repo: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=repo,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_revision.py -q -p no:cacheprovider
```

Expected: `4 passed`.

- [ ] **Step 5: Commit Task 1**

Run:

```powershell
git add src/review_agent/revision.py tests/test_revision.py
git commit -m "feat: resolve immutable review revisions"
```

---

## Task 2: Define Session Manifest schema and strict serialization

**Files:**
- Create: `src/review_agent/session.py`
- Create: `tests/test_session.py`

- [ ] **Step 1: Write failing Session schema tests**

Create `tests/test_session.py` with tests covering:

```python
from review_agent.revision import RepositoryIdentity, ResolvedRevisions
from review_agent.run_state import RunPhase, RunStatus
from review_agent.session import (
    SESSION_SCHEMA_VERSION,
    PhaseStatus,
    ReviewExecutionConfig,
    RevisionChangeKind,
    initial_session_manifest,
    session_manifest_from_dict,
    session_manifest_to_dict,
)


def execution_config() -> ReviewExecutionConfig:
    return ReviewExecutionConfig(
        reviewer_provider="openai-compatible",
        reviewer_model="review-model",
        reviewer_base_url="https://provider.example/v1",
        reviewer_api_key_env="REVIEW_AGENT_API_KEY",
        reviewer_mode="multi",
        reviewer_loop="agent-loop",
        non_interactive=True,
    )


def test_session_manifest_round_trips_with_pending_phases() -> None:
    manifest = initial_session_manifest(
        review_id="review-1",
        repository=RepositoryIdentity("C:/repo", "C:/repo/.git", None),
        revisions=ResolvedRevisions("main", "HEAD", "a" * 40, "b" * 40),
        execution=execution_config(),
        now="2026-07-10T00:00:00Z",
    )

    payload = session_manifest_to_dict(manifest)
    loaded = session_manifest_from_dict(payload)

    assert loaded == manifest
    assert payload["schema_version"] == SESSION_SCHEMA_VERSION
    assert payload["status"] == "created"
    assert payload["current_phase"] == "created"
    assert payload["revisions"]["resolved_head_sha"] == "b" * 40
    assert payload["phases"]["preflight"]["status"] == "pending"


def test_session_manifest_never_serializes_api_key_values() -> None:
    manifest = initial_session_manifest(
        review_id="review-1",
        repository=RepositoryIdentity("C:/repo", "C:/repo/.git", None),
        revisions=ResolvedRevisions("main", "HEAD", "a" * 40, "b" * 40),
        execution=execution_config(),
        now="2026-07-10T00:00:00Z",
    )

    payload = session_manifest_to_dict(manifest)
    execution = payload["execution"]

    assert execution["reviewer_api_key_env"] == "REVIEW_AGENT_API_KEY"
    assert "api_key" not in execution
    assert "authorization" not in str(payload).casefold()


def test_session_manifest_rejects_unsupported_schema_version() -> None:
    manifest = initial_session_manifest(
        review_id="review-1",
        repository=RepositoryIdentity("C:/repo", "C:/repo/.git", None),
        revisions=ResolvedRevisions("main", "HEAD", "a" * 40, "b" * 40),
        execution=execution_config(),
        now="2026-07-10T00:00:00Z",
    )
    payload = session_manifest_to_dict(manifest)
    payload["schema_version"] = SESSION_SCHEMA_VERSION + 1

    try:
        session_manifest_from_dict(payload)
    except ValueError as error:
        assert "schema_version" in str(error)
    else:
        raise AssertionError("expected unsupported schema version to fail")


def test_initial_session_has_initial_lineage() -> None:
    manifest = initial_session_manifest(
        review_id="review-1",
        repository=RepositoryIdentity("C:/repo", "C:/repo/.git", None),
        revisions=ResolvedRevisions("main", "HEAD", "a" * 40, "b" * 40),
        execution=execution_config(),
        now="2026-07-10T00:00:00Z",
    )

    assert manifest.parent_review_id is None
    assert manifest.root_review_id == "review-1"
    assert manifest.revision_change_kind is RevisionChangeKind.INITIAL
    assert manifest.status is RunStatus.CREATED
    assert manifest.current_phase is RunPhase.CREATED
    assert all(item.status is PhaseStatus.PENDING for item in manifest.phases.values())
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_session.py -q -p no:cacheprovider
```

Expected: `ModuleNotFoundError: No module named 'review_agent.session'`.

- [ ] **Step 3: Implement Session models**

Create `src/review_agent/session.py` with:

```python
SESSION_SCHEMA_VERSION = 1
SESSION_PHASES = (
    RunPhase.PREFLIGHT,
    RunPhase.REPOSITORY_INTELLIGENCE,
    RunPhase.REVIEWERS,
    RunPhase.RECONCILIATION,
    RunPhase.COMPLETION,
    RunPhase.FINAL_RISK,
    RunPhase.REPORTING,
)


class PhaseStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INVALIDATED = "invalidated"


class RevisionChangeKind(str, Enum):
    INITIAL = "initial"
    HEAD_MOVED = "head_moved"
    BASE_MOVED = "base_moved"
    BASE_AND_HEAD_MOVED = "base_and_head_moved"
```

Define frozen dataclasses:

```python
@dataclass(frozen=True)
class ReviewExecutionConfig:
    reviewer_provider: str
    reviewer_model: str | None
    reviewer_base_url: str | None
    reviewer_api_key_env: str
    reviewer_mode: str
    reviewer_loop: str
    non_interactive: bool


@dataclass(frozen=True)
class ArtifactDescriptor:
    name: str
    path: str
    sha256: str
    schema: str
    phase: RunPhase
    revision_binding: str | None


@dataclass(frozen=True)
class PhaseCheckpoint:
    status: PhaseStatus = PhaseStatus.PENDING
    attempts: int = 0
    started_at: str | None = None
    completed_at: str | None = None
    artifacts: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class SessionManifest:
    schema_version: int
    review_id: str
    parent_review_id: str | None
    root_review_id: str
    repository: RepositoryIdentity
    revisions: ResolvedRevisions
    original_base_sha: str
    incremental_from_sha: str | None
    revision_change_kind: RevisionChangeKind
    execution: ReviewExecutionConfig
    status: RunStatus
    current_phase: RunPhase
    last_successful_phase: RunPhase | None
    phases: dict[str, PhaseCheckpoint]
    artifacts: dict[str, ArtifactDescriptor]
    errors: list[str]
    created_at: str
    updated_at: str
```

Implement:

- `initial_session_manifest(...)`
- `session_manifest_to_dict(...)`
- `session_manifest_from_dict(...)`

Serialization must explicitly convert every enum and nested dataclass. `from_dict` must reject any schema version other than `SESSION_SCHEMA_VERSION` and must not silently invent missing semantic fields.

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_session.py -q -p no:cacheprovider
```

Expected: `4 passed`.

- [ ] **Step 5: Commit Task 2**

Run:

```powershell
git add src/review_agent/session.py tests/test_session.py
git commit -m "feat: define review session manifest"
```

---

## Task 3: Add atomic SessionStore and artifact registry

**Files:**
- Create: `src/review_agent/session_store.py`
- Create: `tests/test_session_store.py`
- Modify: `src/review_agent/checkpoint.py`
- Modify: `tests/test_checkpoint_reporting.py`

- [ ] **Step 1: Write failing atomic checkpoint and SessionStore tests**

Add to `tests/test_checkpoint_reporting.py`:

```python
def test_checkpoint_store_leaves_no_temp_file_after_json_write(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, "review-atomic")

    store.write_json("request.json", {"head": "HEAD"})

    assert json.loads((store.run_dir / "request.json").read_text(encoding="utf-8")) == {"head": "HEAD"}
    assert list(store.run_dir.glob("*.tmp")) == []
```

Create `tests/test_session_store.py` with helpers that build an initial manifest and tests:

```python
from hashlib import sha256
import json
from pathlib import Path

from review_agent.revision import RepositoryIdentity, ResolvedRevisions
from review_agent.run_state import RunPhase, RunStatus
from review_agent.session import ReviewExecutionConfig, initial_session_manifest
from review_agent.session_store import SessionStore


def manifest():
    return initial_session_manifest(
        review_id="review-1",
        repository=RepositoryIdentity("C:/repo", "C:/repo/.git", None),
        revisions=ResolvedRevisions("main", "HEAD", "a" * 40, "b" * 40),
        execution=ReviewExecutionConfig("fake", None, None, "REVIEW_AGENT_API_KEY", "single", "single-shot", True),
        now="2026-07-10T00:00:00Z",
    )


def test_session_store_atomically_round_trips_manifest(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    store.create(manifest())

    assert store.load() == manifest()
    assert list(tmp_path.glob("*.tmp")) == []


def test_session_store_registers_existing_artifact_hash_and_binding(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    store.create(manifest())
    artifact_path = tmp_path / "request.json"
    artifact_path.write_text('{"head":"HEAD"}', encoding="utf-8")

    updated = store.register_existing_artifact(
        name="request",
        relative_path="request.json",
        schema="review_request_v1",
        phase=RunPhase.PREFLIGHT,
        revision_binding=None,
        now="2026-07-10T00:01:00Z",
    )

    descriptor = updated.artifacts["request"]
    assert descriptor.sha256 == sha256(artifact_path.read_bytes()).hexdigest()
    assert descriptor.path == "request.json"
    assert descriptor.phase is RunPhase.PREFLIGHT
    assert store.validate_artifact(descriptor) is True


def test_session_store_rejects_artifact_path_escape(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    store.create(manifest())

    try:
        store.register_existing_artifact(
            name="secret",
            relative_path="../secret.txt",
            schema="text_v1",
            phase=RunPhase.PREFLIGHT,
            revision_binding=None,
            now="2026-07-10T00:01:00Z",
        )
    except ValueError as error:
        assert "outside" in str(error)
    else:
        raise AssertionError("expected escaped artifact path to fail")


def test_session_store_marks_phase_and_session_completed(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    store.create(manifest())

    store.mark_phase_completed(RunPhase.PREFLIGHT, ["request"], "2026-07-10T00:01:00Z")
    completed = store.mark_session_completed("2026-07-10T00:02:00Z")

    assert completed.status is RunStatus.COMPLETED
    assert completed.current_phase is RunPhase.COMPLETED
    assert completed.last_successful_phase is RunPhase.REPORTING
    assert completed.phases["preflight"].status.value == "completed"


def test_session_store_marks_session_failed_without_losing_last_successful_phase(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    store.create(manifest())
    store.mark_phase_completed(RunPhase.PREFLIGHT, [], "2026-07-10T00:01:00Z")

    failed = store.mark_session_failed(RunPhase.REVIEWERS, "provider unavailable", "2026-07-10T00:02:00Z")

    assert failed.status is RunStatus.FAILED
    assert failed.current_phase is RunPhase.FAILED
    assert failed.last_successful_phase is RunPhase.PREFLIGHT
    assert failed.phases["reviewers"].status.value == "failed"
    assert failed.errors == ["provider unavailable"]
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_session_store.py tests/test_checkpoint_reporting.py::test_checkpoint_store_leaves_no_temp_file_after_json_write -q -p no:cacheprovider
```

Expected: SessionStore import fails and checkpoint test fails until atomic writes are implemented.

- [ ] **Step 3: Make CheckpointStore JSON writes atomic**

In `src/review_agent/checkpoint.py`, add `os` and `uuid`, then replace direct `path.write_text(...)` in `write_json` with:

```python
def _atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
```

`write_json` serializes first and calls `_atomic_write_text(path, content)`.

- [ ] **Step 4: Implement SessionStore**

Create `src/review_agent/session_store.py` implementing:

```python
class SessionStore:
    def __init__(self, run_dir: Path) -> None: ...
    def create(self, manifest: SessionManifest) -> Path: ...
    def load(self) -> SessionManifest: ...
    def write(self, manifest: SessionManifest) -> Path: ...
    def register_existing_artifact(...) -> SessionManifest: ...
    def validate_artifact(self, descriptor: ArtifactDescriptor) -> bool: ...
    def mark_phase_completed(...) -> SessionManifest: ...
    def mark_session_completed(...) -> SessionManifest: ...
    def mark_session_failed(...) -> SessionManifest: ...
```

Rules:

- `create` fails if `session.json` already exists.
- All manifest writes use `_atomic_write_text` from `checkpoint.py`.
- Artifact paths are resolved and must remain under `run_dir`.
- Artifact hash is SHA-256 over raw bytes.
- `register_existing_artifact` updates the manifest atomically.
- `mark_phase_completed` preserves other checkpoints and appends unique artifact names.
- `mark_session_completed` requires every phase through reporting to be completed only after CLI integration marks them; the unit test may mark missing phases completed in a loop before calling it, or implementation may accept an explicit completed phase list. Use the stricter behavior: update the test helper to mark every `SESSION_PHASES` item before completing.
- `mark_session_failed` marks the named active phase failed, preserves `last_successful_phase`, and appends the error.

- [ ] **Step 5: Run tests to verify GREEN**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_session_store.py tests/test_checkpoint_reporting.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 6: Commit Task 3**

Run:

```powershell
git add src/review_agent/checkpoint.py src/review_agent/session_store.py tests/test_checkpoint_reporting.py tests/test_session_store.py
git commit -m "feat: persist atomic review sessions"
```

---

## Task 4: Add resolved revisions to RunState

**Files:**
- Modify: `src/review_agent/run_state.py`
- Modify: `tests/test_run_state.py`

- [ ] **Step 1: Write failing RunState compatibility tests**

Add tests:

```python
def test_run_state_serializes_requested_and_resolved_revisions() -> None:
    state = initial_run_state(
        review_id="review-1",
        repository_path="repo",
        base_revision="main",
        head_revision="HEAD",
        resolved_base_revision="a" * 40,
        resolved_head_revision="b" * 40,
    )

    payload = run_state_to_dict(state)
    loaded = run_state_from_dict(payload)

    assert loaded.resolved_base_revision == "a" * 40
    assert loaded.resolved_head_revision == "b" * 40
    assert payload["base_revision"] == "main"
    assert payload["head_revision"] == "HEAD"


def test_run_state_loads_legacy_payload_without_resolved_revisions() -> None:
    payload = run_state_to_dict(
        initial_run_state(
            review_id="review-1",
            repository_path="repo",
            base_revision="main",
            head_revision="HEAD",
        )
    )
    payload.pop("resolved_base_revision")
    payload.pop("resolved_head_revision")

    loaded = run_state_from_dict(payload)

    assert loaded.resolved_base_revision is None
    assert loaded.resolved_head_revision is None
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_run_state.py -q -p no:cacheprovider
```

Expected: `initial_run_state()` rejects resolved revision keyword arguments.

- [ ] **Step 3: Implement backward-compatible fields**

Add to `RunState`:

```python
resolved_base_revision: str | None = None
resolved_head_revision: str | None = None
```

Thread the fields through `initial_run_state`, `advance_run_state`, `fail_run_state`, `run_state_to_dict`, and `run_state_from_dict`. `from_dict` uses `.get(...)` for compatibility.

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_run_state.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 5: Commit Task 4**

Run:

```powershell
git add src/review_agent/run_state.py tests/test_run_state.py
git commit -m "feat: record resolved review revisions"
```

---

## Task 5: Integrate Session creation and artifact capture into CLI review

**Files:**
- Modify: `src/review_agent/cli.py`
- Modify: `tests/test_cli_smoke.py`
- Modify: `tests/test_cli_resume.py`

- [ ] **Step 1: Write failing CLI Session tests**

Add a CLI smoke test that creates a commit and calls:

```python
main([
    "review",
    "--repo", str(git_repo),
    "--base", "HEAD~1",
    "--head", "HEAD",
    "--reviewer-provider", "fake",
    "--non-interactive",
])
```

Then assert:

```python
session = json.loads((run_dir / "session.json").read_text(encoding="utf-8"))
state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))

assert session["schema_version"] == 1
assert session["revisions"]["requested_base"] == "HEAD~1"
assert session["revisions"]["requested_head"] == "HEAD"
assert session["revisions"]["resolved_base_sha"] == base_sha
assert session["revisions"]["resolved_head_sha"] == head_sha
assert session["execution"]["reviewer_provider"] == "fake"
assert session["execution"]["reviewer_api_key_env"] == "REVIEW_AGENT_API_KEY"
assert "api_key" not in session["execution"]
assert session["status"] == "completed"
assert session["current_phase"] == "completed"
assert session["artifacts"]["request"]["sha256"]
assert session["artifacts"]["review_brief"]["sha256"]
assert session["artifacts"]["report"]["sha256"]
assert state["resolved_base_revision"] == base_sha
assert state["resolved_head_revision"] == head_sha
```

Capture stdout and assert:

```python
assert "Requested base: HEAD~1" in output
assert f"Resolved base: {base_sha}" in output
assert "Requested head: HEAD" in output
assert f"Resolved head: {head_sha}" in output
```

Update completed resume test to assert output includes resolved Base/Head when `session.json` is present.

Add a failure test by monkeypatching `collect_change_summary` to raise after Session creation and assert:

```python
assert session["status"] == "failed"
assert session["last_successful_phase"] is None
assert "RuntimeError: boom" in session["errors"]
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_cli_smoke.py::test_cli_review_writes_session_manifest_with_resolved_revisions tests/test_cli_smoke.py::test_cli_review_records_failed_session_when_collection_fails tests/test_cli_resume.py::test_cli_resume_prints_completed_run_summary -q -p no:cacheprovider
```

Expected: `session.json` is missing and Preflight lacks resolved revision output.

- [ ] **Step 3: Add CLI Session helpers**

Import:

```python
from datetime import datetime, timezone
from review_agent.revision import RevisionResolver
from review_agent.session import ReviewExecutionConfig, initial_session_manifest
from review_agent.session_store import SessionStore
```

Add:

```python
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _execution_config(args: argparse.Namespace) -> ReviewExecutionConfig:
    return ReviewExecutionConfig(
        reviewer_provider=str(args.reviewer_provider),
        reviewer_model=str(args.reviewer_model) if args.reviewer_model else None,
        reviewer_base_url=str(args.reviewer_base_url) if args.reviewer_base_url else None,
        reviewer_api_key_env=str(args.reviewer_api_key_env),
        reviewer_mode=str(args.reviewer_mode),
        reviewer_loop=str(args.reviewer_loop),
        non_interactive=bool(args.non_interactive),
    )
```

At `_run_review()` start:

1. Resolve repository identity and revisions.
2. Create `review_id`, `CheckpointStore`, and `SessionStore`.
3. Create initial manifest.
4. Create `RunState` with requested and resolved revision fields.
5. Use resolved SHA values for all Git, Repository Intelligence, ToolGateway, Observation revision, and Review Brief execution inputs.
6. Keep requested expressions in `ReviewRequest` and human-readable output.

- [ ] **Step 4: Capture Session status and artifacts**

Add CLI helpers:

```python
ARTIFACT_SCHEMAS = {
    "request": "review_request_v1",
    "intent": "intent_packet_v1",
    "risk_packet": "risk_packet_v1",
    "risk": "risk_assessment_v1",
    "assignments": "assignments_v1",
    "quality_gates": "quality_gates_v1",
    "repository_intelligence": "repository_intelligence_v1",
    "multi_reviewer": "multi_reviewer_result_v1",
    "reconciliation": "reconciliation_v1",
    "completion": "completion_v1",
    "final_risk": "final_risk_v1",
    "review_brief": "review_brief_v1",
    "report": "review_brief_markdown_v1",
}
```

After each existing state checkpoint:

- Register newly listed `state.artifacts` that exist and are not already registered with the same hash.
- Mark the corresponding Session phase completed.
- On final completion register `report`, `review_brief`, `final_risk`, and existing observations if present, then mark Session completed.
- On `_record_failed_review_state`, also mark the Session failed at the active phase.

Implement these mechanics through small CLI helpers calling SessionStore; do not duplicate manifest mutation logic in `_run_review()`.

For Batch A, state-tracked artifacts and final report/observations are registered. Per-reviewer attempt isolation and complete raw artifact registration move to Batch B when Reviewer Stage is extracted; no alternate descriptor format is introduced.

- [ ] **Step 5: Update Preflight and resume output**

Preflight prints requested and resolved values separately.

Resume loads `session.json` when available and prints:

```text
Requested Base
Resolved Base
Requested Head
Resolved Head
```

Legacy runs without `session.json` keep the existing summary behavior.

- [ ] **Step 6: Run CLI tests to verify GREEN**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_cli_smoke.py tests/test_cli_resume.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 7: Run Session foundation suite**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_revision.py tests/test_session.py tests/test_session_store.py tests/test_run_state.py tests/test_checkpoint_reporting.py tests/test_cli_smoke.py tests/test_cli_resume.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 8: Commit Task 5**

Run:

```powershell
git add src/review_agent/cli.py tests/test_cli_smoke.py tests/test_cli_resume.py
git commit -m "feat: create revision-bound review sessions"
```

---

## Task 6: Final verification

- [ ] **Step 1: Run architecture and focused tests**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_architecture_boundaries.py tests/test_revision.py tests/test_session.py tests/test_session_store.py tests/test_run_state.py tests/test_checkpoint_reporting.py tests/test_cli_smoke.py tests/test_cli_resume.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 2: Run full suite**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest -q -p no:cacheprovider
```

Expected: PASS. The known Windows `pytest-158 PermissionError` cleanup warning may appear after a successful exit code 0.

- [ ] **Step 3: Manual CLI smoke**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; $env:PYTHONPATH='src'; python -c "from review_agent.cli import main; raise SystemExit(main(['review','--repo','.','--base','HEAD~1','--head','HEAD','--reviewer-provider','fake','--non-interactive']))"
```

Verify output contains Requested/Resolved Base/Head. Inspect the newest `session.json` and confirm:

```text
schema_version = 1
status = completed
resolved revisions are 40-character SHAs
execution has API key environment name but no API key value
request/report/review_brief artifacts have SHA-256 descriptors
```

- [ ] **Step 4: Clean generated artifacts**

Run the existing workspace-safe cleanup for `.review-agent` and Python caches.

---

## Completion checklist

- Every new Review writes `session.json` schema version 1.
- Session repository identity is based on canonical Git common directory.
- Requested Base/Head expressions and resolved commit SHAs are both preserved.
- Internal review reads use resolved commit SHAs.
- Provider execution config is persisted without secret values.
- Checkpoint JSON and Session Manifest writes are atomic.
- Registered artifacts carry path, SHA-256, schema, phase, and revision binding.
- Completed runs mark Session completed; failed runs preserve errors and last successful phase.
- `state.json` remains backward compatible and includes optional resolved revisions.
- Existing review and resume CLI behavior remains compatible.
- Full tests pass.
