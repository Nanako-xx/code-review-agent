from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path

from review_agent.run_state import RunPhase
from review_agent.session import (
    PhaseStatus,
    SESSION_V6_PHASES,
    SessionV6Manifest,
)
from review_agent.session_store import SessionV6Store


class LegacySessionUnsupportedError(RuntimeError):
    def __init__(self, diagnostic: "LegacySessionDiagnostic") -> None:
        self.diagnostic = diagnostic
        super().__init__(
            f"Session schema v{diagnostic.schema_version} is read-only and "
            "cannot be resumed by the v6 product pipeline"
        )


@dataclass(frozen=True)
class LegacyArtifactDiagnostic:
    name: str
    schema: str
    path: str
    valid: bool


@dataclass(frozen=True)
class LegacySessionDiagnostic:
    schema_version: int
    status: str
    current_phase: str
    artifacts: tuple[LegacyArtifactDiagnostic, ...]


@dataclass(frozen=True)
class SessionV6ResumeResult:
    manifest: SessionV6Manifest
    starting_phase: RunPhase | None
    reused_phases: tuple[RunPhase, ...]


def diagnose_legacy_session(run_dir: Path) -> LegacySessionDiagnostic:
    """Hydrate legacy state only through the isolated compatibility module."""

    compatibility = importlib.import_module("review_agent.legacy_resume")
    legacy = compatibility.diagnose_legacy_session(Path(run_dir))
    return LegacySessionDiagnostic(
        schema_version=legacy.schema_version,
        status=legacy.status,
        current_phase=legacy.current_phase,
        artifacts=tuple(
            LegacyArtifactDiagnostic(
                name=item.name,
                schema=item.schema,
                path=item.path,
                valid=item.valid,
            )
            for item in legacy.artifacts
        ),
    )


def require_v6_resume_from_legacy(run_dir: Path) -> None:
    raise LegacySessionUnsupportedError(diagnose_legacy_session(run_dir))


def resume_session_v6(store: SessionV6Store) -> SessionV6ResumeResult:
    if not isinstance(store, SessionV6Store):
        raise ValueError("store must be SessionV6Store")
    manifest = store.load()
    starting_phase = store.next_incomplete_phase()
    reused = tuple(
        phase
        for phase in SESSION_V6_PHASES
        if manifest.phases[phase.value].status is PhaseStatus.COMPLETED
    )
    return SessionV6ResumeResult(
        manifest=manifest,
        starting_phase=starting_phase,
        reused_phases=reused,
    )


__all__ = [
    "LegacyArtifactDiagnostic",
    "LegacySessionDiagnostic",
    "LegacySessionUnsupportedError",
    "SessionV6ResumeResult",
    "diagnose_legacy_session",
    "require_v6_resume_from_legacy",
    "resume_session_v6",
]
