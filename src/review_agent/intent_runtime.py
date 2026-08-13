from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any
import unicodedata

from review_agent.intent_inference import (
    IntentInferenceRun,
    intent_inference_run_to_dict,
    project_inference_goal_v2,
)
from review_agent.pr_workspace import (
    PRWorkspace,
    PRWorkspaceError,
    PRWorkspaceStore,
    SnapshotWorkspace,
)
from review_agent.review_protocol import (
    IntentPacket,
    IntentSource,
    IntentVersionEnvelope,
    ReviewRequest,
    WireProtocolError,
)
from review_agent.safe_io import canonical_json_bytes


class IntentRuntimeError(ValueError):
    pass


class IntentIntegrityError(IntentRuntimeError):
    pass


INTENT_ANALYSIS_SCHEMA = "intent_analysis_record_v2"
_LEGACY_INTENT_ANALYSIS_SCHEMA = "intent_analysis_record_v1"
_ANALYSIS_REF = re.compile(r"\AIA-[0-9a-f]{64}\Z")
INTENT_TRUST_POLICIES = frozenset({"normal", "evaluation_trust_model"})


def _optional_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise IntentRuntimeError(f"{field_name} must be non-empty text or null")
    return value.strip()


def _normalize_uncertainties(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        if type(value) is not str or not value.strip():
            continue
        text = " ".join(unicodedata.normalize("NFKC", value).split())
        if text not in normalized:
            normalized.append(text)
    return tuple(normalized)


@dataclass(frozen=True)
class IntentAnalysisRecord:
    source_snapshot_id: str
    request: ReviewRequest
    declared_goal: str | None
    pr_title: str | None
    pr_description: str | None
    inferred_goal: str | None
    inference_run: IntentInferenceRun | None
    trust_policy: str
    model_inference_promoted: bool
    selection_reason: str
    continued_from_version: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": INTENT_ANALYSIS_SCHEMA,
            "source_snapshot_id": self.source_snapshot_id,
            "public_conversation": [
                message.to_dict() for message in self.request.conversation
            ],
            "explicit_inputs": {
                "declared_goal": self.declared_goal,
                "pr_title": self.pr_title,
                "pr_description": self.pr_description,
            },
            "inferred_goal": self.inferred_goal,
            "inference_run": (
                intent_inference_run_to_dict(self.inference_run)
                if self.inference_run is not None
                else None
            ),
            "trust_policy": self.trust_policy,
            "model_inference_promoted": self.model_inference_promoted,
            "selection_reason": self.selection_reason,
            "continued_from_version": self.continued_from_version,
        }


class IntentRuntime:
    def __init__(self, workspace_store: PRWorkspaceStore) -> None:
        if not isinstance(workspace_store, PRWorkspaceStore):
            raise IntentRuntimeError("Intent Runtime requires a PRWorkspaceStore")
        self._store = workspace_store

    def resolve(
        self,
        workspace: PRWorkspace,
        snapshot: SnapshotWorkspace,
        request: ReviewRequest,
        *,
        declared_goal: str | None = None,
        pr_title: str | None = None,
        pr_description: str | None = None,
        inferred_goal: str | None = None,
        inference_run: IntentInferenceRun | None = None,
        trust_policy: str = "normal",
    ) -> IntentVersionEnvelope:
        self._store.verify_workspace(workspace)
        self._store.verify_snapshot(snapshot)
        if snapshot.workspace != workspace:
            raise IntentIntegrityError("Intent Snapshot is not bound to this PR")
        if not isinstance(request, ReviewRequest):
            raise IntentRuntimeError("Intent request must be a ReviewRequest")
        declared = _optional_text(declared_goal, "declared_goal")
        title = _optional_text(pr_title, "pr_title")
        description = _optional_text(pr_description, "pr_description")
        direct_inference = _optional_text(inferred_goal, "inferred_goal")
        if inference_run is not None and not isinstance(
            inference_run, IntentInferenceRun
        ):
            raise IntentRuntimeError("inference_run must be an IntentInferenceRun")
        if trust_policy not in INTENT_TRUST_POLICIES:
            raise IntentRuntimeError("Intent trust policy is unsupported")

        current = self.load_current(workspace)
        current_model_inference_promoted = False
        if current is not None:
            current_analysis = self.load_analysis_record(
                workspace,
                current.analysis_record_ref,
            )
            current_model_inference_promoted = (
                current_analysis.get("model_inference_promoted") is True
            )
        has_new_analysis = any(
            value is not None
            for value in (
                declared,
                title,
                description,
                direct_inference,
                inference_run,
            )
        )
        if (
            current is not None
            and current.source_snapshot_id == snapshot.snapshot_id
            and not has_new_analysis
        ):
            return current

        explicit = declared or title or description
        selection_reason: str
        model_inference_promoted = False
        projected_inferred_goal: str | None = direct_inference
        inference_uncertainties: tuple[str, ...] = ()
        if inference_run is not None:
            model_goal, inference_uncertainties = project_inference_goal_v2(
                inference_run
            )
            if projected_inferred_goal is None:
                projected_inferred_goal = model_goal
            elif model_goal is not None and model_goal != projected_inferred_goal:
                projected_inferred_goal = None
                inference_uncertainties = _normalize_uncertainties(
                    [
                        *inference_uncertainties,
                        "Intent analysis and deterministic inference conflict.",
                    ]
                )

        if explicit is not None:
            packet = IntentPacket(
                goal=explicit,
                source=IntentSource.EXPLICIT,
                uncertainties=(),
            )
            selection_reason = "explicit_declaration"
        elif (
            current is not None
            and current.packet.source is IntentSource.EXPLICIT
            and not current_model_inference_promoted
        ):
            packet = current.packet
            selection_reason = "explicit_continuation"
        elif projected_inferred_goal is not None:
            model_inference_promoted = (
                trust_policy == "evaluation_trust_model"
                and inference_run is not None
                and inference_run.status == "completed"
            )
            packet = IntentPacket(
                goal=projected_inferred_goal,
                source=(
                    IntentSource.EXPLICIT
                    if model_inference_promoted
                    else IntentSource.INFERRED
                ),
                uncertainties=_normalize_uncertainties(inference_uncertainties),
            )
            selection_reason = (
                "evaluation_model_inference_promoted"
                if model_inference_promoted
                else "inference_revalidated"
            )
        elif (
            current is not None
            and (
                current.packet.source is IntentSource.INFERRED
                or current_model_inference_promoted
            )
            and current.source_snapshot_id != snapshot.snapshot_id
        ):
            packet = IntentPacket(
                goal=None,
                source=None,
                uncertainties=(
                    "The previous inferred Intent requires revalidation for this Snapshot.",
                ),
            )
            selection_reason = (
                "evaluation_model_inference_revalidation_missing"
                if current_model_inference_promoted
                else "inference_revalidation_missing"
            )
        else:
            uncertainties = _normalize_uncertainties(inference_uncertainties)
            if not uncertainties:
                uncertainties = (
                    "No reliable review goal could be established.",
                )
            packet = IntentPacket(
                goal=None,
                source=None,
                uncertainties=uncertainties,
            )
            selection_reason = "reliable_goal_unavailable"

        analysis = IntentAnalysisRecord(
            source_snapshot_id=snapshot.snapshot_id,
            request=request,
            declared_goal=declared,
            pr_title=title,
            pr_description=description,
            inferred_goal=direct_inference,
            inference_run=inference_run,
            trust_policy=trust_policy,
            model_inference_promoted=model_inference_promoted,
            selection_reason=selection_reason,
            continued_from_version=current.version if current is not None else None,
        )
        return self._publish_version(workspace, snapshot, packet, analysis, current)

    def load_current(self, workspace: PRWorkspace) -> IntentVersionEnvelope | None:
        self._store.verify_workspace(workspace)
        pointer = self._store.current_intent_version(workspace)
        if not self._store.intent_current_exists(workspace):
            if pointer is not None:
                raise IntentIntegrityError(
                    "Workspace Intent pointer has no current record"
                )
            return None
        try:
            current = IntentVersionEnvelope.from_dict(
                self._store.read_intent_json(workspace, "current.json")
            )
        except (PRWorkspaceError, WireProtocolError) as error:
            raise IntentIntegrityError("Current Intent record is invalid") from error
        if pointer != current.version:
            raise IntentIntegrityError("Workspace Intent version pointer does not match")
        history_payload = self._store.read_intent_json(
            workspace,
            self._history_relative_path(current.version),
        )
        try:
            historical = IntentVersionEnvelope.from_dict(history_payload)
        except WireProtocolError as error:
            raise IntentIntegrityError("Intent history record is invalid") from error
        if historical != current:
            raise IntentIntegrityError("Current Intent differs from create-only history")
        self.load_analysis_record(workspace, current.analysis_record_ref)
        return current

    def load_current_packet(self, workspace: PRWorkspace) -> IntentPacket | None:
        current = self.load_current(workspace)
        return current.packet if current is not None else None

    def load_analysis_record(
        self,
        workspace: PRWorkspace,
        analysis_record_ref: str,
    ) -> dict[str, Any]:
        if type(analysis_record_ref) is not str or _ANALYSIS_REF.fullmatch(
            analysis_record_ref
        ) is None:
            raise IntentIntegrityError("Intent analysis record reference is invalid")
        relative = self._analysis_relative_path(analysis_record_ref)
        try:
            payload = self._store.read_intent_json(workspace, relative)
        except PRWorkspaceError as error:
            raise IntentIntegrityError("Intent analysis record is unavailable") from error
        legacy_fields = {
            "schema_version",
            "source_snapshot_id",
            "public_conversation",
            "explicit_inputs",
            "inferred_goal",
            "inference_run",
            "selection_reason",
            "continued_from_version",
        }
        expected_fields = {
            *legacy_fields,
            "trust_policy",
            "model_inference_promoted",
        }
        schema = payload.get("schema_version")
        if not (
            schema == INTENT_ANALYSIS_SCHEMA
            and set(payload) == expected_fields
        ) and not (
            schema == _LEGACY_INTENT_ANALYSIS_SCHEMA
            and set(payload) == legacy_fields
        ):
            raise IntentIntegrityError("Intent analysis record schema is invalid")
        digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        if analysis_record_ref != "IA-" + digest:
            raise IntentIntegrityError("Intent analysis record hash does not match")
        return payload

    def history_path(self, workspace: PRWorkspace, version: int) -> Path:
        self._store.verify_workspace(workspace)
        if type(version) is not int or version < 1:
            raise IntentRuntimeError("Intent version must be positive")
        return workspace.path / "Intent" / self._history_relative_path(version)

    def _publish_version(
        self,
        workspace: PRWorkspace,
        snapshot: SnapshotWorkspace,
        packet: IntentPacket,
        analysis: IntentAnalysisRecord,
        current: IntentVersionEnvelope | None,
    ) -> IntentVersionEnvelope:
        analysis_payload = analysis.to_dict()
        analysis_bytes = canonical_json_bytes(analysis_payload)
        analysis_ref = "IA-" + hashlib.sha256(analysis_bytes).hexdigest()
        analysis_filename = self._analysis_relative_path(analysis_ref).split("/", 1)[1]
        self._store.publish_intent_history(
            workspace,
            analysis_filename,
            analysis_bytes,
        )
        version_number = 1 if current is None else current.version + 1
        envelope = IntentVersionEnvelope(
            version=version_number,
            source_snapshot_id=snapshot.snapshot_id,
            packet=packet,
            analysis_record_ref=analysis_ref,
        )
        envelope_bytes = envelope.to_json_bytes()
        history_filename = self._history_relative_path(version_number).split("/", 1)[1]
        self._store.publish_intent_history(
            workspace,
            history_filename,
            envelope_bytes,
        )
        self._store.replace_current_intent(workspace, envelope_bytes)
        self._store.set_current_intent_version(workspace, version_number)
        return envelope

    @staticmethod
    def _history_relative_path(version: int) -> str:
        return f"history/v-{version:08d}.json"

    @staticmethod
    def _analysis_relative_path(analysis_record_ref: str) -> str:
        return f"history/analysis-{analysis_record_ref[3:35]}.json"


__all__ = [
    "IntentAnalysisRecord",
    "IntentIntegrityError",
    "IntentRuntime",
    "IntentRuntimeError",
]
