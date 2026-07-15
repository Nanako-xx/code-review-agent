from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from review_agent.git_repo import ChangeSummary
from review_agent.memory_models import (
    MAX_SNAPSHOT_RECORDS,
    DurableMemoryRecord,
    MemoryKind,
    RecordStatus,
    Sensitivity,
    canonical_sha256,
)
from review_agent.memory_policy import (
    PolicyCompilation,
    PolicyDiagnosticSeverity,
    RaiseRiskFloorAction,
)
from review_agent.models import (
    CompiledRiskFloor,
    IntentPacket,
    MemoryDiagnostic,
    MemoryDiagnosticCode,
    MemoryReference,
    MemoryRiskSignal,
    RiskAssessment,
    RiskAssessmentPacket,
    RiskMemoryProjection,
    RiskLevel,
    hard_policy_overflow_diagnostic,
)


SENSITIVE_PATH_MARKERS = ("auth", "payment", "billing", "security", "migration", "permissions")


class RiskAssessor(Protocol):
    def assess(self, packet: RiskAssessmentPacket) -> RiskAssessment:
        raise NotImplementedError


def build_risk_packet(
    change_summary: ChangeSummary,
    intent_packet: IntentPacket,
    quality_gate_status: Mapping[str, str],
    repository_intelligence: object | None = None,
    memory_projection: RiskMemoryProjection | None = None,
) -> RiskAssessmentPacket:
    """Build the bounded, deterministic input used by local and model risk assessors.

    ``RiskAssessmentPacket.changed_symbols`` and ``signal_catalog`` are intentionally
    constructor keywords.  This implementation therefore depends on the matching
    fields in ``review_agent.models.RiskAssessmentPacket``; the explicit check gives
    a useful failure if this file is combined with an older ``models.py``.
    """

    packet_fields = getattr(RiskAssessmentPacket, "__dataclass_fields__", {})
    required_packet_fields = {"changed_symbols", "signal_catalog"}
    missing_packet_fields = required_packet_fields - set(packet_fields)
    if missing_packet_fields:
        missing = ", ".join(sorted(missing_packet_fields))
        raise RuntimeError(
            "RiskAssessmentPacket is missing model-risk field(s): " + missing
        )

    changed_symbols = _changed_symbol_summaries(repository_intelligence)
    normalized_quality_gates = _normalized_quality_gates(quality_gate_status)
    if memory_projection is not None and not isinstance(
        memory_projection, RiskMemoryProjection
    ):
        raise ValueError("memory_projection must be a RiskMemoryProjection or None")
    normalized_intent_uncertainties = [
        _non_empty_text(item, f"intent uncertainty {index}").strip()
        for index, item in enumerate(intent_packet.uncertainties)
    ]
    signal_catalog = _build_signal_catalog(
        changed_files=change_summary.changed_files,
        changed_symbols=changed_symbols,
        quality_gate_status=normalized_quality_gates,
        intent_status=intent_packet.status.value,
        intent_uncertainties=normalized_intent_uncertainties,
        diff_excerpt=change_summary.diff_excerpt[:80],
    )
    if memory_projection is not None:
        for signal in memory_projection.signals:
            signal_catalog[signal.signal_ref] = signal.summary
        signal_catalog = dict(sorted(signal_catalog.items()))
    return RiskAssessmentPacket(
        change_summary={
            "repository_path": change_summary.repository_path,
            "base_revision": change_summary.base_revision,
            "head_revision": change_summary.head_revision,
            "changed_files": list(change_summary.changed_files),
            "diff_stat": change_summary.diff_stat,
        },
        deterministic_signals={
            "quality_gates": normalized_quality_gates,
            "changed_file_count": len(change_summary.changed_files),
        },
        intent_status=intent_packet.status,
        intent_uncertainties=normalized_intent_uncertainties,
        diff_excerpt=list(change_summary.diff_excerpt[:80]),
        changed_symbols=changed_symbols,
        signal_catalog=signal_catalog,
        memory_projection=memory_projection,
    )


def build_risk_memory_projection(
    records: Sequence[DurableMemoryRecord],
    compilation: PolicyCompilation,
    *,
    diagnostics: Sequence[MemoryDiagnostic] = (),
    max_hard_policy_items: int = 64,
    max_hard_policy_bytes: int = 32_768,
) -> RiskMemoryProjection:
    """Build the minimal risk view; only compiled floor actions carry authority."""

    if not isinstance(compilation, PolicyCompilation):
        raise ValueError("compilation must be a PolicyCompilation")
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ValueError("records must be a sequence of DurableMemoryRecord values")
    if len(records) > MAX_SNAPSHOT_RECORDS:
        raise ValueError("records exceed the bounded Memory projection limit")
    by_id: dict[str, DurableMemoryRecord] = {}
    references: dict[str, MemoryReference] = {}
    signals: list[MemoryRiskSignal] = []
    for item in records:
        if type(item) is not DurableMemoryRecord:
            raise ValueError("records must contain DurableMemoryRecord values")
        canonical = DurableMemoryRecord.from_dict(item.to_dict())
        if canonical != item or canonical.status is not RecordStatus.ACTIVE:
            raise ValueError("risk projection accepts only canonical active records")
        if canonical.sensitivity is Sensitivity.BLOCKED:
            raise ValueError("blocked-sensitivity Memory cannot enter risk projection")
        if canonical.memory_id in by_id:
            raise ValueError("records must not repeat a memory_id")
        by_id[canonical.memory_id] = canonical
        reference = _memory_reference(canonical)
        references[canonical.memory_id] = reference
        if canonical.kind not in {
            MemoryKind.HIGH_RISK_MODULE,
            MemoryKind.INCIDENT_LESSON,
        }:
            continue
        signals.append(
            MemoryRiskSignal(
                signal_ref=f"memory:{canonical.memory_id}",
                summary=canonical.statement,
                memory=reference,
            )
        )

    projected_diagnostics = list(diagnostics)
    projected_diagnostics.extend(
        MemoryDiagnostic(
            code=MemoryDiagnosticCode.POLICY_REJECTED,
            message=item.message,
            memory_ids=(() if item.memory_id is None else (item.memory_id,)),
        )
        for item in compilation.diagnostics
        if item.severity is PolicyDiagnosticSeverity.BLOCKING
    )
    floor_actions = [
        item for item in compilation.actions if type(item) is RaiseRiskFloorAction
    ]
    risk_floor = None
    policy_sources: tuple[MemoryReference, ...] = ()
    if floor_actions:
        order = {level: index for index, level in enumerate(RiskLevel)}
        highest = max(floor_actions, key=lambda item: order[item.minimum_level])
        all_ids = tuple(sorted({mid for item in floor_actions for mid in item.memory_ids}))
        risk_floor = CompiledRiskFloor(highest.minimum_level, all_ids)
        policy_sources = tuple(
            references[memory_id]
            for memory_id in all_ids
            if memory_id in references
        )
        missing_ids = tuple(
            memory_id for memory_id in all_ids if memory_id not in references
        )
        if missing_ids:
            projected_diagnostics.append(
                MemoryDiagnostic(
                    code=MemoryDiagnosticCode.POLICY_REJECTED,
                    message="compiled risk floor provenance is absent from the stage input",
                    memory_ids=missing_ids,
                )
            )
        overflow = hard_policy_overflow_diagnostic(
            "initial_risk",
            tuple(item.to_dict() for item in floor_actions),
            all_ids,
            max_items=max_hard_policy_items,
            max_bytes=max_hard_policy_bytes,
        )
        if overflow is not None:
            projected_diagnostics.append(overflow)
    return RiskMemoryProjection(
        signals=tuple(signals),
        risk_floor=risk_floor,
        policy_sources=policy_sources,
        diagnostics=tuple(projected_diagnostics),
    )


project_memory_for_risk = build_risk_memory_projection


class LocalRiskAssessor:
    """Deterministic offline assessor for tests and provider-free smoke runs.

    Model-assisted planning consumes the same packet, but Runtime always retains
    this assessment as the authoritative lower bound.
    """

    def assess(self, packet: RiskAssessmentPacket) -> RiskAssessment:
        changed_files = [str(path) for path in packet.change_summary["changed_files"]]
        sensitive_files = sorted(
            path
            for path in changed_files
            if any(marker in path.lower() for marker in SENSITIVE_PATH_MARKERS)
        )
        quality_gates = packet.deterministic_signals.get("quality_gates", {})
        if not isinstance(quality_gates, Mapping):
            quality_gates = {}
        failed_gates = sorted(
            str(name)
            for name, status in quality_gates.items()
            if status == "failed"
        )
        unavailable_gates = sorted(
            str(name)
            for name, status in quality_gates.items()
            if status in {"unavailable", "timed_out", "error"}
        )
        signal_refs: list[str] = []
        uncertainties = list(packet.intent_uncertainties)

        if sensitive_files:
            level = RiskLevel.HIGH
            reasons = [f"sensitive path changed: {path}" for path in sensitive_files]
            signal_refs.extend(f"changed_file:{path}" for path in sensitive_files)
            focus = ["caller compatibility", "regression safety", "test adequacy"]
        elif len(changed_files) > 8 and not _all_doc_like(changed_files):
            level = RiskLevel.MEDIUM
            reasons = [f"many non-documentation files changed: {len(changed_files)}"]
            signal_refs.append("changed_file_count")
            focus = ["blast radius", "test adequacy"]
        else:
            level = RiskLevel.LOW
            reasons = ["small or documentation-only non-sensitive change set"]
            focus = ["intent alignment", "changed file sanity"]

        quality_reasons = [
            *(f"quality gate failed: {name}" for name in failed_gates),
            *(
                f"quality gate verification unavailable: {name}"
                for name in unavailable_gates
            ),
        ]
        if failed_gates:
            level = RiskLevel.HIGH
            focus = list(
                dict.fromkeys(["failed quality gate", "regression safety", *focus])
            )
        elif unavailable_gates and level is RiskLevel.LOW:
            level = RiskLevel.MEDIUM
        if unavailable_gates:
            focus = list(dict.fromkeys(["verification gap", "test adequacy", *focus]))
        if quality_reasons:
            reasons = [*quality_reasons, *reasons]
            signal_refs.extend(
                f"quality_gate:{name}"
                for name in [*failed_gates, *unavailable_gates]
            )

        memory = packet.memory_projection
        if memory is not None:
            for signal in memory.signals:
                if signal.memory.local_only:
                    continue
                reasons.append(f"approved memory risk signal: {signal.summary}")
                signal_refs.append(signal.signal_ref)
            uncertainties.extend(
                f"memory {item.code.value}: {item.message}"
                for item in memory.diagnostics
            )
            if memory.signals:
                focus = list(dict.fromkeys(["approved incident lessons", *focus]))
            if memory.risk_floor is not None:
                order = {value: index for index, value in enumerate(RiskLevel)}
                if order[memory.risk_floor.minimum_level] > order[level]:
                    level = memory.risk_floor.minimum_level
                    reasons.append(
                        "compiled Memory risk floor applied: "
                        + memory.risk_floor.minimum_level.value
                    )
                signal_refs.extend(
                    f"memory_floor:{memory_id}"
                    for memory_id in memory.risk_floor.memory_ids
                    if memory_id not in set(memory.local_only_memory_ids)
                )

        return RiskAssessment(
            level=level,
            dimensions={
                "impact": "derived from changed paths and quality gates",
                "blast_radius": "derived from changed file semantics and count",
                "reversibility": "not assessed by local fallback",
                "uncertainty": "derived from intent uncertainties",
                "verification_strength": "derived from quality gates",
            },
            reasons=reasons,
            signal_refs=list(dict.fromkeys(signal_refs)),
            uncertainties=list(dict.fromkeys([*packet.intent_uncertainties, *uncertainties])),
            suggested_focus=focus,
        )


def _all_doc_like(paths: list[str]) -> bool:
    doc_suffixes = (".md", ".rst", ".txt", ".adoc")
    doc_prefixes = ("docs/", "doc/")
    return bool(paths) and all(
        path.lower().endswith(doc_suffixes) or path.lower().startswith(doc_prefixes)
        for path in paths
    )


def _memory_reference(record: DurableMemoryRecord) -> MemoryReference:
    return MemoryReference(
        memory_id=record.memory_id,
        kind=record.kind.value,
        source_refs=tuple(
            [f"memory:{record.memory_id}"]
            + [
                f"memory-source:{canonical_sha256(source.to_dict())}"
                for source in record.source_refs
            ]
        ),
        local_only=record.sensitivity is Sensitivity.LOCAL_ONLY,
    )


def _changed_symbol_summaries(
    repository_intelligence: object | None,
) -> list[dict[str, object]]:
    if repository_intelligence is None:
        return []
    if isinstance(repository_intelligence, Mapping):
        raw_symbols = repository_intelligence.get("changed_symbols", [])
    else:
        raw_symbols = getattr(repository_intelligence, "changed_symbols", [])
    if isinstance(raw_symbols, (str, bytes)) or not isinstance(raw_symbols, Sequence):
        raise ValueError("repository_intelligence.changed_symbols must be a sequence")

    summaries: list[dict[str, object]] = []
    for index, symbol in enumerate(raw_symbols):
        if isinstance(symbol, Mapping):
            source = symbol
            value = source.get
        else:
            value = lambda name, default=None, item=symbol: getattr(
                item,
                name,
                default,
            )
        context = f"repository_intelligence.changed_symbols[{index}]"
        path = _non_empty_text(value("path"), f"{context}.path")
        qualified_name = _non_empty_text(
            value("qualified_name"),
            f"{context}.qualified_name",
        )
        kind = _non_empty_text(value("kind"), f"{context}.kind")
        change_type = _non_empty_text(
            value("change_type"),
            f"{context}.change_type",
        )
        line_start = _positive_int(value("line_start"), f"{context}.line_start")
        line_end = _positive_int(value("line_end"), f"{context}.line_end")
        if line_end < line_start:
            raise ValueError(f"{context}.line_end must be at least line_start")
        summaries.append(
            {
                "path": path,
                "qualified_name": qualified_name,
                "kind": kind,
                "change_type": change_type,
                "line_start": line_start,
                "line_end": line_end,
            }
        )
    return sorted(
        summaries,
        key=lambda item: (
            str(item["path"]),
            int(item["line_start"]),
            str(item["qualified_name"]),
            str(item["change_type"]),
        ),
    )


def _build_signal_catalog(
    *,
    changed_files: Sequence[str],
    changed_symbols: Sequence[Mapping[str, object]],
    quality_gate_status: Mapping[str, str],
    intent_status: str,
    intent_uncertainties: Sequence[str],
    diff_excerpt: Sequence[str],
) -> dict[str, str]:
    catalog: dict[str, str] = {
        "changed_file_count": f"Changed file count: {len(changed_files)}",
        "intent_status": f"Intent status: {intent_status}",
    }
    normalized_paths = [
        _non_empty_text(item, f"changed_files[{index}]")
        for index, item in enumerate(changed_files)
    ]
    for path in sorted(normalized_paths):
        catalog[f"changed_file:{path}"] = f"Changed file: {path}"
    for name, status in sorted(quality_gate_status.items()):
        gate_name = _non_empty_text(name, "quality gate name")
        gate_status = _non_empty_text(status, f"quality gate {gate_name} status")
        catalog[f"quality_gate:{gate_name}"] = (
            f"Quality gate {gate_name}: {gate_status}"
        )
    for symbol in changed_symbols:
        path = str(symbol["path"])
        qualified_name = str(symbol["qualified_name"])
        change_type = str(symbol["change_type"])
        ref = f"changed_symbol:{path}:{qualified_name}:{change_type}"
        catalog[ref] = (
            f"{change_type} {symbol['kind']} {qualified_name} at "
            f"{path}:{symbol['line_start']}-{symbol['line_end']}"
        )
    for index, uncertainty in enumerate(intent_uncertainties):
        text = _non_empty_text(uncertainty, f"intent uncertainty {index}")
        catalog[f"intent_uncertainty:{index:03d}"] = text
    for index, line in enumerate(diff_excerpt):
        if not isinstance(line, str):
            raise ValueError(f"diff excerpt {index} must be a string")
        if not line.strip():
            continue
        catalog[f"diff_excerpt:{index:03d}"] = line
    return {
        ref: _non_empty_text(description, f"signal catalog {ref}").strip()
        for ref, description in sorted(catalog.items())
    }


def _normalized_quality_gates(
    quality_gate_status: Mapping[str, str],
) -> dict[str, str]:
    if not isinstance(quality_gate_status, Mapping):
        raise ValueError("quality_gate_status must be a mapping")
    normalized: dict[str, str] = {}
    for name, status in quality_gate_status.items():
        gate_name = _non_empty_text(name, "quality gate name")
        normalized[gate_name] = _non_empty_text(
            status,
            f"quality gate {gate_name} status",
        )
    return dict(sorted(normalized.items()))


def _non_empty_text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _positive_int(value: object, context: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{context} must be a positive integer")
    return value
