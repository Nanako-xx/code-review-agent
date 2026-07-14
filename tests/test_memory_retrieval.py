from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from conftest import run_git
import review_agent.memory_retrieval as memory_retrieval_module
from review_agent.memory_identity import repository_key
from review_agent.memory_lifecycle import TargetHeadApplicabilityEvaluator
from review_agent.memory_models import (
    Applicability,
    DurableMemoryRecord,
    GenerationMetadata,
    GitCommitSourceRef,
    MemoryConfidence,
    MemoryKind,
    MemoryScope,
    MemorySelectionInput,
    MemorySnapshot,
    PolicyEffect,
    PolicyEffectKind,
    RecordStatus,
    RepositoryRangeSourceRef,
    Sensitivity,
    ValidityPolicy,
    canonical_json,
    stable_event_id,
    stable_id,
)
from review_agent.memory_retrieval import (
    HardPolicyBudgetExceeded,
    MemoryQuery,
    MemorySnapshotBuilder,
    ProjectionBudgetExceeded,
    QueryLimitExceeded,
    QueryScopeViolation,
    RetrievalLimits,
    RetrievalRequest,
    RetrievalStage,
    SemanticRankerViolation,
    SnapshotMemoryQueryService,
    SnapshotMemorySelector,
    SnapshotBudgetExceeded,
    build_disabled_snapshot,
)
from review_agent.memory_sources import SourceValidator, repository_range_hash
from review_agent.revision import RevisionResolver


NOW = "2026-07-15T00:00:00Z"


def _head(repo: Path) -> str:
    return run_git(repo, "rev-parse", "HEAD")


def _repository_key(repo: Path) -> str:
    return repository_key(RevisionResolver().repository_identity(repo))


def _selection(
    repo: Path,
    *,
    head: str | None = None,
    paths: tuple[str, ...] = ("app.py",),
    symbols: tuple[str, ...] = (),
    contracts: tuple[str, ...] = (),
    languages: tuple[str, ...] = ("python",),
    generations: GenerationMetadata | None = None,
) -> MemorySelectionInput:
    revision = head or _head(repo)
    return MemorySelectionInput(
        review_id="review-memory-retrieval",
        repository_key=_repository_key(repo),
        base_sha=revision,
        head_sha=revision,
        changed_paths=paths,
        changed_symbols=symbols,
        contracts=contracts,
        languages=languages,
        generations=generations
        or GenerationMetadata(
            store_schema_version=2,
            memory_generation=7,
            feedback_generation=8,
            knowledge_generation=9,
        ),
    )


def _record(
    repo: Path,
    label: str,
    *,
    valid_from: str | None = None,
    repository_key_value: str | None = None,
    kind: MemoryKind = MemoryKind.REVIEW_RULE,
    statement: str | None = None,
    scope: MemoryScope | None = None,
    source_refs: tuple | None = None,
    policies: tuple[ValidityPolicy, ...] = (ValidityPolicy.MANUAL_UNTIL_REVOKED,),
    effect: PolicyEffect | None = None,
    status: RecordStatus = RecordStatus.ACTIVE,
) -> DurableMemoryRecord:
    revision = valid_from or _head(repo)
    candidate_id = stable_id("MC", "retrieval-test", label)
    return DurableMemoryRecord(
        candidate_id=candidate_id,
        repository_key=repository_key_value or _repository_key(repo),
        kind=kind,
        statement=statement or f"Memory statement for {label}.",
        scope=scope or MemoryScope(paths=("app.py",)),
        source_refs=source_refs or (GitCommitSourceRef(revision),),
        source_bundle_hash=stable_id("BLOB", label).split("-", 1)[1],
        valid_from_sha=revision,
        validity_policies=policies,
        confidence=MemoryConfidence.HIGH,
        sensitivity=Sensitivity.NORMAL,
        policy_effect=effect,
        approved_by="amy",
        approval_event_id=stable_event_id("approve", candidate_id),
        status=status,
        created_at=NOW,
    )


def _builder(repo: Path, limits: RetrievalLimits | None = None) -> MemorySnapshotBuilder:
    return MemorySnapshotBuilder(
        TargetHeadApplicabilityEvaluator(repo, SourceValidator(repo)),
        limits=limits,
    )


def _decisions(snapshot: MemorySnapshot):
    return {item.memory_id: item for item in snapshot.applicability_decisions}


def test_repository_status_target_stage_then_scope_gates_are_authoritative(
    git_repo: Path,
) -> None:
    target = _head(git_repo)
    allowed = _record(
        git_repo,
        "allowed",
        kind=MemoryKind.BUSINESS_INVARIANT,
        scope=MemoryScope(paths=("app.py",)),
    )
    revoked = _record(
        git_repo,
        "revoked",
        kind=MemoryKind.BUSINESS_INVARIANT,
        status=RecordStatus.REVOKED,
    )
    stage_denied = _record(
        git_repo,
        "stage-denied",
        kind=MemoryKind.REVIEW_RULE,
        scope=MemoryScope(paths=("never/**",)),
    )
    run_git(git_repo, "commit", "--allow-empty", "-m", "future")
    future = _record(
        git_repo,
        "future",
        valid_from=_head(git_repo),
        kind=MemoryKind.BUSINESS_INVARIANT,
    )
    foreign = _record(
        git_repo,
        "foreign",
        valid_from=target,
        repository_key_value="f" * 64,
        kind=MemoryKind.BUSINESS_INVARIANT,
    )

    snapshot = _builder(git_repo).build(
        _selection(git_repo, head=target),
        (foreign, future, stage_denied, revoked, allowed),
        stage=RetrievalStage.INTENT_DISCOVERY,
        created_at=NOW,
    )

    assert {record.memory_id for record in snapshot.eligible_records} == {allowed.memory_id}
    decisions = _decisions(snapshot)
    assert foreign.memory_id not in decisions
    assert decisions[revoked.memory_id].applicability is Applicability.REVOKED
    assert decisions[future.memory_id].applicability is Applicability.NOT_YET_VALID
    assert decisions[stage_denied.memory_id].applicability is Applicability.OUT_OF_SCOPE
    assert decisions[stage_denied.memory_id].reason_codes == ("stage_kind_not_allowed",)


def test_repository_gate_precedes_memory_id_collision_and_input_is_bounded(
    git_repo: Path,
    monkeypatch,
) -> None:
    local = _record(git_repo, "repository-first")
    foreign = replace(local, repository_key="f" * 64)

    snapshot = _builder(git_repo).build(
        _selection(git_repo),
        (foreign, local),
        created_at=NOW,
    )

    assert tuple(record.memory_id for record in snapshot.eligible_records) == (
        local.memory_id,
    )

    monkeypatch.setattr(memory_retrieval_module, "MAX_SNAPSHOT_DECISIONS", 2)
    with pytest.raises(SnapshotBudgetExceeded, match="input catalog"):
        _builder(git_repo).build(
            _selection(git_repo),
            (
                _record(git_repo, "bounded-1"),
                _record(git_repo, "bounded-2"),
                _record(git_repo, "bounded-3"),
            ),
            created_at=NOW,
        )


def test_source_changed_missing_and_scope_trigger_never_enter_snapshot(git_repo: Path) -> None:
    gone = git_repo / "gone.py"
    gone.write_text("VALUE = 1\n", encoding="utf-8")
    run_git(git_repo, "add", "gone.py")
    run_git(git_repo, "commit", "-m", "add source")
    valid_from = _head(git_repo)
    app_source = RepositoryRangeSourceRef(
        revision=valid_from,
        path="app.py",
        line_start=1,
        line_end=2,
        content_hash=repository_range_hash(git_repo, valid_from, "app.py", 1, 2),
    )
    gone_source = RepositoryRangeSourceRef(
        revision=valid_from,
        path="gone.py",
        line_start=1,
        line_end=1,
        content_hash=repository_range_hash(git_repo, valid_from, "gone.py", 1, 1),
    )
    changed = _record(
        git_repo,
        "changed-source",
        valid_from=valid_from,
        source_refs=(app_source,),
        policies=(ValidityPolicy.SOURCE_CONTENT_HASH,),
    )
    missing = _record(
        git_repo,
        "missing-source",
        valid_from=valid_from,
        source_refs=(gone_source,),
        policies=(ValidityPolicy.SOURCE_CONTENT_HASH,),
    )
    trigger = _record(
        git_repo,
        "scope-trigger",
        valid_from=valid_from,
        policies=(ValidityPolicy.SCOPE_CHANGE_TRIGGER,),
        scope=MemoryScope(paths=("app.py",)),
    )
    (git_repo / "app.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    gone.unlink()
    run_git(git_repo, "add", "app.py", "gone.py")
    run_git(git_repo, "commit", "-m", "invalidate sources")

    snapshot = _builder(git_repo).build(
        _selection(git_repo, paths=("app.py", "gone.py")),
        (changed, missing, trigger),
        stage=RetrievalStage.REVIEWER,
        created_at=NOW,
    )

    assert snapshot.eligible_records == ()
    decisions = _decisions(snapshot)
    assert decisions[changed.memory_id].applicability is Applicability.SOURCE_CHANGED
    assert decisions[missing.memory_id].applicability is Applicability.SOURCE_MISSING
    assert decisions[trigger.memory_id].applicability is Applicability.SOURCE_CHANGED


def test_path_symbol_contract_and_language_scope_are_or_matched(git_repo: Path) -> None:
    records = (
        _record(git_repo, "path", scope=MemoryScope(paths=("src/**",))),
        _record(git_repo, "symbol", scope=MemoryScope(symbols=("pkg.mod.f",))),
        _record(git_repo, "contract", scope=MemoryScope(contracts=("auth",))),
        _record(git_repo, "language", scope=MemoryScope(languages=("python",))),
        _record(git_repo, "outside", scope=MemoryScope(paths=("docs/**",))),
    )
    snapshot = _builder(git_repo).build(
        _selection(
            git_repo,
            paths=("src/a.py",),
            symbols=("pkg.mod.f",),
            contracts=("AUTH",),
            languages=("Python",),
        ),
        records,
        stage=RetrievalStage.REVIEWER,
        created_at=NOW,
    )
    assert {record.statement for record in snapshot.eligible_records} == {
        "Memory statement for path.",
        "Memory statement for symbol.",
        "Memory statement for contract.",
        "Memory statement for language.",
    }
    assert _decisions(snapshot)[records[-1].memory_id].applicability is Applicability.OUT_OF_SCOPE


def test_typed_policy_lexical_graph_and_memory_id_ranking_are_stable(git_repo: Path) -> None:
    policy = _record(
        git_repo,
        "policy",
        statement="Unrelated hard check.",
        effect=PolicyEffect(PolicyEffectKind.REQUIRE_CHECK, "check.auth"),
    )
    lexical = _record(git_repo, "lexical", statement="Decimal decimal arithmetic invariant.")
    graph = _record(git_repo, "graph", statement="Unrelated context.")
    tie_a = _record(git_repo, "tie-a", statement="Same relevance.")
    tie_b = _record(git_repo, "tie-b", statement="Same relevance.")
    snapshot = _builder(git_repo).build(
        _selection(git_repo),
        (tie_b, graph, lexical, policy, tie_a),
        stage=RetrievalStage.REVIEWER,
        query_text="decimal",
        graph_relevance={graph.memory_id: 100},
        created_at=NOW,
    )
    ordered = [
        decision.memory_id
        for decision in snapshot.applicability_decisions
        if decision.applicability is Applicability.SELECTED
    ]
    assert ordered[0] == policy.memory_id
    assert ordered[1] == lexical.memory_id
    assert ordered[2] == graph.memory_id
    assert ordered[3:] == sorted((tie_a.memory_id, tie_b.memory_id))


def test_semantic_ranker_sees_only_eligible_and_cannot_expand_it(git_repo: Path) -> None:
    first = _record(git_repo, "semantic-first")
    second = _record(git_repo, "semantic-second")
    revoked = _record(git_repo, "semantic-revoked", status=RecordStatus.REVOKED)
    seen: list[tuple[str, ...]] = []

    def ranker(records, request):
        seen.append(tuple(record.memory_id for record in records))
        return {first.memory_id: 0, second.memory_id: 10}

    snapshot = _builder(git_repo).build(
        _selection(git_repo),
        (first, revoked, second),
        stage=RetrievalStage.REVIEWER,
        semantic_ranker=ranker,
        created_at=NOW,
    )
    selected = [
        decision.memory_id
        for decision in snapshot.applicability_decisions
        if decision.applicability is Applicability.SELECTED
    ]
    assert seen == [(first.memory_id, second.memory_id)]
    assert selected[:2] == [second.memory_id, first.memory_id]

    def escaping(records, request):
        return {first.memory_id: 0, second.memory_id: 1, revoked.memory_id: 999}

    with pytest.raises(SemanticRankerViolation):
        _builder(git_repo).build(
            _selection(git_repo),
            (first, revoked, second),
            stage=RetrievalStage.REVIEWER,
            semantic_ranker=escaping,
            created_at=NOW,
        )


def test_snapshot_copies_records_and_top_level_generations_deterministically(git_repo: Path) -> None:
    record = _record(git_repo, "immutable")
    selection = _selection(git_repo)
    first = _builder(git_repo).build(selection, (record,), created_at=NOW)
    second = _builder(git_repo).build(selection, (record,), created_at=NOW)

    assert first is not second
    assert first.eligible_records[0] is not record
    assert first.generations is not selection.generations
    assert first.snapshot_id == second.snapshot_id
    payload = first.to_dict()
    assert payload["memory_generation"] == 7
    assert payload["feedback_generation"] == 8
    assert payload["knowledge_generation"] == 9
    assert "generations" not in payload
    assert MemorySnapshot.from_dict(payload) == first


def test_ordinary_budget_omits_but_hard_policy_count_and_bytes_block(git_repo: Path) -> None:
    ordinary_a = _record(git_repo, "budget-a", statement="A" * 2_000)
    ordinary_b = _record(git_repo, "budget-b", statement="B" * 2_000)
    one_record_limits = RetrievalLimits(max_snapshot_records=1)
    snapshot = _builder(git_repo, one_record_limits).build(
        _selection(git_repo),
        (ordinary_a, ordinary_b),
        created_at=NOW,
    )
    assert len(snapshot.eligible_records) == 1
    assert sum(
        decision.applicability is Applicability.BUDGET_OMITTED
        for decision in snapshot.applicability_decisions
    ) == 1

    hard_a = _record(
        git_repo,
        "hard-a",
        effect=PolicyEffect(PolicyEffectKind.REQUIRE_CHECK, "check.a"),
    )
    hard_b = _record(
        git_repo,
        "hard-b",
        effect=PolicyEffect(PolicyEffectKind.REQUIRE_CHECK, "check.b"),
    )
    with pytest.raises(HardPolicyBudgetExceeded, match="records") as error:
        _builder(git_repo, one_record_limits).build(
            _selection(git_repo),
            (hard_a, hard_b),
            created_at=NOW,
        )
    assert error.value.blocking

    huge_hard = _record(
        git_repo,
        "hard-bytes",
        statement="policy " + "x" * 7_000,
        effect=PolicyEffect(PolicyEffectKind.REQUIRE_CHECK, "check.large"),
    )
    with pytest.raises(HardPolicyBudgetExceeded, match="bytes"):
        _builder(git_repo, RetrievalLimits(max_snapshot_bytes=2_000)).build(
            _selection(git_repo),
            (huge_hard,),
            created_at=NOW,
        )


def test_disabled_and_empty_snapshot_is_canonical_and_bounded(git_repo: Path) -> None:
    selection = _selection(git_repo)
    first = build_disabled_snapshot(selection, created_at=NOW)
    second = build_disabled_snapshot(selection, created_at=NOW)
    assert first.eligible_records == ()
    assert first.applicability_decisions == ()
    assert first.snapshot_id == second.snapshot_id
    assert len(canonical_json(first.to_dict()).encode("utf-8")) < 2_000


def test_snapshot_selector_enforces_per_call_record_and_byte_limits(git_repo: Path) -> None:
    records = tuple(_record(git_repo, f"context-{index}") for index in range(3))
    limits = RetrievalLimits(max_context_records=1, max_context_bytes=20_000)
    snapshot = _builder(git_repo, limits).build(_selection(git_repo), records, created_at=NOW)
    selection = SnapshotMemorySelector(snapshot, limits=limits).select(
        RetrievalRequest(stage=RetrievalStage.REVIEWER, paths=("app.py",))
    )
    assert len(selection.records) == 1
    assert len(selection.omitted_memory_ids) == 2
    assert selection.byte_size <= limits.max_context_bytes


def test_nested_recursive_glob_scopes_have_a_safe_deterministic_intersection(
    git_repo: Path,
) -> None:
    record = _record(
        git_repo,
        "nested-glob",
        scope=MemoryScope(paths=("src/**",)),
    )
    snapshot = _builder(git_repo).build(
        _selection(git_repo, paths=("src/payments/api.py",)),
        (record,),
        created_at=NOW,
    )

    selected = SnapshotMemorySelector(snapshot).select(
        RetrievalRequest(
            stage=RetrievalStage.REVIEWER,
            paths=("src/payments/**",),
        )
    )

    assert selected.selected_memory_ids == (record.memory_id,)


def test_empty_projection_envelope_cannot_exceed_context_or_query_byte_limit(
    git_repo: Path,
) -> None:
    record = _record(git_repo, "tiny-envelope")
    snapshot = _builder(git_repo).build(
        _selection(git_repo),
        (record,),
        created_at=NOW,
    )
    limits = RetrievalLimits(max_context_bytes=1, max_query_bytes=1)

    with pytest.raises(ProjectionBudgetExceeded, match="context projection"):
        SnapshotMemorySelector(snapshot, limits=limits).select(
            RetrievalRequest(stage=RetrievalStage.REVIEWER, paths=("app.py",))
        )

    service = SnapshotMemoryQueryService(
        snapshot,
        assignment_id="A-app",
        assignment_scope=MemoryScope(paths=("app.py",)),
        limits=limits,
    )
    with pytest.raises(ProjectionBudgetExceeded, match="query projection"):
        service.query(MemoryQuery("A-app", path="app.py"))


def test_snapshot_only_query_is_assignment_bound_and_query_bounded(git_repo: Path) -> None:
    src = _record(git_repo, "query-src", scope=MemoryScope(paths=("src/**",)))
    docs = _record(git_repo, "query-docs", scope=MemoryScope(paths=("docs/**",)))
    snapshot = _builder(git_repo).build(
        _selection(git_repo, paths=("src/a.py", "docs/a.md")),
        (src, docs),
        created_at=NOW,
    )
    limits = RetrievalLimits(max_query_results=1, max_query_calls=1)
    service = SnapshotMemoryQueryService(
        snapshot,
        assignment_id="A-src",
        assignment_scope=MemoryScope(paths=("src/**",)),
        limits=limits,
    )
    result = service.query(MemoryQuery("A-src", path="src/a.py", query_text="query"))
    assert [record.memory_id for record in result.records] == [src.memory_id]
    assert result.snapshot_id == snapshot.snapshot_id
    with pytest.raises(QueryLimitExceeded):
        service.query(MemoryQuery("A-src", path="src/a.py"))

    wrong_scope = SnapshotMemoryQueryService(
        snapshot,
        assignment_id="A-src",
        assignment_scope=MemoryScope(paths=("src/**",)),
    )
    with pytest.raises(QueryScopeViolation):
        wrong_scope.query(MemoryQuery("A-src", path="docs/a.md"))
    with pytest.raises(QueryScopeViolation):
        SnapshotMemoryQueryService(
            snapshot,
            assignment_id="A-src",
            assignment_scope=MemoryScope(paths=("src/**",)),
        ).query(MemoryQuery("A-other", path="src/a.py"))


def test_retrieval_module_has_no_pipeline_provider_cli_or_store_import() -> None:
    module = Path("src/review_agent/memory_retrieval.py")
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert not imported.intersection(
        {
            "review_agent.pipeline",
            "review_agent.provider",
            "review_agent.cli",
            "review_agent.memory_store",
        }
    )
