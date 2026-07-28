from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import review_agent_eval.cli as cli_module
from review_agent_eval.analysis_artifacts import AnalysisArtifactStore
from review_agent_eval.artifacts import ArtifactSecurityError


PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "review_agent_eval"
ANALYSIS_MODULES = (
    "analysis_artifacts.py",
    "analysis_exports.py",
    "statistics.py",
    "comparison.py",
    "calibration.py",
    "gates.py",
)
FORBIDDEN_MODULE_PREFIXES = (
    "review_agent.runtime",
    "review_agent.session",
    "review_agent.memory",
    "review_agent.risk",
    "review_agent.orchestrator",
    "review_agent.repository",
    "review_agent_eval.adapters.agent_factory",
    "review_agent_eval.adapters.model_adapter",
)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                prefix = "review_agent_eval."
                result.add(prefix + (node.module or ""))
            elif node.module:
                result.add(node.module)
    return result


def test_task15_analysis_modules_do_not_import_product_runtime_or_network() -> None:
    imported = {
        name: _imports(PACKAGE_ROOT / name)
        for name in ANALYSIS_MODULES
    }

    offenders = {
        name: sorted(
            module
            for module in modules
            if module.startswith(FORBIDDEN_MODULE_PREFIXES)
        )
        for name, modules in imported.items()
    }
    assert {name: values for name, values in offenders.items() if values} == {}


@pytest.mark.parametrize(
    "handler",
    (
        cli_module._handle_compare,
        cli_module._handle_calibrate_export,
        cli_module._handle_calibrate_import_labels,
        cli_module._handle_calibrate_score,
        cli_module._handle_gate_prepare,
        cli_module._handle_gate_evaluate,
    ),
)
def test_analysis_handlers_have_no_agent_judge_or_acquisition_composition(
    handler: object,
) -> None:
    source = inspect.getsource(handler)
    for forbidden in (
        "_agent_adapter(",
        "_judge_for(",
        "_prepare_repository_targets(",
        "build_agent_adapter_factory",
        "build_judge_adapter_factory",
        "RepositoryMode.ACQUIRE",
        "openai-compatible",
    ):
        assert forbidden not in source


def test_analysis_root_rejects_explicit_traversal_before_creation(
    tmp_path: Path,
) -> None:
    escaped = tmp_path / "controlled" / ".." / "escaped-analysis"

    with pytest.raises(ArtifactSecurityError, match="traversal"):
        AnalysisArtifactStore(escaped)

    assert not (tmp_path / "escaped-analysis").exists()


def test_analysis_root_cannot_alias_or_descend_from_run_store(
    tmp_path: Path,
) -> None:
    for path in (
        tmp_path / ".eval-runs",
        tmp_path / ".eval-runs" / ".eval-analyses",
        tmp_path / ".EVAL-RUNS " / "analysis",
    ):
        with pytest.raises((ValueError, ArtifactSecurityError)):
            AnalysisArtifactStore(path)
