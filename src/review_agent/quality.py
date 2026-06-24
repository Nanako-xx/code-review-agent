from __future__ import annotations

from pathlib import Path

from review_agent.models import QualityGateResult


def detect_quality_gates(repo_path: Path) -> list[str]:
    has_python = any(path.suffix == ".py" for path in repo_path.rglob("*.py"))
    return ["python_compile"] if has_python else []


def run_python_compile_gate(repo_path: Path) -> QualityGateResult:
    python_files = [path for path in repo_path.rglob("*.py") if ".git" not in path.parts]
    for path in python_files:
        try:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")
        except SyntaxError as exc:
            return QualityGateResult(
                name="python_compile",
                status="failed",
                command=["python", "-c", "compile(source, filename, 'exec')"],
                summary=f"{type(exc).__name__}: {exc}",
            )

    return QualityGateResult(
        name="python_compile",
        status="passed",
        command=["python", "-c", "compile(source, filename, 'exec')"],
        summary=f"Compiled {len(python_files)} Python files",
    )
