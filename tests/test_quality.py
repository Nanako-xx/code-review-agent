from pathlib import Path

from review_agent.quality import detect_quality_gates, run_python_compile_gate


def test_detect_python_compile_for_python_repo(tmp_path: Path):
    (tmp_path / "app.py").write_text("print('hello')\n", encoding="utf-8")

    gates = detect_quality_gates(tmp_path)

    assert gates == ["python_compile"]


def test_python_compile_gate_passes_for_valid_python(tmp_path: Path):
    (tmp_path / "app.py").write_text("def ok():\n    return 1\n", encoding="utf-8")

    result = run_python_compile_gate(tmp_path)

    assert result.name == "python_compile"
    assert result.status == "passed"


def test_python_compile_gate_passes_for_utf8_bom_python(tmp_path: Path):
    (tmp_path / "app.py").write_text("\ufeffdef ok():\n    return 1\n", encoding="utf-8")

    result = run_python_compile_gate(tmp_path)

    assert result.status == "passed"


def test_python_compile_gate_fails_for_invalid_python(tmp_path: Path):
    (tmp_path / "bad.py").write_text("def broken(:\n    return 1\n", encoding="utf-8")

    result = run_python_compile_gate(tmp_path)

    assert result.status == "failed"
    assert "SyntaxError" in result.summary
