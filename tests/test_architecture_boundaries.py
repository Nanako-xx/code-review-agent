import ast
from pathlib import Path


def test_reviewer_business_modules_do_not_import_model_provider():
    modules = [
        Path("src/review_agent/reviewer.py"),
        Path("src/review_agent/orchestrator.py"),
        Path("src/review_agent/cli.py"),
        Path("src/review_agent/agent_loop.py"),
    ]

    for module in modules:
        text = module.read_text(encoding="utf-8")
        assert "from review_agent.provider import" not in text
        assert "ModelProvider" not in text
        assert "build_provider_from_config" not in text


def test_quality_business_logic_does_not_own_subprocess_execution():
    quality = Path("src/review_agent/quality.py").read_text(encoding="utf-8")
    pipeline = Path("src/review_agent/pipeline.py").read_text(encoding="utf-8")

    assert "subprocess.Popen" not in quality
    assert "_run_bounded_process" not in quality
    assert "subprocess.Popen" not in pipeline


def test_memory_store_depends_only_on_memory_foundations_and_standard_library():
    module = Path("src/review_agent/memory_store.py")
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    review_agent_imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            review_agent_imports.update(
                alias.name
                for alias in node.names
                if alias.name.startswith("review_agent.")
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("review_agent."):
                review_agent_imports.add(node.module)

    assert review_agent_imports <= {
        "review_agent.memory_identity",
        "review_agent.memory_models",
    }
