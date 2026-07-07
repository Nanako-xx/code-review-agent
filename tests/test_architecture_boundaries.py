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
