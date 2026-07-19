"""Module entry point for ``python -m review_agent_eval``."""

from .cli import main


if __name__ == "__main__":  # pragma: no cover - exercised by subprocess tests
    raise SystemExit(main())
