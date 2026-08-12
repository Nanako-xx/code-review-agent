from __future__ import annotations

from pathlib import Path

import pytest

from review_agent.resume import (
    LegacySessionUnsupportedError,
    diagnose_legacy_session,
    require_v6_resume_from_legacy,
)


@pytest.mark.parametrize("version", ("v1", "v2", "v3", "v4"))
def test_legacy_session_is_diagnostic_only_for_v6_product_resume(
    version: str,
) -> None:
    run_dir = Path(__file__).parent / "fixtures" / "sessions" / version

    diagnostic = diagnose_legacy_session(run_dir)

    assert diagnostic.schema_version == int(version[1:])
    assert diagnostic.status
    assert diagnostic.current_phase
    assert all(
        item.name and item.schema and item.path
        for item in diagnostic.artifacts
    )
    with pytest.raises(LegacySessionUnsupportedError) as caught:
        require_v6_resume_from_legacy(run_dir)
    assert caught.value.diagnostic == diagnostic
    assert "read-only" in str(caught.value)
