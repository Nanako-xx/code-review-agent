from pathlib import Path
import json

from review_agent.checkpoint import CheckpointStore
from review_agent.models import IntentStatus, RiskAssessment, RiskLevel
from review_agent.reporting import render_markdown_report


def test_checkpoint_store_writes_json_and_jsonl(tmp_path: Path):
    store = CheckpointStore(tmp_path, review_id="review-1")

    store.write_json("request.json", {"base": "main", "head": "HEAD"})
    store.append_jsonl("evidence.jsonl", {"evidence_id": "ev_1", "summary": "changed app.py"})

    assert json.loads((tmp_path / ".review-agent" / "runs" / "review-1" / "request.json").read_text(encoding="utf-8")) == {
        "base": "main",
        "head": "HEAD",
    }
    assert "ev_1" in (tmp_path / ".review-agent" / "runs" / "review-1" / "evidence.jsonl").read_text(encoding="utf-8")


def test_checkpoint_store_serializes_enum_values(tmp_path: Path):
    store = CheckpointStore(tmp_path, review_id="review-1")

    store.write_json("intent.json", {"status": IntentStatus.PARTIAL})

    assert json.loads((tmp_path / ".review-agent" / "runs" / "review-1" / "intent.json").read_text(encoding="utf-8")) == {
        "status": "partial",
    }


def test_markdown_report_contains_risk_and_uncertainties():
    assessment = RiskAssessment(
        level=RiskLevel.HIGH,
        dimensions={},
        reasons=["sensitive path changed: auth/session.py"],
        evidence_refs=[],
        unknowns=["user did not provide declared intent"],
        suggested_focus=["caller compatibility"],
    )

    report = render_markdown_report(
        review_id="review-1",
        base_revision="main",
        head_revision="HEAD",
        risk_assessment=assessment,
        changed_files=["auth/session.py"],
    )

    assert "# Review Brief" in report
    assert "Risk level: high" in report
    assert "user did not provide declared intent" in report
