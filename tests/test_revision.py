from pathlib import Path

import pytest

from conftest import run_git
from review_agent.revision import RevisionResolver


def test_revision_resolver_resolves_symbolic_revisions_to_commit_shas(
    git_repo: Path,
) -> None:
    base_sha = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "app.py").write_text(
        "def add(a, b):\n    return a - b\n",
        encoding="utf-8",
    )
    run_git(git_repo, "add", "app.py")
    run_git(git_repo, "commit", "-m", "change app")
    head_sha = run_git(git_repo, "rev-parse", "HEAD")

    resolved = RevisionResolver().resolve_pair(git_repo, "HEAD~1", "HEAD")

    assert resolved.requested_base == "HEAD~1"
    assert resolved.requested_head == "HEAD"
    assert resolved.resolved_base_sha == base_sha
    assert resolved.resolved_head_sha == head_sha


def test_revision_resolver_peels_annotated_tag_to_commit(git_repo: Path) -> None:
    head_sha = run_git(git_repo, "rev-parse", "HEAD")
    run_git(git_repo, "tag", "-a", "release-v1", "-m", "release v1")

    resolved_sha = RevisionResolver().resolve_commit(git_repo, "release-v1")

    assert resolved_sha == head_sha
    assert resolved_sha != run_git(git_repo, "rev-parse", "release-v1")


def test_repository_identity_uses_git_common_directory_without_requiring_origin(
    git_repo: Path,
) -> None:
    identity = RevisionResolver().repository_identity(git_repo)

    assert identity.canonical_path == str(git_repo.resolve())
    assert Path(identity.git_common_dir).resolve() == (git_repo / ".git").resolve()
    assert identity.origin_url is None


def test_repository_identity_includes_optional_origin_url(git_repo: Path) -> None:
    origin_url = "https://example.test/acme/review-target.git"
    run_git(git_repo, "remote", "add", "origin", origin_url)

    identity = RevisionResolver().repository_identity(git_repo)

    assert identity.origin_url == origin_url


def test_revision_resolver_reports_invalid_revision(git_repo: Path) -> None:
    with pytest.raises(ValueError, match="missing-revision"):
        RevisionResolver().resolve_commit(git_repo, "missing-revision")


def test_revision_resolver_checks_commit_existence(git_repo: Path) -> None:
    head_sha = run_git(git_repo, "rev-parse", "HEAD")
    blob_sha = run_git(git_repo, "rev-parse", "HEAD:app.py")

    resolver = RevisionResolver()

    assert resolver.commit_exists(git_repo, head_sha) is True
    assert resolver.commit_exists(git_repo, blob_sha) is False
    assert resolver.commit_exists(git_repo, "0" * 40) is False
