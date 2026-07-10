from pathlib import Path
import subprocess

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


@pytest.mark.parametrize(
    ("origin_url", "expected"),
    [
        (
            "https://user:token@example.test/acme/review-target.git",
            "https://example.test/acme/review-target.git",
        ),
        (
            "https://example.test/acme/review-target.git?access_token=secret#credential",
            "https://example.test/acme/review-target.git",
        ),
    ],
)
def test_repository_identity_sanitizes_sensitive_origin_url_components(
    git_repo: Path,
    origin_url: str,
    expected: str,
) -> None:
    run_git(git_repo, "remote", "add", "origin", origin_url)

    identity = RevisionResolver().repository_identity(git_repo)

    assert identity.origin_url == expected
    assert "token" not in identity.origin_url
    assert "secret" not in identity.origin_url


def test_repository_identity_omits_origin_that_cannot_be_safely_expressed(
    git_repo: Path,
) -> None:
    run_git(
        git_repo,
        "remote",
        "add",
        "origin",
        "https://user:token@/review-target.git?access_token=secret",
    )

    identity = RevisionResolver().repository_identity(git_repo)

    assert identity.origin_url is None


@pytest.mark.parametrize(
    "origin_url",
    [
        "ext::sshpass -p supersecret ssh example.test %S repo",
        "ext::invoke-secret-helper",
        "ssh example.test repo-with-supersecret",
        "opaque-origin-alias",
    ],
)
def test_repository_identity_omits_remote_helpers_and_unknown_origin_formats(
    git_repo: Path,
    origin_url: str,
) -> None:
    run_git(git_repo, "remote", "add", "origin", origin_url)

    identity = RevisionResolver().repository_identity(git_repo)

    assert identity.origin_url is None
    assert "supersecret" not in repr(identity)


def test_repository_identity_sanitizes_strict_scp_like_origin(
    git_repo: Path,
) -> None:
    run_git(
        git_repo,
        "remote",
        "add",
        "origin",
        "git@example.test:acme/review-target.git",
    )

    identity = RevisionResolver().repository_identity(git_repo)

    assert identity.origin_url == "example.test:acme/review-target.git"


def test_revision_resolver_reports_invalid_revision(git_repo: Path) -> None:
    with pytest.raises(ValueError, match="missing-revision") as captured:
        RevisionResolver().resolve_commit(git_repo, "missing-revision")

    assert "fatal:" in str(captured.value)


def test_revision_resolver_checks_commit_existence(git_repo: Path) -> None:
    head_sha = run_git(git_repo, "rev-parse", "HEAD")
    blob_sha = run_git(git_repo, "rev-parse", "HEAD:app.py")

    resolver = RevisionResolver()

    assert resolver.commit_exists(git_repo, head_sha) is True
    assert resolver.commit_exists(git_repo, blob_sha) is False
    assert resolver.commit_exists(git_repo, "0" * 40) is False


def test_commit_exists_rejects_revision_names_and_abbreviated_object_ids(
    git_repo: Path,
) -> None:
    head_sha = run_git(git_repo, "rev-parse", "HEAD")
    branch = run_git(git_repo, "branch", "--show-current")
    run_git(git_repo, "tag", "release-v1")

    resolver = RevisionResolver()

    for candidate in ("HEAD", branch, "release-v1", head_sha[:12]):
        with pytest.raises(ValueError, match="full sha1 object ID"):
            resolver.commit_exists(git_repo, candidate)


def test_commit_exists_raises_for_invalid_repository(tmp_path: Path) -> None:
    not_a_repository = tmp_path / "not-a-repository"
    not_a_repository.mkdir()

    with pytest.raises(ValueError, match="object format") as captured:
        RevisionResolver().commit_exists(not_a_repository, "0" * 40)

    assert "not a git repository" in str(captured.value).casefold()


def test_commit_exists_raises_when_git_cannot_read_the_object_database(
    git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    head_sha = run_git(git_repo, "rev-parse", "HEAD")
    real_run = subprocess.run

    def run_with_object_read_failure(
        command: list[str],
        *args: object,
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        if "cat-file" in command:
            return subprocess.CompletedProcess(
                command,
                128,
                stdout="",
                stderr="fatal: permission denied reading object database",
            )
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", run_with_object_read_failure)

    with pytest.raises(ValueError, match="permission denied reading object database"):
        RevisionResolver().commit_exists(git_repo, head_sha)


def test_commit_exists_supports_sha256_repository_object_ids(tmp_path: Path) -> None:
    repo = tmp_path / "sha256-repo"
    repo.mkdir()
    run_git(repo, "init", "--object-format=sha256")
    run_git(repo, "config", "user.email", "review-agent@example.test")
    run_git(repo, "config", "user.name", "Review Agent")
    (repo / "app.py").write_text("print('sha256')\n", encoding="utf-8")
    run_git(repo, "add", "app.py")
    run_git(repo, "commit", "-m", "initial sha256 commit")
    head_sha = run_git(repo, "rev-parse", "HEAD")

    resolver = RevisionResolver()

    assert len(head_sha) == 64
    assert resolver.commit_exists(repo, head_sha) is True
    assert resolver.commit_exists(repo, "0" * 64) is False
    with pytest.raises(ValueError, match="full sha256 object ID"):
        resolver.commit_exists(repo, head_sha[:16])
