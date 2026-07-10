from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess


@dataclass(frozen=True)
class RepositoryIdentity:
    canonical_path: str
    git_common_dir: str
    origin_url: str | None


@dataclass(frozen=True)
class ResolvedRevisions:
    requested_base: str
    requested_head: str
    resolved_base_sha: str
    resolved_head_sha: str


class RevisionResolver:
    def repository_identity(self, repo: Path) -> RepositoryIdentity:
        repository = Path(repo).resolve()
        top_level = Path(
            self._git(repository, ["rev-parse", "--show-toplevel"])
        ).resolve()
        common_raw = self._git(repository, ["rev-parse", "--git-common-dir"])
        common_path = Path(common_raw)
        if not common_path.is_absolute():
            common_path = repository / common_path
        origin_url = self._optional_git(repository, ["remote", "get-url", "origin"])
        return RepositoryIdentity(
            canonical_path=str(top_level),
            git_common_dir=str(common_path.resolve()),
            origin_url=origin_url,
        )

    def resolve_pair(
        self,
        repo: Path,
        base_revision: str,
        head_revision: str,
    ) -> ResolvedRevisions:
        return ResolvedRevisions(
            requested_base=base_revision,
            requested_head=head_revision,
            resolved_base_sha=self.resolve_commit(repo, base_revision),
            resolved_head_sha=self.resolve_commit(repo, head_revision),
        )

    def resolve_commit(self, repo: Path, revision: str) -> str:
        result = self._run_git(
            Path(repo).resolve(),
            ["rev-parse", "--verify", f"{revision}^{{commit}}"],
        )
        if result.returncode != 0:
            raise ValueError(f"revision does not resolve to a commit: {revision}")
        return result.stdout.strip()

    def commit_exists(self, repo: Path, sha: str) -> bool:
        result = self._run_git(
            Path(repo).resolve(),
            ["cat-file", "-e", f"{sha}^{{commit}}"],
        )
        return result.returncode == 0

    def _git(self, repo: Path, args: list[str]) -> str:
        result = self._run_git(repo, args)
        if result.returncode != 0:
            message = result.stderr.strip() or f"git {' '.join(args)} failed"
            raise ValueError(message)
        return result.stdout.strip()

    def _optional_git(self, repo: Path, args: list[str]) -> str | None:
        result = self._run_git(repo, args)
        if result.returncode != 0:
            return None
        value = result.stdout.strip()
        return value or None

    def _run_git(
        self,
        repo: Path,
        args: list[str],
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=repo,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
