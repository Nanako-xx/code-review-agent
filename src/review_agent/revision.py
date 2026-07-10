from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import subprocess
from urllib.parse import urlsplit, urlunsplit


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
        origin_url = _sanitize_origin_url(
            self._optional_git(repository, ["remote", "get-url", "origin"])
        )
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
        try:
            result = self._run_git(
                Path(repo).resolve(),
                ["rev-parse", "--verify", f"{revision}^{{commit}}"],
            )
        except RuntimeError as error:
            raise RuntimeError(f"unable to resolve revision {revision}: {error}") from error
        if result.returncode != 0:
            reason = result.stderr.strip() or result.stdout.strip() or "git rev-parse failed"
            raise ValueError(
                f"revision does not resolve to a commit: {revision}; "
                f"Git reported: {reason}"
            )
        resolved = result.stdout.strip()
        if not resolved:
            raise ValueError(
                f"revision does not resolve to a commit: {revision}; "
                "Git returned an empty object ID"
            )
        return resolved

    def commit_exists(self, repo: Path, sha: str) -> bool:
        repository = Path(repo).resolve()
        object_format = self._object_format(repository)
        object_id_length = self._object_id_length(object_format)
        if not re.fullmatch(rf"[0-9a-fA-F]{{{object_id_length}}}", sha):
            raise ValueError(
                f"expected a full {object_format} object ID "
                f"({object_id_length} hexadecimal characters), got: {sha}"
            )

        result = self._run_git(
            repository,
            ["cat-file", "--batch-check"],
            input_text=f"{sha}\n",
        )
        if result.returncode != 0:
            reason = result.stderr.strip() or result.stdout.strip() or "git cat-file failed"
            raise ValueError(f"unable to inspect object {sha}: {reason}")

        fields = result.stdout.strip().split()
        if len(fields) == 2 and fields[1] == "missing":
            return False
        if len(fields) == 3 and fields[0].casefold() == sha.casefold():
            return fields[1] == "commit"
        raise ValueError(
            f"unable to inspect object {sha}: unexpected git cat-file response: "
            f"{result.stdout.strip()!r}"
        )

    def _object_format(self, repo: Path) -> str:
        result = self._run_git(repo, ["rev-parse", "--show-object-format=storage"])
        if result.returncode != 0:
            reason = result.stderr.strip() or result.stdout.strip() or "git rev-parse failed"
            raise ValueError(
                f"unable to determine repository object format for {repo}: {reason}"
            )
        object_format = result.stdout.strip().casefold()
        if not object_format:
            raise ValueError(
                f"unable to determine repository object format for {repo}: "
                "Git returned an empty value"
            )
        return object_format

    def _object_id_length(self, object_format: str) -> int:
        try:
            return hashlib.new(object_format).digest_size * 2
        except ValueError as error:
            raise ValueError(f"unsupported Git object format: {object_format}") from error

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
        *,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=repo,
                input=input_text,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as error:
            raise RuntimeError(f"failed to execute Git in {repo}: {error}") from error


def _sanitize_origin_url(origin_url: str | None) -> str | None:
    if origin_url is None:
        return None
    origin = origin_url.strip()
    if not origin or any(character in origin for character in "\x00\r\n\t"):
        return None

    if "://" not in origin:
        return _sanitize_scp_or_local_origin(origin)

    try:
        parsed = urlsplit(origin)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return None
    if not parsed.scheme or not hostname:
        return None

    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    authority = hostname if port is None else f"{hostname}:{port}"
    sanitized = urlunsplit(
        (parsed.scheme.casefold(), authority, parsed.path, "", "")
    )
    return sanitized or None


def _sanitize_scp_or_local_origin(origin: str) -> str | None:
    sanitized = origin.split("#", 1)[0].split("?", 1)[0]
    if not sanitized:
        return None

    scp_match = re.fullmatch(
        r"(?:(?P<userinfo>[^@/:]+)@)?(?P<host>\[[^\]]+\]|[^/:]+):(?P<path>.+)",
        sanitized,
    )
    if scp_match is None:
        return sanitized
    return f"{scp_match.group('host')}:{scp_match.group('path')}"
