from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import urlsplit, urlunsplit


_SAFE_ORIGIN_URL_SCHEMES = frozenset({"git", "http", "https", "ssh"})
_SCP_ORIGIN_PATTERN = re.compile(
    r"^(?:(?P<userinfo>[A-Za-z0-9._~-]+)@)?"
    r"(?P<host>[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?|"
    r"\[[0-9A-Fa-f:.%]+\])"
    r":(?P<path>[A-Za-z0-9._~+/@%=-]+)$"
)


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
        origin_url = sanitize_origin_url(
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


def sanitize_origin_url(origin_url: str | None) -> str | None:
    """Return a credential-free origin suitable for local identity metadata.

    Unsafe or ambiguous remote helper syntax is omitted instead of being echoed
    into manifests or error messages.
    """

    if origin_url is None:
        return None
    origin = origin_url.strip()
    if not origin or any(character.isspace() for character in origin):
        return None

    if "://" not in origin:
        return _sanitize_scp_like_origin(origin)

    try:
        parsed = urlsplit(origin)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return None
    scheme = parsed.scheme.casefold()
    if scheme not in _SAFE_ORIGIN_URL_SCHEMES or not hostname:
        return None

    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    authority = hostname if port is None else f"{hostname}:{port}"
    sanitized = urlunsplit((scheme, authority, parsed.path, "", ""))
    return sanitized or None


def normalize_repository_origin(origin_url: str | None) -> str | None:
    """Normalize a sanitized origin for deterministic identity hashing."""

    sanitized = sanitize_origin_url(origin_url)
    if sanitized is None:
        return None
    if "://" not in sanitized:
        host, path = sanitized.split(":", 1)
        normalized_path = path.rstrip("/")
        return f"{host.casefold()}:{normalized_path}"

    parsed = urlsplit(sanitized)
    normalized_path = parsed.path.rstrip("/")
    return urlunsplit(
        (
            parsed.scheme.casefold(),
            parsed.netloc.casefold(),
            normalized_path,
            "",
            "",
        )
    )


def normalize_repository_identity_path(path: str | Path) -> str:
    """Return the local, canonical comparison form used in repository keys."""

    raw_path = str(path)
    if not raw_path or "\0" in raw_path:
        raise ValueError("repository identity path must be a non-empty absolute path")
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        raise ValueError("repository identity path must be an absolute path")
    try:
        resolved = candidate.resolve(strict=False)
    except OSError as error:
        raise ValueError("unable to canonicalize repository identity path") from error
    normalized = os.path.normcase(os.path.normpath(str(resolved)))
    return normalized.replace("\\", "/")


def _sanitize_scp_like_origin(origin: str) -> str | None:
    if "?" in origin or "#" in origin:
        return None
    if re.match(r"^[A-Za-z]:[\\/]", origin):
        return None
    scp_match = _SCP_ORIGIN_PATTERN.fullmatch(origin)
    if scp_match is None:
        return None
    return f"{scp_match.group('host')}:{scp_match.group('path')}"
