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
class RepositoryLayout:
    """Canonical worktree and Git-directory locations for one common dir."""

    git_common_dir: str
    worktree_paths: tuple[str, ...]
    git_dirs: tuple[str, ...]


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
            self._git_path(repository, ["rev-parse", "--show-toplevel"])
        ).resolve()
        common_raw = self._git_path(
            repository,
            ["rev-parse", "--git-common-dir"],
        )
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

    def repository_layout(self, repo: Path) -> RepositoryLayout:
        """Enumerate all worktrees and real Git dirs sharing one common dir.

        ``--porcelain -z`` avoids Git's path quoting and newline ambiguity.  A
        live worktree's real Git directory is resolved by Git itself; linked
        worktree administrative directories are also enumerated from the
        common dir so stale/prunable entries remain protected.
        """

        repository = Path(repo).resolve()
        identity = self.repository_identity(repository)
        common_dir = Path(identity.git_common_dir).resolve()
        worktree_result = self._run_git_bytes(
            repository,
            ["worktree", "list", "--porcelain", "-z"],
        )
        if worktree_result.returncode != 0:
            raise ValueError("unable to enumerate repository worktrees")
        worktree_paths = _parse_worktree_paths(worktree_result.stdout)
        current_path = str(Path(identity.canonical_path).resolve())
        if current_path not in worktree_paths:
            raise ValueError("current repository is missing from Git worktree metadata")

        git_dirs = {str(common_dir)}
        for worktree_value in worktree_paths:
            worktree = Path(worktree_value)
            if not worktree.is_dir():
                continue
            try:
                git_dir_value = self._git_path(
                    worktree,
                    ["rev-parse", "--absolute-git-dir"],
                )
            except ValueError:
                raise ValueError("unable to enumerate repository Git directories")
            git_dir = Path(git_dir_value)
            if not git_dir.is_absolute():
                git_dir = worktree / git_dir
            git_dirs.add(str(git_dir.resolve(strict=False)))

        administrative_root = common_dir / "worktrees"
        if administrative_root.exists() or administrative_root.is_symlink():
            if administrative_root.is_symlink() or not administrative_root.is_dir():
                raise ValueError("repository worktree metadata layout is invalid")
            try:
                administrative_entries = tuple(administrative_root.iterdir())
            except OSError as error:
                raise ValueError(
                    "unable to enumerate repository worktree metadata"
                ) from error
            for entry in administrative_entries:
                git_dirs.add(str(entry.resolve(strict=False)))

        return RepositoryLayout(
            git_common_dir=str(common_dir),
            worktree_paths=tuple(
                sorted(worktree_paths, key=normalize_repository_identity_path)
            ),
            git_dirs=tuple(sorted(git_dirs, key=normalize_repository_identity_path)),
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

    def is_ancestor(
        self,
        repo: Path,
        ancestor_sha: str,
        descendant_sha: str,
    ) -> bool:
        """Return whether two exact commit IDs share the authorized ancestry.

        Symbolic names and abbreviated IDs are deliberately rejected.  Callers
        that make a trust decision from lineage must not let Git resolve a name
        through mutable refs or replacement objects.
        """

        repository = Path(repo).resolve()
        object_format = self._object_format(repository)
        object_id_length = self._object_id_length(object_format)
        object_id_pattern = rf"[0-9a-fA-F]{{{object_id_length}}}"
        for label, value in (
            ("ancestor", ancestor_sha),
            ("descendant", descendant_sha),
        ):
            if not isinstance(value, str) or not re.fullmatch(
                object_id_pattern,
                value,
            ):
                raise ValueError(
                    f"{label} must be a full {object_format} object ID"
                )
            if not self.commit_exists(repository, value):
                raise ValueError(f"{label} commit does not exist")

        result = self._run_git(
            repository,
            ["merge-base", "--is-ancestor", ancestor_sha, descendant_sha],
        )
        if result.returncode == 0:
            return True
        if result.returncode == 1:
            return False
        reason = result.stderr.strip() or result.stdout.strip() or "git merge-base failed"
        raise ValueError(f"unable to inspect commit ancestry: {reason}")

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

    def _git_path(self, repo: Path, args: list[str]) -> str:
        result = self._run_git_bytes(repo, args)
        if result.returncode != 0:
            raise ValueError("Git path query failed")
        return _decode_git_path_output(result.stdout)

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
                ["git", "--no-replace-objects", *args],
                cwd=repo,
                env=sanitized_git_environment(),
                input=input_text,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as error:
            raise RuntimeError(f"failed to execute Git in {repo}: {error}") from error

    def _run_git_bytes(
        self,
        repo: Path,
        args: list[str],
    ) -> subprocess.CompletedProcess[bytes]:
        """Run a Git command whose protocol contains filesystem path bytes."""

        try:
            return subprocess.run(
                ["git", "--no-replace-objects", *args],
                cwd=repo,
                env=sanitized_git_environment(),
                text=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as error:
            raise RuntimeError(f"failed to execute Git in {repo}: {error}") from error


def sanitized_git_environment() -> dict[str, str]:
    """Return an environment that cannot redirect local Git object authority.

    Git honors a broad family of inherited ``GIT_*`` variables for repository,
    worktree, namespace, object database, replacement-ref, and config routing.
    Source validation always derives those locations from its explicit ``cwd``,
    so inherited Git controls are removed wholesale and only fail-closed local
    behavior is added back.
    """

    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("GIT_")
    }
    environment.update(
        {
            "GIT_LITERAL_PATHSPECS": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return environment


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


def _parse_worktree_paths(output: bytes) -> tuple[str, ...]:
    if not isinstance(output, bytes):
        raise ValueError("Git returned invalid worktree metadata")
    if not output:
        raise ValueError("Git returned empty worktree metadata")

    paths: dict[str, str] = {}
    for raw_record in output.split(b"\0\0"):
        if not raw_record:
            continue
        fields = raw_record.split(b"\0")
        if not fields or not fields[0].startswith(b"worktree "):
            raise ValueError("Git returned invalid worktree metadata")
        raw_path = fields[0][len(b"worktree ") :]
        if not raw_path or b"\0" in raw_path:
            raise ValueError("Git returned invalid worktree metadata")
        candidate = Path(os.fsdecode(raw_path))
        if not candidate.is_absolute():
            raise ValueError("Git returned a non-absolute worktree path")
        try:
            canonical = str(candidate.resolve(strict=False))
            normalized = normalize_repository_identity_path(canonical)
        except (OSError, ValueError) as error:
            raise ValueError("unable to canonicalize Git worktree metadata") from error
        if normalized in paths:
            raise ValueError("Git returned duplicate worktree metadata")
        paths[normalized] = canonical

    if not paths:
        raise ValueError("Git returned empty worktree metadata")
    return tuple(paths.values())


def _decode_git_path_output(output: bytes) -> str:
    if not isinstance(output, bytes) or not output or b"\0" in output:
        raise ValueError("Git returned invalid path metadata")
    value = output[:-1] if output.endswith(b"\n") else output
    if value.endswith(b"\r"):
        value = value[:-1]
    if not value:
        raise ValueError("Git returned empty path metadata")
    return os.fsdecode(value)


def _sanitize_scp_like_origin(origin: str) -> str | None:
    if "?" in origin or "#" in origin:
        return None
    if re.match(r"^[A-Za-z]:[\\/]", origin):
        return None
    scp_match = _SCP_ORIGIN_PATTERN.fullmatch(origin)
    if scp_match is None:
        return None
    return f"{scp_match.group('host')}:{scp_match.group('path')}"
