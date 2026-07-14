from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import os
from pathlib import Path
import re
from typing import Any, Mapping

from review_agent.git_repo import ChangeSummary, collect_change_summary
from review_agent.revision import RepositoryIdentity, ResolvedRevisions
from review_agent.session import RevisionChangeKind, SessionManifest


_GIT_OBJECT_ID_PATTERN = re.compile(r"^(?:[0-9A-Fa-f]{40}|[0-9A-Fa-f]{64})$")


@dataclass(frozen=True)
class IncrementalPriorityMap:
    from_revision: str
    to_revision: str
    changed_files: list[str]
    diff_stat: str
    diff_excerpt: list[str]

    def __post_init__(self) -> None:
        for field_name in ("from_revision", "to_revision"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not _GIT_OBJECT_ID_PATTERN.fullmatch(
                value
            ):
                raise ValueError(
                    f"incremental priority {field_name} must be a full Git object ID"
                )
        if len(self.from_revision) != len(self.to_revision):
            raise ValueError(
                "incremental priority revisions must use the same object ID format"
            )
        if not isinstance(self.diff_stat, str):
            raise ValueError("incremental priority diff_stat must be a string")
        _string_list(self.changed_files, "changed_files")
        _string_list(self.diff_excerpt, "diff_excerpt")


def classify_revision_change(
    parent: SessionManifest,
    current: ResolvedRevisions,
) -> RevisionChangeKind:
    base_changed = (
        parent.revisions.resolved_base_sha.casefold()
        != current.resolved_base_sha.casefold()
    )
    head_changed = (
        parent.revisions.resolved_head_sha.casefold()
        != current.resolved_head_sha.casefold()
    )
    if base_changed and head_changed:
        return RevisionChangeKind.BASE_AND_HEAD_MOVED
    if base_changed:
        return RevisionChangeKind.BASE_MOVED
    if head_changed:
        return RevisionChangeKind.HEAD_MOVED
    return RevisionChangeKind.INITIAL


def deterministic_child_review_id(
    *,
    repository: RepositoryIdentity,
    parent_review_id: str,
    revisions: ResolvedRevisions,
) -> str:
    common_dir = os.path.normcase(
        os.path.normpath(str(Path(repository.git_common_dir).resolve()))
    )
    seed = "\0".join(
        [
            common_dir,
            parent_review_id,
            revisions.resolved_base_sha.casefold(),
            revisions.resolved_head_sha.casefold(),
        ]
    )
    return f"review-{sha256(seed.encode('utf-8')).hexdigest()[:12]}"


def build_incremental_priority_map(
    repo: Path,
    *,
    from_revision: str,
    to_revision: str,
) -> IncrementalPriorityMap:
    summary = collect_change_summary(repo, from_revision, to_revision)
    return build_incremental_priority_map_from_summary(
        summary,
        from_revision=from_revision,
        to_revision=to_revision,
    )


def build_incremental_priority_map_from_summary(
    summary: ChangeSummary,
    *,
    from_revision: str,
    to_revision: str,
) -> IncrementalPriorityMap:
    if (
        summary.base_revision.casefold() != from_revision.casefold()
        or summary.head_revision.casefold() != to_revision.casefold()
    ):
        raise ValueError(
            "incremental ChangeSummary revisions must match the priority range"
        )
    return IncrementalPriorityMap(
        from_revision=from_revision,
        to_revision=to_revision,
        changed_files=list(summary.changed_files),
        diff_stat=summary.diff_stat,
        diff_excerpt=list(summary.diff_excerpt),
    )


def incremental_priority_to_dict(
    priority: IncrementalPriorityMap,
) -> dict[str, Any]:
    return asdict(priority)


def incremental_priority_from_dict(
    payload: Mapping[str, Any],
) -> IncrementalPriorityMap:
    expected = {
        "from_revision",
        "to_revision",
        "changed_files",
        "diff_stat",
        "diff_excerpt",
    }
    missing = expected - set(payload)
    if missing:
        raise ValueError(
            "incremental priority map is missing: " + ", ".join(sorted(missing))
        )
    unexpected = set(payload) - expected
    if unexpected:
        raise ValueError(
            "incremental priority map contains unsupported fields: "
            + ", ".join(sorted(str(name) for name in unexpected))
        )
    for field_name in ("from_revision", "to_revision", "diff_stat"):
        if not isinstance(payload[field_name], str):
            raise ValueError(f"incremental priority {field_name} must be a string")
    changed_files = _string_list(payload["changed_files"], "changed_files")
    diff_excerpt = _string_list(payload["diff_excerpt"], "diff_excerpt")
    return IncrementalPriorityMap(
        from_revision=payload["from_revision"],
        to_revision=payload["to_revision"],
        changed_files=changed_files,
        diff_stat=payload["diff_stat"],
        diff_excerpt=diff_excerpt,
    )


def _string_list(value: object, field_name: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"incremental priority {field_name} must be a list of strings")
    return list(value)
