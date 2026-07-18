"""Deterministic, replay-bound location matching for review evaluation.

Location matching is deliberately narrower than issue matching.  A zero
location score says nothing about semantic/root-cause eligibility.

The v1 integer score bands are fixed by :data:`LOCATION_MATCH_POLICY_VERSION`:

* exact range: ``1_000_000``;
* overlapping range: ``900_000``;
* non-overlapping range within policy: ``800_000 - line_distance``;
* no location match: ``0``.

Paths use the sole repository path policy exported by ``repository.py``.
File extents may be built from a verified ``PreparedRepositoryReplay`` and
never inspect a Trial workspace.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from enum import Enum
import re
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .models import (
    DiffSide,
    MAX_IDENTIFIER_CHARS,
    MAX_LINE_NUMBER,
    MAX_TRUTH_FINDINGS,
    MAX_TRUTH_LOCATIONS,
    SubmissionFinding,
    TruthLocation,
    _strict_json_loads,
    canonical_json,
    canonical_sha256,
    stable_id,
)
from .repository import (
    MAX_GIT_BLOB_BYTES,
    MAX_MATERIALIZED_FILES,
    MAX_PATH_BYTES,
    MAX_PATH_COMPONENT_BYTES,
    MAX_PATH_DEPTH,
    PreparedRepositoryReplay,
    RepositoryLimitError,
    RepositoryPolicyError,
    canonical_repository_path,
)


LOCATION_MATCH_POLICY_VERSION = "location_match_policy_v1"
DEFAULT_MAX_LINE_DISTANCE = 5
MAX_CONFIGURED_LINE_DISTANCE = 100_000

EXACT_LOCATION_SCORE = 1_000_000
OVERLAP_LOCATION_SCORE = 900_000
DISTANCE_LOCATION_SCORE_CEILING = 800_000
NO_LOCATION_MATCH_SCORE = 0

# One side is one repository tree, so it uses the repository's tree-file cap.
MAX_SIDE_FILE_EXTENTS = MAX_MATERIALIZED_FILES
MAX_SIDE_PATHS = MAX_SIDE_FILE_EXTENTS  # compatibility constant
MAX_FILE_LINE_COUNT = 1_000_000
MAX_LOCATION_TARGETS = MAX_TRUTH_FINDINGS * MAX_TRUTH_LOCATIONS

MAX_LOCATION_MATCH_RESULT_JSON_BYTES = 16 * 1024
MAX_LOCATION_CANDIDATE_JSON_BYTES = 32 * 1024
MAX_LOCATION_CANDIDATES_JSON_BYTES = 64 * 1024 * 1024

_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_VCS_METADATA_LABELS = frozenset(
    {".git", ".hg", ".svn", ".gitmodules", ".lfsconfig"}
)
_OTHER_SPLITLINE_BOUNDARIES = frozenset(
    ("\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029")
)


class LocationMatchReason(str, Enum):
    """Stable v1 reason codes, ordered from path validation to line scoring."""

    GENERATED_PATH_MISSING = "generated_path_missing"
    GENERATED_PATH_TYPE_INVALID = "generated_path_type_invalid"
    GENERATED_PATH_EMPTY = "generated_path_empty"
    GENERATED_PATH_INVALID_UNICODE = "generated_path_invalid_unicode"
    GENERATED_PATH_TOO_LONG = "generated_path_too_long"
    GENERATED_PATH_NUL = "generated_path_nul"
    GENERATED_PATH_CONTROL = "generated_path_control"
    GENERATED_PATH_WINDOWS_DRIVE = "generated_path_windows_drive"
    GENERATED_PATH_UNC = "generated_path_unc"
    GENERATED_PATH_ABSOLUTE = "generated_path_absolute"
    GENERATED_PATH_BACKSLASH = "generated_path_backslash"
    GENERATED_PATH_EMPTY_COMPONENT = "generated_path_empty_component"
    GENERATED_PATH_DOT_COMPONENT = "generated_path_dot_component"
    GENERATED_PATH_DOT_DOT_COMPONENT = "generated_path_dot_dot_component"
    GENERATED_PATH_COMPONENT_TOO_LONG = "generated_path_component_too_long"
    GENERATED_PATH_TOO_DEEP = "generated_path_too_deep"
    GENERATED_PATH_VCS_METADATA = "generated_path_vcs_metadata"
    GENERATED_PATH_REPOSITORY_POLICY = "generated_path_repository_policy"
    TRUTH_PATH_INVALID = "truth_path_invalid"
    PATH_CASEFOLD_NFC_COLLISION = "path_casefold_nfc_collision"
    PATH_MISMATCH = "path_mismatch"
    PATH_UNAVAILABLE_ON_SIDE = "path_unavailable_on_side"
    TRUTH_PATH_UNAVAILABLE_ON_SIDE = "truth_path_unavailable_on_side"

    GENERATED_SIDE_MISSING = "generated_side_missing"
    GENERATED_SIDE_INVALID = "generated_side_invalid"
    TRUTH_SIDE_MISSING = "truth_side_missing"
    TRUTH_SIDE_INVALID = "truth_side_invalid"
    SIDE_MISMATCH = "side_mismatch"

    GENERATED_LINES_MISSING = "generated_lines_missing"
    GENERATED_RANGE_PARTIAL = "generated_range_partial"
    GENERATED_RANGE_INVALID = "generated_range_invalid"
    GENERATED_RANGE_REVERSED = "generated_range_reversed"
    GENERATED_LINE_COUNT_UNAVAILABLE = "generated_line_count_unavailable"
    GENERATED_RANGE_OUT_OF_BOUNDS = "generated_range_out_of_bounds"
    TRUTH_LINES_MISSING = "truth_lines_missing"
    TRUTH_RANGE_PARTIAL = "truth_range_partial"
    TRUTH_RANGE_INVALID = "truth_range_invalid"
    TRUTH_RANGE_REVERSED = "truth_range_reversed"
    TRUTH_LINE_COUNT_UNAVAILABLE = "truth_line_count_unavailable"
    TRUTH_RANGE_OUT_OF_BOUNDS = "truth_range_out_of_bounds"

    LINE_DISTANCE_EXCEEDED = "line_distance_exceeded"
    WITHIN_LINE_DISTANCE = "within_line_distance"
    OVERLAPPING_RANGE = "overlapping_range"
    EXACT_RANGE = "exact_range"


_REASON_ORDER = {
    reason: index for index, reason in enumerate(LocationMatchReason)
}
_SUCCESS_REASONS = frozenset(
    {
        LocationMatchReason.WITHIN_LINE_DISTANCE,
        LocationMatchReason.OVERLAPPING_RANGE,
        LocationMatchReason.EXACT_RANGE,
    }
)
_GENERATED_PATH_REASONS = frozenset(
    reason
    for reason in LocationMatchReason
    if reason.value.startswith("generated_path_")
)


class SidePathStatus(str, Enum):
    """Exact availability status for one canonical path on one diff side."""

    AVAILABLE = "available"
    CASEFOLD_NFC_COLLISION = "casefold_nfc_collision"
    ABSENT = "absent"


class GeneratedPathError(ValueError):
    """A generated path was rejected by the canonical repository policy."""

    def __init__(self, reason: LocationMatchReason, message: str) -> None:
        if reason not in _GENERATED_PATH_REASONS:
            raise TypeError("GeneratedPathError requires a generated-path reason")
        self.reason = reason
        super().__init__(message)


def _collision_key(path: str) -> str:
    # This exactly follows repository tree collision ordering: NFC, then casefold.
    return unicodedata.normalize("NFC", path).casefold()


def _classify_rejected_generated_path(
    value: Any, error: Exception
) -> LocationMatchReason:
    """Attach useful diagnostics only after the shared policy rejects a path."""

    if value is None:
        return LocationMatchReason.GENERATED_PATH_MISSING
    if type(value) is not str:
        return LocationMatchReason.GENERATED_PATH_TYPE_INVALID
    if not value:
        return LocationMatchReason.GENERATED_PATH_EMPTY
    try:
        encoded = value.encode("utf-8", "strict")
    except UnicodeEncodeError:
        return LocationMatchReason.GENERATED_PATH_INVALID_UNICODE
    if "\x00" in value:
        return LocationMatchReason.GENERATED_PATH_NUL
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        return LocationMatchReason.GENERATED_PATH_CONTROL
    if _WINDOWS_DRIVE_RE.match(value) is not None:
        return LocationMatchReason.GENERATED_PATH_WINDOWS_DRIVE
    if value.startswith("\\\\") or value.startswith("//"):
        return LocationMatchReason.GENERATED_PATH_UNC
    if value.startswith("/"):
        return LocationMatchReason.GENERATED_PATH_ABSOLUTE
    if "\\" in value:
        return LocationMatchReason.GENERATED_PATH_BACKSLASH

    components = value.split("/")
    if any(component == "" for component in components):
        return LocationMatchReason.GENERATED_PATH_EMPTY_COMPONENT
    if any(component == "." for component in components):
        return LocationMatchReason.GENERATED_PATH_DOT_COMPONENT
    if any(component == ".." for component in components):
        return LocationMatchReason.GENERATED_PATH_DOT_DOT_COMPONENT
    if len(components) > MAX_PATH_DEPTH:
        return LocationMatchReason.GENERATED_PATH_TOO_DEEP
    if len(encoded) > MAX_PATH_BYTES:
        return LocationMatchReason.GENERATED_PATH_TOO_LONG
    if any(
        len(component.encode("utf-8", "strict")) > MAX_PATH_COMPONENT_BYTES
        for component in components
    ):
        return LocationMatchReason.GENERATED_PATH_COMPONENT_TOO_LONG
    if any(
        _collision_key(component) in _VCS_METADATA_LABELS
        for component in components
    ):
        return LocationMatchReason.GENERATED_PATH_VCS_METADATA
    # NFC, Windows reserved/forbidden names, ADS syntax, and trailing dot/space
    # are intentionally represented by one stable shared-policy reason rather
    # than reimplementing repository acceptance logic here.
    del error
    return LocationMatchReason.GENERATED_PATH_REPOSITORY_POLICY


def normalize_generated_path(value: Any) -> str:
    """Return the unchanged path only when repository policy accepts it.

    This function performs no correction, case folding, separator conversion,
    or Unicode normalization.  Detailed reason classification occurs only
    after :func:`canonical_repository_path` rejects the value.
    """

    try:
        return canonical_repository_path(value)
    except (RepositoryLimitError, RepositoryPolicyError) as exc:
        reason = _classify_rejected_generated_path(value, exc)
        raise GeneratedPathError(reason, "generated path is noncanonical: %s" % exc) from exc


def _logical_line_count(raw: bytes) -> Optional[int]:
    """Count Evidence-compatible logical lines without allocating a line list."""

    if type(raw) is not bytes:
        raise TypeError("replayed file content must be immutable bytes")
    try:
        text = raw.decode("utf-8", "strict")
    except UnicodeDecodeError:
        return None

    line_count = 0
    index = 0
    length = len(text)
    ended_with_boundary = False
    while index < length:
        character = text[index]
        if character == "\r":
            if index + 1 < length and text[index + 1] == "\n":
                index += 1
            line_count += 1
            ended_with_boundary = True
        elif character == "\n" or character in _OTHER_SPLITLINE_BOUNDARIES:
            line_count += 1
            ended_with_boundary = True
        else:
            ended_with_boundary = False
        if line_count > MAX_FILE_LINE_COUNT:
            return None
        index += 1

    if length and not ended_with_boundary:
        line_count += 1
        if line_count > MAX_FILE_LINE_COUNT:
            return None
    return line_count


@dataclass(frozen=True)
class SideFileExtent:
    """One exact repository file and its replay-derived logical line count."""

    path: str
    line_count: Optional[int]

    def __post_init__(self) -> None:
        canonical_repository_path(self.path)
        if self.line_count is not None:
            if type(self.line_count) is not int:
                raise TypeError("line_count must be an integer or None")
            if not 0 <= self.line_count <= MAX_FILE_LINE_COUNT:
                raise ValueError(
                    "line_count must be between 0 and %d" % MAX_FILE_LINE_COUNT
                )

    def to_dict(self) -> Dict[str, Any]:
        return {"path": self.path, "line_count": self.line_count}


class _PathTrieNode:
    __slots__ = ("name", "is_file", "children")

    def __init__(self, name: Optional[str]) -> None:
        self.name = name
        self.is_file = False
        self.children: Dict[str, "_PathTrieNode"] = {}


def _validate_extent_hierarchy(
    extents: Tuple[SideFileExtent, ...], context: str
) -> None:
    """Reject trees impossible under repository sibling/prefix invariants."""

    root = _PathTrieNode(None)
    for extent in extents:
        node = root
        components = extent.path.split("/")
        for component in components:
            if node.is_file:
                raise ValueError("%s contains a file/directory prefix conflict" % context)
            key = _collision_key(component)
            child = node.children.get(key)
            if child is None:
                child = _PathTrieNode(component)
                node.children[key] = child
            elif child.name != component:
                raise ValueError(
                    "%s contains an NFC/casefold hierarchy collision" % context
                )
            node = child
        if node.is_file:
            raise ValueError("%s contains duplicate path %r" % (context, extent.path))
        if node.children:
            raise ValueError("%s contains a file/directory prefix conflict" % context)
        node.is_file = True


def _extent_sequence(value: Any, context: str) -> Tuple[SideFileExtent, ...]:
    if type(value) not in (list, tuple):
        raise TypeError("%s must be a list or tuple" % context)
    if len(value) > MAX_SIDE_FILE_EXTENTS:
        raise ValueError(
            "%s exceeds the file-extent limit of %d"
            % (context, MAX_SIDE_FILE_EXTENTS)
        )
    if any(type(item) is not SideFileExtent for item in value):
        raise TypeError("%s must contain only SideFileExtent values" % context)
    ordered = tuple(sorted(value, key=lambda item: item.path))
    _validate_extent_hierarchy(ordered, context)
    return ordered


def _collision_entries(
    extents: Tuple[SideFileExtent, ...]
) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted((_collision_key(item.path), item.path) for item in extents))


def _replay_extents(
    replay: PreparedRepositoryReplay, revision: str, context: str
) -> Tuple[SideFileExtent, ...]:
    paths = replay.paths(revision)
    if type(paths) is not tuple:
        raise TypeError("PreparedRepositoryReplay.paths() must return a tuple")
    if len(paths) > MAX_SIDE_FILE_EXTENTS:
        raise ValueError(
            "%s exceeds the file-extent limit of %d"
            % (context, MAX_SIDE_FILE_EXTENTS)
        )

    extents: List[SideFileExtent] = []
    for path in paths:
        canonical_repository_path(path)
        try:
            raw = replay.read_file(revision, path, max_bytes=MAX_GIT_BLOB_BYTES)
        except RepositoryLimitError:
            extents.append(SideFileExtent(path=path, line_count=None))
            continue
        if raw is None:
            raise ValueError("replay path catalog references a missing file")
        extents.append(
            SideFileExtent(path=path, line_count=_logical_line_count(raw))
        )
    return tuple(extents)


@dataclass(frozen=True)
class SidePathCatalog:
    """Immutable base/head file extents from canonical repository snapshots."""

    left_extents: Tuple[SideFileExtent, ...]
    right_extents: Tuple[SideFileExtent, ...]
    _left_paths: Tuple[str, ...] = field(init=False, repr=False, compare=False)
    _right_paths: Tuple[str, ...] = field(init=False, repr=False, compare=False)
    _left_collision_entries: Tuple[Tuple[str, str], ...] = field(
        init=False, repr=False, compare=False
    )
    _right_collision_entries: Tuple[Tuple[str, str], ...] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        left = _extent_sequence(self.left_extents, "left_extents")
        right = _extent_sequence(self.right_extents, "right_extents")
        object.__setattr__(self, "left_extents", left)
        object.__setattr__(self, "right_extents", right)
        object.__setattr__(self, "_left_paths", tuple(item.path for item in left))
        object.__setattr__(self, "_right_paths", tuple(item.path for item in right))
        object.__setattr__(self, "_left_collision_entries", _collision_entries(left))
        object.__setattr__(
            self, "_right_collision_entries", _collision_entries(right)
        )

    @classmethod
    def from_replay(cls, replay: PreparedRepositoryReplay) -> "SidePathCatalog":
        """Derive exact base/head extents from verified replay objects."""

        if type(replay) is not PreparedRepositoryReplay:
            raise TypeError("replay must be a verified PreparedRepositoryReplay")
        return cls(
            left_extents=_replay_extents(
                replay, replay.base_revision, "replay base paths"
            ),
            right_extents=_replay_extents(
                replay, replay.head_revision, "replay head paths"
            ),
        )

    @property
    def left_paths(self) -> Tuple[str, ...]:
        """Read-only compatibility projection of base paths."""

        return self._left_paths

    @property
    def right_paths(self) -> Tuple[str, ...]:
        """Read-only compatibility projection of head paths."""

        return self._right_paths

    def extents_for(self, side: DiffSide) -> Tuple[SideFileExtent, ...]:
        if type(side) is not DiffSide:
            raise TypeError("side must be a DiffSide")
        return self.left_extents if side is DiffSide.LEFT else self.right_extents

    def paths_for(self, side: DiffSide) -> Tuple[str, ...]:
        if type(side) is not DiffSide:
            raise TypeError("side must be a DiffSide")
        return self._left_paths if side is DiffSide.LEFT else self._right_paths

    def extent_for(self, side: DiffSide, path: str) -> Optional[SideFileExtent]:
        if type(side) is not DiffSide:
            raise TypeError("side must be a DiffSide")
        canonical = canonical_repository_path(path)
        paths = self._left_paths if side is DiffSide.LEFT else self._right_paths
        extents = self.left_extents if side is DiffSide.LEFT else self.right_extents
        index = bisect_left(paths, canonical)
        if index < len(paths) and paths[index] == canonical:
            return extents[index]
        return None

    def line_count_for(self, side: DiffSide, path: str) -> Optional[int]:
        extent = self.extent_for(side, path)
        return None if extent is None else extent.line_count

    def availability(self, side: DiffSide, path: str) -> SidePathStatus:
        if type(side) is not DiffSide:
            raise TypeError("side must be a DiffSide")
        canonical = canonical_repository_path(path)
        if self.extent_for(side, canonical) is not None:
            return SidePathStatus.AVAILABLE
        entries = (
            self._left_collision_entries
            if side is DiffSide.LEFT
            else self._right_collision_entries
        )
        key = _collision_key(canonical)
        index = bisect_left(entries, (key, ""))
        if index < len(entries) and entries[index][0] == key:
            return SidePathStatus.CASEFOLD_NFC_COLLISION
        return SidePathStatus.ABSENT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "left_extents": [item.to_dict() for item in self.left_extents],
            "right_extents": [item.to_dict() for item in self.right_extents],
        }

    def digest(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class LocationMatchPolicy:
    """Versioned and digestible bounds for deterministic line matching."""

    version: str = LOCATION_MATCH_POLICY_VERSION
    max_line_distance: int = DEFAULT_MAX_LINE_DISTANCE

    def __post_init__(self) -> None:
        if type(self.version) is not str or self.version != LOCATION_MATCH_POLICY_VERSION:
            raise ValueError(
                "version must be exactly %r" % LOCATION_MATCH_POLICY_VERSION
            )
        if type(self.max_line_distance) is not int:
            raise TypeError("max_line_distance must be an integer (bool is not accepted)")
        if not 0 <= self.max_line_distance <= MAX_CONFIGURED_LINE_DISTANCE:
            raise ValueError(
                "max_line_distance must be between 0 and %d"
                % MAX_CONFIGURED_LINE_DISTANCE
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "max_line_distance": self.max_line_distance,
        }

    def digest(self) -> str:
        return canonical_sha256(self.to_dict())

    @property
    def identity(self) -> str:
        return stable_id("location-match-policy", self.to_dict())


DEFAULT_LOCATION_MATCH_POLICY = LocationMatchPolicy()


def _canonical_reasons(
    reasons: Sequence[LocationMatchReason],
) -> Tuple[LocationMatchReason, ...]:
    unique = set(reasons)
    return tuple(sorted(unique, key=lambda reason: _REASON_ORDER[reason]))


def _strict_object(
    value: Any,
    expected_fields: Sequence[str],
    context: str,
) -> Dict[str, Any]:
    if type(value) is not dict:
        raise TypeError("%s must be an object" % context)
    expected = set(expected_fields)
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            "%s fields are not exact (missing=%r, unexpected=%r)"
            % (context, missing, unexpected)
        )
    return value


def _canonical_json_payload(data: Any, maximum: int, context: str) -> Any:
    payload = _strict_json_loads(data, maximum, context)
    text = data.decode("utf-8", "strict") if type(data) is bytes else data
    if canonical_json(payload) != text:
        raise ValueError("%s must use canonical JSON encoding" % context)
    return payload


def _reasons_from_dict(value: Any) -> Tuple[LocationMatchReason, ...]:
    if type(value) is not list:
        raise TypeError("location match.reasons must be an array")
    if not value:
        raise ValueError("location match.reasons must not be empty")
    if len(value) > len(LocationMatchReason):
        raise ValueError("location match.reasons exceeds the reason-code limit")

    parsed: List[LocationMatchReason] = []
    for item in value:
        if type(item) is not str:
            raise TypeError("location match.reasons must contain strings")
        try:
            parsed.append(LocationMatchReason(item))
        except ValueError as exc:
            raise ValueError("location match.reasons contains an unknown reason") from exc

    reasons = tuple(parsed)
    canonical = _canonical_reasons(reasons)
    if reasons != canonical:
        raise ValueError(
            "location match.reasons must be unique and in canonical enum order"
        )
    return reasons


@dataclass(frozen=True)
class LocationMatchResult:
    """A location-only decision; it intentionally has no semantic status."""

    matched: bool
    score: int
    reasons: Tuple[LocationMatchReason, ...]

    def __post_init__(self) -> None:
        if type(self.matched) is not bool:
            raise TypeError("matched must be a bool")
        if type(self.score) is not int:
            raise TypeError("score must be an integer (bool is not accepted)")
        if not NO_LOCATION_MATCH_SCORE <= self.score <= EXACT_LOCATION_SCORE:
            raise ValueError("score is outside the location score bounds")
        if type(self.reasons) is not tuple:
            raise TypeError("reasons must be a tuple")
        if not self.reasons:
            raise ValueError("reasons must not be empty")
        for reason in self.reasons:
            if type(reason) is not LocationMatchReason:
                raise TypeError("reasons must contain LocationMatchReason values")
        canonical = _canonical_reasons(self.reasons)
        if len(canonical) != len(self.reasons):
            raise ValueError("reasons must not contain duplicates")
        object.__setattr__(self, "reasons", canonical)

        success = tuple(reason for reason in canonical if reason in _SUCCESS_REASONS)
        if self.matched:
            if self.score <= NO_LOCATION_MATCH_SCORE:
                raise ValueError("a matched result must have a positive score")
            if len(canonical) != 1 or len(success) != 1:
                raise ValueError("a matched result requires exactly one success reason")
            success_reason = success[0]
            if (
                success_reason is LocationMatchReason.EXACT_RANGE
                and self.score != EXACT_LOCATION_SCORE
            ):
                raise ValueError("exact range requires the exact score band")
            if (
                success_reason is LocationMatchReason.OVERLAPPING_RANGE
                and self.score != OVERLAP_LOCATION_SCORE
            ):
                raise ValueError("overlapping range requires the overlap score band")
            if success_reason is LocationMatchReason.WITHIN_LINE_DISTANCE and not (
                DISTANCE_LOCATION_SCORE_CEILING - MAX_CONFIGURED_LINE_DISTANCE
                <= self.score
                < DISTANCE_LOCATION_SCORE_CEILING
            ):
                raise ValueError("line distance score is outside the distance band")
        else:
            if self.score != NO_LOCATION_MATCH_SCORE:
                raise ValueError("an unmatched result must have score zero")
            if success:
                raise ValueError("an unmatched result cannot contain a success reason")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matched": self.matched,
            "score": self.score,
            "reasons": [reason.value for reason in self.reasons],
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Any) -> "LocationMatchResult":
        payload = _strict_object(
            value,
            ("matched", "score", "reasons"),
            "location match",
        )
        if type(payload["matched"]) is not bool:
            raise TypeError("location match.matched must be a bool")
        if type(payload["score"]) is not int:
            raise TypeError(
                "location match.score must be an integer (bool is not accepted)"
            )
        return cls(
            matched=payload["matched"],
            score=payload["score"],
            reasons=_reasons_from_dict(payload["reasons"]),
        )

    @classmethod
    def from_json(cls, data: Any) -> "LocationMatchResult":
        return cls.from_dict(
            _canonical_json_payload(
                data,
                MAX_LOCATION_MATCH_RESULT_JSON_BYTES,
                "LocationMatchResult JSON",
            )
        )


@dataclass(frozen=True)
class TruthLocationTarget:
    """Caller-supplied stable issue identity and per-issue location index."""

    truth_id: str
    truth_index: int
    location: TruthLocation

    def __post_init__(self) -> None:
        if type(self.truth_id) is not str or not self.truth_id:
            raise TypeError("truth_id must be a non-empty string")
        if len(self.truth_id) > MAX_IDENTIFIER_CHARS:
            raise ValueError("truth_id exceeds the identifier limit")
        if self.truth_id != self.truth_id.strip() or any(
            character.isspace()
            or ord(character) < 32
            or ord(character) == 127
            for character in self.truth_id
        ):
            raise ValueError("truth_id must not contain whitespace or controls")
        if type(self.truth_index) is not int:
            raise TypeError("truth_index must be an integer (bool is not accepted)")
        if not 0 <= self.truth_index < MAX_TRUTH_LOCATIONS:
            raise ValueError("truth_index is outside the per-finding location bounds")
        if type(self.location) is not TruthLocation:
            raise TypeError("location must be a TruthLocation")


@dataclass(frozen=True)
class LocationCandidate:
    """One explicit truth identity and its deterministic location result."""

    truth_id: str
    truth_index: int
    truth_location: TruthLocation
    match: LocationMatchResult

    def __post_init__(self) -> None:
        TruthLocationTarget(self.truth_id, self.truth_index, self.truth_location)
        if type(self.match) is not LocationMatchResult:
            raise TypeError("match must be a LocationMatchResult")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "truth_id": self.truth_id,
            "truth_index": self.truth_index,
            "truth_location": self.truth_location.to_dict(),
            "match": self.match.to_dict(),
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(
        cls,
        value: Any,
        *,
        truth_target: TruthLocationTarget,
    ) -> "LocationCandidate":
        if type(truth_target) is not TruthLocationTarget:
            raise TypeError("truth_target must be a TruthLocationTarget")
        payload = _strict_object(
            value,
            ("truth_id", "truth_index", "truth_location", "match"),
            "location candidate",
        )
        candidate = cls(
            truth_id=payload["truth_id"],
            truth_index=payload["truth_index"],
            truth_location=TruthLocation.from_dict(payload["truth_location"]),
            match=LocationMatchResult.from_dict(payload["match"]),
        )
        if (
            candidate.truth_id != truth_target.truth_id
            or candidate.truth_index != truth_target.truth_index
            or candidate.truth_location != truth_target.location
        ):
            raise ValueError(
                "location candidate does not match its bound truth identity/index/location"
            )
        return candidate

    @classmethod
    def from_json(
        cls,
        data: Any,
        *,
        truth_target: TruthLocationTarget,
    ) -> "LocationCandidate":
        return cls.from_dict(
            _canonical_json_payload(
                data,
                MAX_LOCATION_CANDIDATE_JSON_BYTES,
                "LocationCandidate JSON",
            ),
            truth_target=truth_target,
        )


def _candidate_sort_key(candidate: LocationCandidate) -> Tuple[Any, ...]:
    return (
        -candidate.match.score,
        candidate.truth_id,
        candidate.truth_index,
        canonical_json(candidate.truth_location.to_dict()),
    )


def _canonical_candidates(values: Any) -> Tuple[LocationCandidate, ...]:
    if type(values) not in (list, tuple):
        raise TypeError("location candidates must be a list or tuple")
    if len(values) > MAX_LOCATION_TARGETS:
        raise ValueError(
            "location candidates exceeds the case-wide candidate limit of %d"
            % MAX_LOCATION_TARGETS
        )
    if any(type(item) is not LocationCandidate for item in values):
        raise TypeError("location candidates must contain only LocationCandidate values")

    candidates = tuple(values)
    if candidates != tuple(sorted(candidates, key=_candidate_sort_key)):
        raise ValueError("location candidates must use canonical score/identity order")

    seen = set()
    for candidate in candidates:
        identity = (candidate.truth_id, candidate.truth_index)
        if identity in seen:
            raise ValueError(
                "location candidates contains duplicate truth identity/index %r"
                % (identity,)
            )
        seen.add(identity)
    return candidates


def _append_reason(
    reasons: List[LocationMatchReason], reason: LocationMatchReason
) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _side_or_reason(
    value: Any,
    missing: LocationMatchReason,
    invalid: LocationMatchReason,
    reasons: List[LocationMatchReason],
) -> Optional[DiffSide]:
    if value is None:
        _append_reason(reasons, missing)
        return None
    if type(value) is not DiffSide:
        _append_reason(reasons, invalid)
        return None
    return value


def _range_or_reason(
    from_line: Any,
    to_line: Any,
    *,
    missing: LocationMatchReason,
    partial: LocationMatchReason,
    invalid: LocationMatchReason,
    reversed_reason: LocationMatchReason,
    reasons: List[LocationMatchReason],
) -> Optional[Tuple[int, int]]:
    if from_line is None and to_line is None:
        _append_reason(reasons, missing)
        return None
    if (from_line is None) != (to_line is None):
        _append_reason(reasons, partial)
        return None
    if (
        type(from_line) is not int
        or type(to_line) is not int
        or not 1 <= from_line <= MAX_LINE_NUMBER
        or not 1 <= to_line <= MAX_LINE_NUMBER
    ):
        _append_reason(reasons, invalid)
        return None
    if to_line < from_line:
        _append_reason(reasons, reversed_reason)
        return None
    return (from_line, to_line)


def _validate_range_extent(
    line_range: Optional[Tuple[int, int]],
    extent: Optional[SideFileExtent],
    *,
    unavailable: LocationMatchReason,
    out_of_bounds: LocationMatchReason,
    reasons: List[LocationMatchReason],
) -> None:
    if line_range is None or extent is None:
        return
    if extent.line_count is None:
        _append_reason(reasons, unavailable)
    elif line_range[1] > extent.line_count:
        _append_reason(reasons, out_of_bounds)


def _no_match(reasons: Sequence[LocationMatchReason]) -> LocationMatchResult:
    return LocationMatchResult(
        matched=False,
        score=NO_LOCATION_MATCH_SCORE,
        reasons=_canonical_reasons(reasons),
    )


@dataclass(frozen=True)
class LocationMatcher:
    """Pure matcher bound to one immutable replay-derived extent snapshot."""

    side_paths: SidePathCatalog
    policy: LocationMatchPolicy = DEFAULT_LOCATION_MATCH_POLICY

    def __post_init__(self) -> None:
        if type(self.side_paths) is not SidePathCatalog:
            raise TypeError("side_paths must be a SidePathCatalog")
        if type(self.policy) is not LocationMatchPolicy:
            raise TypeError("policy must be a LocationMatchPolicy")

    def match(
        self, finding: SubmissionFinding, truth: TruthLocation
    ) -> LocationMatchResult:
        if type(finding) is not SubmissionFinding:
            raise TypeError("finding must be a SubmissionFinding")
        if type(truth) is not TruthLocation:
            raise TypeError("truth must be a TruthLocation")

        reasons: List[LocationMatchReason] = []
        generated_path: Optional[str]
        try:
            generated_path = normalize_generated_path(finding.path)
        except GeneratedPathError as exc:
            generated_path = None
            _append_reason(reasons, exc.reason)

        truth_path: Optional[str]
        try:
            truth_path = canonical_repository_path(truth.path)
        except (RepositoryLimitError, RepositoryPolicyError):
            truth_path = None
            _append_reason(reasons, LocationMatchReason.TRUTH_PATH_INVALID)

        generated_side = _side_or_reason(
            finding.side,
            LocationMatchReason.GENERATED_SIDE_MISSING,
            LocationMatchReason.GENERATED_SIDE_INVALID,
            reasons,
        )
        truth_side = _side_or_reason(
            truth.side,
            LocationMatchReason.TRUTH_SIDE_MISSING,
            LocationMatchReason.TRUTH_SIDE_INVALID,
            reasons,
        )
        if (
            generated_side is not None
            and truth_side is not None
            and generated_side is not truth_side
        ):
            _append_reason(reasons, LocationMatchReason.SIDE_MISMATCH)

        generated_range = _range_or_reason(
            finding.from_line,
            finding.to_line,
            missing=LocationMatchReason.GENERATED_LINES_MISSING,
            partial=LocationMatchReason.GENERATED_RANGE_PARTIAL,
            invalid=LocationMatchReason.GENERATED_RANGE_INVALID,
            reversed_reason=LocationMatchReason.GENERATED_RANGE_REVERSED,
            reasons=reasons,
        )
        truth_range = _range_or_reason(
            truth.from_line,
            truth.to_line,
            missing=LocationMatchReason.TRUTH_LINES_MISSING,
            partial=LocationMatchReason.TRUTH_RANGE_PARTIAL,
            invalid=LocationMatchReason.TRUTH_RANGE_INVALID,
            reversed_reason=LocationMatchReason.TRUTH_RANGE_REVERSED,
            reasons=reasons,
        )

        if generated_path is not None and truth_path is not None:
            if generated_path != truth_path:
                if _collision_key(generated_path) == _collision_key(truth_path):
                    _append_reason(
                        reasons,
                        LocationMatchReason.PATH_CASEFOLD_NFC_COLLISION,
                    )
                else:
                    _append_reason(reasons, LocationMatchReason.PATH_MISMATCH)

        generated_extent: Optional[SideFileExtent] = None
        if generated_path is not None and generated_side is not None:
            availability = self.side_paths.availability(
                generated_side, generated_path
            )
            if availability is SidePathStatus.AVAILABLE:
                generated_extent = self.side_paths.extent_for(
                    generated_side, generated_path
                )
            elif availability is SidePathStatus.CASEFOLD_NFC_COLLISION:
                _append_reason(
                    reasons,
                    LocationMatchReason.PATH_CASEFOLD_NFC_COLLISION,
                )
            else:
                _append_reason(
                    reasons, LocationMatchReason.PATH_UNAVAILABLE_ON_SIDE
                )

        truth_extent: Optional[SideFileExtent] = None
        if truth_path is not None and truth_side is not None:
            truth_extent = self.side_paths.extent_for(truth_side, truth_path)
            if truth_extent is None:
                _append_reason(
                    reasons,
                    LocationMatchReason.TRUTH_PATH_UNAVAILABLE_ON_SIDE,
                )

        _validate_range_extent(
            generated_range,
            generated_extent,
            unavailable=LocationMatchReason.GENERATED_LINE_COUNT_UNAVAILABLE,
            out_of_bounds=LocationMatchReason.GENERATED_RANGE_OUT_OF_BOUNDS,
            reasons=reasons,
        )
        _validate_range_extent(
            truth_range,
            truth_extent,
            unavailable=LocationMatchReason.TRUTH_LINE_COUNT_UNAVAILABLE,
            out_of_bounds=LocationMatchReason.TRUTH_RANGE_OUT_OF_BOUNDS,
            reasons=reasons,
        )

        if reasons:
            return _no_match(reasons)

        assert generated_range is not None
        assert truth_range is not None
        generated_from, generated_to = generated_range
        truth_from, truth_to = truth_range

        if generated_range == truth_range:
            return LocationMatchResult(
                matched=True,
                score=EXACT_LOCATION_SCORE,
                reasons=(LocationMatchReason.EXACT_RANGE,),
            )
        if max(generated_from, truth_from) <= min(generated_to, truth_to):
            return LocationMatchResult(
                matched=True,
                score=OVERLAP_LOCATION_SCORE,
                reasons=(LocationMatchReason.OVERLAPPING_RANGE,),
            )

        if generated_to < truth_from:
            line_distance = truth_from - generated_to
        else:
            line_distance = generated_from - truth_to
        if line_distance <= self.policy.max_line_distance:
            return LocationMatchResult(
                matched=True,
                score=DISTANCE_LOCATION_SCORE_CEILING - line_distance,
                reasons=(LocationMatchReason.WITHIN_LINE_DISTANCE,),
            )
        return _no_match((LocationMatchReason.LINE_DISTANCE_EXCEEDED,))

    def generate_candidates(
        self,
        finding: SubmissionFinding,
        truth_locations: Sequence[TruthLocationTarget],
    ) -> Tuple[LocationCandidate, ...]:
        """Evaluate explicit truth identities in deterministic score order."""

        if type(finding) is not SubmissionFinding:
            raise TypeError("finding must be a SubmissionFinding")
        targets = _canonical_targets(truth_locations)
        candidates = tuple(
            LocationCandidate(
                truth_id=target.truth_id,
                truth_index=target.truth_index,
                truth_location=target.location,
                match=self.match(finding, target.location),
            )
            for target in targets
        )
        return tuple(sorted(candidates, key=_candidate_sort_key))


def _canonical_targets(values: Any) -> Tuple[TruthLocationTarget, ...]:
    if type(values) not in (list, tuple):
        raise TypeError("truth_locations must be a list or tuple")
    if len(values) > MAX_LOCATION_TARGETS:
        raise ValueError(
            "truth_locations exceeds the case-wide target limit of %d"
            % MAX_LOCATION_TARGETS
        )
    if any(type(item) is not TruthLocationTarget for item in values):
        raise TypeError(
            "truth_locations must contain only explicit TruthLocationTarget values"
        )
    ordered = tuple(
        sorted(
            values,
            key=lambda target: (
                target.truth_id,
                target.truth_index,
                canonical_json(target.location.to_dict()),
            ),
        )
    )
    seen = set()
    for target in ordered:
        identity = (target.truth_id, target.truth_index)
        if identity in seen:
            raise ValueError(
                "truth_locations contains duplicate truth identity/index %r"
                % (identity,)
            )
        seen.add(identity)
    return ordered


def location_candidates_to_dict(
    candidates: Sequence[LocationCandidate],
) -> List[Dict[str, Any]]:
    """Return the strict embedded JSON projection for a candidate collection."""

    canonical = _canonical_candidates(candidates)
    payload = [candidate.to_dict() for candidate in canonical]
    if (
        len(canonical_json(payload).encode("utf-8"))
        > MAX_LOCATION_CANDIDATES_JSON_BYTES
    ):
        raise ValueError("location candidates exceeds the canonical JSON byte limit")
    return payload


def location_candidates_to_json(
    candidates: Sequence[LocationCandidate],
) -> str:
    return canonical_json(location_candidates_to_dict(candidates))


def location_candidates_from_dict(
    value: Any,
    *,
    truth_locations: Sequence[TruthLocationTarget],
) -> Tuple[LocationCandidate, ...]:
    """Hydrate a complete candidate set bound to the supplied truth targets."""

    targets = _canonical_targets(truth_locations)
    if type(value) is not list:
        raise TypeError("location candidates must be an array")
    if len(value) > MAX_LOCATION_TARGETS:
        raise ValueError(
            "location candidates exceeds the case-wide candidate limit of %d"
            % MAX_LOCATION_TARGETS
        )
    if len(value) != len(targets):
        raise ValueError(
            "location candidates must contain exactly one record per truth target"
        )

    targets_by_identity = {
        (target.truth_id, target.truth_index): target for target in targets
    }
    candidates: List[LocationCandidate] = []
    for item in value:
        payload = _strict_object(
            item,
            ("truth_id", "truth_index", "truth_location", "match"),
            "location candidate",
        )
        truth_id = payload["truth_id"]
        truth_index = payload["truth_index"]
        if type(truth_id) is not str:
            raise TypeError("location candidate.truth_id must be a string")
        if type(truth_index) is not int:
            raise TypeError(
                "location candidate.truth_index must be an integer (bool is not accepted)"
            )
        target = targets_by_identity.get((truth_id, truth_index))
        if target is None:
            raise ValueError(
                "location candidate truth identity/index is not present in bound targets"
            )
        candidates.append(LocationCandidate.from_dict(payload, truth_target=target))

    canonical = _canonical_candidates(candidates)
    if {
        (candidate.truth_id, candidate.truth_index) for candidate in canonical
    } != set(targets_by_identity):
        raise ValueError(
            "location candidates do not bind the complete truth target collection"
        )
    return canonical


def location_candidates_from_json(
    data: Any,
    *,
    truth_locations: Sequence[TruthLocationTarget],
) -> Tuple[LocationCandidate, ...]:
    return location_candidates_from_dict(
        _canonical_json_payload(
            data,
            MAX_LOCATION_CANDIDATES_JSON_BYTES,
            "LocationCandidate collection JSON",
        ),
        truth_locations=truth_locations,
    )


def match_location(
    finding: SubmissionFinding,
    truth: TruthLocation,
    side_paths: SidePathCatalog,
    policy: LocationMatchPolicy = DEFAULT_LOCATION_MATCH_POLICY,
) -> LocationMatchResult:
    """Match one generated finding against one truth location."""

    return LocationMatcher(side_paths=side_paths, policy=policy).match(finding, truth)


def generate_location_candidates(
    finding: SubmissionFinding,
    truth_locations: Sequence[TruthLocationTarget],
    side_paths: SidePathCatalog,
    policy: LocationMatchPolicy = DEFAULT_LOCATION_MATCH_POLICY,
) -> Tuple[LocationCandidate, ...]:
    """Generate deterministic candidates for explicit truth identities."""

    return LocationMatcher(side_paths=side_paths, policy=policy).generate_candidates(
        finding, truth_locations
    )


__all__ = [
    "DEFAULT_LOCATION_MATCH_POLICY",
    "DEFAULT_MAX_LINE_DISTANCE",
    "DISTANCE_LOCATION_SCORE_CEILING",
    "EXACT_LOCATION_SCORE",
    "GeneratedPathError",
    "LOCATION_MATCH_POLICY_VERSION",
    "LocationCandidate",
    "LocationMatchPolicy",
    "LocationMatchReason",
    "LocationMatchResult",
    "LocationMatcher",
    "MAX_CONFIGURED_LINE_DISTANCE",
    "MAX_FILE_LINE_COUNT",
    "MAX_LOCATION_CANDIDATE_JSON_BYTES",
    "MAX_LOCATION_CANDIDATES_JSON_BYTES",
    "MAX_LOCATION_MATCH_RESULT_JSON_BYTES",
    "MAX_LOCATION_TARGETS",
    "MAX_SIDE_FILE_EXTENTS",
    "MAX_SIDE_PATHS",
    "NO_LOCATION_MATCH_SCORE",
    "OVERLAP_LOCATION_SCORE",
    "SideFileExtent",
    "SidePathCatalog",
    "SidePathStatus",
    "TruthLocationTarget",
    "generate_location_candidates",
    "location_candidates_from_dict",
    "location_candidates_from_json",
    "location_candidates_to_dict",
    "location_candidates_to_json",
    "match_location",
    "normalize_generated_path",
]
