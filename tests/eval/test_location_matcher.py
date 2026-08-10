from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, fields
import hashlib
from pathlib import Path
import re
from types import MappingProxyType

import pytest

import review_agent_eval.repository as repository_module
from review_agent_eval.match_location import (
    DISTANCE_LOCATION_SCORE_CEILING,
    EXACT_LOCATION_SCORE,
    LOCATION_MATCH_POLICY_VERSION,
    MAX_CONFIGURED_LINE_DISTANCE,
    MAX_FILE_LINE_COUNT,
    MAX_LOCATION_TARGETS,
    MAX_SIDE_FILE_EXTENTS,
    NO_LOCATION_MATCH_SCORE,
    OVERLAP_LOCATION_SCORE,
    GeneratedPathError,
    LocationCandidate,
    LocationMatchPolicy,
    LocationMatchReason,
    LocationMatchResult,
    LocationMatcher,
    SideFileExtent,
    SidePathCatalog,
    SidePathStatus,
    TruthLocationTarget,
    generate_location_candidates,
    location_candidates_from_dict,
    location_candidates_from_json,
    location_candidates_to_dict,
    location_candidates_to_json,
    match_location,
    normalize_generated_path,
)
from review_agent_eval.models import (
    DiffSide,
    FindingSeverity,
    MAX_LINE_NUMBER,
    MAX_TRUTH_LOCATIONS,
    SubmissionFinding,
    TruthLocation,
    canonical_json,
)
from review_agent_eval.repository import (
    PreparedRepositoryReplay,
    RepositoryPolicyError,
)


APP = "src/app.py"
OTHER = "src/other.py"
DELETED = "src/deleted.py"


def finding(
    *,
    path=APP,
    side=DiffSide.RIGHT,
    from_line=10,
    to_line=12,
) -> SubmissionFinding:
    return SubmissionFinding(
        finding_id="finding-1",
        claim="The changed branch can return the wrong result.",
        severity=FindingSeverity.HIGH,
        path=path,
        side=side,
        from_line=from_line,
        to_line=to_line,
        evidence_refs=(),
        suggested_action=None,
    )


def truth(
    *,
    path=APP,
    side=DiffSide.RIGHT,
    from_line=10,
    to_line=12,
) -> TruthLocation:
    return TruthLocation(
        path=path,
        side=side,
        from_line=from_line,
        to_line=to_line,
    )


def extent(path: str, line_count=100) -> SideFileExtent:
    return SideFileExtent(path=path, line_count=line_count)


def catalog(
    *,
    left=(APP, OTHER, DELETED),
    right=(APP, OTHER),
    line_count=100,
) -> SidePathCatalog:
    return SidePathCatalog(
        left_extents=tuple(extent(path, line_count) for path in left),
        right_extents=tuple(extent(path, line_count) for path in right),
    )


def catalog_for_count(line_count, *, path=APP) -> SidePathCatalog:
    file_extent = extent(path, line_count)
    return SidePathCatalog(
        left_extents=(file_extent,),
        right_extents=(file_extent,),
    )


def target(
    truth_id: str,
    *,
    truth_index: int = 0,
    location: TruthLocation = None,
) -> TruthLocationTarget:
    return TruthLocationTarget(
        truth_id=truth_id,
        truth_index=truth_index,
        location=truth() if location is None else location,
    )


def replay_with_files(
    base_files: dict[str, bytes], head_files: dict[str, bytes]
) -> PreparedRepositoryReplay:
    base_revision = "a" * 40
    head_revision = "b" * 40
    objects = {}
    files_by_revision = {}
    for revision, side_name, files in (
        (base_revision, "base", base_files),
        (head_revision, "head", head_files),
    ):
        indexed = {}
        for path, raw in sorted(files.items()):
            oid = hashlib.sha1(
                side_name.encode("ascii")
                + b"\0"
                + path.encode("utf-8")
                + b"\0"
                + raw
            ).hexdigest()
            objects[oid] = repository_module._GitObject(oid, "blob", raw)
            indexed[path] = oid
        files_by_revision[revision] = MappingProxyType(indexed)
    return PreparedRepositoryReplay(
        prepared_repository_id="prepared-test",
        repository_descriptor_digest="d" * 64,
        base_revision=base_revision,
        head_revision=head_revision,
        _git_dir=Path("unused-replay.git"),
        _runner=None,
        _open_check=lambda: None,
        _verify_cache=lambda: None,
        _objects=MappingProxyType(objects),
        _files_by_revision=MappingProxyType(files_by_revision),
    )


@pytest.mark.parametrize(
    ("path", "reason"),
    [
        (None, LocationMatchReason.GENERATED_PATH_MISSING),
        ("", LocationMatchReason.GENERATED_PATH_EMPTY),
        ("/src/app.py", LocationMatchReason.GENERATED_PATH_ABSOLUTE),
        ("C:/src/app.py", LocationMatchReason.GENERATED_PATH_WINDOWS_DRIVE),
        (r"\\server\share\app.py", LocationMatchReason.GENERATED_PATH_UNC),
        (r"src\app.py", LocationMatchReason.GENERATED_PATH_BACKSLASH),
        ("src//app.py", LocationMatchReason.GENERATED_PATH_EMPTY_COMPONENT),
        ("src/./app.py", LocationMatchReason.GENERATED_PATH_DOT_COMPONENT),
        ("src/../app.py", LocationMatchReason.GENERATED_PATH_DOT_DOT_COMPONENT),
        ("src/app.py\x00tail", LocationMatchReason.GENERATED_PATH_NUL),
        ("src/app.py\n", LocationMatchReason.GENERATED_PATH_CONTROL),
        ("src/.GiT/config", LocationMatchReason.GENERATED_PATH_VCS_METADATA),
        ("a" * 256, LocationMatchReason.GENERATED_PATH_COMPONENT_TOO_LONG),
        ("/".join(("a",) * 65), LocationMatchReason.GENERATED_PATH_TOO_DEEP),
        ("/".join(("a" * 200,) * 6), LocationMatchReason.GENERATED_PATH_TOO_LONG),
    ],
)
def test_generated_path_reasons_classify_shared_policy_rejections(
    path, reason: LocationMatchReason
) -> None:
    with pytest.raises(GeneratedPathError) as captured:
        normalize_generated_path(path)

    assert captured.value.reason is reason


def test_gitmodules_is_a_valid_parent_repository_path() -> None:
    assert normalize_generated_path(".gitmodules") == ".gitmodules"
    assert normalize_generated_path("deps/.gitmodules") == "deps/.gitmodules"


@pytest.mark.parametrize(
    "path",
    [
        "src/cafe\u0301.py",
        "src/CON.py",
        "src/a:b.py",
        "src/trailing.",
        "src/trailing ",
    ],
)
def test_shared_repository_policy_rejects_noncanonical_counterexamples(
    path: str,
) -> None:
    with pytest.raises(GeneratedPathError) as captured:
        normalize_generated_path(path)
    with pytest.raises(RepositoryPolicyError):
        SideFileExtent(path=path, line_count=10)

    assert (
        captured.value.reason
        is LocationMatchReason.GENERATED_PATH_REPOSITORY_POLICY
    )


def test_c1_path_character_is_accepted_exactly_like_repository_policy() -> None:
    c1_path = "src/next\x85line.py"
    side_paths = SidePathCatalog(
        left_extents=(extent(c1_path, 2),),
        right_extents=(extent(c1_path, 2),),
    )

    assert normalize_generated_path(c1_path) == c1_path
    assert match_location(
        finding(path=c1_path, from_line=1, to_line=1),
        truth(path=c1_path, from_line=1, to_line=1),
        side_paths,
    ).matched


def test_decomposed_path_is_never_corrected_into_an_exact_match() -> None:
    decomposed = "src/cafe\u0301.py"
    composed = "src/café.py"
    side_paths = SidePathCatalog(
        left_extents=(), right_extents=(extent(composed, 10),)
    )

    result = match_location(
        finding(path=decomposed, from_line=1, to_line=1),
        truth(path=composed, from_line=1, to_line=1),
        side_paths,
    )

    assert result.matched is False
    assert result.reasons == (
        LocationMatchReason.GENERATED_PATH_REPOSITORY_POLICY,
    )


@pytest.mark.parametrize(
    "bad_path",
    ["/src/app.py", "../src/app.py", r"src\app.py", ".git/config"],
)
def test_invalid_generated_path_is_a_stable_non_match(bad_path: str) -> None:
    result = match_location(finding(path=bad_path), truth(), catalog())

    assert result.matched is False
    assert result.score == NO_LOCATION_MATCH_SCORE
    assert len(result.reasons) == 1
    assert result.reasons[0].value.startswith("generated_path_")


@pytest.mark.parametrize("bad_path", ["src/CON.py", "a" * 256])
def test_noncanonical_truth_path_is_rejected_by_shared_policy(
    bad_path: str,
) -> None:
    result = match_location(finding(), truth(path=bad_path), catalog())

    assert result.reasons == (LocationMatchReason.TRUTH_PATH_INVALID,)


def test_wrong_path_does_not_match_when_both_files_exist() -> None:
    result = match_location(finding(path=OTHER), truth(path=APP), catalog())

    assert result == LocationMatchResult(
        matched=False,
        score=0,
        reasons=(LocationMatchReason.PATH_MISMATCH,),
    )


def test_casefold_collision_is_reported_without_matching() -> None:
    side_paths = SidePathCatalog(
        left_extents=(), right_extents=(extent("src/app.py", 20),)
    )

    result = match_location(
        finding(path="src/App.py"),
        truth(path="src/app.py"),
        side_paths,
    )

    assert result.matched is False
    assert result.reasons == (
        LocationMatchReason.PATH_CASEFOLD_NFC_COLLISION,
    )
    assert LocationMatchReason.PATH_MISMATCH not in result.reasons


def test_file_extent_catalog_is_immutable_sorted_and_exposes_line_counts() -> None:
    left_input = [extent(OTHER, 8), extent(APP, 12)]
    side_paths = SidePathCatalog(
        left_extents=left_input,
        right_extents=(extent(APP, 20),),
    )
    left_input.append(extent(DELETED, 3))

    assert side_paths.left_paths == (APP, OTHER)
    assert side_paths.left_extents == (extent(APP, 12), extent(OTHER, 8))
    assert side_paths.extent_for(DiffSide.LEFT, APP) == extent(APP, 12)
    assert side_paths.line_count_for(DiffSide.RIGHT, APP) == 20
    assert side_paths.availability(DiffSide.LEFT, APP) is SidePathStatus.AVAILABLE
    assert (
        side_paths.availability(DiffSide.LEFT, "src/App.py")
        is SidePathStatus.CASEFOLD_NFC_COLLISION
    )
    assert side_paths.availability(DiffSide.LEFT, DELETED) is SidePathStatus.ABSENT
    assert side_paths.digest() == SidePathCatalog(
        left_extents=(extent(APP, 12), extent(OTHER, 8)),
        right_extents=(extent(APP, 20),),
    ).digest()
    with pytest.raises(FrozenInstanceError):
        side_paths.left_extents = ()


@pytest.mark.parametrize(
    "paths",
    [
        ("A/x.py", "a/y.py"),
        ("dir/X.py", "dir/x.py"),
        ("file", "file/child.py"),
        ("File", "file/child.py"),
        (APP, APP),
    ],
)
def test_catalog_rejects_hierarchy_collisions_and_prefix_conflicts(paths) -> None:
    with pytest.raises(ValueError):
        SidePathCatalog(
            left_extents=tuple(extent(path, 1) for path in paths),
            right_extents=(),
        )


def test_catalog_allows_shared_exact_directory_prefixes() -> None:
    side_paths = SidePathCatalog(
        left_extents=(extent("dir/a.py", 1), extent("dir/b.py", 2)),
        right_extents=(),
    )

    assert side_paths.left_paths == ("dir/a.py", "dir/b.py")


@pytest.mark.parametrize("line_count", [-1, MAX_FILE_LINE_COUNT + 1, True, 1.5])
def test_file_extent_rejects_invalid_line_count(line_count) -> None:
    with pytest.raises((TypeError, ValueError)):
        SideFileExtent(path=APP, line_count=line_count)


def test_catalog_rejects_oversized_extent_collection_before_sorting() -> None:
    oversized = (extent(APP, 1),) * (MAX_SIDE_FILE_EXTENTS + 1)

    with pytest.raises(ValueError, match="file-extent limit"):
        SidePathCatalog(left_extents=oversized, right_extents=())


def test_from_replay_counts_evidence_compatible_lines_and_unknown_sources() -> None:
    boundaries = b"a\vb\fc\x1cd\x1de\x1ef\xc2\x85g\xe2\x80\xa8h\xe2\x80\xa9last"
    replay = replay_with_files(
        {
            "boundaries.txt": boundaries,
            "crlf.txt": b"one\r\ntwo\rthree\n",
            "deleted.py": b"deleted\n",
            "empty.txt": b"",
            "invalid.bin": b"valid\xffinvalid\n",
            "max-lines.txt": b"\n" * MAX_FILE_LINE_COUNT,
            "many-lines.txt": b"\n" * (MAX_FILE_LINE_COUNT + 1),
        },
        {
            "src/app.py": b"alpha\r\nbeta\nlast",
        },
    )

    side_paths = SidePathCatalog.from_replay(replay)

    assert side_paths.line_count_for(DiffSide.LEFT, "boundaries.txt") == 9
    assert side_paths.line_count_for(DiffSide.LEFT, "crlf.txt") == 3
    assert side_paths.line_count_for(DiffSide.LEFT, "empty.txt") == 0
    assert side_paths.line_count_for(DiffSide.LEFT, "invalid.bin") is None
    assert (
        side_paths.line_count_for(DiffSide.LEFT, "max-lines.txt")
        == MAX_FILE_LINE_COUNT
    )
    assert side_paths.line_count_for(DiffSide.LEFT, "many-lines.txt") is None
    assert side_paths.line_count_for(DiffSide.RIGHT, APP) == 3
    assert DELETED.replace("src/", "") not in side_paths.right_paths


@pytest.mark.parametrize("path", ["invalid.bin", "many-lines.txt"])
def test_replay_source_with_unknown_line_count_can_never_line_match(path: str) -> None:
    replay = replay_with_files(
        {
            "invalid.bin": b"\xff",
            "many-lines.txt": b"\n" * (MAX_FILE_LINE_COUNT + 1),
        },
        {},
    )
    side_paths = SidePathCatalog.from_replay(replay)

    result = match_location(
        finding(path=path, side=DiffSide.LEFT, from_line=1, to_line=1),
        truth(path=path, side=DiffSide.LEFT, from_line=1, to_line=1),
        side_paths,
    )

    assert result.matched is False
    assert result.reasons == (
        LocationMatchReason.GENERATED_LINE_COUNT_UNAVAILABLE,
        LocationMatchReason.TRUTH_LINE_COUNT_UNAVAILABLE,
    )


def test_deleted_file_matches_on_left_and_is_rejected_on_right() -> None:
    replay = replay_with_files({DELETED: b"old\n"}, {})
    side_paths = SidePathCatalog.from_replay(replay)

    left = match_location(
        finding(path=DELETED, side=DiffSide.LEFT, from_line=1, to_line=1),
        truth(path=DELETED, side=DiffSide.LEFT, from_line=1, to_line=1),
        side_paths,
    )
    wrong_side = match_location(
        finding(path=DELETED, side=DiffSide.RIGHT, from_line=1, to_line=1),
        truth(path=DELETED, side=DiffSide.LEFT, from_line=1, to_line=1),
        side_paths,
    )

    assert left.matched is True
    assert wrong_side.reasons == (
        LocationMatchReason.PATH_UNAVAILABLE_ON_SIDE,
        LocationMatchReason.SIDE_MISMATCH,
    )


def test_side_mismatch_never_location_matches_when_file_exists_on_both_sides() -> None:
    both_sides = catalog(right=(APP, OTHER, DELETED))

    result = match_location(
        finding(side=DiffSide.LEFT),
        truth(side=DiffSide.RIGHT),
        both_sides,
    )

    assert result.reasons == (LocationMatchReason.SIDE_MISMATCH,)


def test_missing_generated_location_fields_remain_missing() -> None:
    result = match_location(
        finding(path=None, side=None, from_line=None, to_line=None),
        truth(),
        catalog(),
    )

    assert result == LocationMatchResult(
        matched=False,
        score=0,
        reasons=(
            LocationMatchReason.GENERATED_PATH_MISSING,
            LocationMatchReason.GENERATED_SIDE_MISSING,
            LocationMatchReason.GENERATED_LINES_MISSING,
        ),
    )


def test_missing_truth_side_and_lines_are_deterministic_non_matches() -> None:
    result = match_location(
        finding(),
        truth(side=None, from_line=None, to_line=None),
        catalog(),
    )

    assert result.reasons == (
        LocationMatchReason.TRUTH_SIDE_MISSING,
        LocationMatchReason.TRUTH_LINES_MISSING,
    )


@pytest.mark.parametrize(
    ("from_line", "to_line", "expected_reason"),
    [
        (10, None, LocationMatchReason.GENERATED_RANGE_PARTIAL),
        (None, 12, LocationMatchReason.GENERATED_RANGE_PARTIAL),
        (12, 10, LocationMatchReason.GENERATED_RANGE_REVERSED),
    ],
)
def test_partial_and_reversed_generated_ranges_do_not_match(
    from_line, to_line, expected_reason: LocationMatchReason
) -> None:
    result = match_location(
        finding(from_line=from_line, to_line=to_line),
        truth(),
        catalog(),
    )

    assert result.matched is False
    assert result.reasons == (expected_reason,)


def test_eof_is_valid_but_eof_plus_one_is_generated_out_of_bounds() -> None:
    side_paths = catalog_for_count(20)

    eof = match_location(
        finding(from_line=20, to_line=20),
        truth(from_line=20, to_line=20),
        side_paths,
    )
    beyond = match_location(
        finding(from_line=1, to_line=21),
        truth(from_line=1, to_line=20),
        side_paths,
    )

    assert eof.matched is True
    assert beyond.reasons == (
        LocationMatchReason.GENERATED_RANGE_OUT_OF_BOUNDS,
    )


def test_truth_range_is_independently_checked_against_file_extent() -> None:
    result = match_location(
        finding(from_line=1, to_line=20),
        truth(from_line=1, to_line=21),
        catalog_for_count(20),
    )

    assert result.reasons == (
        LocationMatchReason.TRUTH_RANGE_OUT_OF_BOUNDS,
    )


def test_huge_synthetic_range_never_overlap_matches_short_file() -> None:
    result = match_location(
        finding(from_line=1, to_line=MAX_LINE_NUMBER),
        truth(from_line=10, to_line=12),
        catalog_for_count(20),
    )

    assert result.matched is False
    assert result.reasons == (
        LocationMatchReason.GENERATED_RANGE_OUT_OF_BOUNDS,
    )


def test_empty_file_rejects_every_positive_range_on_both_sides() -> None:
    result = match_location(
        finding(from_line=1, to_line=1),
        truth(from_line=1, to_line=1),
        catalog_for_count(0),
    )

    assert result.reasons == (
        LocationMatchReason.GENERATED_RANGE_OUT_OF_BOUNDS,
        LocationMatchReason.TRUTH_RANGE_OUT_OF_BOUNDS,
    )


def test_generated_and_truth_unavailable_line_counts_have_distinct_reasons() -> None:
    side_paths = SidePathCatalog(
        left_extents=(),
        right_extents=(extent(APP, None), extent(OTHER, 20)),
    )
    generated_unknown = match_location(
        finding(path=APP, from_line=1, to_line=1),
        truth(path=OTHER, from_line=1, to_line=1),
        side_paths,
    )
    truth_unknown = match_location(
        finding(path=OTHER, from_line=1, to_line=1),
        truth(path=APP, from_line=1, to_line=1),
        side_paths,
    )

    assert LocationMatchReason.GENERATED_LINE_COUNT_UNAVAILABLE in generated_unknown.reasons
    assert LocationMatchReason.TRUTH_LINE_COUNT_UNAVAILABLE not in generated_unknown.reasons
    assert LocationMatchReason.TRUTH_LINE_COUNT_UNAVAILABLE in truth_unknown.reasons
    assert LocationMatchReason.GENERATED_LINE_COUNT_UNAVAILABLE not in truth_unknown.reasons


def test_exact_range_has_the_highest_bounded_integer_score() -> None:
    result = match_location(finding(), truth(), catalog())

    assert result == LocationMatchResult(
        matched=True,
        score=EXACT_LOCATION_SCORE,
        reasons=(LocationMatchReason.EXACT_RANGE,),
    )
    assert type(result.score) is int


def test_overlapping_range_matches_below_exact() -> None:
    result = match_location(
        finding(from_line=10, to_line=15),
        truth(from_line=12, to_line=20),
        catalog(),
    )

    assert result == LocationMatchResult(
        matched=True,
        score=OVERLAP_LOCATION_SCORE,
        reasons=(LocationMatchReason.OVERLAPPING_RANGE,),
    )
    assert EXACT_LOCATION_SCORE > result.score


def test_distance_boundary_matches_and_nearer_scores_above_farther() -> None:
    policy = LocationMatchPolicy(max_line_distance=5)
    near = match_location(
        finding(from_line=10, to_line=12),
        truth(from_line=14, to_line=16),
        catalog(),
        policy,
    )
    boundary = match_location(
        finding(from_line=10, to_line=12),
        truth(from_line=17, to_line=19),
        catalog(),
        policy,
    )

    assert near.score == DISTANCE_LOCATION_SCORE_CEILING - 2
    assert boundary.score == DISTANCE_LOCATION_SCORE_CEILING - 5
    assert OVERLAP_LOCATION_SCORE > near.score > boundary.score > 0
    assert boundary.reasons == (LocationMatchReason.WITHIN_LINE_DISTANCE,)


def test_beyond_distance_is_a_zero_score_non_match() -> None:
    result = match_location(
        finding(from_line=10, to_line=12),
        truth(from_line=18, to_line=20),
        catalog(),
        LocationMatchPolicy(max_line_distance=5),
    )

    assert result == LocationMatchResult(
        matched=False,
        score=0,
        reasons=(LocationMatchReason.LINE_DISTANCE_EXCEEDED,),
    )


def test_location_match_result_persistence_round_trip_is_canonical() -> None:
    result = match_location(
        finding(path=None, side=None, from_line=None, to_line=None),
        truth(),
        catalog(),
    )

    payload = result.to_dict()

    assert payload == {
        "matched": False,
        "score": 0,
        "reasons": [
            "generated_path_missing",
            "generated_side_missing",
            "generated_lines_missing",
        ],
    }
    assert result.to_json() == canonical_json(payload)
    assert LocationMatchResult.from_dict(payload) == result
    assert LocationMatchResult.from_json(result.to_json()) == result
    assert LocationMatchResult.from_json(result.to_json().encode("utf-8")) == result


@pytest.mark.parametrize(
    "payload",
    [
        {
            "matched": False,
            "score": EXACT_LOCATION_SCORE,
            "reasons": ["path_mismatch"],
        },
        {
            "matched": True,
            "score": 0,
            "reasons": ["exact_range"],
        },
        {
            "matched": True,
            "score": OVERLAP_LOCATION_SCORE,
            "reasons": ["exact_range"],
        },
        {
            "matched": False,
            "score": 0,
            "reasons": ["not_a_location_reason"],
        },
        {
            "matched": False,
            "score": 0,
            "reasons": ["generated_side_missing", "generated_path_missing"],
        },
        {
            "matched": False,
            "score": 0,
            "reasons": ["path_mismatch", "path_mismatch"],
        },
    ],
)
def test_location_match_result_hydration_rejects_tampering(payload) -> None:
    with pytest.raises((TypeError, ValueError)):
        LocationMatchResult.from_dict(payload)


def test_location_match_result_hydration_requires_exact_fields() -> None:
    payload = match_location(finding(), truth(), catalog()).to_dict()
    payload["semantic_eligible"] = True

    with pytest.raises(ValueError, match="fields are not exact"):
        LocationMatchResult.from_dict(payload)


@pytest.mark.parametrize(
    "data",
    [
        '{ "matched":true,"reasons":["exact_range"],"score":1000000}',
        '{"matched":true,"matched":true,"reasons":["exact_range"],"score":1000000}',
    ],
)
def test_location_match_result_from_json_rejects_noncanonical_or_duplicate_keys(
    data: str,
) -> None:
    with pytest.raises(ValueError):
        LocationMatchResult.from_json(data)


def test_policy_is_versioned_validated_and_has_stable_identity() -> None:
    first = LocationMatchPolicy(max_line_distance=7)
    same = LocationMatchPolicy(max_line_distance=7)
    changed = LocationMatchPolicy(max_line_distance=8)

    assert first.version == LOCATION_MATCH_POLICY_VERSION
    assert re.fullmatch(r"[0-9a-f]{64}", first.digest())
    assert first.digest() == same.digest()
    assert first.identity == same.identity
    assert first.digest() != changed.digest()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"version": "location_match_policy_v1"},
        {"max_line_distance": -1},
        {"max_line_distance": MAX_CONFIGURED_LINE_DISTANCE + 1},
        {"max_line_distance": True},
        {"max_line_distance": 1.5},
    ],
)
def test_policy_rejects_unknown_version_and_invalid_bounds(kwargs) -> None:
    with pytest.raises((TypeError, ValueError)):
        LocationMatchPolicy(**kwargs)


def test_candidate_generation_requires_explicit_truth_targets() -> None:
    with pytest.raises(TypeError, match="TruthLocationTarget"):
        LocationMatcher(catalog()).generate_candidates(finding(), (truth(),))


def test_location_candidate_persistence_round_trip_is_truth_bound() -> None:
    truth_target = target(
        "truth-a",
        truth_index=3,
        location=truth(from_line=16, to_line=18),
    )
    candidate = generate_location_candidates(
        finding(),
        (truth_target,),
        catalog(),
    )[0]

    assert LocationCandidate.from_dict(
        candidate.to_dict(), truth_target=truth_target
    ) == candidate
    assert LocationCandidate.from_json(
        candidate.to_json(), truth_target=truth_target
    ) == candidate
    assert LocationCandidate.from_json(
        candidate.to_json().encode("utf-8"), truth_target=truth_target
    ) == candidate


@pytest.mark.parametrize("tampered_field", ["truth_id", "truth_index", "truth_location"])
def test_location_candidate_hydration_rejects_truth_binding_tampering(
    tampered_field: str,
) -> None:
    truth_target = target("truth-a", location=truth())
    candidate = generate_location_candidates(finding(), (truth_target,), catalog())[0]
    payload = candidate.to_dict()
    if tampered_field == "truth_id":
        payload["truth_id"] = "truth-b"
    elif tampered_field == "truth_index":
        payload["truth_index"] = 1
    else:
        payload["truth_location"] = truth(path=OTHER).to_dict()

    with pytest.raises(ValueError, match="bound truth identity/index/location"):
        LocationCandidate.from_dict(payload, truth_target=truth_target)


def test_location_candidate_hydration_rejects_tampered_match_consistency() -> None:
    truth_target = target("truth-a", location=truth())
    candidate = generate_location_candidates(finding(), (truth_target,), catalog())[0]
    payload = candidate.to_dict()
    payload["match"]["matched"] = False

    with pytest.raises(ValueError, match="unmatched result must have score zero"):
        LocationCandidate.from_dict(payload, truth_target=truth_target)


def test_candidates_preserve_shared_locations_for_distinct_truth_issues() -> None:
    targets = (
        target("truth-b", location=truth()),
        target("truth-a", location=truth()),
    )

    candidates = generate_location_candidates(finding(), targets, catalog())

    assert [(item.truth_id, item.truth_index) for item in candidates] == [
        ("truth-a", 0),
        ("truth-b", 0),
    ]
    assert all(item.match.matched for item in candidates)


def test_sixty_five_truth_issues_are_accepted_and_order_invariant() -> None:
    targets = tuple(target("truth-%03d" % index) for index in range(65))
    matcher = LocationMatcher(catalog())

    forward = matcher.generate_candidates(finding(), targets)
    reverse = matcher.generate_candidates(finding(), tuple(reversed(targets)))

    assert len(forward) == 65
    assert forward == reverse
    assert [item.truth_id for item in forward] == sorted(item.truth_id for item in forward)


def test_candidate_scores_and_explicit_identity_sort_deterministically() -> None:
    targets = (
        target("truth-wrong", location=truth(path=OTHER)),
        target("truth-near", location=truth(from_line=16, to_line=18)),
        target("truth-overlap", location=truth(from_line=11, to_line=20)),
        target("truth-exact", location=truth()),
    )

    candidates = generate_location_candidates(finding(), targets, catalog())

    assert [item.truth_id for item in candidates] == [
        "truth-exact",
        "truth-overlap",
        "truth-near",
        "truth-wrong",
    ]
    assert tuple(item.match.score for item in candidates) == (
        EXACT_LOCATION_SCORE,
        OVERLAP_LOCATION_SCORE,
        DISTANCE_LOCATION_SCORE_CEILING - 4,
        0,
    )


def test_location_candidate_collection_round_trip_is_complete_and_canonical() -> None:
    targets = (
        target("truth-wrong", location=truth(path=OTHER)),
        target("truth-near", location=truth(from_line=16, to_line=18)),
        target("truth-exact", location=truth()),
    )
    candidates = generate_location_candidates(finding(), targets, catalog())

    payload = location_candidates_to_dict(candidates)
    encoded = location_candidates_to_json(candidates)

    assert encoded == canonical_json(payload)
    assert location_candidates_from_dict(
        payload,
        truth_locations=tuple(reversed(targets)),
    ) == candidates
    assert location_candidates_from_json(
        encoded,
        truth_locations=targets,
    ) == candidates
    assert location_candidates_from_json(
        encoded.encode("utf-8"),
        truth_locations=targets,
    ) == candidates


def test_location_candidate_collection_rejects_noncanonical_order() -> None:
    targets = (
        target("truth-wrong", location=truth(path=OTHER)),
        target("truth-exact", location=truth()),
    )
    candidates = generate_location_candidates(finding(), targets, catalog())
    reversed_candidates = tuple(reversed(candidates))

    with pytest.raises(ValueError, match="canonical score/identity order"):
        location_candidates_to_dict(reversed_candidates)
    with pytest.raises(ValueError, match="canonical score/identity order"):
        location_candidates_from_dict(
            [candidate.to_dict() for candidate in reversed_candidates],
            truth_locations=targets,
        )


def test_location_candidate_collection_rejects_missing_or_duplicate_target() -> None:
    targets = (
        target("truth-a", location=truth()),
        target("truth-b", location=truth(from_line=20, to_line=21)),
    )
    candidates = generate_location_candidates(finding(), targets, catalog())
    payload = location_candidates_to_dict(candidates)

    with pytest.raises(ValueError, match="exactly one record"):
        location_candidates_from_dict(payload[:-1], truth_locations=targets)

    duplicated = [deepcopy(payload[0]), deepcopy(payload[0])]
    with pytest.raises(ValueError, match="duplicate truth identity/index"):
        location_candidates_from_dict(duplicated, truth_locations=targets)


def test_location_candidate_collection_from_json_rejects_noncanonical_text() -> None:
    truth_target = target("truth-a", location=truth())
    candidates = generate_location_candidates(finding(), (truth_target,), catalog())
    encoded = location_candidates_to_json(candidates)

    with pytest.raises(ValueError, match="canonical JSON encoding"):
        location_candidates_from_json(
            " " + encoded,
            truth_locations=(truth_target,),
        )


def test_duplicate_truth_identity_index_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate truth identity/index"):
        generate_location_candidates(
            finding(),
            (
                target("truth-a", location=truth()),
                target("truth-a", location=truth(from_line=20, to_line=21)),
            ),
            catalog(),
        )


def test_per_finding_truth_location_index_remains_bounded() -> None:
    with pytest.raises(ValueError, match="per-finding"):
        target("truth-a", truth_index=MAX_TRUTH_LOCATIONS)


def test_case_wide_candidate_limit_is_checked_before_duplicate_sorting() -> None:
    repeated = (target("truth-a"),) * (MAX_LOCATION_TARGETS + 1)

    with pytest.raises(ValueError, match="case-wide target limit"):
        LocationMatcher(catalog()).generate_candidates(finding(), repeated)


def test_location_non_match_exposes_no_semantic_ineligibility_decision() -> None:
    result = match_location(finding(path=OTHER), truth(), catalog())

    assert [item.name for item in fields(result)] == ["matched", "score", "reasons"]
    assert not hasattr(result, "semantic_eligible")
    assert not hasattr(result, "issue_match")
    assert all(
        token not in reason.value
        for reason in result.reasons
        for token in ("semantic", "fabricated", "ineligible")
    )


def test_public_matcher_values_are_frozen() -> None:
    result = match_location(finding(), truth(), catalog())
    policy = LocationMatchPolicy()
    file_extent = extent(APP, 100)
    side_paths = catalog()
    truth_target = target("truth-a")
    candidate = generate_location_candidates(
        finding(), (truth_target,), side_paths
    )[0]

    for value, attribute in (
        (result, "score"),
        (policy, "max_line_distance"),
        (file_extent, "line_count"),
        (side_paths, "right_extents"),
        (truth_target, "truth_index"),
        (candidate, "truth_index"),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(value, attribute, None)
