from __future__ import annotations

from dataclasses import FrozenInstanceError
from itertools import permutations, product, repeat
import random

import pytest

from review_agent_eval.assignment import (
    ASSIGNMENT_POLICY_VERSION,
    MAX_ASSIGNMENT_EDGES,
    MAX_ASSIGNMENT_IDENTIFIER_CHARS,
    MAX_ASSIGNMENT_LEFT_ITEMS,
    MAX_ASSIGNMENT_RIGHT_ITEMS,
    MAX_ASSIGNMENT_WEIGHT,
    AssignedPair,
    WeightedAssignmentEdge,
    WeightedAssignmentResult,
    maximum_weight_bipartite_assignment,
)


def _edge(left_id: str, right_id: str, weight: int) -> WeightedAssignmentEdge:
    return WeightedAssignmentEdge(left_id, right_id, weight)


def _pairs(result: WeightedAssignmentResult) -> tuple[tuple[str, str, int], ...]:
    return tuple(
        (match.left_id, match.right_id, match.weight)
        for match in result.matches
    )


def test_global_optimum_beats_greedy_edge_selection() -> None:
    result = maximum_weight_bipartite_assignment(
        ["left-b", "left-a"],
        ["right-y", "right-x"],
        [
            _edge("left-a", "right-x", 10),
            _edge("left-a", "right-y", 9),
            _edge("left-b", "right-x", 9),
        ],
    )

    assert result.policy_version == ASSIGNMENT_POLICY_VERSION
    assert result.total_weight == 18
    assert _pairs(result) == (
        ("left-a", "right-y", 9),
        ("left-b", "right-x", 9),
    )
    assert result.unmatched_left == ()
    assert result.unmatched_right == ()


def test_all_input_edge_and_dict_orders_produce_the_same_result() -> None:
    left = ("left-b", "left-a")
    right = ("right-y", "right-x")
    edges = (
        _edge("left-a", "right-x", 5),
        _edge("left-a", "right-y", 5),
        _edge("left-b", "right-x", 5),
        _edge("left-b", "right-y", 5),
    )
    expected = WeightedAssignmentResult(
        policy_version=ASSIGNMENT_POLICY_VERSION,
        matches=(
            AssignedPair("left-a", "right-x", 5),
            AssignedPair("left-b", "right-y", 5),
        ),
        unmatched_left=(),
        unmatched_right=(),
        total_weight=10,
    )

    # This exhausts every order of both item collections and all 24 edge
    # insertion orders.  dict views are intentional: insertion order must not
    # become solver policy.
    for left_order in permutations(left):
        for right_order in permutations(right):
            for edge_order in permutations(edges):
                left_dict = dict.fromkeys(left_order)
                right_dict = dict.fromkeys(right_order)
                edge_dict = {
                    (edge.left_id, edge.right_id): edge
                    for edge in edge_order
                }
                assert maximum_weight_bipartite_assignment(
                    left_dict.keys(), right_dict.keys(), edge_dict.values()
                ) == expected


def test_equal_weight_ties_use_the_explicit_assignment_vector_policy() -> None:
    complete_tie = maximum_weight_bipartite_assignment(
        ["b", "a"],
        ["y", "x"],
        [_edge(left, right, 1) for left in ("b", "a") for right in ("y", "x")],
    )
    assert _pairs(complete_tie) == (("a", "x", 1), ("b", "y", 1))

    # Both alternatives total 2:
    #   (x, unmatched) versus (y, x).
    # The former wins at the first vector coordinate even though it has fewer
    # matches; unmatched is an abstract sentinel after every real right ID.
    cardinality_tie = maximum_weight_bipartite_assignment(
        ["b", "a"],
        ["y", "x"],
        [
            _edge("a", "x", 2),
            _edge("a", "y", 1),
            _edge("b", "x", 1),
        ],
    )
    assert _pairs(cardinality_tie) == (("a", "x", 2),)
    assert cardinality_tie.unmatched_left == ("b",)
    assert cardinality_tie.unmatched_right == ("y",)

    earlier_match = maximum_weight_bipartite_assignment(
        ["b", "a"],
        ["x"],
        [_edge("b", "x", 1), _edge("a", "x", 1)],
    )
    assert _pairs(earlier_match) == (("a", "x", 1),)
    assert earlier_match.unmatched_left == ("b",)


def test_one_to_one_matching_leaves_extra_duplicate_claim_unmatched() -> None:
    result = maximum_weight_bipartite_assignment(
        ["duplicate-b", "duplicate-a"],
        ["truth"],
        [
            _edge("duplicate-b", "truth", 50),
            _edge("duplicate-a", "truth", 50),
        ],
    )

    assert _pairs(result) == (("duplicate-a", "truth", 50),)
    assert result.unmatched_left == ("duplicate-b",)
    assert result.unmatched_right == ()


def test_optional_unmatched_choice_can_outweigh_a_larger_cardinality() -> None:
    result = maximum_weight_bipartite_assignment(
        ["b", "a"],
        ["y", "x"],
        [
            _edge("a", "x", 100),
            _edge("a", "y", 49),
            _edge("b", "x", 49),
        ],
    )

    assert result.total_weight == 100
    assert _pairs(result) == (("a", "x", 100),)
    assert result.unmatched_left == ("b",)
    assert result.unmatched_right == ("y",)


def test_empty_sides_are_supported_and_canonical() -> None:
    no_left = maximum_weight_bipartite_assignment([], ["z", "a"], [])
    assert no_left == WeightedAssignmentResult(
        ASSIGNMENT_POLICY_VERSION, (), (), ("a", "z"), 0
    )

    no_right = maximum_weight_bipartite_assignment(["z", "a"], [], [])
    assert no_right == WeightedAssignmentResult(
        ASSIGNMENT_POLICY_VERSION, (), ("a", "z"), (), 0
    )

    empty = maximum_weight_bipartite_assignment([], [], [])
    assert empty == WeightedAssignmentResult(
        ASSIGNMENT_POLICY_VERSION, (), (), (), 0
    )


@pytest.mark.parametrize(
    "bad_weight",
    [True, False, 0, -1, MAX_ASSIGNMENT_WEIGHT + 1, 1.0, "1"],
)
def test_weights_must_be_exact_bounded_positive_integers(
    bad_weight: object,
) -> None:
    with pytest.raises(ValueError, match="weight"):
        WeightedAssignmentEdge("left", "right", bad_weight)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad_id",
    [
        None,
        1,
        "",
        " ",
        " leading",
        "trailing ",
        "internal space",
        "line\nbreak",
        "control\x00",
        "\ud800",
        "x" * (MAX_ASSIGNMENT_IDENTIFIER_CHARS + 1),
    ],
)
def test_identifiers_must_be_canonical_non_empty_strings(bad_id: object) -> None:
    with pytest.raises(ValueError):
        maximum_weight_bipartite_assignment([bad_id], [], [])  # type: ignore[list-item]


def test_duplicate_ids_edges_unknown_endpoints_and_edge_types_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate identifier"):
        maximum_weight_bipartite_assignment(["left", "left"], ["right"], [])
    with pytest.raises(ValueError, match="duplicate identifier"):
        maximum_weight_bipartite_assignment(["left"], ["right", "right"], [])
    with pytest.raises(ValueError, match="duplicate left/right"):
        maximum_weight_bipartite_assignment(
            ["left"],
            ["right"],
            [_edge("left", "right", 1), _edge("left", "right", 2)],
        )
    with pytest.raises(ValueError, match="unknown left"):
        maximum_weight_bipartite_assignment(
            ["left"], ["right"], [_edge("other", "right", 1)]
        )
    with pytest.raises(ValueError, match="unknown right"):
        maximum_weight_bipartite_assignment(
            ["left"], ["right"], [_edge("left", "other", 1)]
        )
    with pytest.raises(ValueError, match="WeightedAssignmentEdge"):
        maximum_weight_bipartite_assignment(
            ["left"], ["right"], [("left", "right", 1)]  # type: ignore[list-item]
        )


def test_hard_item_edge_and_weight_budgets_are_enforced() -> None:
    with pytest.raises(ValueError, match="left_ids exceeds"):
        maximum_weight_bipartite_assignment(
            (f"left-{index}" for index in range(MAX_ASSIGNMENT_LEFT_ITEMS + 1)),
            [],
            [],
        )
    with pytest.raises(ValueError, match="right_ids exceeds"):
        maximum_weight_bipartite_assignment(
            [],
            (f"right-{index}" for index in range(MAX_ASSIGNMENT_RIGHT_ITEMS + 1)),
            [],
        )
    with pytest.raises(ValueError, match="edges exceeds"):
        maximum_weight_bipartite_assignment(
            ["left"],
            ["right"],
            repeat(_edge("left", "right", 1), MAX_ASSIGNMENT_EDGES + 1),
        )

    maximum = maximum_weight_bipartite_assignment(
        ["left"],
        ["right"],
        [_edge("left", "right", MAX_ASSIGNMENT_WEIGHT)],
    )
    assert maximum.total_weight == MAX_ASSIGNMENT_WEIGHT


def test_public_values_are_deeply_immutable_and_self_validate() -> None:
    edge = _edge("left", "right", 3)
    pair = AssignedPair("left", "right", 3)
    result = WeightedAssignmentResult(
        ASSIGNMENT_POLICY_VERSION, (pair,), (), (), 3
    )

    with pytest.raises(FrozenInstanceError):
        edge.weight = 4  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        pair.right_id = "other"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.total_weight = 4  # type: ignore[misc]
    with pytest.raises(ValueError, match="total_weight"):
        WeightedAssignmentResult(
            ASSIGNMENT_POLICY_VERSION, (pair,), (), (), 4
        )


def _brute_force_expected(
    left_ids: list[str],
    right_ids: list[str],
    edges: list[WeightedAssignmentEdge],
) -> tuple[int, tuple[int, ...], tuple[tuple[str, str, int], ...]]:
    left = sorted(left_ids)
    right = sorted(right_ids)
    right_rank = {identifier: index for index, identifier in enumerate(right)}
    weights = {(edge.left_id, edge.right_id): edge.weight for edge in edges}
    sentinel = len(right)
    best: tuple[int, tuple[int, ...], tuple[tuple[str, str, int], ...]] | None = None

    for vector in product(range(sentinel + 1), repeat=len(left)):
        real = [rank for rank in vector if rank != sentinel]
        if len(set(real)) != len(real):
            continue
        matches: list[tuple[str, str, int]] = []
        total = 0
        valid = True
        for left_id, rank in zip(left, vector):
            if rank == sentinel:
                continue
            right_id = right[rank]
            weight = weights.get((left_id, right_id))
            if weight is None:
                valid = False
                break
            total += weight
            matches.append((left_id, right_id, weight))
        if not valid:
            continue
        candidate = (total, vector, tuple(matches))
        if best is None or (-total, vector) < (-best[0], best[1]):
            best = candidate

    assert best is not None
    return best


def test_small_random_graphs_match_an_independent_exhaustive_oracle() -> None:
    random_source = random.Random(20260717)
    for _ in range(80):
        left = [f"left-{index}" for index in range(random_source.randrange(5))]
        right = [f"right-{index}" for index in range(random_source.randrange(5))]
        edges = [
            _edge(left_id, right_id, random_source.randrange(1, 8))
            for left_id in left
            for right_id in right
            if random_source.random() < 0.65
        ]
        random_source.shuffle(left)
        random_source.shuffle(right)
        random_source.shuffle(edges)

        expected_weight, _, expected_matches = _brute_force_expected(
            left, right, edges
        )
        result = maximum_weight_bipartite_assignment(left, right, edges)
        assert result.total_weight == expected_weight
        assert _pairs(result) == expected_matches


def test_medium_sparse_adversarial_graph_is_practical_and_exact() -> None:
    size = 192
    left = [f"left-{index:03d}" for index in range(size)]
    right = [f"right-{index:03d}" for index in range(size)]
    edge_weights: dict[tuple[str, str], int] = {}

    # Ninety-six independent greedy traps establish the exact optimum.  The
    # low-weight cyclic decoys make the graph substantially larger and highly
    # connected without constructing a dense matrix.
    for index in range(0, size, 2):
        edge_weights[(left[index], right[index])] = 1_000
        edge_weights[(left[index], right[index + 1])] = 999
        edge_weights[(left[index + 1], right[index])] = 999
    for left_index, left_id in enumerate(left):
        for offset in range(1, 33):
            right_id = right[(left_index + offset) % size]
            edge_weights.setdefault((left_id, right_id), 1)

    edges = [
        _edge(left_id, right_id, weight)
        for (left_id, right_id), weight in reversed(tuple(edge_weights.items()))
    ]
    result = maximum_weight_bipartite_assignment(
        reversed(left), reversed(right), edges
    )

    expected = tuple(
        pair
        for index in range(0, size, 2)
        for pair in (
            (left[index], right[index + 1], 999),
            (left[index + 1], right[index], 999),
        )
    )
    assert len(edges) < MAX_ASSIGNMENT_EDGES
    assert result.total_weight == size * 999
    assert _pairs(result) == expected
    assert result.unmatched_left == ()
    assert result.unmatched_right == ()
