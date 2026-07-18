"""Exact, deterministic maximum-weight bipartite assignment.

The public policy is deliberately independent of caller iteration order.  Let
``L`` and ``R`` be the left and right identifiers sorted by Python's exact
Unicode code-point string order.  For an assignment, define a vector with one
entry for every identifier in ``L``: the matched right identifier, or an
abstract unmatched sentinel that compares *after* every identifier in ``R``.

Policy v1 first maximizes the sum of edge weights.  Among assignments with the
same maximum sum, it chooses the lexicographically smallest such vector.  This
also specifies ties between assignments of different cardinalities.

The implementation has two exact polynomial-time phases:

* sparse successive-shortest-path min-cost flow obtains a maximum-weight
  assignment while representing every left item as either matched or
  unmatched;
* zero-reduced-cost residual cycles fix each canonical left choice in turn.
  A candidate choice can occur in another optimum exactly when its residual
  edge belongs to such a cycle.  This realizes the policy directly rather
  than depending on traversal order or encoding the vector in a huge integer.

For ``n`` left items, ``m`` right items, and ``e`` supplied edges, residual
storage is ``O(n + m + e)``.  Runtime is
``O(n * (n + m + e) * log(n + m) + n * (n + m + e))`` under the hard limits
below.  No dense ``n * m`` matrix is constructed.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
from typing import Iterable, Optional, Sequence, Tuple


ASSIGNMENT_POLICY_VERSION = "maximum_weight_bipartite_assignment_v1"
"""Version of the primary-weight and stable lexicographic selection policy."""

# Compatibility-friendly descriptive alias for consumers that qualify policy
# constants by the result type.
WEIGHTED_ASSIGNMENT_POLICY_VERSION = ASSIGNMENT_POLICY_VERSION

# These limits cover the eval protocol's largest individual item collections
# while preventing an accidentally dense candidate matrix from reaching the
# solver.  Intermediate scalar costs are naturally bounded by
# MAX_ASSIGNMENT_TOTAL_WEIGHT; no positional tie-break encoding is used.
MAX_ASSIGNMENT_ITEMS = 2_048
MAX_ASSIGNMENT_LEFT_ITEMS = MAX_ASSIGNMENT_ITEMS
MAX_ASSIGNMENT_RIGHT_ITEMS = MAX_ASSIGNMENT_ITEMS
MAX_ASSIGNMENT_EDGES = 65_536
MAX_ASSIGNMENT_WEIGHT = 1_000_000_000
MAX_ASSIGNMENT_TOTAL_WEIGHT = MAX_ASSIGNMENT_ITEMS * MAX_ASSIGNMENT_WEIGHT
MAX_ASSIGNMENT_IDENTIFIER_CHARS = 512
_MAX_ASSIGNMENT_PAIR_SPACE = MAX_ASSIGNMENT_LEFT_ITEMS * MAX_ASSIGNMENT_RIGHT_ITEMS


def _identifier(value: object, context: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{context} must be a string")
    if not value:
        raise ValueError(f"{context} must be non-empty")
    try:
        value.encode("utf-8", "strict")
    except UnicodeEncodeError as exc:
        raise ValueError(
            f"{context} must contain valid Unicode scalar values"
        ) from exc
    if len(value) > MAX_ASSIGNMENT_IDENTIFIER_CHARS:
        raise ValueError(
            f"{context} exceeds the character limit of "
            f"{MAX_ASSIGNMENT_IDENTIFIER_CHARS}"
        )
    if value != value.strip() or any(
        character.isspace()
        or ord(character) < 32
        or ord(character) == 127
        for character in value
    ):
        raise ValueError(
            f"{context} must be an opaque identifier without whitespace "
            "or controls"
        )
    return value


def _weight(value: object, context: str) -> int:
    if type(value) is not int:
        raise ValueError(
            f"{context} must be an integer (bool is not accepted)"
        )
    if value < 1 or value > MAX_ASSIGNMENT_WEIGHT:
        raise ValueError(
            f"{context} must be between 1 and {MAX_ASSIGNMENT_WEIGHT}"
        )
    return value


@dataclass(frozen=True)
class WeightedAssignmentEdge:
    """One eligible left-to-right edge with a positive integral weight."""

    left_id: str
    right_id: str
    weight: int

    def __post_init__(self) -> None:
        _identifier(self.left_id, "edge.left_id")
        _identifier(self.right_id, "edge.right_id")
        _weight(self.weight, "edge.weight")


@dataclass(frozen=True)
class AssignedPair:
    """One selected edge in a weighted assignment result."""

    left_id: str
    right_id: str
    weight: int

    def __post_init__(self) -> None:
        _identifier(self.left_id, "match.left_id")
        _identifier(self.right_id, "match.right_id")
        _weight(self.weight, "match.weight")


@dataclass(frozen=True)
class WeightedAssignmentResult:
    """Canonical immutable result of :func:`maximum_weight_bipartite_assignment`."""

    policy_version: str
    matches: Tuple[AssignedPair, ...]
    unmatched_left: Tuple[str, ...]
    unmatched_right: Tuple[str, ...]
    total_weight: int

    def __post_init__(self) -> None:
        if self.policy_version != ASSIGNMENT_POLICY_VERSION:
            raise ValueError("result.policy_version is not supported")
        if type(self.matches) is not tuple:
            raise ValueError("result.matches must be an immutable tuple")
        if type(self.unmatched_left) is not tuple:
            raise ValueError("result.unmatched_left must be an immutable tuple")
        if type(self.unmatched_right) is not tuple:
            raise ValueError("result.unmatched_right must be an immutable tuple")
        if type(self.total_weight) is not int:
            raise ValueError(
                "result.total_weight must be an integer (bool is not accepted)"
            )
        if not 0 <= self.total_weight <= MAX_ASSIGNMENT_TOTAL_WEIGHT:
            raise ValueError("result.total_weight is outside the assignment budget")

        for index, match in enumerate(self.matches):
            if type(match) is not AssignedPair:
                raise ValueError(
                    f"result.matches[{index}] must be an AssignedPair"
                )
        match_order = tuple(
            sorted(self.matches, key=lambda item: (item.left_id, item.right_id))
        )
        if self.matches != match_order:
            raise ValueError("result.matches must be in canonical order")

        matched_left = tuple(match.left_id for match in self.matches)
        matched_right = tuple(match.right_id for match in self.matches)
        if len(set(matched_left)) != len(matched_left):
            raise ValueError("result.matches contains a duplicate left identifier")
        if len(set(matched_right)) != len(matched_right):
            raise ValueError("result.matches contains a duplicate right identifier")

        _validate_canonical_id_tuple(self.unmatched_left, "result.unmatched_left")
        _validate_canonical_id_tuple(self.unmatched_right, "result.unmatched_right")
        if set(matched_left).intersection(self.unmatched_left):
            raise ValueError("a left identifier is both matched and unmatched")
        if set(matched_right).intersection(self.unmatched_right):
            raise ValueError("a right identifier is both matched and unmatched")
        if sum(match.weight for match in self.matches) != self.total_weight:
            raise ValueError("result.total_weight does not equal its match weights")


def _validate_canonical_id_tuple(values: Tuple[str, ...], context: str) -> None:
    for index, value in enumerate(values):
        _identifier(value, f"{context}[{index}]")
    if tuple(sorted(values)) != values:
        raise ValueError(f"{context} must be in canonical order")
    if len(set(values)) != len(values):
        raise ValueError(f"{context} contains a duplicate identifier")


class _ResidualArc:
    __slots__ = ("to", "reverse_index", "capacity", "cost")

    def __init__(
        self, to: int, reverse_index: int, capacity: int, cost: int
    ) -> None:
        self.to = to
        self.reverse_index = reverse_index
        self.capacity = capacity
        self.cost = cost


@dataclass(frozen=True)
class _Choice:
    right_id: Optional[str]
    target_node: int
    arc_index: int
    weight: int


def _bounded_values(
    values: Iterable[object], context: str, maximum: int
) -> list[object]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{context} must be an iterable of items")
    try:
        iterator = iter(values)
    except TypeError as exc:
        raise ValueError(f"{context} must be iterable") from exc

    result: list[object] = []
    for value in iterator:
        if len(result) == maximum:
            raise ValueError(f"{context} exceeds its item limit of {maximum}")
        result.append(value)
    return result


def _canonical_ids(
    values: Iterable[str], context: str, maximum: int
) -> Tuple[str, ...]:
    raw = _bounded_values(values, context, maximum)
    checked = [_identifier(value, f"{context}[{index}]") for index, value in enumerate(raw)]
    if len(set(checked)) != len(checked):
        raise ValueError(f"{context} contains a duplicate identifier")
    return tuple(sorted(checked))


def _canonical_edges(
    values: Iterable[WeightedAssignmentEdge],
    left_ids: Sequence[str],
    right_ids: Sequence[str],
    maximum: int = MAX_ASSIGNMENT_EDGES,
) -> Tuple[WeightedAssignmentEdge, ...]:
    if (
        type(maximum) is not int
        or maximum < 1
        or maximum > _MAX_ASSIGNMENT_PAIR_SPACE
    ):
        raise ValueError("maximum edge budget is outside the assignment pair space")
    raw = _bounded_values(values, "edges", maximum)
    checked: list[WeightedAssignmentEdge] = []
    for index, value in enumerate(raw):
        if not isinstance(value, WeightedAssignmentEdge):
            raise ValueError(
                f"edges[{index}] must be a WeightedAssignmentEdge"
            )
        # Revalidate at the trust boundary even if a hostile subclass bypassed
        # the base dataclass's post-init method.
        _identifier(value.left_id, f"edges[{index}].left_id")
        _identifier(value.right_id, f"edges[{index}].right_id")
        _weight(value.weight, f"edges[{index}].weight")
        checked.append(value)

    left_set = set(left_ids)
    right_set = set(right_ids)
    checked.sort(key=lambda edge: (edge.left_id, edge.right_id, edge.weight))
    previous_endpoints: Optional[tuple[str, str]] = None
    for edge in checked:
        endpoints = (edge.left_id, edge.right_id)
        if endpoints == previous_endpoints:
            raise ValueError(
                "edges contains a duplicate left/right endpoint pair"
            )
        previous_endpoints = endpoints
        if edge.left_id not in left_set:
            raise ValueError(
                f"edge references unknown left identifier {edge.left_id!r}"
            )
        if edge.right_id not in right_set:
            raise ValueError(
                f"edge references unknown right identifier {edge.right_id!r}"
            )
    return tuple(checked)


def _add_arc(
    graph: list[list[_ResidualArc]], source: int, target: int, cost: int
) -> int:
    source_index = len(graph[source])
    target_index = len(graph[target])
    graph[source].append(_ResidualArc(target, target_index, 1, cost))
    graph[target].append(_ResidualArc(source, source_index, 0, -cost))
    return source_index


def _initial_potentials(
    node_count: int,
    left_count: int,
    right_count: int,
    sink: int,
    edges: Sequence[WeightedAssignmentEdge],
    right_index: dict[str, int],
) -> list[int]:
    """Return exact shortest-path potentials for the initial acyclic network."""

    potentials = [0] * node_count
    has_incoming = [False] * right_count
    for edge in edges:
        index = right_index[edge.right_id]
        node = 1 + left_count + index
        candidate = -edge.weight
        if not has_incoming[index] or candidate < potentials[node]:
            potentials[node] = candidate
            has_incoming[index] = True
    potentials[sink] = min(
        [0]
        + [
            potentials[1 + left_count + index]
            for index in range(right_count)
            if has_incoming[index]
        ]
    )
    return potentials


def _send_all_left_flow(
    graph: list[list[_ResidualArc]],
    source: int,
    sink: int,
    units: int,
    potentials: list[int],
) -> None:
    """Compute an exact minimum-cost integral flow of the requested size."""

    node_count = len(graph)
    for _ in range(units):
        distances: list[Optional[int]] = [None] * node_count
        previous_node = [-1] * node_count
        previous_arc = [-1] * node_count
        distances[source] = 0
        queue: list[tuple[int, int]] = [(0, source)]

        while queue:
            distance, node = heapq.heappop(queue)
            if distances[node] != distance:
                continue
            for arc_index, arc in enumerate(graph[node]):
                if arc.capacity == 0:
                    continue
                reduced_cost = (
                    arc.cost + potentials[node] - potentials[arc.to]
                )
                if reduced_cost < 0:
                    raise RuntimeError(
                        "assignment solver lost feasible cost potentials"
                    )
                candidate = distance + reduced_cost
                known = distances[arc.to]
                if known is None or candidate < known:
                    distances[arc.to] = candidate
                    previous_node[arc.to] = node
                    previous_arc[arc.to] = arc_index
                    heapq.heappush(queue, (candidate, arc.to))

        if distances[sink] is None:
            raise RuntimeError("assignment flow network unexpectedly became infeasible")
        for node, distance in enumerate(distances):
            if distance is not None:
                potentials[node] += distance

        node = sink
        while node != source:
            parent = previous_node[node]
            arc_index = previous_arc[node]
            if parent < 0 or arc_index < 0:
                raise RuntimeError("assignment augmenting path is incomplete")
            _augment_arc(graph, parent, arc_index)
            node = parent


def _augment_arc(
    graph: list[list[_ResidualArc]], node: int, arc_index: int
) -> None:
    arc = graph[node][arc_index]
    if arc.capacity <= 0:
        raise RuntimeError("assignment attempted to exceed a residual capacity")
    arc.capacity -= 1
    reverse = graph[arc.to][arc.reverse_index]
    reverse.capacity += 1


def _reduced_cost(
    potentials: Sequence[int], node: int, arc: _ResidualArc
) -> int:
    return arc.cost + potentials[node] - potentials[arc.to]


def _reverse_zero_paths(
    graph: list[list[_ResidualArc]],
    potentials: Sequence[int],
    destination: int,
    source: int,
    locked_left_nodes: set[int],
) -> tuple[list[bool], list[Optional[tuple[int, int]]]]:
    """Find every node with a zero-reduced-cost residual path to destination."""

    reachable = [False] * len(graph)
    next_arc: list[Optional[tuple[int, int]]] = [None] * len(graph)
    reachable[destination] = True
    queue = deque([destination])

    while queue:
        node = queue.popleft()
        for paired_arc in graph[node]:
            predecessor = paired_arc.to
            if predecessor == source or predecessor in locked_left_nodes:
                continue
            if node in locked_left_nodes or reachable[predecessor]:
                continue
            incoming_index = paired_arc.reverse_index
            incoming = graph[predecessor][incoming_index]
            if incoming.capacity == 0:
                continue
            if _reduced_cost(potentials, predecessor, incoming) != 0:
                continue
            reachable[predecessor] = True
            next_arc[predecessor] = (predecessor, incoming_index)
            queue.append(predecessor)

    return reachable, next_arc


def _apply_zero_cycle(
    graph: list[list[_ResidualArc]],
    choice_node: int,
    choice_arc_index: int,
    next_arc: Sequence[Optional[tuple[int, int]]],
) -> None:
    first = graph[choice_node][choice_arc_index]
    target = first.to
    _augment_arc(graph, choice_node, choice_arc_index)

    node = target
    steps_remaining = len(graph)
    while node != choice_node:
        path_arc = next_arc[node]
        if path_arc is None or steps_remaining == 0:
            raise RuntimeError("assignment zero-cost cycle reconstruction failed")
        parent, arc_index = path_arc
        arc = graph[parent][arc_index]
        _augment_arc(graph, parent, arc_index)
        node = arc.to
        steps_remaining -= 1


def _selected_choice(
    graph: Sequence[Sequence[_ResidualArc]],
    left_node: int,
    choices: Sequence[_Choice],
) -> tuple[int, _Choice]:
    selected = [
        (index, choice)
        for index, choice in enumerate(choices)
        if graph[left_node][choice.arc_index].capacity == 0
    ]
    if len(selected) != 1:
        raise RuntimeError("assignment flow does not select exactly one left choice")
    return selected[0]


def _lexicographically_refine(
    graph: list[list[_ResidualArc]],
    choices_by_left: Sequence[Sequence[_Choice]],
    potentials: Sequence[int],
    source: int,
) -> None:
    """Greedily fix the smallest feasible choice for each canonical left ID.

    Feasibility here means preserving the already fixed prefix and the scalar
    optimum.  With feasible min-cost-flow potentials, every residual cost is
    non-negative.  Therefore a different choice belongs to an equal-cost flow
    iff its forward edge can be completed to a cycle consisting entirely of
    zero-reduced-cost residual edges.
    """

    locked_left_nodes: set[int] = set()
    for left_offset, choices in enumerate(choices_by_left):
        left_node = 1 + left_offset
        current_index, _ = _selected_choice(graph, left_node, choices)
        earlier_tight = []
        for choice in choices[:current_index]:
            arc = graph[left_node][choice.arc_index]
            if arc.capacity and _reduced_cost(potentials, left_node, arc) == 0:
                earlier_tight.append(choice)

        if earlier_tight:
            reachable, next_arc = _reverse_zero_paths(
                graph,
                potentials,
                left_node,
                source,
                locked_left_nodes,
            )
            for choice in earlier_tight:
                if reachable[choice.target_node]:
                    _apply_zero_cycle(
                        graph, left_node, choice.arc_index, next_arc
                    )
                    break
        locked_left_nodes.add(left_node)


def maximum_weight_bipartite_assignment(
    left_ids: Iterable[str],
    right_ids: Iterable[str],
    edges: Iterable[WeightedAssignmentEdge],
    *,
    edge_limit: Optional[int] = None,
) -> WeightedAssignmentResult:
    """Return the exact policy-v1 maximum-weight bipartite assignment.

    Every supplied edge is optional.  Each left and right identifier can occur
    in at most one returned match, and all remaining identifiers are returned
    in their canonical unmatched tuple.

    ``edge_limit`` defaults to the generic policy budget.  A caller with its
    own independently enforced, versioned cardinality bound may raise that
    ingestion limit, but never beyond the finite left/right pair space.

    Inputs may be any finite iterable.  They are consumed only through one
    item beyond the applicable hard budget, so even an accidentally unbounded
    iterable is rejected without unbounded consumption.
    """

    canonical_left = _canonical_ids(
        left_ids, "left_ids", MAX_ASSIGNMENT_LEFT_ITEMS
    )
    canonical_right = _canonical_ids(
        right_ids, "right_ids", MAX_ASSIGNMENT_RIGHT_ITEMS
    )
    maximum_edges = MAX_ASSIGNMENT_EDGES if edge_limit is None else edge_limit
    if (
        type(maximum_edges) is not int
        or maximum_edges < 1
        or maximum_edges > _MAX_ASSIGNMENT_PAIR_SPACE
    ):
        raise ValueError("edge_limit is outside the assignment pair space")
    canonical_edges = _canonical_edges(
        edges, canonical_left, canonical_right, maximum_edges
    )

    left_count = len(canonical_left)
    right_count = len(canonical_right)
    if left_count == 0:
        return WeightedAssignmentResult(
            policy_version=ASSIGNMENT_POLICY_VERSION,
            matches=(),
            unmatched_left=(),
            unmatched_right=canonical_right,
            total_weight=0,
        )

    left_index = {identifier: index for index, identifier in enumerate(canonical_left)}
    right_index = {
        identifier: index for index, identifier in enumerate(canonical_right)
    }

    source = 0
    first_left = 1
    first_right = first_left + left_count
    sink = first_right + right_count
    graph: list[list[_ResidualArc]] = [
        [] for _ in range(sink + 1)
    ]

    for offset in range(left_count):
        _add_arc(graph, source, first_left + offset, 0)
    rights_with_edges = {edge.right_id for edge in canonical_edges}
    for offset, identifier in enumerate(canonical_right):
        if identifier in rights_with_edges:
            _add_arc(graph, first_right + offset, sink, 0)

    choices_by_left: list[list[_Choice]] = [
        [] for _ in range(left_count)
    ]
    for edge in canonical_edges:
        left_offset = left_index[edge.left_id]
        right_offset = right_index[edge.right_id]
        left_node = first_left + left_offset
        right_node = first_right + right_offset
        arc_index = _add_arc(graph, left_node, right_node, -edge.weight)
        choices_by_left[left_offset].append(
            _Choice(edge.right_id, right_node, arc_index, edge.weight)
        )
    for left_offset in range(left_count):
        left_node = first_left + left_offset
        arc_index = _add_arc(graph, left_node, sink, 0)
        # Real choices were inserted in canonical right-ID order because the
        # validated edge sequence is sorted.  The abstract unmatched sentinel
        # is appended last, exactly as specified by policy v1.
        choices_by_left[left_offset].append(
            _Choice(None, sink, arc_index, 0)
        )

    potentials = _initial_potentials(
        len(graph),
        left_count,
        right_count,
        sink,
        canonical_edges,
        right_index,
    )
    _send_all_left_flow(graph, source, sink, left_count, potentials)
    _lexicographically_refine(graph, choices_by_left, potentials, source)

    matches: list[AssignedPair] = []
    unmatched_left: list[str] = []
    matched_right: set[str] = set()
    total_weight = 0
    for left_offset, left_id in enumerate(canonical_left):
        left_node = first_left + left_offset
        _, choice = _selected_choice(
            graph, left_node, choices_by_left[left_offset]
        )
        if choice.right_id is None:
            unmatched_left.append(left_id)
            continue
        matches.append(AssignedPair(left_id, choice.right_id, choice.weight))
        matched_right.add(choice.right_id)
        total_weight += choice.weight

    unmatched_right = tuple(
        identifier
        for identifier in canonical_right
        if identifier not in matched_right
    )
    return WeightedAssignmentResult(
        policy_version=ASSIGNMENT_POLICY_VERSION,
        matches=tuple(matches),
        unmatched_left=tuple(unmatched_left),
        unmatched_right=unmatched_right,
        total_weight=total_weight,
    )


__all__ = [
    "ASSIGNMENT_POLICY_VERSION",
    "WEIGHTED_ASSIGNMENT_POLICY_VERSION",
    "MAX_ASSIGNMENT_ITEMS",
    "MAX_ASSIGNMENT_LEFT_ITEMS",
    "MAX_ASSIGNMENT_RIGHT_ITEMS",
    "MAX_ASSIGNMENT_EDGES",
    "MAX_ASSIGNMENT_WEIGHT",
    "MAX_ASSIGNMENT_TOTAL_WEIGHT",
    "MAX_ASSIGNMENT_IDENTIFIER_CHARS",
    "WeightedAssignmentEdge",
    "AssignedPair",
    "WeightedAssignmentResult",
    "maximum_weight_bipartite_assignment",
]
