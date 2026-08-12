from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from importlib import resources
import json
import re
from typing import Any

from review_agent.review_policy import (
    DEFAULT_DEVELOPER_REVIEW_POLICY,
    DeveloperReviewPolicy,
)
from review_agent.review_protocol import ReviewerAssignment
from review_agent.safe_io import canonical_json_bytes


BUILTIN_DEVELOPER_RULE_RESOLVER_VERSION = "assignment-glob-first-match-v1"
_CATALOG_RESOURCE = "system_rules.json"
_RULE_DIRECTORY = "rule_docs"
_MAX_CATALOG_BYTES = 128 * 1024
_MAX_RULE_BYTES = 512 * 1024
_RULE_FILE = re.compile(r"\A[a-z0-9][a-z0-9_-]*\.md\Z")
_SYMBOL_SEPARATOR = "::"
_HUNK_SUFFIX = re.compile(r"#hunk-[0-9]+\Z")
_UNSUPPORTED_TOOL_NAMES = ("`file_read`", "`code_search`")
_QUALITY_EVIDENCE_PREAMBLE = (
    "Quality-tool evidence rule: only treat a compiler, formatter, linter, "
    "type checker, test runner, or static analyzer as having covered an issue "
    "when <PreflightResults> explicitly records that check as successfully "
    "completed. If no such successful check is recorded, do not assume it ran."
)


class DeveloperRuleCatalogError(ValueError):
    pass


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DeveloperRuleCatalogError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_non_finite(token: str) -> None:
    raise DeveloperRuleCatalogError(
        f"catalog contains a non-finite number: {token}"
    )


def _parse_catalog_document(payload: bytes) -> tuple[str, tuple[tuple[str, str], ...]]:
    if type(payload) is not bytes or not payload or len(payload) > _MAX_CATALOG_BYTES:
        raise DeveloperRuleCatalogError("catalog bytes are unavailable or too large")
    try:
        document = json.loads(
            payload.decode("utf-8", "strict"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_non_finite,
        )
    except DeveloperRuleCatalogError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise DeveloperRuleCatalogError("catalog JSON is invalid") from error
    if type(document) is not dict or set(document) != {
        "default_rule",
        "path_rule_map",
    }:
        raise DeveloperRuleCatalogError("catalog fields are not canonical")
    default = _rule_file_name(document["default_rule"])
    mapping = document["path_rule_map"]
    if type(mapping) is not dict:
        raise DeveloperRuleCatalogError("path_rule_map must be an object")
    entries: list[tuple[str, str]] = []
    for pattern, rule_name in mapping.items():
        if type(pattern) is not str or not pattern or "\x00" in pattern:
            raise DeveloperRuleCatalogError("path rule pattern is invalid")
        if "\\" in pattern or pattern.startswith("/"):
            raise DeveloperRuleCatalogError("path rule pattern is not repository-relative")
        entries.append((pattern, _rule_file_name(rule_name)))
    if not entries:
        raise DeveloperRuleCatalogError("path_rule_map must not be empty")
    return default, tuple(entries)


def _rule_file_name(value: object) -> str:
    if type(value) is not str or _RULE_FILE.fullmatch(value) is None:
        raise DeveloperRuleCatalogError("rule file name is unsafe")
    return value


def _read_resource_bytes(node: Any, *, limit: int, label: str) -> bytes:
    try:
        payload = node.read_bytes()
    except (OSError, TypeError) as error:
        raise DeveloperRuleCatalogError(f"{label} is unavailable") from error
    if not payload or len(payload) > limit:
        raise DeveloperRuleCatalogError(f"{label} is empty or too large")
    return payload


def _read_rule(node: Any, name: str) -> "DeveloperRule":
    payload = _read_resource_bytes(
        node.joinpath(name),
        limit=_MAX_RULE_BYTES,
        label=f"developer rule {name}",
    )
    try:
        decoded = payload.decode("utf-8", "strict")
    except UnicodeError as error:
        raise DeveloperRuleCatalogError(
            f"developer rule {name} must be UTF-8"
        ) from error
    content = decoded.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not content or "\x00" in content:
        raise DeveloperRuleCatalogError(f"developer rule {name} is invalid")
    if any(name in content for name in _UNSUPPORTED_TOOL_NAMES):
        raise DeveloperRuleCatalogError(
            f"developer rule {name} references an unsupported tool"
        )
    return DeveloperRule(
        name=name,
        content=content,
        source_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )


def _expand_braces(pattern: str) -> tuple[str, ...]:
    opening = pattern.find("{")
    if opening < 0:
        return (pattern,)
    closing = pattern.find("}", opening + 1)
    if closing < 0:
        return (pattern,)
    options = pattern[opening + 1 : closing].split(",")
    if any(not option for option in options):
        raise DeveloperRuleCatalogError("brace pattern contains an empty option")
    prefix = pattern[:opening]
    suffix = pattern[closing + 1 :]
    expanded: list[str] = []
    for option in options:
        expanded.extend(_expand_braces(prefix + option + suffix))
    return tuple(expanded)


def _match_segments(pattern: tuple[str, ...], path: tuple[str, ...]) -> bool:
    if not pattern:
        return not path
    head = pattern[0]
    if head == "**":
        return _match_segments(pattern[1:], path) or (
            bool(path) and _match_segments(pattern, path[1:])
        )
    return bool(path) and _match_segment(head, path[0]) and _match_segments(
        pattern[1:], path[1:]
    )


@lru_cache(maxsize=2048)
def _match_segment(pattern: str, value: str) -> bool:
    pieces = [r"\A"]
    for character in pattern:
        if character == "*":
            pieces.append("[^/]*")
        elif character == "?":
            pieces.append("[^/]")
        else:
            pieces.append(re.escape(character))
    pieces.append(r"\Z")
    return re.fullmatch("".join(pieces), value) is not None


def _matches(pattern: str, path: str) -> bool:
    normalized = path.casefold()
    return any(
        _match_segments(
            tuple(part for part in expanded.casefold().split("/") if part),
            tuple(part for part in normalized.split("/") if part),
        )
        for expanded in _expand_braces(pattern)
    )


def _assignment_paths(assignment: ReviewerAssignment) -> tuple[str, ...]:
    if type(assignment) is not ReviewerAssignment:
        raise DeveloperRuleCatalogError("assignment must be a ReviewerAssignment")
    paths: set[str] = set(assignment.targets.files)
    for symbol in assignment.targets.symbols:
        path, separator, _name = symbol.partition(_SYMBOL_SEPARATOR)
        if separator and path:
            paths.add(path)
    for hunk in assignment.targets.hunks:
        path = _HUNK_SUFFIX.sub("", hunk)
        if path != hunk:
            paths.add(path)
    return tuple(sorted(paths, key=str.casefold))


@dataclass(frozen=True)
class DeveloperRule:
    name: str
    content: str
    source_sha256: str


@dataclass(frozen=True)
class DeveloperRulePattern:
    pattern: str
    rule_name: str


@dataclass(frozen=True)
class DeveloperRuleCatalog:
    resolver_version: str
    default_rule: str
    patterns: tuple[DeveloperRulePattern, ...]
    rules: tuple[DeveloperRule, ...]
    digest: str

    def rule_for_path(self, path: str) -> DeveloperRule:
        if (
            type(path) is not str
            or not path
            or "\x00" in path
            or "\\" in path
            or path.startswith("/")
            or path.endswith("/")
            or ":" in path
            or any(part in {"", ".", ".."} for part in path.split("/"))
        ):
            raise DeveloperRuleCatalogError("path must be a canonical repository path")
        rule_by_name = {rule.name: rule for rule in self.rules}
        for entry in self.patterns:
            if _matches(entry.pattern, path):
                return rule_by_name[entry.rule_name]
        return rule_by_name[self.default_rule]

    def to_manifest(self) -> dict[str, Any]:
        return _catalog_manifest(
            resolver_version=self.resolver_version,
            default_rule=self.default_rule,
            patterns=self.patterns,
            rules=self.rules,
        )


def _catalog_manifest(
    *,
    resolver_version: str,
    default_rule: str,
    patterns: tuple[DeveloperRulePattern, ...],
    rules: tuple[DeveloperRule, ...],
) -> dict[str, Any]:
    return {
        "resolver_version": resolver_version,
        "default_rule": default_rule,
        "path_rule_map": [
            {"pattern": item.pattern, "rule": item.rule_name}
            for item in patterns
        ],
        "rules": [
            {
                "name": rule.name,
                "content_sha256": hashlib.sha256(
                    rule.content.encode("utf-8")
                ).hexdigest(),
                "source_sha256": rule.source_sha256,
            }
            for rule in rules
        ],
    }


@lru_cache(maxsize=1)
def load_builtin_developer_rule_catalog() -> DeveloperRuleCatalog:
    package = resources.files(__package__)
    catalog_bytes = _read_resource_bytes(
        package.joinpath(_CATALOG_RESOURCE),
        limit=_MAX_CATALOG_BYTES,
        label="developer rule catalog",
    )
    default, entries = _parse_catalog_document(catalog_bytes)
    referenced_names = tuple(dict.fromkeys([default, *(name for _, name in entries)]))
    rule_root = package.joinpath(_RULE_DIRECTORY)
    rules = tuple(_read_rule(rule_root, name) for name in referenced_names)
    present_names = {rule.name for rule in rules}
    if present_names != set(referenced_names):
        raise DeveloperRuleCatalogError("developer rule catalog is incomplete")
    try:
        packaged_names = {
            item.name
            for item in rule_root.iterdir()
            if item.is_file() and item.name.endswith(".md")
        }
    except (OSError, TypeError) as error:
        raise DeveloperRuleCatalogError("developer rule directory is unavailable") from error
    if packaged_names != present_names:
        raise DeveloperRuleCatalogError("developer rule directory has unregistered files")
    patterns = tuple(
        DeveloperRulePattern(pattern=pattern, rule_name=name)
        for pattern, name in entries
    )
    manifest = _catalog_manifest(
        resolver_version=BUILTIN_DEVELOPER_RULE_RESOLVER_VERSION,
        default_rule=default,
        patterns=patterns,
        rules=rules,
    )
    return DeveloperRuleCatalog(
        resolver_version=BUILTIN_DEVELOPER_RULE_RESOLVER_VERSION,
        default_rule=default,
        patterns=patterns,
        rules=rules,
        digest=hashlib.sha256(canonical_json_bytes(manifest)).hexdigest(),
    )


class DeveloperRuleResolver:
    def __init__(self) -> None:
        self._catalog = load_builtin_developer_rule_catalog()

    @property
    def catalog_digest(self) -> str:
        return self._catalog.digest

    def rules_for_assignment(
        self,
        assignment: ReviewerAssignment,
    ) -> tuple[DeveloperRule, ...]:
        selected: dict[str, DeveloperRule] = {}
        for path in _assignment_paths(assignment):
            rule = self._catalog.rule_for_path(path)
            selected[rule.name] = rule
        if not selected:
            default = self._catalog.rule_for_path("unknown")
            selected[default.name] = default
        order = {rule.name: index for index, rule in enumerate(self._catalog.rules)}
        return tuple(sorted(selected.values(), key=lambda rule: order[rule.name]))

    def policy_for_assignment(
        self,
        assignment: ReviewerAssignment,
    ) -> DeveloperReviewPolicy:
        rules = self.rules_for_assignment(assignment)
        sections = [
            DEFAULT_DEVELOPER_REVIEW_POLICY.content,
            _QUALITY_EVIDENCE_PREAMBLE,
        ]
        sections.extend(
            f'<BuiltInReviewRule name="{rule.name}">\n{rule.content}\n</BuiltInReviewRule>'
            for rule in rules
        )
        identity = {
            "catalog_sha256": self._catalog.digest,
            "rule_names": [rule.name for rule in rules],
        }
        return DeveloperReviewPolicy(
            policy_id=(
                "builtin-assignment-policy-"
                + hashlib.sha256(canonical_json_bytes(identity)).hexdigest()[:24]
            ),
            content="\n\n".join(sections),
            locked_topics=DEFAULT_DEVELOPER_REVIEW_POLICY.locked_topics,
        )


__all__ = [
    "BUILTIN_DEVELOPER_RULE_RESOLVER_VERSION",
    "DeveloperRule",
    "DeveloperRuleCatalog",
    "DeveloperRuleCatalogError",
    "DeveloperRulePattern",
    "DeveloperRuleResolver",
    "load_builtin_developer_rule_catalog",
]
