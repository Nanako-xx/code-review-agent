from __future__ import annotations

from dataclasses import asdict, dataclass, field
import ast
import io
import math
from pathlib import Path, PurePosixPath
import re
import subprocess
import time
import tokenize
from typing import Any, Callable

try:  # Python 3.11+; tests also exercise the package under an older interpreter.
    import tomllib as _tomllib
except ModuleNotFoundError:  # pragma: no cover - selected by the test runtime.
    _tomllib = None

from review_agent.models import (
    Assignment,
    QualityGateResult,
    RiskAssessment,
)


QUALITY_GATE_STATUSES = frozenset(
    {"passed", "failed", "skipped", "unavailable", "timed_out", "error"}
)
QUALITY_GATE_CATEGORIES = frozenset(
    {"compile", "format", "type", "lint", "build", "test", "security"}
)
QUALITY_GATE_COSTS = frozenset({"cheap", "expensive"})
QUALITY_GATE_SOURCES = frozenset({"builtin", "repository_config"})
RISK_LEVELS = ("low", "medium", "high", "critical")
DEFAULT_CHEAP_TIMEOUT_SECONDS = 60.0
DEFAULT_EXPENSIVE_TIMEOUT_SECONDS = 300.0
# Discovery reads immutable Git objects rather than running repository code.
# Large official repositories backed by verified loose-object stores can need
# several minutes for ls-tree/cat-file on Windows, so keep this watchdog
# finite but aligned with the evaluation repository watchdog.
DEFAULT_GIT_TIMEOUT_SECONDS = 3600.0
MAX_CONFIG_BYTES = 1_048_576
MAX_REVISION_BLOB_BYTES = 268_435_456
MAX_REVISION_FILES = 20_000

_GATE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_MODULE_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
_ALLOWED_GATE_MODULES = {
    "compile": frozenset(),
    "format": frozenset({"ruff"}),
    "type": frozenset({"mypy", "pyright"}),
    "lint": frozenset({"ruff"}),
    "build": frozenset(),
    "test": frozenset({"pytest", "unittest"}),
    "security": frozenset({"bandit", "pip_audit"}),
}
_MUTATING_GATE_ARGUMENTS = frozenset(
    {"--fix", "--unsafe-fixes", "--update", "install", "uninstall"}
)


@dataclass(frozen=True)
class QualityGateDefinition:
    name: str
    category: str
    cost: str
    source: str
    command: list[str]
    blocking: bool = False
    timeout_seconds: float = DEFAULT_CHEAP_TIMEOUT_SECONDS
    trigger_risks: list[str] = field(default_factory=lambda: list(RISK_LEVELS))

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _GATE_NAME_PATTERN.fullmatch(
            self.name
        ):
            raise ValueError("quality gate name must be a stable lowercase identifier")
        if self.category not in QUALITY_GATE_CATEGORIES:
            raise ValueError("quality gate category is unsupported")
        if self.cost not in QUALITY_GATE_COSTS:
            raise ValueError("quality gate cost is unsupported")
        if self.source not in QUALITY_GATE_SOURCES:
            raise ValueError("quality gate source is unsupported")
        if not isinstance(self.command, list) or not self.command:
            raise ValueError("quality gate command must be a non-empty argv list")
        for argument in self.command:
            if (
                not isinstance(argument, str)
                or not argument
                or argument != argument.strip()
                or "\x00" in argument
                or "\n" in argument
                or "\r" in argument
            ):
                raise ValueError("quality gate command contains an invalid argument")
        if type(self.blocking) is not bool:
            raise ValueError("quality gate blocking must be a boolean")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("quality gate timeout_seconds must be positive and finite")
        trigger_risks = list(self.trigger_risks or RISK_LEVELS)
        if (
            not trigger_risks
            or len(set(trigger_risks)) != len(trigger_risks)
            or any(item not in RISK_LEVELS for item in trigger_risks)
        ):
            raise ValueError("quality gate trigger_risks are invalid")
        _validate_gate_command(
            self.name,
            self.category,
            self.command,
            self.source,
        )
        object.__setattr__(self, "command", list(self.command))
        object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))
        object.__setattr__(self, "trigger_risks", trigger_risks)


@dataclass(frozen=True)
class QualityGatePlan:
    revision: str
    gates: list[QualityGateDefinition]
    discovery_issues: list[str]

    def __post_init__(self) -> None:
        if not isinstance(self.revision, str) or not self.revision:
            raise ValueError("quality gate plan revision must be non-empty")
        if not isinstance(self.gates, list) or not all(
            isinstance(item, QualityGateDefinition) for item in self.gates
        ):
            raise ValueError("quality gate plan gates must be definitions")
        names = [gate.name for gate in self.gates]
        if len(names) != len(set(names)):
            raise ValueError("quality gate plan contains duplicate names")
        if not isinstance(self.discovery_issues, list) or any(
            not isinstance(item, str) or not item.strip()
            for item in self.discovery_issues
        ):
            raise ValueError("quality gate discovery_issues must be non-empty strings")
        object.__setattr__(self, "gates", list(self.gates))
        object.__setattr__(self, "discovery_issues", list(self.discovery_issues))


@dataclass(frozen=True)
class QualityGateExecution:
    result: QualityGateResult
    raw_output: str

    def __post_init__(self) -> None:
        if not isinstance(self.result, QualityGateResult):
            raise ValueError("quality gate execution result is invalid")
        if not isinstance(self.raw_output, str):
            raise ValueError("quality gate execution raw_output must be text")


@dataclass(frozen=True)
class _TreeBlob:
    path: str
    object_id: str
    mode: str


def discover_quality_gate_plan(repo_path: Path, revision: str) -> QualityGatePlan:
    repo = Path(repo_path)
    blobs = _tree_blobs_at_revision(repo, revision)
    by_path = {blob.path: blob for blob in blobs}
    paths = set(by_path)
    issues: list[str] = []
    pyproject: dict[str, Any] = {}
    if "pyproject.toml" in by_path:
        raw = _read_blob(repo, by_path["pyproject.toml"].object_id)
        if len(raw) > MAX_CONFIG_BYTES:
            issues.append("pyproject.toml exceeds the Quality Gate config size limit")
        else:
            try:
                parsed = _parse_toml(raw.decode("utf-8"))
                if isinstance(parsed, dict):
                    pyproject = parsed
            except (UnicodeDecodeError, ValueError) as error:
                issues.append(f"pyproject.toml could not be parsed: {error}")

    definitions = _builtin_gate_definitions(
        paths,
        pyproject,
        read_text=lambda path: _revision_text(repo, by_path, path),
    )
    custom, custom_issues = _configured_gate_definitions(pyproject)
    issues.extend(custom_issues)
    existing = {gate.name for gate in definitions}
    for gate in custom:
        if gate.name in existing:
            issues.append(
                f"configured Quality Gate duplicates discovered name: {gate.name}"
            )
            continue
        existing.add(gate.name)
        definitions.append(gate)
    return QualityGatePlan(
        revision=revision,
        gates=definitions,
        discovery_issues=_dedupe(issues),
    )


def quality_gate_plan_to_dict(plan: QualityGatePlan) -> dict[str, Any]:
    return {
        "revision": plan.revision,
        "gates": [asdict(gate) for gate in plan.gates],
        "discovery_issues": list(plan.discovery_issues),
    }


def detect_quality_gates(repo_path: Path, revision: str | None = None) -> list[str]:
    if revision is not None:
        return [
            gate.name for gate in discover_quality_gate_plan(repo_path, revision).gates
        ]
    repo = Path(repo_path)
    paths = {
        path.relative_to(repo).as_posix()
        for path in repo.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and ".git" not in path.parts
        and ".review-agent" not in path.parts
    }
    pyproject: dict[str, Any] = {}
    pyproject_path = repo / "pyproject.toml"
    if pyproject_path.is_file() and pyproject_path.stat().st_size <= MAX_CONFIG_BYTES:
        try:
            pyproject = _parse_toml(pyproject_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError):
            pyproject = {}
    definitions = _builtin_gate_definitions(
        paths,
        pyproject,
        read_text=lambda path: _worktree_text(repo, path),
    )
    custom, _issues = _configured_gate_definitions(pyproject)
    existing = {gate.name for gate in definitions}
    definitions.extend(gate for gate in custom if gate.name not in existing)
    return [gate.name for gate in definitions]


def run_python_compile_gate(
    repo_path: Path,
    revision: str | None = None,
) -> QualityGateResult:
    started = time.monotonic()
    try:
        if revision is None:
            python_paths = [
                path
                for path in Path(repo_path).rglob("*.py")
                if ".git" not in path.parts
                and ".review-agent" not in path.parts
                and not path.is_symlink()
            ]
            _validate_compile_file_count(len(python_paths))
            python_sources = [
                (str(path), path.read_bytes()) for path in python_paths
            ]
        else:
            python_blobs = _python_blobs_at_revision(
                Path(repo_path),
                revision,
            )
            _validate_compile_file_count(len(python_blobs))
            blob_contents = _read_blobs(
                Path(repo_path),
                [object_id for _relative_path, object_id in python_blobs],
                max_total_bytes=MAX_REVISION_BLOB_BYTES,
            )
            python_sources = [
                (
                    f"{revision}:{relative_path}",
                    blob_contents[object_id],
                )
                for relative_path, object_id in python_blobs
            ]
    except (OSError, RuntimeError, UnicodeError) as error:
        return _errored_python_compile_result(
            error,
            duration_seconds=time.monotonic() - started,
        )

    for filename, raw_source in python_sources:
        try:
            source = _decode_python_source(raw_source)
            compile(source, filename, "exec")
        except (LookupError, SyntaxError, UnicodeDecodeError) as exc:
            return _failed_python_compile_result(
                filename,
                exc,
                duration_seconds=time.monotonic() - started,
            )

    return QualityGateResult(
        name="python_compile",
        status="passed",
        command=["python", "-c", "compile(source, filename, 'exec')"],
        summary=f"Compiled {len(python_sources)} Python files",
        category="compile",
        cost="cheap",
        source="builtin",
        duration_seconds=time.monotonic() - started,
        sandbox="git_blob_compile",
    )


def quality_gate_policy_decision(
    gate: QualityGateDefinition,
    risk: RiskAssessment,
    assignments: list[Assignment],
) -> tuple[bool, str]:
    if gate.cost != "expensive":
        raise ValueError("deep Quality Gate policy accepts only expensive gates")
    if risk.level.value in gate.trigger_risks:
        return True, f"risk level {risk.level.value} is in gate trigger policy"
    roles = [assignment.role.casefold() for assignment in assignments]
    if gate.category == "security" and any("security" in role for role in roles):
        return True, "Security Specialist Reviewer requires the gate"
    focus = [item.casefold() for item in risk.suggested_focus]
    if gate.category == "test" and any("test adequacy" in item for item in focus):
        return True, "risk investigation focus requires test adequacy evidence"
    return (
        False,
        f"risk level {risk.level.value} and reviewer portfolio did not trigger the gate",
    )


def _builtin_gate_definitions(
    paths: set[str],
    pyproject: dict[str, Any],
    *,
    read_text: Callable[[str], str],
) -> list[QualityGateDefinition]:
    tool_value = pyproject.get("tool", {})
    tool = tool_value if isinstance(tool_value, dict) else {}
    definitions: list[QualityGateDefinition] = []
    if any(PurePosixPath(path).suffix == ".py" for path in paths):
        definitions.append(
            QualityGateDefinition(
                name="python_compile",
                category="compile",
                cost="cheap",
                source="builtin",
                command=["python", "-c", "compile(source, filename, 'exec')"],
                trigger_risks=list(RISK_LEVELS),
            )
        )
    if "ruff.toml" in paths or ".ruff.toml" in paths or "ruff" in tool:
        definitions.append(
            _python_module_gate("ruff", "lint", "cheap", ["check", "."])
        )
    setup_cfg = read_text("setup.cfg") if "setup.cfg" in paths else ""
    if (
        "mypy.ini" in paths
        or ".mypy.ini" in paths
        or "mypy" in tool
        or "[mypy]" in setup_cfg.casefold()
    ):
        definitions.append(_python_module_gate("mypy", "type", "cheap", ["."]))
    if "pyrightconfig.json" in paths or "pyright" in tool:
        definitions.append(_python_module_gate("pyright", "type", "cheap", []))
    has_pytest_config = (
        "pytest.ini" in paths
        or "pytest" in tool
        or "[pytest]" in setup_cfg.casefold()
        or "[tool:pytest]" in setup_cfg.casefold()
    )
    has_tests = any(_looks_like_pytest_path(path) for path in paths)
    if has_pytest_config or has_tests:
        definitions.append(
            _python_module_gate(
                "pytest",
                "test",
                "expensive",
                ["-q"],
                timeout_seconds=DEFAULT_EXPENSIVE_TIMEOUT_SECONDS,
                trigger_risks=["medium", "high", "critical"],
            )
        )
    return definitions


def _python_module_gate(
    name: str,
    category: str,
    cost: str,
    arguments: list[str],
    *,
    timeout_seconds: float = DEFAULT_CHEAP_TIMEOUT_SECONDS,
    trigger_risks: list[str] | None = None,
) -> QualityGateDefinition:
    return QualityGateDefinition(
        name=name,
        category=category,
        cost=cost,
        source="builtin",
        command=["python", "-m", name, *arguments],
        timeout_seconds=timeout_seconds,
        trigger_risks=trigger_risks or list(RISK_LEVELS),
    )


def _configured_gate_definitions(
    pyproject: dict[str, Any],
) -> tuple[list[QualityGateDefinition], list[str]]:
    tool = pyproject.get("tool", {})
    if not isinstance(tool, dict):
        return [], []
    review_agent = tool.get("review-agent", tool.get("review_agent", {}))
    if review_agent in ({}, None):
        return [], []
    if not isinstance(review_agent, dict):
        return [], ["tool.review-agent must be a TOML table"]
    rows = review_agent.get("quality-gates", review_agent.get("quality_gates", []))
    if rows in ([], None):
        return [], []
    if not isinstance(rows, list):
        return [], ["tool.review-agent.quality-gates must be an array of tables"]

    definitions: list[QualityGateDefinition] = []
    issues: list[str] = []
    for index, row in enumerate(rows):
        context = f"tool.review-agent.quality-gates[{index}]"
        if not isinstance(row, dict):
            issues.append(f"{context} must be a table")
            continue
        allowed = {
            "name",
            "category",
            "cost",
            "command",
            "blocking",
            "timeout_seconds",
            "trigger_risks",
        }
        unknown = sorted(set(row) - allowed)
        if unknown:
            issues.append(f"{context} has unknown fields: {', '.join(unknown)}")
            continue
        try:
            name = row["name"]
            category = row.get("category", "security")
            cost = row.get("cost", "expensive")
            command = row["command"]
            blocking = row.get("blocking", False)
            timeout = row.get(
                "timeout_seconds",
                DEFAULT_EXPENSIVE_TIMEOUT_SECONDS
                if cost == "expensive"
                else DEFAULT_CHEAP_TIMEOUT_SECONDS,
            )
            triggers = row.get("trigger_risks")
            if triggers is None:
                triggers = (
                    list(RISK_LEVELS)
                    if blocking or cost == "cheap"
                    else ["high", "critical"]
                )
            definition = QualityGateDefinition(
                name=name,
                category=category,
                cost=cost,
                source="repository_config",
                command=command,
                blocking=blocking,
                timeout_seconds=timeout,
                trigger_risks=triggers,
            )
        except (KeyError, TypeError, ValueError) as error:
            issues.append(f"{context} is invalid: {error}")
            continue
        definitions.append(definition)
    return definitions, issues


def _validate_gate_command(
    name: str,
    category: str,
    command: list[str],
    source: str,
) -> None:
    if name == "python_compile" and source == "builtin":
        return
    if len(command) < 3 or command[0] != "python" or command[1] != "-m":
        raise ValueError("Quality Gate commands must use explicit 'python -m module' argv")
    module = command[2]
    if not _MODULE_NAME_PATTERN.fullmatch(module):
        raise ValueError("Quality Gate Python module name is invalid")
    if module not in _ALLOWED_GATE_MODULES[category]:
        raise ValueError(
            f"Python module {module!r} is not allowed for {category} Quality Gates"
        )
    for argument in command[3:]:
        normalized = argument.replace("\\", "/")
        if normalized.casefold() in _MUTATING_GATE_ARGUMENTS:
            raise ValueError("Quality Gate command contains a mutating argument")
        if (
            PurePosixPath(normalized).is_absolute()
            or re.match(r"^[A-Za-z]:/", normalized)
            or normalized.startswith("//")
            or re.search(r"(?:^|[=/])\.\.(?:$|/)", normalized)
            or "://" in normalized
        ):
            raise ValueError("Quality Gate command arguments may not escape the snapshot")


def _decode_python_source(raw_source: bytes) -> str:
    encoding, _ = tokenize.detect_encoding(io.BytesIO(raw_source).readline)
    return raw_source.decode(encoding)


def _failed_python_compile_result(
    filename: str,
    error: Exception,
    *,
    duration_seconds: float,
) -> QualityGateResult:
    return QualityGateResult(
        name="python_compile",
        status="failed",
        command=["python", "-c", "compile(source, filename, 'exec')"],
        summary=f"{filename}: {type(error).__name__}: {error}",
        category="compile",
        cost="cheap",
        source="builtin",
        exit_code=1,
        duration_seconds=duration_seconds,
        sandbox="git_blob_compile",
    )


def _errored_python_compile_result(
    error: Exception,
    *,
    duration_seconds: float,
) -> QualityGateResult:
    reason = f"{type(error).__name__}: {error}"
    return QualityGateResult(
        name="python_compile",
        status="error",
        command=["python", "-c", "compile(source, filename, 'exec')"],
        summary=f"python_compile runner error: {reason}",
        category="compile",
        cost="cheap",
        source="builtin",
        reason=reason,
        duration_seconds=duration_seconds,
        sandbox="git_blob_compile",
    )


def _validate_compile_file_count(file_count: int) -> None:
    if file_count > MAX_REVISION_FILES:
        raise RuntimeError(
            "Python compile input exceeds file limit "
            f"({file_count} > {MAX_REVISION_FILES})"
        )


def _python_blobs_at_revision(
    repo_path: Path,
    revision: str,
) -> list[tuple[str, str]]:
    return [
        (blob.path, blob.object_id)
        for blob in _tree_blobs_at_revision(repo_path, revision)
        if blob.mode in {"100644", "100755"}
        and PurePosixPath(blob.path).suffix == ".py"
    ]


def _tree_blobs_at_revision(repo_path: Path, revision: str) -> list[_TreeBlob]:
    result = _run_git(
        repo_path,
        ["ls-tree", "-r", "-z", "--full-tree", revision, "--"],
    )
    blobs: list[_TreeBlob] = []
    for record in result.stdout.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode, object_type, object_id = metadata.split(b" ", 2)
        except ValueError as error:
            raise RuntimeError("git ls-tree returned a malformed entry") from error
        if object_type not in {b"blob", b"commit"}:
            continue
        path = raw_path.decode("utf-8", errors="surrogateescape")
        blobs.append(
            _TreeBlob(
                path=path,
                object_id=object_id.decode("ascii"),
                mode=mode.decode("ascii"),
            )
        )
    return blobs


def _revision_text(
    repo: Path,
    by_path: dict[str, _TreeBlob],
    path: str,
) -> str:
    blob = by_path.get(path)
    if blob is None:
        return ""
    raw = _read_blob(repo, blob.object_id)
    if len(raw) > MAX_CONFIG_BYTES:
        return ""
    return raw.decode("utf-8", errors="replace")


def _worktree_text(repo: Path, path: str) -> str:
    target = repo / path
    try:
        if not target.is_file() or target.stat().st_size > MAX_CONFIG_BYTES:
            return ""
        return target.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _read_blob(repo_path: Path, object_id: str) -> bytes:
    return _run_git(repo_path, ["cat-file", "blob", object_id]).stdout


def _read_blobs(
    repo_path: Path,
    object_ids: list[str],
    *,
    max_total_bytes: int,
) -> dict[str, bytes]:
    if type(max_total_bytes) is not int or max_total_bytes < 1:
        raise ValueError("max_total_bytes must be a positive integer")
    unique_ids = list(dict.fromkeys(object_ids))
    if not unique_ids:
        return {}
    request = b"".join(object_id.encode("ascii") + b"\n" for object_id in unique_ids)
    size_output = _run_git(
        repo_path,
        ["cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
        input_data=request,
    ).stdout
    size_lines = size_output.splitlines()
    if len(size_lines) != len(unique_ids):
        raise RuntimeError("git cat-file --batch-check returned an unexpected result")
    total_bytes = 0
    for object_id, header in zip(unique_ids, size_lines):
        try:
            returned_id, object_type, raw_size = header.split(b" ", 2)
            size = int(raw_size)
        except (ValueError, UnicodeError) as error:
            raise RuntimeError(
                "git cat-file --batch-check returned an invalid header"
            ) from error
        if (
            returned_id.decode("ascii").casefold() != object_id.casefold()
            or object_type != b"blob"
            or size < 0
        ):
            raise RuntimeError("git cat-file --batch-check returned an unexpected object")
        total_bytes += size
        if total_bytes > max_total_bytes:
            raise RuntimeError(
                "revision blobs exceed byte limit "
                f"({total_bytes} > {max_total_bytes})"
            )

    stream = io.BytesIO(
        _run_git(repo_path, ["cat-file", "--batch"], input_data=request).stdout
    )
    contents: dict[str, bytes] = {}
    for object_id in unique_ids:
        header = stream.readline()
        try:
            returned_id, object_type, raw_size = header.rstrip(b"\n").split(b" ", 2)
            size = int(raw_size)
        except (ValueError, UnicodeError) as error:
            raise RuntimeError("git cat-file --batch returned an invalid header") from error
        if (
            returned_id.decode("ascii").casefold() != object_id.casefold()
            or object_type != b"blob"
            or size < 0
        ):
            raise RuntimeError("git cat-file --batch returned an unexpected object")
        raw = _read_exact(stream, size)
        if stream.read(1) != b"\n":
            raise RuntimeError("git cat-file --batch omitted the blob delimiter")
        contents[object_id] = raw
    if stream.read(1):
        raise RuntimeError("git cat-file --batch returned trailing data")
    return contents


def _read_exact(stream: Any, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise RuntimeError("git cat-file --batch ended before the blob was complete")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _run_git(
    repo_path: Path,
    args: list[str],
    *,
    input_data: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=Path(repo_path),
            check=False,
            input=input_data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=DEFAULT_GIT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("Git command timed out") from error
    except OSError as error:
        raise RuntimeError(f"unable to execute Git: {error}") from error
    if result.returncode != 0:
        reason = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Git command failed: {reason or 'unknown Git error'}")
    return result


def _looks_like_pytest_path(path: str) -> bool:
    candidate = PurePosixPath(path)
    if candidate.suffix != ".py":
        return False
    name = candidate.name
    return (
        name.startswith("test_")
        or name.endswith("_test.py")
        or "tests" in candidate.parts
    )


def _parse_toml(content: str) -> dict[str, Any]:
    if _tomllib is not None:
        return _tomllib.loads(content)
    return _parse_minimal_toml(content)


def _parse_minimal_toml(content: str) -> dict[str, Any]:
    """Parse the Quality Gate subset on pre-3.11 Python.

    The package targets Python 3.11+, where ``tomllib`` is authoritative. This
    compatibility path understands ordinary tables, arrays of tables, scalar
    values, and one-line arrays so the persisted test/runtime behavior remains
    available under older local interpreters without adding a network-fetched
    dependency.
    """

    root: dict[str, Any] = {}
    current = root
    for line_number, raw_line in enumerate(content.splitlines(), start=1):
        line = _strip_toml_comment(raw_line).strip()
        if not line:
            continue
        if line.startswith("[[") and line.endswith("]]"):
            parts = _toml_key_parts(line[2:-2].strip(), line_number)
            parent = _toml_table(root, parts[:-1], line_number)
            name = parts[-1]
            rows = parent.setdefault(name, [])
            if not isinstance(rows, list):
                raise ValueError(f"TOML line {line_number}: table type conflict")
            row: dict[str, Any] = {}
            rows.append(row)
            current = row
            continue
        if line.startswith("[") and line.endswith("]"):
            parts = _toml_key_parts(line[1:-1].strip(), line_number)
            current = _toml_table(root, parts, line_number)
            continue
        if "=" not in line:
            raise ValueError(f"TOML line {line_number}: expected key/value pair")
        raw_key, raw_value = line.split("=", 1)
        key_parts = _toml_key_parts(raw_key.strip(), line_number)
        target = _toml_table(current, key_parts[:-1], line_number)
        key = key_parts[-1]
        if key in target:
            raise ValueError(f"TOML line {line_number}: duplicate key {key}")
        target[key] = _minimal_toml_value(raw_value.strip(), line_number)
    return root


def _toml_table(
    root: dict[str, Any],
    parts: list[str],
    line_number: int,
) -> dict[str, Any]:
    current = root
    for part in parts:
        value = current.setdefault(part, {})
        if not isinstance(value, dict):
            raise ValueError(f"TOML line {line_number}: table type conflict")
        current = value
    return current


def _toml_key_parts(raw: str, line_number: int) -> list[str]:
    parts = [part.strip().strip('"\'') for part in raw.split(".")]
    if not parts or any(not part for part in parts):
        raise ValueError(f"TOML line {line_number}: invalid key")
    return parts


def _minimal_toml_value(raw: str, line_number: int) -> Any:
    if raw == "true":
        return True
    if raw == "false":
        return False
    try:
        return ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        try:
            if any(marker in raw.casefold() for marker in ("nan", "inf")):
                raise ValueError
            return float(raw) if any(marker in raw for marker in (".", "e", "E")) else int(raw)
        except ValueError:
            # Unrelated TOML date/inline values need not prevent detection of
            # tool tables. Explicit Quality Gate fields still fail strict
            # validation when they reach _configured_gate_definitions().
            return raw


def _strip_toml_comment(line: str) -> str:
    quote: str | None = None
    escaped = False
    for index, character in enumerate(line):
        if escaped:
            escaped = False
            continue
        if character == "\\" and quote == '"':
            escaped = True
            continue
        if quote is not None:
            if character == quote:
                quote = None
            continue
        if character in {'"', "'"}:
            quote = character
            continue
        if character == "#":
            return line[:index]
    return line


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
