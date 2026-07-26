"""Public API for the canonical code-review evaluation v2 protocols."""

from .models import *
from .models import __all__

# Runner is kept out of the canonical model wildcard to avoid importing
# subprocess/repository orchestration for protocol-only consumers.  The
# symbols are nevertheless available from the package root for the CLI layer.
from .runner import (
    ADAPTER_IDENTITY_MISMATCH,
    AgentRunner,
    AdapterDiagnostic,
    CapabilityIssue,
    CapabilityPolicy,
    CapabilityPreflight,
    ClarificationScriptProvider,
    EvalRunner,
    PreflightMode,
    RunnerError,
    RunIncompatibilityError,
    RunResult,
    RunSetup,
    TrialResult,
    TrialRunner,
)
from .evidence_checker import *
from .evidence_checker import __all__ as _evidence_checker_all
from .match_location import *
from .match_location import __all__ as _match_location_all
from .assignment import *
from .assignment import __all__ as _assignment_all
from .intent_evaluator import *
from .intent_evaluator import __all__ as _intent_evaluator_all
from .judge_exports import JUDGE_PUBLIC_NAMES as _judge_all
from .review_exports import REVIEW_PUBLIC_NAMES as _review_all
from .metrics_exports import METRICS_PUBLIC_NAMES as _metrics_all
from .report_exports import REPORT_PUBLIC_NAMES as _report_all
from .analysis_exports import ANALYSIS_PUBLIC_NAMES as _analysis_all
from .materialization import (
    MaterializationError,
    MaterializationRequest,
    PreparedTargetMaterialization,
    RepositoryTargetMaterializer,
    TargetMaterializer,
    repository_replay_binding_digest,
)

__all__ = list(__all__) + [
    "ADAPTER_IDENTITY_MISMATCH",
    "AgentRunner",
    "AdapterDiagnostic",
    "CapabilityIssue",
    "CapabilityPolicy",
    "CapabilityPreflight",
    "ClarificationScriptProvider",
    "EvalRunner",
    "PreflightMode",
    "RunnerError",
    "RunIncompatibilityError",
    "RunResult",
    "RunSetup",
    "TrialResult",
    "TrialRunner",
    "MaterializationError",
    "MaterializationRequest",
    "PreparedTargetMaterialization",
    "RepositoryTargetMaterializer",
    "TargetMaterializer",
    "repository_replay_binding_digest",
] + list(_match_location_all) + list(_evidence_checker_all) + list(
    _assignment_all
) + list(_intent_evaluator_all) + list(_judge_all) + list(_review_all) + list(
    _metrics_all
) + list(_report_all) + list(_analysis_all)


def __getattr__(name):
    if name in _judge_all:
        from importlib import import_module

        value = getattr(import_module(".judge", __name__), name)
        globals()[name] = value
        return value
    if name in _review_all:
        from importlib import import_module

        value = getattr(import_module(".review_evaluator", __name__), name)
        globals()[name] = value
        return value
    if name in _metrics_all:
        from importlib import import_module

        value = getattr(import_module(".metrics", __name__), name)
        globals()[name] = value
        return value
    if name in _report_all:
        from importlib import import_module

        value = getattr(import_module(".report", __name__), name)
        globals()[name] = value
        return value
    if name in _analysis_all:
        from importlib import import_module

        value = getattr(import_module(".analysis_artifacts", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def __dir__():
    return sorted(set(globals()).union(__all__))
