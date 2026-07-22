"""Authority-minimal interface shared by every Agent-under-test adapter."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    FrozenSet,
    Protocol,
    runtime_checkable,
)

from ..artifacts import TargetAccess
from ..cases import WireContractV2
from ..config import (
    AdapterCapabilitiesV2,
    AgentConfigSnapshot,
    ClarificationMatcherSnapshot,
    EvalRunConfig,
    ResourceBudgets,
    validate_run_id,
    validate_trial_id,
)
from ..models import (
    EvalInput,
    EvalSubmission,
    FailureCode,
    ReviewTargetKind,
    SchemaError,
    _digest,
    _identifier,
    _integer,
)

if TYPE_CHECKING:
    from ..clarification import ClarificationChannel


@dataclass(frozen=True, init=False)
class AgentRunConfig:
    """Truth-free runtime projection for exactly one canonical Trial.

    This is deliberately not another persisted protocol.  Agent identity and
    resource budgets remain the canonical Task 3 objects; this projection only
    binds them to one input and one derived Trial identity.
    """

    _CONSTRUCTION_TOKEN: ClassVar[object] = object()

    run_id: str
    task_id: str
    eval_input_digest: str
    wire_contract: WireContractV2
    adapter_capabilities: AdapterCapabilitiesV2
    adapter_capabilities_digest: str
    clarification_matcher: ClarificationMatcherSnapshot
    clarification_matcher_config_digest: str
    trial_index: int
    trial_id: str
    agent: AgentConfigSnapshot
    budgets: ResourceBudgets

    def __init__(
        self,
        *,
        run_id: str,
        task_id: str,
        eval_input_digest: str,
        wire_contract: WireContractV2,
        adapter_capabilities: AdapterCapabilitiesV2,
        adapter_capabilities_digest: str,
        clarification_matcher: ClarificationMatcherSnapshot,
        clarification_matcher_config_digest: str,
        trial_index: int,
        trial_id: str,
        agent: AgentConfigSnapshot,
        budgets: ResourceBudgets,
        _construction_token: object,
    ) -> None:
        if _construction_token is not self._CONSTRUCTION_TOKEN:
            raise TypeError(
                "AgentRunConfig must be created from a verified Run binding"
            )
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "eval_input_digest", eval_input_digest)
        object.__setattr__(self, "wire_contract", wire_contract)
        object.__setattr__(self, "adapter_capabilities", adapter_capabilities)
        object.__setattr__(
            self,
            "adapter_capabilities_digest",
            adapter_capabilities_digest,
        )
        object.__setattr__(self, "clarification_matcher", clarification_matcher)
        object.__setattr__(
            self,
            "clarification_matcher_config_digest",
            clarification_matcher_config_digest,
        )
        object.__setattr__(self, "trial_index", trial_index)
        object.__setattr__(self, "trial_id", trial_id)
        object.__setattr__(self, "agent", agent)
        object.__setattr__(self, "budgets", budgets)
        self.__post_init__()

    def __post_init__(self) -> None:
        validate_run_id(self.run_id)
        _identifier(self.task_id, "agent run config.task_id")
        _digest(self.eval_input_digest, "agent run config.eval_input_digest")
        if not isinstance(self.wire_contract, WireContractV2):
            raise SchemaError(
                "agent run config.wire_contract must be WireContractV2"
            )
        if not isinstance(self.adapter_capabilities, AdapterCapabilitiesV2):
            raise SchemaError(
                "agent run config.adapter_capabilities must be AdapterCapabilitiesV2"
            )
        _digest(
            self.adapter_capabilities_digest,
            "agent run config.adapter_capabilities_digest",
        )
        if self.adapter_capabilities_digest != self.adapter_capabilities.digest():
            raise SchemaError(
                "agent run config adapter capability digest drifted"
            )
        if (
            self.adapter_capabilities.input_schema_version
            != self.wire_contract.input_schema_version
            or self.adapter_capabilities.submission_schema_version
            != self.wire_contract.submission_schema_version
        ):
            raise SchemaError(
                "agent run config adapter capabilities do not match wire contract"
            )
        if not isinstance(
            self.clarification_matcher,
            ClarificationMatcherSnapshot,
        ):
            raise SchemaError(
                "agent run config.clarification_matcher must be ClarificationMatcherSnapshot"
            )
        _digest(
            self.clarification_matcher_config_digest,
            "agent run config.clarification_matcher_config_digest",
        )
        _integer(
            self.trial_index,
            "agent run config.trial_index",
            minimum=1,
        )
        validate_trial_id(
            self.trial_id,
            self.run_id,
            self.task_id,
            self.trial_index,
        )
        if not isinstance(self.agent, AgentConfigSnapshot):
            raise SchemaError("agent run config.agent must be AgentConfigSnapshot")
        if not isinstance(self.budgets, ResourceBudgets):
            raise SchemaError("agent run config.budgets must be ResourceBudgets")
        if (
            self.clarification_matcher_config_digest
            != self.clarification_matcher.digest()
        ):
            raise SchemaError(
                "clarification matcher digest does not match its snapshot"
            )

    @classmethod
    def bind(
        cls,
        run_config: EvalRunConfig,
        eval_input: EvalInput,
        trial_index: int,
    ) -> "AgentRunConfig":
        if not isinstance(run_config, EvalRunConfig):
            raise SchemaError("run_config must be EvalRunConfig")
        if not isinstance(eval_input, EvalInput):
            raise SchemaError("eval_input must be EvalInput")
        case = run_config.suite.case(eval_input.task_id)
        if eval_input.digest() != case.eval_input_digest:
            raise SchemaError(
                "eval_input does not match the immutable Suite case binding"
            )
        return cls._from_verified_binding(
            run_id=run_config.run_id,
            task_id=eval_input.task_id,
            eval_input_digest=eval_input.digest(),
            wire_contract=run_config.wire_contract,
            adapter_capabilities=run_config.adapter_capabilities,
            adapter_capabilities_digest=(
                run_config.adapter_capabilities_digest
            ),
            clarification_matcher=run_config.clarification_matcher,
            clarification_matcher_config_digest=(
                run_config.clarification_matcher_config_digest
            ),
            trial_index=trial_index,
            trial_id=run_config.trial_id(eval_input.task_id, trial_index),
            agent=run_config.agent,
            budgets=run_config.resource_budgets,
        )

    @classmethod
    def _from_verified_binding(
        cls,
        *,
        run_id: str,
        task_id: str,
        eval_input_digest: str,
        wire_contract: WireContractV2,
        adapter_capabilities: AdapterCapabilitiesV2,
        adapter_capabilities_digest: str,
        clarification_matcher: ClarificationMatcherSnapshot,
        clarification_matcher_config_digest: str,
        trial_index: int,
        trial_id: str,
        agent: AgentConfigSnapshot,
        budgets: ResourceBudgets,
    ) -> "AgentRunConfig":
        """Internal projection constructor used after an equivalent verified binding."""

        return cls(
            run_id=run_id,
            task_id=task_id,
            eval_input_digest=eval_input_digest,
            wire_contract=wire_contract,
            adapter_capabilities=adapter_capabilities,
            adapter_capabilities_digest=adapter_capabilities_digest,
            clarification_matcher=clarification_matcher,
            clarification_matcher_config_digest=(
                clarification_matcher_config_digest
            ),
            trial_index=trial_index,
            trial_id=trial_id,
            agent=agent,
            budgets=budgets,
            _construction_token=cls._CONSTRUCTION_TOKEN,
        )

    @property
    def agent_id(self) -> str:
        return self.agent.agent_id

    @property
    def timeout_seconds(self) -> Any:
        return self.budgets.agent_timeout_seconds

    @property
    def max_output_bytes(self) -> int:
        return self.budgets.max_agent_output_bytes

    @property
    def max_trace_bytes(self) -> int:
        return self.budgets.max_trace_bytes

    @property
    def target_kind(self) -> ReviewTargetKind:
        return self.wire_contract.review_target_kind


class AgentAdapterError(RuntimeError):
    """A bounded adapter failure expressed with the canonical failure taxonomy."""

    def __init__(
        self,
        code: FailureCode,
        message: str,
        *,
        retryable: bool,
    ) -> None:
        if not isinstance(code, FailureCode):
            raise TypeError("adapter error code must be FailureCode")
        if type(message) is not str or not message or len(message) > 4_096:
            raise ValueError("adapter error message must be bounded non-empty text")
        if type(retryable) is not bool:
            raise TypeError("adapter error retryable must be bool")
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class AgentInputCapability(str, Enum):
    """Optional canonical input features an Adapter may or may not support."""

    EXISTING_CI_EVIDENCE = "existing_ci_evidence"


class AdapterIncompatibilityReason(str, Enum):
    EXISTING_CI_EVIDENCE = "adapter_incompatible.existing_ci_evidence"
    CANONICAL_MATERIAL_CLAIM_UNAVAILABLE = (
        "adapter_incompatible.canonical_material_claim_unavailable"
    )
    TARGET_KIND = "adapter_incompatible.target_kind"
    CAPABILITY_MISMATCH = "adapter_incompatible.capability_mismatch"


class AgentAdapterIncompatibleError(RuntimeError):
    """The configured Adapter/interaction protocol cannot represent this Trial."""

    def __init__(self, reason: AdapterIncompatibilityReason) -> None:
        if not isinstance(reason, AdapterIncompatibilityReason):
            raise TypeError("incompatibility reason must be AdapterIncompatibilityReason")
        super().__init__(reason.value)
        self.reason = reason


@dataclass(frozen=True)
class AdapterCompatibility:
    """Pre-Trial suite-gating result; incompatibility is not an Agent failure."""

    unsupported: FrozenSet[AgentInputCapability] = frozenset()

    def __post_init__(self) -> None:
        if not isinstance(self.unsupported, frozenset) or any(
            not isinstance(item, AgentInputCapability) for item in self.unsupported
        ):
            raise TypeError(
                "adapter compatibility.unsupported must be AgentInputCapability values"
            )

    @property
    def compatible(self) -> bool:
        return not self.unsupported


@runtime_checkable
class AgentUnderTestAdapter(Protocol):
    """The only execution interface understood by the Eval Runner.

    Adapters are trusted Harness integration code.  ``run`` must observe the
    supplied cancellation event and terminate every process/HTTP/IPC resource
    it owns before returning.  Untrusted Agents must remain behind that
    Adapter-managed boundary; accepting and ignoring cancellation is a
    protocol violation.
    """

    def compatibility(
        self,
        eval_input: EvalInput,
        config: AgentRunConfig,
    ) -> AdapterCompatibility:
        ...

    def run(
        self,
        eval_input: EvalInput,
        workspace: Path,
        config: AgentRunConfig,
        clarification_channel: ClarificationChannel,
        *,
        target_access: TargetAccess,
        target_materialization_id: str,
        cancel_event: Any,
    ) -> EvalSubmission:
        ...


__all__ = [
    "AdapterCompatibility",
    "AdapterIncompatibilityReason",
    "AgentAdapterError",
    "AgentAdapterIncompatibleError",
    "AgentInputCapability",
    "AgentRunConfig",
    "AgentUnderTestAdapter",
]
