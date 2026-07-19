"""Explicit Eval boundary for the project's unified model-adapter API.

Eval business modules import the protocol through this module so product
runtime dependencies remain confined to one trusted adapter boundary.  The
protocol objects are aliases, not parallel DTOs: Judge calls still use the
project's single ``ModelAdapter`` / ``ModelTurnRequest`` protocol end to end.

The factory relay is intentionally defined here as well.  Composition roots
such as the Eval CLI can construct a Judge without importing product runtime
modules or handling credential values.  Only an environment-variable *name*
crosses this API; the product factory resolves the value at construction time.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Protocol
from urllib.parse import urlsplit

from review_agent.model_adapter import (
    MAX_ALLOWED_RESPONSE_BYTES,
    ModelAdapter,
    ModelAdapterCapabilities,
)
from review_agent.model_protocol import (
    ModelResponseKind,
    ModelTurnRequest,
    ModelTurnResponse,
)
from ..config import JudgeExecutionBudgets, validate_safe_text


EVAL_JUDGE_STAGE_LABEL = "eval-judge"

_ENVIRONMENT_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_STAGE_LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_SUPPORTED_PROVIDERS = frozenset({"none", "fake", "openai-compatible"})
_MAX_MODEL_NAME_CHARS = 512
_MAX_BASE_URL_CHARS = 4_096
_MAX_TIMEOUT_SECONDS = 7 * 24 * 60 * 60


class AdapterConfigError(ValueError):
    """Sanitized model-adapter configuration failure at the Eval boundary."""


@dataclass(frozen=True)
class ModelAdapterConfig:
    """Runtime-only, credential-free input to the Eval model factory.

    ``api_key_env`` is the name of the one authorized environment variable,
    never its value.  ``timeout_seconds`` and ``max_response_bytes`` are
    transport enforcement budgets, not model-generation parameters.
    """

    provider_name: str | None
    model: str | None
    base_url: str | None
    api_key_env: str
    stage_label: str = EVAL_JUDGE_STAGE_LABEL
    timeout_seconds: int | float | None = None
    max_response_bytes: int | None = None

    def __post_init__(self) -> None:
        provider_name = self.provider_name
        if provider_name is None:
            provider_name = "none"
        if type(provider_name) is not str or provider_name not in _SUPPORTED_PROVIDERS:
            raise AdapterConfigError("unsupported Eval model provider")
        if self.model is not None and (
            type(self.model) is not str
            or not self.model
            or len(self.model) > _MAX_MODEL_NAME_CHARS
            or "\x00" in self.model
        ):
            raise AdapterConfigError("model must be bounded non-empty text or None")
        if self.base_url is not None:
            _validate_base_url(self.base_url)
        if (
            type(self.api_key_env) is not str
            or _ENVIRONMENT_KEY_RE.fullmatch(self.api_key_env) is None
        ):
            raise AdapterConfigError(
                "api_key_env must name exactly one environment variable"
            )
        _validate_stage_label(self.stage_label)
        if self.timeout_seconds is not None:
            _validate_timeout(self.timeout_seconds)
        if self.max_response_bytes is not None:
            _validate_response_limit(self.max_response_bytes)

    def redacted_dict(self) -> dict[str, Any]:
        """Return a non-secret diagnostic projection, never the credential."""

        return {
            "provider_name": self.provider_name or "none",
            "model": self.model,
            "base_url": self.base_url,
            "credential_env_name": self.api_key_env,
            "stage_label": self.stage_label,
            "timeout_seconds": self.timeout_seconds,
            "max_response_bytes": self.max_response_bytes,
        }


# The longer name makes call sites self-documenting while preserving the
# familiar unified-factory name for callers migrating from the product API.
EvalModelAdapterConfig = ModelAdapterConfig


class ModelAdapterFactory(Protocol):
    def create(self) -> ModelAdapter:
        ...


def build_model_adapter_factory_from_config(
    config: ModelAdapterConfig,
    *,
    stage_label: str | None = None,
    timeout_seconds: int | float | None = None,
    max_response_bytes: int | None = None,
) -> ModelAdapterFactory | None:
    """Relay Eval configuration to the unified product factory.

    The product factory import is delayed until this function is called.  This
    keeps ordinary Eval package imports independent of product composition
    code and prevents CLI handlers from importing it directly.
    """

    if type(config) is not ModelAdapterConfig:
        raise TypeError("config must be an Eval ModelAdapterConfig")
    resolved_stage_label = config.stage_label if stage_label is None else stage_label
    _validate_stage_label(resolved_stage_label)
    if (
        timeout_seconds is not None
        and config.timeout_seconds is not None
        and timeout_seconds != config.timeout_seconds
    ):
        raise AdapterConfigError(
            "timeout_seconds conflicts with the immutable Eval adapter config"
        )
    if (
        max_response_bytes is not None
        and config.max_response_bytes is not None
        and max_response_bytes != config.max_response_bytes
    ):
        raise AdapterConfigError(
            "max_response_bytes conflicts with the immutable Eval adapter config"
        )
    resolved_timeout = (
        config.timeout_seconds if timeout_seconds is None else timeout_seconds
    )
    resolved_response_limit = (
        config.max_response_bytes
        if max_response_bytes is None
        else max_response_bytes
    )
    if resolved_timeout is not None:
        _validate_timeout(resolved_timeout)
    if resolved_response_limit is not None:
        _validate_response_limit(resolved_response_limit)

    # This is the only Eval -> product-factory transfer point.  No credential
    # value is read in Eval code or copied into this immutable boundary config.
    from review_agent.model_adapter_factory import (
        AdapterConfigError as ProductAdapterConfigError,
        ModelAdapterConfig as ProductModelAdapterConfig,
        build_model_adapter_factory_from_config as build_product_factory,
    )

    product_config = ProductModelAdapterConfig(
        provider_name=config.provider_name,
        model=config.model,
        base_url=config.base_url,
        api_key_env=config.api_key_env,
        stage_label=resolved_stage_label,
        timeout_seconds=resolved_timeout,
        max_response_bytes=resolved_response_limit,
    )
    try:
        return build_product_factory(
            product_config,
            stage_label=resolved_stage_label,
        )
    except ProductAdapterConfigError as exc:
        # Product errors contain option/environment names, never environment
        # values.  Re-raise through the Eval boundary so callers do not need a
        # product-module import merely to handle configuration failures.
        raise AdapterConfigError(str(exc)) from exc


def build_judge_model_adapter_factory(
    config: ModelAdapterConfig,
    *,
    budgets: JudgeExecutionBudgets | None = None,
    timeout_seconds: int | float | None = None,
    max_response_bytes: int | None = None,
) -> ModelAdapterFactory | None:
    """Build an ``eval-judge`` factory bound to Judge execution budgets.

    ``budgets`` may be the canonical ``JudgeExecutionBudgets``.  Explicit
    scalar arguments are also accepted for small composition roots and tests,
    but conflicting values fail closed.  This helper does not attest or alter
    Adapter capabilities; ``SemanticJudge`` remains the authority that checks
    tool-choice, timeout, response-limit, and provider identity capabilities.
    """

    if budgets is not None:
        if type(budgets) is not JudgeExecutionBudgets:
            raise TypeError("budgets must be a JudgeExecutionBudgets")
        budget_timeout = budgets.attempt_timeout_seconds
        budget_response_limit = budgets.max_model_response_bytes
        if timeout_seconds is not None and timeout_seconds != budget_timeout:
            raise AdapterConfigError(
                "timeout_seconds conflicts with JudgeExecutionBudgets"
            )
        if (
            max_response_bytes is not None
            and max_response_bytes != budget_response_limit
        ):
            raise AdapterConfigError(
                "max_response_bytes conflicts with JudgeExecutionBudgets"
            )
        timeout_seconds = budget_timeout
        max_response_bytes = budget_response_limit

    return build_model_adapter_factory_from_config(
        config,
        stage_label=EVAL_JUDGE_STAGE_LABEL,
        timeout_seconds=timeout_seconds,
        max_response_bytes=max_response_bytes,
    )


def _validate_stage_label(value: Any) -> str:
    if type(value) is not str or _STAGE_LABEL_RE.fullmatch(value) is None:
        raise AdapterConfigError(
            "stage_label must be a bounded option-name component"
        )
    return value


def _validate_timeout(value: Any) -> int | float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
        or value > _MAX_TIMEOUT_SECONDS
    ):
        raise AdapterConfigError(
            "timeout_seconds must be positive, finite, and within the Eval limit"
        )
    return value


def _validate_response_limit(value: Any) -> int:
    if (
        type(value) is not int
        or value < 1
        or value > MAX_ALLOWED_RESPONSE_BYTES
    ):
        raise AdapterConfigError(
            "max_response_bytes must be a positive bounded integer"
        )
    return value


def _validate_base_url(value: Any) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > _MAX_BASE_URL_CHARS
        or "\x00" in value
        or any(character in value for character in "\r\n")
    ):
        raise AdapterConfigError("base_url must be bounded non-empty text or None")
    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise AdapterConfigError("base_url is invalid") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise AdapterConfigError("base_url must be an HTTP(S) endpoint")
    if parsed.username is not None or parsed.password is not None or "@" in parsed.netloc:
        raise AdapterConfigError("base_url may not contain credentials")
    if parsed.query or parsed.fragment:
        raise AdapterConfigError("base_url may not contain a query or fragment")
    try:
        # Reuse the Eval artifact safety policy for query/path strings too;
        # credentials must not be smuggled through a URL merely because they
        # are not in the authority component.
        validate_safe_text(value, "base_url")
    except Exception as exc:
        raise AdapterConfigError("base_url contains forbidden sensitive material") from exc
    return value


__all__ = [
    "AdapterConfigError",
    "EVAL_JUDGE_STAGE_LABEL",
    "EvalModelAdapterConfig",
    "ModelAdapter",
    "ModelAdapterCapabilities",
    "ModelAdapterConfig",
    "ModelAdapterFactory",
    "ModelResponseKind",
    "ModelTurnRequest",
    "ModelTurnResponse",
    "build_judge_model_adapter_factory",
    "build_model_adapter_factory_from_config",
]
