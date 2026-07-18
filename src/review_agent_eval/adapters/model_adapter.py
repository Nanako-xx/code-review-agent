"""Narrow integration boundary from the Eval harness to the unified model API.

Eval business modules import the protocol through this module so product
runtime dependencies remain confined to an explicit adapter boundary.  The
objects are aliases, not parallel DTOs: Judge calls still use the project's
single ``ModelAdapter`` / ``ModelTurnRequest`` protocol end to end.
"""

from review_agent.model_adapter import ModelAdapter, ModelAdapterCapabilities
from review_agent.model_protocol import (
    ModelResponseKind,
    ModelTurnRequest,
    ModelTurnResponse,
)

__all__ = [
    "ModelAdapter",
    "ModelAdapterCapabilities",
    "ModelResponseKind",
    "ModelTurnRequest",
    "ModelTurnResponse",
]
