"""Explicit machine-local repository locator-to-authority bindings.

This module never copies or re-keys repository Memory.  A binding says that a
freshly verified live repository identity (the locator) uses an explicitly
selected older repository identity (the authority).  The registry is local to
one Memory root and deliberately has no discovery, origin matching, import, or
export behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hmac
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import sys
import threading
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Union

from review_agent.memory_identity import (
    MemoryIdentityError,
    PathInput,
    RepositoryIdentityDescriptor,
    RepositoryRelinkDescriptor,
    ResolvedMemoryRoot,
    VerifiedRepositoryIdentity,
    build_relink_descriptor,
    plan_repository_memory_namespace,
    repository_namespace_path,
    resolve_memory_root,
)
from review_agent.memory_models import (
    canonical_json,
    canonical_sha256,
    stable_event_id,
    stable_repository_binding_id,
    validate_stable_id,
)
from review_agent.memory_store import (
    MemoryStore,
    MemoryStoreBusyError,
    MemoryStoreError,
)
from review_agent.revision import RevisionResolver


REPOSITORY_RELINK_REGISTRY_SCHEMA_VERSION = 1
REPOSITORY_RELINK_REGISTRY_FILENAME = "repository-relinks.sqlite3"
REPOSITORY_RELINK_EVENT_TYPE = "repository_authority_bound"

_APPLICATION_ID = 0x52454C4B  # "RELK"
_GENESIS_EVENT_HASH = "0" * 64
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_TIMESTAMP_PATTERN = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z$"
)
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400
_SQLITE_SIDECAR_SUFFIXES = ("-journal", "-wal", "-shm")
_EXPECTED_TABLES = frozenset(
    {"registry_state", "bindings", "events", "request_receipts"}
)

IdentityDescriptorInput = Union[
    RepositoryIdentityDescriptor,
    VerifiedRepositoryIdentity,
]
TransactionHook = Callable[[str], None]


class RepositoryRelinkError(ValueError):
    """A stable, content-free repository relink error."""


class RepositoryRelinkValidationError(RepositoryRelinkError):
    """A relink request or canonical payload is invalid."""


class RepositoryRelinkConflictError(RepositoryRelinkError):
    """A relink precondition or compare-and-swap check failed."""


class RepositoryRelinkIntegrityError(RepositoryRelinkError):
    """The root-scoped registry failed integrity verification."""


def repository_binding_id(
    locator_repository_key: str,
    authority_repository_key: str,
) -> str:
    """Return the stable RB identity for one locator/authority pair."""

    locator = _digest(locator_repository_key, "locator_repository_key")
    authority = _digest(authority_repository_key, "authority_repository_key")
    if hmac.compare_digest(locator, authority):
        raise RepositoryRelinkValidationError(
            "repository relink requires different repository keys"
        )
    return stable_repository_binding_id(locator, authority)


def repository_authority_resolution_hash(
    locator_repository_key: str,
    authority_repository_key: str,
    *,
    binding_id: Optional[str] = None,
) -> str:
    """Use the canonical candidate-authority resolution hash contract."""

    # Imported lazily so memory_identity remains independent from sources and
    # the service layer is the sole bridge between the two foundations.
    from review_agent.memory_sources import candidate_authority_resolution_hash

    try:
        return candidate_authority_resolution_hash(
            locator_repository_key,
            authority_repository_key,
            binding_id=binding_id,
        )
    except (TypeError, ValueError):
        raise RepositoryRelinkValidationError(
            "repository authority resolution is invalid"
        ) from None


@dataclass(frozen=True)
class RepositoryAuthorityResolution:
    """A verified live locator and its direct or explicitly bound authority."""

    locator_identity: RepositoryIdentityDescriptor = field(repr=False)
    authority_identity: RepositoryIdentityDescriptor = field(repr=False)
    binding_id: Optional[str]
    authority_resolution_hash: str
    registry_generation: int
    schema_version: int = REPOSITORY_RELINK_REGISTRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema_version(self.schema_version, "authority_resolution")
        locator = _identity_descriptor(self.locator_identity, "locator identity")
        authority = _identity_descriptor(
            self.authority_identity, "authority identity"
        )
        generation = _generation(self.registry_generation)
        direct = hmac.compare_digest(
            locator.repository_key, authority.repository_key
        )
        if direct:
            if self.binding_id is not None:
                raise RepositoryRelinkValidationError(
                    "direct authority resolution cannot contain a binding"
                )
            binding = None
        else:
            binding = _stable_id(self.binding_id, "RB", "binding_id")
            expected_binding = repository_binding_id(
                locator.repository_key, authority.repository_key
            )
            if not hmac.compare_digest(binding, expected_binding):
                raise RepositoryRelinkValidationError(
                    "repository binding identity is not canonical"
                )
        resolution_hash = _digest(
            self.authority_resolution_hash,
            "authority_resolution_hash",
        )
        expected_hash = repository_authority_resolution_hash(
            locator.repository_key,
            authority.repository_key,
            binding_id=binding,
        )
        if not hmac.compare_digest(resolution_hash, expected_hash):
            raise RepositoryRelinkValidationError(
                "repository authority resolution hash is not canonical"
            )
        object.__setattr__(self, "locator_identity", locator)
        object.__setattr__(self, "authority_identity", authority)
        object.__setattr__(self, "binding_id", binding)
        object.__setattr__(self, "authority_resolution_hash", resolution_hash)
        object.__setattr__(self, "registry_generation", generation)

    @property
    def locator_repository_key(self) -> str:
        return self.locator_identity.repository_key

    @property
    def authority_repository_key(self) -> str:
        return self.authority_identity.repository_key

    @property
    def is_bound(self) -> bool:
        return self.binding_id is not None

    def to_payload(self) -> Dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "locator_identity": self.locator_identity.to_payload(),
            "authority_identity": self.authority_identity.to_payload(),
            "binding_id": self.binding_id,
            "authority_resolution_hash": self.authority_resolution_hash,
            "registry_generation": self.registry_generation,
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, object]
    ) -> "RepositoryAuthorityResolution":
        root = _object(payload, "authority resolution")
        _exact_fields(
            root,
            {
                "schema_version",
                "locator_identity",
                "authority_identity",
                "binding_id",
                "authority_resolution_hash",
                "registry_generation",
            },
            "authority resolution",
        )
        try:
            resolution = cls(
                locator_identity=RepositoryIdentityDescriptor.from_payload(
                    _object(root["locator_identity"], "locator identity")
                ),
                authority_identity=RepositoryIdentityDescriptor.from_payload(
                    _object(root["authority_identity"], "authority identity")
                ),
                binding_id=root["binding_id"],
                authority_resolution_hash=root["authority_resolution_hash"],
                registry_generation=root["registry_generation"],
                schema_version=root["schema_version"],
            )
        except MemoryIdentityError:
            raise RepositoryRelinkValidationError(
                "authority resolution identity is invalid"
            ) from None
        if resolution.to_payload() != dict(root):
            raise RepositoryRelinkValidationError(
                "authority resolution payload is not canonical"
            )
        return resolution


@dataclass(frozen=True)
class PreparedRepositoryRelink:
    """An immutable, write-free relink plan bound to registry state."""

    descriptor: RepositoryRelinkDescriptor = field(repr=False)
    from_repository_key: str
    request_id: str
    actor: str = field(repr=False)
    reason: str = field(repr=False)
    old_authority_state_token: str
    new_namespace_empty: bool
    registry_generation: int
    registry_root_hash: str
    schema_version: int = REPOSITORY_RELINK_REGISTRY_SCHEMA_VERSION
    authority_descriptor_hash: str = field(init=False)
    locator_descriptor_hash: str = field(init=False)
    descriptor_hash: str = field(init=False)
    binding_id: str = field(init=False)
    authority_resolution_hash: str = field(init=False)
    semantic_hash: str = field(init=False)
    prepared_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _schema_version(self.schema_version, "prepared repository relink")
        descriptor = _relink_descriptor(self.descriptor)
        from_repository_key = _digest(
            self.from_repository_key,
            "from_repository_key",
        )
        if not hmac.compare_digest(
            from_repository_key,
            descriptor.authority_repository_key,
        ):
            raise RepositoryRelinkValidationError(
                "repository relink from-key does not match the authority"
            )
        request_id = _stable_id(self.request_id, "REQ", "request_id")
        actor = _text(self.actor, "actor", 512)
        reason = _text(self.reason, "reason", 4096)
        old_token = _digest(
            self.old_authority_state_token,
            "old_authority_state_token",
        )
        if self.new_namespace_empty is not True:
            raise RepositoryRelinkConflictError(
                "new repository namespace must be explicitly empty"
            )
        generation = _generation(self.registry_generation)
        root_hash = _digest(self.registry_root_hash, "registry_root_hash")
        authority_hash = canonical_sha256(
            descriptor.authority_identity.to_payload()
        )
        locator_hash = canonical_sha256(descriptor.locator_identity.to_payload())
        descriptor_hash = canonical_sha256(descriptor.to_payload())
        if not hmac.compare_digest(descriptor_hash, descriptor.descriptor_hash):
            raise RepositoryRelinkValidationError(
                "repository relink descriptor hash is not canonical"
            )
        binding = repository_binding_id(
            descriptor.locator_repository_key,
            descriptor.authority_repository_key,
        )
        resolution_hash = repository_authority_resolution_hash(
            descriptor.locator_repository_key,
            descriptor.authority_repository_key,
            binding_id=binding,
        )
        object.__setattr__(self, "descriptor", descriptor)
        object.__setattr__(self, "from_repository_key", from_repository_key)
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "actor", actor)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "old_authority_state_token", old_token)
        object.__setattr__(self, "registry_generation", generation)
        object.__setattr__(self, "registry_root_hash", root_hash)
        object.__setattr__(self, "authority_descriptor_hash", authority_hash)
        object.__setattr__(self, "locator_descriptor_hash", locator_hash)
        object.__setattr__(self, "descriptor_hash", descriptor_hash)
        object.__setattr__(self, "binding_id", binding)
        object.__setattr__(self, "authority_resolution_hash", resolution_hash)
        semantic_hash = canonical_sha256(self.semantic_payload())
        object.__setattr__(self, "semantic_hash", semantic_hash)
        object.__setattr__(
            self,
            "prepared_hash",
            canonical_sha256(
                {
                    "request_id": request_id,
                    "semantic_hash": semantic_hash,
                    "cas": self.cas_payload(),
                }
            ),
        )

    @property
    def authority_identity(self) -> RepositoryIdentityDescriptor:
        return self.descriptor.authority_identity

    @property
    def locator_identity(self) -> RepositoryIdentityDescriptor:
        return self.descriptor.locator_identity

    @property
    def authority_repository_key(self) -> str:
        return self.descriptor.authority_repository_key

    @property
    def locator_repository_key(self) -> str:
        return self.descriptor.locator_repository_key

    def semantic_payload(self) -> Dict[str, object]:
        """Return request semantics, excluding mutable CAS observations."""

        return {
            "schema_version": self.schema_version,
            "descriptor": self.descriptor.to_payload(),
            "from_repository_key": self.from_repository_key,
            "actor": self.actor,
            "reason": self.reason,
            "authority_descriptor_hash": self.authority_descriptor_hash,
            "locator_descriptor_hash": self.locator_descriptor_hash,
            "descriptor_hash": self.descriptor_hash,
            "binding_id": self.binding_id,
            "authority_resolution_hash": self.authority_resolution_hash,
        }

    def cas_payload(self) -> Dict[str, object]:
        """Return the prepare-time observations that apply must compare."""

        return {
            "old_authority_state_token": self.old_authority_state_token,
            "new_namespace_empty": self.new_namespace_empty,
            "registry_generation": self.registry_generation,
            "registry_root_hash": self.registry_root_hash,
        }

    def to_payload(self) -> Dict[str, object]:
        return {
            **self.semantic_payload(),
            **self.cas_payload(),
            "request_id": self.request_id,
            "semantic_hash": self.semantic_hash,
            "prepared_hash": self.prepared_hash,
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, object]
    ) -> "PreparedRepositoryRelink":
        root = _object(payload, "prepared repository relink")
        expected = {
            "schema_version",
            "descriptor",
            "from_repository_key",
            "request_id",
            "actor",
            "reason",
            "old_authority_state_token",
            "new_namespace_empty",
            "registry_generation",
            "registry_root_hash",
            "authority_descriptor_hash",
            "locator_descriptor_hash",
            "descriptor_hash",
            "binding_id",
            "authority_resolution_hash",
            "semantic_hash",
            "prepared_hash",
        }
        _exact_fields(root, expected, "prepared repository relink")
        try:
            prepared = cls(
                descriptor=RepositoryRelinkDescriptor.from_payload(
                    _object(root["descriptor"], "relink descriptor")
                ),
                from_repository_key=root["from_repository_key"],
                request_id=root["request_id"],
                actor=root["actor"],
                reason=root["reason"],
                old_authority_state_token=root[
                    "old_authority_state_token"
                ],
                new_namespace_empty=root["new_namespace_empty"],
                registry_generation=root["registry_generation"],
                registry_root_hash=root["registry_root_hash"],
                schema_version=root["schema_version"],
            )
        except MemoryIdentityError:
            raise RepositoryRelinkValidationError(
                "prepared repository relink identity is invalid"
            ) from None
        if prepared.to_payload() != dict(root):
            raise RepositoryRelinkValidationError(
                "prepared repository relink payload is not canonical"
            )
        return prepared


@dataclass(frozen=True)
class RepositoryRelinkEvent:
    """One append-only hash-chained authority binding event."""

    sequence: int
    generation: int
    request_id: str
    descriptor: RepositoryRelinkDescriptor = field(repr=False)
    binding_id: str
    old_authority_state_token: str
    actor: str = field(repr=False)
    reason: str = field(repr=False)
    previous_event_hash: str
    created_at: str
    event_type: str = REPOSITORY_RELINK_EVENT_TYPE
    schema_version: int = REPOSITORY_RELINK_REGISTRY_SCHEMA_VERSION
    event_id: str = field(init=False)
    event_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _schema_version(self.schema_version, "repository relink event")
        if self.event_type != REPOSITORY_RELINK_EVENT_TYPE:
            raise RepositoryRelinkValidationError(
                "repository relink event type is invalid"
            )
        sequence = _positive_int(self.sequence, "event sequence")
        generation = _positive_int(self.generation, "event generation")
        if sequence != generation:
            raise RepositoryRelinkValidationError(
                "repository relink event order is invalid"
            )
        request_id = _stable_id(self.request_id, "REQ", "request_id")
        descriptor = _relink_descriptor(self.descriptor)
        binding = _stable_id(self.binding_id, "RB", "binding_id")
        expected_binding = repository_binding_id(
            descriptor.locator_repository_key,
            descriptor.authority_repository_key,
        )
        if not hmac.compare_digest(binding, expected_binding):
            raise RepositoryRelinkValidationError(
                "repository relink event binding is invalid"
            )
        old_token = _digest(
            self.old_authority_state_token,
            "old_authority_state_token",
        )
        actor = _text(self.actor, "actor", 512)
        reason = _text(self.reason, "reason", 4096)
        previous_hash = _digest(self.previous_event_hash, "previous_event_hash")
        created_at = _timestamp(self.created_at)
        event_id = stable_event_id(
            REPOSITORY_RELINK_EVENT_TYPE,
            request_id,
            binding,
            generation,
        )
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(self, "generation", generation)
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "descriptor", descriptor)
        object.__setattr__(self, "binding_id", binding)
        object.__setattr__(self, "old_authority_state_token", old_token)
        object.__setattr__(self, "actor", actor)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "previous_event_hash", previous_hash)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "event_hash", canonical_sha256(self._body()))

    @property
    def locator_repository_key(self) -> str:
        return self.descriptor.locator_repository_key

    @property
    def authority_repository_key(self) -> str:
        return self.descriptor.authority_repository_key

    def _body(self) -> Dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "event_type": self.event_type,
            "sequence": self.sequence,
            "generation": self.generation,
            "event_id": self.event_id,
            "request_id": self.request_id,
            "descriptor": self.descriptor.to_payload(),
            "binding_id": self.binding_id,
            "old_authority_state_token": self.old_authority_state_token,
            "actor": self.actor,
            "reason": self.reason,
            "previous_event_hash": self.previous_event_hash,
            "created_at": self.created_at,
        }

    def to_payload(self) -> Dict[str, object]:
        return {**self._body(), "event_hash": self.event_hash}

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "RepositoryRelinkEvent":
        root = _object(payload, "repository relink event")
        _exact_fields(
            root,
            {
                "schema_version",
                "event_type",
                "sequence",
                "generation",
                "event_id",
                "request_id",
                "descriptor",
                "binding_id",
                "old_authority_state_token",
                "actor",
                "reason",
                "previous_event_hash",
                "created_at",
                "event_hash",
            },
            "repository relink event",
        )
        try:
            event = cls(
                sequence=root["sequence"],
                generation=root["generation"],
                request_id=root["request_id"],
                descriptor=RepositoryRelinkDescriptor.from_payload(
                    _object(root["descriptor"], "relink descriptor")
                ),
                binding_id=root["binding_id"],
                old_authority_state_token=root[
                    "old_authority_state_token"
                ],
                actor=root["actor"],
                reason=root["reason"],
                previous_event_hash=root["previous_event_hash"],
                created_at=root["created_at"],
                event_type=root["event_type"],
                schema_version=root["schema_version"],
            )
        except MemoryIdentityError:
            raise RepositoryRelinkValidationError(
                "repository relink event identity is invalid"
            ) from None
        if event.to_payload() != dict(root):
            raise RepositoryRelinkValidationError(
                "repository relink event payload is not canonical"
            )
        return event


@dataclass(frozen=True)
class RepositoryRelinkResult:
    """The immutable outcome stored in a request receipt."""

    request_id: str
    resolution: RepositoryAuthorityResolution
    event: RepositoryRelinkEvent
    outcome: str
    schema_version: int = REPOSITORY_RELINK_REGISTRY_SCHEMA_VERSION
    result_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _schema_version(self.schema_version, "repository relink result")
        request_id = _stable_id(self.request_id, "REQ", "request_id")
        if not isinstance(self.resolution, RepositoryAuthorityResolution):
            raise RepositoryRelinkValidationError(
                "repository relink result resolution is invalid"
            )
        resolution = RepositoryAuthorityResolution.from_payload(
            self.resolution.to_payload()
        )
        if not isinstance(self.event, RepositoryRelinkEvent):
            raise RepositoryRelinkValidationError(
                "repository relink result event is invalid"
            )
        event = RepositoryRelinkEvent.from_payload(self.event.to_payload())
        if self.outcome not in {"applied", "already_bound"}:
            raise RepositoryRelinkValidationError(
                "repository relink result outcome is invalid"
            )
        if (
            resolution.binding_id != event.binding_id
            or resolution.locator_repository_key
            != event.locator_repository_key
            or resolution.authority_repository_key
            != event.authority_repository_key
            or event.generation > resolution.registry_generation
        ):
            raise RepositoryRelinkValidationError(
                "repository relink result is inconsistent"
            )
        if self.outcome == "applied" and (
            event.request_id != request_id
            or event.generation != resolution.registry_generation
        ):
            raise RepositoryRelinkValidationError(
                "applied repository relink result is inconsistent"
            )
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "resolution", resolution)
        object.__setattr__(self, "event", event)
        object.__setattr__(self, "result_hash", canonical_sha256(self._body()))

    @property
    def applied(self) -> bool:
        return self.outcome == "applied"

    def _body(self) -> Dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "resolution": self.resolution.to_payload(),
            "event": self.event.to_payload(),
            "outcome": self.outcome,
        }

    def to_payload(self) -> Dict[str, object]:
        return {**self._body(), "result_hash": self.result_hash}

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "RepositoryRelinkResult":
        root = _object(payload, "repository relink result")
        _exact_fields(
            root,
            {
                "schema_version",
                "request_id",
                "resolution",
                "event",
                "outcome",
                "result_hash",
            },
            "repository relink result",
        )
        result = cls(
            request_id=root["request_id"],
            resolution=RepositoryAuthorityResolution.from_payload(
                _object(root["resolution"], "authority resolution")
            ),
            event=RepositoryRelinkEvent.from_payload(
                _object(root["event"], "repository relink event")
            ),
            outcome=root["outcome"],
            schema_version=root["schema_version"],
        )
        if result.to_payload() != dict(root):
            raise RepositoryRelinkValidationError(
                "repository relink result payload is not canonical"
            )
        return result


@dataclass(frozen=True)
class RepositoryRelinkRequestReceipt:
    """An immutable audited request-ID to canonical result binding."""

    request_id: str
    semantic_hash: str
    prepared: PreparedRepositoryRelink = field(repr=False)
    result: RepositoryRelinkResult
    created_at: str
    schema_version: int = REPOSITORY_RELINK_REGISTRY_SCHEMA_VERSION
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _schema_version(self.schema_version, "repository relink receipt")
        request_id = _stable_id(self.request_id, "REQ", "request_id")
        semantic_hash = _digest(self.semantic_hash, "semantic_hash")
        if not isinstance(self.prepared, PreparedRepositoryRelink):
            raise RepositoryRelinkValidationError(
                "repository relink receipt request is invalid"
            )
        prepared = PreparedRepositoryRelink.from_payload(
            self.prepared.to_payload()
        )
        if not isinstance(self.result, RepositoryRelinkResult):
            raise RepositoryRelinkValidationError(
                "repository relink receipt result is invalid"
            )
        result = RepositoryRelinkResult.from_payload(self.result.to_payload())
        if (
            prepared.request_id != request_id
            or result.request_id != request_id
            or not hmac.compare_digest(semantic_hash, prepared.semantic_hash)
        ):
            raise RepositoryRelinkValidationError(
                "repository relink receipt request is inconsistent"
            )
        resolution = result.resolution
        event = result.event
        if (
            resolution.locator_identity.to_payload()
            != prepared.locator_identity.to_payload()
            or resolution.authority_identity.to_payload()
            != prepared.authority_identity.to_payload()
            or resolution.binding_id != prepared.binding_id
            or resolution.authority_resolution_hash
            != prepared.authority_resolution_hash
            or event.binding_id != prepared.binding_id
        ):
            raise RepositoryRelinkValidationError(
                "repository relink receipt result is inconsistent"
            )
        if result.outcome == "applied":
            if (
                event.request_id != request_id
                or event.descriptor.to_payload()
                != prepared.descriptor.to_payload()
                or event.actor != prepared.actor
                or event.reason != prepared.reason
                or event.old_authority_state_token
                != prepared.old_authority_state_token
                or event.generation != prepared.registry_generation + 1
                or resolution.registry_generation != event.generation
            ):
                raise RepositoryRelinkValidationError(
                    "repository relink receipt audit is inconsistent"
                )
        elif (
            event.request_id == request_id
            or event.generation > prepared.registry_generation
            or resolution.registry_generation != prepared.registry_generation
        ):
            raise RepositoryRelinkValidationError(
                "repository relink receipt audit is inconsistent"
            )
        created_at = _timestamp(self.created_at)
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "semantic_hash", semantic_hash)
        object.__setattr__(self, "prepared", prepared)
        object.__setattr__(self, "result", result)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "receipt_hash", canonical_sha256(self._body()))

    def _body(self) -> Dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "semantic_hash": self.semantic_hash,
            "prepared": self.prepared.to_payload(),
            "result": self.result.to_payload(),
            "created_at": self.created_at,
        }

    def to_payload(self) -> Dict[str, object]:
        return {**self._body(), "receipt_hash": self.receipt_hash}

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, object]
    ) -> "RepositoryRelinkRequestReceipt":
        root = _object(payload, "repository relink receipt")
        _exact_fields(
            root,
            {
                "schema_version",
                "request_id",
                "semantic_hash",
                "prepared",
                "result",
                "created_at",
                "receipt_hash",
            },
            "repository relink receipt",
        )
        receipt = cls(
            request_id=root["request_id"],
            semantic_hash=root["semantic_hash"],
            prepared=PreparedRepositoryRelink.from_payload(
                _object(root["prepared"], "prepared repository relink")
            ),
            result=RepositoryRelinkResult.from_payload(
                _object(root["result"], "repository relink result")
            ),
            created_at=root["created_at"],
            schema_version=root["schema_version"],
        )
        if receipt.to_payload() != dict(root):
            raise RepositoryRelinkValidationError(
                "repository relink receipt payload is not canonical"
            )
        return receipt


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS registry_state (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    schema_version INTEGER NOT NULL,
    generation INTEGER NOT NULL CHECK (generation >= 0),
    root_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    sequence INTEGER PRIMARY KEY CHECK (sequence > 0),
    generation INTEGER NOT NULL UNIQUE CHECK (generation > 0),
    event_id TEXT NOT NULL UNIQUE,
    request_id TEXT NOT NULL UNIQUE,
    binding_id TEXT NOT NULL,
    previous_event_hash TEXT NOT NULL,
    event_hash TEXT NOT NULL UNIQUE,
    event_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS bindings (
    locator_repository_key TEXT PRIMARY KEY,
    authority_repository_key TEXT NOT NULL,
    binding_id TEXT NOT NULL UNIQUE,
    locator_descriptor_json TEXT NOT NULL,
    authority_descriptor_json TEXT NOT NULL,
    descriptor_hash TEXT NOT NULL,
    event_id TEXT NOT NULL UNIQUE,
    generation INTEGER NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS bindings_authority_key
    ON bindings(authority_repository_key);
CREATE TABLE IF NOT EXISTS request_receipts (
    request_id TEXT PRIMARY KEY,
    semantic_hash TEXT NOT NULL,
    result_hash TEXT NOT NULL,
    receipt_hash TEXT NOT NULL UNIQUE,
    receipt_json TEXT NOT NULL
);
"""


class RepositoryRelinkRegistry:
    """SQLite authority-binding registry scoped to exactly one Memory root."""

    def __init__(
        self,
        memory_root: Union[PathInput, ResolvedMemoryRoot],
        *,
        revision_resolver: Optional[RevisionResolver] = None,
        transaction_hook: Optional[TransactionHook] = None,
        sqlite_timeout_seconds: float = 5.0,
    ) -> None:
        root_value = (
            memory_root.path
            if isinstance(memory_root, ResolvedMemoryRoot)
            else memory_root
        )
        try:
            root = resolve_memory_root(root_value, env={}, create=False)
        except MemoryIdentityError:
            raise RepositoryRelinkValidationError(
                "Memory root is not safe for repository relinks"
            ) from None
        if (
            not isinstance(sqlite_timeout_seconds, (int, float))
            or isinstance(sqlite_timeout_seconds, bool)
            or sqlite_timeout_seconds <= 0
            or sqlite_timeout_seconds > 30
        ):
            raise RepositoryRelinkValidationError(
                "repository relink lock timeout is invalid"
            )
        if transaction_hook is not None and not callable(transaction_hook):
            raise RepositoryRelinkValidationError(
                "repository relink transaction hook is invalid"
            )
        self._root = root
        self._database_path = root / REPOSITORY_RELINK_REGISTRY_FILENAME
        self._revision_resolver = revision_resolver or RevisionResolver()
        self._transaction_hook = transaction_hook
        self._timeout = float(sqlite_timeout_seconds)
        self._lock = threading.RLock()
        self._validate_bounded_paths()
        self._inspect_storage_paths()

    @property
    def memory_root(self) -> Path:
        return self._root

    @property
    def database_path(self) -> Path:
        return self._database_path

    @property
    def registry_root_hash(self) -> str:
        return canonical_sha256(
            {
                "schema": "repository_relink_registry_root_v1",
                "memory_root": os.path.normcase(str(self._root)),
            }
        )

    def prepare_relink(
        self,
        old_identity: IdentityDescriptorInput,
        new_identity: IdentityDescriptorInput,
        *,
        from_repository_key: str,
        actor: str,
        reason: str,
        request_id: str,
        revision_resolver: Optional[RevisionResolver] = None,
    ) -> PreparedRepositoryRelink:
        """Verify and prepare a relink without writing any filesystem state."""

        authority = _identity_descriptor(old_identity, "authority identity")
        from_key = _digest(from_repository_key, "from_repository_key")
        if not hmac.compare_digest(from_key, authority.repository_key):
            raise RepositoryRelinkConflictError(
                "repository relink --from-key does not match the selected authority"
            )
        locator_claim = _identity_descriptor(new_identity, "locator identity")
        locator = self._verify_live_locator(
            locator_claim,
            revision_resolver=revision_resolver,
        )
        if locator.to_payload() != locator_claim.to_payload():
            raise RepositoryRelinkConflictError(
                "live repository locator changed during relink preparation"
            )
        try:
            descriptor = build_relink_descriptor(authority, locator)
        except MemoryIdentityError:
            raise RepositoryRelinkValidationError(
                "repository relink descriptor is invalid"
            ) from None
        with self._lock:
            generation, bindings, _, _ = self._read_verified_state()
        self._assert_binding_policy(
            bindings,
            descriptor.locator_repository_key,
            descriptor.authority_repository_key,
            repository_binding_id(
                descriptor.locator_repository_key,
                descriptor.authority_repository_key,
            ),
        )
        try:
            authority_namespace = repository_namespace_path(
                self._root,
                descriptor.authority_repository_key,
            )
            locator_namespace = repository_namespace_path(
                self._root,
                descriptor.locator_repository_key,
            )
            authority_snapshot = MemoryStore(
                authority_namespace,
                read_only=True,
            ).repository_authority_snapshot(
                descriptor.authority_repository_key
            )
            locator_namespace_empty = MemoryStore.namespace_has_no_store_state(
                locator_namespace
            )
        except MemoryIdentityError:
            raise RepositoryRelinkValidationError(
                "repository relink namespace is invalid"
            ) from None
        except MemoryStoreError:
            raise RepositoryRelinkValidationError(
                "selected repository authority is unavailable"
            ) from None
        if (
            authority_snapshot.repository_identity.to_payload()
            != descriptor.authority_identity.to_payload()
        ):
            raise RepositoryRelinkConflictError(
                "selected repository authority descriptor changed"
            )
        if not locator_namespace_empty:
            raise RepositoryRelinkConflictError(
                "new repository namespace is not empty"
            )
        return PreparedRepositoryRelink(
            descriptor=descriptor,
            from_repository_key=from_key,
            request_id=request_id,
            actor=actor,
            reason=reason,
            old_authority_state_token=authority_snapshot.state_token,
            new_namespace_empty=True,
            registry_generation=generation,
            registry_root_hash=self.registry_root_hash,
        )

    prepare = prepare_relink

    def apply_relink(
        self,
        prepared: PreparedRepositoryRelink,
        *,
        revision_resolver: Optional[RevisionResolver] = None,
    ) -> RepositoryRelinkResult:
        """Atomically apply one exact prepared relink or replay its receipt."""

        canonical_prepared = self._canonical_prepared(prepared)
        if not hmac.compare_digest(
            canonical_prepared.registry_root_hash,
            self.registry_root_hash,
        ):
            raise RepositoryRelinkConflictError(
                "prepared relink belongs to a different Memory root"
            )
        try:
            authority_namespace = repository_namespace_path(
                self._root,
                canonical_prepared.authority_repository_key,
            )
            locator_namespace = repository_namespace_path(
                self._root,
                canonical_prepared.locator_repository_key,
            )
        except MemoryIdentityError:
            raise RepositoryRelinkValidationError(
                "repository relink namespace is invalid"
            ) from None

        try:
            with MemoryStore.lock_namespaces(
                authority_namespace,
                locator_namespace,
                busy_timeout_ms=max(1, int(round(self._timeout * 1000.0))),
            ):
                return self._apply_relink_locked(
                    canonical_prepared,
                    authority_namespace=authority_namespace,
                    locator_namespace=locator_namespace,
                    revision_resolver=revision_resolver,
                )
        except MemoryStoreBusyError:
            raise RepositoryRelinkConflictError(
                "repository namespace is busy during relink"
            ) from None
        except MemoryStoreError:
            raise RepositoryRelinkError(
                "repository namespace lock failed during relink"
            ) from None

    def _apply_relink_locked(
        self,
        canonical_prepared: PreparedRepositoryRelink,
        *,
        authority_namespace: Path,
        locator_namespace: Path,
        revision_resolver: Optional[RevisionResolver],
    ) -> RepositoryRelinkResult:
        """Apply while both repository namespace authority locks are held."""

        live_locator = self._verify_live_locator(
            canonical_prepared.locator_identity,
            revision_resolver=revision_resolver,
            conflict_on_failure=True,
        )
        if live_locator.to_payload() != canonical_prepared.locator_identity.to_payload():
            raise RepositoryRelinkConflictError(
                "live repository locator changed after relink preparation"
            )

        with self._lock:
            # A receipt lookup precedes Store state verification so an exact
            # retry can return the originally committed result.  Both namespace
            # locks already cover the live Git verification above and remain
            # held through any registry commit below.
            existing_receipt = self._read_receipt_if_present(
                canonical_prepared.request_id
            )
            if existing_receipt is not None:
                return self._replay_receipt(existing_receipt, canonical_prepared)

            try:
                authority_snapshot = MemoryStore(
                    authority_namespace,
                    read_only=True,
                ).repository_authority_snapshot(
                    canonical_prepared.authority_repository_key
                )
                namespace_is_empty = MemoryStore.namespace_has_no_store_state(
                    locator_namespace
                )
            except MemoryStoreError:
                raise RepositoryRelinkConflictError(
                    "repository relink authority state is unavailable"
                ) from None
            if (
                authority_snapshot.repository_identity.to_payload()
                != canonical_prepared.authority_identity.to_payload()
            ):
                raise RepositoryRelinkConflictError(
                    "old repository authority descriptor changed"
                )
            old_token = authority_snapshot.state_token
            if not hmac.compare_digest(
                old_token,
                canonical_prepared.old_authority_state_token,
            ):
                raise RepositoryRelinkConflictError(
                    "old repository authority state changed"
                )
            if namespace_is_empty is not True:
                raise RepositoryRelinkConflictError(
                    "new repository namespace is not empty"
                )

            connection: Optional[sqlite3.Connection] = None
            try:
                needs_initialization = not self._database_path.exists()
                connection = self._connect_writable()
                if needs_initialization:
                    self._bootstrap_schema(connection)
                connection.execute("BEGIN IMMEDIATE")
                generation, bindings, events, receipts = self._verify_connection(
                    connection
                )
                concurrent_receipt = receipts.get(canonical_prepared.request_id)
                if concurrent_receipt is not None:
                    result = self._replay_receipt(
                        concurrent_receipt, canonical_prepared
                    )
                    connection.rollback()
                    return result
                if generation != canonical_prepared.registry_generation:
                    raise RepositoryRelinkConflictError(
                        "repository relink registry generation changed"
                    )

                self._assert_binding_policy(
                    bindings,
                    canonical_prepared.locator_repository_key,
                    canonical_prepared.authority_repository_key,
                    canonical_prepared.binding_id,
                )
                existing = bindings.get(
                    canonical_prepared.locator_repository_key
                )
                if existing is not None:
                    event = events[existing["event_id"]]
                    result = self._already_bound_result(
                        canonical_prepared,
                        event,
                        generation,
                    )
                    self._insert_receipt(
                        connection,
                        canonical_prepared,
                        result,
                    )
                    self._run_transaction_hook("after_receipt_insert")
                    connection.commit()
                    return result

                next_generation = generation + 1
                previous_hash = (
                    _GENESIS_EVENT_HASH
                    if not events
                    else events[max(events, key=lambda event_id: events[event_id].sequence)].event_hash
                )
                event = RepositoryRelinkEvent(
                    sequence=next_generation,
                    generation=next_generation,
                    request_id=canonical_prepared.request_id,
                    descriptor=canonical_prepared.descriptor,
                    binding_id=canonical_prepared.binding_id,
                    old_authority_state_token=(
                        canonical_prepared.old_authority_state_token
                    ),
                    actor=canonical_prepared.actor,
                    reason=canonical_prepared.reason,
                    previous_event_hash=previous_hash,
                    created_at=_utc_now(),
                )
                self._insert_binding(connection, canonical_prepared, event)
                self._run_transaction_hook("after_binding_insert")
                self._insert_event(connection, event)
                self._run_transaction_hook("after_event_insert")
                cursor = connection.execute(
                    """
                    UPDATE registry_state
                    SET generation = ?
                    WHERE singleton = 1 AND generation = ?
                    """,
                    (next_generation, generation),
                )
                if cursor.rowcount != 1:
                    raise RepositoryRelinkConflictError(
                        "repository relink registry generation changed"
                    )
                self._run_transaction_hook("after_generation_update")
                resolution = RepositoryAuthorityResolution(
                    locator_identity=canonical_prepared.locator_identity,
                    authority_identity=canonical_prepared.authority_identity,
                    binding_id=canonical_prepared.binding_id,
                    authority_resolution_hash=(
                        canonical_prepared.authority_resolution_hash
                    ),
                    registry_generation=next_generation,
                )
                result = RepositoryRelinkResult(
                    request_id=canonical_prepared.request_id,
                    resolution=resolution,
                    event=event,
                    outcome="applied",
                )
                self._insert_receipt(
                    connection,
                    canonical_prepared,
                    result,
                )
                self._run_transaction_hook("after_receipt_insert")
                connection.commit()
                return result
            except RepositoryRelinkError:
                if connection is not None:
                    connection.rollback()
                raise
            except (MemoryIdentityError, sqlite3.Error, OSError, TypeError, ValueError):
                if connection is not None:
                    connection.rollback()
                raise RepositoryRelinkError(
                    "repository relink transaction failed"
                ) from None
            except Exception:
                if connection is not None:
                    connection.rollback()
                raise RepositoryRelinkError(
                    "repository relink transaction failed"
                ) from None
            finally:
                if connection is not None:
                    connection.close()

    apply = apply_relink

    def resolve_authority(
        self,
        locator_identity: IdentityDescriptorInput,
        *,
        revision_resolver: Optional[RevisionResolver] = None,
    ) -> RepositoryAuthorityResolution:
        """Resolve a freshly verified live locator without origin inference."""

        locator_claim = _identity_descriptor(locator_identity, "locator identity")
        locator = self._verify_live_locator(
            locator_claim,
            revision_resolver=revision_resolver,
        )
        with self._lock:
            generation, bindings, _, _ = self._read_verified_state()
        binding = bindings.get(locator.repository_key)
        if binding is None:
            return RepositoryAuthorityResolution(
                locator_identity=locator,
                authority_identity=locator,
                binding_id=None,
                authority_resolution_hash=repository_authority_resolution_hash(
                    locator.repository_key,
                    locator.repository_key,
                ),
                registry_generation=generation,
            )
        authority = binding["authority_identity"]
        binding_id = binding["binding_id"]
        return RepositoryAuthorityResolution(
            locator_identity=locator,
            authority_identity=authority,
            binding_id=binding_id,
            authority_resolution_hash=repository_authority_resolution_hash(
                locator.repository_key,
                authority.repository_key,
                binding_id=binding_id,
            ),
            registry_generation=generation,
        )

    resolve = resolve_authority

    def generation(self) -> int:
        with self._lock:
            generation, _, _, _ = self._read_verified_state()
            return generation

    def get_request_receipt(
        self,
        request_id: str,
    ) -> Optional[RepositoryRelinkRequestReceipt]:
        """Return one verified root-local apply receipt without writing state."""

        canonical_request_id = _stable_id(request_id, "REQ", "request_id")
        with self._lock:
            _, _, _, receipts = self._read_verified_state()
        return receipts.get(canonical_request_id)

    def verify_event_chain(self) -> Tuple[RepositoryRelinkEvent, ...]:
        with self._lock:
            _, _, events, _ = self._read_verified_state()
        return tuple(sorted(events.values(), key=lambda item: item.sequence))

    def _verify_live_locator(
        self,
        identity: RepositoryIdentityDescriptor,
        *,
        revision_resolver: Optional[RevisionResolver],
        conflict_on_failure: bool = False,
    ) -> RepositoryIdentityDescriptor:
        try:
            plan = plan_repository_memory_namespace(
                identity,
                self._root,
                revision_resolver=(revision_resolver or self._revision_resolver),
            )
            return plan.locator.identity.descriptor
        except (MemoryIdentityError, OSError, RuntimeError, TypeError, ValueError):
            if conflict_on_failure:
                raise RepositoryRelinkConflictError(
                    "live repository locator changed after relink preparation"
                ) from None
            raise RepositoryRelinkValidationError(
                "unable to verify live repository locator"
            ) from None

    def _canonical_prepared(
        self, prepared: PreparedRepositoryRelink
    ) -> PreparedRepositoryRelink:
        if not isinstance(prepared, PreparedRepositoryRelink):
            raise RepositoryRelinkValidationError(
                "prepared repository relink is invalid"
            )
        try:
            hydrated = PreparedRepositoryRelink.from_payload(prepared.to_payload())
        except (RepositoryRelinkError, AttributeError, TypeError, ValueError):
            raise RepositoryRelinkValidationError(
                "prepared repository relink is invalid"
            ) from None
        if hydrated != prepared:
            raise RepositoryRelinkValidationError(
                "prepared repository relink is not canonical"
            )
        return hydrated

    def _read_receipt_if_present(
        self, request_id: str
    ) -> Optional[RepositoryRelinkRequestReceipt]:
        _, _, _, receipts = self._read_verified_state()
        return receipts.get(request_id)

    @staticmethod
    def _replay_receipt(
        receipt: RepositoryRelinkRequestReceipt,
        prepared: PreparedRepositoryRelink,
    ) -> RepositoryRelinkResult:
        if not hmac.compare_digest(receipt.semantic_hash, prepared.semantic_hash):
            raise RepositoryRelinkConflictError(
                "repository relink request ID already has different semantics"
            )
        return receipt.result

    @staticmethod
    def _assert_binding_policy(
        bindings: Mapping[str, Mapping[str, Any]],
        locator_key: str,
        authority_key: str,
        binding_id: str,
    ) -> None:
        existing = bindings.get(locator_key)
        if existing is not None:
            if (
                existing["authority_repository_key"] != authority_key
                or existing["binding_id"] != binding_id
            ):
                raise RepositoryRelinkConflictError(
                    "repository locator already has a different authority"
                )
            return
        if authority_key in bindings:
            raise RepositoryRelinkConflictError(
                "repository relink authority must be an unbound root"
            )
        if any(
            item["authority_repository_key"] == locator_key
            for item in bindings.values()
        ):
            raise RepositoryRelinkConflictError(
                "repository relink would create an authority chain"
            )

    @staticmethod
    def _already_bound_result(
        prepared: PreparedRepositoryRelink,
        event: RepositoryRelinkEvent,
        generation: int,
    ) -> RepositoryRelinkResult:
        resolution = RepositoryAuthorityResolution(
            locator_identity=prepared.locator_identity,
            authority_identity=prepared.authority_identity,
            binding_id=prepared.binding_id,
            authority_resolution_hash=prepared.authority_resolution_hash,
            registry_generation=generation,
        )
        return RepositoryRelinkResult(
            request_id=prepared.request_id,
            resolution=resolution,
            event=event,
            outcome="already_bound",
        )

    def _connect_readonly(self) -> sqlite3.Connection:
        self._inspect_storage_paths(require_database=True)
        before = _directory_identity(self._root)
        try:
            uri = self._database_path.as_uri() + "?mode=ro"
            connection = sqlite3.connect(
                uri,
                uri=True,
                timeout=self._timeout,
                isolation_level=None,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only = ON")
            connection.execute("PRAGMA busy_timeout = %d" % int(self._timeout * 1000))
        except (OSError, sqlite3.Error, ValueError):
            raise RepositoryRelinkIntegrityError(
                "repository relink registry is unavailable"
            ) from None
        if before != _directory_identity(self._root):
            connection.close()
            raise RepositoryRelinkIntegrityError(
                "repository relink registry root changed"
            )
        return connection

    def _connect_writable(self) -> sqlite3.Connection:
        try:
            materialized = resolve_memory_root(self._root, env={}, create=True)
        except MemoryIdentityError:
            raise RepositoryRelinkValidationError(
                "Memory root is not safe for repository relinks"
            ) from None
        if materialized != self._root:
            raise RepositoryRelinkIntegrityError(
                "repository relink registry root changed"
            )
        self._inspect_storage_paths()
        before = _directory_identity(self._root)
        try:
            connection = sqlite3.connect(
                str(self._database_path),
                timeout=self._timeout,
                isolation_level=None,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout = %d" % int(self._timeout * 1000))
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA synchronous = FULL")
        except (OSError, sqlite3.Error):
            raise RepositoryRelinkError(
                "repository relink registry is unavailable"
            ) from None
        self._inspect_storage_paths(require_database=True)
        if before != _directory_identity(self._root):
            connection.close()
            raise RepositoryRelinkIntegrityError(
                "repository relink registry root changed"
            )
        return connection

    @staticmethod
    def _initialize_schema(
        connection: sqlite3.Connection,
        registry_root_hash: str,
    ) -> None:
        # ``executescript`` performs an implicit commit in sqlite3.  Execute the
        # fixed statements individually so schema creation remains in the same
        # rollback boundary as binding, event, generation, and receipt writes.
        for statement in _SCHEMA_SQL.split(";"):
            if statement.strip():
                connection.execute(statement)
        connection.execute(
            "INSERT OR IGNORE INTO registry_state(singleton, schema_version, generation, root_hash) VALUES (1, ?, 0, ?)",
            (
                REPOSITORY_RELINK_REGISTRY_SCHEMA_VERSION,
                registry_root_hash,
            ),
        )
        connection.execute("PRAGMA application_id = %d" % _APPLICATION_ID)
        connection.execute(
            "PRAGMA user_version = %d"
            % REPOSITORY_RELINK_REGISTRY_SCHEMA_VERSION
        )

    def _bootstrap_schema(self, connection: sqlite3.Connection) -> None:
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._initialize_schema(connection, self.registry_root_hash)
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    def _read_verified_state(
        self,
    ) -> Tuple[
        int,
        Dict[str, Mapping[str, Any]],
        Dict[str, RepositoryRelinkEvent],
        Dict[str, RepositoryRelinkRequestReceipt],
    ]:
        self._inspect_storage_paths()
        if not self._database_path.exists():
            return 0, {}, {}, {}
        connection: Optional[sqlite3.Connection] = None
        try:
            connection = self._connect_readonly()
            connection.execute("BEGIN")
            verified = self._verify_connection(connection)
            connection.rollback()
            return verified
        except RepositoryRelinkIntegrityError:
            raise
        except RepositoryRelinkError:
            raise RepositoryRelinkIntegrityError(
                "repository relink registry failed integrity verification"
            ) from None
        except (MemoryIdentityError, sqlite3.Error, OSError, TypeError, ValueError):
            raise RepositoryRelinkIntegrityError(
                "repository relink registry failed integrity verification"
            ) from None
        finally:
            if connection is not None:
                connection.close()

    def _verify_connection(
        self,
        connection: sqlite3.Connection,
    ) -> Tuple[
        int,
        Dict[str, Mapping[str, Any]],
        Dict[str, RepositoryRelinkEvent],
        Dict[str, RepositoryRelinkRequestReceipt],
    ]:
        try:
            tables = {
                row["name"]
                for row in connection.execute(
                    "SELECT name FROM sqlite_schema WHERE type = 'table'"
                ).fetchall()
                if not row["name"].startswith("sqlite_")
            }
            if tables != _EXPECTED_TABLES:
                raise RepositoryRelinkIntegrityError(
                    "repository relink registry schema is invalid"
                )
            application_id = connection.execute(
                "PRAGMA application_id"
            ).fetchone()[0]
            user_version = connection.execute("PRAGMA user_version").fetchone()[0]
            if (
                application_id != _APPLICATION_ID
                or user_version != REPOSITORY_RELINK_REGISTRY_SCHEMA_VERSION
            ):
                raise RepositoryRelinkIntegrityError(
                    "repository relink registry schema is invalid"
                )
            check = connection.execute("PRAGMA quick_check").fetchall()
            if len(check) != 1 or check[0][0] != "ok":
                raise RepositoryRelinkIntegrityError(
                    "repository relink registry storage is invalid"
                )
            state_rows = connection.execute(
                "SELECT singleton, schema_version, generation, root_hash FROM registry_state"
            ).fetchall()
            if len(state_rows) != 1:
                raise RepositoryRelinkIntegrityError(
                    "repository relink registry state is invalid"
                )
            state = state_rows[0]
            generation = _generation(state["generation"])
            if (
                state["singleton"] != 1
                or state["schema_version"]
                != REPOSITORY_RELINK_REGISTRY_SCHEMA_VERSION
                or state["root_hash"] != self.registry_root_hash
            ):
                raise RepositoryRelinkIntegrityError(
                    "repository relink registry state is invalid"
                )

            events: Dict[str, RepositoryRelinkEvent] = {}
            previous_hash = _GENESIS_EVENT_HASH
            event_rows = connection.execute(
                "SELECT * FROM events ORDER BY sequence"
            ).fetchall()
            for expected_sequence, row in enumerate(event_rows, start=1):
                payload = _canonical_json_object(
                    row["event_json"], "repository relink event"
                )
                event = RepositoryRelinkEvent.from_payload(payload)
                if (
                    event.sequence != expected_sequence
                    or event.generation != expected_sequence
                    or event.sequence != row["sequence"]
                    or event.generation != row["generation"]
                    or event.event_id != row["event_id"]
                    or event.request_id != row["request_id"]
                    or event.binding_id != row["binding_id"]
                    or event.previous_event_hash != row["previous_event_hash"]
                    or event.event_hash != row["event_hash"]
                    or not hmac.compare_digest(
                        event.previous_event_hash, previous_hash
                    )
                    or event.event_id in events
                ):
                    raise RepositoryRelinkIntegrityError(
                        "repository relink event chain is invalid"
                    )
                events[event.event_id] = event
                previous_hash = event.event_hash
            if generation != len(event_rows):
                raise RepositoryRelinkIntegrityError(
                    "repository relink generation is invalid"
                )

            bindings: Dict[str, Mapping[str, Any]] = {}
            binding_rows = connection.execute(
                "SELECT * FROM bindings ORDER BY locator_repository_key"
            ).fetchall()
            for row in binding_rows:
                locator = RepositoryIdentityDescriptor.from_payload(
                    _canonical_json_object(
                        row["locator_descriptor_json"], "locator descriptor"
                    )
                )
                authority = RepositoryIdentityDescriptor.from_payload(
                    _canonical_json_object(
                        row["authority_descriptor_json"], "authority descriptor"
                    )
                )
                descriptor = build_relink_descriptor(authority, locator)
                binding = repository_binding_id(
                    locator.repository_key, authority.repository_key
                )
                event = events.get(row["event_id"])
                if (
                    row["locator_repository_key"] != locator.repository_key
                    or row["authority_repository_key"] != authority.repository_key
                    or row["binding_id"] != binding
                    or row["descriptor_hash"]
                    != canonical_sha256(descriptor.to_payload())
                    or event is None
                    or row["generation"] != event.generation
                    or event.descriptor.to_payload() != descriptor.to_payload()
                    or event.binding_id != binding
                    or locator.repository_key in bindings
                ):
                    raise RepositoryRelinkIntegrityError(
                        "repository relink binding registry is invalid"
                    )
                bindings[locator.repository_key] = {
                    "locator_identity": locator,
                    "authority_identity": authority,
                    "authority_repository_key": authority.repository_key,
                    "binding_id": binding,
                    "event_id": event.event_id,
                    "generation": event.generation,
                }
            if len(bindings) != len(events):
                raise RepositoryRelinkIntegrityError(
                    "repository relink binding registry is invalid"
                )
            locator_keys = set(bindings)
            if any(
                item["authority_repository_key"] in locator_keys
                for item in bindings.values()
            ):
                raise RepositoryRelinkIntegrityError(
                    "repository relink authority chain is invalid"
                )

            receipts: Dict[str, RepositoryRelinkRequestReceipt] = {}
            receipt_rows = connection.execute(
                "SELECT * FROM request_receipts ORDER BY request_id"
            ).fetchall()
            for row in receipt_rows:
                receipt = RepositoryRelinkRequestReceipt.from_payload(
                    _canonical_json_object(
                        row["receipt_json"], "repository relink receipt"
                    )
                )
                if (
                    receipt.request_id != row["request_id"]
                    or receipt.semantic_hash != row["semantic_hash"]
                    or receipt.result.result_hash != row["result_hash"]
                    or receipt.receipt_hash != row["receipt_hash"]
                    or receipt.prepared.registry_root_hash
                    != self.registry_root_hash
                    or receipt.request_id in receipts
                    or receipt.result.event.event_id not in events
                ):
                    raise RepositoryRelinkIntegrityError(
                        "repository relink request receipt is invalid"
                    )
                receipts[receipt.request_id] = receipt
            if any(event.request_id not in receipts for event in events.values()):
                raise RepositoryRelinkIntegrityError(
                    "repository relink event receipt is missing"
                )
            return generation, bindings, events, receipts
        except RepositoryRelinkIntegrityError:
            raise
        except RepositoryRelinkError:
            raise RepositoryRelinkIntegrityError(
                "repository relink registry failed integrity verification"
            ) from None
        except (MemoryIdentityError, sqlite3.Error, TypeError, ValueError):
            raise RepositoryRelinkIntegrityError(
                "repository relink registry failed integrity verification"
            ) from None

    @staticmethod
    def _insert_binding(
        connection: sqlite3.Connection,
        prepared: PreparedRepositoryRelink,
        event: RepositoryRelinkEvent,
    ) -> None:
        connection.execute(
            """
            INSERT INTO bindings(
                locator_repository_key, authority_repository_key, binding_id,
                locator_descriptor_json, authority_descriptor_json,
                descriptor_hash, event_id, generation
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                prepared.locator_repository_key,
                prepared.authority_repository_key,
                prepared.binding_id,
                canonical_json(prepared.locator_identity.to_payload()),
                canonical_json(prepared.authority_identity.to_payload()),
                prepared.descriptor_hash,
                event.event_id,
                event.generation,
            ),
        )

    @staticmethod
    def _insert_event(
        connection: sqlite3.Connection,
        event: RepositoryRelinkEvent,
    ) -> None:
        connection.execute(
            """
            INSERT INTO events(
                sequence, generation, event_id, request_id, binding_id,
                previous_event_hash, event_hash, event_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.sequence,
                event.generation,
                event.event_id,
                event.request_id,
                event.binding_id,
                event.previous_event_hash,
                event.event_hash,
                canonical_json(event.to_payload()),
            ),
        )

    @staticmethod
    def _insert_receipt(
        connection: sqlite3.Connection,
        prepared: PreparedRepositoryRelink,
        result: RepositoryRelinkResult,
    ) -> RepositoryRelinkRequestReceipt:
        receipt = RepositoryRelinkRequestReceipt(
            request_id=result.request_id,
            semantic_hash=prepared.semantic_hash,
            prepared=prepared,
            result=result,
            created_at=_utc_now(),
        )
        connection.execute(
            """
            INSERT INTO request_receipts(
                request_id, semantic_hash, result_hash, receipt_hash,
                receipt_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                receipt.request_id,
                receipt.semantic_hash,
                receipt.result.result_hash,
                receipt.receipt_hash,
                canonical_json(receipt.to_payload()),
            ),
        )
        return receipt

    def _run_transaction_hook(self, stage: str) -> None:
        if self._transaction_hook is None:
            return
        try:
            self._transaction_hook(stage)
        except Exception:
            raise RepositoryRelinkError(
                "repository relink transaction failed"
            ) from None

    def _validate_bounded_paths(self) -> None:
        if len(REPOSITORY_RELINK_REGISTRY_FILENAME) > 64:
            raise RepositoryRelinkValidationError(
                "repository relink registry filename is invalid"
            )
        if sys.platform.startswith("win"):
            root_units = len(str(self._root).encode("utf-16-le")) // 2
            file_units = len(str(self._database_path).encode("utf-16-le")) // 2
            if root_units > 247 or file_units > 259:
                raise RepositoryRelinkValidationError(
                    "repository relink registry exceeds the bounded path policy"
                )
        if self._database_path.parent != self._root:
            raise RepositoryRelinkValidationError(
                "repository relink registry escapes the Memory root"
            )

    def _inspect_storage_paths(self, *, require_database: bool = False) -> None:
        try:
            canonical = resolve_memory_root(self._root, env={}, create=False)
        except MemoryIdentityError:
            raise RepositoryRelinkIntegrityError(
                "repository relink registry path is unsafe"
            ) from None
        if canonical != self._root:
            raise RepositoryRelinkIntegrityError(
                "repository relink registry path changed"
            )
        for path in (
            self._database_path,
            *(Path(str(self._database_path) + suffix) for suffix in _SQLITE_SIDECAR_SUFFIXES),
        ):
            try:
                metadata = path.lstat()
            except FileNotFoundError:
                continue
            except OSError:
                raise RepositoryRelinkIntegrityError(
                    "unable to inspect repository relink registry"
                ) from None
            attributes = getattr(metadata, "st_file_attributes", 0)
            if (
                stat.S_ISLNK(metadata.st_mode)
                or attributes & _FILE_ATTRIBUTE_REPARSE_POINT
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
            ):
                raise RepositoryRelinkIntegrityError(
                    "repository relink registry path is unsafe"
                )
        if require_database and not self._database_path.is_file():
            raise RepositoryRelinkIntegrityError(
                "repository relink registry is unavailable"
            )


def resolve_repository_authority(
    memory_root: Union[PathInput, ResolvedMemoryRoot],
    locator_identity: IdentityDescriptorInput,
    *,
    revision_resolver: Optional[RevisionResolver] = None,
) -> RepositoryAuthorityResolution:
    return RepositoryRelinkRegistry(
        memory_root,
        revision_resolver=revision_resolver,
    ).resolve_authority(locator_identity)


def get_repository_relink_receipt(
    memory_root: Union[PathInput, ResolvedMemoryRoot],
    request_id: str,
) -> Optional[RepositoryRelinkRequestReceipt]:
    """Read one verified apply audit receipt from a root-scoped registry."""

    return RepositoryRelinkRegistry(memory_root).get_request_receipt(request_id)


def prepare_relink(
    memory_root: Union[PathInput, ResolvedMemoryRoot],
    old_identity: IdentityDescriptorInput,
    new_identity: IdentityDescriptorInput,
    *,
    from_repository_key: str,
    actor: str,
    reason: str,
    request_id: str,
    revision_resolver: Optional[RevisionResolver] = None,
) -> PreparedRepositoryRelink:
    return RepositoryRelinkRegistry(
        memory_root,
        revision_resolver=revision_resolver,
    ).prepare_relink(
        old_identity,
        new_identity,
        from_repository_key=from_repository_key,
        actor=actor,
        reason=reason,
        request_id=request_id,
    )


prepare_repository_relink = prepare_relink


def apply_relink(
    memory_root: Union[PathInput, ResolvedMemoryRoot],
    prepared: PreparedRepositoryRelink,
    *,
    revision_resolver: Optional[RevisionResolver] = None,
) -> RepositoryRelinkResult:
    return RepositoryRelinkRegistry(
        memory_root,
        revision_resolver=revision_resolver,
    ).apply_relink(
        prepared,
        revision_resolver=revision_resolver,
    )


apply_repository_relink = apply_relink


def _identity_descriptor(
    identity: IdentityDescriptorInput,
    field_name: str,
) -> RepositoryIdentityDescriptor:
    descriptor = (
        identity.descriptor
        if isinstance(identity, VerifiedRepositoryIdentity)
        else identity
    )
    if not isinstance(descriptor, RepositoryIdentityDescriptor):
        raise RepositoryRelinkValidationError("%s is invalid" % field_name)
    try:
        hydrated = RepositoryIdentityDescriptor.from_payload(
            descriptor.to_payload()
        )
    except (MemoryIdentityError, AttributeError, TypeError, ValueError):
        raise RepositoryRelinkValidationError("%s is invalid" % field_name) from None
    if hydrated.to_payload() != descriptor.to_payload():
        raise RepositoryRelinkValidationError(
            "%s is not canonical" % field_name
        )
    return hydrated


def _relink_descriptor(value: Any) -> RepositoryRelinkDescriptor:
    if not isinstance(value, RepositoryRelinkDescriptor):
        raise RepositoryRelinkValidationError(
            "repository relink descriptor is invalid"
        )
    try:
        hydrated = RepositoryRelinkDescriptor.from_payload(value.to_payload())
    except (MemoryIdentityError, AttributeError, TypeError, ValueError):
        raise RepositoryRelinkValidationError(
            "repository relink descriptor is invalid"
        ) from None
    if hydrated.to_payload() != value.to_payload():
        raise RepositoryRelinkValidationError(
            "repository relink descriptor is not canonical"
        )
    return hydrated


def _object(value: Any, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RepositoryRelinkValidationError("%s payload is invalid" % context)
    return value


def _exact_fields(
    payload: Mapping[str, object],
    expected: set,
    context: str,
) -> None:
    if set(payload) != expected:
        raise RepositoryRelinkValidationError("%s payload is invalid" % context)


def _schema_version(value: Any, context: str) -> int:
    if type(value) is not int or value != REPOSITORY_RELINK_REGISTRY_SCHEMA_VERSION:
        raise RepositoryRelinkValidationError("%s schema is invalid" % context)
    return value


def _positive_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise RepositoryRelinkValidationError("%s is invalid" % field_name)
    return value


def _generation(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise RepositoryRelinkValidationError(
            "repository relink generation is invalid"
        )
    return value


def _digest(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise RepositoryRelinkValidationError("%s is invalid" % field_name)
    return value


def _stable_id(value: Any, prefix: str, field_name: str) -> str:
    try:
        return validate_stable_id(value, prefix, field_name)
    except ValueError:
        raise RepositoryRelinkValidationError("%s is invalid" % field_name) from None


def _text(value: Any, field_name: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
        or any(ord(character) < 32 for character in value)
    ):
        raise RepositoryRelinkValidationError("%s is invalid" % field_name)
    return value


def _timestamp(value: Any) -> str:
    if not isinstance(value, str) or _TIMESTAMP_PATTERN.fullmatch(value) is None:
        raise RepositoryRelinkValidationError(
            "repository relink timestamp is invalid"
        )
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        raise RepositoryRelinkValidationError(
            "repository relink timestamp is invalid"
        ) from None
    if parsed.tzinfo != timezone.utc:
        raise RepositoryRelinkValidationError(
            "repository relink timestamp is invalid"
        )
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _canonical_json_object(text: Any, context: str) -> Mapping[str, object]:
    if not isinstance(text, str):
        raise RepositoryRelinkIntegrityError("%s storage is invalid" % context)
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        raise RepositoryRelinkIntegrityError("%s storage is invalid" % context) from None
    if not isinstance(payload, Mapping) or canonical_json(payload) != text:
        raise RepositoryRelinkIntegrityError("%s storage is invalid" % context)
    return payload


def _directory_identity(path: Path) -> Tuple[int, int]:
    try:
        metadata = path.lstat()
    except OSError:
        raise RepositoryRelinkIntegrityError(
            "repository relink registry root is unavailable"
        ) from None
    attributes = getattr(metadata, "st_file_attributes", 0)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or attributes & _FILE_ATTRIBUTE_REPARSE_POINT
    ):
        raise RepositoryRelinkIntegrityError(
            "repository relink registry root is unsafe"
        )
    return int(metadata.st_dev), int(metadata.st_ino)


__all__ = [
    "REPOSITORY_RELINK_EVENT_TYPE",
    "REPOSITORY_RELINK_REGISTRY_FILENAME",
    "REPOSITORY_RELINK_REGISTRY_SCHEMA_VERSION",
    "PreparedRepositoryRelink",
    "RepositoryAuthorityResolution",
    "RepositoryRelinkConflictError",
    "RepositoryRelinkError",
    "RepositoryRelinkEvent",
    "RepositoryRelinkIntegrityError",
    "RepositoryRelinkRegistry",
    "RepositoryRelinkRequestReceipt",
    "RepositoryRelinkResult",
    "RepositoryRelinkValidationError",
    "apply_relink",
    "apply_repository_relink",
    "get_repository_relink_receipt",
    "prepare_relink",
    "prepare_repository_relink",
    "repository_authority_resolution_hash",
    "repository_binding_id",
    "resolve_repository_authority",
]
