"""Exact-revision, immutable Repository Knowledge cache.

The cache is a policy and lifecycle layer over the canonical durable-memory
models and :class:`~review_agent.memory_store.MemoryStore`.  Cache payloads are
content-addressed blobs; :class:`RepositoryKnowledgeEntry` instances are the
immutable revision manifests.  A hit is valid only after the exact key,
manifest metadata, blob size, and blob hash have all been verified.

The module deliberately knows nothing about Pipeline permissions.  Callers
must build a key only from repository inputs they are already authorized to
read, and a miss always invokes the caller-provided builder.  Consequently the
cache cannot become an alternate repository-read capability.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import re
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Tuple,
    Union,
)

from review_agent.memory_models import (
    MemoryMode,
    RepositoryKnowledgeCapability,
    RepositoryKnowledgeEntry,
    RepositoryKnowledgeKey,
    canonical_sha256,
    stable_request_id,
)
from review_agent.memory_store import (
    MemoryStore,
    MemoryStoreConflictError,
    MemoryStoreCorruptionError,
    MemoryStoreError,
    MemoryStoreReadOnlyError,
)


MAX_CACHE_ARTIFACT_BYTES = 512 * 1024 * 1024
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+#/@-]{0,511}$")
_CONTENT_TYPE_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*/"
    r"[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*$"
)


class RepositoryCacheError(RuntimeError):
    """Base error for deterministic Repository Knowledge cache failures."""


class RepositoryCacheValidationError(RepositoryCacheError, ValueError):
    """The caller supplied an invalid key, artifact, or cache operation."""


class RepositoryCacheConflictError(RepositoryCacheError):
    """An exact immutable key already names different canonical content."""


class RepositoryCacheModeError(RepositoryCacheError):
    """The requested persistent mutation is forbidden by the configured mode."""


class RepositoryCacheStatus(str, Enum):
    OFF = "off"
    HIT = "hit"
    MISS = "miss"
    REBUILD = "rebuild"


@dataclass(frozen=True)
class RepositoryCapabilityMetadata:
    capability: RepositoryKnowledgeCapability
    artifact_schema: str
    content_type: str


def _capability_metadata(
    capability: RepositoryKnowledgeCapability,
    media_name: str,
) -> RepositoryCapabilityMetadata:
    return RepositoryCapabilityMetadata(
        capability=capability,
        artifact_schema=capability.value + "_v1",
        content_type="application/vnd.review-agent.%s+json" % media_name,
    )


CAPABILITY_METADATA: Mapping[
    RepositoryKnowledgeCapability, RepositoryCapabilityMetadata
] = MappingProxyType(
    {
        RepositoryKnowledgeCapability.FILE_INDEX: _capability_metadata(
            RepositoryKnowledgeCapability.FILE_INDEX, "file-index"
        ),
        RepositoryKnowledgeCapability.SYMBOL_INDEX: _capability_metadata(
            RepositoryKnowledgeCapability.SYMBOL_INDEX, "symbol-index"
        ),
        RepositoryKnowledgeCapability.DEFINITIONS: _capability_metadata(
            RepositoryKnowledgeCapability.DEFINITIONS, "definitions"
        ),
        RepositoryKnowledgeCapability.REFERENCES: _capability_metadata(
            RepositoryKnowledgeCapability.REFERENCES, "references"
        ),
        RepositoryKnowledgeCapability.CALLS: _capability_metadata(
            RepositoryKnowledgeCapability.CALLS, "calls"
        ),
        RepositoryKnowledgeCapability.TESTS: _capability_metadata(
            RepositoryKnowledgeCapability.TESTS, "tests"
        ),
        RepositoryKnowledgeCapability.PROJECT_CONFIG: _capability_metadata(
            RepositoryKnowledgeCapability.PROJECT_CONFIG, "project-config"
        ),
        RepositoryKnowledgeCapability.GIT_SUMMARY: _capability_metadata(
            RepositoryKnowledgeCapability.GIT_SUMMARY, "git-summary"
        ),
    }
)


@dataclass(frozen=True)
class RepositoryKnowledgeArtifact:
    """A validated immutable payload before it is attached to a key manifest."""

    content: bytes
    content_type: str
    artifact_schema: str
    summary_hash: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.content, (bytes, bytearray, memoryview)):
            raise RepositoryCacheValidationError("cache artifact content must be bytes")
        content = bytes(self.content)
        if len(content) > MAX_CACHE_ARTIFACT_BYTES:
            raise RepositoryCacheValidationError(
                "cache artifact exceeds the supported size limit"
            )
        object.__setattr__(self, "content", content)

        if not isinstance(self.content_type, str):
            raise RepositoryCacheValidationError(
                "cache artifact content_type must be a media type"
            )
        normalized_content_type = self.content_type.casefold()
        if _CONTENT_TYPE_PATTERN.fullmatch(normalized_content_type) is None:
            raise RepositoryCacheValidationError(
                "cache artifact content_type must be a media type"
            )
        object.__setattr__(self, "content_type", normalized_content_type)

        if (
            not isinstance(self.artifact_schema, str)
            or _TOKEN_PATTERN.fullmatch(self.artifact_schema) is None
        ):
            raise RepositoryCacheValidationError(
                "cache artifact_schema must be a stable token"
            )
        object.__setattr__(self, "artifact_schema", self.artifact_schema.casefold())

        if self.summary_hash is not None:
            if (
                not isinstance(self.summary_hash, str)
                or _SHA256_PATTERN.fullmatch(self.summary_hash.casefold()) is None
            ):
                raise RepositoryCacheValidationError(
                    "cache artifact summary_hash must be a SHA-256 digest"
                )
            object.__setattr__(self, "summary_hash", self.summary_hash.casefold())

    @property
    def blob_hash(self) -> str:
        return hashlib.sha256(self.content).hexdigest()


def _canonical_fallback(value: Any) -> Tuple[Tuple[str, str], ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        raw_items = list(value.items())
    elif isinstance(value, tuple):
        raw_items = list(value)
    else:
        raise RepositoryCacheValidationError("fallback provenance must be a mapping")
    items = []
    for pair in raw_items:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise RepositoryCacheValidationError(
                "fallback provenance must contain key/value pairs"
            )
        key, item = pair
        if (
            not isinstance(key, str)
            or not key
            or len(key) > 512
            or _TOKEN_PATTERN.fullmatch(key) is None
        ):
            raise RepositoryCacheValidationError(
                "fallback provenance keys must be stable tokens"
            )
        if (
            not isinstance(item, str)
            or not item.strip()
            or item != item.strip()
            or len(item) > 512
        ):
            raise RepositoryCacheValidationError(
                "fallback provenance values must be non-empty strings"
            )
        items.append((key, item))
    if len({key for key, _ in items}) != len(items):
        raise RepositoryCacheValidationError("fallback provenance keys must be unique")
    return tuple(sorted(items))


@dataclass(frozen=True)
class RepositoryCacheProvenance:
    """Session-safe provenance for a hit, miss, or deterministic rebuild."""

    status: RepositoryCacheStatus
    key: RepositoryKnowledgeKey
    entry_id: Optional[str]
    blob_hash: Optional[str]
    persistent: bool
    session_pinned: bool
    fallback: Tuple[Tuple[str, str], ...] = ()
    corruption_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, RepositoryCacheStatus):
            raise RepositoryCacheValidationError(
                "cache provenance status must be a RepositoryCacheStatus"
            )
        if not isinstance(self.key, RepositoryKnowledgeKey):
            raise RepositoryCacheValidationError(
                "cache provenance key must be canonical"
            )
        if type(self.persistent) is not bool or type(self.session_pinned) is not bool:
            raise RepositoryCacheValidationError(
                "cache provenance persistence flags must be booleans"
            )
        if self.blob_hash is not None and (
            not isinstance(self.blob_hash, str)
            or _SHA256_PATTERN.fullmatch(self.blob_hash) is None
        ):
            raise RepositoryCacheValidationError(
                "cache provenance blob_hash must be a SHA-256 digest"
            )
        if self.entry_id is not None and (
            not isinstance(self.entry_id, str)
            or re.fullmatch(r"RKE-[0-9a-f]{64}", self.entry_id) is None
        ):
            raise RepositoryCacheValidationError(
                "cache provenance entry_id must be a repository knowledge ID"
            )
        if self.corruption_reason is not None and (
            not isinstance(self.corruption_reason, str)
            or _TOKEN_PATTERN.fullmatch(self.corruption_reason) is None
        ):
            raise RepositoryCacheValidationError(
                "cache provenance corruption_reason must be a stable token"
            )
        if self.persistent and (self.entry_id is None or self.blob_hash is None):
            raise RepositoryCacheValidationError(
                "persistent cache provenance requires an entry and blob"
            )
        if self.session_pinned and not self.persistent:
            raise RepositoryCacheValidationError(
                "Session-pinned cache provenance must be persistent"
            )
        if self.status is RepositoryCacheStatus.HIT and not self.persistent:
            raise RepositoryCacheValidationError(
                "cache hit provenance must name persistent validated content"
            )
        if self.status is RepositoryCacheStatus.OFF and (
            self.persistent or self.entry_id is not None
        ):
            raise RepositoryCacheValidationError(
                "off cache provenance cannot name persistent content"
            )
        object.__setattr__(self, "fallback", _canonical_fallback(self.fallback))

    @property
    def key_hash(self) -> str:
        return self.key.key_hash

    @property
    def analyzer_name(self) -> str:
        return self.key.analyzer_name

    @property
    def analyzer_version(self) -> str:
        return self.key.analyzer_version

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "key_hash": self.key.key_hash,
            "repository_key": self.key.repository_key,
            "revision_binding": self.key.revision_binding,
            "capability": self.key.capability.value,
            "configuration_digest": self.key.configuration_digest,
            "input_digest": self.key.input_digest,
            "analyzer": {
                "name": self.key.analyzer_name,
                "version": self.key.analyzer_version,
            },
            "entry_id": self.entry_id,
            "blob_hash": self.blob_hash,
            "persistent": self.persistent,
            "session_pinned": self.session_pinned,
            "fallback": dict(self.fallback),
            "corruption_reason": self.corruption_reason,
        }


@dataclass(frozen=True)
class RepositoryCacheResult:
    content: Optional[bytes]
    entry: Optional[RepositoryKnowledgeEntry]
    provenance: RepositoryCacheProvenance

    def __post_init__(self) -> None:
        if self.content is not None:
            if not isinstance(self.content, bytes):
                raise RepositoryCacheValidationError(
                    "cache result content must be immutable bytes"
                )
            if self.provenance.blob_hash != hashlib.sha256(self.content).hexdigest():
                raise RepositoryCacheValidationError(
                    "cache result content does not match provenance"
                )
        if self.entry is not None:
            if self.provenance.entry_id != self.entry.entry_id:
                raise RepositoryCacheValidationError(
                    "cache result entry does not match provenance"
                )
            if self.provenance.blob_hash != self.entry.blob_hash:
                raise RepositoryCacheValidationError(
                    "cache result blob does not match its manifest"
                )

    @property
    def manifest(self) -> Optional[RepositoryKnowledgeEntry]:
        return self.entry

    @property
    def is_hit(self) -> bool:
        return self.provenance.status is RepositoryCacheStatus.HIT


@dataclass(frozen=True)
class RepositoryCacheGCResult:
    candidate_entry_ids: Tuple[str, ...]
    deleted_entry_ids: Tuple[str, ...]
    retained_pinned_entry_ids: Tuple[str, ...]
    deleted_blob_hashes: Tuple[str, ...]
    dry_run: bool


@dataclass(frozen=True)
class _LookupState:
    result: RepositoryCacheResult
    invalid_entry: Optional[RepositoryKnowledgeEntry] = None


ArtifactBuilder = Callable[
    [], Union[RepositoryKnowledgeArtifact, bytes, bytearray, memoryview]
]
ArtifactValidator = Callable[[bytes], Any]
Clock = Callable[[], str]


def digest_repository_configuration(configuration: Any) -> str:
    """Return the canonical digest used for every analyzer configuration input."""

    try:
        return canonical_sha256(configuration)
    except (TypeError, ValueError) as error:
        raise RepositoryCacheValidationError(
            "repository cache configuration is not canonical JSON"
        ) from error


def digest_repository_input(inputs: Any) -> str:
    """Return the canonical digest used for authorized analyzer inputs."""

    try:
        return canonical_sha256(inputs)
    except (TypeError, ValueError) as error:
        raise RepositoryCacheValidationError(
            "repository cache input is not canonical JSON"
        ) from error


def build_repository_knowledge_key(
    *,
    repository_key: str,
    revision_binding: str,
    capability: RepositoryKnowledgeCapability,
    analyzer_name: str,
    analyzer_version: str,
    configuration: Any = None,
    inputs: Any = None,
    configuration_digest: Optional[str] = None,
    input_digest: Optional[str] = None,
) -> RepositoryKnowledgeKey:
    """Build the canonical key while preventing partially bound dimensions."""

    if configuration_digest is not None and configuration is not None:
        raise RepositoryCacheValidationError(
            "provide configuration or configuration_digest, not both"
        )
    if input_digest is not None and inputs is not None:
        raise RepositoryCacheValidationError(
            "provide inputs or input_digest, not both"
        )
    resolved_configuration_digest = (
        digest_repository_configuration({} if configuration is None else configuration)
        if configuration_digest is None
        else configuration_digest
    )
    resolved_input_digest = (
        digest_repository_input({} if inputs is None else inputs)
        if input_digest is None
        else input_digest
    )
    try:
        return RepositoryKnowledgeKey(
            repository_key=repository_key,
            revision_binding=revision_binding,
            capability=capability,
            analyzer_name=analyzer_name,
            analyzer_version=analyzer_version,
            configuration_digest=resolved_configuration_digest,
            input_digest=resolved_input_digest,
        )
    except ValueError as error:
        raise RepositoryCacheValidationError(str(error)) from error


def validate_repository_knowledge_manifest(
    entry: RepositoryKnowledgeEntry,
    key: RepositoryKnowledgeKey,
    content: Union[bytes, bytearray, memoryview],
    *,
    content_type: Optional[str] = None,
    artifact_schema: Optional[str] = None,
) -> RepositoryKnowledgeEntry:
    """Validate an exact manifest and its content-addressed payload."""

    if not isinstance(entry, RepositoryKnowledgeEntry):
        raise RepositoryCacheValidationError(
            "repository knowledge manifest must be canonical"
        )
    if not isinstance(key, RepositoryKnowledgeKey):
        raise RepositoryCacheValidationError("repository knowledge key must be canonical")
    if not isinstance(content, (bytes, bytearray, memoryview)):
        raise RepositoryCacheValidationError("repository knowledge content must be bytes")
    raw = bytes(content)
    if entry.key != key or entry.key.key_hash != key.key_hash:
        raise RepositoryCacheValidationError(
            "repository knowledge manifest does not match the exact key"
        )
    if entry.size_bytes != len(raw):
        raise RepositoryCacheValidationError(
            "repository knowledge manifest size does not match its blob"
        )
    if entry.blob_hash != hashlib.sha256(raw).hexdigest():
        raise RepositoryCacheValidationError(
            "repository knowledge manifest hash does not match its blob"
        )
    if content_type is not None and entry.content_type != content_type.casefold():
        raise RepositoryCacheValidationError(
            "repository knowledge manifest content type does not match"
        )
    if artifact_schema is not None and entry.artifact_schema != artifact_schema.casefold():
        raise RepositoryCacheValidationError(
            "repository knowledge manifest artifact schema does not match"
        )
    return entry


class RepositoryKnowledgeCache:
    """Mode-aware immutable cache using canonical MemoryStore operations."""

    def __init__(
        self,
        store: Optional[MemoryStore] = None,
        *,
        mode: MemoryMode = MemoryMode.READ_WRITE,
        clock: Optional[Clock] = None,
    ) -> None:
        if not isinstance(mode, MemoryMode):
            raise RepositoryCacheValidationError("cache mode must be a MemoryMode")
        if clock is not None and not callable(clock):
            raise RepositoryCacheValidationError("cache clock must be callable")
        self._store = store
        self.mode = mode
        self._clock = clock or _utc_now

    @property
    def store(self) -> Optional[MemoryStore]:
        return self._store

    def lookup(
        self,
        key: RepositoryKnowledgeKey,
        *,
        review_id: Optional[str] = None,
        validator: Optional[ArtifactValidator] = None,
        fallback_provenance: Optional[Mapping[str, str]] = None,
        content_type: Optional[str] = None,
        artifact_schema: Optional[str] = None,
    ) -> RepositoryCacheResult:
        return self._lookup_state(
            key,
            review_id=review_id,
            validator=validator,
            fallback=_canonical_fallback(fallback_provenance),
            content_type=content_type,
            artifact_schema=artifact_schema,
        ).result

    def get_or_build(
        self,
        key: RepositoryKnowledgeKey,
        builder: ArtifactBuilder,
        *,
        review_id: Optional[str] = None,
        validator: Optional[ArtifactValidator] = None,
        fallback_provenance: Optional[Mapping[str, str]] = None,
        content_type: Optional[str] = None,
        artifact_schema: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> RepositoryCacheResult:
        if not callable(builder):
            raise RepositoryCacheValidationError("cache builder must be callable")
        fallback = _canonical_fallback(fallback_provenance)
        lookup = self._lookup_state(
            key,
            review_id=review_id,
            validator=validator,
            fallback=fallback,
            content_type=content_type,
            artifact_schema=artifact_schema,
        )
        if lookup.result.content is not None:
            return lookup.result

        artifact = self._normalize_artifact(
            key,
            builder(),
            content_type=content_type,
            artifact_schema=artifact_schema,
        )
        _validate_artifact_content(artifact.content, validator, cache_read=False)
        return self._persist_or_session(
            key,
            artifact,
            status=lookup.result.provenance.status,
            review_id=review_id,
            fallback=fallback,
            corruption_reason=lookup.result.provenance.corruption_reason,
            invalid_entry=lookup.invalid_entry,
            created_at=created_at,
            conflict_is_error=False,
        )

    def write(
        self,
        key: RepositoryKnowledgeKey,
        artifact: Union[RepositoryKnowledgeArtifact, bytes, bytearray, memoryview],
        *,
        review_id: Optional[str] = None,
        validator: Optional[ArtifactValidator] = None,
        fallback_provenance: Optional[Mapping[str, str]] = None,
        content_type: Optional[str] = None,
        artifact_schema: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> RepositoryCacheResult:
        fallback = _canonical_fallback(fallback_provenance)
        normalized = self._normalize_artifact(
            key,
            artifact,
            content_type=content_type,
            artifact_schema=artifact_schema,
        )
        _validate_artifact_content(normalized.content, validator, cache_read=False)

        lookup = self._lookup_state(
            key,
            review_id=review_id,
            validator=validator,
            fallback=fallback,
            content_type=normalized.content_type,
            artifact_schema=normalized.artifact_schema,
        )
        if lookup.result.content is not None:
            if not _entry_matches_artifact(lookup.result.entry, normalized):
                raise RepositoryCacheConflictError(
                    "exact cache key already has different immutable content"
                )
            return lookup.result
        return self._persist_or_session(
            key,
            normalized,
            status=lookup.result.provenance.status,
            review_id=review_id,
            fallback=fallback,
            corruption_reason=lookup.result.provenance.corruption_reason,
            invalid_entry=lookup.invalid_entry,
            created_at=created_at,
            conflict_is_error=True,
        )

    write_manifest = write

    def pin(
        self,
        entry_id: str,
        review_id: str,
        *,
        created_at: Optional[str] = None,
    ) -> bool:
        if self.mode is MemoryMode.OFF or self._store is None:
            return False
        return self._store.pin_knowledge_entry(
            entry_id,
            review_id,
            created_at=created_at,
        )

    def unpin(self, entry_id: str, review_id: str) -> bool:
        if self.mode is MemoryMode.OFF or self._store is None:
            return False
        return self._store.unpin_knowledge_entry(entry_id, review_id)

    def gc_unpinned_entries(
        self,
        repository_key: str,
        *,
        entry_ids: Optional[Iterable[str]] = None,
        dry_run: bool = True,
    ) -> RepositoryCacheGCResult:
        if type(dry_run) is not bool:
            raise RepositoryCacheValidationError("cache GC dry_run must be a boolean")
        if self.mode is MemoryMode.OFF or self._store is None:
            return RepositoryCacheGCResult((), (), (), (), dry_run)
        if not dry_run and self.mode is not MemoryMode.READ_WRITE:
            raise RepositoryCacheModeError(
                "persistent cache GC requires read-write mode"
            )

        if entry_ids is None:
            selected_ids = None
        else:
            if isinstance(entry_ids, (str, bytes)):
                raise RepositoryCacheValidationError(
                    "cache GC entry_ids must be an iterable of entry IDs"
                )
            try:
                selected_values = tuple(entry_ids)
            except TypeError:
                raise RepositoryCacheValidationError(
                    "cache GC entry_ids must be an iterable of entry IDs"
                ) from None
            if any(not isinstance(entry_id, str) for entry_id in selected_values):
                raise RepositoryCacheValidationError(
                    "cache GC entry_ids must contain strings"
                )
            selected_ids = frozenset(selected_values)
        entries = self._store.list_knowledge_entries(repository_key)
        if selected_ids is not None:
            entries = tuple(entry for entry in entries if entry.entry_id in selected_ids)
        candidates = tuple(
            sorted(entry.entry_id for entry in entries if not entry.pinned_by_review_ids)
        )
        retained = set(
            entry.entry_id for entry in entries if entry.pinned_by_review_ids
        )
        if dry_run:
            return RepositoryCacheGCResult(
                candidate_entry_ids=candidates,
                deleted_entry_ids=(),
                retained_pinned_entry_ids=tuple(sorted(retained)),
                deleted_blob_hashes=(),
                dry_run=True,
            )

        deleted = []
        by_id = {entry.entry_id: entry for entry in entries}
        timestamp = self._timestamp(None)
        for entry_id in candidates:
            entry = by_id[entry_id]
            try:
                generation = self._store.get_generations(
                    entry.key.repository_key
                ).knowledge_generation
                self._store.delete_knowledge_entry(
                    entry_id,
                    request_id=stable_request_id(
                        "repository_cache_gc",
                        entry.key.repository_key,
                        entry_id,
                        generation,
                    ),
                    expected_generation=generation,
                    created_at=timestamp,
                )
                deleted.append(entry_id)
            except MemoryStoreConflictError:
                # A concurrent Session pin wins over ordinary cache collection.
                retained.add(entry_id)
        blob_result = self._store.gc_blobs(dry_run=False)
        return RepositoryCacheGCResult(
            candidate_entry_ids=candidates,
            deleted_entry_ids=tuple(sorted(deleted)),
            retained_pinned_entry_ids=tuple(sorted(retained)),
            deleted_blob_hashes=tuple(sorted(blob_result.deleted_hashes)),
            dry_run=False,
        )

    gc = gc_unpinned_entries

    def _lookup_state(
        self,
        key: RepositoryKnowledgeKey,
        *,
        review_id: Optional[str],
        validator: Optional[ArtifactValidator],
        fallback: Tuple[Tuple[str, str], ...],
        content_type: Optional[str],
        artifact_schema: Optional[str],
    ) -> _LookupState:
        _require_key(key)
        if self.mode is MemoryMode.OFF:
            return _LookupState(
                self._empty_result(
                    key,
                    status=RepositoryCacheStatus.OFF,
                    fallback=fallback,
                )
            )
        if self._store is None:
            return _LookupState(
                self._empty_result(
                    key,
                    status=RepositoryCacheStatus.MISS,
                    fallback=fallback,
                    corruption_reason="unavailable",
                )
            )

        try:
            entry = self._store.find_knowledge_by_key(key)
        except MemoryStoreCorruptionError as error:
            return _LookupState(
                self._empty_result(
                    key,
                    status=RepositoryCacheStatus.REBUILD,
                    fallback=fallback,
                    corruption_reason=error.code.value,
                )
            )
        except MemoryStoreError as error:
            return _LookupState(
                self._empty_result(
                    key,
                    status=RepositoryCacheStatus.MISS,
                    fallback=fallback,
                    corruption_reason=error.code.value,
                )
            )
        if entry is None:
            return _LookupState(
                self._empty_result(
                    key,
                    status=RepositoryCacheStatus.MISS,
                    fallback=fallback,
                )
            )

        pinned = False
        pin_added = False
        projected = entry
        if review_id is not None:
            try:
                pin_added = self._store.pin_knowledge_entry(entry.entry_id, review_id)
                projected = self._store.get_knowledge_entry(entry.entry_id)
                pinned = review_id in projected.pinned_by_review_ids
            except MemoryStoreReadOnlyError:
                pinned = False
            except MemoryStoreError:
                pinned = False

        try:
            content = self._store.read_blob(entry.blob_hash)
            validate_repository_knowledge_manifest(
                entry,
                key,
                content,
                content_type=content_type,
                artifact_schema=artifact_schema,
            )
        except (MemoryStoreCorruptionError, RepositoryCacheValidationError) as error:
            if pin_added and review_id is not None:
                self._best_effort_unpin(entry.entry_id, review_id)
            reason = (
                error.code.value
                if isinstance(error, MemoryStoreCorruptionError)
                else "corruption"
            )
            return _LookupState(
                self._empty_result(
                    key,
                    status=RepositoryCacheStatus.REBUILD,
                    fallback=fallback,
                    corruption_reason=reason,
                ),
                # A semantically invalid, hash-valid artifact can be replaced
                # through immutable entry deletion.  A missing/corrupt blob is
                # repaired in place so pins held by earlier Sessions survive.
                invalid_entry=(
                    entry
                    if isinstance(error, RepositoryCacheValidationError)
                    else None
                ),
            )
        except MemoryStoreError as error:
            if pin_added and review_id is not None:
                self._best_effort_unpin(entry.entry_id, review_id)
            return _LookupState(
                self._empty_result(
                    key,
                    status=RepositoryCacheStatus.MISS,
                    fallback=fallback,
                    corruption_reason=error.code.value,
                )
            )

        try:
            _validate_artifact_content(content, validator, cache_read=True)
        except RepositoryCacheValidationError:
            if pin_added and review_id is not None:
                self._best_effort_unpin(entry.entry_id, review_id)
            return _LookupState(
                self._empty_result(
                    key,
                    status=RepositoryCacheStatus.REBUILD,
                    fallback=fallback,
                    corruption_reason="corruption",
                ),
                invalid_entry=entry,
            )
        return _LookupState(
            RepositoryCacheResult(
                content=content,
                entry=projected,
                provenance=self._provenance(
                    key,
                    status=RepositoryCacheStatus.HIT,
                    entry=projected,
                    blob_hash=projected.blob_hash,
                    persistent=True,
                    session_pinned=pinned,
                    fallback=fallback,
                ),
            )
        )

    def _persist_or_session(
        self,
        key: RepositoryKnowledgeKey,
        artifact: RepositoryKnowledgeArtifact,
        *,
        status: RepositoryCacheStatus,
        review_id: Optional[str],
        fallback: Tuple[Tuple[str, str], ...],
        corruption_reason: Optional[str],
        invalid_entry: Optional[RepositoryKnowledgeEntry],
        created_at: Optional[str],
        conflict_is_error: bool,
    ) -> RepositoryCacheResult:
        if self.mode is not MemoryMode.READ_WRITE or self._store is None:
            return self._session_result(
                key,
                artifact,
                status=status,
                fallback=fallback,
                corruption_reason=corruption_reason,
            )

        timestamp = self._timestamp(created_at)
        if invalid_entry is not None:
            try:
                generation = self._store.get_generations(
                    key.repository_key
                ).knowledge_generation
                self._store.delete_knowledge_entry(
                    invalid_entry.entry_id,
                    request_id=stable_request_id(
                        "repository_cache_rebuild",
                        key.key_hash,
                        invalid_entry.entry_id,
                        artifact.blob_hash,
                        generation,
                    ),
                    expected_generation=generation,
                    created_at=timestamp,
                )
            except MemoryStoreConflictError:
                # A corrupt manifest pinned by an existing Session remains an
                # audit object.  The rebuilt bytes are still safe Session data.
                return self._session_result(
                    key,
                    artifact,
                    status=RepositoryCacheStatus.REBUILD,
                    fallback=fallback,
                    corruption_reason="pinned_corrupt_entry",
                )
            except MemoryStoreError as error:
                return self._session_result(
                    key,
                    artifact,
                    status=RepositoryCacheStatus.REBUILD,
                    fallback=fallback,
                    corruption_reason=error.code.value,
                )

        try:
            blob = self._put_blob_with_repair(artifact, status)
            entry = RepositoryKnowledgeEntry(
                key=key,
                blob_hash=blob.blob_hash,
                size_bytes=blob.size_bytes,
                content_type=artifact.content_type,
                artifact_schema=artifact.artifact_schema,
                summary_hash=artifact.summary_hash,
                created_at=timestamp,
                pinned_by_review_ids=(review_id,) if review_id is not None else (),
            )
            generation = self._store.get_generations(
                key.repository_key
            ).knowledge_generation
            self._store.put_knowledge_entry(
                entry,
                request_id=stable_request_id(
                    "repository_cache_write", entry.to_dict(), generation
                ),
                expected_generation=generation,
            )
            projected = self._store.get_knowledge_entry(entry.entry_id)
            content = self._store.read_blob(projected.blob_hash)
            validate_repository_knowledge_manifest(
                projected,
                key,
                content,
                content_type=artifact.content_type,
                artifact_schema=artifact.artifact_schema,
            )
            if content != artifact.content:
                raise RepositoryCacheConflictError(
                    "exact cache key resolved to different immutable content"
                )
            pinned = review_id is not None and review_id in projected.pinned_by_review_ids
            return RepositoryCacheResult(
                content=content,
                entry=projected,
                provenance=self._provenance(
                    key,
                    status=status,
                    entry=projected,
                    blob_hash=projected.blob_hash,
                    persistent=True,
                    session_pinned=pinned,
                    fallback=fallback,
                    corruption_reason=corruption_reason,
                ),
            )
        except RepositoryCacheConflictError:
            raise
        except MemoryStoreConflictError as error:
            concurrent = self._lookup_state(
                key,
                review_id=review_id,
                validator=None,
                fallback=fallback,
                content_type=artifact.content_type,
                artifact_schema=artifact.artifact_schema,
            ).result
            if (
                concurrent.content is not None
                and concurrent.entry is not None
                and _entry_matches_artifact(concurrent.entry, artifact)
            ):
                return concurrent
            if conflict_is_error:
                raise RepositoryCacheConflictError(
                    "exact cache key already has different immutable content"
                ) from error
            return self._session_result(
                key,
                artifact,
                status=RepositoryCacheStatus.REBUILD,
                fallback=fallback,
                corruption_reason=error.code.value,
            )
        except MemoryStoreError as error:
            return self._session_result(
                key,
                artifact,
                status=status,
                fallback=fallback,
                corruption_reason=error.code.value,
            )
        except RepositoryCacheError:
            return self._session_result(
                key,
                artifact,
                status=status,
                fallback=fallback,
                corruption_reason="unavailable",
            )
        except ValueError as error:
            raise RepositoryCacheValidationError(str(error)) from error

    def _put_blob_with_repair(
        self,
        artifact: RepositoryKnowledgeArtifact,
        status: RepositoryCacheStatus,
    ):
        try:
            return self._store.put_blob(
                artifact.content,
                media_type=artifact.content_type,
                expected_hash=artifact.blob_hash,
                expected_size=len(artifact.content),
            )
        except MemoryStoreCorruptionError:
            if status is not RepositoryCacheStatus.REBUILD:
                raise
            # Repair stays inside MemoryStore so promotion/read/GC remain
            # serialized across processes.  Cache code never unlinks Store
            # paths directly.
            info = self._store.get_blob_info(artifact.blob_hash, validate=False)
            if (
                info.size_bytes != len(artifact.content)
                or info.media_type != artifact.content_type
            ):
                raise
            return self._store.repair_blob(
                artifact.content,
                media_type=artifact.content_type,
                expected_hash=artifact.blob_hash,
                expected_size=len(artifact.content),
            )

    def _normalize_artifact(
        self,
        key: RepositoryKnowledgeKey,
        value: Union[RepositoryKnowledgeArtifact, bytes, bytearray, memoryview],
        *,
        content_type: Optional[str],
        artifact_schema: Optional[str],
    ) -> RepositoryKnowledgeArtifact:
        _require_key(key)
        if isinstance(value, RepositoryKnowledgeArtifact):
            if content_type is not None and value.content_type != content_type.casefold():
                raise RepositoryCacheValidationError(
                    "artifact and requested content_type disagree"
                )
            if (
                artifact_schema is not None
                and value.artifact_schema != artifact_schema.casefold()
            ):
                raise RepositoryCacheValidationError(
                    "artifact and requested artifact_schema disagree"
                )
            return value
        if not isinstance(value, (bytes, bytearray, memoryview)):
            raise RepositoryCacheValidationError(
                "cache builder must return bytes or RepositoryKnowledgeArtifact"
            )
        metadata = CAPABILITY_METADATA[key.capability]
        return RepositoryKnowledgeArtifact(
            content=bytes(value),
            content_type=content_type or metadata.content_type,
            artifact_schema=artifact_schema or metadata.artifact_schema,
        )

    def _best_effort_unpin(self, entry_id: str, review_id: str) -> None:
        try:
            self._store.unpin_knowledge_entry(entry_id, review_id)
        except MemoryStoreError:
            pass

    def _timestamp(self, value: Optional[str]) -> str:
        timestamp = self._clock() if value is None else value
        if not isinstance(timestamp, str):
            raise RepositoryCacheValidationError("cache clock must return a timestamp")
        try:
            datetime.fromisoformat(timestamp[:-1] + "+00:00")
        except (ValueError, TypeError):
            raise RepositoryCacheValidationError(
                "cache timestamp must be RFC 3339 UTC ending in Z"
            ) from None
        if not timestamp.endswith("Z"):
            raise RepositoryCacheValidationError(
                "cache timestamp must be RFC 3339 UTC ending in Z"
            )
        return timestamp

    def _empty_result(
        self,
        key: RepositoryKnowledgeKey,
        *,
        status: RepositoryCacheStatus,
        fallback: Tuple[Tuple[str, str], ...],
        corruption_reason: Optional[str] = None,
    ) -> RepositoryCacheResult:
        return RepositoryCacheResult(
            content=None,
            entry=None,
            provenance=self._provenance(
                key,
                status=status,
                entry=None,
                blob_hash=None,
                persistent=False,
                session_pinned=False,
                fallback=fallback,
                corruption_reason=corruption_reason,
            ),
        )

    def _session_result(
        self,
        key: RepositoryKnowledgeKey,
        artifact: RepositoryKnowledgeArtifact,
        *,
        status: RepositoryCacheStatus,
        fallback: Tuple[Tuple[str, str], ...],
        corruption_reason: Optional[str],
    ) -> RepositoryCacheResult:
        return RepositoryCacheResult(
            content=artifact.content,
            entry=None,
            provenance=self._provenance(
                key,
                status=status,
                entry=None,
                blob_hash=artifact.blob_hash,
                persistent=False,
                session_pinned=False,
                fallback=fallback,
                corruption_reason=corruption_reason,
            ),
        )

    @staticmethod
    def _provenance(
        key: RepositoryKnowledgeKey,
        *,
        status: RepositoryCacheStatus,
        entry: Optional[RepositoryKnowledgeEntry],
        blob_hash: Optional[str],
        persistent: bool,
        session_pinned: bool,
        fallback: Tuple[Tuple[str, str], ...],
        corruption_reason: Optional[str] = None,
    ) -> RepositoryCacheProvenance:
        return RepositoryCacheProvenance(
            status=status,
            key=key,
            entry_id=None if entry is None else entry.entry_id,
            blob_hash=blob_hash,
            persistent=persistent,
            session_pinned=session_pinned,
            fallback=fallback,
            corruption_reason=corruption_reason,
        )


def _entry_matches_artifact(
    entry: Optional[RepositoryKnowledgeEntry],
    artifact: RepositoryKnowledgeArtifact,
) -> bool:
    return bool(
        entry is not None
        and entry.blob_hash == artifact.blob_hash
        and entry.size_bytes == len(artifact.content)
        and entry.content_type == artifact.content_type
        and entry.artifact_schema == artifact.artifact_schema
        and entry.summary_hash == artifact.summary_hash
    )


def _validate_artifact_content(
    content: bytes,
    validator: Optional[ArtifactValidator],
    *,
    cache_read: bool,
) -> None:
    if validator is None:
        return
    if not callable(validator):
        raise RepositoryCacheValidationError("cache validator must be callable")
    try:
        result = validator(content)
    except RepositoryCacheValidationError:
        raise
    except Exception as error:
        message = (
            "cached repository artifact failed validation"
            if cache_read
            else "built repository artifact failed validation"
        )
        raise RepositoryCacheValidationError(message) from error
    if result is False:
        message = (
            "cached repository artifact failed validation"
            if cache_read
            else "built repository artifact failed validation"
        )
        raise RepositoryCacheValidationError(message)


def _require_key(key: RepositoryKnowledgeKey) -> None:
    if not isinstance(key, RepositoryKnowledgeKey):
        raise RepositoryCacheValidationError(
            "repository cache key must be a canonical RepositoryKnowledgeKey"
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


# Concise aliases for callers that already use cache terminology.
RepositoryKnowledgeManifest = RepositoryKnowledgeEntry
repository_configuration_digest = digest_repository_configuration
repository_input_digest = digest_repository_input


__all__ = [
    "CAPABILITY_METADATA",
    "RepositoryCacheConflictError",
    "RepositoryCacheError",
    "RepositoryCacheGCResult",
    "RepositoryCacheModeError",
    "RepositoryCacheProvenance",
    "RepositoryCacheResult",
    "RepositoryCacheStatus",
    "RepositoryCacheValidationError",
    "RepositoryCapabilityMetadata",
    "RepositoryKnowledgeArtifact",
    "RepositoryKnowledgeCache",
    "RepositoryKnowledgeManifest",
    "build_repository_knowledge_key",
    "digest_repository_configuration",
    "digest_repository_input",
    "repository_configuration_digest",
    "repository_input_digest",
    "validate_repository_knowledge_manifest",
]
