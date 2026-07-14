"""Transactional durable Memory storage.

SQLite is the authority for metadata, projections, generations, and the audit
chain.  Large immutable payloads live in a SHA-256 addressed blob tree beside
the database.  This module deliberately sits below Pipeline/CLI/provider code;
its only project-level inputs are canonical Memory models and repository
identity descriptors.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import stat
import time
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)
import uuid

from review_agent.memory_identity import (
    RepositoryIdentityDescriptor,
    RepositoryMemoryNamespace,
)
from review_agent.memory_models import (
    CandidateStatus,
    DurableMemoryRecord,
    FeedbackRecord,
    FeedbackStatus,
    GenerationMetadata,
    MemoryCandidate,
    RecordStatus,
    RepositoryKnowledgeEntry,
    RepositoryKnowledgeKey,
    Sensitivity,
    SourceBundleDescriptor,
    canonical_json,
    canonical_sha256,
    stable_event_id,
    stable_request_id,
    validate_stable_id,
)


STORE_SCHEMA_NAME = "memory_store_schema_v1"
STORE_SCHEMA_VERSION = 1
EVENT_SCHEMA_VERSION = 1
EXPORT_SCHEMA_NAME = "memory_store_export_v1"
EXPORT_SCHEMA_VERSION = 1
DEFAULT_BUSY_TIMEOUT_MS = 5_000
MAX_BUSY_TIMEOUT_MS = 120_000
MAX_BLOB_SIZE_BYTES = 512 * 1024 * 1024
ZERO_EVENT_HASH = "0" * 64

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REPOSITORY_KEY_PATTERN = _SHA256_PATTERN
_UTC_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$"
)
_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+/@-]{0,511}$")
_MEDIA_TYPE_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*/[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*$"
)
_SUBJECT_TYPES = frozenset(
    {"candidate", "record", "feedback", "knowledge", "source_bundle"}
)
_PIN_TYPES = frozenset({"session", "source_bundle", "manual", "knowledge"})


_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE metadata (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE schema_migrations (
        schema_version INTEGER PRIMARY KEY,
        schema_name TEXT NOT NULL,
        definition_hash TEXT NOT NULL,
        applied_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE repositories (
        repository_key TEXT PRIMARY KEY,
        identity_schema TEXT,
        canonical_path TEXT,
        git_common_dir TEXT,
        origin_url TEXT,
        identity_json TEXT,
        created_at TEXT NOT NULL,
        last_accessed_at TEXT NOT NULL,
        CHECK (
            length(repository_key) = 64
            AND repository_key NOT GLOB '*[^0-9a-f]*'
        )
    )
    """,
    """
    CREATE TABLE generations (
        repository_key TEXT PRIMARY KEY
            REFERENCES repositories(repository_key) ON DELETE CASCADE,
        memory_generation INTEGER NOT NULL DEFAULT 0 CHECK (memory_generation >= 0),
        feedback_generation INTEGER NOT NULL DEFAULT 0 CHECK (feedback_generation >= 0),
        knowledge_generation INTEGER NOT NULL DEFAULT 0 CHECK (knowledge_generation >= 0)
    )
    """,
    """
    CREATE TABLE blobs (
        blob_hash TEXT PRIMARY KEY,
        size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
        media_type TEXT NOT NULL,
        created_at TEXT NOT NULL,
        CHECK (length(blob_hash) = 64)
    )
    """,
    """
    CREATE TABLE blob_pins (
        blob_hash TEXT NOT NULL REFERENCES blobs(blob_hash) ON DELETE CASCADE,
        pin_type TEXT NOT NULL,
        pin_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (blob_hash, pin_type, pin_id)
    )
    """,
    """
    CREATE TABLE candidates (
        candidate_id TEXT PRIMARY KEY,
        repository_key TEXT NOT NULL
            REFERENCES repositories(repository_key) ON DELETE CASCADE,
        content_fingerprint TEXT NOT NULL,
        model_json TEXT NOT NULL,
        body_hash TEXT NOT NULL,
        current_status TEXT NOT NULL,
        generation INTEGER NOT NULL CHECK (generation > 0),
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE source_bundles (
        bundle_hash TEXT PRIMARY KEY,
        repository_key TEXT NOT NULL
            REFERENCES repositories(repository_key) ON DELETE CASCADE,
        candidate_id TEXT NOT NULL
            REFERENCES candidates(candidate_id) ON DELETE RESTRICT,
        blob_hash TEXT NOT NULL REFERENCES blobs(blob_hash) ON DELETE RESTRICT,
        model_json TEXT NOT NULL,
        body_hash TEXT NOT NULL,
        generation INTEGER NOT NULL CHECK (generation > 0),
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE records (
        memory_id TEXT PRIMARY KEY,
        candidate_id TEXT NOT NULL UNIQUE
            REFERENCES candidates(candidate_id) ON DELETE RESTRICT,
        repository_key TEXT NOT NULL
            REFERENCES repositories(repository_key) ON DELETE CASCADE,
        source_bundle_hash TEXT NOT NULL
            REFERENCES source_bundles(bundle_hash) ON DELETE RESTRICT,
        model_json TEXT NOT NULL,
        body_hash TEXT NOT NULL,
        current_status TEXT NOT NULL,
        generation INTEGER NOT NULL CHECK (generation > 0),
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE feedback (
        feedback_id TEXT PRIMARY KEY,
        repository_key TEXT NOT NULL
            REFERENCES repositories(repository_key) ON DELETE CASCADE,
        review_id TEXT NOT NULL,
        finding_id TEXT NOT NULL,
        model_json TEXT NOT NULL,
        body_hash TEXT NOT NULL,
        current_status TEXT NOT NULL,
        generation INTEGER NOT NULL CHECK (generation > 0),
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE knowledge_entries (
        entry_id TEXT PRIMARY KEY,
        repository_key TEXT NOT NULL
            REFERENCES repositories(repository_key) ON DELETE CASCADE,
        key_hash TEXT NOT NULL UNIQUE,
        blob_hash TEXT NOT NULL REFERENCES blobs(blob_hash) ON DELETE RESTRICT,
        model_json TEXT NOT NULL,
        body_hash TEXT NOT NULL,
        generation INTEGER NOT NULL CHECK (generation > 0),
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE events (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id TEXT NOT NULL UNIQUE,
        schema_version INTEGER NOT NULL,
        repository_key TEXT NOT NULL
            REFERENCES repositories(repository_key) ON DELETE CASCADE,
        subject_type TEXT NOT NULL,
        subject_id TEXT NOT NULL,
        action TEXT NOT NULL,
        actor_type TEXT NOT NULL,
        actor_id TEXT NOT NULL,
        reason_code TEXT NOT NULL,
        reason TEXT,
        previous_status TEXT,
        new_status TEXT,
        request_id TEXT NOT NULL UNIQUE,
        created_at TEXT NOT NULL,
        previous_hash TEXT NOT NULL,
        current_hash TEXT NOT NULL,
        generation_kind TEXT NOT NULL,
        generation INTEGER NOT NULL CHECK (generation >= 0),
        CHECK (schema_version = 1),
        CHECK (length(previous_hash) = 64),
        CHECK (length(current_hash) = 64)
    )
    """,
    """
    CREATE TABLE event_chain_heads (
        repository_key TEXT PRIMARY KEY
            REFERENCES repositories(repository_key) ON DELETE CASCADE,
        event_count INTEGER NOT NULL DEFAULT 0 CHECK (event_count >= 0),
        head_sequence INTEGER,
        head_hash TEXT NOT NULL,
        CHECK (length(head_hash) = 64)
    )
    """,
    """
    CREATE TABLE outbox_receipts (
        request_id TEXT PRIMARY KEY,
        repository_key TEXT NOT NULL
            REFERENCES repositories(repository_key) ON DELETE CASCADE,
        operation TEXT NOT NULL,
        request_hash TEXT NOT NULL,
        subject_id TEXT NOT NULL,
        event_id TEXT REFERENCES events(event_id) ON DELETE RESTRICT,
        result_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        CHECK (length(request_hash) = 64)
    )
    """,
    "CREATE INDEX idx_candidates_repository_status ON candidates(repository_key, current_status)",
    "CREATE INDEX idx_candidates_fingerprint ON candidates(repository_key, content_fingerprint)",
    "CREATE INDEX idx_records_repository_status ON records(repository_key, current_status)",
    "CREATE INDEX idx_feedback_repository_status ON feedback(repository_key, current_status)",
    "CREATE INDEX idx_feedback_review_finding ON feedback(repository_key, review_id, finding_id)",
    "CREATE INDEX idx_knowledge_repository_key ON knowledge_entries(repository_key, key_hash)",
    "CREATE INDEX idx_events_repository_sequence ON events(repository_key, sequence)",
    "CREATE INDEX idx_events_subject ON events(repository_key, subject_type, subject_id, sequence)",
    "CREATE INDEX idx_blob_pins_kind ON blob_pins(pin_type, pin_id)",
    """
    CREATE TRIGGER events_no_update
    BEFORE UPDATE ON events
    BEGIN
        SELECT RAISE(ABORT, 'events are append-only');
    END
    """,
    """
    CREATE TRIGGER events_no_delete
    BEFORE DELETE ON events
    BEGIN
        SELECT RAISE(ABORT, 'events are append-only');
    END
    """,
    """
    CREATE TRIGGER candidates_body_immutable
    BEFORE UPDATE OF candidate_id, repository_key, content_fingerprint,
                     model_json, body_hash, created_at ON candidates
    BEGIN
        SELECT RAISE(ABORT, 'candidate bodies are immutable');
    END
    """,
    """
    CREATE TRIGGER records_body_immutable
    BEFORE UPDATE OF memory_id, candidate_id, repository_key, source_bundle_hash,
                     model_json, body_hash, created_at ON records
    BEGIN
        SELECT RAISE(ABORT, 'record bodies are immutable');
    END
    """,
    """
    CREATE TRIGGER feedback_body_immutable
    BEFORE UPDATE OF feedback_id, repository_key, review_id, finding_id,
                     model_json, body_hash, created_at ON feedback
    BEGIN
        SELECT RAISE(ABORT, 'feedback bodies are immutable');
    END
    """,
    """
    CREATE TRIGGER knowledge_body_immutable
    BEFORE UPDATE ON knowledge_entries
    BEGIN
        SELECT RAISE(ABORT, 'knowledge entries are immutable');
    END
    """,
    """
    CREATE TRIGGER source_bundles_immutable
    BEFORE UPDATE ON source_bundles
    BEGIN
        SELECT RAISE(ABORT, 'source bundles are immutable');
    END
    """,
    """
    CREATE TRIGGER blobs_immutable
    BEFORE UPDATE ON blobs
    BEGIN
        SELECT RAISE(ABORT, 'blobs are immutable');
    END
    """,
)

SCHEMA_DEFINITION_HASH = hashlib.sha256(
    "\n".join(statement.strip() for statement in _SCHEMA_STATEMENTS).encode("utf-8")
).hexdigest()

_REQUIRED_TABLES = frozenset(
    {
        "metadata",
        "schema_migrations",
        "repositories",
        "generations",
        "blobs",
        "blob_pins",
        "candidates",
        "source_bundles",
        "records",
        "feedback",
        "knowledge_entries",
        "events",
        "event_chain_heads",
        "outbox_receipts",
    }
)
_REQUIRED_TRIGGERS = frozenset(
    {
        "events_no_update",
        "events_no_delete",
        "candidates_body_immutable",
        "records_body_immutable",
        "feedback_body_immutable",
        "knowledge_body_immutable",
        "source_bundles_immutable",
        "blobs_immutable",
    }
)


class MemoryStoreErrorCode(str, Enum):
    UNAVAILABLE = "unavailable"
    UNSUPPORTED_SCHEMA = "unsupported_schema"
    CORRUPTION = "corruption"
    CONFLICT = "conflict"
    BUSY = "busy"
    VALIDATION = "validation"
    NOT_FOUND = "not_found"
    READ_ONLY = "read_only"
    MIGRATION = "migration"


class MemoryStoreError(RuntimeError):
    """Base class carrying a stable, non-sensitive error category."""

    def __init__(self, message: str, code: MemoryStoreErrorCode) -> None:
        super().__init__(message)
        self.code = code


class MemoryStoreUnavailableError(MemoryStoreError):
    def __init__(self, message: str = "memory store is unavailable") -> None:
        super().__init__(message, MemoryStoreErrorCode.UNAVAILABLE)


class MemoryStoreSchemaError(MemoryStoreError):
    def __init__(self, message: str = "memory store schema is unsupported") -> None:
        super().__init__(message, MemoryStoreErrorCode.UNSUPPORTED_SCHEMA)


class MemoryStoreCorruptionError(MemoryStoreError):
    def __init__(self, message: str = "memory store integrity validation failed") -> None:
        super().__init__(message, MemoryStoreErrorCode.CORRUPTION)


class MemoryStoreConflictError(MemoryStoreError):
    def __init__(self, message: str = "memory store compare-and-swap conflict") -> None:
        super().__init__(message, MemoryStoreErrorCode.CONFLICT)


class MemoryStoreBusyError(MemoryStoreError):
    def __init__(self, message: str = "memory store is busy") -> None:
        super().__init__(message, MemoryStoreErrorCode.BUSY)


class MemoryStoreValidationError(MemoryStoreError):
    def __init__(self, message: str = "memory store input validation failed") -> None:
        super().__init__(message, MemoryStoreErrorCode.VALIDATION)


class MemoryStoreNotFoundError(MemoryStoreError):
    def __init__(self, message: str = "memory store subject was not found") -> None:
        super().__init__(message, MemoryStoreErrorCode.NOT_FOUND)


class MemoryStoreReadOnlyError(MemoryStoreError):
    def __init__(self, message: str = "memory store is read-only") -> None:
        super().__init__(message, MemoryStoreErrorCode.READ_ONLY)


class MemoryStoreMigrationError(MemoryStoreError):
    def __init__(self, message: str = "staged memory store migration failed") -> None:
        super().__init__(message, MemoryStoreErrorCode.MIGRATION)


@dataclass(frozen=True)
class BlobInfo:
    blob_hash: str
    size_bytes: int
    media_type: str
    created_at: str
    path: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "blob_hash": self.blob_hash,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class WriteResult:
    operation: str
    subject_id: str
    event_id: Optional[str]
    generations: GenerationMetadata
    applied: bool
    replayed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "subject_id": self.subject_id,
            "event_id": self.event_id,
            "generations": self.generations.to_dict(),
            "applied": self.applied,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WriteResult":
        expected = {
            "operation",
            "subject_id",
            "event_id",
            "generations",
            "applied",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise MemoryStoreCorruptionError("memory request receipt is invalid")
        return cls(
            operation=_required_token(payload["operation"], "operation"),
            subject_id=_required_text(payload["subject_id"], "subject_id", 512),
            event_id=(
                None
                if payload["event_id"] is None
                else _event_id(payload["event_id"])
            ),
            generations=GenerationMetadata.from_dict(payload["generations"]),
            applied=_required_bool(payload["applied"], "applied"),
        )


@dataclass(frozen=True)
class MemoryEvent:
    sequence: int
    event_id: str
    repository_key: str
    subject_type: str
    subject_id: str
    action: str
    actor_type: str
    actor_id: str
    reason_code: str
    reason: Optional[str]
    previous_status: Optional[str]
    new_status: Optional[str]
    request_id: str
    created_at: str
    previous_hash: str
    current_hash: str
    generation_kind: str
    generation: int
    schema_version: int = EVENT_SCHEMA_VERSION

    def hash_payload(self) -> Dict[str, Any]:
        return {
            "sequence": self.sequence,
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "repository_key": self.repository_key,
            "subject_type": self.subject_type,
            "subject_id": self.subject_id,
            "action": self.action,
            "actor_type": self.actor_type,
            "actor_id": self.actor_id,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "previous_status": self.previous_status,
            "new_status": self.new_status,
            "request_id": self.request_id,
            "created_at": self.created_at,
            "previous_hash": self.previous_hash,
            "generation_kind": self.generation_kind,
            "generation": self.generation,
        }

    def to_dict(self) -> Dict[str, Any]:
        payload = self.hash_payload()
        payload["current_hash"] = self.current_hash
        return payload


@dataclass(frozen=True)
class BlobGCResult:
    candidate_hashes: Tuple[str, ...]
    deleted_hashes: Tuple[str, ...]
    orphan_paths: Tuple[str, ...]
    deleted_orphan_paths: Tuple[str, ...]
    reclaimed_bytes: int
    dry_run: bool


@dataclass(frozen=True)
class MemoryStoreReadView:
    generations: GenerationMetadata
    records: Tuple[DurableMemoryRecord, ...]
    feedback: Tuple[FeedbackRecord, ...]
    knowledge_entries: Tuple[RepositoryKnowledgeEntry, ...]


@dataclass(frozen=True)
class ImportPlan:
    repository_keys: Tuple[str, ...]
    candidate_count: int
    record_count: int
    feedback_count: int
    knowledge_count: int
    source_bundle_count: int
    event_count: int
    blob_count: int
    outbox_receipt_count: int
    redacted: bool
    restorable: bool
    applied: bool = False


@dataclass(frozen=True)
class IntegrityReport:
    repository_count: int
    event_count: int
    blob_count: int
    candidate_count: int
    record_count: int
    feedback_count: int
    knowledge_count: int


PathInput = Union[str, os.PathLike]
NamespaceInput = Union[PathInput, RepositoryMemoryNamespace]


class MemoryStore:
    """A repository-scoped SQLite/event/blob store.

    Passing a directory uses ``<directory>/memory.sqlite3``.  Passing an
    existing file, or a path ending in ``.sqlite3``, addresses that database
    directly.  A :class:`RepositoryMemoryNamespace` additionally registers its
    sanitized identity descriptor.
    """

    def __init__(
        self,
        namespace: NamespaceInput,
        *,
        busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
        read_only: bool = False,
    ) -> None:
        if type(busy_timeout_ms) is not int or not (
            0 <= busy_timeout_ms <= MAX_BUSY_TIMEOUT_MS
        ):
            raise MemoryStoreValidationError("busy timeout is outside the supported range")
        if type(read_only) is not bool:
            raise MemoryStoreValidationError("read_only must be a boolean")

        descriptor: Optional[RepositoryIdentityDescriptor] = None
        if isinstance(namespace, RepositoryMemoryNamespace):
            raw_path = Path(namespace.namespace_path)
            descriptor = namespace.metadata
        else:
            raw_path = Path(namespace)

        try:
            resolved = raw_path.resolve(strict=False)
        except OSError:
            raise MemoryStoreUnavailableError() from None
        if resolved.exists() and resolved.is_file():
            database_path = resolved
            namespace_path = resolved.parent
        elif resolved.name.endswith(".sqlite3"):
            database_path = resolved
            namespace_path = resolved.parent
        else:
            namespace_path = resolved
            database_path = namespace_path / "memory.sqlite3"

        self.namespace_path = namespace_path
        self.database_path = database_path
        self.blob_root = namespace_path / "blobs" / "sha256"
        self._blob_temp_root = namespace_path / "blobs" / ".tmp"
        self._blob_lock_path = namespace_path / "blobs" / ".blob-store.lock"
        self._busy_timeout_ms = busy_timeout_ms
        self._read_only = read_only

        if not read_only:
            try:
                self.namespace_path.mkdir(parents=True, exist_ok=True)
            except OSError:
                raise MemoryStoreUnavailableError() from None
        self._initialize_or_validate()
        if not read_only:
            try:
                self._initialize_blob_layout()
            except OSError:
                raise MemoryStoreUnavailableError() from None
        if descriptor is not None:
            if read_only:
                self._require_registered_repository(descriptor.repository_key)
            else:
                self.register_repository(descriptor)

    @property
    def busy_timeout_ms(self) -> int:
        return self._busy_timeout_ms

    @property
    def read_only(self) -> bool:
        return self._read_only

    def close(self) -> None:
        """Compatibility no-op; operations use short-lived connections."""

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _connect(self, *, read_only: bool) -> sqlite3.Connection:
        if not read_only and self._read_only:
            raise MemoryStoreReadOnlyError()
        try:
            if read_only:
                uri_path = self.database_path.as_posix().replace("?", "%3f").replace(
                    "#", "%23"
                )
                connection = sqlite3.connect(
                    "file:%s?mode=ro" % uri_path,
                    uri=True,
                    timeout=self._busy_timeout_ms / 1000.0,
                    isolation_level=None,
                )
            else:
                connection = sqlite3.connect(
                    str(self.database_path),
                    timeout=self._busy_timeout_ms / 1000.0,
                    isolation_level=None,
                )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = %d" % self._busy_timeout_ms)
            if read_only:
                connection.execute("PRAGMA query_only = ON")
            return connection
        except sqlite3.Error as error:
            raise _translate_sqlite_error(error) from None

    @contextmanager
    def open_connection(
        self,
        *,
        read_only: bool = True,
    ) -> Iterator[sqlite3.Connection]:
        """Open a read-only diagnostic connection and always close it.

        Authority writes must go through Store transactions so events,
        generations, receipts, and projections cannot diverge.
        """

        if read_only is not True:
            raise MemoryStoreReadOnlyError(
                "raw writable connections are not part of the MemoryStore API"
            )
        connection = self._connect(read_only=True)
        try:
            yield connection
        except MemoryStoreError:
            if connection.in_transaction:
                connection.rollback()
            raise
        except sqlite3.Error as error:
            if connection.in_transaction:
                connection.rollback()
            raise _translate_sqlite_error(error) from None
        finally:
            connection.close()

    @contextmanager
    def _maintenance_connection(self) -> Iterator[sqlite3.Connection]:
        """Internal/test-only connection for checkpoints and fault injection."""

        connection = self._connect(read_only=False)
        try:
            yield connection
            if connection.in_transaction:
                connection.commit()
        except MemoryStoreError:
            if connection.in_transaction:
                connection.rollback()
            raise
        except sqlite3.Error as error:
            if connection.in_transaction:
                connection.rollback()
            raise _translate_sqlite_error(error) from None
        finally:
            connection.close()

    @contextmanager
    def _reader(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect(read_only=True)
        try:
            yield connection
        except MemoryStoreError:
            raise
        except sqlite3.Error as error:
            raise _translate_sqlite_error(error) from None
        finally:
            connection.close()

    @contextmanager
    def _authority(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect(read_only=False)
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except MemoryStoreError:
            if connection.in_transaction:
                connection.rollback()
            raise
        except sqlite3.Error as error:
            if connection.in_transaction:
                connection.rollback()
            raise _translate_sqlite_error(error) from None
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize_or_validate(self) -> None:
        if self._read_only and not self.database_path.is_file():
            raise MemoryStoreUnavailableError()
        connection = self._connect(read_only=self._read_only)
        try:
            tables = _table_names(connection)
            if not tables:
                if self._read_only:
                    raise MemoryStoreSchemaError()
                mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
                if str(mode).casefold() != "wal":
                    raise MemoryStoreUnavailableError(
                        "memory store could not enable write-ahead logging"
                    )
                connection.execute("BEGIN IMMEDIATE")
                try:
                    # Another initializer may have completed while this process waited.
                    if _table_names(connection):
                        _validate_schema_connection(connection)
                    else:
                        created_at = _utc_now()
                        for statement in _SCHEMA_STATEMENTS:
                            connection.execute(statement)
                        metadata = (
                            ("schema_name", STORE_SCHEMA_NAME),
                            ("schema_version", str(STORE_SCHEMA_VERSION)),
                            ("schema_definition_hash", SCHEMA_DEFINITION_HASH),
                            ("schema_created_at", created_at),
                        )
                        connection.executemany(
                            "INSERT INTO metadata(key, value) VALUES (?, ?)",
                            metadata,
                        )
                        connection.execute(
                            """
                            INSERT INTO schema_migrations(
                                schema_version, schema_name, definition_hash, applied_at
                            ) VALUES (?, ?, ?, ?)
                            """,
                            (
                                STORE_SCHEMA_VERSION,
                                STORE_SCHEMA_NAME,
                                SCHEMA_DEFINITION_HASH,
                                created_at,
                            ),
                        )
                        connection.execute(
                            "PRAGMA user_version = %d" % STORE_SCHEMA_VERSION
                        )
                    connection.commit()
                except Exception:
                    if connection.in_transaction:
                        connection.rollback()
                    raise
            else:
                # Validate before any mutating PRAGMA so unknown schemas fail closed.
                _validate_schema_connection(connection)
                if not self._read_only:
                    mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
                    if str(mode).casefold() != "wal":
                        raise MemoryStoreUnavailableError(
                            "memory store could not enable write-ahead logging"
                        )

            _validate_schema_connection(connection)
            journal_mode = str(
                connection.execute("PRAGMA journal_mode").fetchone()[0]
            ).casefold()
            if journal_mode != "wal":
                raise MemoryStoreCorruptionError(
                    "memory store write-ahead logging is not enabled"
                )
            if connection.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
                raise MemoryStoreUnavailableError(
                    "memory store could not enable foreign-key validation"
                )
        except MemoryStoreError:
            raise
        except sqlite3.Error as error:
            raise _translate_sqlite_error(error) from None
        finally:
            connection.close()

    def metadata(self) -> Dict[str, str]:
        with self._reader() as connection:
            return {
                str(row["key"]): str(row["value"])
                for row in connection.execute(
                    "SELECT key, value FROM metadata ORDER BY key"
                )
            }

    def register_repository(
        self,
        descriptor: RepositoryIdentityDescriptor,
        *,
        accessed_at: Optional[str] = None,
    ) -> None:
        if not isinstance(descriptor, RepositoryIdentityDescriptor):
            raise MemoryStoreValidationError(
                "repository metadata must be a canonical identity descriptor"
            )
        now = _timestamp(accessed_at or _utc_now(), "accessed_at")
        identity_json = canonical_json(descriptor.to_payload())
        with self._authority() as connection:
            existing = connection.execute(
                "SELECT identity_json FROM repositories WHERE repository_key = ?",
                (descriptor.repository_key,),
            ).fetchone()
            if existing is not None and existing["identity_json"] not in (
                None,
                identity_json,
            ):
                raise MemoryStoreConflictError(
                    "repository identity does not match the registered namespace"
                )
            connection.execute(
                """
                INSERT INTO repositories(
                    repository_key, identity_schema, canonical_path, git_common_dir,
                    origin_url, identity_json, created_at, last_accessed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(repository_key) DO UPDATE SET
                    identity_schema = excluded.identity_schema,
                    canonical_path = excluded.canonical_path,
                    git_common_dir = excluded.git_common_dir,
                    origin_url = excluded.origin_url,
                    identity_json = excluded.identity_json,
                    last_accessed_at = excluded.last_accessed_at
                """,
                (
                    descriptor.repository_key,
                    descriptor.schema,
                    descriptor.canonical_path,
                    descriptor.git_common_dir,
                    descriptor.origin_url,
                    identity_json,
                    now,
                    now,
                ),
            )
            self._ensure_repository_rows(connection, descriptor.repository_key, now)

    def _require_registered_repository(self, repository_key: str) -> None:
        key = _repository_key(repository_key)
        with self._reader() as connection:
            row = connection.execute(
                "SELECT 1 FROM repositories WHERE repository_key = ?", (key,)
            ).fetchone()
        if row is None:
            raise MemoryStoreNotFoundError("repository is not registered in this store")

    @staticmethod
    def _ensure_repository_rows(
        connection: sqlite3.Connection,
        repository_key: str,
        now: Optional[str] = None,
    ) -> None:
        key = _repository_key(repository_key)
        timestamp = _timestamp(now or _utc_now(), "repository timestamp")
        connection.execute(
            """
            INSERT OR IGNORE INTO repositories(
                repository_key, created_at, last_accessed_at
            ) VALUES (?, ?, ?)
            """,
            (key, timestamp, timestamp),
        )
        connection.execute(
            "INSERT OR IGNORE INTO generations(repository_key) VALUES (?)", (key,)
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO event_chain_heads(
                repository_key, event_count, head_sequence, head_hash
            ) VALUES (?, 0, NULL, ?)
            """,
            (key, ZERO_EVENT_HASH),
        )

    def get_generations(self, repository_key: str) -> GenerationMetadata:
        key = _repository_key(repository_key)
        with self._reader() as connection:
            return self._generations_from_connection(connection, key)

    generation_snapshot = get_generations

    @staticmethod
    def _generations_from_connection(
        connection: sqlite3.Connection,
        repository_key: str,
    ) -> GenerationMetadata:
        row = connection.execute(
            """
            SELECT memory_generation, feedback_generation, knowledge_generation
            FROM generations WHERE repository_key = ?
            """,
            (repository_key,),
        ).fetchone()
        if row is None:
            return GenerationMetadata(
                store_schema_version=STORE_SCHEMA_VERSION,
                memory_generation=0,
                feedback_generation=0,
                knowledge_generation=0,
            )
        try:
            return GenerationMetadata(
                store_schema_version=STORE_SCHEMA_VERSION,
                memory_generation=row["memory_generation"],
                feedback_generation=row["feedback_generation"],
                knowledge_generation=row["knowledge_generation"],
            )
        except (TypeError, ValueError):
            raise MemoryStoreCorruptionError("memory generations are invalid") from None

    @staticmethod
    def _assert_generation(
        connection: sqlite3.Connection,
        repository_key: str,
        kind: str,
        expected: Optional[int],
    ) -> int:
        if expected is not None and (type(expected) is not int or expected < 0):
            raise MemoryStoreValidationError("expected generation must be non-negative")
        generations = MemoryStore._generations_from_connection(
            connection, repository_key
        )
        value = getattr(generations, "%s_generation" % kind)
        if expected is not None and value != expected:
            raise MemoryStoreConflictError()
        return value

    @staticmethod
    def _bump_generation(
        connection: sqlite3.Connection,
        repository_key: str,
        kind: str,
    ) -> int:
        if kind not in {"memory", "feedback", "knowledge"}:
            raise MemoryStoreValidationError("unknown generation kind")
        column = "%s_generation" % kind
        connection.execute(
            "UPDATE generations SET %s = %s + 1 WHERE repository_key = ?"
            % (column, column),
            (repository_key,),
        )
        row = connection.execute(
            "SELECT %s AS value FROM generations WHERE repository_key = ?" % column,
            (repository_key,),
        ).fetchone()
        if row is None or type(row["value"]) is not int:
            raise MemoryStoreCorruptionError("memory generation update failed")
        return row["value"]

    @staticmethod
    def _request_receipt(
        connection: sqlite3.Connection,
        *,
        request_id: str,
        repository_key: Optional[str],
        operation: str,
        request_hash: str,
    ) -> Optional[WriteResult]:
        row = connection.execute(
            """
            SELECT repository_key, operation, request_hash, result_json
            FROM outbox_receipts WHERE request_id = ?
            """,
            (request_id,),
        ).fetchone()
        if row is None:
            return None
        if (
            (repository_key is not None and row["repository_key"] != repository_key)
            or row["operation"] != operation
            or not hmac.compare_digest(str(row["request_hash"]), request_hash)
        ):
            raise MemoryStoreConflictError("request ID was reused for different content")
        try:
            payload = json.loads(row["result_json"])
            result = WriteResult.from_dict(payload)
        except (json.JSONDecodeError, TypeError, ValueError, MemoryStoreError):
            raise MemoryStoreCorruptionError("memory request receipt is invalid") from None
        return replace(result, applied=False, replayed=True)

    @staticmethod
    def _store_request_receipt(
        connection: sqlite3.Connection,
        *,
        request_id: str,
        repository_key: str,
        operation: str,
        request_hash: str,
        result: WriteResult,
        created_at: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO outbox_receipts(
                request_id, repository_key, operation, request_hash, subject_id,
                event_id, result_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                repository_key,
                operation,
                request_hash,
                result.subject_id,
                result.event_id,
                canonical_json(result.to_dict()),
                created_at,
            ),
        )

    @staticmethod
    def _append_event(
        connection: sqlite3.Connection,
        *,
        repository_key: str,
        subject_type: str,
        subject_id: str,
        action: str,
        actor_type: str,
        actor_id: str,
        reason_code: str,
        reason: Optional[str],
        previous_status: Optional[str],
        new_status: Optional[str],
        request_id: str,
        created_at: str,
        generation_kind: str,
        generation: int,
        event_id: Optional[str] = None,
    ) -> MemoryEvent:
        key = _repository_key(repository_key)
        checked_subject_type = _subject_type(subject_type)
        checked_subject_id = _required_text(subject_id, "subject_id", 512)
        checked_action = _required_token(action, "action")
        checked_actor_type = _required_token(actor_type, "actor_type")
        checked_actor_id = _required_text(actor_id, "actor_id", 512)
        checked_reason_code = _required_token(reason_code, "reason_code")
        checked_reason = _optional_text(reason, "reason", 2_048)
        checked_previous = _optional_text(previous_status, "previous_status", 128)
        checked_new = _optional_text(new_status, "new_status", 128)
        checked_request = _request_id(request_id)
        checked_created_at = _timestamp(created_at, "created_at")
        if generation_kind not in {"memory", "feedback", "knowledge"}:
            raise MemoryStoreValidationError("unknown event generation kind")
        if type(generation) is not int or generation < 0:
            raise MemoryStoreValidationError("event generation must be non-negative")
        actual_event_id = _event_id(
            event_id
            or stable_event_id(
                STORE_SCHEMA_NAME,
                key,
                checked_subject_type,
                checked_subject_id,
                checked_action,
                checked_request,
            )
        )

        head = connection.execute(
            """
            SELECT event_count, head_sequence, head_hash
            FROM event_chain_heads WHERE repository_key = ?
            """,
            (key,),
        ).fetchone()
        if head is None:
            raise MemoryStoreCorruptionError("event chain head is missing")
        previous_hash = str(head["head_hash"])
        _digest(previous_hash, "previous event hash")
        max_row = connection.execute(
            """
            SELECT MAX(value) AS maximum FROM (
                SELECT COALESCE(MAX(sequence), 0) AS value FROM events
                UNION ALL
                SELECT COALESCE(MAX(head_sequence), 0) AS value FROM event_chain_heads
            )
            """
        ).fetchone()
        sequence = int(max_row["maximum"]) + 1
        provisional = MemoryEvent(
            sequence=sequence,
            event_id=actual_event_id,
            repository_key=key,
            subject_type=checked_subject_type,
            subject_id=checked_subject_id,
            action=checked_action,
            actor_type=checked_actor_type,
            actor_id=checked_actor_id,
            reason_code=checked_reason_code,
            reason=checked_reason,
            previous_status=checked_previous,
            new_status=checked_new,
            request_id=checked_request,
            created_at=checked_created_at,
            previous_hash=previous_hash,
            current_hash=ZERO_EVENT_HASH,
            generation_kind=generation_kind,
            generation=generation,
        )
        current_hash = canonical_sha256(provisional.hash_payload())
        event = replace(provisional, current_hash=current_hash)
        connection.execute(
            """
            INSERT INTO events(
                sequence, event_id, schema_version, repository_key, subject_type,
                subject_id, action, actor_type, actor_id, reason_code, reason,
                previous_status, new_status, request_id, created_at, previous_hash,
                current_hash, generation_kind, generation
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.sequence,
                event.event_id,
                event.schema_version,
                event.repository_key,
                event.subject_type,
                event.subject_id,
                event.action,
                event.actor_type,
                event.actor_id,
                event.reason_code,
                event.reason,
                event.previous_status,
                event.new_status,
                event.request_id,
                event.created_at,
                event.previous_hash,
                event.current_hash,
                event.generation_kind,
                event.generation,
            ),
        )
        connection.execute(
            """
            UPDATE event_chain_heads
            SET event_count = event_count + 1,
                head_sequence = ?,
                head_hash = ?
            WHERE repository_key = ?
            """,
            (event.sequence, event.current_hash, key),
        )
        return event

    def list_events(
        self,
        repository_key: str,
        *,
        subject_type: Optional[str] = None,
        subject_id: Optional[str] = None,
        validate: bool = True,
    ) -> Tuple[MemoryEvent, ...]:
        key = _repository_key(repository_key)
        checked_type = None if subject_type is None else _subject_type(subject_type)
        checked_id = (
            None
            if subject_id is None
            else _required_text(subject_id, "subject_id", 512)
        )
        if checked_id is not None and checked_type is None:
            raise MemoryStoreValidationError("subject_id requires subject_type")
        with self._reader() as connection:
            if validate:
                self._verify_event_chain_connection(connection, key)
            query = "SELECT * FROM events WHERE repository_key = ?"
            parameters: List[Any] = [key]
            if checked_type is not None:
                query += " AND subject_type = ?"
                parameters.append(checked_type)
            if checked_id is not None:
                query += " AND subject_id = ?"
                parameters.append(checked_id)
            query += " ORDER BY sequence"
            return tuple(
                _event_from_row(row)
                for row in connection.execute(query, tuple(parameters))
            )

    def verify_event_chain(self, repository_key: Optional[str] = None) -> int:
        with self._reader() as connection:
            if repository_key is not None:
                return self._verify_event_chain_connection(
                    connection, _repository_key(repository_key)
                )
            keys = {
                str(row["repository_key"])
                for row in connection.execute(
                    "SELECT repository_key FROM event_chain_heads"
                )
            }
            keys.update(
                str(row["repository_key"])
                for row in connection.execute(
                    "SELECT DISTINCT repository_key FROM events"
                )
            )
            return sum(
                self._verify_event_chain_connection(connection, key)
                for key in sorted(keys)
            )

    @staticmethod
    def _verify_event_chain_connection(
        connection: sqlite3.Connection,
        repository_key: str,
    ) -> int:
        rows = connection.execute(
            "SELECT * FROM events WHERE repository_key = ? ORDER BY sequence",
            (repository_key,),
        ).fetchall()
        expected_previous = ZERO_EVENT_HASH
        previous_sequence = 0
        last_generation = {"memory": 0, "feedback": 0, "knowledge": 0}
        for row in rows:
            event = _event_from_row(row)
            if event.sequence <= previous_sequence:
                raise MemoryStoreCorruptionError("memory event order is invalid")
            if not hmac.compare_digest(event.previous_hash, expected_previous):
                raise MemoryStoreCorruptionError("memory event chain is discontinuous")
            expected_current = canonical_sha256(event.hash_payload())
            if not hmac.compare_digest(event.current_hash, expected_current):
                raise MemoryStoreCorruptionError("memory event hash is invalid")
            if event.generation != last_generation[event.generation_kind] + 1:
                raise MemoryStoreCorruptionError(
                    "memory event generation sequence is invalid"
                )
            last_generation[event.generation_kind] = event.generation
            expected_previous = event.current_hash
            previous_sequence = event.sequence

        head = connection.execute(
            """
            SELECT event_count, head_sequence, head_hash
            FROM event_chain_heads WHERE repository_key = ?
            """,
            (repository_key,),
        ).fetchone()
        if head is None:
            if rows:
                raise MemoryStoreCorruptionError("memory event chain head is missing")
            return 0
        expected_sequence = None if not rows else rows[-1]["sequence"]
        if (
            head["event_count"] != len(rows)
            or head["head_sequence"] != expected_sequence
            or not hmac.compare_digest(str(head["head_hash"]), expected_previous)
        ):
            raise MemoryStoreCorruptionError("memory event chain head is invalid")
        generations = MemoryStore._generations_from_connection(
            connection, repository_key
        )
        if (
            generations.memory_generation != last_generation["memory"]
            or generations.feedback_generation != last_generation["feedback"]
            or generations.knowledge_generation != last_generation["knowledge"]
        ):
            raise MemoryStoreCorruptionError(
                "memory generation does not match the event chain"
            )
        return len(rows)

    def _verify_projection_connection(
        self,
        connection: sqlite3.Connection,
        repository_key: str,
    ) -> None:
        """Verify mutable projections are exact derivatives of audited events."""

        key = _repository_key(repository_key)
        latest: Dict[Tuple[str, str], MemoryEvent] = {}
        for row in connection.execute(
            "SELECT * FROM events WHERE repository_key = ? ORDER BY sequence",
            (key,),
        ):
            event = _event_from_row(row)
            latest[(event.subject_type, event.subject_id)] = event

        def require_projection_event(
            *,
            subject_type: str,
            subject_id: str,
            status: str,
            generation: Any,
            generation_kind: str,
        ) -> MemoryEvent:
            event = latest.get((subject_type, subject_id))
            if (
                event is None
                or event.new_status != status
                or event.generation_kind != generation_kind
                or type(generation) is not int
                or event.generation != generation
            ):
                raise MemoryStoreCorruptionError(
                    "%s projection does not match its latest event" % subject_type
                )
            return event

        candidate_rows = connection.execute(
            "SELECT * FROM candidates WHERE repository_key = ?",
            (key,),
        ).fetchall()
        candidate_ids: Set[str] = set()
        for row in candidate_rows:
            candidate = _candidate_from_row(row)
            candidate_ids.add(candidate.candidate_id)
            require_projection_event(
                subject_type="candidate",
                subject_id=candidate.candidate_id,
                status=candidate.status.value,
                generation=row["generation"],
                generation_kind="memory",
            )

        bundle_rows = connection.execute(
            "SELECT * FROM source_bundles WHERE repository_key = ?",
            (key,),
        ).fetchall()
        bundle_ids: Set[str] = set()
        for row in bundle_rows:
            bundle = _source_bundle_from_row(row)
            bundle_ids.add(bundle.bundle_hash)
            require_projection_event(
                subject_type="source_bundle",
                subject_id=bundle.bundle_hash,
                status="stored",
                generation=row["generation"],
                generation_kind="memory",
            )

        record_rows = connection.execute(
            "SELECT * FROM records WHERE repository_key = ?",
            (key,),
        ).fetchall()
        record_ids: Set[str] = set()
        for row in record_rows:
            record = _record_from_row(row)
            record_ids.add(record.memory_id)
            event = latest.get(("record", record.memory_id))
            if event is None:
                approval = latest.get(("candidate", record.candidate_id))
                if (
                    approval is None
                    or approval.action != "approve"
                    or approval.new_status != CandidateStatus.APPROVED.value
                    or record.status is not RecordStatus.ACTIVE
                    or approval.generation_kind != "memory"
                    or type(row["generation"]) is not int
                    or approval.generation != row["generation"]
                ):
                    raise MemoryStoreCorruptionError(
                        "record projection does not match candidate approval"
                    )
            else:
                require_projection_event(
                    subject_type="record",
                    subject_id=record.memory_id,
                    status=record.status.value,
                    generation=row["generation"],
                    generation_kind="memory",
                )

        feedback_rows = connection.execute(
            "SELECT * FROM feedback WHERE repository_key = ?",
            (key,),
        ).fetchall()
        feedback_ids: Set[str] = set()
        for row in feedback_rows:
            feedback = _feedback_from_row(row)
            feedback_ids.add(feedback.feedback_id)
            require_projection_event(
                subject_type="feedback",
                subject_id=feedback.feedback_id,
                status=feedback.status.value,
                generation=row["generation"],
                generation_kind="feedback",
            )

        knowledge_rows = connection.execute(
            "SELECT * FROM knowledge_entries WHERE repository_key = ?",
            (key,),
        ).fetchall()
        knowledge_ids: Set[str] = set()
        for row in knowledge_rows:
            entry = _knowledge_from_row(connection, row)
            knowledge_ids.add(entry.entry_id)
            require_projection_event(
                subject_type="knowledge",
                subject_id=entry.entry_id,
                status="stored",
                generation=row["generation"],
                generation_kind="knowledge",
            )

        present = {
            "candidate": candidate_ids,
            "source_bundle": bundle_ids,
            "record": record_ids,
            "feedback": feedback_ids,
            "knowledge": knowledge_ids,
        }
        for (subject_type, subject_id), event in latest.items():
            if subject_type == "knowledge" and event.new_status == "deleted":
                if subject_id in knowledge_ids:
                    raise MemoryStoreCorruptionError(
                        "deleted knowledge projection still exists"
                    )
                continue
            if subject_id not in present[subject_type]:
                raise MemoryStoreCorruptionError(
                    "%s event has no matching projection" % subject_type
                )

    def put_candidate(
        self,
        candidate: MemoryCandidate,
        *,
        request_id: Optional[str] = None,
        expected_generation: Optional[int] = None,
        actor_type: Optional[str] = None,
        actor_id: Optional[str] = None,
        reason_code: str = "candidate_submitted",
        reason: Optional[str] = None,
    ) -> WriteResult:
        if not isinstance(candidate, MemoryCandidate):
            raise MemoryStoreValidationError(
                "candidate must be a canonical MemoryCandidate"
            )
        if candidate.sensitivity is Sensitivity.BLOCKED:
            raise MemoryStoreValidationError("blocked candidate content cannot be persisted")
        operation = "put_candidate"
        checked_request = _request_id(
            request_id or stable_request_id(operation, candidate.candidate_id)
        )
        producer_type = actor_type or candidate.producer.producer_type.value
        producer_id = actor_id or candidate.producer.name
        request_hash = canonical_sha256(
            {
                "operation": operation,
                "candidate": candidate.to_dict(),
                "expected_generation": expected_generation,
                "actor_type": producer_type,
                "actor_id": producer_id,
                "reason_code": reason_code,
                "reason": reason,
            }
        )
        model_json, body_hash = _model_storage(candidate)
        with self._authority() as connection:
            self._ensure_repository_rows(
                connection, candidate.repository_key, candidate.created_at
            )
            replay = self._request_receipt(
                connection,
                request_id=checked_request,
                repository_key=candidate.repository_key,
                operation=operation,
                request_hash=request_hash,
            )
            if replay is not None:
                return replay
            self._assert_generation(
                connection,
                candidate.repository_key,
                "memory",
                expected_generation,
            )
            existing = connection.execute(
                "SELECT * FROM candidates WHERE candidate_id = ?",
                (candidate.candidate_id,),
            ).fetchone()
            if existing is not None:
                if (
                    existing["repository_key"] != candidate.repository_key
                    or existing["model_json"] != model_json
                    or existing["body_hash"] != body_hash
                ):
                    raise MemoryStoreConflictError(
                        "candidate ID already exists with different canonical content"
                    )
                result = WriteResult(
                    operation=operation,
                    subject_id=candidate.candidate_id,
                    event_id=None,
                    generations=self._generations_from_connection(
                        connection, candidate.repository_key
                    ),
                    applied=False,
                )
                self._store_request_receipt(
                    connection,
                    request_id=checked_request,
                    repository_key=candidate.repository_key,
                    operation=operation,
                    request_hash=request_hash,
                    result=result,
                    created_at=candidate.created_at,
                )
                return result

            generation = self._bump_generation(
                connection, candidate.repository_key, "memory"
            )
            connection.execute(
                """
                INSERT INTO candidates(
                    candidate_id, repository_key, content_fingerprint, model_json,
                    body_hash, current_status, generation, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate.candidate_id,
                    candidate.repository_key,
                    candidate.content_fingerprint,
                    model_json,
                    body_hash,
                    candidate.status.value,
                    generation,
                    candidate.created_at,
                ),
            )
            event = self._append_event(
                connection,
                repository_key=candidate.repository_key,
                subject_type="candidate",
                subject_id=candidate.candidate_id,
                action="candidate_submitted",
                actor_type=producer_type,
                actor_id=producer_id,
                reason_code=reason_code,
                reason=reason,
                previous_status=None,
                new_status=candidate.status.value,
                request_id=checked_request,
                created_at=candidate.created_at,
                generation_kind="memory",
                generation=generation,
            )
            result = WriteResult(
                operation=operation,
                subject_id=candidate.candidate_id,
                event_id=event.event_id,
                generations=self._generations_from_connection(
                    connection, candidate.repository_key
                ),
                applied=True,
            )
            self._store_request_receipt(
                connection,
                request_id=checked_request,
                repository_key=candidate.repository_key,
                operation=operation,
                request_hash=request_hash,
                result=result,
                created_at=candidate.created_at,
            )
            return result

    store_candidate = put_candidate

    def find_candidate(self, candidate_id: str) -> Optional[MemoryCandidate]:
        checked_id = _stable_subject_id(candidate_id, "MC", "candidate_id")
        with self._reader() as connection:
            row = connection.execute(
                "SELECT * FROM candidates WHERE candidate_id = ?", (checked_id,)
            ).fetchone()
            return None if row is None else _candidate_from_row(row)

    def get_candidate(self, candidate_id: str) -> MemoryCandidate:
        candidate = self.find_candidate(candidate_id)
        if candidate is None:
            raise MemoryStoreNotFoundError("memory candidate was not found")
        return candidate

    def list_candidates(
        self,
        repository_key: str,
        *,
        status: Optional[CandidateStatus] = None,
    ) -> Tuple[MemoryCandidate, ...]:
        key = _repository_key(repository_key)
        if status is not None and not isinstance(status, CandidateStatus):
            raise MemoryStoreValidationError("candidate status is invalid")
        query = "SELECT * FROM candidates WHERE repository_key = ?"
        parameters: List[Any] = [key]
        if status is not None:
            query += " AND current_status = ?"
            parameters.append(status.value)
        query += " ORDER BY candidate_id"
        with self._reader() as connection:
            return tuple(
                _candidate_from_row(row)
                for row in connection.execute(query, tuple(parameters))
            )

    def transition_candidate(
        self,
        candidate_id: str,
        *,
        expected_status: CandidateStatus,
        new_status: CandidateStatus,
        action: str,
        actor_type: str,
        actor_id: str,
        reason_code: str,
        request_id: str,
        created_at: Optional[str] = None,
        reason: Optional[str] = None,
        expected_generation: Optional[int] = None,
    ) -> WriteResult:
        checked_id = _stable_subject_id(candidate_id, "MC", "candidate_id")
        if not isinstance(expected_status, CandidateStatus) or not isinstance(
            new_status, CandidateStatus
        ):
            raise MemoryStoreValidationError("candidate status is invalid")
        operation = "transition_candidate"
        timestamp = _timestamp(created_at or _utc_now(), "created_at")
        checked_request = _request_id(request_id)
        request_hash = canonical_sha256(
            {
                "operation": operation,
                "candidate_id": checked_id,
                "expected_status": expected_status.value,
                "new_status": new_status.value,
                "action": action,
                "actor_type": actor_type,
                "actor_id": actor_id,
                "reason_code": reason_code,
                "reason": reason,
                "expected_generation": expected_generation,
            }
        )
        with self._authority() as connection:
            row = connection.execute(
                "SELECT * FROM candidates WHERE candidate_id = ?", (checked_id,)
            ).fetchone()
            if row is None:
                raise MemoryStoreNotFoundError("memory candidate was not found")
            candidate = _candidate_from_row(row)
            replay = self._request_receipt(
                connection,
                request_id=checked_request,
                repository_key=candidate.repository_key,
                operation=operation,
                request_hash=request_hash,
            )
            if replay is not None:
                return replay
            self._assert_generation(
                connection,
                candidate.repository_key,
                "memory",
                expected_generation,
            )
            if candidate.status is not expected_status:
                raise MemoryStoreConflictError()
            if candidate.status is new_status:
                result = WriteResult(
                    operation=operation,
                    subject_id=checked_id,
                    event_id=None,
                    generations=self._generations_from_connection(
                        connection, candidate.repository_key
                    ),
                    applied=False,
                )
            else:
                generation = self._bump_generation(
                    connection, candidate.repository_key, "memory"
                )
                connection.execute(
                    """
                    UPDATE candidates SET current_status = ?, generation = ?
                    WHERE candidate_id = ? AND current_status = ?
                    """,
                    (new_status.value, generation, checked_id, expected_status.value),
                )
                if connection.execute("SELECT changes()").fetchone()[0] != 1:
                    raise MemoryStoreConflictError()
                event = self._append_event(
                    connection,
                    repository_key=candidate.repository_key,
                    subject_type="candidate",
                    subject_id=checked_id,
                    action=action,
                    actor_type=actor_type,
                    actor_id=actor_id,
                    reason_code=reason_code,
                    reason=reason,
                    previous_status=expected_status.value,
                    new_status=new_status.value,
                    request_id=checked_request,
                    created_at=timestamp,
                    generation_kind="memory",
                    generation=generation,
                )
                result = WriteResult(
                    operation=operation,
                    subject_id=checked_id,
                    event_id=event.event_id,
                    generations=self._generations_from_connection(
                        connection, candidate.repository_key
                    ),
                    applied=True,
                )
            self._store_request_receipt(
                connection,
                request_id=checked_request,
                repository_key=candidate.repository_key,
                operation=operation,
                request_hash=request_hash,
                result=result,
                created_at=timestamp,
            )
            return result

    def blob_path(self, blob_hash: str) -> Path:
        digest = _digest(blob_hash, "blob_hash")
        candidate = self.blob_root / digest[:2] / digest
        self._require_safe_blob_path(candidate)
        return candidate

    def _initialize_blob_layout(self) -> None:
        blobs_directory = self.namespace_path / "blobs"
        for directory in (blobs_directory, self.blob_root, self._blob_temp_root):
            if directory.is_symlink():
                raise MemoryStoreCorruptionError(
                    "memory blob layout traverses a symbolic link"
                )
            if directory.exists() and not directory.is_dir():
                raise MemoryStoreCorruptionError("memory blob layout is invalid")
            directory.mkdir(exist_ok=True)
        self._require_safe_blob_path(self.blob_root / "00" / ("0" * 64))

    def _require_safe_blob_path(self, candidate: Path) -> None:
        components = (
            self.namespace_path / "blobs",
            self.blob_root,
            self._blob_temp_root,
            candidate.parent,
        )
        for component in components:
            if component.is_symlink():
                raise MemoryStoreCorruptionError(
                    "memory blob layout traverses a symbolic link"
                )
            if component.exists() and not component.is_dir():
                raise MemoryStoreCorruptionError("memory blob layout is invalid")
        try:
            resolved = candidate.resolve(strict=False)
            root = self.namespace_path.resolve(strict=False)
            if os.path.commonpath((str(resolved), str(root))) != str(root):
                raise MemoryStoreCorruptionError("memory blob path escapes its namespace")
        except ValueError:
            raise MemoryStoreCorruptionError("memory blob path escapes its namespace") from None

    def put_blob(
        self,
        content: Union[bytes, bytearray, memoryview],
        *,
        media_type: str,
        expected_hash: Optional[str] = None,
        expected_size: Optional[int] = None,
        created_at: Optional[str] = None,
    ) -> BlobInfo:
        if self._read_only:
            raise MemoryStoreReadOnlyError()
        with _exclusive_file_lock(self._blob_lock_path, self._busy_timeout_ms):
            return self._put_blob_locked(
                content,
                media_type=media_type,
                expected_hash=expected_hash,
                expected_size=expected_size,
                created_at=created_at,
            )

    def _put_blob_locked(
        self,
        content: Union[bytes, bytearray, memoryview],
        *,
        media_type: str,
        expected_hash: Optional[str] = None,
        expected_size: Optional[int] = None,
        created_at: Optional[str] = None,
    ) -> BlobInfo:
        if not isinstance(content, (bytes, bytearray, memoryview)):
            raise MemoryStoreValidationError("blob content must be bytes")
        raw = bytes(content)
        if len(raw) > MAX_BLOB_SIZE_BYTES:
            raise MemoryStoreValidationError("blob exceeds the supported size limit")
        checked_media_type = _media_type(media_type)
        digest = hashlib.sha256(raw).hexdigest()
        size = len(raw)
        if expected_hash is not None and not hmac.compare_digest(
            digest, _digest(expected_hash, "expected_hash")
        ):
            raise MemoryStoreValidationError("blob hash validation failed")
        if expected_size is not None:
            if type(expected_size) is not int or expected_size < 0:
                raise MemoryStoreValidationError("expected blob size is invalid")
            if expected_size != size:
                raise MemoryStoreValidationError("blob size validation failed")
        timestamp = _timestamp(created_at or _utc_now(), "created_at")
        destination = self.blob_path(digest)
        self._promote_blob(raw, digest, size, destination)

        with self._authority() as connection:
            existing = connection.execute(
                "SELECT * FROM blobs WHERE blob_hash = ?", (digest,)
            ).fetchone()
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO blobs(blob_hash, size_bytes, media_type, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (digest, size, checked_media_type, timestamp),
                )
                stored_created_at = timestamp
            else:
                if (
                    existing["size_bytes"] != size
                    or existing["media_type"] != checked_media_type
                ):
                    raise MemoryStoreCorruptionError("blob metadata is inconsistent")
                stored_created_at = str(existing["created_at"])
        return BlobInfo(
            blob_hash=digest,
            size_bytes=size,
            media_type=checked_media_type,
            created_at=stored_created_at,
            path=str(destination),
        )

    store_blob = put_blob

    def repair_blob(
        self,
        content: Union[bytes, bytearray, memoryview],
        *,
        media_type: str,
        expected_hash: Optional[str] = None,
        expected_size: Optional[int] = None,
    ) -> BlobInfo:
        """Atomically restore a corrupt/missing file for existing blob metadata.

        Repository-cache rebuilds must not unlink content-addressed files behind
        the Store's back.  Repair is serialized with readers, promotion, and GC,
        and it never creates or changes canonical blob metadata.
        """

        if self._read_only:
            raise MemoryStoreReadOnlyError()
        if not isinstance(content, (bytes, bytearray, memoryview)):
            raise MemoryStoreValidationError("blob content must be bytes")
        raw = bytes(content)
        if len(raw) > MAX_BLOB_SIZE_BYTES:
            raise MemoryStoreValidationError("blob exceeds the supported size limit")
        checked_media_type = _media_type(media_type)
        digest = hashlib.sha256(raw).hexdigest()
        size = len(raw)
        if expected_hash is not None and not hmac.compare_digest(
            digest, _digest(expected_hash, "expected_hash")
        ):
            raise MemoryStoreValidationError("blob hash validation failed")
        if expected_size is not None:
            if type(expected_size) is not int or expected_size < 0:
                raise MemoryStoreValidationError("expected blob size is invalid")
            if expected_size != size:
                raise MemoryStoreValidationError("blob size validation failed")

        with _exclusive_file_lock(self._blob_lock_path, self._busy_timeout_ms):
            destination = self.blob_path(digest)
            with self._authority() as connection:
                row = connection.execute(
                    "SELECT * FROM blobs WHERE blob_hash = ?", (digest,)
                ).fetchone()
                if row is None:
                    raise MemoryStoreCorruptionError(
                        "cannot repair unregistered memory blob"
                    )
                if (
                    row["size_bytes"] != size
                    or row["media_type"] != checked_media_type
                ):
                    raise MemoryStoreCorruptionError(
                        "blob metadata is inconsistent"
                    )
                created_at = _timestamp(row["created_at"], "blob created_at")
                try:
                    self._validate_blob_file_values(destination, digest, size)
                except MemoryStoreCorruptionError:
                    self._replace_blob_file(raw, digest, size, destination)

            return BlobInfo(
                blob_hash=digest,
                size_bytes=size,
                media_type=checked_media_type,
                created_at=created_at,
                path=str(destination),
            )

    def _promote_blob(
        self,
        content: bytes,
        digest: str,
        size: int,
        destination: Path,
    ) -> None:
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            self._blob_temp_root.mkdir(parents=True, exist_ok=True)
            if destination.exists() or destination.is_symlink():
                self._validate_blob_file_values(destination, digest, size)
                return
            temporary = self._blob_temp_root / (".%s.%s.tmp" % (digest, uuid.uuid4().hex))
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_BINARY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(str(temporary), flags, 0o600)
            try:
                offset = 0
                while offset < size:
                    written = os.write(descriptor, content[offset : offset + 1024 * 1024])
                    if written <= 0:
                        raise OSError("short blob write")
                    offset += written
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            try:
                self._validate_blob_file_values(temporary, digest, size)
                os.replace(str(temporary), str(destination))
                _fsync_directory(destination.parent)
                self._validate_blob_file_values(destination, digest, size)
            finally:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
        except MemoryStoreError:
            raise
        except OSError:
            raise MemoryStoreUnavailableError("memory blob store is unavailable") from None

    def _replace_blob_file(
        self,
        content: bytes,
        digest: str,
        size: int,
        destination: Path,
    ) -> None:
        """Replace one known corrupt blob while the process lock is held."""

        temporary = self._blob_temp_root / (".%s.%s.repair" % (digest, uuid.uuid4().hex))
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            self._blob_temp_root.mkdir(parents=True, exist_ok=True)
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_BINARY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(str(temporary), flags, 0o600)
            try:
                offset = 0
                while offset < size:
                    written = os.write(
                        descriptor,
                        content[offset : offset + 1024 * 1024],
                    )
                    if written <= 0:
                        raise OSError("short blob repair write")
                    offset += written
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            self._validate_blob_file_values(temporary, digest, size)
            os.replace(str(temporary), str(destination))
            _fsync_directory(destination.parent)
            self._validate_blob_file_values(destination, digest, size)
        except MemoryStoreError:
            raise
        except OSError:
            raise MemoryStoreUnavailableError(
                "memory blob store is unavailable"
            ) from None
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    @staticmethod
    def _validate_blob_file_values(path: Path, digest: str, size: int) -> None:
        try:
            if path.is_symlink():
                raise MemoryStoreCorruptionError("memory blob is not a regular file")
            file_status = path.stat()
            if not stat.S_ISREG(file_status.st_mode) or file_status.st_size != size:
                raise MemoryStoreCorruptionError("memory blob size validation failed")
            actual = hashlib.sha256()
            with path.open("rb") as stream:
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    actual.update(chunk)
            if not hmac.compare_digest(actual.hexdigest(), digest):
                raise MemoryStoreCorruptionError("memory blob hash validation failed")
        except MemoryStoreError:
            raise
        except (OSError, ValueError):
            raise MemoryStoreCorruptionError("memory blob is missing or unreadable") from None

    def _blob_info_from_connection(
        self,
        connection: sqlite3.Connection,
        blob_hash: str,
        *,
        validate_file: bool,
    ) -> BlobInfo:
        digest = _digest(blob_hash, "blob_hash")
        row = connection.execute(
            "SELECT * FROM blobs WHERE blob_hash = ?", (digest,)
        ).fetchone()
        if row is None:
            raise MemoryStoreCorruptionError("database references an unknown memory blob")
        try:
            size = int(row["size_bytes"])
            media_type = _media_type(row["media_type"])
            created_at = _timestamp(row["created_at"], "blob created_at")
        except (TypeError, ValueError, MemoryStoreError):
            raise MemoryStoreCorruptionError("memory blob metadata is invalid") from None
        info = BlobInfo(
            blob_hash=digest,
            size_bytes=size,
            media_type=media_type,
            created_at=created_at,
            path=str(self.blob_path(digest)),
        )
        if validate_file:
            self._validate_blob_file_values(Path(info.path), digest, size)
        return info

    def get_blob_info(self, blob_hash: str, *, validate: bool = True) -> BlobInfo:
        with self._reader() as connection:
            return self._blob_info_from_connection(
                connection, blob_hash, validate_file=validate
            )

    def validate_blob(self, blob_hash: str) -> BlobInfo:
        return self.get_blob_info(blob_hash, validate=True)

    def read_blob(self, blob_hash: str) -> bytes:
        with _exclusive_file_lock(self._blob_lock_path, self._busy_timeout_ms):
            info = self.get_blob_info(blob_hash, validate=True)
            try:
                return Path(info.path).read_bytes()
            except OSError:
                raise MemoryStoreCorruptionError(
                    "memory blob is missing or unreadable"
                ) from None

    def pin_blob(
        self,
        blob_hash: str,
        *,
        pin_type: str,
        pin_id: str,
        created_at: Optional[str] = None,
    ) -> bool:
        digest = _digest(blob_hash, "blob_hash")
        checked_type = _pin_type(pin_type)
        checked_id = _required_text(pin_id, "pin_id", 512)
        timestamp = _timestamp(created_at or _utc_now(), "created_at")
        with self._authority() as connection:
            self._blob_info_from_connection(connection, digest, validate_file=True)
            connection.execute(
                """
                INSERT OR IGNORE INTO blob_pins(blob_hash, pin_type, pin_id, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (digest, checked_type, checked_id, timestamp),
            )
            return connection.execute("SELECT changes()").fetchone()[0] == 1

    def unpin_blob(self, blob_hash: str, *, pin_type: str, pin_id: str) -> bool:
        digest = _digest(blob_hash, "blob_hash")
        checked_type = _pin_type(pin_type)
        if checked_type == "source_bundle":
            raise MemoryStoreConflictError("source-bundle pins are permanent")
        checked_id = _required_text(pin_id, "pin_id", 512)
        with self._authority() as connection:
            connection.execute(
                "DELETE FROM blob_pins WHERE blob_hash = ? AND pin_type = ? AND pin_id = ?",
                (digest, checked_type, checked_id),
            )
            return connection.execute("SELECT changes()").fetchone()[0] == 1

    def put_source_bundle(
        self,
        bundle: SourceBundleDescriptor,
        *,
        request_id: Optional[str] = None,
        expected_generation: Optional[int] = None,
    ) -> WriteResult:
        if not isinstance(bundle, SourceBundleDescriptor):
            raise MemoryStoreValidationError(
                "source bundle must be a canonical SourceBundleDescriptor"
            )
        operation = "put_source_bundle"
        checked_request = _request_id(
            request_id or stable_request_id(operation, bundle.bundle_hash)
        )
        request_hash = canonical_sha256(
            {
                "operation": operation,
                "bundle": bundle.to_dict(),
                "expected_generation": expected_generation,
            }
        )
        model_json, body_hash = _model_storage(bundle)
        with self._authority() as connection:
            self._ensure_repository_rows(connection, bundle.repository_key, bundle.created_at)
            replay = self._request_receipt(
                connection,
                request_id=checked_request,
                repository_key=bundle.repository_key,
                operation=operation,
                request_hash=request_hash,
            )
            if replay is not None:
                return replay
            self._assert_generation(
                connection, bundle.repository_key, "memory", expected_generation
            )
            candidate_row = connection.execute(
                "SELECT * FROM candidates WHERE candidate_id = ?",
                (bundle.candidate_id,),
            ).fetchone()
            if candidate_row is None:
                raise MemoryStoreNotFoundError("source bundle candidate was not found")
            candidate = _candidate_from_row(candidate_row)
            if candidate.repository_key != bundle.repository_key:
                raise MemoryStoreConflictError("source bundle repository does not match")
            blob = self._blob_info_from_connection(
                connection, bundle.blob_hash, validate_file=True
            )
            if blob.size_bytes != bundle.size_bytes or blob.media_type != bundle.media_type:
                raise MemoryStoreConflictError("source bundle blob metadata does not match")
            existing = connection.execute(
                "SELECT * FROM source_bundles WHERE bundle_hash = ?",
                (bundle.bundle_hash,),
            ).fetchone()
            if existing is not None:
                if existing["model_json"] != model_json or existing["body_hash"] != body_hash:
                    raise MemoryStoreConflictError(
                        "source bundle hash already has different canonical content"
                    )
                result = WriteResult(
                    operation=operation,
                    subject_id=bundle.bundle_hash,
                    event_id=None,
                    generations=self._generations_from_connection(
                        connection, bundle.repository_key
                    ),
                    applied=False,
                )
            else:
                generation = self._bump_generation(
                    connection, bundle.repository_key, "memory"
                )
                connection.execute(
                    """
                    INSERT INTO source_bundles(
                        bundle_hash, repository_key, candidate_id, blob_hash,
                        model_json, body_hash, generation, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        bundle.bundle_hash,
                        bundle.repository_key,
                        bundle.candidate_id,
                        bundle.blob_hash,
                        model_json,
                        body_hash,
                        generation,
                        bundle.created_at,
                    ),
                )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO blob_pins(blob_hash, pin_type, pin_id, created_at)
                    VALUES (?, 'source_bundle', ?, ?)
                    """,
                    (bundle.blob_hash, bundle.bundle_hash, bundle.created_at),
                )
                event = self._append_event(
                    connection,
                    repository_key=bundle.repository_key,
                    subject_type="source_bundle",
                    subject_id=bundle.bundle_hash,
                    action="source_bundle_stored",
                    actor_type="runtime",
                    actor_id="memory_sources",
                    reason_code="source_bundle_materialized",
                    reason=None,
                    previous_status=None,
                    new_status="stored",
                    request_id=checked_request,
                    created_at=bundle.created_at,
                    generation_kind="memory",
                    generation=generation,
                )
                result = WriteResult(
                    operation=operation,
                    subject_id=bundle.bundle_hash,
                    event_id=event.event_id,
                    generations=self._generations_from_connection(
                        connection, bundle.repository_key
                    ),
                    applied=True,
                )
            self._store_request_receipt(
                connection,
                request_id=checked_request,
                repository_key=bundle.repository_key,
                operation=operation,
                request_hash=request_hash,
                result=result,
                created_at=bundle.created_at,
            )
            return result

    store_source_bundle = put_source_bundle

    def find_source_bundle(self, bundle_hash: str) -> Optional[SourceBundleDescriptor]:
        digest = _digest(bundle_hash, "bundle_hash")
        with self._reader() as connection:
            row = connection.execute(
                "SELECT * FROM source_bundles WHERE bundle_hash = ?", (digest,)
            ).fetchone()
            if row is None:
                return None
            bundle = _validated_source_bundle_from_row(connection, row)
            blob = self._blob_info_from_connection(
                connection, bundle.blob_hash, validate_file=True
            )
            if blob.size_bytes != bundle.size_bytes or blob.media_type != bundle.media_type:
                raise MemoryStoreCorruptionError("source bundle blob metadata is invalid")
            return bundle

    def get_source_bundle(self, bundle_hash: str) -> SourceBundleDescriptor:
        bundle = self.find_source_bundle(bundle_hash)
        if bundle is None:
            raise MemoryStoreNotFoundError("source bundle was not found")
        return bundle

    def approve_candidate_with_source_bundle(
        self,
        record: DurableMemoryRecord,
        bundle: SourceBundleDescriptor,
        *,
        request_id: str,
        expected_candidate_status: CandidateStatus,
        expected_generation: Optional[int] = None,
        actor_type: str = "human",
        actor_id: Optional[str] = None,
        reason_code: str = "approved",
        reason: Optional[str] = None,
        supersede_memory_id: Optional[str] = None,
        expected_supersede_status: Optional[RecordStatus] = None,
    ) -> WriteResult:
        """Atomically pin evidence, approve a candidate, and optionally supersede.

        The content-addressed blob may have been promoted before this call; a
        blob by itself carries no Memory authority.  The SourceBundle row, its
        permanent pin, the approved Record, Candidate projection, events, and
        optional predecessor supersession are committed in one SQLite authority
        transaction.  A failure therefore leaves no observable half-authority
        state.
        """

        if not isinstance(record, DurableMemoryRecord):
            raise MemoryStoreValidationError(
                "record must be a canonical DurableMemoryRecord"
            )
        if not isinstance(bundle, SourceBundleDescriptor):
            raise MemoryStoreValidationError(
                "source bundle must be a canonical SourceBundleDescriptor"
            )
        if record.sensitivity is Sensitivity.BLOCKED:
            raise MemoryStoreValidationError("blocked memory content cannot be persisted")
        if record.status is not RecordStatus.ACTIVE:
            raise MemoryStoreValidationError(
                "a newly approved durable memory record must be active"
            )
        if not isinstance(expected_candidate_status, CandidateStatus):
            raise MemoryStoreValidationError("candidate status is invalid")
        if expected_candidate_status is CandidateStatus.APPROVED:
            raise MemoryStoreValidationError(
                "candidate approval requires a pre-approval status"
            )
        checked_supersede_id = (
            None
            if supersede_memory_id is None
            else _stable_subject_id(
                supersede_memory_id,
                "MEM",
                "supersede_memory_id",
            )
        )
        if checked_supersede_id is None:
            if expected_supersede_status is not None:
                raise MemoryStoreValidationError(
                    "expected supersede status requires a predecessor record"
                )
        else:
            if not isinstance(expected_supersede_status, RecordStatus):
                raise MemoryStoreValidationError(
                    "predecessor record status is invalid"
                )
            if expected_supersede_status not in {
                RecordStatus.ACTIVE,
                RecordStatus.REVALIDATION_REQUIRED,
            }:
                raise MemoryStoreValidationError(
                    "only active or revalidation-required records can be superseded"
                )
            if checked_supersede_id == record.memory_id:
                raise MemoryStoreValidationError(
                    "a record cannot supersede itself"
                )
        if (
            bundle.repository_key != record.repository_key
            or bundle.candidate_id != record.candidate_id
            or bundle.source_refs != record.source_refs
        ):
            raise MemoryStoreValidationError(
                "source bundle does not match the approved record"
            )
        if bundle.created_at != record.created_at:
            raise MemoryStoreValidationError(
                "source bundle and approved record timestamps must match"
            )

        operation = "approve_candidate_with_source_bundle"
        checked_request = _request_id(request_id)
        approver = actor_id or record.approved_by
        request_hash = canonical_sha256(
            {
                "operation": operation,
                "record": record.to_dict(),
                "bundle": bundle.to_dict(),
                "expected_candidate_status": expected_candidate_status.value,
                "expected_generation": expected_generation,
                "actor_type": actor_type,
                "actor_id": approver,
                "reason_code": reason_code,
                "reason": reason,
                "supersede_memory_id": checked_supersede_id,
                "expected_supersede_status": (
                    None
                    if expected_supersede_status is None
                    else expected_supersede_status.value
                ),
            }
        )
        record_json, record_body_hash = _model_storage(record)
        bundle_json, bundle_body_hash = _model_storage(bundle)

        with self._authority() as connection:
            self._ensure_repository_rows(
                connection,
                record.repository_key,
                record.created_at,
            )
            replay = self._request_receipt(
                connection,
                request_id=checked_request,
                repository_key=record.repository_key,
                operation=operation,
                request_hash=request_hash,
            )
            if replay is not None:
                return replay
            self._assert_generation(
                connection,
                record.repository_key,
                "memory",
                expected_generation,
            )

            candidate_row = connection.execute(
                "SELECT * FROM candidates WHERE candidate_id = ?",
                (record.candidate_id,),
            ).fetchone()
            if candidate_row is None:
                raise MemoryStoreNotFoundError("memory candidate was not found")
            candidate = _candidate_from_row(candidate_row)
            if not _record_matches_candidate(record, candidate):
                raise MemoryStoreConflictError(
                    "record body does not match its immutable candidate"
                )
            if candidate.status is not expected_candidate_status:
                raise MemoryStoreConflictError()
            if connection.execute(
                "SELECT 1 FROM records WHERE candidate_id = ?",
                (record.candidate_id,),
            ).fetchone() is not None:
                raise MemoryStoreConflictError(
                    "candidate already has a durable memory record"
                )

            predecessor: Optional[DurableMemoryRecord] = None
            if checked_supersede_id is not None:
                predecessor_row = connection.execute(
                    "SELECT * FROM records WHERE memory_id = ?",
                    (checked_supersede_id,),
                ).fetchone()
                if predecessor_row is None:
                    raise MemoryStoreNotFoundError(
                        "predecessor durable memory record was not found"
                    )
                predecessor = _validated_record_from_row(
                    connection,
                    predecessor_row,
                )
                if predecessor.repository_key != record.repository_key:
                    raise MemoryStoreConflictError(
                        "replacement and predecessor repositories differ"
                    )
                if predecessor.status is not expected_supersede_status:
                    raise MemoryStoreConflictError()

            blob = self._blob_info_from_connection(
                connection,
                bundle.blob_hash,
                validate_file=True,
            )
            if (
                blob.size_bytes != bundle.size_bytes
                or blob.media_type != bundle.media_type
            ):
                raise MemoryStoreConflictError(
                    "source bundle blob metadata does not match"
                )

            bundle_row = connection.execute(
                "SELECT * FROM source_bundles WHERE bundle_hash = ?",
                (bundle.bundle_hash,),
            ).fetchone()
            if bundle_row is None:
                bundle_generation = self._bump_generation(
                    connection,
                    bundle.repository_key,
                    "memory",
                )
                connection.execute(
                    """
                    INSERT INTO source_bundles(
                        bundle_hash, repository_key, candidate_id, blob_hash,
                        model_json, body_hash, generation, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        bundle.bundle_hash,
                        bundle.repository_key,
                        bundle.candidate_id,
                        bundle.blob_hash,
                        bundle_json,
                        bundle_body_hash,
                        bundle_generation,
                        bundle.created_at,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO blob_pins(blob_hash, pin_type, pin_id, created_at)
                    VALUES (?, 'source_bundle', ?, ?)
                    """,
                    (
                        bundle.blob_hash,
                        bundle.bundle_hash,
                        bundle.created_at,
                    ),
                )
                self._append_event(
                    connection,
                    repository_key=bundle.repository_key,
                    subject_type="source_bundle",
                    subject_id=bundle.bundle_hash,
                    action="source_bundle_stored",
                    actor_type="runtime",
                    actor_id="memory_lifecycle",
                    reason_code="source_bundle_materialized",
                    reason=None,
                    previous_status=None,
                    new_status="stored",
                    request_id=stable_request_id(
                        operation,
                        "source_bundle",
                        checked_request,
                        bundle.bundle_hash,
                    ),
                    created_at=bundle.created_at,
                    generation_kind="memory",
                    generation=bundle_generation,
                )
            else:
                existing_bundle = _validated_source_bundle_from_row(
                    connection,
                    bundle_row,
                )
                if existing_bundle != bundle:
                    raise MemoryStoreConflictError(
                        "source bundle hash already has different canonical content"
                    )
                existing_pin = connection.execute(
                    """
                    SELECT 1 FROM blob_pins
                    WHERE blob_hash = ? AND pin_type = 'source_bundle' AND pin_id = ?
                    """,
                    (bundle.blob_hash, bundle.bundle_hash),
                ).fetchone()
                if existing_pin is None:
                    raise MemoryStoreCorruptionError(
                        "source bundle permanent pin is missing"
                    )

            approval_generation = self._bump_generation(
                connection,
                record.repository_key,
                "memory",
            )
            connection.execute(
                """
                UPDATE candidates SET current_status = ?, generation = ?
                WHERE candidate_id = ? AND current_status = ?
                """,
                (
                    CandidateStatus.APPROVED.value,
                    approval_generation,
                    record.candidate_id,
                    expected_candidate_status.value,
                ),
            )
            if connection.execute("SELECT changes()").fetchone()[0] != 1:
                raise MemoryStoreConflictError()
            connection.execute(
                """
                INSERT INTO records(
                    memory_id, candidate_id, repository_key, source_bundle_hash,
                    model_json, body_hash, current_status, generation, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.memory_id,
                    record.candidate_id,
                    record.repository_key,
                    record.source_bundle_hash,
                    record_json,
                    record_body_hash,
                    record.status.value,
                    approval_generation,
                    record.created_at,
                ),
            )
            approval_event = self._append_event(
                connection,
                repository_key=record.repository_key,
                subject_type="candidate",
                subject_id=record.candidate_id,
                action="approve",
                actor_type=actor_type,
                actor_id=approver,
                reason_code=reason_code,
                reason=reason,
                previous_status=expected_candidate_status.value,
                new_status=CandidateStatus.APPROVED.value,
                request_id=checked_request,
                created_at=record.created_at,
                generation_kind="memory",
                generation=approval_generation,
                event_id=record.approval_event_id,
            )

            if predecessor is not None:
                supersede_generation = self._bump_generation(
                    connection,
                    record.repository_key,
                    "memory",
                )
                connection.execute(
                    """
                    UPDATE records SET current_status = ?, generation = ?
                    WHERE memory_id = ? AND current_status = ?
                    """,
                    (
                        RecordStatus.SUPERSEDED.value,
                        supersede_generation,
                        predecessor.memory_id,
                        expected_supersede_status.value,
                    ),
                )
                if connection.execute("SELECT changes()").fetchone()[0] != 1:
                    raise MemoryStoreConflictError()
                self._append_event(
                    connection,
                    repository_key=record.repository_key,
                    subject_type="record",
                    subject_id=predecessor.memory_id,
                    action="supersede",
                    actor_type=actor_type,
                    actor_id=approver,
                    reason_code="revalidated",
                    reason=reason,
                    previous_status=expected_supersede_status.value,
                    new_status=RecordStatus.SUPERSEDED.value,
                    request_id=stable_request_id(
                        operation,
                        "supersede",
                        checked_request,
                        predecessor.memory_id,
                        record.memory_id,
                    ),
                    created_at=record.created_at,
                    generation_kind="memory",
                    generation=supersede_generation,
                )

            result = WriteResult(
                operation=operation,
                subject_id=record.memory_id,
                event_id=approval_event.event_id,
                generations=self._generations_from_connection(
                    connection,
                    record.repository_key,
                ),
                applied=True,
            )
            self._store_request_receipt(
                connection,
                request_id=checked_request,
                repository_key=record.repository_key,
                operation=operation,
                request_hash=request_hash,
                result=result,
                created_at=record.created_at,
            )
            return result

    def approve_candidate(
        self,
        record: DurableMemoryRecord,
        *,
        request_id: str,
        expected_candidate_status: CandidateStatus,
        expected_generation: Optional[int] = None,
        actor_type: str = "human",
        actor_id: Optional[str] = None,
        reason_code: str = "approved",
        reason: Optional[str] = None,
    ) -> WriteResult:
        if not isinstance(record, DurableMemoryRecord):
            raise MemoryStoreValidationError(
                "record must be a canonical DurableMemoryRecord"
            )
        if record.sensitivity is Sensitivity.BLOCKED:
            raise MemoryStoreValidationError("blocked memory content cannot be persisted")
        if record.status is not RecordStatus.ACTIVE:
            raise MemoryStoreValidationError(
                "a newly approved durable memory record must be active"
            )
        if not isinstance(expected_candidate_status, CandidateStatus):
            raise MemoryStoreValidationError("candidate status is invalid")
        operation = "approve_candidate"
        checked_request = _request_id(request_id)
        approver = actor_id or record.approved_by
        request_hash = canonical_sha256(
            {
                "operation": operation,
                "record": record.to_dict(),
                "expected_candidate_status": expected_candidate_status.value,
                "expected_generation": expected_generation,
                "actor_type": actor_type,
                "actor_id": approver,
                "reason_code": reason_code,
                "reason": reason,
            }
        )
        model_json, body_hash = _model_storage(record)
        with self._authority() as connection:
            self._ensure_repository_rows(connection, record.repository_key, record.created_at)
            replay = self._request_receipt(
                connection,
                request_id=checked_request,
                repository_key=record.repository_key,
                operation=operation,
                request_hash=request_hash,
            )
            if replay is not None:
                return replay
            self._assert_generation(
                connection, record.repository_key, "memory", expected_generation
            )
            candidate_row = connection.execute(
                "SELECT * FROM candidates WHERE candidate_id = ?",
                (record.candidate_id,),
            ).fetchone()
            if candidate_row is None:
                raise MemoryStoreNotFoundError("memory candidate was not found")
            candidate = _candidate_from_row(candidate_row)
            if not _record_matches_candidate(record, candidate):
                raise MemoryStoreConflictError(
                    "record body does not match its immutable candidate"
                )
            existing_record = connection.execute(
                "SELECT * FROM records WHERE candidate_id = ?",
                (record.candidate_id,),
            ).fetchone()
            if existing_record is not None:
                raise MemoryStoreConflictError(
                    "candidate already has a durable memory record"
                )
            if candidate.status is not expected_candidate_status:
                raise MemoryStoreConflictError()
            bundle_row = connection.execute(
                "SELECT * FROM source_bundles WHERE bundle_hash = ?",
                (record.source_bundle_hash,),
            ).fetchone()
            if bundle_row is None:
                raise MemoryStoreConflictError("record source bundle is missing")
            bundle = _validated_source_bundle_from_row(connection, bundle_row)
            if bundle.candidate_id != record.candidate_id:
                raise MemoryStoreConflictError("record source bundle belongs to another candidate")
            self._blob_info_from_connection(connection, bundle.blob_hash, validate_file=True)

            generation = self._bump_generation(
                connection, record.repository_key, "memory"
            )
            connection.execute(
                """
                UPDATE candidates SET current_status = ?, generation = ?
                WHERE candidate_id = ? AND current_status = ?
                """,
                (
                    CandidateStatus.APPROVED.value,
                    generation,
                    record.candidate_id,
                    expected_candidate_status.value,
                ),
            )
            if connection.execute("SELECT changes()").fetchone()[0] != 1:
                raise MemoryStoreConflictError()
            connection.execute(
                """
                INSERT INTO records(
                    memory_id, candidate_id, repository_key, source_bundle_hash,
                    model_json, body_hash, current_status, generation, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.memory_id,
                    record.candidate_id,
                    record.repository_key,
                    record.source_bundle_hash,
                    model_json,
                    body_hash,
                    record.status.value,
                    generation,
                    record.created_at,
                ),
            )
            event = self._append_event(
                connection,
                repository_key=record.repository_key,
                subject_type="candidate",
                subject_id=record.candidate_id,
                action="approve",
                actor_type=actor_type,
                actor_id=approver,
                reason_code=reason_code,
                reason=reason,
                previous_status=expected_candidate_status.value,
                new_status=CandidateStatus.APPROVED.value,
                request_id=checked_request,
                created_at=record.created_at,
                generation_kind="memory",
                generation=generation,
                event_id=record.approval_event_id,
            )
            result = WriteResult(
                operation=operation,
                subject_id=record.memory_id,
                event_id=event.event_id,
                generations=self._generations_from_connection(
                    connection, record.repository_key
                ),
                applied=True,
            )
            self._store_request_receipt(
                connection,
                request_id=checked_request,
                repository_key=record.repository_key,
                operation=operation,
                request_hash=request_hash,
                result=result,
                created_at=record.created_at,
            )
            return result

    put_record = approve_candidate
    create_record = approve_candidate

    def find_record(self, memory_id: str) -> Optional[DurableMemoryRecord]:
        checked_id = _stable_subject_id(memory_id, "MEM", "memory_id")
        with self._reader() as connection:
            row = connection.execute(
                "SELECT * FROM records WHERE memory_id = ?", (checked_id,)
            ).fetchone()
            if row is None:
                return None
            record = _validated_record_from_row(connection, row)
            bundle_row = connection.execute(
                "SELECT * FROM source_bundles WHERE bundle_hash = ?",
                (record.source_bundle_hash,),
            ).fetchone()
            if bundle_row is None:
                raise MemoryStoreCorruptionError("record source bundle is missing")
            bundle = _validated_source_bundle_from_row(connection, bundle_row)
            self._blob_info_from_connection(connection, bundle.blob_hash, validate_file=True)
            return record

    def get_record(self, memory_id: str) -> DurableMemoryRecord:
        record = self.find_record(memory_id)
        if record is None:
            raise MemoryStoreNotFoundError("durable memory record was not found")
        return record

    def count_records(self, repository_key: str) -> int:
        key = _repository_key(repository_key)
        with self._reader() as connection:
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM records WHERE repository_key = ?", (key,)
                ).fetchone()[0]
            )

    def list_records(
        self,
        repository_key: str,
        *,
        status: Optional[RecordStatus] = None,
    ) -> Tuple[DurableMemoryRecord, ...]:
        key = _repository_key(repository_key)
        if status is not None and not isinstance(status, RecordStatus):
            raise MemoryStoreValidationError("record status is invalid")
        query = "SELECT * FROM records WHERE repository_key = ?"
        parameters: List[Any] = [key]
        if status is not None:
            query += " AND current_status = ?"
            parameters.append(status.value)
        query += " ORDER BY memory_id"
        with self._reader() as connection:
            records = tuple(
                _validated_record_from_row(connection, row)
                for row in connection.execute(query, tuple(parameters))
            )
            for record in records:
                bundle_row = connection.execute(
                    "SELECT * FROM source_bundles WHERE bundle_hash = ?",
                    (record.source_bundle_hash,),
                ).fetchone()
                if bundle_row is None:
                    raise MemoryStoreCorruptionError("record source bundle is missing")
                bundle = _validated_source_bundle_from_row(connection, bundle_row)
                self._blob_info_from_connection(
                    connection, bundle.blob_hash, validate_file=True
                )
            return records

    def transition_record(
        self,
        memory_id: str,
        *,
        expected_status: RecordStatus,
        new_status: RecordStatus,
        action: str,
        actor_type: str,
        actor_id: str,
        reason_code: str,
        request_id: str,
        created_at: Optional[str] = None,
        reason: Optional[str] = None,
        expected_generation: Optional[int] = None,
    ) -> WriteResult:
        checked_id = _stable_subject_id(memory_id, "MEM", "memory_id")
        if not isinstance(expected_status, RecordStatus) or not isinstance(
            new_status, RecordStatus
        ):
            raise MemoryStoreValidationError("record status is invalid")
        operation = "transition_record"
        timestamp = _timestamp(created_at or _utc_now(), "created_at")
        checked_request = _request_id(request_id)
        request_hash = canonical_sha256(
            {
                "operation": operation,
                "memory_id": checked_id,
                "expected_status": expected_status.value,
                "new_status": new_status.value,
                "action": action,
                "actor_type": actor_type,
                "actor_id": actor_id,
                "reason_code": reason_code,
                "reason": reason,
                "expected_generation": expected_generation,
            }
        )
        with self._authority() as connection:
            row = connection.execute(
                "SELECT * FROM records WHERE memory_id = ?", (checked_id,)
            ).fetchone()
            if row is None:
                raise MemoryStoreNotFoundError("durable memory record was not found")
            record = _validated_record_from_row(connection, row)
            replay = self._request_receipt(
                connection,
                request_id=checked_request,
                repository_key=record.repository_key,
                operation=operation,
                request_hash=request_hash,
            )
            if replay is not None:
                return replay
            self._assert_generation(
                connection,
                record.repository_key,
                "memory",
                expected_generation,
            )
            if record.status is not expected_status:
                raise MemoryStoreConflictError()
            generation = self._bump_generation(
                connection, record.repository_key, "memory"
            )
            connection.execute(
                """
                UPDATE records SET current_status = ?, generation = ?
                WHERE memory_id = ? AND current_status = ?
                """,
                (new_status.value, generation, checked_id, expected_status.value),
            )
            if connection.execute("SELECT changes()").fetchone()[0] != 1:
                raise MemoryStoreConflictError()
            event = self._append_event(
                connection,
                repository_key=record.repository_key,
                subject_type="record",
                subject_id=checked_id,
                action=action,
                actor_type=actor_type,
                actor_id=actor_id,
                reason_code=reason_code,
                reason=reason,
                previous_status=expected_status.value,
                new_status=new_status.value,
                request_id=checked_request,
                created_at=timestamp,
                generation_kind="memory",
                generation=generation,
            )
            result = WriteResult(
                operation=operation,
                subject_id=checked_id,
                event_id=event.event_id,
                generations=self._generations_from_connection(
                    connection, record.repository_key
                ),
                applied=True,
            )
            self._store_request_receipt(
                connection,
                request_id=checked_request,
                repository_key=record.repository_key,
                operation=operation,
                request_hash=request_hash,
                result=result,
                created_at=timestamp,
            )
            return result

    def put_feedback(
        self,
        feedback: FeedbackRecord,
        *,
        request_id: Optional[str] = None,
        expected_generation: Optional[int] = None,
    ) -> WriteResult:
        if not isinstance(feedback, FeedbackRecord):
            raise MemoryStoreValidationError(
                "feedback must be a canonical FeedbackRecord"
            )
        operation = "put_feedback"
        checked_request = _request_id(
            request_id or stable_request_id(operation, feedback.feedback_id)
        )
        request_hash = canonical_sha256(
            {
                "operation": operation,
                "feedback": feedback.to_dict(),
                "expected_generation": expected_generation,
            }
        )
        model_json, body_hash = _model_storage(feedback)
        with self._authority() as connection:
            self._ensure_repository_rows(
                connection, feedback.repository_key, feedback.created_at
            )
            replay = self._request_receipt(
                connection,
                request_id=checked_request,
                repository_key=feedback.repository_key,
                operation=operation,
                request_hash=request_hash,
            )
            if replay is not None:
                return replay
            self._assert_generation(
                connection,
                feedback.repository_key,
                "feedback",
                expected_generation,
            )
            existing = connection.execute(
                "SELECT * FROM feedback WHERE feedback_id = ?",
                (feedback.feedback_id,),
            ).fetchone()
            if existing is not None:
                if existing["model_json"] != model_json or existing["body_hash"] != body_hash:
                    raise MemoryStoreConflictError(
                        "feedback ID already has different canonical content"
                    )
                result = WriteResult(
                    operation=operation,
                    subject_id=feedback.feedback_id,
                    event_id=None,
                    generations=self._generations_from_connection(
                        connection, feedback.repository_key
                    ),
                    applied=False,
                )
            else:
                generation = self._bump_generation(
                    connection, feedback.repository_key, "feedback"
                )
                connection.execute(
                    """
                    INSERT INTO feedback(
                        feedback_id, repository_key, review_id, finding_id,
                        model_json, body_hash, current_status, generation, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        feedback.feedback_id,
                        feedback.repository_key,
                        feedback.review_id,
                        feedback.finding_id,
                        model_json,
                        body_hash,
                        feedback.status.value,
                        generation,
                        feedback.created_at,
                    ),
                )
                event = self._append_event(
                    connection,
                    repository_key=feedback.repository_key,
                    subject_type="feedback",
                    subject_id=feedback.feedback_id,
                    action="feedback_recorded",
                    actor_type="human",
                    actor_id=feedback.actor,
                    reason_code=feedback.reason_code.value,
                    reason=feedback.reason,
                    previous_status=None,
                    new_status=feedback.status.value,
                    request_id=checked_request,
                    created_at=feedback.created_at,
                    generation_kind="feedback",
                    generation=generation,
                )
                result = WriteResult(
                    operation=operation,
                    subject_id=feedback.feedback_id,
                    event_id=event.event_id,
                    generations=self._generations_from_connection(
                        connection, feedback.repository_key
                    ),
                    applied=True,
                )
            self._store_request_receipt(
                connection,
                request_id=checked_request,
                repository_key=feedback.repository_key,
                operation=operation,
                request_hash=request_hash,
                result=result,
                created_at=feedback.created_at,
            )
            return result

    store_feedback = put_feedback

    def find_feedback(self, feedback_id: str) -> Optional[FeedbackRecord]:
        checked_id = _stable_subject_id(feedback_id, "FB", "feedback_id")
        with self._reader() as connection:
            row = connection.execute(
                "SELECT * FROM feedback WHERE feedback_id = ?", (checked_id,)
            ).fetchone()
            return None if row is None else _feedback_from_row(row)

    def get_feedback(self, feedback_id: str) -> FeedbackRecord:
        feedback = self.find_feedback(feedback_id)
        if feedback is None:
            raise MemoryStoreNotFoundError("feedback record was not found")
        return feedback

    def list_feedback(
        self,
        repository_key: str,
        *,
        status: Optional[FeedbackStatus] = None,
    ) -> Tuple[FeedbackRecord, ...]:
        key = _repository_key(repository_key)
        if status is not None and not isinstance(status, FeedbackStatus):
            raise MemoryStoreValidationError("feedback status is invalid")
        query = "SELECT * FROM feedback WHERE repository_key = ?"
        parameters: List[Any] = [key]
        if status is not None:
            query += " AND current_status = ?"
            parameters.append(status.value)
        query += " ORDER BY feedback_id"
        with self._reader() as connection:
            return tuple(
                _feedback_from_row(row)
                for row in connection.execute(query, tuple(parameters))
            )

    def transition_feedback(
        self,
        feedback_id: str,
        *,
        expected_status: FeedbackStatus,
        new_status: FeedbackStatus,
        action: str,
        actor_id: str,
        reason_code: str,
        request_id: str,
        created_at: Optional[str] = None,
        reason: Optional[str] = None,
        expected_generation: Optional[int] = None,
    ) -> WriteResult:
        checked_id = _stable_subject_id(feedback_id, "FB", "feedback_id")
        if not isinstance(expected_status, FeedbackStatus) or not isinstance(
            new_status, FeedbackStatus
        ):
            raise MemoryStoreValidationError("feedback status is invalid")
        operation = "transition_feedback"
        timestamp = _timestamp(created_at or _utc_now(), "created_at")
        checked_request = _request_id(request_id)
        request_hash = canonical_sha256(
            {
                "operation": operation,
                "feedback_id": checked_id,
                "expected_status": expected_status.value,
                "new_status": new_status.value,
                "action": action,
                "actor_id": actor_id,
                "reason_code": reason_code,
                "reason": reason,
                "expected_generation": expected_generation,
            }
        )
        with self._authority() as connection:
            row = connection.execute(
                "SELECT * FROM feedback WHERE feedback_id = ?", (checked_id,)
            ).fetchone()
            if row is None:
                raise MemoryStoreNotFoundError("feedback record was not found")
            feedback = _feedback_from_row(row)
            replay = self._request_receipt(
                connection,
                request_id=checked_request,
                repository_key=feedback.repository_key,
                operation=operation,
                request_hash=request_hash,
            )
            if replay is not None:
                return replay
            self._assert_generation(
                connection,
                feedback.repository_key,
                "feedback",
                expected_generation,
            )
            if feedback.status is not expected_status:
                raise MemoryStoreConflictError()
            generation = self._bump_generation(
                connection, feedback.repository_key, "feedback"
            )
            connection.execute(
                """
                UPDATE feedback SET current_status = ?, generation = ?
                WHERE feedback_id = ? AND current_status = ?
                """,
                (new_status.value, generation, checked_id, expected_status.value),
            )
            if connection.execute("SELECT changes()").fetchone()[0] != 1:
                raise MemoryStoreConflictError()
            event = self._append_event(
                connection,
                repository_key=feedback.repository_key,
                subject_type="feedback",
                subject_id=checked_id,
                action=action,
                actor_type="human",
                actor_id=actor_id,
                reason_code=reason_code,
                reason=reason,
                previous_status=expected_status.value,
                new_status=new_status.value,
                request_id=checked_request,
                created_at=timestamp,
                generation_kind="feedback",
                generation=generation,
            )
            result = WriteResult(
                operation=operation,
                subject_id=checked_id,
                event_id=event.event_id,
                generations=self._generations_from_connection(
                    connection, feedback.repository_key
                ),
                applied=True,
            )
            self._store_request_receipt(
                connection,
                request_id=checked_request,
                repository_key=feedback.repository_key,
                operation=operation,
                request_hash=request_hash,
                result=result,
                created_at=timestamp,
            )
            return result

    def put_knowledge_entry(
        self,
        entry: RepositoryKnowledgeEntry,
        *,
        request_id: Optional[str] = None,
        expected_generation: Optional[int] = None,
    ) -> WriteResult:
        if not isinstance(entry, RepositoryKnowledgeEntry):
            raise MemoryStoreValidationError(
                "knowledge entry must be a canonical RepositoryKnowledgeEntry"
            )
        operation = "put_knowledge_entry"
        checked_request = _request_id(
            request_id or stable_request_id(operation, entry.entry_id)
        )
        request_hash = canonical_sha256(
            {
                "operation": operation,
                "entry": entry.to_dict(),
                "expected_generation": expected_generation,
            }
        )
        model_json, body_hash = _model_storage(entry)
        repository_key = entry.key.repository_key
        with self._authority() as connection:
            self._ensure_repository_rows(connection, repository_key, entry.created_at)
            replay = self._request_receipt(
                connection,
                request_id=checked_request,
                repository_key=repository_key,
                operation=operation,
                request_hash=request_hash,
            )
            if replay is not None:
                return replay
            self._assert_generation(
                connection, repository_key, "knowledge", expected_generation
            )
            blob = self._blob_info_from_connection(
                connection, entry.blob_hash, validate_file=True
            )
            if blob.size_bytes != entry.size_bytes or blob.media_type != entry.content_type:
                raise MemoryStoreConflictError("knowledge blob metadata does not match")
            existing_key = connection.execute(
                "SELECT * FROM knowledge_entries WHERE key_hash = ?",
                (entry.key.key_hash,),
            ).fetchone()
            existing_id = connection.execute(
                "SELECT * FROM knowledge_entries WHERE entry_id = ?",
                (entry.entry_id,),
            ).fetchone()
            existing = existing_key or existing_id
            if existing is not None:
                if existing["entry_id"] != entry.entry_id:
                    raise MemoryStoreConflictError(
                        "knowledge key already maps to different immutable content"
                    )
                stored = _knowledge_from_row(connection, existing)
                if not _knowledge_identity_equal(stored, entry):
                    raise MemoryStoreConflictError(
                        "knowledge entry already has different canonical content"
                    )
                for review_id in entry.pinned_by_review_ids:
                    self._pin_knowledge_blob(
                        connection, entry, review_id, entry.created_at
                    )
                result = WriteResult(
                    operation=operation,
                    subject_id=entry.entry_id,
                    event_id=None,
                    generations=self._generations_from_connection(
                        connection, repository_key
                    ),
                    applied=False,
                )
            else:
                generation = self._bump_generation(
                    connection, repository_key, "knowledge"
                )
                connection.execute(
                    """
                    INSERT INTO knowledge_entries(
                        entry_id, repository_key, key_hash, blob_hash, model_json,
                        body_hash, generation, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entry.entry_id,
                        repository_key,
                        entry.key.key_hash,
                        entry.blob_hash,
                        model_json,
                        body_hash,
                        generation,
                        entry.created_at,
                    ),
                )
                for review_id in entry.pinned_by_review_ids:
                    self._pin_knowledge_blob(
                        connection, entry, review_id, entry.created_at
                    )
                event = self._append_event(
                    connection,
                    repository_key=repository_key,
                    subject_type="knowledge",
                    subject_id=entry.entry_id,
                    action="knowledge_stored",
                    actor_type="runtime",
                    actor_id=entry.key.analyzer_name,
                    reason_code="exact_revision_cache",
                    reason=None,
                    previous_status=None,
                    new_status="stored",
                    request_id=checked_request,
                    created_at=entry.created_at,
                    generation_kind="knowledge",
                    generation=generation,
                )
                result = WriteResult(
                    operation=operation,
                    subject_id=entry.entry_id,
                    event_id=event.event_id,
                    generations=self._generations_from_connection(
                        connection, repository_key
                    ),
                    applied=True,
                )
            self._store_request_receipt(
                connection,
                request_id=checked_request,
                repository_key=repository_key,
                operation=operation,
                request_hash=request_hash,
                result=result,
                created_at=entry.created_at,
            )
            return result

    store_knowledge_entry = put_knowledge_entry

    @staticmethod
    def _pin_knowledge_blob(
        connection: sqlite3.Connection,
        entry: RepositoryKnowledgeEntry,
        review_id: str,
        created_at: str,
    ) -> None:
        checked_review = _required_text(review_id, "review_id", 512)
        connection.execute(
            """
            INSERT OR IGNORE INTO blob_pins(blob_hash, pin_type, pin_id, created_at)
            VALUES (?, 'knowledge', ?, ?)
            """,
            (entry.blob_hash, "%s:%s" % (entry.entry_id, checked_review), created_at),
        )

    def find_knowledge_entry(
        self, entry_id: str
    ) -> Optional[RepositoryKnowledgeEntry]:
        checked_id = _stable_subject_id(entry_id, "RKE", "entry_id")
        with self._reader() as connection:
            row = connection.execute(
                "SELECT * FROM knowledge_entries WHERE entry_id = ?", (checked_id,)
            ).fetchone()
            if row is None:
                return None
            entry = _knowledge_from_row(connection, row)
            blob = self._blob_info_from_connection(
                connection, entry.blob_hash, validate_file=True
            )
            if blob.size_bytes != entry.size_bytes or blob.media_type != entry.content_type:
                raise MemoryStoreCorruptionError("knowledge blob metadata is invalid")
            return entry

    def get_knowledge_entry(self, entry_id: str) -> RepositoryKnowledgeEntry:
        entry = self.find_knowledge_entry(entry_id)
        if entry is None:
            raise MemoryStoreNotFoundError("repository knowledge entry was not found")
        return entry

    def find_knowledge_by_key(
        self, key: RepositoryKnowledgeKey
    ) -> Optional[RepositoryKnowledgeEntry]:
        if not isinstance(key, RepositoryKnowledgeKey):
            raise MemoryStoreValidationError(
                "knowledge key must be a canonical RepositoryKnowledgeKey"
            )
        with self._reader() as connection:
            row = connection.execute(
                "SELECT * FROM knowledge_entries WHERE key_hash = ?", (key.key_hash,)
            ).fetchone()
            if row is None:
                return None
            entry = _knowledge_from_row(connection, row)
            if entry.key != key:
                raise MemoryStoreCorruptionError("knowledge key projection is invalid")
            self._blob_info_from_connection(connection, entry.blob_hash, validate_file=True)
            return entry

    def list_knowledge_entries(
        self, repository_key: str
    ) -> Tuple[RepositoryKnowledgeEntry, ...]:
        key = _repository_key(repository_key)
        with self._reader() as connection:
            entries = tuple(
                _knowledge_from_row(connection, row)
                for row in connection.execute(
                    """
                    SELECT * FROM knowledge_entries
                    WHERE repository_key = ? ORDER BY entry_id
                    """,
                    (key,),
                )
            )
            for entry in entries:
                self._blob_info_from_connection(
                    connection, entry.blob_hash, validate_file=True
                )
            return entries

    def pin_knowledge_entry(
        self,
        entry_id: str,
        review_id: str,
        *,
        created_at: Optional[str] = None,
    ) -> bool:
        checked_id = _stable_subject_id(entry_id, "RKE", "entry_id")
        timestamp = _timestamp(created_at or _utc_now(), "created_at")
        with self._authority() as connection:
            row = connection.execute(
                "SELECT * FROM knowledge_entries WHERE entry_id = ?", (checked_id,)
            ).fetchone()
            if row is None:
                raise MemoryStoreNotFoundError("repository knowledge entry was not found")
            entry = _knowledge_from_row(connection, row)
            self._pin_knowledge_blob(connection, entry, review_id, timestamp)
            return connection.execute("SELECT changes()").fetchone()[0] == 1

    def unpin_knowledge_entry(self, entry_id: str, review_id: str) -> bool:
        checked_id = _stable_subject_id(entry_id, "RKE", "entry_id")
        checked_review = _required_text(review_id, "review_id", 512)
        with self._authority() as connection:
            row = connection.execute(
                "SELECT blob_hash FROM knowledge_entries WHERE entry_id = ?",
                (checked_id,),
            ).fetchone()
            if row is None:
                raise MemoryStoreNotFoundError("repository knowledge entry was not found")
            connection.execute(
                """
                DELETE FROM blob_pins
                WHERE blob_hash = ? AND pin_type = 'knowledge' AND pin_id = ?
                """,
                (row["blob_hash"], "%s:%s" % (checked_id, checked_review)),
            )
            return connection.execute("SELECT changes()").fetchone()[0] == 1

    def delete_knowledge_entry(
        self,
        entry_id: str,
        *,
        request_id: str,
        expected_generation: Optional[int] = None,
        created_at: Optional[str] = None,
    ) -> WriteResult:
        checked_id = _stable_subject_id(entry_id, "RKE", "entry_id")
        checked_request = _request_id(request_id)
        timestamp = _timestamp(created_at or _utc_now(), "created_at")
        operation = "delete_knowledge_entry"
        request_hash = canonical_sha256(
            {
                "operation": operation,
                "entry_id": checked_id,
                "expected_generation": expected_generation,
            }
        )
        with self._authority() as connection:
            replay = self._request_receipt(
                connection,
                request_id=checked_request,
                repository_key=None,
                operation=operation,
                request_hash=request_hash,
            )
            if replay is not None:
                return replay
            row = connection.execute(
                "SELECT * FROM knowledge_entries WHERE entry_id = ?", (checked_id,)
            ).fetchone()
            if row is None:
                raise MemoryStoreNotFoundError("repository knowledge entry was not found")
            entry = _knowledge_from_row(connection, row)
            repository_key = entry.key.repository_key
            self._assert_generation(
                connection, repository_key, "knowledge", expected_generation
            )
            pin_count = connection.execute(
                """
                SELECT COUNT(*) FROM blob_pins
                WHERE blob_hash = ? AND pin_type = 'knowledge'
                  AND pin_id LIKE ? ESCAPE '\\'
                """,
                (entry.blob_hash, _escape_like(checked_id) + ":%"),
            ).fetchone()[0]
            if pin_count:
                raise MemoryStoreConflictError("pinned knowledge entry cannot be deleted")
            generation = self._bump_generation(connection, repository_key, "knowledge")
            connection.execute(
                "DELETE FROM knowledge_entries WHERE entry_id = ?", (checked_id,)
            )
            event = self._append_event(
                connection,
                repository_key=repository_key,
                subject_type="knowledge",
                subject_id=checked_id,
                action="knowledge_deleted",
                actor_type="runtime",
                actor_id="repository_cache",
                reason_code="cache_gc",
                reason=None,
                previous_status="stored",
                new_status="deleted",
                request_id=checked_request,
                created_at=timestamp,
                generation_kind="knowledge",
                generation=generation,
            )
            result = WriteResult(
                operation=operation,
                subject_id=checked_id,
                event_id=event.event_id,
                generations=self._generations_from_connection(connection, repository_key),
                applied=True,
            )
            self._store_request_receipt(
                connection,
                request_id=checked_request,
                repository_key=repository_key,
                operation=operation,
                request_hash=request_hash,
                result=result,
                created_at=timestamp,
            )
            return result

    def gc_blobs(
        self,
        *,
        dry_run: bool = True,
        grace_seconds: float = 0,
        now: Optional[float] = None,
    ) -> BlobGCResult:
        with _exclusive_file_lock(self._blob_lock_path, self._busy_timeout_ms):
            return self._gc_blobs_locked(
                dry_run=dry_run,
                grace_seconds=grace_seconds,
                now=now,
            )

    def _gc_blobs_locked(
        self,
        *,
        dry_run: bool = True,
        grace_seconds: float = 0,
        now: Optional[float] = None,
    ) -> BlobGCResult:
        if type(dry_run) is not bool:
            raise MemoryStoreValidationError("dry_run must be a boolean")
        if not isinstance(grace_seconds, (int, float)) or grace_seconds < 0:
            raise MemoryStoreValidationError("GC grace period must be non-negative")
        cutoff = (time.time() if now is None else float(now)) - float(grace_seconds)
        with self._reader() as connection:
            rows = connection.execute(
                """
                SELECT b.blob_hash, b.size_bytes
                FROM blobs AS b
                WHERE NOT EXISTS (
                    SELECT 1 FROM source_bundles AS s WHERE s.blob_hash = b.blob_hash
                )
                  AND NOT EXISTS (
                    SELECT 1 FROM knowledge_entries AS k WHERE k.blob_hash = b.blob_hash
                )
                  AND NOT EXISTS (
                    SELECT 1 FROM blob_pins AS p WHERE p.blob_hash = b.blob_hash
                )
                ORDER BY b.blob_hash
                """
            ).fetchall()
            known_hashes = {
                str(row["blob_hash"])
                for row in connection.execute("SELECT blob_hash FROM blobs")
            }

        candidates: List[Tuple[str, int]] = []
        for row in rows:
            digest = str(row["blob_hash"])
            path = self.blob_path(digest)
            try:
                modified = path.stat().st_mtime if path.exists() else 0.0
            except OSError:
                modified = 0.0
            if modified <= cutoff:
                candidates.append((digest, int(row["size_bytes"])))

        orphan_paths: List[Path] = []
        if self.blob_root.exists():
            try:
                for prefix in self.blob_root.iterdir():
                    if prefix.is_symlink() or not prefix.is_dir():
                        continue
                    for path in prefix.iterdir():
                        if path.is_symlink() or not path.is_file():
                            continue
                        name = path.name
                        if (
                            _SHA256_PATTERN.fullmatch(name) is None
                            or prefix.name != name[:2]
                            or name in known_hashes
                        ):
                            continue
                        if path.stat().st_mtime <= cutoff:
                            orphan_paths.append(path)
                if self._blob_temp_root.exists():
                    for path in self._blob_temp_root.iterdir():
                        if path.is_symlink() or not path.is_file():
                            continue
                        if path.stat().st_mtime <= cutoff:
                            orphan_paths.append(path)
            except OSError:
                raise MemoryStoreUnavailableError("memory blob GC scan failed") from None

        candidate_hashes = tuple(digest for digest, _ in candidates)
        if dry_run:
            return BlobGCResult(
                candidate_hashes=candidate_hashes,
                deleted_hashes=(),
                orphan_paths=tuple(sorted(str(path) for path in orphan_paths)),
                deleted_orphan_paths=(),
                reclaimed_bytes=sum(size for _, size in candidates),
                dry_run=True,
            )

        deleted: List[str] = []
        reclaimed = 0
        with self._authority() as connection:
            for digest, size in candidates:
                eligible = connection.execute(
                    """
                    SELECT 1 FROM blobs AS b
                    WHERE b.blob_hash = ?
                      AND NOT EXISTS (SELECT 1 FROM source_bundles s WHERE s.blob_hash = b.blob_hash)
                      AND NOT EXISTS (SELECT 1 FROM knowledge_entries k WHERE k.blob_hash = b.blob_hash)
                      AND NOT EXISTS (SELECT 1 FROM blob_pins p WHERE p.blob_hash = b.blob_hash)
                    """,
                    (digest,),
                ).fetchone()
                if eligible is None:
                    continue
                connection.execute("DELETE FROM blobs WHERE blob_hash = ?", (digest,))
                deleted.append(digest)
                reclaimed += size

        deleted_orphans: List[str] = []
        for digest in deleted:
            path = self.blob_path(digest)
            try:
                path.unlink(missing_ok=True)
                deleted_orphans.append(str(path))
            except OSError:
                # A committed DB deletion followed by a failed unlink is a safe orphan;
                # the next GC pass will retry it.
                pass
        for path in sorted(orphan_paths, key=lambda item: str(item)):
            try:
                size = path.stat().st_size
                path.unlink(missing_ok=True)
                reclaimed += size
                deleted_orphans.append(str(path))
            except OSError:
                pass
        return BlobGCResult(
            candidate_hashes=candidate_hashes,
            deleted_hashes=tuple(deleted),
            orphan_paths=tuple(sorted(str(path) for path in orphan_paths)),
            deleted_orphan_paths=tuple(sorted(set(deleted_orphans))),
            reclaimed_bytes=reclaimed,
            dry_run=False,
        )

    def read_view(self, repository_key: str) -> MemoryStoreReadView:
        """Capture validated projections and generations in one read snapshot."""

        key = _repository_key(repository_key)
        connection = self._connect(read_only=True)
        try:
            connection.execute("BEGIN")
            self._verify_event_chain_connection(connection, key)
            self._verify_projection_connection(connection, key)
            generations = self._generations_from_connection(connection, key)
            records = tuple(
                _validated_record_from_row(connection, row)
                for row in connection.execute(
                    "SELECT * FROM records WHERE repository_key = ? ORDER BY memory_id",
                    (key,),
                )
            )
            feedback = tuple(
                _feedback_from_row(row)
                for row in connection.execute(
                    "SELECT * FROM feedback WHERE repository_key = ? ORDER BY feedback_id",
                    (key,),
                )
            )
            knowledge = tuple(
                _knowledge_from_row(connection, row)
                for row in connection.execute(
                    """
                    SELECT * FROM knowledge_entries
                    WHERE repository_key = ? ORDER BY entry_id
                    """,
                    (key,),
                )
            )
            for record in records:
                bundle_row = connection.execute(
                    "SELECT * FROM source_bundles WHERE bundle_hash = ?",
                    (record.source_bundle_hash,),
                ).fetchone()
                if bundle_row is None:
                    raise MemoryStoreCorruptionError("record source bundle is missing")
                bundle = _validated_source_bundle_from_row(connection, bundle_row)
                self._blob_info_from_connection(
                    connection, bundle.blob_hash, validate_file=True
                )
            for entry in knowledge:
                self._blob_info_from_connection(
                    connection, entry.blob_hash, validate_file=True
                )
            connection.commit()
            return MemoryStoreReadView(
                generations=generations,
                records=records,
                feedback=feedback,
                knowledge_entries=knowledge,
            )
        except MemoryStoreError:
            if connection.in_transaction:
                connection.rollback()
            raise
        except sqlite3.Error as error:
            if connection.in_transaction:
                connection.rollback()
            raise _translate_sqlite_error(error) from None
        finally:
            connection.close()

    validated_read_view = read_view

    def validate_integrity(
        self,
        *,
        validate_blob_files: bool = True,
    ) -> IntegrityReport:
        if type(validate_blob_files) is not bool:
            raise MemoryStoreValidationError("validate_blob_files must be a boolean")
        with self._reader() as connection:
            _validate_schema_connection(connection)
            integrity = connection.execute("PRAGMA integrity_check").fetchall()
            if len(integrity) != 1 or str(integrity[0][0]).casefold() != "ok":
                raise MemoryStoreCorruptionError("SQLite integrity validation failed")
            if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
                raise MemoryStoreCorruptionError("memory foreign-key validation failed")

            repositories = connection.execute(
                "SELECT * FROM repositories ORDER BY repository_key"
            ).fetchall()
            for repository in repositories:
                try:
                    key = _repository_key(repository["repository_key"])
                    if connection.execute(
                        "SELECT 1 FROM generations WHERE repository_key = ?", (key,)
                    ).fetchone() is None:
                        raise ValueError
                    self._generations_from_connection(connection, key)
                    if connection.execute(
                        "SELECT 1 FROM event_chain_heads WHERE repository_key = ?", (key,)
                    ).fetchone() is None:
                        raise ValueError
                    identity_json = repository["identity_json"]
                    if identity_json is not None:
                        payload = json.loads(identity_json)
                        if canonical_json(payload) != identity_json:
                            raise ValueError
                        descriptor = RepositoryIdentityDescriptor(
                            repository_key=payload["repository_key"],
                            canonical_path=payload["canonical_path"],
                            git_common_dir=payload["git_common_dir"],
                            origin_url=payload["origin_url"],
                            schema=payload["schema"],
                        )
                        if descriptor.repository_key != key:
                            raise ValueError
                except (json.JSONDecodeError, KeyError, TypeError, ValueError, MemoryStoreError):
                    raise MemoryStoreCorruptionError(
                        "repository identity metadata is invalid"
                    ) from None
            candidates = connection.execute("SELECT * FROM candidates").fetchall()
            bundles = connection.execute("SELECT * FROM source_bundles").fetchall()
            records = connection.execute("SELECT * FROM records").fetchall()
            feedback_rows = connection.execute("SELECT * FROM feedback").fetchall()
            knowledge_rows = connection.execute("SELECT * FROM knowledge_entries").fetchall()
            blobs = connection.execute("SELECT * FROM blobs").fetchall()
            for row in candidates:
                _candidate_from_row(row)
            for row in bundles:
                bundle = _validated_source_bundle_from_row(connection, row)
                info = self._blob_info_from_connection(
                    connection, bundle.blob_hash, validate_file=validate_blob_files
                )
                if info.size_bytes != bundle.size_bytes or info.media_type != bundle.media_type:
                    raise MemoryStoreCorruptionError("source bundle blob metadata is invalid")
            for row in records:
                record = _validated_record_from_row(connection, row)
                bundle_row = connection.execute(
                    "SELECT * FROM source_bundles WHERE bundle_hash = ?",
                    (record.source_bundle_hash,),
                ).fetchone()
                if bundle_row is None:
                    raise MemoryStoreCorruptionError("record source bundle is missing")
            for row in feedback_rows:
                _feedback_from_row(row)
            for row in knowledge_rows:
                entry = _knowledge_from_row(connection, row)
                info = self._blob_info_from_connection(
                    connection, entry.blob_hash, validate_file=validate_blob_files
                )
                if info.size_bytes != entry.size_bytes or info.media_type != entry.content_type:
                    raise MemoryStoreCorruptionError("knowledge blob metadata is invalid")
            for row in blobs:
                self._blob_info_from_connection(
                    connection,
                    str(row["blob_hash"]),
                    validate_file=validate_blob_files,
                )
            for row in connection.execute(
                "SELECT * FROM outbox_receipts ORDER BY request_id"
            ):
                receipt = _receipt_from_row(row)
                if receipt["event_id"] is not None:
                    event = connection.execute(
                        "SELECT request_id FROM events WHERE event_id = ?",
                        (receipt["event_id"],),
                    ).fetchone()
                    if event is None or event["request_id"] != receipt["request_id"]:
                        raise MemoryStoreCorruptionError(
                            "memory request receipt event is invalid"
                        )
            event_count = 0
            for row in connection.execute(
                "SELECT repository_key FROM event_chain_heads ORDER BY repository_key"
            ):
                repository_key = str(row["repository_key"])
                event_count += self._verify_event_chain_connection(
                    connection, repository_key
                )
                self._verify_projection_connection(connection, repository_key)
            return IntegrityReport(
                repository_count=len(repositories),
                event_count=event_count,
                blob_count=len(blobs),
                candidate_count=len(candidates),
                record_count=len(records),
                feedback_count=len(feedback_rows),
                knowledge_count=len(knowledge_rows),
            )

    def checkpoint(self, mode: str = "PASSIVE") -> Tuple[int, int, int]:
        checked_mode = str(mode).upper()
        if checked_mode not in {"PASSIVE", "FULL", "RESTART", "TRUNCATE"}:
            raise MemoryStoreValidationError("unsupported WAL checkpoint mode")
        with self._maintenance_connection() as connection:
            row = connection.execute(
                "PRAGMA wal_checkpoint(%s)" % checked_mode
            ).fetchone()
            return int(row[0]), int(row[1]), int(row[2])

    def build_export_manifest(
        self,
        *,
        redact: bool = True,
        created_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        if type(redact) is not bool:
            raise MemoryStoreValidationError("redact must be a boolean")
        timestamp = _timestamp(created_at or _utc_now(), "created_at")
        self.validate_integrity()
        with self._reader() as connection:
            connection.execute("BEGIN")
            for row in connection.execute(
                "SELECT repository_key FROM event_chain_heads ORDER BY repository_key"
            ):
                self._verify_event_chain_connection(
                    connection, str(row["repository_key"])
                )
            repositories: List[Dict[str, Any]] = []
            for row in connection.execute(
                "SELECT * FROM repositories ORDER BY repository_key"
            ):
                repositories.append(
                    {
                        "repository_key": str(row["repository_key"]),
                        "identity_schema": row["identity_schema"],
                        "canonical_path": None if redact else row["canonical_path"],
                        "git_common_dir": None if redact else row["git_common_dir"],
                        "origin_url": row["origin_url"],
                        "created_at": str(row["created_at"]),
                    }
                )
            generations = [
                {
                    "repository_key": str(row["repository_key"]),
                    "generations": self._generations_from_connection(
                        connection, str(row["repository_key"])
                    ).to_dict(),
                }
                for row in connection.execute(
                    "SELECT repository_key FROM generations ORDER BY repository_key"
                )
            ]
            blobs = [
                {
                    "blob_hash": str(row["blob_hash"]),
                    "size_bytes": int(row["size_bytes"]),
                    "media_type": str(row["media_type"]),
                    "created_at": str(row["created_at"]),
                }
                for row in connection.execute("SELECT * FROM blobs ORDER BY blob_hash")
            ]
            pins = [
                {
                    "blob_hash": str(row["blob_hash"]),
                    "pin_type": str(row["pin_type"]),
                    "pin_id": str(row["pin_id"]),
                    "created_at": str(row["created_at"]),
                }
                for row in connection.execute(
                    "SELECT * FROM blob_pins ORDER BY blob_hash, pin_type, pin_id"
                )
            ]
            candidates = [
                _export_model_row(
                    row,
                    id_column="candidate_id",
                    redact=redact,
                )
                for row in connection.execute(
                    "SELECT * FROM candidates ORDER BY candidate_id"
                )
            ]
            source_bundles = [
                _export_model_row(row, id_column="bundle_hash", redact=redact)
                for row in connection.execute(
                    "SELECT * FROM source_bundles ORDER BY bundle_hash"
                )
            ]
            records = [
                _export_model_row(
                    row,
                    id_column="memory_id",
                    redact=redact,
                )
                for row in connection.execute("SELECT * FROM records ORDER BY memory_id")
            ]
            feedback = [
                _export_model_row(
                    row,
                    id_column="feedback_id",
                    redact=redact,
                )
                for row in connection.execute("SELECT * FROM feedback ORDER BY feedback_id")
            ]
            knowledge = [
                _export_model_row(row, id_column="entry_id", redact=False)
                for row in connection.execute(
                    "SELECT * FROM knowledge_entries ORDER BY entry_id"
                )
            ]
            events: List[Dict[str, Any]] = []
            for row in connection.execute("SELECT * FROM events ORDER BY sequence"):
                event = _event_from_row(row)
                payload = event.to_dict()
                reason_hash = None
                reason_redacted = False
                if redact and payload["reason"] is not None:
                    reason_hash = hashlib.sha256(
                        str(payload["reason"]).encode("utf-8")
                    ).hexdigest()
                    payload["reason"] = None
                    reason_redacted = True
                events.append(
                    {
                        "event": payload,
                        "reason_hash": reason_hash,
                        "reason_redacted": reason_redacted,
                    }
                )
            outbox_receipts = [
                _receipt_export_from_row(row)
                for row in connection.execute(
                    "SELECT * FROM outbox_receipts ORDER BY request_id"
                )
            ]
            connection.commit()

        body: Dict[str, Any] = {
            "schema_name": EXPORT_SCHEMA_NAME,
            "schema_version": EXPORT_SCHEMA_VERSION,
            "store_schema_name": STORE_SCHEMA_NAME,
            "store_schema_version": STORE_SCHEMA_VERSION,
            "created_at": timestamp,
            "redacted": redact,
            "restorable": not redact,
            "repositories": repositories,
            "generations": generations,
            "blobs": blobs,
            "blob_pins": pins,
            "candidates": candidates,
            "source_bundles": source_bundles,
            "records": records,
            "feedback": feedback,
            "knowledge_entries": knowledge,
            "events": events,
            "outbox_receipts": outbox_receipts,
        }
        body["manifest_hash"] = canonical_sha256(body)
        return body

    def export_manifest(
        self,
        path: PathInput,
        *,
        redact: bool = True,
        created_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        manifest = self.build_export_manifest(redact=redact, created_at=created_at)
        destination = Path(path).resolve(strict=False)
        _atomic_write_bytes(destination, canonical_json(manifest).encode("utf-8"))
        return manifest

    def export_to_directory(
        self,
        directory: PathInput,
        *,
        redact: bool = True,
        include_blobs: bool = False,
        created_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        destination = Path(directory).resolve(strict=False)
        try:
            destination.mkdir(parents=True, exist_ok=True)
        except OSError:
            raise MemoryStoreUnavailableError("memory export destination is unavailable") from None
        manifest = self.export_manifest(
            destination / "manifest.json", redact=redact, created_at=created_at
        )
        if include_blobs:
            for blob in manifest["blobs"]:
                digest = blob["blob_hash"]
                info = self.get_blob_info(digest, validate=True)
                target = destination / "blobs" / "sha256" / digest[:2] / digest
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_name(".%s.%s.tmp" % (digest, uuid.uuid4().hex))
                    shutil.copyfile(info.path, str(temporary))
                    self._validate_blob_file_values(
                        temporary, digest, int(blob["size_bytes"])
                    )
                    os.replace(str(temporary), str(target))
                    _fsync_directory(target.parent)
                except MemoryStoreError:
                    raise
                except OSError:
                    raise MemoryStoreUnavailableError("memory blob export failed") from None
                finally:
                    try:
                        temporary.unlink(missing_ok=True)
                    except (OSError, UnboundLocalError):
                        pass
        return manifest

    def validate_import_manifest(
        self, manifest: Mapping[str, Any]
    ) -> ImportPlan:
        expected_fields = {
            "schema_name",
            "schema_version",
            "store_schema_name",
            "store_schema_version",
            "created_at",
            "redacted",
            "restorable",
            "repositories",
            "generations",
            "blobs",
            "blob_pins",
            "candidates",
            "source_bundles",
            "records",
            "feedback",
            "knowledge_entries",
            "events",
            "outbox_receipts",
            "manifest_hash",
        }
        if not isinstance(manifest, Mapping) or set(manifest) != expected_fields:
            raise MemoryStoreValidationError("memory import manifest fields are invalid")
        if (
            manifest["schema_name"] != EXPORT_SCHEMA_NAME
            or manifest["schema_version"] != EXPORT_SCHEMA_VERSION
            or manifest["store_schema_name"] != STORE_SCHEMA_NAME
            or manifest["store_schema_version"] != STORE_SCHEMA_VERSION
        ):
            raise MemoryStoreSchemaError("memory import schema is unsupported")
        _timestamp(manifest["created_at"], "manifest created_at")
        redacted = _required_bool(manifest["redacted"], "redacted")
        restorable = _required_bool(manifest["restorable"], "restorable")
        if restorable == redacted:
            raise MemoryStoreValidationError("memory import restoration flags are invalid")
        supplied_hash = _digest(manifest["manifest_hash"], "manifest_hash")
        body = {key: value for key, value in manifest.items() if key != "manifest_hash"}
        if not hmac.compare_digest(supplied_hash, canonical_sha256(body)):
            raise MemoryStoreValidationError("memory import manifest hash is invalid")
        list_fields = (
            "repositories",
            "generations",
            "blobs",
            "blob_pins",
            "candidates",
            "source_bundles",
            "records",
            "feedback",
            "knowledge_entries",
            "events",
            "outbox_receipts",
        )
        for field in list_fields:
            if not isinstance(manifest[field], list):
                raise MemoryStoreValidationError("memory import manifest collection is invalid")

        repository_keys: List[str] = []
        for repository in manifest["repositories"]:
            expected = {
                "repository_key",
                "identity_schema",
                "canonical_path",
                "git_common_dir",
                "origin_url",
                "created_at",
            }
            if not isinstance(repository, Mapping) or set(repository) != expected:
                raise MemoryStoreValidationError("memory import repository is invalid")
            repository_key = _repository_key(repository["repository_key"])
            repository_keys.append(repository_key)
            _timestamp(repository["created_at"], "repository created_at")
            if redacted and (
                repository["canonical_path"] is not None
                or repository["git_common_dir"] is not None
            ):
                raise MemoryStoreValidationError("redacted repository paths are not empty")
            identity_schema = repository["identity_schema"]
            if identity_schema is None:
                if any(
                    repository[field] is not None
                    for field in (
                        "canonical_path",
                        "git_common_dir",
                        "origin_url",
                    )
                ):
                    raise MemoryStoreValidationError(
                        "memory import repository identity is incomplete"
                    )
            else:
                audit_path = str(self.namespace_path)
                try:
                    RepositoryIdentityDescriptor(
                        repository_key=repository_key,
                        canonical_path=(
                            audit_path
                            if repository["canonical_path"] is None
                            else repository["canonical_path"]
                        ),
                        git_common_dir=(
                            audit_path
                            if repository["git_common_dir"] is None
                            else repository["git_common_dir"]
                        ),
                        origin_url=repository["origin_url"],
                        schema=identity_schema,
                    )
                except ValueError:
                    raise MemoryStoreValidationError(
                        "memory import repository identity is invalid"
                    ) from None
        if repository_keys != sorted(set(repository_keys)):
            raise MemoryStoreValidationError("memory import repositories are not canonical")

        generation_keys: List[str] = []
        for item in manifest["generations"]:
            if not isinstance(item, Mapping) or set(item) != {
                "repository_key",
                "generations",
            }:
                raise MemoryStoreValidationError("memory import generations are invalid")
            generation_keys.append(_repository_key(item["repository_key"]))
            GenerationMetadata.from_dict(item["generations"])
        if generation_keys != sorted(set(generation_keys)):
            raise MemoryStoreValidationError("memory import generations are not canonical")

        blob_hashes: List[str] = []
        for blob in manifest["blobs"]:
            if not isinstance(blob, Mapping) or set(blob) != {
                "blob_hash",
                "size_bytes",
                "media_type",
                "created_at",
            }:
                raise MemoryStoreValidationError("memory import blob metadata is invalid")
            blob_hashes.append(_digest(blob["blob_hash"], "blob_hash"))
            if type(blob["size_bytes"]) is not int or blob["size_bytes"] < 0:
                raise MemoryStoreValidationError("memory import blob size is invalid")
            _media_type(blob["media_type"])
            _timestamp(blob["created_at"], "blob created_at")
        if blob_hashes != sorted(set(blob_hashes)):
            raise MemoryStoreValidationError("memory import blobs are not canonical")

        for pin in manifest["blob_pins"]:
            if not isinstance(pin, Mapping) or set(pin) != {
                "blob_hash",
                "pin_type",
                "pin_id",
                "created_at",
            }:
                raise MemoryStoreValidationError("memory import blob pin is invalid")
            if _digest(pin["blob_hash"], "blob_hash") not in blob_hashes:
                raise MemoryStoreValidationError("memory import blob pin target is missing")
            _pin_type(pin["pin_type"])
            _required_text(pin["pin_id"], "pin_id", 512)
            _timestamp(pin["created_at"], "pin created_at")

        _validate_export_model_rows(
            manifest["candidates"], MemoryCandidate, "candidate_id", redacted
        )
        _validate_export_model_rows(
            manifest["source_bundles"],
            SourceBundleDescriptor,
            "bundle_hash",
            redacted,
        )
        _validate_export_model_rows(
            manifest["records"], DurableMemoryRecord, "memory_id", redacted
        )
        _validate_export_model_rows(
            manifest["feedback"], FeedbackRecord, "feedback_id", redacted
        )
        _validate_export_model_rows(
            manifest["knowledge_entries"], RepositoryKnowledgeEntry, "entry_id", False
        )
        _validate_export_events(manifest["events"], redacted)
        _validate_export_receipts(manifest["outbox_receipts"])
        _validate_manifest_relationships(manifest)

        return ImportPlan(
            repository_keys=tuple(repository_keys),
            candidate_count=len(manifest["candidates"]),
            record_count=len(manifest["records"]),
            feedback_count=len(manifest["feedback"]),
            knowledge_count=len(manifest["knowledge_entries"]),
            source_bundle_count=len(manifest["source_bundles"]),
            event_count=len(manifest["events"]),
            blob_count=len(manifest["blobs"]),
            outbox_receipt_count=len(manifest["outbox_receipts"]),
            redacted=redacted,
            restorable=restorable,
        )

    def import_manifest(
        self,
        manifest_or_path: Union[Mapping[str, Any], PathInput],
        *,
        dry_run: bool = True,
        blob_source_root: Optional[PathInput] = None,
    ) -> ImportPlan:
        if type(dry_run) is not bool:
            raise MemoryStoreValidationError("dry_run must be a boolean")
        manifest = _load_manifest(manifest_or_path)
        plan = self.validate_import_manifest(manifest)
        self._validate_import_target_identity(plan.repository_keys)
        if dry_run:
            return plan
        if not plan.restorable:
            raise MemoryStoreValidationError("redacted memory exports cannot be applied")
        self._apply_import_manifest(manifest, blob_source_root=blob_source_root)
        return replace(plan, applied=True)

    def _validate_import_target_identity(
        self,
        imported_repository_keys: Sequence[str],
    ) -> None:
        imported = set(imported_repository_keys)
        with self._reader() as connection:
            existing = {
                str(row["repository_key"])
                for row in connection.execute(
                    "SELECT repository_key FROM repositories"
                )
            }
        if existing and existing != imported:
            raise MemoryStoreConflictError(
                "memory import repository identity requires explicit relink"
            )

    def _apply_import_manifest(
        self,
        manifest: Mapping[str, Any],
        *,
        blob_source_root: Optional[PathInput],
    ) -> None:
        source_root = (
            None
            if blob_source_root is None
            else Path(blob_source_root).resolve(strict=False)
        )
        for blob in manifest["blobs"]:
            digest = str(blob["blob_hash"])
            try:
                existing = self.get_blob_info(digest, validate=True)
            except MemoryStoreCorruptionError:
                existing = None
            if existing is not None:
                if (
                    existing.size_bytes != blob["size_bytes"]
                    or existing.media_type != blob["media_type"]
                ):
                    raise MemoryStoreConflictError("import blob metadata conflicts")
                continue
            if source_root is None:
                raise MemoryStoreValidationError(
                    "restorable import requires validated blob content"
                )
            candidates = (
                source_root / "blobs" / "sha256" / digest[:2] / digest,
                source_root / "sha256" / digest[:2] / digest,
                source_root / digest[:2] / digest,
            )
            source = next((path for path in candidates if path.is_file()), None)
            if source is None:
                raise MemoryStoreValidationError("import blob content is missing")
            try:
                raw = source.read_bytes()
            except OSError:
                raise MemoryStoreValidationError("import blob content is unreadable") from None
            self.put_blob(
                raw,
                media_type=blob["media_type"],
                expected_hash=digest,
                expected_size=blob["size_bytes"],
                created_at=blob["created_at"],
            )

        repository_keys = tuple(
            str(item["repository_key"]) for item in manifest["repositories"]
        )
        with self._authority() as connection:
            for key in repository_keys:
                counts = sum(
                    int(
                        connection.execute(
                            "SELECT COUNT(*) FROM %s WHERE repository_key = ?" % table,
                            (key,),
                        ).fetchone()[0]
                    )
                    for table in (
                        "candidates",
                        "source_bundles",
                        "records",
                        "feedback",
                        "knowledge_entries",
                        "events",
                        "outbox_receipts",
                    )
                )
                generations = self._generations_from_connection(connection, key)
                if counts or any(
                    (
                        generations.memory_generation,
                        generations.feedback_generation,
                        generations.knowledge_generation,
                    )
                ):
                    raise MemoryStoreConflictError(
                        "memory import target repository is not empty"
                    )
            if connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]:
                raise MemoryStoreConflictError(
                    "memory import requires an empty target event log"
                )

            for repository in manifest["repositories"]:
                key = str(repository["repository_key"])
                created_at = str(repository["created_at"])
                self._ensure_repository_rows(connection, key, created_at)
                identity_json = None
                if repository["identity_schema"] is not None:
                    identity_json = canonical_json(
                        {
                            "schema": repository["identity_schema"],
                            "repository_key": key,
                            "canonical_path": repository["canonical_path"],
                            "git_common_dir": repository["git_common_dir"],
                            "origin_url": repository["origin_url"],
                        }
                    )
                connection.execute(
                    """
                    UPDATE repositories SET
                        identity_schema = ?, canonical_path = ?, git_common_dir = ?,
                        origin_url = ?, identity_json = ?, created_at = ?,
                        last_accessed_at = ?
                    WHERE repository_key = ?
                    """,
                    (
                        repository["identity_schema"],
                        repository["canonical_path"],
                        repository["git_common_dir"],
                        repository["origin_url"],
                        identity_json,
                        created_at,
                        created_at,
                        key,
                    ),
                )

            for envelope in manifest["candidates"]:
                payload = envelope["model"]
                candidate = MemoryCandidate.from_dict(payload)
                connection.execute(
                    """
                    INSERT INTO candidates(
                        candidate_id, repository_key, content_fingerprint, model_json,
                        body_hash, current_status, generation, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        candidate.candidate_id,
                        candidate.repository_key,
                        candidate.content_fingerprint,
                        canonical_json(payload),
                        envelope["body_hash"],
                        envelope["current_status"],
                        envelope["generation"],
                        candidate.created_at,
                    ),
                )
            for envelope in manifest["source_bundles"]:
                payload = envelope["model"]
                bundle = SourceBundleDescriptor.from_dict(payload)
                connection.execute(
                    """
                    INSERT INTO source_bundles(
                        bundle_hash, repository_key, candidate_id, blob_hash,
                        model_json, body_hash, generation, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        bundle.bundle_hash,
                        bundle.repository_key,
                        bundle.candidate_id,
                        bundle.blob_hash,
                        canonical_json(payload),
                        envelope["body_hash"],
                        envelope["generation"],
                        bundle.created_at,
                    ),
                )
            for envelope in manifest["records"]:
                payload = envelope["model"]
                record = DurableMemoryRecord.from_dict(payload)
                connection.execute(
                    """
                    INSERT INTO records(
                        memory_id, candidate_id, repository_key, source_bundle_hash,
                        model_json, body_hash, current_status, generation, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.memory_id,
                        record.candidate_id,
                        record.repository_key,
                        record.source_bundle_hash,
                        canonical_json(payload),
                        envelope["body_hash"],
                        envelope["current_status"],
                        envelope["generation"],
                        record.created_at,
                    ),
                )
            for envelope in manifest["feedback"]:
                payload = envelope["model"]
                feedback = FeedbackRecord.from_dict(payload)
                connection.execute(
                    """
                    INSERT INTO feedback(
                        feedback_id, repository_key, review_id, finding_id,
                        model_json, body_hash, current_status, generation, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        feedback.feedback_id,
                        feedback.repository_key,
                        feedback.review_id,
                        feedback.finding_id,
                        canonical_json(payload),
                        envelope["body_hash"],
                        envelope["current_status"],
                        envelope["generation"],
                        feedback.created_at,
                    ),
                )
            for envelope in manifest["knowledge_entries"]:
                payload = envelope["model"]
                entry = RepositoryKnowledgeEntry.from_dict(payload)
                connection.execute(
                    """
                    INSERT INTO knowledge_entries(
                        entry_id, repository_key, key_hash, blob_hash, model_json,
                        body_hash, generation, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entry.entry_id,
                        entry.key.repository_key,
                        entry.key.key_hash,
                        entry.blob_hash,
                        canonical_json(payload),
                        envelope["body_hash"],
                        envelope["generation"],
                        entry.created_at,
                    ),
                )
            for pin in manifest["blob_pins"]:
                connection.execute(
                    """
                    INSERT INTO blob_pins(blob_hash, pin_type, pin_id, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        pin["blob_hash"],
                        pin["pin_type"],
                        pin["pin_id"],
                        pin["created_at"],
                    ),
                )
            for envelope in manifest["events"]:
                event = _event_from_payload(envelope["event"])
                connection.execute(
                    """
                    INSERT INTO events(
                        sequence, event_id, schema_version, repository_key,
                        subject_type, subject_id, action, actor_type, actor_id,
                        reason_code, reason, previous_status, new_status, request_id,
                        created_at, previous_hash, current_hash, generation_kind,
                        generation
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.sequence,
                        event.event_id,
                        event.schema_version,
                        event.repository_key,
                        event.subject_type,
                        event.subject_id,
                        event.action,
                        event.actor_type,
                        event.actor_id,
                        event.reason_code,
                        event.reason,
                        event.previous_status,
                        event.new_status,
                        event.request_id,
                        event.created_at,
                        event.previous_hash,
                        event.current_hash,
                        event.generation_kind,
                        event.generation,
                    ),
                )
            for receipt in manifest["outbox_receipts"]:
                connection.execute(
                    """
                    INSERT INTO outbox_receipts(
                        request_id, repository_key, operation, request_hash,
                        subject_id, event_id, result_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        receipt["request_id"],
                        receipt["repository_key"],
                        receipt["operation"],
                        receipt["request_hash"],
                        receipt["subject_id"],
                        receipt["event_id"],
                        canonical_json(receipt["result"]),
                        receipt["created_at"],
                    ),
                )
            for key in repository_keys:
                events = [
                    _event_from_payload(envelope["event"])
                    for envelope in manifest["events"]
                    if envelope["event"]["repository_key"] == key
                ]
                connection.execute(
                    """
                    UPDATE event_chain_heads SET
                        event_count = ?, head_sequence = ?, head_hash = ?
                    WHERE repository_key = ?
                    """,
                    (
                        len(events),
                        None if not events else events[-1].sequence,
                        ZERO_EVENT_HASH if not events else events[-1].current_hash,
                        key,
                    ),
                )
            for item in manifest["generations"]:
                generations = GenerationMetadata.from_dict(item["generations"])
                connection.execute(
                    """
                    UPDATE generations SET
                        memory_generation = ?, feedback_generation = ?,
                        knowledge_generation = ?
                    WHERE repository_key = ?
                    """,
                    (
                        generations.memory_generation,
                        generations.feedback_generation,
                        generations.knowledge_generation,
                        item["repository_key"],
                    ),
                )
        self.validate_integrity()

    def backup_to(self, destination: PathInput) -> Path:
        if self._read_only:
            # Read-only stores may still be backed up; this guards only destination writes.
            pass
        target = Path(destination).resolve(strict=False)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            raise MemoryStoreUnavailableError("memory backup destination is unavailable") from None
        staging = target.with_name(".%s.%s.backup" % (target.name, uuid.uuid4().hex))
        try:
            self.validate_integrity()
            source = self._connect(read_only=True)
            backup = sqlite3.connect(str(staging), isolation_level=None)
            try:
                source.backup(backup)
            finally:
                backup.close()
                source.close()
            staged_store = MemoryStore(
                staging,
                busy_timeout_ms=self._busy_timeout_ms,
                read_only=True,
            )
            staged_store.validate_integrity(validate_blob_files=False)
            os.replace(str(staging), str(target))
            _fsync_directory(target.parent)
            return target
        except MemoryStoreError:
            raise
        except (OSError, sqlite3.Error):
            raise MemoryStoreUnavailableError("memory backup failed") from None
        finally:
            try:
                staging.unlink(missing_ok=True)
            except OSError:
                pass

    def replace_with_staged_copy(
        self,
        migration: Callable[[sqlite3.Connection], None],
        *,
        validator: Optional[Callable[[sqlite3.Connection], None]] = None,
    ) -> None:
        if self._read_only:
            raise MemoryStoreReadOnlyError()
        if not callable(migration) or (validator is not None and not callable(validator)):
            raise MemoryStoreValidationError("migration callbacks must be callable")
        staging = self.database_path.with_name(
            ".%s.%s.migration.sqlite3"
            % (self.database_path.name, uuid.uuid4().hex)
        )
        with _exclusive_file_lock(
            self.namespace_path / ".memory-store.lock", self._busy_timeout_ms
        ):
            try:
                self.checkpoint("TRUNCATE")
                source = self._connect(read_only=True)
                staged_connection = sqlite3.connect(str(staging), isolation_level=None)
                try:
                    source.backup(staged_connection)
                finally:
                    staged_connection.close()
                    source.close()

                staged_connection = sqlite3.connect(
                    str(staging),
                    timeout=self._busy_timeout_ms / 1000.0,
                    isolation_level=None,
                )
                staged_connection.row_factory = sqlite3.Row
                try:
                    staged_connection.execute("PRAGMA foreign_keys = ON")
                    staged_connection.execute("BEGIN IMMEDIATE")
                    migration(staged_connection)
                    staged_connection.commit()
                    if validator is not None:
                        validator(staged_connection)
                    staged_connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except Exception:
                    if staged_connection.in_transaction:
                        staged_connection.rollback()
                    raise MemoryStoreMigrationError() from None
                finally:
                    staged_connection.close()

                staged_store = MemoryStore(
                    staging,
                    busy_timeout_ms=self._busy_timeout_ms,
                    read_only=True,
                )
                staged_store.validate_integrity()
                os.replace(str(staging), str(self.database_path))
                for suffix in ("-wal", "-shm"):
                    try:
                        Path(str(self.database_path) + suffix).unlink(missing_ok=True)
                    except OSError:
                        pass
                _fsync_directory(self.database_path.parent)
                self._initialize_or_validate()
            except MemoryStoreMigrationError:
                raise
            except MemoryStoreError:
                raise MemoryStoreMigrationError() from None
            except (OSError, sqlite3.Error):
                raise MemoryStoreMigrationError() from None
            finally:
                for path in (
                    staging,
                    Path(str(staging) + "-wal"),
                    Path(str(staging) + "-shm"),
                ):
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        pass


def _table_names(connection: sqlite3.Connection) -> Set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            """
        )
    }


def _validate_schema_connection(connection: sqlite3.Connection) -> None:
    tables = _table_names(connection)
    if "metadata" not in tables:
        raise MemoryStoreSchemaError()
    try:
        metadata = {
            str(row["key"]): str(row["value"])
            for row in connection.execute("SELECT key, value FROM metadata")
        }
    except (sqlite3.Error, TypeError, KeyError):
        raise MemoryStoreSchemaError() from None
    if (
        metadata.get("schema_name") != STORE_SCHEMA_NAME
        or metadata.get("schema_version") != str(STORE_SCHEMA_VERSION)
    ):
        raise MemoryStoreSchemaError()
    if connection.execute("PRAGMA user_version").fetchone()[0] != STORE_SCHEMA_VERSION:
        raise MemoryStoreSchemaError()
    missing = _REQUIRED_TABLES - tables
    if missing:
        raise MemoryStoreCorruptionError("memory store schema is incomplete")
    triggers = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger'"
        )
    }
    if _REQUIRED_TRIGGERS - triggers:
        raise MemoryStoreCorruptionError("memory store append-only guards are incomplete")
    if metadata.get("schema_definition_hash") != SCHEMA_DEFINITION_HASH:
        raise MemoryStoreCorruptionError("memory store schema definition is invalid")
    migration = connection.execute(
        """
        SELECT schema_name, definition_hash FROM schema_migrations
        WHERE schema_version = ?
        """,
        (STORE_SCHEMA_VERSION,),
    ).fetchone()
    if (
        migration is None
        or migration["schema_name"] != STORE_SCHEMA_NAME
        or migration["definition_hash"] != SCHEMA_DEFINITION_HASH
    ):
        raise MemoryStoreCorruptionError("memory store migration metadata is invalid")


def _translate_sqlite_error(error: sqlite3.Error) -> MemoryStoreError:
    message = str(error).casefold()
    if "locked" in message or "busy" in message:
        return MemoryStoreBusyError()
    if "readonly" in message or "read-only" in message:
        return MemoryStoreReadOnlyError()
    if isinstance(error, sqlite3.IntegrityError):
        return MemoryStoreConflictError()
    corruption_markers = (
        "malformed",
        "not a database",
        "disk image",
        "no such table",
        "database schema has changed",
    )
    if isinstance(error, sqlite3.DatabaseError) and any(
        marker in message for marker in corruption_markers
    ):
        return MemoryStoreCorruptionError()
    return MemoryStoreUnavailableError()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _timestamp(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _UTC_PATTERN.fullmatch(value) is None:
        raise MemoryStoreValidationError("%s must be a canonical UTC timestamp" % field_name)
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        raise MemoryStoreValidationError(
            "%s must be a canonical UTC timestamp" % field_name
        ) from None
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise MemoryStoreValidationError("%s must use UTC" % field_name)
    return value


def _required_text(value: Any, field_name: str, max_length: int) -> str:
    if not isinstance(value, str):
        raise MemoryStoreValidationError("%s must be text" % field_name)
    normalized = value.strip()
    if not normalized or len(normalized) > max_length or "\0" in normalized:
        raise MemoryStoreValidationError("%s is outside the supported bounds" % field_name)
    return normalized


def _optional_text(value: Any, field_name: str, max_length: int) -> Optional[str]:
    if value is None:
        return None
    return _required_text(value, field_name, max_length)


def _required_token(value: Any, field_name: str) -> str:
    token = _required_text(value, field_name, 512)
    if _TOKEN_PATTERN.fullmatch(token) is None:
        raise MemoryStoreValidationError("%s is not a canonical token" % field_name)
    return token


def _required_bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise MemoryStoreValidationError("%s must be a boolean" % field_name)
    return value


def _repository_key(value: Any) -> str:
    key = _required_text(value, "repository_key", 512)
    if _REPOSITORY_KEY_PATTERN.fullmatch(key) is None:
        raise MemoryStoreValidationError(
            "repository_key must be a lowercase SHA-256 digest"
        )
    return key


def _digest(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise MemoryStoreValidationError("%s must be a SHA-256 digest" % field_name)
    return value


def _media_type(value: Any) -> str:
    if not isinstance(value, str) or _MEDIA_TYPE_PATTERN.fullmatch(value) is None:
        raise MemoryStoreValidationError("media type is invalid")
    return value.casefold()


def _pin_type(value: Any) -> str:
    checked = _required_token(value, "pin_type")
    if checked not in _PIN_TYPES:
        raise MemoryStoreValidationError("blob pin type is unsupported")
    return checked


def _subject_type(value: Any) -> str:
    checked = _required_token(value, "subject_type")
    if checked not in _SUBJECT_TYPES:
        raise MemoryStoreValidationError("memory event subject type is unsupported")
    return checked


def _stable_subject_id(value: Any, prefix: str, field_name: str) -> str:
    try:
        return validate_stable_id(value, prefix, field_name)
    except ValueError:
        raise MemoryStoreValidationError("%s is invalid" % field_name) from None


def _request_id(value: Any) -> str:
    return _stable_subject_id(value, "REQ", "request_id")


def _event_id(value: Any) -> str:
    return _stable_subject_id(value, "EVT", "event_id")


def _model_storage(model: Any) -> Tuple[str, str]:
    try:
        payload = model.to_dict()
        text = canonical_json(payload)
        return text, canonical_sha256(payload)
    except (AttributeError, TypeError, ValueError):
        raise MemoryStoreValidationError("canonical memory model serialization failed") from None


def _hydrate_model_row(row: sqlite3.Row, model_type: Any) -> Any:
    try:
        text = row["model_json"]
        body_hash = row["body_hash"]
        if not isinstance(text, str) or not isinstance(body_hash, str):
            raise ValueError
        payload = json.loads(text)
        if canonical_json(payload) != text:
            raise ValueError
        if not hmac.compare_digest(canonical_sha256(payload), body_hash):
            raise ValueError
        model = model_type.from_dict(payload)
        if canonical_json(model.to_dict()) != text:
            raise ValueError
        return model
    except (json.JSONDecodeError, TypeError, ValueError, KeyError, MemoryStoreError):
        raise MemoryStoreCorruptionError("canonical memory row is invalid") from None


def _candidate_from_row(row: sqlite3.Row) -> MemoryCandidate:
    candidate = _hydrate_model_row(row, MemoryCandidate)
    try:
        status = CandidateStatus(row["current_status"])
        if row["repository_key"] != candidate.repository_key:
            raise ValueError
        return replace(candidate, status=status)
    except (TypeError, ValueError):
        raise MemoryStoreCorruptionError("candidate projection is invalid") from None


def _source_bundle_from_row(row: sqlite3.Row) -> SourceBundleDescriptor:
    bundle = _hydrate_model_row(row, SourceBundleDescriptor)
    if (
        row["bundle_hash"] != bundle.bundle_hash
        or row["repository_key"] != bundle.repository_key
        or row["candidate_id"] != bundle.candidate_id
        or row["blob_hash"] != bundle.blob_hash
    ):
        raise MemoryStoreCorruptionError("source bundle projection is invalid")
    return bundle


def _validated_source_bundle_from_row(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
) -> SourceBundleDescriptor:
    bundle = _source_bundle_from_row(row)
    candidate_row = connection.execute(
        "SELECT * FROM candidates WHERE candidate_id = ?",
        (bundle.candidate_id,),
    ).fetchone()
    if candidate_row is None:
        raise MemoryStoreCorruptionError("source bundle candidate is missing")
    candidate = _candidate_from_row(candidate_row)
    if (
        bundle.repository_key != candidate.repository_key
        or bundle.candidate_id != candidate.candidate_id
        or bundle.source_refs != candidate.source_refs
    ):
        raise MemoryStoreCorruptionError(
            "source bundle does not match its canonical candidate"
        )
    return bundle


def _record_from_row(row: sqlite3.Row) -> DurableMemoryRecord:
    record = _hydrate_model_row(row, DurableMemoryRecord)
    try:
        status = RecordStatus(row["current_status"])
        if (
            row["memory_id"] != record.memory_id
            or row["candidate_id"] != record.candidate_id
            or row["repository_key"] != record.repository_key
            or row["source_bundle_hash"] != record.source_bundle_hash
        ):
            raise ValueError
        return replace(record, status=status)
    except (TypeError, ValueError):
        raise MemoryStoreCorruptionError("record projection is invalid") from None


def _validated_record_from_row(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
) -> DurableMemoryRecord:
    record = _record_from_row(row)
    candidate_row = connection.execute(
        "SELECT * FROM candidates WHERE candidate_id = ?",
        (record.candidate_id,),
    ).fetchone()
    if candidate_row is None:
        raise MemoryStoreCorruptionError("record candidate is missing")
    candidate = _candidate_from_row(candidate_row)
    if not _record_matches_candidate(record, candidate):
        raise MemoryStoreCorruptionError(
            "record body does not match its canonical candidate"
        )
    return record


def _feedback_from_row(row: sqlite3.Row) -> FeedbackRecord:
    feedback = _hydrate_model_row(row, FeedbackRecord)
    try:
        status = FeedbackStatus(row["current_status"])
        if (
            row["feedback_id"] != feedback.feedback_id
            or row["repository_key"] != feedback.repository_key
            or row["review_id"] != feedback.review_id
            or row["finding_id"] != feedback.finding_id
        ):
            raise ValueError
        return replace(feedback, status=status)
    except (TypeError, ValueError):
        raise MemoryStoreCorruptionError("feedback projection is invalid") from None


def _knowledge_from_row(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
) -> RepositoryKnowledgeEntry:
    entry = _hydrate_model_row(row, RepositoryKnowledgeEntry)
    if (
        row["entry_id"] != entry.entry_id
        or row["repository_key"] != entry.key.repository_key
        or row["key_hash"] != entry.key.key_hash
        or row["blob_hash"] != entry.blob_hash
    ):
        raise MemoryStoreCorruptionError("knowledge projection is invalid")
    prefix = entry.entry_id + ":"
    pins = tuple(
        sorted(
            str(pin["pin_id"])[len(prefix) :]
            for pin in connection.execute(
                """
                SELECT pin_id FROM blob_pins
                WHERE blob_hash = ? AND pin_type = 'knowledge'
                  AND pin_id LIKE ? ESCAPE '\\'
                ORDER BY pin_id
                """,
                (entry.blob_hash, _escape_like(entry.entry_id) + ":%"),
            )
            if str(pin["pin_id"]).startswith(prefix)
        )
    )
    try:
        return replace(entry, pinned_by_review_ids=pins)
    except ValueError:
        raise MemoryStoreCorruptionError("knowledge pin projection is invalid") from None


def _record_matches_candidate(
    record: DurableMemoryRecord,
    candidate: MemoryCandidate,
) -> bool:
    return (
        record.candidate_id == candidate.candidate_id
        and record.repository_key == candidate.repository_key
        and record.kind is candidate.kind
        and record.statement == candidate.statement
        and record.scope == candidate.scope
        and record.source_refs == candidate.source_refs
        and record.valid_from_sha == candidate.valid_from_sha
        and record.validity_policies == candidate.validity_policies
        and record.confidence is candidate.confidence
        and record.sensitivity is candidate.sensitivity
        and record.policy_effect == candidate.policy_effect
    )


def _knowledge_identity_equal(
    left: RepositoryKnowledgeEntry,
    right: RepositoryKnowledgeEntry,
) -> bool:
    return (
        left.entry_id == right.entry_id
        and left.key == right.key
        and left.blob_hash == right.blob_hash
        and left.size_bytes == right.size_bytes
        and left.content_type == right.content_type
        and left.artifact_schema == right.artifact_schema
        and left.summary_hash == right.summary_hash
    )


def _event_from_row(row: sqlite3.Row) -> MemoryEvent:
    payload = {
        "sequence": row["sequence"],
        "event_id": row["event_id"],
        "schema_version": row["schema_version"],
        "repository_key": row["repository_key"],
        "subject_type": row["subject_type"],
        "subject_id": row["subject_id"],
        "action": row["action"],
        "actor_type": row["actor_type"],
        "actor_id": row["actor_id"],
        "reason_code": row["reason_code"],
        "reason": row["reason"],
        "previous_status": row["previous_status"],
        "new_status": row["new_status"],
        "request_id": row["request_id"],
        "created_at": row["created_at"],
        "previous_hash": row["previous_hash"],
        "current_hash": row["current_hash"],
        "generation_kind": row["generation_kind"],
        "generation": row["generation"],
    }
    return _event_from_payload(payload)


def _event_from_payload(payload: Mapping[str, Any]) -> MemoryEvent:
    expected = {
        "sequence",
        "event_id",
        "schema_version",
        "repository_key",
        "subject_type",
        "subject_id",
        "action",
        "actor_type",
        "actor_id",
        "reason_code",
        "reason",
        "previous_status",
        "new_status",
        "request_id",
        "created_at",
        "previous_hash",
        "current_hash",
        "generation_kind",
        "generation",
    }
    try:
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError
        if type(payload["sequence"]) is not int or payload["sequence"] <= 0:
            raise ValueError
        if payload["schema_version"] != EVENT_SCHEMA_VERSION:
            raise ValueError
        if type(payload["generation"]) is not int or payload["generation"] < 0:
            raise ValueError
        generation_kind = str(payload["generation_kind"])
        if generation_kind not in {"memory", "feedback", "knowledge"}:
            raise ValueError
        return MemoryEvent(
            sequence=payload["sequence"],
            event_id=_event_id(payload["event_id"]),
            schema_version=EVENT_SCHEMA_VERSION,
            repository_key=_repository_key(payload["repository_key"]),
            subject_type=_subject_type(payload["subject_type"]),
            subject_id=_required_text(payload["subject_id"], "subject_id", 512),
            action=_required_token(payload["action"], "action"),
            actor_type=_required_token(payload["actor_type"], "actor_type"),
            actor_id=_required_text(payload["actor_id"], "actor_id", 512),
            reason_code=_required_token(payload["reason_code"], "reason_code"),
            reason=_optional_text(payload["reason"], "reason", 2_048),
            previous_status=_optional_text(
                payload["previous_status"], "previous_status", 128
            ),
            new_status=_optional_text(payload["new_status"], "new_status", 128),
            request_id=_request_id(payload["request_id"]),
            created_at=_timestamp(payload["created_at"], "created_at"),
            previous_hash=_digest(payload["previous_hash"], "previous_hash"),
            current_hash=_digest(payload["current_hash"], "current_hash"),
            generation_kind=generation_kind,
            generation=payload["generation"],
        )
    except (TypeError, ValueError, MemoryStoreError):
        raise MemoryStoreCorruptionError("memory event row is invalid") from None


def _export_model_row(
    row: sqlite3.Row,
    *,
    id_column: str,
    redact: bool,
) -> Dict[str, Any]:
    return {
        "id": str(row[id_column]),
        "model": None if redact else json.loads(row["model_json"]),
        "current_status": (
            row["current_status"] if "current_status" in row.keys() else None
        ),
        "generation": int(row["generation"]),
        "body_hash": str(row["body_hash"]),
        "redacted": redact,
    }


def _validate_export_model_rows(
    rows: Sequence[Mapping[str, Any]],
    model_type: Any,
    id_attribute: str,
    redaction_allowed: bool,
) -> None:
    ids: List[str] = []
    expected = {"id", "model", "current_status", "generation", "body_hash", "redacted"}
    for envelope in rows:
        if not isinstance(envelope, Mapping) or set(envelope) != expected:
            raise MemoryStoreValidationError("memory import model envelope is invalid")
        identifier = _required_text(envelope["id"], "model id", 512)
        ids.append(identifier)
        redacted = _required_bool(envelope["redacted"], "model redacted")
        if redacted:
            if not redaction_allowed or envelope["model"] is not None:
                raise MemoryStoreValidationError("memory import redaction is invalid")
        else:
            if not isinstance(envelope["model"], Mapping):
                raise MemoryStoreValidationError("memory import model is missing")
            try:
                model = model_type.from_dict(envelope["model"])
            except (TypeError, ValueError):
                raise MemoryStoreValidationError("memory import model is invalid") from None
            if getattr(model, id_attribute) != identifier:
                raise MemoryStoreValidationError("memory import model ID is invalid")
            if not hmac.compare_digest(
                canonical_sha256(envelope["model"]),
                _digest(envelope["body_hash"], "body_hash"),
            ):
                raise MemoryStoreValidationError("memory import model hash is invalid")
        if type(envelope["generation"]) is not int or envelope["generation"] <= 0:
            raise MemoryStoreValidationError("memory import model generation is invalid")
        if envelope["current_status"] is not None:
            _required_token(envelope["current_status"], "current_status")
        _digest(envelope["body_hash"], "body_hash")
    if ids != sorted(set(ids)):
        raise MemoryStoreValidationError("memory import model rows are not canonical")


def _validate_export_events(
    rows: Sequence[Mapping[str, Any]],
    redacted_manifest: bool,
) -> None:
    previous_by_repository: Dict[str, str] = {}
    sequence = 0
    event_ids: Set[str] = set()
    request_ids: Set[str] = set()
    for envelope in rows:
        if not isinstance(envelope, Mapping) or set(envelope) != {
            "event",
            "reason_hash",
            "reason_redacted",
        }:
            raise MemoryStoreValidationError("memory import event envelope is invalid")
        reason_redacted = _required_bool(
            envelope["reason_redacted"], "reason_redacted"
        )
        try:
            event = _event_from_payload(envelope["event"])
        except MemoryStoreError:
            raise MemoryStoreValidationError("memory import event is invalid") from None
        if event.sequence <= sequence:
            raise MemoryStoreValidationError("memory import event order is invalid")
        sequence = event.sequence
        if event.event_id in event_ids or event.request_id in request_ids:
            raise MemoryStoreValidationError("memory import event identity is duplicated")
        event_ids.add(event.event_id)
        request_ids.add(event.request_id)
        expected_previous = previous_by_repository.get(
            event.repository_key, ZERO_EVENT_HASH
        )
        if not hmac.compare_digest(event.previous_hash, expected_previous):
            raise MemoryStoreValidationError("memory import event chain is discontinuous")
        if reason_redacted:
            if not redacted_manifest or event.reason is not None:
                raise MemoryStoreValidationError("memory import event redaction is invalid")
            _digest(envelope["reason_hash"], "reason_hash")
        else:
            if envelope["reason_hash"] is not None:
                raise MemoryStoreValidationError("memory import reason hash is unexpected")
            expected_hash = canonical_sha256(event.hash_payload())
            if not hmac.compare_digest(expected_hash, event.current_hash):
                raise MemoryStoreValidationError("memory import event hash is invalid")
        previous_by_repository[event.repository_key] = event.current_hash


def _receipt_from_row(row: sqlite3.Row) -> Dict[str, Any]:
    try:
        result_payload = json.loads(row["result_json"])
        if canonical_json(result_payload) != row["result_json"]:
            raise ValueError
        result = WriteResult.from_dict(result_payload)
        event_id = None if row["event_id"] is None else _event_id(row["event_id"])
        if result.event_id != event_id or result.subject_id != row["subject_id"]:
            raise ValueError
        return {
            "request_id": _request_id(row["request_id"]),
            "repository_key": _repository_key(row["repository_key"]),
            "operation": _required_token(row["operation"], "operation"),
            "request_hash": _digest(row["request_hash"], "request_hash"),
            "subject_id": _required_text(row["subject_id"], "subject_id", 512),
            "event_id": event_id,
            "result": result.to_dict(),
            "created_at": _timestamp(row["created_at"], "receipt created_at"),
        }
    except (json.JSONDecodeError, TypeError, ValueError, MemoryStoreError):
        raise MemoryStoreCorruptionError("memory request receipt is invalid") from None


def _receipt_export_from_row(row: sqlite3.Row) -> Dict[str, Any]:
    return _receipt_from_row(row)


def _validate_export_receipts(rows: Sequence[Mapping[str, Any]]) -> None:
    expected = {
        "request_id",
        "repository_key",
        "operation",
        "request_hash",
        "subject_id",
        "event_id",
        "result",
        "created_at",
    }
    request_ids: List[str] = []
    for receipt in rows:
        if not isinstance(receipt, Mapping) or set(receipt) != expected:
            raise MemoryStoreValidationError("memory import request receipt is invalid")
        request_id = _request_id(receipt["request_id"])
        request_ids.append(request_id)
        _repository_key(receipt["repository_key"])
        operation = _required_token(receipt["operation"], "operation")
        _digest(receipt["request_hash"], "request_hash")
        subject_id = _required_text(receipt["subject_id"], "subject_id", 512)
        event_id = (
            None if receipt["event_id"] is None else _event_id(receipt["event_id"])
        )
        _timestamp(receipt["created_at"], "receipt created_at")
        try:
            result = WriteResult.from_dict(receipt["result"])
        except (TypeError, ValueError, MemoryStoreError):
            raise MemoryStoreValidationError("memory import request result is invalid") from None
        if (
            result.operation != operation
            or result.subject_id != subject_id
            or result.event_id != event_id
        ):
            raise MemoryStoreValidationError("memory import request result is inconsistent")
    if request_ids != sorted(set(request_ids)):
        raise MemoryStoreValidationError("memory import request receipts are not canonical")


def _validate_manifest_relationships(manifest: Mapping[str, Any]) -> None:
    repository_keys = {
        str(item["repository_key"]) for item in manifest["repositories"]
    }
    generation_map = {
        str(item["repository_key"]): GenerationMetadata.from_dict(item["generations"])
        for item in manifest["generations"]
    }
    if set(generation_map) != repository_keys:
        raise MemoryStoreValidationError(
            "memory import repository generations are incomplete"
        )
    blob_map = {str(item["blob_hash"]): item for item in manifest["blobs"]}
    candidate_envelopes = {str(item["id"]): item for item in manifest["candidates"]}
    bundle_envelopes = {
        str(item["id"]): item for item in manifest["source_bundles"]
    }
    candidate_models: Dict[str, MemoryCandidate] = {}
    for identifier, envelope in candidate_envelopes.items():
        if envelope["model"] is None:
            continue
        candidate = MemoryCandidate.from_dict(envelope["model"])
        candidate_models[identifier] = candidate
        if candidate.repository_key not in repository_keys:
            raise MemoryStoreValidationError("memory import candidate repository is missing")
        try:
            CandidateStatus(envelope["current_status"])
        except ValueError:
            raise MemoryStoreValidationError(
                "memory import candidate projection is invalid"
            ) from None
        if envelope["generation"] > generation_map[
            candidate.repository_key
        ].memory_generation:
            raise MemoryStoreValidationError("memory import candidate generation is invalid")

    bundle_models: Dict[str, SourceBundleDescriptor] = {}
    for identifier, envelope in bundle_envelopes.items():
        if envelope["model"] is None:
            continue
        bundle = SourceBundleDescriptor.from_dict(envelope["model"])
        bundle_models[identifier] = bundle
        if (
            bundle.repository_key not in repository_keys
            or bundle.candidate_id not in candidate_envelopes
            or bundle.blob_hash not in blob_map
        ):
            raise MemoryStoreValidationError("memory import source bundle reference is missing")
        blob = blob_map[bundle.blob_hash]
        if blob["size_bytes"] != bundle.size_bytes or blob["media_type"] != bundle.media_type:
            raise MemoryStoreValidationError("memory import source bundle blob is invalid")
        candidate = candidate_models.get(bundle.candidate_id)
        if candidate is not None and (
            bundle.repository_key != candidate.repository_key
            or bundle.source_refs != candidate.source_refs
        ):
            raise MemoryStoreValidationError(
                "memory import source bundle does not match its candidate"
            )
        if envelope["generation"] > generation_map[
            bundle.repository_key
        ].memory_generation:
            raise MemoryStoreValidationError("memory import source bundle generation is invalid")

    for envelope in manifest["records"]:
        if envelope["model"] is None:
            continue
        record = DurableMemoryRecord.from_dict(envelope["model"])
        if (
            record.repository_key not in repository_keys
            or record.candidate_id not in candidate_envelopes
            or record.source_bundle_hash not in bundle_envelopes
        ):
            raise MemoryStoreValidationError("memory import record reference is missing")
        try:
            RecordStatus(envelope["current_status"])
        except ValueError:
            raise MemoryStoreValidationError("memory import record projection is invalid") from None
        candidate = candidate_models.get(record.candidate_id)
        if candidate is not None and not _record_matches_candidate(record, candidate):
            raise MemoryStoreValidationError(
                "memory import record does not match its candidate"
            )
        bundle = bundle_models.get(record.source_bundle_hash)
        if bundle is not None and bundle.candidate_id != record.candidate_id:
            raise MemoryStoreValidationError(
                "memory import record source bundle is inconsistent"
            )
        if envelope["generation"] > generation_map[
            record.repository_key
        ].memory_generation:
            raise MemoryStoreValidationError("memory import record generation is invalid")

    for envelope in manifest["feedback"]:
        if envelope["model"] is None:
            continue
        feedback = FeedbackRecord.from_dict(envelope["model"])
        if feedback.repository_key not in repository_keys:
            raise MemoryStoreValidationError("memory import feedback repository is missing")
        try:
            FeedbackStatus(envelope["current_status"])
        except ValueError:
            raise MemoryStoreValidationError(
                "memory import feedback projection is invalid"
            ) from None
        if envelope["generation"] > generation_map[
            feedback.repository_key
        ].feedback_generation:
            raise MemoryStoreValidationError("memory import feedback generation is invalid")

    for envelope in manifest["knowledge_entries"]:
        entry = RepositoryKnowledgeEntry.from_dict(envelope["model"])
        repository_key = entry.key.repository_key
        if repository_key not in repository_keys or entry.blob_hash not in blob_map:
            raise MemoryStoreValidationError("memory import knowledge reference is missing")
        blob = blob_map[entry.blob_hash]
        if blob["size_bytes"] != entry.size_bytes or blob["media_type"] != entry.content_type:
            raise MemoryStoreValidationError("memory import knowledge blob is invalid")
        if envelope["generation"] > generation_map[
            repository_key
        ].knowledge_generation:
            raise MemoryStoreValidationError("memory import knowledge generation is invalid")

    last_generation: Dict[Tuple[str, str], int] = {}
    event_ids: Set[str] = set()
    for envelope in manifest["events"]:
        event = _event_from_payload(envelope["event"])
        if event.repository_key not in repository_keys:
            raise MemoryStoreValidationError("memory import event repository is missing")
        key = (event.repository_key, event.generation_kind)
        expected = last_generation.get(key, 0) + 1
        if event.generation != expected:
            raise MemoryStoreValidationError("memory import event generation is invalid")
        last_generation[key] = event.generation
        event_ids.add(event.event_id)
    for repository_key, generations in generation_map.items():
        if (
            last_generation.get((repository_key, "memory"), 0)
            != generations.memory_generation
            or last_generation.get((repository_key, "feedback"), 0)
            != generations.feedback_generation
            or last_generation.get((repository_key, "knowledge"), 0)
            != generations.knowledge_generation
        ):
            raise MemoryStoreValidationError(
                "memory import generations do not match the event log"
            )
    for receipt in manifest["outbox_receipts"]:
        if receipt["repository_key"] not in repository_keys:
            raise MemoryStoreValidationError("memory import receipt repository is missing")
        if receipt["event_id"] is not None and receipt["event_id"] not in event_ids:
            raise MemoryStoreValidationError("memory import receipt event is missing")


def _load_manifest(
    manifest_or_path: Union[Mapping[str, Any], PathInput]
) -> Mapping[str, Any]:
    if isinstance(manifest_or_path, Mapping):
        return manifest_or_path
    path = Path(manifest_or_path).resolve(strict=False)
    try:
        if path.stat().st_size > 64 * 1024 * 1024:
            raise MemoryStoreValidationError("memory import manifest is too large")
        payload = json.loads(path.read_text(encoding="utf-8"))
    except MemoryStoreError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise MemoryStoreValidationError("memory import manifest is unreadable") from None
    if not isinstance(payload, Mapping):
        raise MemoryStoreValidationError("memory import manifest must be an object")
    return payload


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    temporary = path.with_name(".%s.%s.tmp" % (path.name, uuid.uuid4().hex))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            str(temporary),
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            offset = 0
            while offset < len(content):
                written = os.write(descriptor, content[offset : offset + 1024 * 1024])
                if written <= 0:
                    raise OSError("short write")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(str(temporary), str(path))
        _fsync_directory(path.parent)
    except OSError:
        raise MemoryStoreUnavailableError("atomic memory file write failed") from None
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(str(path), flags)
    except OSError:
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            pass
    finally:
        os.close(descriptor)


@contextmanager
def _exclusive_file_lock(path: Path, timeout_ms: int) -> Iterator[None]:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            raise OSError("lock path is a symbolic link")
        descriptor = os.open(
            str(path),
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        stream = os.fdopen(descriptor, "r+b", buffering=0)
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
    except OSError:
        raise MemoryStoreUnavailableError("memory migration lock is unavailable") from None

    acquired = False
    deadline = time.monotonic() + timeout_ms / 1000.0
    try:
        while not acquired:
            try:
                if os.name == "nt":
                    import msvcrt

                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except (OSError, IOError):
                if time.monotonic() >= deadline:
                    raise MemoryStoreBusyError("memory migration lock is busy") from None
                time.sleep(0.01)
        yield
    finally:
        if acquired:
            try:
                if os.name == "nt":
                    import msvcrt

                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            except (OSError, IOError):
                pass
        stream.close()


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


__all__ = [
    "BlobGCResult",
    "BlobInfo",
    "DEFAULT_BUSY_TIMEOUT_MS",
    "EVENT_SCHEMA_VERSION",
    "EXPORT_SCHEMA_NAME",
    "EXPORT_SCHEMA_VERSION",
    "ImportPlan",
    "IntegrityReport",
    "MemoryEvent",
    "MemoryStore",
    "MemoryStoreBusyError",
    "MemoryStoreConflictError",
    "MemoryStoreCorruptionError",
    "MemoryStoreError",
    "MemoryStoreErrorCode",
    "MemoryStoreMigrationError",
    "MemoryStoreNotFoundError",
    "MemoryStoreReadOnlyError",
    "MemoryStoreReadView",
    "MemoryStoreSchemaError",
    "MemoryStoreUnavailableError",
    "MemoryStoreValidationError",
    "SCHEMA_DEFINITION_HASH",
    "STORE_SCHEMA_NAME",
    "STORE_SCHEMA_VERSION",
    "WriteResult",
]
