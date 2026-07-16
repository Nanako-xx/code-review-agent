"""Capability-limited scripted clarification for the evaluation harness."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from threading import Lock
import unicodedata
from typing import (
    Callable,
    FrozenSet,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

from . import models as _canonical
from .adapters.base import AgentRunConfig
from .config import ClarificationMatcherSnapshot
from .models import (
    MAX_CLARIFICATION_QUESTIONS,
    ClarificationAction,
    ClarificationAnswer,
    ClarificationScript,
    IntentDimension,
    SchemaError,
    SubmissionClarificationExchange,
)


class ClarificationProtocolError(RuntimeError):
    """A channel operation cannot be represented by the canonical protocol."""


@runtime_checkable
class MaterialClaimMatcher(Protocol):
    """Versionable semantic boundary for matching an asked material claim."""

    @property
    def binding_digest(self) -> str:
        ...

    def equivalent(
        self,
        dimension: IntentDimension,
        actual_claim: str,
        scripted_claim: str,
    ) -> bool:
        ...


@runtime_checkable
class MaterialClaimMatcherFactory(Protocol):
    """Build one matcher strictly from a persisted Run snapshot."""

    def build(
        self,
        snapshot: ClarificationMatcherSnapshot,
    ) -> MaterialClaimMatcher:
        ...


_CANONICAL_MATCHER_IMPLEMENTATION_DIGEST = hashlib.sha256(
    b"review-agent-eval:split-whitespace-then-casefold-equality:v1"
).hexdigest()
_CANONICAL_MATCHER_RUBRIC_DIGEST = hashlib.sha256(
    b"review-agent-eval:dimension-plus-canonical-material-claim-equivalence:v1"
).hexdigest()


def canonical_material_claim_matcher_snapshot() -> ClarificationMatcherSnapshot:
    """Return the exact built-in snapshot for canonicalized product claims."""

    return ClarificationMatcherSnapshot(
        matcher_id="canonical-material-claim",
        matcher_version="1.0.0",
        implementation_digest=_CANONICAL_MATCHER_IMPLEMENTATION_DIGEST,
        model_artifact_digest=None,
        rubric_digest=_CANONICAL_MATCHER_RUBRIC_DIGEST,
        normalization_version="unicode-whitespace-casefold-v1",
        threshold=None,
        parameters={"unicode_version": unicodedata.unidata_version},
    )


class NormalizedMaterialClaimMatcher:
    """Deterministic matcher for adapters that emit canonical claim text."""

    __slots__ = ("__binding_digest",)

    def __init__(self, binding_digest: str) -> None:
        self.__binding_digest = _canonical._digest(
            binding_digest,
            "canonical material claim matcher.binding_digest",
        )

    @property
    def binding_digest(self) -> str:
        return self.__binding_digest

    @staticmethod
    def equivalent(
        dimension: IntentDimension,
        actual_claim: str,
        scripted_claim: str,
    ) -> bool:
        del dimension
        return " ".join(actual_claim.split()).casefold() == " ".join(
            scripted_claim.split()
        ).casefold()


class BuiltinMaterialClaimMatcherFactory:
    """Fail-closed registry for matchers shipped with this Harness build."""

    @staticmethod
    def build(
        snapshot: ClarificationMatcherSnapshot,
    ) -> MaterialClaimMatcher:
        if not isinstance(snapshot, ClarificationMatcherSnapshot):
            raise TypeError("matcher factory requires ClarificationMatcherSnapshot")
        if snapshot != canonical_material_claim_matcher_snapshot():
            raise ClarificationProtocolError(
                "clarification matcher snapshot is unsupported by this Harness"
            )
        return NormalizedMaterialClaimMatcher(snapshot.digest())


class MaterialClaimMatchOutcome(str, Enum):
    MATCHED = "matched"
    UNMATCHED = "unmatched"
    AMBIGUOUS = "ambiguous"
    ROUND_LIMIT = "round_limit"


@dataclass(frozen=True)
class MaterialClaimCandidateDecision:
    answer_id: str
    request_digest: str
    equivalent: bool
    action_eligible: bool


@dataclass(frozen=True)
class MaterialClaimMatchReceipt:
    turn_index: int
    question_id: str
    dimension: IntentDimension
    actual_claim_digest: str
    matcher_digest: str
    candidates: Tuple[MaterialClaimCandidateDecision, ...]
    outcome: MaterialClaimMatchOutcome
    matched_answer_id: str | None


@runtime_checkable
class ClarificationChannel(Protocol):
    """The complete clarification capability visible to an Adapter."""

    def ask(
        self,
        *,
        question_id: str,
        dimension: IntentDimension,
        question: str,
        material_claim: str,
        proposed_values: Sequence[str] = (),
    ) -> SubmissionClarificationExchange:
        ...


class _BoundClarificationChannel:
    __slots__ = ("__ask_question",)

    def __init__(self, ask_question: Callable[..., SubmissionClarificationExchange]):
        self.__ask_question = ask_question

    def ask(
        self,
        *,
        question_id: str,
        dimension: IntentDimension,
        question: str,
        material_claim: str,
        proposed_values: Sequence[str] = (),
    ) -> SubmissionClarificationExchange:
        return self.__ask_question(
            question_id=question_id,
            dimension=dimension,
            question=question,
            material_claim=material_claim,
            proposed_values=proposed_values,
        )

class ClarificationSession:
    """Harness-owned script controller exposed only to trusted Adapter code.

    The facade is an authority-minimal API, not an in-process security sandbox.
    Untrusted Agents must remain behind an Adapter-owned process/IPC boundary and
    receive only the single answer returned by ``ask``.
    """

    __slots__ = (
        "__channel",
        "__consumed_answer_ids",
        "__lock",
        "__matcher",
        "__match_receipts",
        "__question_ids",
        "__script",
        "__transcript",
    )

    def __init__(
        self,
        script: ClarificationScript,
        *,
        run_binding: AgentRunConfig,
        matcher_factory: MaterialClaimMatcherFactory | None = None,
    ) -> None:
        if not isinstance(script, ClarificationScript):
            raise SchemaError(
                "clarification session.script must be a ClarificationScript"
            )
        if not isinstance(run_binding, AgentRunConfig):
            raise TypeError(
                "clarification session requires a verified AgentRunConfig"
            )
        snapshot = run_binding.clarification_matcher
        expected_matcher_digest = _canonical._digest(
            run_binding.clarification_matcher_config_digest,
            "clarification matcher config digest",
        )
        if snapshot.digest() != expected_matcher_digest:
            raise ClarificationProtocolError(
                "clarification matcher snapshot does not match the Run binding"
            )
        selected_factory = matcher_factory or BuiltinMaterialClaimMatcherFactory()
        if not isinstance(selected_factory, MaterialClaimMatcherFactory):
            raise TypeError(
                "clarification matcher factory must implement MaterialClaimMatcherFactory"
            )
        matcher = selected_factory.build(snapshot)
        if not isinstance(matcher, MaterialClaimMatcher):
            raise TypeError("clarification matcher must implement MaterialClaimMatcher")
        if matcher.binding_digest != expected_matcher_digest:
            raise ClarificationProtocolError(
                "clarification matcher does not match the run binding"
            )
        self.__script = script
        self.__matcher = matcher
        self.__match_receipts: list[MaterialClaimMatchReceipt] = []
        self.__transcript: list[SubmissionClarificationExchange] = []
        self.__consumed_answer_ids: set[str] = set()
        self.__question_ids: set[str] = set()
        self.__lock = Lock()
        self.__channel: ClarificationChannel = _BoundClarificationChannel(self.__ask)

    @property
    def channel(self) -> ClarificationChannel:
        return self.__channel

    @property
    def transcript(self) -> Tuple[SubmissionClarificationExchange, ...]:
        with self.__lock:
            return tuple(self.__transcript)

    @property
    def consumed_answer_ids(self) -> FrozenSet[str]:
        with self.__lock:
            return frozenset(self.__consumed_answer_ids)

    @property
    def match_receipts(self) -> Tuple[MaterialClaimMatchReceipt, ...]:
        with self.__lock:
            return tuple(self.__match_receipts)

    def __ask(
        self,
        *,
        question_id: str,
        dimension: IntentDimension,
        question: str,
        material_claim: str,
        proposed_values: Sequence[str] = (),
    ) -> SubmissionClarificationExchange:
        _canonical._identifier(question_id, "clarification question.question_id")
        _canonical._require_enum(
            IntentDimension,
            dimension,
            "clarification question.dimension",
        )
        asked_text = _canonical._string(
            question,
            "clarification question.question",
            _canonical.MAX_QUESTION_CHARS,
        )
        asked_claim = _canonical._string(
            material_claim,
            "clarification question.material_claim",
            _canonical.MAX_CLAIM_CHARS,
        )
        proposed = _canonical._text_tuple(
            proposed_values,
            "clarification question.proposed_values",
            _canonical.MAX_TEXT_LIST_ITEMS,
            _canonical.MAX_ANSWER_CHARS,
        )

        with self.__lock:
            if question_id in self.__question_ids:
                raise ClarificationProtocolError(
                    "duplicate clarification question_id: %r" % question_id
                )
            if len(self.__transcript) >= MAX_CLARIFICATION_QUESTIONS:
                raise ClarificationProtocolError(
                    "canonical clarification question limit of %d exceeded"
                    % MAX_CLARIFICATION_QUESTIONS
                )

            turn_index = len(self.__transcript) + 1
            matched_answer = None
            if turn_index <= self.__script.max_rounds:
                matched_answer, match_receipt = self.__find_matching_answer(
                    turn_index=turn_index,
                    question_id=question_id,
                    dimension=dimension,
                    material_claim=asked_claim,
                    proposed_values=proposed,
                )
            else:
                match_receipt = self.__match_receipt(
                    turn_index=turn_index,
                    question_id=question_id,
                    dimension=dimension,
                    material_claim=asked_claim,
                    candidates=(),
                    outcome=MaterialClaimMatchOutcome.ROUND_LIMIT,
                    matched_answer_id=None,
                )
            exchange = self.__make_exchange(
                turn_index=turn_index,
                question_id=question_id,
                dimension=dimension,
                question=asked_text,
                material_claim=asked_claim,
                proposed_values=proposed,
                matched_answer=matched_answer,
            )
            self.__question_ids.add(question_id)
            if matched_answer is not None:
                self.__consumed_answer_ids.add(matched_answer.answer_id)
            self.__match_receipts.append(match_receipt)
            self.__transcript.append(exchange)
            return exchange

    def __find_matching_answer(
        self,
        *,
        turn_index: int,
        question_id: str,
        dimension: IntentDimension,
        material_claim: str,
        proposed_values: Tuple[str, ...],
    ) -> Tuple[ClarificationAnswer | None, MaterialClaimMatchReceipt]:
        matches = []
        decisions = []
        for scripted_answer in self.__script.answers:
            if scripted_answer.answer_id in self.__consumed_answer_ids:
                continue
            if scripted_answer.dimension is not dimension:
                continue
            try:
                equivalent = self.__matcher.equivalent(
                    dimension,
                    material_claim,
                    scripted_answer.material_claim,
                )
            except Exception as exc:
                raise ClarificationProtocolError(
                    "material claim matcher failed"
                ) from exc
            if type(equivalent) is not bool:
                raise ClarificationProtocolError(
                    "material claim matcher must return bool"
                )
            action_eligible = not (
                scripted_answer.action is ClarificationAction.CONFIRM
                and not proposed_values
            )
            decisions.append(
                MaterialClaimCandidateDecision(
                    answer_id=scripted_answer.answer_id,
                    request_digest=_canonical.canonical_sha256(
                        {
                            "matcher_digest": self.__matcher.binding_digest,
                            "dimension": dimension.value,
                            "actual_claim": material_claim,
                            "scripted_claim": scripted_answer.material_claim,
                            "answer_id": scripted_answer.answer_id,
                        }
                    ),
                    equivalent=equivalent,
                    action_eligible=action_eligible,
                )
            )
            if equivalent and action_eligible:
                matches.append(scripted_answer)
        matched = matches[0] if len(matches) == 1 else None
        outcome = (
            MaterialClaimMatchOutcome.MATCHED
            if matched is not None
            else (
                MaterialClaimMatchOutcome.AMBIGUOUS
                if len(matches) > 1
                else MaterialClaimMatchOutcome.UNMATCHED
            )
        )
        return matched, self.__match_receipt(
            turn_index=turn_index,
            question_id=question_id,
            dimension=dimension,
            material_claim=material_claim,
            candidates=tuple(decisions),
            outcome=outcome,
            matched_answer_id=(None if matched is None else matched.answer_id),
        )

    def __match_receipt(
        self,
        *,
        turn_index: int,
        question_id: str,
        dimension: IntentDimension,
        material_claim: str,
        candidates: Tuple[MaterialClaimCandidateDecision, ...],
        outcome: MaterialClaimMatchOutcome,
        matched_answer_id: str | None,
    ) -> MaterialClaimMatchReceipt:
        return MaterialClaimMatchReceipt(
            turn_index=turn_index,
            question_id=question_id,
            dimension=dimension,
            actual_claim_digest=_canonical.canonical_sha256(
                {
                    "dimension": dimension.value,
                    "material_claim": material_claim,
                }
            ),
            matcher_digest=self.__matcher.binding_digest,
            candidates=candidates,
            outcome=outcome,
            matched_answer_id=matched_answer_id,
        )

    @staticmethod
    def __make_exchange(
        *,
        turn_index: int,
        question_id: str,
        dimension: IntentDimension,
        question: str,
        material_claim: str,
        proposed_values: Tuple[str, ...],
        matched_answer: ClarificationAnswer | None,
    ) -> SubmissionClarificationExchange:
        if matched_answer is None:
            return SubmissionClarificationExchange(
                turn_index=turn_index,
                question_id=question_id,
                dimension=dimension,
                question=question,
                material_claim=material_claim,
                matched_answer_id=None,
                action=None,
                response=None,
                resolved_values=(),
            )
        if matched_answer.action is ClarificationAction.CONFIRM:
            resolved_values = proposed_values
        elif matched_answer.action is ClarificationAction.CORRECT:
            resolved_values = matched_answer.corrected_values
        else:
            resolved_values = ()
        return SubmissionClarificationExchange(
            turn_index=turn_index,
            question_id=question_id,
            dimension=dimension,
            question=question,
            material_claim=material_claim,
            matched_answer_id=matched_answer.answer_id,
            action=matched_answer.action,
            response=matched_answer.response,
            resolved_values=resolved_values,
        )


__all__ = [
    "BuiltinMaterialClaimMatcherFactory",
    "ClarificationChannel",
    "ClarificationProtocolError",
    "ClarificationSession",
    "MaterialClaimMatcher",
    "MaterialClaimMatcherFactory",
    "MaterialClaimCandidateDecision",
    "MaterialClaimMatchOutcome",
    "MaterialClaimMatchReceipt",
    "NormalizedMaterialClaimMatcher",
    "canonical_material_claim_matcher_snapshot",
]
