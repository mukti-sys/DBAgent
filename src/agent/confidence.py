"""
Confidence scorer — combines schema-match / join-confidence /
result-sanity into one gate decision.

Per Architecture.md §2: Combines schema-match / join-confidence /
result-sanity into one gate decision.

Per Rules.md §2: If confidence is below threshold, refuse or flag —
do not answer anyway and hope.

Per Design.md §2: Confidence levels are 'high', 'flagged: <reason>',
'refused: <reason>'.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.config import load_settings

logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Confidence levels per Design.md §2."""
    HIGH = "high"
    FLAGGED = "flagged"
    REFUSED = "refused"


@dataclass
class ConfidenceSignal:
    """A single signal contributing to overall confidence."""
    name: str
    score: float  # 0.0 - 1.0
    weight: float = 1.0
    reason: str = ""


@dataclass
class ConfidenceResult:
    """The overall confidence gate decision."""
    level: ConfidenceLevel
    score: float  # 0.0 - 1.0 composite score
    signals: list[ConfidenceSignal] = field(default_factory=list)
    reason: str = ""

    def display(self) -> str:
        """Format confidence for display per Design.md §2."""
        if self.level == ConfidenceLevel.HIGH:
            return "high"
        elif self.level == ConfidenceLevel.FLAGGED:
            return f"flagged: {self.reason}"
        else:
            return f"refused: {self.reason}"

    def should_proceed(self) -> bool:
        """Whether to proceed with execution (high or flagged, not refused)."""
        return self.level != ConfidenceLevel.REFUSED


class ConfidenceScorer:
    """
    Combines multiple confidence signals into a single gate decision.

    Signals:
    - schema_match: how well the query's tables/columns match the schema
    - join_confidence: quality and safety of join conditions
    - result_sanity: whether result patterns look reasonable
    - verification: results from the verifier

    Per Rules.md §2: refuse below threshold, flag for uncertainty,
    never guess silently.
    """

    def __init__(
        self,
        refusal_threshold: float | None = None,
        flag_threshold: float | None = None,
    ):
        settings = load_settings()
        conf = settings.get("confidence", {})
        self._refusal_threshold = refusal_threshold or conf.get("refusal_threshold", 0.3)
        self._flag_threshold = flag_threshold or conf.get("flag_threshold", 0.6)

    def score(
        self,
        schema_match: float = 1.0,
        join_confidence: float = 1.0,
        verification_score: float = 1.0,
        result_sanity: float = 1.0,
        has_hallucinated_refs: bool = False,
        has_cartesian_join: bool = False,
        is_ambiguous: bool = False,
    ) -> ConfidenceResult:
        """
        Compute composite confidence from individual signals.

        Args:
            schema_match: How well query tables/columns match schema (0-1).
            join_confidence: Quality of join conditions (0-1).
            verification_score: Verifier result (0-1, 0 if errors found).
            result_sanity: Whether results look reasonable (0-1).
            has_hallucinated_refs: Whether hallucinated tables/columns found.
            has_cartesian_join: Whether a cartesian join was detected.
            is_ambiguous: Whether the question was ambiguous.

        Returns:
            ConfidenceResult with level, score, and reason.
        """
        signals = []

        # Schema match signal
        signals.append(ConfidenceSignal(
            name="schema_match",
            score=schema_match,
            weight=2.0,
            reason="How well query references match known schema",
        ))

        # Join confidence signal
        signals.append(ConfidenceSignal(
            name="join_confidence",
            score=join_confidence,
            weight=1.5,
            reason="Quality and safety of join conditions",
        ))

        # Verification signal
        signals.append(ConfidenceSignal(
            name="verification",
            score=verification_score,
            weight=2.0,
            reason="Static verification result",
        ))

        # Result sanity signal
        signals.append(ConfidenceSignal(
            name="result_sanity",
            score=result_sanity,
            weight=1.0,
            reason="Whether results look reasonable",
        ))

        # Hard penalties for critical issues
        if has_hallucinated_refs:
            signals.append(ConfidenceSignal(
                name="hallucination_penalty",
                score=0.0,
                weight=3.0,
                reason="Hallucinated table/column references detected",
            ))

        if has_cartesian_join:
            signals.append(ConfidenceSignal(
                name="cartesian_penalty",
                score=0.0,
                weight=2.5,
                reason="Cartesian join detected",
            ))

        if is_ambiguous:
            signals.append(ConfidenceSignal(
                name="ambiguity_penalty",
                score=0.3,
                weight=1.5,
                reason="Question is ambiguous",
            ))

        # Compute weighted average
        total_weight = sum(s.weight for s in signals)
        if total_weight == 0:
            composite = 0.0
        else:
            composite = sum(s.score * s.weight for s in signals) / total_weight

        # Determine level
        reasons = []
        if has_hallucinated_refs:
            reasons.append("references to non-existent schema elements")
        if has_cartesian_join:
            reasons.append("unsafe cartesian join detected")
        if is_ambiguous:
            reasons.append("question is ambiguous")
        if verification_score < 0.5:
            reasons.append("verification found issues")
        if schema_match < 0.5:
            reasons.append("poor schema match")

        if composite <= self._refusal_threshold:
            level = ConfidenceLevel.REFUSED
            reason = "Confidence too low to proceed: " + "; ".join(reasons) if reasons else "Composite score below refusal threshold"
        elif composite <= self._flag_threshold:
            level = ConfidenceLevel.FLAGGED
            reason = "; ".join(reasons) if reasons else "Composite score below high-confidence threshold"
        else:
            level = ConfidenceLevel.HIGH
            reason = ""

        return ConfidenceResult(
            level=level,
            score=round(composite, 3),
            signals=signals,
            reason=reason,
        )

    def score_from_verification(
        self,
        verification_errors: int,
        verification_warnings: int,
        total_tables: int,
        hallucinated_tables: int,
        is_ambiguous: bool = False,
    ) -> ConfidenceResult:
        """
        Convenience method to score from verification results.

        Translates verification counts into the signal-based scoring.
        """
        schema_match = 1.0 - (hallucinated_tables / max(total_tables, 1))
        verification_score = 1.0 if verification_errors == 0 else max(0.0, 1.0 - verification_errors * 0.4)
        join_confidence = 0.8 if verification_warnings > 0 else 1.0

        return self.score(
            schema_match=schema_match,
            join_confidence=join_confidence,
            verification_score=verification_score,
            has_hallucinated_refs=hallucinated_tables > 0,
            has_cartesian_join=verification_errors > 0 and hallucinated_tables == 0,
            is_ambiguous=is_ambiguous,
        )
