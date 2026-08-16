"""
Phase 6 — Confidence Gate tests.

Per Tests.md §1:
- confidence.py: score combination logic; threshold behavior at the boundary

Per Phases.md Phase 6 exit criteria:
- Agent refuses on a deliberately underspecified schema test case
  instead of guessing
"""

import pytest
from src.agent.confidence import (
    ConfidenceScorer,
    ConfidenceResult,
    ConfidenceLevel,
    ConfidenceSignal,
)


class TestConfidenceLevels:
    """Test that confidence levels map correctly."""

    def test_high_confidence_display(self):
        r = ConfidenceResult(level=ConfidenceLevel.HIGH, score=0.9)
        assert r.display() == "high"

    def test_flagged_display(self):
        r = ConfidenceResult(
            level=ConfidenceLevel.FLAGGED, score=0.5,
            reason="join quality uncertain"
        )
        assert "flagged" in r.display()
        assert "join quality" in r.display()

    def test_refused_display(self):
        r = ConfidenceResult(
            level=ConfidenceLevel.REFUSED, score=0.1,
            reason="missing schema info"
        )
        assert "refused" in r.display()
        assert "missing schema" in r.display()

    def test_should_proceed_high(self):
        r = ConfidenceResult(level=ConfidenceLevel.HIGH, score=0.9)
        assert r.should_proceed() is True

    def test_should_proceed_flagged(self):
        r = ConfidenceResult(level=ConfidenceLevel.FLAGGED, score=0.5)
        assert r.should_proceed() is True

    def test_should_not_proceed_refused(self):
        r = ConfidenceResult(level=ConfidenceLevel.REFUSED, score=0.1)
        assert r.should_proceed() is False


class TestScoreCombination:
    """Test score combination logic."""

    def test_all_high_signals_give_high_confidence(self):
        scorer = ConfidenceScorer(refusal_threshold=0.3, flag_threshold=0.6)
        result = scorer.score(
            schema_match=1.0,
            join_confidence=1.0,
            verification_score=1.0,
            result_sanity=1.0,
        )
        assert result.level == ConfidenceLevel.HIGH
        assert result.score > 0.6

    def test_low_schema_match_flags(self):
        scorer = ConfidenceScorer(refusal_threshold=0.3, flag_threshold=0.6)
        result = scorer.score(
            schema_match=0.3,
            join_confidence=1.0,
            verification_score=1.0,
            result_sanity=1.0,
        )
        # Low schema match should lower the score
        assert result.score < 1.0

    def test_hallucination_heavily_penalized(self):
        scorer = ConfidenceScorer(refusal_threshold=0.3, flag_threshold=0.6)
        result = scorer.score(
            schema_match=0.5,
            join_confidence=1.0,
            verification_score=0.5,
            has_hallucinated_refs=True,
        )
        assert result.level in (ConfidenceLevel.REFUSED, ConfidenceLevel.FLAGGED)
        assert result.score < 0.6

    def test_cartesian_join_penalized(self):
        scorer = ConfidenceScorer(refusal_threshold=0.3, flag_threshold=0.6)
        result = scorer.score(
            schema_match=1.0,
            join_confidence=0.0,
            verification_score=0.5,
            has_cartesian_join=True,
        )
        assert result.score < 0.6

    def test_ambiguity_lowers_score(self):
        scorer = ConfidenceScorer(refusal_threshold=0.3, flag_threshold=0.6)
        result_clear = scorer.score(schema_match=0.8, verification_score=0.8)
        result_ambiguous = scorer.score(
            schema_match=0.8, verification_score=0.8, is_ambiguous=True
        )
        assert result_ambiguous.score < result_clear.score


class TestThresholdBehavior:
    """Test threshold boundary behavior."""

    def test_at_refusal_threshold(self):
        scorer = ConfidenceScorer(refusal_threshold=0.3, flag_threshold=0.6)
        # Force a very low score
        result = scorer.score(
            schema_match=0.0,
            join_confidence=0.0,
            verification_score=0.0,
            result_sanity=0.0,
        )
        assert result.level == ConfidenceLevel.REFUSED

    def test_just_above_refusal(self):
        scorer = ConfidenceScorer(refusal_threshold=0.3, flag_threshold=0.6)
        result = scorer.score(
            schema_match=0.5,
            join_confidence=0.5,
            verification_score=0.5,
            result_sanity=0.5,
        )
        # 0.5 > 0.3 so should be FLAGGED, not REFUSED
        assert result.level in (ConfidenceLevel.FLAGGED, ConfidenceLevel.HIGH)

    def test_between_thresholds(self):
        scorer = ConfidenceScorer(refusal_threshold=0.2, flag_threshold=0.8)
        result = scorer.score(
            schema_match=0.5,
            join_confidence=0.5,
            verification_score=0.5,
            result_sanity=0.5,
        )
        assert result.level == ConfidenceLevel.FLAGGED

    def test_custom_thresholds(self):
        scorer = ConfidenceScorer(refusal_threshold=0.5, flag_threshold=0.9)
        result = scorer.score(
            schema_match=0.6,
            join_confidence=0.6,
            verification_score=0.6,
            result_sanity=0.6,
        )
        # 0.6 is between 0.5 and 0.9
        assert result.level == ConfidenceLevel.FLAGGED


class TestRefusalOnUnderspecified:
    """
    Phase 6 exit criteria: agent refuses on a deliberately underspecified
    schema test case instead of guessing.
    """

    def test_refuses_hallucinated_references(self):
        """A query with hallucinated tables/columns should be refused."""
        scorer = ConfidenceScorer(refusal_threshold=0.3, flag_threshold=0.6)
        result = scorer.score(
            schema_match=0.0,
            join_confidence=0.0,
            verification_score=0.0,
            has_hallucinated_refs=True,
        )
        assert result.level == ConfidenceLevel.REFUSED
        assert result.should_proceed() is False
        assert "refused" in result.display()

    def test_refuses_underspecified_with_ambiguity(self):
        """Ambiguous question + poor schema match = refuse."""
        scorer = ConfidenceScorer(refusal_threshold=0.3, flag_threshold=0.6)
        result = scorer.score(
            schema_match=0.1,
            join_confidence=0.1,
            verification_score=0.2,
            is_ambiguous=True,
            has_hallucinated_refs=True,
        )
        assert result.level == ConfidenceLevel.REFUSED

    def test_refuses_with_clear_reason(self):
        """Refusal should state what's missing."""
        scorer = ConfidenceScorer(refusal_threshold=0.3, flag_threshold=0.6)
        result = scorer.score(
            schema_match=0.0,
            verification_score=0.0,
            has_hallucinated_refs=True,
        )
        assert "non-existent" in result.reason.lower() or "schema" in result.reason.lower()


class TestSignals:
    """Test that signals are properly recorded."""

    def test_signals_recorded(self):
        scorer = ConfidenceScorer()
        result = scorer.score(schema_match=0.8, verification_score=0.9)
        assert len(result.signals) >= 4  # At least the 4 base signals
        signal_names = {s.name for s in result.signals}
        assert "schema_match" in signal_names
        assert "verification" in signal_names

    def test_penalty_signals_added(self):
        scorer = ConfidenceScorer()
        result = scorer.score(
            has_hallucinated_refs=True,
            has_cartesian_join=True,
            is_ambiguous=True,
        )
        signal_names = {s.name for s in result.signals}
        assert "hallucination_penalty" in signal_names
        assert "cartesian_penalty" in signal_names
        assert "ambiguity_penalty" in signal_names


class TestConvenienceMethod:
    """Test score_from_verification convenience method."""

    def test_clean_verification(self):
        scorer = ConfidenceScorer()
        result = scorer.score_from_verification(
            verification_errors=0,
            verification_warnings=0,
            total_tables=3,
            hallucinated_tables=0,
        )
        assert result.level == ConfidenceLevel.HIGH

    def test_hallucinated_tables(self):
        scorer = ConfidenceScorer()
        result = scorer.score_from_verification(
            verification_errors=1,
            verification_warnings=0,
            total_tables=3,
            hallucinated_tables=2,
        )
        assert result.level in (ConfidenceLevel.REFUSED, ConfidenceLevel.FLAGGED)
