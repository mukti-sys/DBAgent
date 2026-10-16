"""
Unit tests for Counterfactual Verification.
"""

from unittest.mock import MagicMock
from src.agent.counterfactual import (
    CounterfactualVerifier,
    CounterfactualResult,
)


def test_fast_path_skip_simple_query():
    verifier = CounterfactualVerifier()
    # Simple single table, high confidence -> should fast-path skip
    sql = "SELECT CustomerId, FirstName FROM Customer"
    assert verifier.should_verify(sql, confidence_score=0.95, uncertainty_score=0.05) is False
    res = verifier.verify(sql, confidence_score=0.95, uncertainty_score=0.05)
    assert res.skipped_fast_path is True
    assert res.executed is False
    assert res.passed is True


def test_complex_query_triggers_verification():
    verifier = CounterfactualVerifier()
    # Join + aggregation -> must verify
    sql = "SELECT c.CustomerId, SUM(i.Total) FROM Customer c JOIN Invoice i ON c.CustomerId = i.CustomerId GROUP BY c.CustomerId"
    assert verifier.should_verify(sql, confidence_score=0.90) is True


def test_low_confidence_triggers_verification():
    verifier = CounterfactualVerifier()
    # Simple table but low confidence in warning band -> must verify
    sql = "SELECT CustomerId FROM Customer WHERE Country = 'USA'"
    assert verifier.should_verify(sql, confidence_score=0.70) is True


def test_drop_filter_monotonicity_passed():
    mock_executor = MagicMock()
    # Mocking rows returned: original count = 5, when filter dropped = 15
    mock_executor.execute.side_effect = [
        MagicMock(rows=[{}] * 15, error=None),  # drop filter result
        MagicMock(rows=[{}] * 10, error=None),  # invert filter result
    ]

    verifier = CounterfactualVerifier(executor=mock_executor)
    sql = "SELECT CustomerId FROM Customer WHERE Country = 'USA'"
    res = verifier.verify(sql, original_rows=[{}] * 5, confidence_score=0.6)

    assert res.executed is True
    assert res.passed is True
    assert len(res.checks) >= 1
    drop_chk = next(c for c in res.checks if c.mutation_type == "drop_filter")
    assert drop_chk.passed is True
    assert drop_chk.original_count == 5
    assert drop_chk.mutated_count == 15


def test_drop_filter_monotonicity_failed_flags_bug():
    mock_executor = MagicMock()
    # Original has 10 rows, but dropping filter mysteriously returned 2 rows
    mock_executor.execute.side_effect = [
        MagicMock(rows=[{}] * 2, error=None),
        MagicMock(rows=[{}] * 8, error=None),
    ]

    verifier = CounterfactualVerifier(executor=mock_executor)
    sql = "SELECT CustomerId FROM Customer WHERE Country = 'USA'"
    res = verifier.verify(sql, original_rows=[{}] * 10, confidence_score=0.6)

    assert res.executed is True
    assert res.passed is False
    assert any("Monotonicity violation" in f for f in res.findings)


def test_non_select_refused_for_safety():
    verifier = CounterfactualVerifier()
    sql = "DELETE FROM Customer WHERE CustomerId = 1"
    res = verifier.verify(sql, confidence_score=0.5)
    assert res.executed is False
    assert res.passed is False
    assert "not a SELECT" in res.findings[0]
