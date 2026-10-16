"""
Unit tests for Honest Answer Set mode.
"""

from unittest.mock import MagicMock
from src.agent.answer_set import (
    AnswerSetGenerator,
    HonestAnswerSet,
    SQLCandidate,
)
from src.agent.uncertainty import (
    DimensionUncertainty,
    TargetedClarification,
    UncertaintyDecomposition,
)


def make_dummy_uncertainty(dominant="schema_linking", unc_score=0.55):
    d = DimensionUncertainty(dominant, unc_score, 1 - unc_score, ["ambiguity"], ["opt1", "opt2"])
    other = DimensionUncertainty("other", 0.1, 0.9)
    return UncertaintyDecomposition(
        schema_linking=d if dominant == "schema_linking" else other,
        join_path=d if dominant == "join_path" else other,
        aggregation=d if dominant == "aggregation" else other,
        value_grounding=d if dominant == "value_grounding" else other,
        composite_uncertainty=unc_score,
        composite_confidence=1 - unc_score,
        dominant_dimension=dominant,
        needs_clarification=True,
        clarification=TargetedClarification(dominant, "Which one?"),
    )


def test_should_generate_answer_set_on_ambiguity():
    gen = AnswerSetGenerator()
    assert gen.should_generate_answer_set(0.9, is_ambiguous=True) is True
    assert gen.should_generate_answer_set(0.5, is_ambiguous=False) is True  # Medium band
    assert gen.should_generate_answer_set(0.9, is_ambiguous=False) is False  # High confidence


def test_build_answer_set_with_explicit_alternatives():
    gen = AnswerSetGenerator()
    primary_sql = "SELECT CustomerId, SUM(Total) FROM Invoice GROUP BY CustomerId"
    alts = [
        {
            "title": "By Count of Orders",
            "sql": "SELECT CustomerId, COUNT(*) FROM Invoice GROUP BY CustomerId",
            "description": "Calculates order frequency instead of dollar total",
            "probability": 0.35,
        }
    ]
    res = gen.build_answer_set(
        question="Who are the top customers?",
        primary_sql=primary_sql,
        alternative_sqls=alts,
    )
    assert res.has_alternatives is True
    assert len(res.candidates) == 2
    assert res.best_candidate().sql == primary_sql
    display = res.format_display()
    assert "Option 1" in display
    assert "Option 2" in display
    assert "⭐ (Recommended)" in display


def test_build_answer_set_with_executor():
    gen = AnswerSetGenerator()
    mock_executor = MagicMock()
    mock_res = MagicMock()
    mock_res.rows = [{"val": 100}]
    mock_res.row_count = 1
    mock_res.column_names = ["val"]
    mock_res.error = None
    mock_res.execution_time_ms = 4.2
    mock_executor.execute.return_value = mock_res

    alts = [{"title": "Alt", "sql": "SELECT 2", "probability": 0.3}]
    res = gen.build_answer_set(
        question="test",
        primary_sql="SELECT 1",
        alternative_sqls=alts,
        executor=mock_executor,
    )
    assert len(res.candidates) == 2
    assert res.candidates[0].row_count == 1
    assert res.candidates[1].row_count == 1
    mock_executor.execute.assert_called()


def test_answer_set_to_dict():
    gen = AnswerSetGenerator()
    res = gen.build_answer_set(
        question="test",
        primary_sql="SELECT 1",
        alternative_sqls=[{"title": "Alt", "sql": "SELECT 2", "probability": 0.4}],
    )
    d = res.to_dict()
    assert d["question"] == "test"
    assert len(d["candidates"]) == 2
    assert d["candidates"][0]["sql"] == "SELECT 1"
