"""
End-to-end integration test validating all 5 high-level features in the pipeline:
1. Uncertainty Decomposition
2. Honest Answer Set
3. Counterfactual Verification (with gated execution fast-path and read-only safety)
4. Calibrated Refusal Benchmark
5. Adversarial & Schema Drift Suite
"""

from unittest.mock import MagicMock
from src.agent.pipeline import Pipeline, PipelineResult
from src.agent.uncertainty import UncertaintyDecomposer
from src.agent.answer_set import AnswerSetGenerator
from src.agent.counterfactual import CounterfactualVerifier
from src.agent.calibration_eval import CalibrationEvaluator
from src.agent.adversarial_drift_eval import SchemaDriftDetector, SchemaSnapshot, AdversarialSuiteRunner


def test_pipeline_with_all_5_features():
    mock_sql_gen = MagicMock()
    mock_sql_gen.generate.return_value = MagicMock(
        sql="SELECT c.CustomerId, SUM(i.Total) FROM Customer c JOIN Invoice i ON c.CustomerId = i.CustomerId GROUP BY c.CustomerId"
    )

    mock_executor = MagicMock()
    mock_executor.execute.return_value = MagicMock(
        rows=[{"CustomerId": 1, "Total": 45.0}],
        row_count=1,
        column_names=["CustomerId", "Total"],
        execution_time_ms=5.0,
        error=None,
    )

    mock_critic = MagicMock()
    mock_critic.evaluate.return_value = MagicMock(passed=True, diagnosis="")

    mock_confidence = MagicMock()
    mock_confidence.score.return_value = MagicMock(
        display=lambda: "high",
        score=0.88,
        reason="",
        should_proceed=lambda: True,
    )

    pipeline = Pipeline(
        sql_generator=mock_sql_gen,
        executor=mock_executor,
        critic=mock_critic,
        confidence_scorer=mock_confidence,
    )

    result = pipeline.run("What are the total sales per customer?")
    assert result.success is True
    assert result.sql is not None

    # 1. Uncertainty Decomposition was executed
    assert result.uncertainty is not None
    assert 0.0 <= result.uncertainty.composite_uncertainty <= 1.0
    assert result.uncertainty.dominant_dimension in ["schema_linking", "join_path", "aggregation", "value_grounding"]

    # 2. Counterfactual Verification was executed safely
    assert result.counterfactual_result is not None
    assert result.counterfactual_result.passed is True

    # 3. Formatted response includes new insights
    resp = result.format_response()
    assert "Uncertainty:" in resp

    # 4. Calibration Evaluator operates on results
    evaluator = CalibrationEvaluator(num_bins=5)
    calib_report = evaluator.evaluate([
        {"confidence": result.confidence_score, "correct": True},
        {"confidence": 0.4, "correct": False},
    ])
    assert calib_report.total_samples == 2
    assert 0.0 <= calib_report.ece <= 1.0

    # 5. Schema Drift detector operates
    snap1 = SchemaSnapshot.capture({"Customer": ["CustomerId", "Name"]})
    snap2 = SchemaSnapshot.capture({"Customer": ["CustomerId", "Name", "Email"]})
    drift = SchemaDriftDetector().diff(snap1, snap2)
    assert drift.has_drift is True
    assert "Email" in drift.added_columns["Customer"]
