import pytest
from src.agent.calibration_eval import CalibrationEvaluator


def test_perfect_calibration():
    evaluator = CalibrationEvaluator(num_bins=5)
    # 10 cases with confidence 0.9 and all 10 correct
    preds = [{"confidence": 0.9, "correct": True} for _ in range(10)]
    report = evaluator.evaluate(preds)
    assert report.total_samples == 10
    # In bin [0.8, 1.0], avg conf is 0.9, accuracy is 1.0 -> gap = 0.1
    assert report.ece <= 0.15
    assert report.brier_score == pytest.approx(0.01)


def test_selective_classification_sweep():
    evaluator = CalibrationEvaluator()
    # Mixed predictions
    preds = [
        {"confidence": 0.95, "correct": True},
        {"confidence": 0.85, "correct": True},
        {"confidence": 0.70, "correct": True},
        {"confidence": 0.40, "correct": False},
        {"confidence": 0.20, "correct": False},
    ]
    report = evaluator.evaluate(preds)
    assert len(report.curve_points) == 20

    # At tau = 0.0, coverage should be 1.0, accuracy should be 3/5 = 0.60
    point_0 = next(p for p in report.curve_points if p.threshold == 0.0)
    assert point_0.coverage == 1.0
    assert point_0.selective_accuracy == 0.6

    # At tau = 0.65, only 3 high confidence correct queries accepted -> selective acc = 1.0!
    point_65 = next(p for p in report.curve_points if p.threshold == 0.65)
    assert point_65.selective_accuracy == 1.0
    assert point_65.coverage == 0.60
    assert point_65.false_accept_rate == 0.0


def test_ascii_report_formatting():
    evaluator = CalibrationEvaluator()
    preds = [
        {"confidence": 0.9, "correct": True},
        {"confidence": 0.3, "correct": False},
    ]
    report = evaluator.evaluate(preds)
    text = report.format_ascii_report()
    assert "CALIBRATED REFUSAL BENCHMARK" in text
    assert "Expected Calibration Error" in text
    assert "Selective Classification" in text
