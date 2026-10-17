"""
Calibrated Refusal Benchmark — computes rigorous calibration metrics and
accuracy-refusal curves (selective classification) for Text-to-SQL agents.

Metrics computed:
1. Expected Calibration Error (ECE):
   Partition predictions into M confidence bins and compute:
   ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
2. Maximum Calibration Error (MCE):
   MCE = max_{m=1}^M |acc(B_m) - conf(B_m)|
3. Accuracy-Refusal Tradeoff (Selective Classification):
   Sweeps refusal thresholds tau in [0.0, 1.0] to compute:
   - Coverage(tau): fraction of queries answered
   - Selective Accuracy(tau): accuracy on answered subset
   - False Acceptance Rate (FAR): fraction of incorrect answers accepted
4. ASCII Reliability Diagram & Curve Generation for terminal/paper reporting.
"""

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CalibrationBin:
    """A single confidence bin in the reliability diagram."""
    bin_index: int
    bin_lower: float
    bin_upper: float
    sample_count: int = 0
    correct_count: int = 0
    avg_confidence: float = 0.0
    accuracy: float = 0.0

    @property
    def calibration_gap(self) -> float:
        return abs(self.accuracy - self.avg_confidence)


@dataclass
class AccuracyRefusalPoint:
    """A point on the Accuracy-Refusal curve (Selective Classification)."""
    threshold: float
    coverage: float            # Fraction answered
    selective_accuracy: float  # Accuracy on answered
    refusal_rate: float        # 1.0 - coverage
    false_accept_rate: float   # Answered but incorrect / total


@dataclass
class CalibrationReport:
    """Complete calibration evaluation report."""
    total_samples: int
    ece: float  # Expected Calibration Error (0.0 to 1.0)
    mce: float  # Maximum Calibration Error (0.0 to 1.0)
    brier_score: float
    bins: list[CalibrationBin] = field(default_factory=list)
    curve_points: list[AccuracyRefusalPoint] = field(default_factory=list)
    optimal_refusal_threshold: float = 0.40

    def format_ascii_report(self) -> str:
        """Render a clean, formatted calibration report with ASCII visualization."""
        lines = [
            "=" * 64,
            "           DBAGENT CALIBRATED REFUSAL BENCHMARK REPORT          ",
            "=" * 64,
            f" Total Evaluated Cases:       {self.total_samples}",
            f" Expected Calibration Error:  {self.ece:.1%} (ECE - Lower is better)",
            f" Maximum Calibration Error:   {self.mce:.1%} (MCE)",
            f" Brier Score:                 {self.brier_score:.4f}",
            f" Optimal Refusal Threshold:   {self.optimal_refusal_threshold:.2f}",
            "",
            "--- Reliability Diagram (Binned Calibration) ---",
            f"{'Bin':<12} | {'Count':<7} | {'Avg Conf':<10} | {'Accuracy':<10} | {'Gap':<8}",
            "-" * 64,
        ]

        for b in self.bins:
            if b.sample_count > 0:
                bin_label = f"[{b.bin_lower:.1f}, {b.bin_upper:.1f})"
                lines.append(
                    f"{bin_label:<12} | {b.sample_count:<7} | {b.avg_confidence:<10.1%} | {b.accuracy:<10.1%} | {b.calibration_gap:<8.1%}"
                )

        lines.extend([
            "",
            "--- Selective Classification (Accuracy vs. Refusal Curve) ---",
            f"{'Threshold (τ)':<15} | {'Coverage':<10} | {'Selective Acc':<15} | {'FAR (Risk)':<10}",
            "-" * 64,
        ])

        # Sample every other point to keep table readable
        for p in self.curve_points[::2]:
            lines.append(
                f"τ = {p.threshold:<11.2f} | {p.coverage:<10.1%} | {p.selective_accuracy:<15.1%} | {p.false_accept_rate:<10.1%}"
            )

        # ASCII Curve Visualization
        lines.extend([
            "",
            "--- ASCII Accuracy-Coverage Tradeoff ---",
            " Acc  ^",
        ])
        for p in self.curve_points[::3]:
            bar_len = int(p.selective_accuracy * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            lines.append(f"{p.selective_accuracy:>5.0%} │ {bar} (Cov: {p.coverage:.0%}, τ={p.threshold:.2f})")
        lines.append("      └" + "─" * 32 + "> Refusal Strictness")

        return "\n".join(lines)


class CalibrationEvaluator:
    """
    Computes calibration metrics from predictions and ground-truth outcomes.
    """

    def __init__(self, num_bins: int = 10):
        self.num_bins = num_bins

    def evaluate(
        self,
        predictions: list[dict[str, Any]],
    ) -> CalibrationReport:
        """
        Evaluate calibration across a set of test cases.
        
        Each item in predictions should contain:
        - confidence: float in [0.0, 1.0]
        - correct: bool (whether the answer was correct)
        - was_refused: bool (optional)
        """
        n = len(predictions)
        if n == 0:
            return CalibrationReport(
                total_samples=0,
                ece=0.0,
                mce=0.0,
                brier_score=0.0,
            )

        # 1. Compute Brier Score: mean((conf - y)^2)
        brier_sum = sum((p["confidence"] - (1.0 if p["correct"] else 0.0)) ** 2 for p in predictions)
        brier_score = brier_sum / n

        # 2. Compute Bins for ECE and MCE
        bins: list[CalibrationBin] = []
        bin_width = 1.0 / self.num_bins

        for i in range(self.num_bins):
            lower = i * bin_width
            upper = (i + 1) * bin_width
            # Upper edge inclusive for the last bin
            if i == self.num_bins - 1:
                bin_preds = [p for p in predictions if lower <= p["confidence"] <= upper]
            else:
                bin_preds = [p for p in predictions if lower <= p["confidence"] < upper]

            count = len(bin_preds)
            if count > 0:
                correct = sum(1 for p in bin_preds if p["correct"])
                avg_conf = sum(p["confidence"] for p in bin_preds) / count
                accuracy = correct / count
            else:
                correct = 0
                avg_conf = (lower + upper) / 2.0
                accuracy = 0.0

            bins.append(
                CalibrationBin(
                    bin_index=i,
                    bin_lower=round(lower, 2),
                    bin_upper=round(upper, 2),
                    sample_count=count,
                    correct_count=correct,
                    avg_confidence=round(avg_conf, 3),
                    accuracy=round(accuracy, 3),
                )
            )

        # ECE: sum (|B_m| / N) * |acc(B_m) - conf(B_m)| for non-empty bins
        ece = sum((b.sample_count / n) * b.calibration_gap for b in bins if b.sample_count > 0)
        # MCE: max |acc(B_m) - conf(B_m)| for non-empty bins
        non_empty_gaps = [b.calibration_gap for b in bins if b.sample_count > 0]
        mce = max(non_empty_gaps) if non_empty_gaps else 0.0

        # 3. Accuracy-Refusal Curve (Selective Classification)
        thresholds = [round(t * 0.05, 2) for t in range(20)]  # 0.0, 0.05, ..., 0.95
        curve_points: list[AccuracyRefusalPoint] = []

        best_f1 = -1.0
        optimal_tau = 0.40

        for tau in thresholds:
            # Queries where confidence >= tau are accepted (answered)
            accepted = [p for p in predictions if p["confidence"] >= tau]
            coverage = len(accepted) / n
            refusal_rate = 1.0 - coverage

            if len(accepted) > 0:
                correct_accepted = sum(1 for p in accepted if p["correct"])
                selective_acc = correct_accepted / len(accepted)
                false_accepts = sum(1 for p in accepted if not p["correct"])
                far = false_accepts / n
            else:
                selective_acc = 1.0  # Perfect accuracy on empty set
                far = 0.0

            curve_points.append(
                AccuracyRefusalPoint(
                    threshold=tau,
                    coverage=round(coverage, 3),
                    selective_accuracy=round(selective_acc, 3),
                    refusal_rate=round(refusal_rate, 3),
                    false_accept_rate=round(far, 3),
                )
            )

            # Heuristic for optimal threshold: maximize harmonic mean of accuracy and coverage
            if coverage > 0.3 and selective_acc > 0.0:
                f_score = 2 * (selective_acc * coverage) / (selective_acc + coverage)
                if f_score > best_f1:
                    best_f1 = f_score
                    optimal_tau = tau

        return CalibrationReport(
            total_samples=n,
            ece=round(ece, 4),
            mce=round(mce, 4),
            brier_score=round(brier_score, 4),
            bins=bins,
            curve_points=curve_points,
            optimal_refusal_threshold=optimal_tau,
        )
