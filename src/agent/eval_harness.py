"""
Eval harness — hand-labeled eval set with execution-match + semantic-match
scoring, producing accuracy and calibration reports.

Per Phases.md Phase 12:
- Hand-labeled eval set (grows over time)
- Execution-match + manual semantic-match scoring
- Exit criteria: harness runs end-to-end and produces an accuracy +
  calibration report

Per Tests.md §4:
- Easy tier (clear questions, clean schema) and hard tier (ambiguous
  questions, messy schema, including questions that should trigger refusal)
- Track accuracy and calibration over time
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EvalCase:
    """A single eval test case."""
    case_id: str
    tier: str                       # "easy" or "hard"
    question: str
    expected_sql: str | None        # None for questions that should be refused
    expected_answer: Any | None     # The expected answer value
    accepted_equivalents: list[str] = field(default_factory=list)
    should_refuse: bool = False     # True if agent should refuse this question
    should_flag: bool = False       # True if agent should flag (not refuse)
    tags: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class EvalResult:
    """Result of evaluating a single case."""
    case_id: str
    tier: str
    question: str
    passed: bool
    match_type: str                 # "execution_match", "semantic_match", "refusal_correct", "refusal_incorrect", etc.
    generated_sql: str | None = None
    expected_sql: str | None = None
    generated_answer: Any | None = None
    expected_answer: Any | None = None
    confidence_level: str | None = None
    was_refused: bool = False
    was_flagged: bool = False
    details: str = ""


@dataclass
class EvalReport:
    """Aggregate eval report with accuracy and calibration metrics."""
    total_cases: int
    passed: int
    failed: int
    accuracy: float

    # Per-tier breakdown
    easy_total: int = 0
    easy_passed: int = 0
    easy_accuracy: float = 0.0
    hard_total: int = 0
    hard_passed: int = 0
    hard_accuracy: float = 0.0

    # Calibration metrics
    total_refusals: int = 0
    correct_refusals: int = 0        # refused and should have refused
    incorrect_refusals: int = 0      # refused but answer was correct
    missed_refusals: int = 0         # didn't refuse but should have
    total_flags: int = 0
    correct_flags: int = 0

    # Detailed results
    results: list[EvalResult] = field(default_factory=list)

    def calibration_score(self) -> float:
        """How well-calibrated is the agent's confidence?
        High = good: refuses when it should, answers when it can."""
        total_decisions = self.correct_refusals + self.incorrect_refusals + self.missed_refusals
        if total_decisions == 0:
            return 1.0  # No refusal decisions to evaluate
        return self.correct_refusals / total_decisions if total_decisions > 0 else 0.0

    def format_report(self) -> str:
        """Format a human-readable report."""
        lines = [
            "=" * 60,
            "DBAgent Eval Report",
            "=" * 60,
            "",
            f"Total cases:     {self.total_cases}",
            f"Passed:          {self.passed}",
            f"Failed:          {self.failed}",
            f"Accuracy:        {self.accuracy:.1%}",
            "",
            "--- Tier Breakdown ---",
            f"Easy:   {self.easy_passed}/{self.easy_total} ({self.easy_accuracy:.1%})",
            f"Hard:   {self.hard_passed}/{self.hard_total} ({self.hard_accuracy:.1%})",
            "",
            "--- Calibration ---",
            f"Correct refusals:    {self.correct_refusals}",
            f"Incorrect refusals:  {self.incorrect_refusals}",
            f"Missed refusals:     {self.missed_refusals}",
            f"Calibration score:   {self.calibration_score():.1%}",
            "",
            "--- Details ---",
        ]

        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"  [{status}] {r.case_id} ({r.tier}): {r.match_type}")
            if not r.passed:
                lines.append(f"         {r.details}")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)


class EvalHarness:
    """
    Runs eval cases against the agent and produces accuracy/calibration reports.

    Per Tests.md §4: score via execution-match, noted honestly as imperfect —
    two different correct SQL statements can both pass, or a semantically-wrong
    query can accidentally match.
    """

    def __init__(self, eval_cases: list[EvalCase] | None = None):
        self._cases = eval_cases or []

    @property
    def case_count(self) -> int:
        return len(self._cases)

    def add_case(self, case: EvalCase) -> None:
        self._cases.append(case)

    def load_cases_from_json(self, path: str | Path) -> None:
        """Load eval cases from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            self._cases.append(EvalCase(**item))

    def evaluate(
        self,
        run_fn=None,
    ) -> EvalReport:
        """
        Run all eval cases and produce a report.

        Args:
            run_fn: Optional callable(question) -> dict with keys:
                - sql: generated SQL string
                - answer: computed answer
                - confidence: "high", "flagged: ...", "refused: ..."
                - was_refused: bool
                - was_flagged: bool
                If None, uses a mock runner for harness testing.
        """
        results: list[EvalResult] = []

        for case in self._cases:
            if run_fn:
                output = run_fn(case.question)
            else:
                # Mock runner for harness-level testing
                output = self._mock_run(case)

            result = self._evaluate_case(case, output)
            results.append(result)

        return self._build_report(results)

    def _evaluate_case(self, case: EvalCase, output: dict) -> EvalResult:
        """Evaluate a single case against its expected outcome."""
        generated_sql = output.get("sql")
        generated_answer = output.get("answer")
        confidence = output.get("confidence", "")
        was_refused = output.get("was_refused", False)
        was_flagged = output.get("was_flagged", False)

        # Case 1: Should have refused
        if case.should_refuse:
            if was_refused:
                return EvalResult(
                    case_id=case.case_id,
                    tier=case.tier,
                    question=case.question,
                    passed=True,
                    match_type="refusal_correct",
                    generated_sql=generated_sql,
                    expected_sql=case.expected_sql,
                    confidence_level=confidence,
                    was_refused=True,
                    details="Correctly refused an unanswerable question.",
                )
            else:
                return EvalResult(
                    case_id=case.case_id,
                    tier=case.tier,
                    question=case.question,
                    passed=False,
                    match_type="refusal_missed",
                    generated_sql=generated_sql,
                    expected_sql=case.expected_sql,
                    generated_answer=generated_answer,
                    confidence_level=confidence,
                    was_refused=False,
                    details="Should have refused but attempted an answer.",
                )

        # Case 2: Should not have refused
        if was_refused and not case.should_refuse:
            return EvalResult(
                case_id=case.case_id,
                tier=case.tier,
                question=case.question,
                passed=False,
                match_type="refusal_incorrect",
                generated_sql=generated_sql,
                expected_sql=case.expected_sql,
                confidence_level=confidence,
                was_refused=True,
                details="Refused but should have answered.",
            )

        # Case 3: Check execution match
        if case.expected_answer is not None and generated_answer is not None:
            if self._answers_match(generated_answer, case.expected_answer):
                return EvalResult(
                    case_id=case.case_id,
                    tier=case.tier,
                    question=case.question,
                    passed=True,
                    match_type="execution_match",
                    generated_sql=generated_sql,
                    expected_sql=case.expected_sql,
                    generated_answer=generated_answer,
                    expected_answer=case.expected_answer,
                    confidence_level=confidence,
                    was_flagged=was_flagged,
                    details="Answer matches expected value.",
                )
            else:
                # Answer mismatch — fail even if SQL matches.
                # Per Tests.md §4: a semantically-wrong query can accidentally
                # match on SQL; answer is the definitive check.
                return EvalResult(
                    case_id=case.case_id,
                    tier=case.tier,
                    question=case.question,
                    passed=False,
                    match_type="answer_mismatch",
                    generated_sql=generated_sql,
                    expected_sql=case.expected_sql,
                    generated_answer=generated_answer,
                    expected_answer=case.expected_answer,
                    confidence_level=confidence,
                    was_flagged=was_flagged,
                    details=f"Answer mismatch: got {generated_answer}, expected {case.expected_answer}",
                )

        # Case 4: Check SQL semantic match (only when no expected_answer to compare)
        if case.expected_sql and generated_sql:
            if self._sql_matches(generated_sql, case.expected_sql, case.accepted_equivalents):
                return EvalResult(
                    case_id=case.case_id,
                    tier=case.tier,
                    question=case.question,
                    passed=True,
                    match_type="semantic_match",
                    generated_sql=generated_sql,
                    expected_sql=case.expected_sql,
                    generated_answer=generated_answer,
                    expected_answer=case.expected_answer,
                    confidence_level=confidence,
                    was_flagged=was_flagged,
                    details="SQL matches expected or accepted equivalent.",
                )

        # Case 5: No match
        return EvalResult(
            case_id=case.case_id,
            tier=case.tier,
            question=case.question,
            passed=False,
            match_type="no_match",
            generated_sql=generated_sql,
            expected_sql=case.expected_sql,
            generated_answer=generated_answer,
            expected_answer=case.expected_answer,
            confidence_level=confidence,
            was_flagged=was_flagged,
            details=f"Generated SQL/answer did not match expected. Got SQL: {generated_sql}",
        )

    def _answers_match(self, generated: Any, expected: Any) -> bool:
        """Check if generated answer matches expected answer."""
        # Exact match
        if generated == expected:
            return True

        # Numeric comparison with tolerance
        try:
            gen_num = float(generated)
            exp_num = float(expected)
            if abs(gen_num - exp_num) < 0.01:
                return True
        except (ValueError, TypeError):
            pass

        # String comparison (case-insensitive)
        if str(generated).strip().lower() == str(expected).strip().lower():
            return True

        return False

    def _sql_matches(
        self,
        generated: str,
        expected: str,
        equivalents: list[str],
    ) -> bool:
        """Check if generated SQL matches expected or any accepted equivalent."""
        gen_norm = self._normalize_sql(generated)
        exp_norm = self._normalize_sql(expected)

        if gen_norm == exp_norm:
            return True

        for equiv in equivalents:
            if gen_norm == self._normalize_sql(equiv):
                return True

        return False

    @staticmethod
    def _normalize_sql(sql: str) -> str:
        """Normalize SQL for comparison (lowercase, strip whitespace)."""
        import re
        sql = sql.strip().lower()
        sql = re.sub(r'\s+', ' ', sql)
        sql = sql.rstrip(';')
        return sql

    def _mock_run(self, case: EvalCase) -> dict:
        """Mock runner that returns expected values for harness testing."""
        if case.should_refuse:
            return {
                "sql": None,
                "answer": None,
                "confidence": "refused: insufficient schema",
                "was_refused": True,
                "was_flagged": False,
            }

        return {
            "sql": case.expected_sql,
            "answer": case.expected_answer,
            "confidence": "high" if not case.should_flag else "flagged: uncertain",
            "was_refused": False,
            "was_flagged": case.should_flag,
        }

    def _build_report(self, results: list[EvalResult]) -> EvalReport:
        """Build aggregate report from individual results."""
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed

        easy_results = [r for r in results if r.tier == "easy"]
        hard_results = [r for r in results if r.tier == "hard"]

        easy_passed = sum(1 for r in easy_results if r.passed)
        hard_passed = sum(1 for r in hard_results if r.passed)

        # Calibration
        correct_refusals = sum(1 for r in results if r.match_type == "refusal_correct")
        incorrect_refusals = sum(1 for r in results if r.match_type == "refusal_incorrect")
        missed_refusals = sum(1 for r in results if r.match_type == "refusal_missed")
        total_refusals = correct_refusals + incorrect_refusals
        total_flags = sum(1 for r in results if r.was_flagged)
        correct_flags = sum(1 for r in results if r.was_flagged and r.passed)

        return EvalReport(
            total_cases=total,
            passed=passed,
            failed=failed,
            accuracy=passed / total if total > 0 else 0.0,
            easy_total=len(easy_results),
            easy_passed=easy_passed,
            easy_accuracy=easy_passed / len(easy_results) if easy_results else 0.0,
            hard_total=len(hard_results),
            hard_passed=hard_passed,
            hard_accuracy=hard_passed / len(hard_results) if hard_results else 0.0,
            total_refusals=total_refusals,
            correct_refusals=correct_refusals,
            incorrect_refusals=incorrect_refusals,
            missed_refusals=missed_refusals,
            total_flags=total_flags,
            correct_flags=correct_flags,
            results=results,
        )
