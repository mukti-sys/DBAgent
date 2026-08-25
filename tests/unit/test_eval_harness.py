"""
Phase 12 — Eval Harness unit tests.

Per Phases.md Phase 12 exit criteria:
- Harness runs end-to-end and produces an accuracy + calibration report
"""

import json
import pytest
from pathlib import Path

from src.agent.eval_harness import (
    EvalHarness,
    EvalCase,
    EvalResult,
    EvalReport,
)


@pytest.fixture
def sample_cases():
    return [
        EvalCase(
            case_id="easy_001",
            tier="easy",
            question="How many users are there?",
            expected_sql="SELECT COUNT(*) FROM users",
            expected_answer=42,
        ),
        EvalCase(
            case_id="easy_002",
            tier="easy",
            question="What is the total revenue?",
            expected_sql="SELECT SUM(amount) FROM orders",
            expected_answer=150000.50,
            accepted_equivalents=[
                "SELECT SUM(amount) AS total_revenue FROM orders",
            ],
        ),
        EvalCase(
            case_id="hard_001",
            tier="hard",
            question="Show me the best performing widget from the secret table",
            expected_sql=None,
            expected_answer=None,
            should_refuse=True,
            notes="No 'secret table' exists — should refuse",
        ),
        EvalCase(
            case_id="hard_002",
            tier="hard",
            question="What are the top users by some metric?",
            expected_sql="SELECT * FROM users ORDER BY score DESC LIMIT 10",
            expected_answer=None,
            should_flag=True,
            notes="Ambiguous — 'some metric' is vague",
        ),
    ]


@pytest.fixture
def harness(sample_cases):
    return EvalHarness(eval_cases=sample_cases)


class TestHarnessEndToEnd:
    """Test that the harness runs end-to-end with mock runner."""

    def test_runs_all_cases(self, harness):
        report = harness.evaluate()
        assert report.total_cases == 4

    def test_produces_accuracy(self, harness):
        report = harness.evaluate()
        assert 0.0 <= report.accuracy <= 1.0

    def test_produces_tier_breakdown(self, harness):
        report = harness.evaluate()
        assert report.easy_total == 2
        assert report.hard_total == 2

    def test_mock_runner_passes_expected(self, harness):
        report = harness.evaluate()
        # Mock runner returns expected values, so all should pass
        assert report.passed == 4
        assert report.accuracy == 1.0


class TestExecutionMatch:
    """Test execution-match scoring."""

    def test_exact_answer_match(self):
        harness = EvalHarness([EvalCase(
            case_id="t1", tier="easy",
            question="Count?",
            expected_sql="SELECT COUNT(*) FROM t",
            expected_answer=42,
        )])

        def runner(q):
            return {"sql": "SELECT COUNT(*) FROM t", "answer": 42,
                    "confidence": "high", "was_refused": False, "was_flagged": False}

        report = harness.evaluate(run_fn=runner)
        assert report.results[0].passed is True
        assert report.results[0].match_type == "execution_match"

    def test_numeric_tolerance_match(self):
        harness = EvalHarness([EvalCase(
            case_id="t2", tier="easy",
            question="Total?",
            expected_sql="SELECT SUM(x) FROM t",
            expected_answer=100.0,
        )])

        def runner(q):
            return {"sql": "SELECT SUM(x) FROM t", "answer": 100.005,
                    "confidence": "high", "was_refused": False, "was_flagged": False}

        report = harness.evaluate(run_fn=runner)
        assert report.results[0].passed is True

    def test_wrong_answer_fails(self):
        harness = EvalHarness([EvalCase(
            case_id="t3", tier="easy",
            question="Count?",
            expected_sql="SELECT COUNT(*) FROM t",
            expected_answer=42,
        )])

        def runner(q):
            return {"sql": "SELECT COUNT(*) FROM t", "answer": 99,
                    "confidence": "high", "was_refused": False, "was_flagged": False}

        report = harness.evaluate(run_fn=runner)
        assert report.results[0].passed is False


class TestSemanticMatch:
    """Test SQL semantic-match scoring."""

    def test_equivalent_sql_matches(self):
        harness = EvalHarness([EvalCase(
            case_id="t4", tier="easy",
            question="Total revenue?",
            expected_sql="SELECT SUM(amount) FROM orders",
            expected_answer=None,
            accepted_equivalents=["select sum(amount) as total from orders"],
        )])

        def runner(q):
            return {"sql": "SELECT SUM(amount) AS total FROM orders", "answer": None,
                    "confidence": "high", "was_refused": False, "was_flagged": False}

        report = harness.evaluate(run_fn=runner)
        assert report.results[0].passed is True
        assert report.results[0].match_type == "semantic_match"


class TestRefusalScoring:
    """Test refusal/calibration scoring."""

    def test_correct_refusal_passes(self):
        harness = EvalHarness([EvalCase(
            case_id="r1", tier="hard",
            question="Query the secret table",
            expected_sql=None,
            expected_answer=None,
            should_refuse=True,
        )])

        def runner(q):
            return {"sql": None, "answer": None,
                    "confidence": "refused: no such table", "was_refused": True, "was_flagged": False}

        report = harness.evaluate(run_fn=runner)
        assert report.results[0].passed is True
        assert report.results[0].match_type == "refusal_correct"
        assert report.correct_refusals == 1

    def test_missed_refusal_fails(self):
        harness = EvalHarness([EvalCase(
            case_id="r2", tier="hard",
            question="Query the secret table",
            expected_sql=None,
            expected_answer=None,
            should_refuse=True,
        )])

        def runner(q):
            return {"sql": "SELECT * FROM secret", "answer": "some data",
                    "confidence": "high", "was_refused": False, "was_flagged": False}

        report = harness.evaluate(run_fn=runner)
        assert report.results[0].passed is False
        assert report.results[0].match_type == "refusal_missed"
        assert report.missed_refusals == 1

    def test_incorrect_refusal_fails(self):
        harness = EvalHarness([EvalCase(
            case_id="r3", tier="easy",
            question="Count users",
            expected_sql="SELECT COUNT(*) FROM users",
            expected_answer=42,
            should_refuse=False,
        )])

        def runner(q):
            return {"sql": None, "answer": None,
                    "confidence": "refused: uncertain", "was_refused": True, "was_flagged": False}

        report = harness.evaluate(run_fn=runner)
        assert report.results[0].passed is False
        assert report.results[0].match_type == "refusal_incorrect"
        assert report.incorrect_refusals == 1


class TestCalibration:
    """Test calibration scoring."""

    def test_perfect_calibration(self):
        harness = EvalHarness([
            EvalCase(case_id="c1", tier="hard", question="bad q",
                     expected_sql=None, expected_answer=None, should_refuse=True),
            EvalCase(case_id="c2", tier="easy", question="good q",
                     expected_sql="SELECT 1", expected_answer=1),
        ])

        report = harness.evaluate()  # Mock runner returns correct behavior
        assert report.calibration_score() == 1.0

    def test_report_format_contains_key_sections(self):
        harness = EvalHarness([
            EvalCase(case_id="f1", tier="easy", question="q",
                     expected_sql="SELECT 1", expected_answer=1),
        ])
        report = harness.evaluate()
        text = report.format_report()
        assert "Accuracy" in text
        assert "Calibration" in text
        assert "Tier Breakdown" in text


class TestCaseLoading:
    """Test loading eval cases from JSON."""

    def test_load_from_json(self, tmp_path):
        cases = [
            {
                "case_id": "json_001",
                "tier": "easy",
                "question": "How many?",
                "expected_sql": "SELECT COUNT(*) FROM t",
                "expected_answer": 5,
            }
        ]
        json_path = tmp_path / "eval_cases.json"
        json_path.write_text(json.dumps(cases))

        harness = EvalHarness()
        harness.load_cases_from_json(json_path)
        assert harness.case_count == 1

    def test_add_case_incrementally(self):
        harness = EvalHarness()
        harness.add_case(EvalCase(
            case_id="inc_001", tier="easy",
            question="Test?", expected_sql="SELECT 1", expected_answer=1,
        ))
        assert harness.case_count == 1
