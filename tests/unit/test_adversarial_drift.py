"""
Unit tests for Schema Drift Detector and Adversarial Evaluation.
"""

from unittest.mock import MagicMock
from src.agent.adversarial_drift_eval import (
    SchemaSnapshot,
    SchemaDriftDetector,
    AdversarialSuiteRunner,
    AdversarialCase,
)


def test_schema_drift_detection_added_and_dropped():
    detector = SchemaDriftDetector()
    base = SchemaSnapshot.capture({
        "Customer": ["CustomerId", "Name", "OldCol"],
        "Orders": ["OrderId", "CustomerId"],
    })
    current = SchemaSnapshot.capture({
        "Customer": ["CustomerId", "Name", "NewCol"],
        "Invoices": ["InvoiceId", "Total"],
    })

    report = detector.diff(base, current)
    assert report.has_drift is True
    assert "Invoices" in report.added_tables
    assert "Orders" in report.dropped_tables
    assert "OldCol" in report.dropped_columns["Customer"]
    assert "NewCol" in report.added_columns["Customer"]
    assert report.severity == "critical"
    summary = report.summary()
    assert "Dropped Tables: Orders" in summary


def test_audit_glossary_flags_stale_terms():
    detector = SchemaDriftDetector()
    current = SchemaSnapshot.capture({
        "Customer": ["CustomerId", "Name"],
    })
    # Legacy glossary references dropped 'Orders' table
    glossary = [
        {"term": "Total Sales", "sql_pattern": "SELECT SUM(Total) FROM Orders"},
        {"term": "Active Customers", "sql_pattern": "SELECT * FROM Customer"},
    ]
    stale_terms, _ = detector.audit_glossary_and_corrections(current, glossary_terms=glossary)
    assert len(stale_terms) == 1
    assert "Total Sales" in stale_terms[0]
    assert "Orders" in stale_terms[0]


def test_adversarial_runner_default_cases():
    runner = AdversarialSuiteRunner()
    mock_pipeline = MagicMock()
    # Mock defense blocking injection
    mock_pipeline.run.return_value = MagicMock(
        refused=True,
        success=False,
        error="Blocked suspected injection",
        refusal_reason="Blocked",
    )
    results = runner.run_suite(pipeline=mock_pipeline)
    assert len(results) == len(runner.DEFAULT_ADVERSARIAL_CASES)
    assert all(r.passed for r in results)
    assert all(r.defended for r in results)
