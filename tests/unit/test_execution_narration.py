"""
Phase 7 — Execution & Narration tests.

Per Phases.md Phase 7 exit criteria:
- Read-only execution with timeout guard
- Narration numbers match query results exactly across the eval set
"""

import pytest
from src.agent.executor import QueryExecutor, ExecutionResult
from src.agent.narrator import ResultNarrator, NarrationResult


# --- Executor tests ---

class TestQueryExecutor:
    """Test query execution with guards."""

    def test_successful_execution(self):
        def mock_execute(sql, params):
            return [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]

        executor = QueryExecutor(execute_fn=mock_execute)
        result = executor.execute("SELECT * FROM users")
        assert result.is_success
        assert result.row_count == 2
        assert result.column_names == ["id", "name"]

    def test_empty_result(self):
        def mock_execute(sql, params):
            return []

        executor = QueryExecutor(execute_fn=mock_execute)
        result = executor.execute("SELECT * FROM users WHERE 1=0")
        assert result.is_success
        assert result.row_count == 0

    def test_execution_error_handled(self):
        def mock_execute(sql, params):
            raise RuntimeError("Connection refused")

        executor = QueryExecutor(execute_fn=mock_execute)
        result = executor.execute("SELECT 1")
        assert not result.is_success
        assert "Connection refused" in result.error

    def test_row_limit_applied(self):
        def mock_execute(sql, params):
            return [{"id": i} for i in range(2000)]

        executor = QueryExecutor(execute_fn=mock_execute, max_result_rows=100)
        result = executor.execute("SELECT * FROM big_table")
        assert result.is_success
        assert result.row_count == 100
        assert result.was_truncated is True
        assert result.truncation_limit == 100

    def test_no_truncation_under_limit(self):
        def mock_execute(sql, params):
            return [{"id": i} for i in range(50)]

        executor = QueryExecutor(execute_fn=mock_execute, max_result_rows=100)
        result = executor.execute("SELECT * FROM small_table")
        assert result.was_truncated is False

    def test_execution_time_recorded(self):
        def mock_execute(sql, params):
            return [{"id": 1}]

        executor = QueryExecutor(execute_fn=mock_execute)
        result = executor.execute("SELECT 1")
        assert result.execution_time_ms >= 0

    def test_no_execution_fn_returns_error(self):
        executor = QueryExecutor(execute_fn=None)
        result = executor.execute("SELECT 1")
        assert not result.is_success
        assert "No execution function" in result.error

    def test_params_passed_through(self):
        received_params = {}

        def mock_execute(sql, params):
            received_params.update(params)
            return [{"count": 5}]

        executor = QueryExecutor(execute_fn=mock_execute)
        executor.execute("SELECT COUNT(*) FROM users WHERE status = :status",
                         params={"status": "active"})
        assert received_params == {"status": "active"}


# --- Narrator tests ---

class TestNarrator:
    """Test narration with real computed numbers."""

    def test_single_scalar_result(self):
        narrator = ResultNarrator()
        result = narrator.narrate(
            rows=[{"count": 42}],
            column_names=["count"],
            question="How many users?",
            sql="SELECT COUNT(*) as count FROM users",
            confidence_display="high",
        )
        assert "42" in result.answer
        assert result.confidence == "high"

    def test_empty_result(self):
        narrator = ResultNarrator()
        result = narrator.narrate(
            rows=[],
            column_names=[],
            question="Users in Antarctica?",
            sql="SELECT * FROM users WHERE region = 'Antarctica'",
            confidence_display="high",
        )
        assert "no results" in result.answer.lower()

    def test_multi_row_result(self):
        narrator = ResultNarrator()
        rows = [
            {"category": "electronics", "total": 1500},
            {"category": "books", "total": 300},
            {"category": "clothing", "total": 700},
        ]
        result = narrator.narrate(
            rows=rows,
            column_names=["category", "total"],
            question="Revenue by category",
            sql="SELECT category, SUM(total) FROM orders GROUP BY category",
            confidence_display="high",
        )
        assert "3" in result.answer  # 3 rows
        assert "2,500" in result.answer or "2500" in result.answer  # total sum

    def test_narration_numbers_match_exactly(self):
        """
        Phase 7 exit criteria: narration numbers match query results exactly.

        The narrator must compute and display the exact values from the
        result rows, not approximate or LLM-generated values.
        """
        narrator = ResultNarrator()
        rows = [
            {"region": "North", "revenue": 12345.67},
            {"region": "South", "revenue": 8901.23},
            {"region": "East", "revenue": 4567.89},
        ]
        result = narrator.narrate(
            rows=rows,
            column_names=["region", "revenue"],
            question="Revenue by region",
            sql="SELECT region, SUM(amount) as revenue FROM orders GROUP BY region",
            confidence_display="high",
        )
        # The narrator should compute the real total
        expected_total = 12345.67 + 8901.23 + 4567.89
        # Check the total appears in the answer
        assert f"{expected_total:,.2f}" in result.answer

    def test_narration_format_per_design(self):
        """Output format must match Design.md §1."""
        narrator = ResultNarrator()
        result = narrator.narrate(
            rows=[{"count": 100}],
            column_names=["count"],
            question="User count",
            sql="SELECT COUNT(*) as count FROM users",
            confidence_display="high",
            interpretation="Count all users in the users table",
            notes=["Using glossary definition of 'user'"],
        )
        output = result.format_output()
        assert "**Interpretation:**" in output
        assert "**SQL:**" in output
        assert "**Confidence:**" in output
        assert "**Answer:**" in output
        assert "**Notes:**" in output

    def test_sql_always_shown(self):
        """Per Rules.md §2: Always show the generated SQL. No exceptions."""
        narrator = ResultNarrator()
        result = narrator.narrate(
            rows=[{"id": 1}],
            column_names=["id"],
            question="test",
            sql="SELECT id FROM users LIMIT 1",
            confidence_display="high",
        )
        output = result.format_output()
        assert "SELECT id FROM users LIMIT 1" in output

    def test_confidence_always_shown(self):
        """Per Design.md §1: Confidence never hidden, never omitted."""
        narrator = ResultNarrator()
        for conf in ["high", "flagged: uncertain join", "refused: missing schema"]:
            result = narrator.narrate(
                rows=[],
                column_names=[],
                question="test",
                sql="SELECT 1",
                confidence_display=conf,
            )
            output = result.format_output()
            assert conf in output

    def test_single_row_multiple_columns(self):
        narrator = ResultNarrator()
        result = narrator.narrate(
            rows=[{"name": "Alice", "email": "alice@example.com", "age": 30}],
            column_names=["name", "email", "age"],
            question="User details",
            sql="SELECT name, email, age FROM users WHERE id = 1",
            confidence_display="high",
        )
        assert "Alice" in result.answer
        assert "alice@example.com" in result.answer

    def test_null_values_handled(self):
        narrator = ResultNarrator()
        result = narrator.narrate(
            rows=[{"name": "Bob", "phone": None}],
            column_names=["name", "phone"],
            question="User phone",
            sql="SELECT name, phone FROM users WHERE id = 2",
            confidence_display="high",
        )
        assert "NULL" in result.answer or "None" in result.answer


class TestNarratorComputation:
    """Verify that numbers are computed from real data, not hallucinated."""

    def test_sum_computed_correctly(self):
        narrator = ResultNarrator()
        rows = [{"amount": 100}, {"amount": 200}, {"amount": 300}]
        stats = narrator._compute_stats(rows, ["amount"])
        assert stats["amount_sum"] == 600
        assert stats["amount_avg"] == 200
        assert stats["amount_min"] == 100
        assert stats["amount_max"] == 300

    def test_count_computed_correctly(self):
        narrator = ResultNarrator()
        rows = [{"id": i} for i in range(10)]
        stats = narrator._compute_stats(rows, ["id"])
        assert stats["row_count"] == 10

    def test_mixed_types_handled(self):
        narrator = ResultNarrator()
        rows = [
            {"name": "Alice", "score": 95},
            {"name": "Bob", "score": 87},
        ]
        stats = narrator._compute_stats(rows, ["name", "score"])
        assert "score_sum" in stats
        assert stats["score_sum"] == 182
        # "name" is not numeric, should not have sum
        assert "name_sum" not in stats

    def test_empty_rows_handled(self):
        narrator = ResultNarrator()
        stats = narrator._compute_stats([], [])
        assert stats["row_count"] == 0
