"""
Phase 5 — Verification-First Mode tests.

Per Tests.md §1:
- verifier.py: flags cartesian joins, flags queries referencing nonexistent
  tables/columns, flags queries touching unexpected tables, flags absurd
  row-count estimates

Per Phases.md Phase 5 exit criteria:
- Verifier flags a deliberately-broken cartesian-join test case
- AND a test case referencing a hallucinated (nonexistent) column
"""

import pytest
from src.agent.verifier import SQLVerifier, VerificationResult, VerificationIssue


KNOWN_TABLES = {
    "users": {"id", "name", "email", "created_at"},
    "orders": {"id", "user_id", "total", "status"},
    "products": {"id", "name", "price", "category"},
}


class TestHallucinatedReferences:
    """Test detection of hallucinated (nonexistent) tables and columns."""

    def test_valid_table_passes(self):
        v = SQLVerifier(known_tables=KNOWN_TABLES)
        result = v.verify("SELECT * FROM users")
        assert result.is_safe is True
        assert len(result.errors()) == 0

    def test_hallucinated_table_flagged(self):
        v = SQLVerifier(known_tables=KNOWN_TABLES)
        result = v.verify("SELECT * FROM nonexistent_table")
        assert result.is_safe is False
        errors = [i for i in result.issues if i.category == "hallucinated_table"]
        assert len(errors) >= 1
        assert "nonexistent_table" in errors[0].description

    def test_hallucinated_column_flagged(self):
        """
        Phase 5 exit criteria: flags a test case referencing a
        hallucinated (nonexistent) column.
        """
        v = SQLVerifier(known_tables=KNOWN_TABLES)
        result = v.verify("SELECT users.fake_column FROM users")
        assert result.is_safe is False
        errors = [i for i in result.issues if i.category == "hallucinated_column"]
        assert len(errors) >= 1
        assert "fake_column" in errors[0].description

    def test_valid_column_passes(self):
        v = SQLVerifier(known_tables=KNOWN_TABLES)
        result = v.verify("SELECT users.name, users.email FROM users")
        col_errors = [i for i in result.issues if i.category == "hallucinated_column"]
        assert len(col_errors) == 0

    def test_multiple_hallucinated_columns(self):
        v = SQLVerifier(known_tables=KNOWN_TABLES)
        result = v.verify("SELECT users.fake1, users.fake2 FROM users")
        col_errors = [i for i in result.issues if i.category == "hallucinated_column"]
        assert len(col_errors) >= 2


class TestCartesianJoins:
    """Test detection of cartesian joins (joins without ON)."""

    def test_proper_join_passes(self):
        v = SQLVerifier(known_tables=KNOWN_TABLES)
        result = v.verify(
            "SELECT * FROM users JOIN orders ON users.id = orders.user_id"
        )
        cartesian = [i for i in result.issues if i.category == "cartesian_join"]
        assert len(cartesian) == 0

    def test_cartesian_join_flagged(self):
        """
        Phase 5 exit criteria: verifier flags a deliberately-broken
        cartesian-join test case.
        """
        v = SQLVerifier(known_tables=KNOWN_TABLES)
        result = v.verify("SELECT * FROM users CROSS JOIN orders")
        cartesian = [i for i in result.issues if i.category == "cartesian_join"]
        assert len(cartesian) >= 1
        assert "cartesian" in cartesian[0].description.lower()

    def test_implicit_cartesian_flagged(self):
        """Multiple tables in FROM with no WHERE — implicit cartesian."""
        v = SQLVerifier(known_tables=KNOWN_TABLES)
        result = v.verify("SELECT * FROM users, orders")
        cartesian = [i for i in result.issues if i.category == "cartesian_join"]
        assert len(cartesian) >= 1

    def test_join_with_using_passes(self):
        """JOIN with USING clause should not be flagged as cartesian."""
        v = SQLVerifier(known_tables=KNOWN_TABLES)
        result = v.verify("SELECT * FROM users JOIN orders USING (id)")
        cartesian = [i for i in result.issues if i.category == "cartesian_join"]
        assert len(cartesian) == 0


class TestUnexpectedTables:
    """Test flagging of unexpected table access."""

    def test_expected_tables_pass(self):
        v = SQLVerifier(known_tables=KNOWN_TABLES)
        result = v.verify(
            "SELECT * FROM users",
            expected_tables={"users"},
        )
        unexpected = [i for i in result.issues if i.category == "unexpected_table"]
        assert len(unexpected) == 0

    def test_unexpected_table_flagged(self):
        v = SQLVerifier(known_tables=KNOWN_TABLES)
        result = v.verify(
            "SELECT * FROM users JOIN orders ON users.id = orders.user_id",
            expected_tables={"users"},
        )
        unexpected = [i for i in result.issues if i.category == "unexpected_table"]
        assert len(unexpected) >= 1
        assert "orders" in unexpected[0].description


class TestCostAndRowEstimates:
    """Test EXPLAIN result verification."""

    def test_reasonable_cost_passes(self):
        v = SQLVerifier(known_tables=KNOWN_TABLES, max_reasonable_cost=1000)
        result = VerificationResult(sql="SELECT 1", is_safe=True)
        explain = [{"QUERY PLAN": "Seq Scan on users  (cost=0.00..100.00 rows=100)"}]
        v.verify_explain_result(explain, result)
        assert result.estimated_cost == 100.0
        cost_warnings = [i for i in result.issues if i.category == "cost"]
        assert len(cost_warnings) == 0

    def test_absurd_cost_flagged(self):
        v = SQLVerifier(known_tables=KNOWN_TABLES, max_reasonable_cost=1000)
        result = VerificationResult(sql="SELECT 1", is_safe=True)
        explain = [{"QUERY PLAN": "Seq Scan on users  (cost=0.00..999999.00 rows=100)"}]
        v.verify_explain_result(explain, result)
        cost_warnings = [i for i in result.issues if i.category == "cost"]
        assert len(cost_warnings) >= 1

    def test_absurd_rows_flagged(self):
        v = SQLVerifier(known_tables=KNOWN_TABLES, max_reasonable_rows=1000)
        result = VerificationResult(sql="SELECT 1", is_safe=True)
        explain = [{"QUERY PLAN": "Seq Scan on users  (cost=0.00..100.00 rows=99999999)"}]
        v.verify_explain_result(explain, result)
        row_warnings = [i for i in result.issues if i.category == "row_count"]
        assert len(row_warnings) >= 1


class TestParseErrors:
    """Test handling of unparseable SQL."""

    def test_valid_sql_parses(self):
        v = SQLVerifier()
        result = v.verify("SELECT 1")
        assert result.is_safe is True

    def test_multiple_valid_queries(self):
        v = SQLVerifier(known_tables=KNOWN_TABLES)
        queries = [
            "SELECT * FROM users WHERE id = 1",
            "SELECT COUNT(*) FROM orders GROUP BY status",
            "SELECT u.name, SUM(o.total) FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.name",
        ]
        for sql in queries:
            result = v.verify(sql)
            assert len(result.errors()) == 0, f"Unexpected errors for: {sql}"


class TestCombinedChecks:
    """Test that multiple issues are caught in a single query."""

    def test_hallucinated_table_and_cartesian(self):
        v = SQLVerifier(known_tables=KNOWN_TABLES)
        result = v.verify("SELECT * FROM users CROSS JOIN fake_table")
        assert result.is_safe is False
        categories = {i.category for i in result.issues}
        assert "hallucinated_table" in categories
        assert "cartesian_join" in categories

    def test_safe_query_has_no_issues(self):
        v = SQLVerifier(known_tables=KNOWN_TABLES)
        result = v.verify(
            "SELECT u.name, SUM(o.total) as total_spent "
            "FROM users u JOIN orders o ON u.id = o.user_id "
            "WHERE o.status = 'completed' "
            "GROUP BY u.name ORDER BY total_spent DESC LIMIT 10"
        )
        assert len(result.errors()) == 0
