"""
Verifier — runs EXPLAIN, checks join cardinality, touched tables,
cost estimates. Flags broken joins and hallucinated schema elements.

Per Architecture.md §2: Runs EXPLAIN; checks join cardinality, touched
tables, cost estimate.

Per Architecture.md §3 Model Routing: Verification is code, not LLM.
Haiku only for a natural-language explanation of a flagged issue.
Routes through llm_client for that optional call.

Per Rules.md §4: Mitigate schema/SQL hallucination by verifying
references against known schema.

This module performs static analysis of generated SQL using sqlglot.
Live EXPLAIN checks require a DB connection (integration tests).
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import sqlglot
from sqlglot import exp

from src.agent.llm_client import LLMClient as UnifiedLLMClient

logger = logging.getLogger(__name__)


@dataclass
class VerificationIssue:
    """A single verification issue found in generated SQL."""
    severity: str  # "error", "warning", "info"
    category: str  # "hallucinated_table", "hallucinated_column", "cartesian_join", "cost", "row_count"
    description: str
    suggestion: str = ""


@dataclass
class VerificationResult:
    """Complete verification result for a SQL query."""
    sql: str
    is_safe: bool  # False if any errors found
    issues: list[VerificationIssue] = field(default_factory=list)
    tables_verified: list[str] = field(default_factory=list)
    joins_verified: list[str] = field(default_factory=list)
    estimated_cost: float | None = None
    estimated_rows: int | None = None

    def errors(self) -> list[VerificationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    def warnings(self) -> list[VerificationIssue]:
        return [i for i in self.issues if i.severity == "warning"]


class SQLVerifier:
    """
    Verifies generated SQL for safety and correctness.

    Checks performed (statically via sqlglot):
    1. Referenced tables exist in schema
    2. Referenced columns exist in their tables
    3. Joins have proper ON conditions (no cartesian joins)
    4. Unexpected tables not touched
    5. Cost/row-count sanity (via EXPLAIN when DB available)

    Per Rules.md §4, this is a best-effort mitigation for
    hallucination — not a guarantee.
    """

    def __init__(
        self,
        known_tables: dict[str, set[str]] | None = None,
        max_reasonable_cost: float = 1_000_000,
        max_reasonable_rows: int = 10_000_000,
        llm_client: UnifiedLLMClient | None = None,
    ):
        """
        Args:
            known_tables: dict of table_name -> set of column_names.
            max_reasonable_cost: cost threshold above which to warn.
            max_reasonable_rows: row count above which to warn.
            llm_client: Optional unified LLM client for natural-language
                explanation of flagged issues (Architecture.md Model Routing).
        """
        self._known_tables = known_tables or {}
        self._max_cost = max_reasonable_cost
        self._llm_client = llm_client
        self._max_rows = max_reasonable_rows

    def verify(
        self,
        sql: str,
        expected_tables: set[str] | None = None,
    ) -> VerificationResult:
        """
        Verify a SQL query against known schema and safety checks.

        Args:
            sql: The SQL query to verify.
            expected_tables: If provided, tables that SHOULD be referenced.

        Returns:
            VerificationResult with all issues found.
        """
        result = VerificationResult(sql=sql, is_safe=True)

        try:
            parsed = sqlglot.parse(sql, read="postgres")
        except sqlglot.errors.ParseError as e:
            result.is_safe = False
            result.issues.append(VerificationIssue(
                severity="error",
                category="parse_error",
                description=f"SQL parse error: {e}",
            ))
            return result

        for statement in parsed:
            if statement is None:
                continue

            # Check table references
            self._check_tables(statement, result)

            # Check column references
            self._check_columns(statement, result)

            # Check for cartesian joins
            self._check_joins(statement, result)

            # Check for unexpected tables
            if expected_tables:
                self._check_unexpected_tables(statement, result, expected_tables)

        # Final safety determination
        result.is_safe = len(result.errors()) == 0
        return result

    def verify_explain_result(
        self,
        explain_output: list[dict[str, Any]],
        result: VerificationResult,
    ) -> VerificationResult:
        """
        Verify an EXPLAIN result for cost and row count sanity.

        Args:
            explain_output: EXPLAIN output rows.
            result: Existing VerificationResult to augment.

        Returns:
            Updated VerificationResult.
        """
        for row in explain_output:
            plan = row.get("QUERY PLAN", "")
            if isinstance(plan, str):
                # Try to extract cost and rows from EXPLAIN output
                import re
                cost_match = re.search(r"cost=[\d.]+\.\.([\d.]+)", plan)
                rows_match = re.search(r"rows=(\d+)", plan)

                if cost_match:
                    cost = float(cost_match.group(1))
                    result.estimated_cost = cost
                    if cost > self._max_cost:
                        result.issues.append(VerificationIssue(
                            severity="warning",
                            category="cost",
                            description=f"Estimated cost ({cost:.0f}) exceeds threshold ({self._max_cost:.0f})",
                            suggestion="Consider adding indexes or limiting the query scope.",
                        ))

                if rows_match:
                    rows = int(rows_match.group(1))
                    result.estimated_rows = rows
                    if rows > self._max_rows:
                        result.issues.append(VerificationIssue(
                            severity="warning",
                            category="row_count",
                            description=f"Estimated rows ({rows:,}) exceeds threshold ({self._max_rows:,})",
                            suggestion="Consider adding a LIMIT or more restrictive WHERE clause.",
                        ))

        result.is_safe = len(result.errors()) == 0
        return result

    def _check_tables(self, statement: exp.Expression, result: VerificationResult) -> None:
        """Check that all referenced tables exist in known schema."""
        for table_node in statement.find_all(exp.Table):
            table_name = table_node.name
            if not table_name:
                continue

            result.tables_verified.append(table_name)

            if self._known_tables and table_name.lower() not in {
                t.lower() for t in self._known_tables
            }:
                result.issues.append(VerificationIssue(
                    severity="error",
                    category="hallucinated_table",
                    description=f"Table '{table_name}' not found in schema",
                    suggestion=f"Known tables: {', '.join(sorted(self._known_tables.keys()))}",
                ))

    def _check_columns(self, statement: exp.Expression, result: VerificationResult) -> None:
        """Check that referenced columns exist in their tables."""
        if not self._known_tables:
            return  # Can't verify without schema

        for col_node in statement.find_all(exp.Column):
            col_name = col_node.name
            table_ref = col_node.table

            if not col_name:
                continue

            # Skip columns with * (SELECT *)
            if col_name == "*":
                continue

            if table_ref:
                # Find the actual table name (might be aliased)
                actual_table = self._resolve_alias(statement, table_ref)
                if actual_table and actual_table.lower() in {
                    t.lower() for t in self._known_tables
                }:
                    known_cols = self._get_columns_for_table(actual_table)
                    if known_cols and col_name.lower() not in {
                        c.lower() for c in known_cols
                    }:
                        result.issues.append(VerificationIssue(
                            severity="error",
                            category="hallucinated_column",
                            description=(
                                f"Column '{col_name}' not found in table "
                                f"'{actual_table}'"
                            ),
                            suggestion=f"Known columns: {', '.join(sorted(known_cols))}",
                        ))

    def _check_joins(self, statement: exp.Expression, result: VerificationResult) -> None:
        """Check for cartesian joins (joins without ON conditions)."""
        joins = list(statement.find_all(exp.Join))

        for join in joins:
            join_str = join.sql()
            result.joins_verified.append(join_str)

            # Check join kind and conditions via args dict
            join_kind = join.args.get("kind", "")
            on_condition = join.args.get("on")
            using_condition = join.args.get("using")

            # CROSS JOIN or join without ON/USING is a cartesian join
            if join_kind == "CROSS" or (on_condition is None and using_condition is None):
                result.issues.append(VerificationIssue(
                    severity="error",
                    category="cartesian_join",
                    description=f"Cartesian join detected (no ON/USING condition): {join_str[:100]}",
                    suggestion="Add an ON condition to specify the join relationship.",
                ))

        # Also check for implicit cartesian joins (FROM a, b without WHERE join)
        from_clause = statement.find(exp.From)
        if from_clause:
            tables_in_from = list(from_clause.find_all(exp.Table))
            if len(tables_in_from) > 1 and not joins:
                # Multiple tables in FROM without explicit JOINs
                # Check for WHERE-based join condition
                where = statement.find(exp.Where)
                if where is None:
                    result.issues.append(VerificationIssue(
                        severity="error",
                        category="cartesian_join",
                        description="Multiple tables in FROM without JOIN or WHERE condition — possible cartesian product",
                        suggestion="Use explicit JOIN with ON conditions.",
                    ))

    def _check_unexpected_tables(
        self,
        statement: exp.Expression,
        result: VerificationResult,
        expected: set[str],
    ) -> None:
        """Check if the query touches tables not in the expected set."""
        for table_node in statement.find_all(exp.Table):
            table_name = table_node.name
            if table_name and table_name.lower() not in {t.lower() for t in expected}:
                result.issues.append(VerificationIssue(
                    severity="warning",
                    category="unexpected_table",
                    description=f"Query touches unexpected table: '{table_name}'",
                    suggestion=f"Expected only: {', '.join(sorted(expected))}",
                ))

    def _resolve_alias(self, statement: exp.Expression, alias: str) -> str | None:
        """Resolve a table alias to the actual table name."""
        for table_node in statement.find_all(exp.Table):
            table_alias = table_node.alias
            if table_alias and table_alias.lower() == alias.lower():
                return table_node.name
            if table_node.name and table_node.name.lower() == alias.lower():
                return table_node.name
        return alias  # Return as-is if we can't resolve

    def _get_columns_for_table(self, table_name: str) -> set[str] | None:
        """Get known columns for a table (case-insensitive lookup)."""
        for t, cols in self._known_tables.items():
            if t.lower() == table_name.lower():
                return cols
        return None
