"""
Critic — pre-execution SQL validator that catches errors before
they hit the database.

Checks:
1. Hallucinated tables/columns (not in known schema)
2. Cartesian joins (missing join conditions)
3. Dialect mismatches (e.g., PostgreSQL syntax in SQLite context)
4. Missing WHERE on UPDATE/DELETE (safety)
5. Execution error classification (for retry routing)

Returns a CriticResult with pass/fail, diagnosis, error category, and
fix strategy so the reasoning engine can retry intelligently.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Error categories for retry routing."""
    NONE = "none"
    COLUMN_ERROR = "column_error"
    TABLE_ERROR = "table_error"
    SYNTAX_ERROR = "syntax_error"
    DIALECT_ERROR = "dialect_error"
    JOIN_ERROR = "join_error"
    SAFETY_ERROR = "safety_error"
    EXECUTION_ERROR = "execution_error"
    UNKNOWN = "unknown"


@dataclass
class CriticResult:
    """Result of SQL critique."""
    passed: bool
    diagnosis: str = ""
    error_category: ErrorCategory = ErrorCategory.NONE
    fix_strategy: str = ""
    details: list[str] = field(default_factory=list)
    sql: str = ""

    @property
    def needs_retry(self) -> bool:
        return not self.passed


class Critic:
    """
    Pre-execution SQL validator.

    Usage:
        critic = Critic(
            known_tables=["Customer", "Invoice"],
            known_columns={"Customer": ["CustomerId", "FirstName"], ...},
            dialect="sqlite",
        )
        result = critic.evaluate("SELECT * FROM Customers")
    """

    # PostgreSQL patterns that don't work in SQLite
    _PG_PATTERNS = [
        (r"\bEXTRACT\s*\(", "EXTRACT() — use strftime() in SQLite"),
        (r"\bINTERVAL\s+'", "INTERVAL — use date() arithmetic in SQLite"),
        (r"\bNOW\s*\(\)", "NOW() — use datetime('now') in SQLite"),
        (r"\bCURRENT_DATE\b(?!\s*\))", "CURRENT_DATE is supported but CURRENT_DATE + INTERVAL is not"),
        (r"\bFULL\s+OUTER\s+JOIN\b", "FULL OUTER JOIN — not supported in SQLite"),
        (r"\bILIKE\b", "ILIKE — use LIKE (SQLite is case-insensitive by default)"),
        (r"::\s*\w+", ":: type casting — use CAST() in SQLite"),
        (r"\bARRAY\[", "ARRAY[] — not supported in SQLite"),
        (r"\bGENERATE_SERIES\s*\(", "generate_series() — not available in SQLite"),
        (r"\bSTRING_AGG\s*\(", "STRING_AGG() — use GROUP_CONCAT() in SQLite"),
        (r"\bTO_CHAR\s*\(", "TO_CHAR() — use strftime() in SQLite"),
        (r"\bTO_DATE\s*\(", "TO_DATE() — use date() in SQLite"),
        (r"\bDATE_TRUNC\s*\(", "DATE_TRUNC() — use strftime() in SQLite"),
        (r"\bCOALESCE\s*\(\s*NULLIF\b.*?::", "Complex casting — use simpler CAST() in SQLite"),
    ]

    def __init__(
        self,
        known_tables: list[str] | None = None,
        known_columns: dict[str, list[str]] | None = None,
        dialect: str = "sqlite",
    ):
        self._known_tables = {t.lower(): t for t in (known_tables or [])}
        self._known_columns: dict[str, list[str]] = {}
        for table, cols in (known_columns or {}).items():
            self._known_columns[table.lower()] = [c.lower() for c in cols]
        self._dialect = dialect.lower()

    def evaluate(
        self,
        sql: str,
        execution_error: str | None = None,
        execution_result: Any = None,
        verification_issues: list[str] | None = None,
    ) -> CriticResult:
        """
        Evaluate SQL for issues.

        If execution_error is provided, classifies the runtime error
        instead of doing static analysis.
        """
        if execution_error:
            return self._classify_execution_error(sql, execution_error)

        issues = []

        # Check dialect mismatches
        dialect_issues = self._check_dialect(sql)
        if dialect_issues:
            return CriticResult(
                passed=False,
                diagnosis=f"Dialect mismatch ({self._dialect}): {'; '.join(dialect_issues)}",
                error_category=ErrorCategory.DIALECT_ERROR,
                fix_strategy=f"Rewrite using {self._dialect}-compatible syntax",
                details=dialect_issues,
                sql=sql,
            )

        # Check hallucinated tables
        table_issues = self._check_tables(sql)
        if table_issues:
            issues.extend(table_issues)
            return CriticResult(
                passed=False,
                diagnosis=f"Hallucinated table(s): {'; '.join(table_issues)}",
                error_category=ErrorCategory.TABLE_ERROR,
                fix_strategy="Use only tables from the schema",
                details=table_issues,
                sql=sql,
            )

        # Check hallucinated columns
        column_issues = self._check_columns(sql)
        if column_issues:
            issues.extend(column_issues)
            return CriticResult(
                passed=False,
                diagnosis=f"Hallucinated column(s): {'; '.join(column_issues)}",
                error_category=ErrorCategory.COLUMN_ERROR,
                fix_strategy="Use only columns from the schema",
                details=column_issues,
                sql=sql,
            )

        # Check cartesian joins
        join_issues = self._check_joins(sql)
        if join_issues:
            issues.extend(join_issues)
            return CriticResult(
                passed=False,
                diagnosis=f"Cartesian join detected: {'; '.join(join_issues)}",
                error_category=ErrorCategory.JOIN_ERROR,
                fix_strategy="Add proper join conditions",
                details=join_issues,
                sql=sql,
            )

        # Check safety (missing WHERE on UPDATE/DELETE)
        safety_issues = self._check_safety(sql)
        if safety_issues:
            return CriticResult(
                passed=False,
                diagnosis=f"Safety issue: {'; '.join(safety_issues)}",
                error_category=ErrorCategory.SAFETY_ERROR,
                fix_strategy="Add WHERE clause or confirm intent",
                details=safety_issues,
                sql=sql,
            )

        return CriticResult(passed=True, sql=sql)

    def _check_dialect(self, sql: str) -> list[str]:
        """Check for dialect-specific syntax mismatches."""
        if self._dialect != "sqlite":
            return []

        issues = []
        for pattern, description in self._PG_PATTERNS:
            if re.search(pattern, sql, re.IGNORECASE):
                issues.append(description)
        return issues

    def _check_tables(self, sql: str) -> list[str]:
        """Check for tables not in known schema."""
        if not self._known_tables:
            return []

        issues = []
        # Extract table references from FROM and JOIN clauses
        table_pattern = r'(?:FROM|JOIN)\s+"?(\w+)"?'
        referenced = re.findall(table_pattern, sql, re.IGNORECASE)

        for table in referenced:
            if table.lower() not in self._known_tables:
                # Find closest match for suggestion
                suggestion = self._find_closest(table, list(self._known_tables.values()))
                msg = f"Table '{table}' not found in schema"
                if suggestion:
                    msg += f" (did you mean '{suggestion}'?)"
                issues.append(msg)

        return issues

    def _check_columns(self, sql: str) -> list[str]:
        """Check for columns not in known schema."""
        if not self._known_columns:
            return []

        issues = []
        # Extract table.column references
        qualified_pattern = r'"?(\w+)"?\."?(\w+)"?'
        for table, column in re.findall(qualified_pattern, sql):
            table_lower = table.lower()
            if table_lower in self._known_columns:
                known = self._known_columns[table_lower]
                if column.lower() not in known:
                    suggestion = self._find_closest(column, known)
                    msg = f"Column '{column}' not found in '{table}'"
                    if suggestion:
                        msg += f" (did you mean '{suggestion}'?)"
                    issues.append(msg)

        return issues

    def _check_joins(self, sql: str) -> list[str]:
        """Check for cartesian joins (multiple FROM tables without join conditions)."""
        issues = []
        # Simple heuristic: multiple tables in FROM without JOIN or WHERE
        from_match = re.search(
            r'FROM\s+"?(\w+)"?\s*,\s*"?(\w+)"?',
            sql, re.IGNORECASE
        )
        if from_match:
            # Check if there's a WHERE with a join condition
            if not re.search(r'WHERE\s+.*?\.\w+\s*=\s*.*?\.\w+', sql, re.IGNORECASE):
                issues.append(
                    f"Tables {from_match.group(1)} and {from_match.group(2)} "
                    f"appear in FROM without a join condition"
                )
        return issues

    def _check_safety(self, sql: str) -> list[str]:
        """Check for dangerous operations without WHERE."""
        issues = []
        sql_upper = sql.strip().upper()

        if sql_upper.startswith("DELETE") and "WHERE" not in sql_upper:
            issues.append("DELETE without WHERE clause")
        if sql_upper.startswith("UPDATE") and "WHERE" not in sql_upper:
            issues.append("UPDATE without WHERE clause")
        if "DROP " in sql_upper:
            issues.append("DROP statement detected")

        return issues

    def _classify_execution_error(self, sql: str, error: str) -> CriticResult:
        """Classify a runtime execution error for retry routing."""
        error_lower = error.lower()

        if "no such column" in error_lower or "unknown column" in error_lower:
            col_match = re.search(r'no such column:\s*(\S+)', error_lower)
            col_name = col_match.group(1) if col_match else "unknown"
            return CriticResult(
                passed=False,
                diagnosis=f"Column error: {error}",
                error_category=ErrorCategory.COLUMN_ERROR,
                fix_strategy=f"Column '{col_name}' doesn't exist. Check schema for correct name.",
                sql=sql,
            )

        if "no such table" in error_lower:
            return CriticResult(
                passed=False,
                diagnosis=f"Table error: {error}",
                error_category=ErrorCategory.TABLE_ERROR,
                fix_strategy="Table doesn't exist. Check schema for correct name.",
                sql=sql,
            )

        if "syntax error" in error_lower or "near " in error_lower:
            return CriticResult(
                passed=False,
                diagnosis=f"Syntax error: {error}",
                error_category=ErrorCategory.SYNTAX_ERROR,
                fix_strategy="Fix SQL syntax",
                sql=sql,
            )

        return CriticResult(
            passed=False,
            diagnosis=f"Execution error: {error}",
            error_category=ErrorCategory.EXECUTION_ERROR,
            fix_strategy="Review and fix the query",
            sql=sql,
        )

    @staticmethod
    def _find_closest(name: str, candidates: list[str], threshold: float = 0.6) -> str | None:
        """Find closest match using simple ratio."""
        from difflib import SequenceMatcher
        best_score = 0.0
        best_match = None
        for candidate in candidates:
            score = SequenceMatcher(
                None, name.lower(), candidate.lower()
            ).ratio()
            if score > best_score and score >= threshold:
                best_score = score
                best_match = candidate
        return best_match
