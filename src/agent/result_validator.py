"""
Result validator — post-execution sanity checks on query results.

Checks:
1. Sanity checks (negative counts, impossible values, empty results)
2. Cross-query verification (generates a simpler verification query)
3. Assumption surfacing (deleted_at columns, date ranges, filters)
"""

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class FindingSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Finding:
    """A single validation finding."""
    check_name: str
    severity: FindingSeverity
    message: str
    column: str = ""
    value: Any = None

    def display(self) -> str:
        icon = {"info": "ℹ", "warning": "⚠", "critical": "🚩"}
        return f"{icon.get(self.severity.value, '•')} [{self.severity.value}] {self.check_name}: {self.message}"


@dataclass
class Assumption:
    """An assumption the query made that the user should know about."""
    description: str
    category: str = "general"


@dataclass
class ValidationResult:
    """Complete validation result."""
    is_trustworthy: bool = True
    findings: list[Finding] = field(default_factory=list)
    assumptions: list[Assumption] = field(default_factory=list)
    cross_check_sql: str | None = None

    @property
    def has_critical(self) -> bool:
        return any(f.severity == FindingSeverity.CRITICAL for f in self.findings)


class ResultValidator:
    """
    Post-execution result validator.

    Usage:
        validator = ResultValidator()
        result = validator.validate(
            sql="SELECT COUNT(*) FROM users",
            question="How many users?",
            rows=[{"count": -5}],
            column_names=["count"],
            row_count=1,
        )
    """

    def validate(
        self,
        sql: str,
        question: str,
        rows: list[dict[str, Any]],
        column_names: list[str],
        row_count: int | None = None,
    ) -> ValidationResult:
        """Run all validation checks."""
        if row_count is None:
            row_count = len(rows)

        result = ValidationResult()

        # Run sanity checks
        result.findings.extend(self._check_negative_counts(rows, column_names, question))
        result.findings.extend(self._check_empty_results(rows, row_count, question))
        result.findings.extend(self._check_extreme_values(rows, column_names))
        result.findings.extend(self._check_null_heavy(rows, column_names))

        # Surface assumptions
        result.assumptions.extend(self._surface_assumptions(sql, question, column_names))

        # Generate cross-check SQL
        result.cross_check_sql = self._generate_cross_check(sql, question)

        # Mark as untrustworthy if critical findings
        if result.has_critical:
            result.is_trustworthy = False

        return result

    def _check_negative_counts(self, rows: list[dict], columns: list[str],
                                question: str) -> list[Finding]:
        """Check for negative values in count/sum columns."""
        findings = []
        count_keywords = {"count", "total", "sum", "amount", "quantity", "num", "number"}

        for col in columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in count_keywords):
                for row in rows:
                    val = row.get(col)
                    if isinstance(val, (int, float)) and val < 0:
                        findings.append(Finding(
                            check_name="negative_value",
                            severity=FindingSeverity.WARNING,
                            message=f"'{col}' has negative value: {val}",
                            column=col,
                            value=val,
                        ))
        return findings

    def _check_empty_results(self, rows: list[dict], row_count: int,
                              question: str) -> list[Finding]:
        """Flag empty results for questions that expect data."""
        findings = []
        expects_data = any(kw in question.lower() for kw in [
            "how many", "total", "list", "show", "what", "count", "sum"
        ])
        if row_count == 0 and expects_data:
            findings.append(Finding(
                check_name="empty_result",
                severity=FindingSeverity.WARNING,
                message="Query returned 0 rows but the question expects data",
            ))
        return findings

    def _check_extreme_values(self, rows: list[dict], columns: list[str]) -> list[Finding]:
        """Check for suspiciously extreme values."""
        findings = []
        if not rows:
            return findings

        for col in columns:
            numeric_vals = [
                row[col] for row in rows
                if col in row and isinstance(row[col], (int, float))
            ]
            if len(numeric_vals) < 2:
                continue

            avg = sum(numeric_vals) / len(numeric_vals)
            if avg == 0:
                continue

            for val in numeric_vals:
                ratio = abs(val / avg) if avg != 0 else 0
                if ratio > 100:
                    findings.append(Finding(
                        check_name="extreme_value",
                        severity=FindingSeverity.WARNING,
                        message=f"'{col}' has extreme outlier: {val} (avg: {avg:.2f})",
                        column=col,
                        value=val,
                    ))
                    break  # One warning per column is enough

        return findings

    def _check_null_heavy(self, rows: list[dict], columns: list[str]) -> list[Finding]:
        """Check for columns that are mostly NULL."""
        findings = []
        if len(rows) < 3:
            return findings

        for col in columns:
            null_count = sum(1 for row in rows if row.get(col) is None)
            null_pct = null_count / len(rows) if rows else 0

            if null_pct > 0.8:
                findings.append(Finding(
                    check_name="null_heavy",
                    severity=FindingSeverity.INFO,
                    message=f"'{col}' is {null_pct:.0%} NULL ({null_count}/{len(rows)} rows)",
                    column=col,
                ))

        return findings

    def _surface_assumptions(self, sql: str, question: str,
                              columns: list[str]) -> list[Assumption]:
        """Surface implicit assumptions the query makes."""
        assumptions = []
        sql_lower = sql.lower()
        q_lower = question.lower()

        # Check for soft-delete columns
        if "deleted_at" not in sql_lower:
            # If the schema likely has deleted_at, surface it
            assumptions.append(Assumption(
                description="Query does not filter soft-deleted rows (if table has 'deleted_at')",
                category="soft_delete",
            ))

        # Check for date range assumptions
        date_keywords = ["this year", "this month", "today", "recent", "last"]
        if any(kw in q_lower for kw in date_keywords):
            if not re.search(r"WHERE\s+.*(?:date|time|created)", sql_lower):
                assumptions.append(Assumption(
                    description=f"Question mentions time but SQL has no date filter",
                    category="date_range",
                ))

        # Check for LIMIT without ORDER BY
        if "limit" in sql_lower and "order by" not in sql_lower:
            assumptions.append(Assumption(
                description="Query uses LIMIT without ORDER BY — results are non-deterministic",
                category="ordering",
            ))

        return assumptions

    def _generate_cross_check(self, sql: str, question: str) -> str | None:
        """Generate a simpler verification query."""
        sql_lower = sql.lower()

        # If it's a COUNT, generate a SUM cross-check
        if "count(" in sql_lower:
            return None  # COUNT is simple enough

        # If it's a SUM with GROUP BY, generate a total SUM
        sum_match = re.search(r'SUM\(([^)]+)\)', sql, re.IGNORECASE)
        group_match = re.search(r'GROUP\s+BY', sql, re.IGNORECASE)
        if sum_match and group_match:
            col = sum_match.group(1)
            # Extract FROM clause
            from_match = re.search(r'FROM\s+(.+?)(?:WHERE|GROUP|ORDER|LIMIT|$)',
                                    sql, re.IGNORECASE | re.DOTALL)
            if from_match:
                return f"SELECT SUM({col}) AS total FROM {from_match.group(1).strip()}"

        return None
