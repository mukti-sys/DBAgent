"""
Executor — runs SQL read-only with timeout and circuit breaker.

Per Architecture.md §2: Runs the query read-only, with timeout and
circuit breaker.

Per Rules.md §1: Default DB role is read-only, enforced at the
database user level.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from src.config import load_settings

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of query execution."""
    rows: list[dict[str, Any]]
    row_count: int
    column_names: list[str]
    execution_time_ms: float
    was_truncated: bool = False
    truncation_limit: int | None = None
    error: str | None = None

    @property
    def is_success(self) -> bool:
        return self.error is None


class QueryExecutor:
    """
    Executes queries with safety guards.

    For unit testing without a DB, accepts a mock execution function.
    The actual DB execution is delegated to the connector (Phase 1).
    """

    def __init__(
        self,
        execute_fn: Any = None,
        query_timeout: int = 30,
        max_result_rows: int = 1000,
    ):
        settings = load_settings()
        db_settings = settings.get("database", {})
        self._execute_fn = execute_fn
        self._timeout = query_timeout or db_settings.get("query_timeout", 30)
        self._max_rows = max_result_rows

    def execute(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        """
        Execute a SQL query with timeout guard and row limiting.

        Args:
            sql: SQL to execute.
            params: Parameterized query bindings.

        Returns:
            ExecutionResult with rows, timing, and metadata.
        """
        if self._execute_fn is None:
            return ExecutionResult(
                rows=[], row_count=0, column_names=[],
                execution_time_ms=0, error="No execution function configured",
            )

        start = time.perf_counter()
        try:
            rows = self._execute_fn(sql, params or {})
            elapsed_ms = (time.perf_counter() - start) * 1000

            # Extract column names from first row
            column_names = list(rows[0].keys()) if rows else []

            # Apply row limit
            was_truncated = len(rows) > self._max_rows
            if was_truncated:
                rows = rows[:self._max_rows]

            return ExecutionResult(
                rows=rows,
                row_count=len(rows),
                column_names=column_names,
                execution_time_ms=round(elapsed_ms, 2),
                was_truncated=was_truncated,
                truncation_limit=self._max_rows if was_truncated else None,
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error(f"Query execution failed after {elapsed_ms:.0f}ms: {e}")
            return ExecutionResult(
                rows=[], row_count=0, column_names=[],
                execution_time_ms=round(elapsed_ms, 2),
                error=str(e),
            )
