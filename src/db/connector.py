"""
DB connector — SQLAlchemy-based connection management.

Read-only role enforced at the DB user level, not just app level (Rules.md §1).
Uses parameterized queries exclusively — no raw string interpolation, ever.
"""

from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from src.config import load_settings
from src.db.access_control import AccessController, AccessDeniedError


class DBConnector:
    """
    Manages database connections with enforced read-only access by default.

    All queries go through parameterized execution — never raw string
    interpolation (Rules.md §1).
    """

    def __init__(
        self,
        db_url: str | None = None,
        read_only: bool = True,
        query_timeout: int = 30,
    ):
        settings = load_settings()
        db_settings = settings.get("database", {})

        self._db_url = db_url or db_settings.get("url", "")
        self._read_only = read_only if read_only is not None else db_settings.get("read_only", True)
        self._query_timeout = query_timeout or db_settings.get("query_timeout", 30)
        self._engine: Engine | None = None
        self._access_controller = AccessController(write_mode_enabled=not self._read_only)

    @property
    def access_controller(self) -> AccessController:
        return self._access_controller

    def connect(self, db_url: str | None = None) -> Engine:
        """Create or return the SQLAlchemy engine."""
        url = db_url or self._db_url
        if not url:
            raise ValueError("No database URL provided. Set DB_URL env var or config/settings.yaml database.url")

        if self._engine is None:
            connect_args = {}
            # For PostgreSQL, set statement_timeout and default_transaction_read_only
            if "postgresql" in url or "postgres" in url:
                connect_args["options"] = (
                    f"-c statement_timeout={self._query_timeout * 1000} "
                    f"-c default_transaction_read_only={'on' if self._read_only else 'off'}"
                )

            self._engine = create_engine(
                url,
                connect_args=connect_args,
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=2,
            )

        return self._engine

    def execute_query(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        user_input: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Execute a SQL query with parameterized binding.

        Args:
            sql: SQL string with :param_name placeholders for parameters.
            params: Parameter dict to bind safely.
            user_input: The original user input (for injection detection).

        Returns:
            List of row dicts.

        Raises:
            AccessDeniedError: If the query is blocked by access control.
        """
        # Defense-in-depth: check for direct interpolation
        if user_input and not self._access_controller.validate_no_injection(sql, user_input):
            raise AccessDeniedError(
                "Potential SQL injection detected: user input appears to be "
                "directly interpolated into the query. Use parameterized queries."
            )

        # Access control check
        allowed, reason = self._access_controller.check_query_allowed(sql)
        if not allowed:
            raise AccessDeniedError(reason)

        if self._engine is None:
            raise RuntimeError("Not connected. Call connect() first.")

        with self._engine.connect() as conn:
            result = conn.execute(text(sql), params or {})
            if result.returns_rows:
                return [dict(row._mapping) for row in result.fetchall()]
            return []

    def close(self):
        """Dispose of the engine and connection pool."""
        if self._engine:
            self._engine.dispose()
            self._engine = None
