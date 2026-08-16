"""
Access control — enforces read-only DB role; blocks DDL/DML
unless write-mode is explicitly enabled (Rules.md §1).

Security rules (non-negotiable, per Rules.md §1):
- Default DB role is read-only, enforced at the database user level.
- Never interpolate user input directly into SQL strings.
- Block DDL/DML (DROP, DELETE, UPDATE, INSERT, ALTER, TRUNCATE) unless
  explicit write-mode is enabled, and even then require separate confirmation.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import sqlglot
from sqlglot import exp


class QueryType(Enum):
    """Classification of SQL statement types."""
    SELECT = "SELECT"
    DDL = "DDL"      # CREATE, ALTER, DROP, TRUNCATE
    DML = "DML"      # INSERT, UPDATE, DELETE
    OTHER = "OTHER"


class AccessDeniedError(Exception):
    """Raised when a query is blocked by access control."""
    pass


class WriteConfirmationRequired(Exception):
    """Raised when write-mode is on but confirmation is needed."""
    pass


# DDL/DML statement types that sqlglot recognizes
_DDL_TYPES = (
    exp.Create, exp.Drop, exp.Alter,
)

_DML_TYPES = (
    exp.Insert, exp.Update, exp.Delete,
)

# Regex fallback for statements sqlglot may not parse cleanly
# (e.g., TRUNCATE, which sqlglot may not always model)
_DANGEROUS_PATTERN = re.compile(
    r"^\s*(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|CREATE)\b",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class AccessController:
    """
    Enforces query-level access control.

    - Default mode is read-only: only SELECT queries are allowed.
    - Write mode can be explicitly enabled, but DDL/DML still requires
      a separate confirmation step before execution.
    """

    write_mode_enabled: bool = False
    _pending_confirmation: dict[str, Any] = field(default_factory=dict)

    def classify_query(self, sql: str) -> QueryType:
        """
        Classify a SQL string as SELECT, DDL, DML, or OTHER.

        Uses sqlglot for structured parsing with a regex fallback
        for edge cases (e.g. TRUNCATE).
        """
        # Regex fallback first — catches things sqlglot might not model
        if _DANGEROUS_PATTERN.search(sql):
            match = _DANGEROUS_PATTERN.search(sql)
            keyword = match.group(1).upper()
            if keyword in ("DROP", "ALTER", "TRUNCATE", "CREATE"):
                return QueryType.DDL
            elif keyword in ("DELETE", "UPDATE", "INSERT"):
                return QueryType.DML

        # Structured parse via sqlglot
        try:
            parsed = sqlglot.parse(sql)
            for statement in parsed:
                if statement is None:
                    continue
                if isinstance(statement, _DDL_TYPES):
                    return QueryType.DDL
                if isinstance(statement, _DML_TYPES):
                    return QueryType.DML
                if isinstance(statement, exp.Select):
                    return QueryType.SELECT
            # If we got here with parsed statements but none matched known types
            if parsed and any(s is not None for s in parsed):
                return QueryType.OTHER
        except sqlglot.errors.ParseError:
            pass

        # If nothing parsed, do a simple keyword check
        stripped = sql.strip().upper()
        if stripped.startswith("SELECT") or stripped.startswith("WITH"):
            return QueryType.SELECT

        return QueryType.OTHER

    def check_query_allowed(self, sql: str) -> tuple[bool, str]:
        """
        Check whether a SQL query is allowed under current access control.

        Returns:
            (allowed: bool, reason: str)
        """
        query_type = self.classify_query(sql)

        if query_type == QueryType.SELECT:
            return True, "SELECT queries are always allowed."

        if query_type in (QueryType.DDL, QueryType.DML):
            if not self.write_mode_enabled:
                return False, (
                    f"Blocked: {query_type.value} operation not allowed in "
                    f"read-only mode. Enable write mode explicitly to proceed."
                )
            else:
                # Write mode is on, but we still need confirmation
                raise WriteConfirmationRequired(
                    f"{query_type.value} operation detected. Write mode is enabled, "
                    f"but a separate confirmation is required before executing: {sql[:100]}"
                )

        # OTHER — allow with caution (e.g., EXPLAIN, SHOW)
        return True, "Non-modifying query allowed."

    def validate_no_injection(self, sql: str, user_input: str) -> bool:
        """
        Verify that user_input is NOT directly interpolated into the SQL string.

        This is a defense-in-depth check: the primary protection is always using
        parameterized queries. This catches cases where someone accidentally
        bypasses the parameterized path.

        Returns True if the SQL appears safe (user input not found verbatim
        in a suspicious position), False if potential injection detected.
        """
        if not user_input or len(user_input) < 3:
            return True

        # If the user input appears verbatim in the SQL in a non-parameterized
        # position, flag it. This is a heuristic, not a replacement for
        # always using parameterized queries.
        # Check for common injection patterns
        injection_patterns = [
            f"'{user_input}'",          # Directly quoted
            f'"{user_input}"',          # Double-quoted
            user_input,                 # Raw interpolation
        ]

        sql_upper = sql.upper()
        input_upper = user_input.upper()

        # Only flag if the input looks like it could be a SQL fragment
        sql_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER",
                        "TRUNCATE", "--", ";", "UNION", "OR 1=1", "' OR"]
        input_has_sql = any(kw in input_upper for kw in sql_keywords)

        if input_has_sql:
            for pattern in injection_patterns:
                if pattern in sql:
                    return False

        return True

    def enable_write_mode(self) -> str:
        """Enable write mode. Returns a warning message."""
        self.write_mode_enabled = True
        return (
            "WARNING: Write mode enabled. DDL/DML operations will still "
            "require explicit confirmation before execution."
        )

    def disable_write_mode(self) -> str:
        """Disable write mode, returning to read-only default."""
        self.write_mode_enabled = False
        self._pending_confirmation = {}
        return "Write mode disabled. Returning to read-only mode."

    def confirm_write(self, sql: str) -> tuple[bool, str]:
        """
        Confirm a write operation after write-mode check.

        Returns (allowed, reason).
        """
        if not self.write_mode_enabled:
            return False, "Write mode is not enabled."

        query_type = self.classify_query(sql)
        if query_type not in (QueryType.DDL, QueryType.DML):
            return True, "Not a write operation; no confirmation needed."

        return True, f"Write operation confirmed: {query_type.value}"
