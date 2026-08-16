"""
SQL generator — produces SQL from natural language using retrieved schema,
glossary context, and correction memory.

Per Architecture.md §2: LLM call using retrieved schema slice + glossary
+ correction memory.

Per Rules.md §3: use sqlglot for SQL parsing/analysis.

Per Architecture.md §3: Every module routes through llm_client — nothing
calls a provider SDK directly.

This module wraps the LLM call and validates the output with sqlglot.
For unit testing without a real LLM, the LLM call is injectable via
either the unified LLMClient or a simple callable.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import sqlglot

from src.agent.llm_client import LLMClient as UnifiedLLMClient

logger = logging.getLogger(__name__)


class LLMClientProtocol(Protocol):
    """Protocol for LLM clients — allows injection for testing."""
    def generate(self, prompt: str) -> str: ...


@dataclass
class SQLGenerationResult:
    """Result of SQL generation."""
    sql: str
    raw_llm_response: str
    is_valid: bool
    parse_errors: list[str] = field(default_factory=list)
    tables_referenced: list[str] = field(default_factory=list)
    dialect: str = "postgres"


class MockLLMClient:
    """
    Mock LLM client for testing — returns predefined responses
    or generates simple SQL from patterns.
    """

    def __init__(self, responses: dict[str, str] | None = None):
        self._responses = responses or {}
        self._default_response = "SELECT 1"

    def generate(self, prompt: str) -> str:
        # Check for keyword-based response matching
        for keyword, response in self._responses.items():
            if keyword.lower() in prompt.lower():
                return response
        return self._default_response


class SQLGenerator:
    """
    Generates SQL from natural language queries.

    Uses:
    - Retrieved schema context (relevant tables/columns only)
    - Glossary definitions for term resolution
    - Correction memory rules for known fixes
    - sqlglot for syntax validation

    Routes through llm_client (Architecture.md §3). The actual LLM call
    is delegated to an injected client, making this testable without API calls.
    """

    def __init__(
        self,
        llm_client=None,
        dialect: str = "postgres",
    ):
        """
        Args:
            llm_client: Either a UnifiedLLMClient (from llm_client.py),
                a MockLLMClient, or any object with a generate(prompt) method.
            dialect: SQL dialect for validation (default: postgres).
        """
        if llm_client is None:
            self._llm = MockLLMClient()
        elif isinstance(llm_client, UnifiedLLMClient):
            # Wrap unified client to match simple protocol
            self._llm = _UnifiedClientWrapper(llm_client)
        else:
            self._llm = llm_client
        self._dialect = dialect

    def generate(
        self,
        question: str,
        schema_context: str,
        glossary_context: str = "",
        correction_rules: list[str] | None = None,
        restatement: str | None = None,
    ) -> SQLGenerationResult:
        """
        Generate SQL for a natural language question.

        Args:
            question: The user's natural language question.
            schema_context: Relevant schema info (from SchemaIndex).
            glossary_context: Resolved glossary terms.
            correction_rules: Previously learned correction rules.
            restatement: Intent restatement (from IntentReconstructor).

        Returns:
            SQLGenerationResult with validated SQL.
        """
        prompt = self._build_prompt(
            question=question,
            schema_context=schema_context,
            glossary_context=glossary_context,
            correction_rules=correction_rules or [],
            restatement=restatement,
        )

        raw_response = self._llm.generate(prompt)
        sql = self._extract_sql(raw_response)
        validation = self._validate_sql(sql)

        return SQLGenerationResult(
            sql=sql,
            raw_llm_response=raw_response,
            is_valid=validation["is_valid"],
            parse_errors=validation["errors"],
            tables_referenced=validation["tables"],
            dialect=self._dialect,
        )

    def _build_prompt(
        self,
        question: str,
        schema_context: str,
        glossary_context: str,
        correction_rules: list[str],
        restatement: str | None,
    ) -> str:
        """Build the SQL generation prompt."""
        parts = [
            "Generate a PostgreSQL SQL query for the following question.",
            "Use ONLY the tables and columns listed in the schema below.",
            "Do NOT reference any table or column not in the schema.",
            "Use parameterized queries with :param_name syntax for any user-provided values.",
            "Return ONLY the SQL query, no explanation.",
            "",
        ]

        if restatement:
            parts.append(f"Intent (restatement): {restatement}")
            parts.append("")

        parts.append(f"Question: {question}")
        parts.append("")
        parts.append(f"Schema:\n{schema_context}")

        if glossary_context:
            parts.append(f"\n{glossary_context}")

        if correction_rules:
            parts.append("\nCorrection rules (apply these):")
            for rule in correction_rules:
                parts.append(f"- {rule}")

        parts.append("\nSQL:")
        return "\n".join(parts)

    def _extract_sql(self, response: str) -> str:
        """Extract SQL from LLM response, stripping markdown fences etc."""
        sql = response.strip()

        # Remove markdown code fences
        if sql.startswith("```sql"):
            sql = sql[6:]
        elif sql.startswith("```"):
            sql = sql[3:]
        if sql.endswith("```"):
            sql = sql[:-3]

        sql = sql.strip()

        # Remove trailing semicolons for consistency
        if sql.endswith(";"):
            sql = sql[:-1].strip()

        return sql

    def _validate_sql(self, sql: str) -> dict[str, Any]:
        """
        Validate SQL syntax using sqlglot.

        Per Rules.md §3: use sqlglot for SQL parsing/analysis.
        """
        errors: list[str] = []
        tables: list[str] = []

        try:
            parsed = sqlglot.parse(sql, read="postgres")
            for statement in parsed:
                if statement is None:
                    continue
                # Extract referenced tables
                for table in statement.find_all(sqlglot.exp.Table):
                    table_name = table.name
                    if table_name:
                        tables.append(table_name)
        except sqlglot.errors.ParseError as e:
            errors.append(str(e))
        except Exception as e:
            errors.append(f"Unexpected parse error: {e}")

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "tables": list(set(tables)),
        }

    def validate_against_schema(
        self,
        result: SQLGenerationResult,
        known_tables: set[str],
    ) -> list[str]:
        """
        Check that all tables referenced in SQL actually exist in the schema.

        Per Rules.md §4: mitigate schema hallucination by checking references.
        """
        hallucinated = []
        for table in result.tables_referenced:
            if table.lower() not in {t.lower() for t in known_tables}:
                hallucinated.append(table)
        return hallucinated


class _UnifiedClientWrapper:
    """
    Wraps the unified LLMClient to match the simple generate(prompt) protocol.

    Routes through llm_client with the 'generation' role per Architecture.md
    Model Routing table.
    """

    def __init__(self, client: UnifiedLLMClient, role: str = "generation"):
        self._client = client
        self._role = role

    def generate(self, prompt: str) -> str:
        return self._client.generate(
            role=self._role,
            messages=[{"role": "user", "content": prompt}],
        )

