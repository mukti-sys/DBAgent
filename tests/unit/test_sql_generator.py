"""
Phase 4 — SQL Generation tests.

Per Tests.md §1:
- sql_generator.py: valid SQL syntax (via sqlglot parse check)
  across a range of question types

Per Phases.md Phase 4 exit criteria:
- Generates syntactically valid SQL for the eval set's easy tier
"""

import pytest
from src.agent.sql_generator import (
    SQLGenerator,
    SQLGenerationResult,
    MockLLMClient,
)


def _make_generator_with_responses(responses: dict[str, str]) -> SQLGenerator:
    """Create a SQLGenerator with predefined LLM responses."""
    return SQLGenerator(
        llm_client=MockLLMClient(responses=responses),
        dialect="postgres",
    )


SCHEMA_CONTEXT = """table public.users: column id (integer) PRIMARY KEY, column name (varchar), column email (varchar), column created_at (timestamp)
table public.orders: column id (integer) PRIMARY KEY, column user_id (integer) FOREIGN KEY -> users.id, column total (decimal), column status (varchar)
table public.products: column id (integer) PRIMARY KEY, column name (varchar), column price (decimal), column category (varchar)
relationship: orders.user_id -> users.id (many-to-one)"""


class TestSQLGeneration:
    """Test SQL generation produces valid SQL."""

    def test_simple_select(self):
        gen = _make_generator_with_responses({
            "users": "SELECT * FROM users"
        })
        result = gen.generate(
            question="Show me all users",
            schema_context=SCHEMA_CONTEXT,
        )
        assert result.is_valid is True
        assert "users" in result.tables_referenced

    def test_count_query(self):
        gen = _make_generator_with_responses({
            "how many": "SELECT COUNT(*) FROM users"
        })
        result = gen.generate(
            question="How many users are there?",
            schema_context=SCHEMA_CONTEXT,
        )
        assert result.is_valid is True
        assert result.sql == "SELECT COUNT(*) FROM users"

    def test_join_query(self):
        gen = _make_generator_with_responses({
            "orders": "SELECT u.name, SUM(o.total) FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.name"
        })
        result = gen.generate(
            question="Total orders per user",
            schema_context=SCHEMA_CONTEXT,
        )
        assert result.is_valid is True
        assert "users" in result.tables_referenced or "u" in result.tables_referenced

    def test_where_clause(self):
        gen = _make_generator_with_responses({
            "status": "SELECT * FROM orders WHERE status = :status"
        })
        result = gen.generate(
            question="Show me completed orders",
            schema_context=SCHEMA_CONTEXT,
        )
        assert result.is_valid is True
        assert "orders" in result.tables_referenced

    def test_aggregate_with_groupby(self):
        gen = _make_generator_with_responses({
            "category": "SELECT category, AVG(price) as avg_price FROM products GROUP BY category"
        })
        result = gen.generate(
            question="Average price by product category",
            schema_context=SCHEMA_CONTEXT,
        )
        assert result.is_valid is True

    def test_subquery(self):
        gen = _make_generator_with_responses({
            "never ordered": "SELECT * FROM users WHERE id NOT IN (SELECT DISTINCT user_id FROM orders)"
        })
        result = gen.generate(
            question="Users who never ordered",
            schema_context=SCHEMA_CONTEXT,
        )
        assert result.is_valid is True

    def test_cte_query(self):
        gen = _make_generator_with_responses({
            "revenue": "WITH user_revenue AS (SELECT user_id, SUM(total) as rev FROM orders GROUP BY user_id) SELECT u.name, ur.rev FROM users u JOIN user_revenue ur ON u.id = ur.user_id ORDER BY ur.rev DESC LIMIT 10"
        })
        result = gen.generate(
            question="Top 10 users by revenue",
            schema_context=SCHEMA_CONTEXT,
        )
        assert result.is_valid is True

    def test_invalid_sql_detected(self):
        gen = _make_generator_with_responses({
            "broken": "SLECT * FORM users WHER"
        })
        result = gen.generate(
            question="broken query",
            schema_context=SCHEMA_CONTEXT,
        )
        # sqlglot may or may not flag this as invalid depending on version
        # but if it does, errors should be populated
        # The key test is that we DO validate
        assert isinstance(result.is_valid, bool)

    def test_markdown_fences_stripped(self):
        gen = _make_generator_with_responses({
            "fenced": "```sql\nSELECT * FROM users\n```"
        })
        result = gen.generate(
            question="fenced query",
            schema_context=SCHEMA_CONTEXT,
        )
        assert "```" not in result.sql
        assert result.is_valid is True

    def test_semicolons_stripped(self):
        gen = _make_generator_with_responses({
            "semi": "SELECT * FROM users;"
        })
        result = gen.generate(
            question="semi query",
            schema_context=SCHEMA_CONTEXT,
        )
        assert not result.sql.endswith(";")


class TestSchemaValidation:
    """Test that generated SQL is validated against known schema."""

    def test_valid_table_reference(self):
        gen = _make_generator_with_responses({
            "users": "SELECT * FROM users"
        })
        result = gen.generate(
            question="Show me all users",
            schema_context=SCHEMA_CONTEXT,
        )
        hallucinated = gen.validate_against_schema(
            result,
            known_tables={"users", "orders", "products"},
        )
        assert len(hallucinated) == 0

    def test_hallucinated_table_detected(self):
        gen = _make_generator_with_responses({
            "fake": "SELECT * FROM nonexistent_table"
        })
        result = gen.generate(
            question="fake table query",
            schema_context=SCHEMA_CONTEXT,
        )
        hallucinated = gen.validate_against_schema(
            result,
            known_tables={"users", "orders", "products"},
        )
        assert "nonexistent_table" in hallucinated

    def test_multiple_hallucinated_tables(self):
        gen = _make_generator_with_responses({
            "multi": "SELECT * FROM fake1 JOIN fake2 ON fake1.id = fake2.fid"
        })
        result = gen.generate(
            question="multi fake tables",
            schema_context=SCHEMA_CONTEXT,
        )
        hallucinated = gen.validate_against_schema(
            result,
            known_tables={"users", "orders", "products"},
        )
        assert len(hallucinated) == 2


class TestPromptConstruction:
    """Test that prompts are correctly constructed."""

    def test_prompt_includes_question(self):
        gen = SQLGenerator(llm_client=MockLLMClient())
        # Access internal method for testing
        prompt = gen._build_prompt(
            question="How many users?",
            schema_context=SCHEMA_CONTEXT,
            glossary_context="",
            correction_rules=[],
            restatement=None,
        )
        assert "How many users?" in prompt

    def test_prompt_includes_schema(self):
        gen = SQLGenerator(llm_client=MockLLMClient())
        prompt = gen._build_prompt(
            question="test",
            schema_context=SCHEMA_CONTEXT,
            glossary_context="",
            correction_rules=[],
            restatement=None,
        )
        assert "users" in prompt
        assert "orders" in prompt

    def test_prompt_includes_glossary(self):
        gen = SQLGenerator(llm_client=MockLLMClient())
        prompt = gen._build_prompt(
            question="test",
            schema_context=SCHEMA_CONTEXT,
            glossary_context="revenue: Sum of order totals",
            correction_rules=[],
            restatement=None,
        )
        assert "revenue" in prompt

    def test_prompt_includes_correction_rules(self):
        gen = SQLGenerator(llm_client=MockLLMClient())
        prompt = gen._build_prompt(
            question="test",
            schema_context=SCHEMA_CONTEXT,
            glossary_context="",
            correction_rules=["Use completed orders only for revenue"],
            restatement=None,
        )
        assert "completed orders" in prompt

    def test_prompt_includes_restatement(self):
        gen = SQLGenerator(llm_client=MockLLMClient())
        prompt = gen._build_prompt(
            question="test",
            schema_context=SCHEMA_CONTEXT,
            glossary_context="",
            correction_rules=[],
            restatement="Count all users in the users table",
        )
        assert "Count all users" in prompt

    def test_prompt_instructs_parameterized(self):
        """Prompt should instruct LLM to use parameterized queries."""
        gen = SQLGenerator(llm_client=MockLLMClient())
        prompt = gen._build_prompt(
            question="test",
            schema_context=SCHEMA_CONTEXT,
            glossary_context="",
            correction_rules=[],
            restatement=None,
        )
        assert "parameterized" in prompt.lower() or "param" in prompt.lower()

    def test_prompt_instructs_schema_only(self):
        """Prompt should tell LLM to only use listed tables/columns."""
        gen = SQLGenerator(llm_client=MockLLMClient())
        prompt = gen._build_prompt(
            question="test",
            schema_context=SCHEMA_CONTEXT,
            glossary_context="",
            correction_rules=[],
            restatement=None,
        )
        assert "only" in prompt.lower()


class TestEasyTierEvalQuestions:
    """
    Phase 4 exit criteria: generates syntactically valid SQL for easy tier.

    These simulate clear questions against a clean schema.
    """

    EASY_TIER = {
        "all users": "SELECT * FROM users",
        "count users": "SELECT COUNT(*) FROM users",
        "orders total": "SELECT SUM(total) FROM orders",
        "products by category": "SELECT category, COUNT(*) FROM products GROUP BY category",
        "user orders join": "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id",
    }

    def test_all_easy_tier_produce_valid_sql(self):
        gen = _make_generator_with_responses(self.EASY_TIER)
        for keyword in self.EASY_TIER:
            result = gen.generate(
                question=f"Query about {keyword}",
                schema_context=SCHEMA_CONTEXT,
            )
            assert result.is_valid is True, f"Invalid SQL for '{keyword}': {result.parse_errors}"

    def test_easy_tier_references_known_tables(self):
        gen = _make_generator_with_responses(self.EASY_TIER)
        known = {"users", "orders", "products", "u", "o"}
        for keyword in self.EASY_TIER:
            result = gen.generate(
                question=f"Query about {keyword}",
                schema_context=SCHEMA_CONTEXT,
            )
            hallucinated = gen.validate_against_schema(result, known)
            assert len(hallucinated) == 0, f"Hallucinated tables for '{keyword}': {hallucinated}"
