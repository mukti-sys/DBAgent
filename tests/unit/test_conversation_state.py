"""
Phase 11 — Multi-Turn Conversational State unit tests.

Per Phases.md Phase 11 exit criteria:
- "now break that down by region" correctly inherits prior filters in
  eval test cases
- Graceful degradation (ask, don't assume) when state is uncertain
"""

import pytest
from src.agent.conversation_state import (
    ConversationState,
    QueryContext,
    ContextInheritanceResult,
)


@pytest.fixture
def state():
    return ConversationState()


class TestFollowUpDetection:
    """Test follow-up vs fresh question detection."""

    def test_breakdown_detected_as_follow_up(self, state):
        state.add_turn(QueryContext(
            question="Show total revenue by month",
            sql="SELECT ...",
            filters={"metric": "revenue"},
            tables_used=["orders"],
        ))
        result = state.resolve_context("now break that down by region")
        assert result.is_follow_up is True

    def test_group_by_detected_as_follow_up(self, state):
        state.add_turn(QueryContext(
            question="How many orders last quarter?",
            sql="SELECT ...",
            filters={"time_range": "last_quarter"},
            tables_used=["orders"],
        ))
        result = state.resolve_context("group that by category")
        assert result.is_follow_up is True

    def test_fresh_question_not_follow_up(self, state):
        state.add_turn(QueryContext(
            question="Show total revenue",
            sql="SELECT ...",
        ))
        result = state.resolve_context("What is the total number of users?")
        assert result.is_follow_up is False

    def test_no_history_never_follow_up(self, state):
        result = state.resolve_context("now break that down by region")
        assert result.is_follow_up is False

    def test_but_for_detected_as_follow_up(self, state):
        state.add_turn(QueryContext(
            question="Show me sales in US",
            sql="SELECT ...",
            filters={"region": "US"},
            tables_used=["sales"],
        ))
        result = state.resolve_context("but for Europe instead")
        assert result.is_follow_up is True

    def test_also_detected_as_follow_up(self, state):
        state.add_turn(QueryContext(
            question="Show revenue",
            sql="SELECT ...",
        ))
        result = state.resolve_context("also show the count")
        assert result.is_follow_up is True


class TestFilterInheritance:
    """Test that prior filters are correctly inherited."""

    def test_inherits_filters_from_last_turn(self, state):
        state.add_turn(QueryContext(
            question="Show revenue for enterprise customers",
            sql="SELECT ...",
            filters={"customer_type": "enterprise", "metric": "revenue"},
            tables_used=["orders", "customers"],
        ))
        result = state.resolve_context("now break that down by region")
        assert result.is_follow_up is True
        assert result.inherited_filters["customer_type"] == "enterprise"
        assert result.inherited_filters["metric"] == "revenue"

    def test_inherits_tables_from_last_turn(self, state):
        state.add_turn(QueryContext(
            question="Show revenue",
            sql="SELECT ...",
            tables_used=["orders", "products"],
        ))
        result = state.resolve_context("now break that down by category")
        assert "orders" in result.inherited_tables
        assert "products" in result.inherited_tables

    def test_inherits_aggregation(self, state):
        state.add_turn(QueryContext(
            question="Count orders by status",
            sql="SELECT ...",
            aggregation="count",
        ))
        result = state.resolve_context("but for last month only")
        assert result.inherited_aggregation == "count"

    def test_inherits_time_range(self, state):
        state.add_turn(QueryContext(
            question="Show sales last quarter",
            sql="SELECT ...",
            time_range="last_quarter",
            tables_used=["sales"],
        ))
        result = state.resolve_context("break that down by region")
        assert result.inherited_time_range == "last_quarter"

    def test_breakdown_by_region_inherits_all_prior_context(self, state):
        """Key eval test: 'break that down by region' inherits prior filters."""
        state.add_turn(QueryContext(
            question="What is total revenue for Q3 2024 enterprise customers?",
            sql="SELECT SUM(amount) FROM orders WHERE ...",
            filters={
                "time_range": "Q3 2024",
                "customer_type": "enterprise",
            },
            tables_used=["orders", "customers"],
            aggregation="sum",
            time_range="Q3 2024",
        ))

        result = state.resolve_context("now break that down by region")

        assert result.is_follow_up is True
        assert result.inherited_filters["time_range"] == "Q3 2024"
        assert result.inherited_filters["customer_type"] == "enterprise"
        assert result.inherited_filters["group_by"] == "region"
        assert "orders" in result.inherited_tables
        assert result.inherited_aggregation == "sum"
        assert result.inherited_time_range == "Q3 2024"

    def test_session_filters_accumulate(self, state):
        state.add_turn(QueryContext(
            question="Show orders for enterprise",
            filters={"customer_type": "enterprise"},
            tables_used=["orders"],
        ))
        state.add_turn(QueryContext(
            question="filter to last month",
            filters={"time_range": "last_month"},
            tables_used=["orders"],
        ))
        result = state.resolve_context("now break that down by region")
        # Both filters should be inherited
        assert result.inherited_filters.get("customer_type") == "enterprise"
        assert result.inherited_filters.get("time_range") == "last_month"


class TestAmbiguityHandling:
    """Test graceful degradation when context is uncertain."""

    def test_ambiguous_reference_with_multiple_table_contexts(self, state):
        """When 3+ turns reference different tables, pronoun references are ambiguous."""
        state.add_turn(QueryContext(
            question="Show user signups",
            tables_used=["users"],
        ))
        state.add_turn(QueryContext(
            question="Show order counts",
            tables_used=["orders"],
        ))
        state.add_turn(QueryContext(
            question="Show product categories",
            tables_used=["products", "categories"],
        ))
        # "filter that" is ambiguous — which prior query?
        result = state.resolve_context("filter that by status")
        assert result.is_follow_up is True
        assert result.is_ambiguous is True
        assert result.clarifying_question is not None

    def test_clear_reference_not_ambiguous(self, state):
        state.add_turn(QueryContext(
            question="Show total revenue",
            tables_used=["orders"],
            filters={"metric": "revenue"},
        ))
        result = state.resolve_context("break that down by region")
        # Only 1 prior turn — reference is clear
        assert result.is_ambiguous is False


class TestContextPromptBuilding:
    """Test that context prompts are built correctly."""

    def test_follow_up_prompt_includes_filters(self, state):
        state.add_turn(QueryContext(
            question="Show revenue",
            filters={"metric": "revenue"},
            tables_used=["orders"],
        ))
        result = state.resolve_context("break that down by region")
        prompt = state.build_context_prompt("break that down by region", result)
        assert "revenue" in prompt
        assert "follow-up" in prompt.lower()
        assert "orders" in prompt

    def test_fresh_question_prompt_no_inheritance(self, state):
        result = state.resolve_context("What is the total number of users?")
        prompt = state.build_context_prompt("What is the total number of users?", result)
        assert "follow-up" not in prompt.lower()


class TestSessionManagement:
    """Test session lifecycle."""

    def test_clear_resets_state(self, state):
        state.add_turn(QueryContext(question="test", filters={"a": 1}))
        assert state.turn_count == 1
        state.clear()
        assert state.turn_count == 0
        assert state.session_filters == {}

    def test_max_history_enforced(self):
        state = ConversationState(max_history=3)
        for i in range(5):
            state.add_turn(QueryContext(question=f"q{i}"))
        assert state.turn_count == 3
        assert state.last_context.question == "q4"
