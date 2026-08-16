"""
Phase 3 — Glossary & Intent Reconstruction tests.

Per Tests.md §1:
- intent.py: ambiguous vs. clear question classification; glossary term resolution

Per Phases.md Phase 3 exit criteria:
- Ambiguous test questions trigger a clarifying question instead of a guess
"""

import pytest
from src.agent.glossary import GlossaryStore, GlossaryTerm
from src.agent.intent import (
    IntentReconstructor,
    IntentResult,
    AmbiguityType,
)


def _make_glossary() -> GlossaryStore:
    return GlossaryStore(terms=[
        GlossaryTerm(
            term="active user",
            definition="A user who logged in at least once in the last 30 days",
            sql_expression="last_login >= NOW() - INTERVAL '30 days'",
            alternatives=["active users"],
        ),
        GlossaryTerm(
            term="revenue",
            definition="Sum of completed order totals, excluding refunds",
            sql_expression="SUM(orders.total) WHERE orders.status = 'completed'",
        ),
        GlossaryTerm(
            term="churn",
            definition="Users who had an account 90+ days ago but no login in last 30 days",
            sql_expression="",
            notes="Definition varies by team — marketing uses 60 days",
        ),
    ])


class TestGlossaryStore:
    """Test glossary lookup and conflict detection."""

    def test_lookup_exact_term(self):
        gs = _make_glossary()
        matches = gs.lookup("How many active users do we have?")
        assert len(matches) >= 1
        assert any(t.term == "active user" for t in matches)

    def test_lookup_alternative(self):
        gs = _make_glossary()
        matches = gs.lookup("Show me all active users")
        assert len(matches) >= 1

    def test_lookup_no_match(self):
        gs = _make_glossary()
        matches = gs.lookup("What is the weather today?")
        assert len(matches) == 0

    def test_lookup_multiple_terms(self):
        gs = _make_glossary()
        matches = gs.lookup("What is the revenue from active users?")
        terms = {t.term for t in matches}
        assert "revenue" in terms
        assert "active user" in terms

    def test_conflict_detection(self):
        gs = GlossaryStore(terms=[
            GlossaryTerm(term="revenue", definition="Total sales"),
            GlossaryTerm(term="revenue", definition="Net sales minus returns"),
        ])
        conflicts = gs.find_conflicts("revenue")
        assert len(conflicts) == 2

    def test_no_conflict_single_definition(self):
        gs = _make_glossary()
        conflicts = gs.find_conflicts("revenue")
        assert len(conflicts) == 0  # Only one definition

    def test_format_for_prompt(self):
        gs = _make_glossary()
        matches = gs.lookup("active user revenue")
        formatted = gs.format_for_prompt(matches)
        assert "active user" in formatted.lower()
        assert "revenue" in formatted.lower()
        assert "Glossary" in formatted

    def test_format_empty(self):
        gs = _make_glossary()
        assert gs.format_for_prompt([]) == ""

    def test_add_term(self):
        gs = GlossaryStore()
        gs.add_term(GlossaryTerm(term="test", definition="A test term"))
        assert len(gs.get_all_terms()) == 1

    def test_from_config(self):
        """Glossary loads from config without error."""
        gs = GlossaryStore.from_config()
        assert isinstance(gs, GlossaryStore)


class TestIntentReconstructor:
    """Test ambiguity detection and intent analysis."""

    def test_clear_question_not_ambiguous(self):
        """A clear, specific question should not be flagged as ambiguous."""
        ir = IntentReconstructor(glossary=_make_glossary())
        result = ir.analyze("How many users signed up in January 2024?")
        assert result.is_ambiguous is False
        assert result.clarifying_question is None

    def test_vague_top_triggers_ambiguity(self):
        """'top users' without criteria should trigger clarification."""
        ir = IntentReconstructor(glossary=_make_glossary())
        result = ir.analyze("Show me the top users")
        assert result.is_ambiguous is True
        assert result.needs_clarification() is True
        assert any(
            a.ambiguity_type == AmbiguityType.MISSING_FILTER
            for a in result.ambiguities
        )

    def test_vague_recent_triggers_ambiguity(self):
        """'recent orders' without time range should trigger clarification."""
        ir = IntentReconstructor(glossary=_make_glossary())
        result = ir.analyze("Show me recent orders")
        assert result.is_ambiguous is True
        assert any(
            a.ambiguity_type == AmbiguityType.UNCLEAR_TIMEFRAME
            for a in result.ambiguities
        )

    def test_undefined_metric_triggers_ambiguity(self):
        """A metric word not in the glossary should flag as unclear."""
        ir = IntentReconstructor(glossary=GlossaryStore())  # Empty glossary
        result = ir.analyze("What is the revenue this month?")
        assert result.is_ambiguous is True
        assert any(
            a.ambiguity_type == AmbiguityType.UNCLEAR_METRIC
            for a in result.ambiguities
        )

    def test_defined_metric_not_ambiguous(self):
        """A metric word WITH a glossary definition should NOT flag."""
        ir = IntentReconstructor(glossary=_make_glossary())
        result = ir.analyze("What is the revenue this month?")
        # Revenue IS in glossary, so no UNCLEAR_METRIC
        metric_ambiguities = [
            a for a in result.ambiguities
            if a.ambiguity_type == AmbiguityType.UNCLEAR_METRIC
            and "revenue" in a.description
        ]
        assert len(metric_ambiguities) == 0

    def test_conflicting_definitions_trigger_ambiguity(self):
        """Multiple glossary definitions for same term should trigger."""
        gs = GlossaryStore(terms=[
            GlossaryTerm(term="revenue", definition="Total sales"),
            GlossaryTerm(term="revenue", definition="Net sales minus returns"),
        ])
        ir = IntentReconstructor(glossary=gs)
        result = ir.analyze("What is the revenue?")
        assert result.is_ambiguous is True
        assert any(
            a.ambiguity_type == AmbiguityType.CONFLICTING_TERMS
            for a in result.ambiguities
        )

    def test_clarifying_question_generated(self):
        """When ambiguous, a clarifying question should be generated."""
        ir = IntentReconstructor(glossary=_make_glossary())
        result = ir.analyze("Show me the top users recently")
        assert result.clarifying_question is not None
        assert "clarify" in result.clarifying_question.lower() or "need" in result.clarifying_question.lower()

    def test_glossary_terms_resolved(self):
        """Matched glossary terms should be included in the result."""
        ir = IntentReconstructor(glossary=_make_glossary())
        result = ir.analyze("Count of active users with churn risk")
        assert len(result.matched_glossary_terms) >= 1
        terms = {t.term for t in result.matched_glossary_terms}
        assert "active user" in terms or "churn" in terms

    def test_ambiguous_question_returns_clarification_not_guess(self):
        """
        Phase 3 exit criteria: ambiguous test questions trigger a clarifying
        question instead of a guess.
        """
        ir = IntentReconstructor(glossary=GlossaryStore())
        result = ir.analyze("Show me the best users with high engagement recently")
        assert result.is_ambiguous is True
        assert result.needs_clarification() is True
        assert result.clarifying_question is not None
        # Should not have a restatement (that would be a guess)
        # Instead it should ask for clarification

    def test_restatement_prompt_built(self):
        """Restatement prompt includes question, glossary, and schema."""
        ir = IntentReconstructor(glossary=_make_glossary())
        prompt = ir.build_restatement_prompt(
            question="What is the revenue?",
            glossary_context="revenue: Sum of completed order totals",
            schema_context="table orders: id, total, status",
        )
        assert "revenue" in prompt
        assert "orders" in prompt
        assert "Restatement" in prompt

    def test_multiple_ambiguities_all_reported(self):
        """A deeply ambiguous question should report all ambiguities."""
        ir = IntentReconstructor(glossary=GlossaryStore())
        result = ir.analyze("Show me the top users with the best engagement recently")
        assert len(result.ambiguities) >= 2  # At least "top" and "best" and "recently"
