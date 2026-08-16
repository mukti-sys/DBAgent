"""
Phase 8 — Correction Memory tests.

Per Tests.md §1:
- correction_memory.py: a stored rule is retrieved and applied on a
  matching future query

Per Phases.md Phase 8 exit criteria:
- A corrected mistake does not recur on a repeat/near-repeat question
"""

import json
import os
import tempfile
import pytest
from pathlib import Path
from src.agent.correction_memory import CorrectionMemory, CorrectionRule


@pytest.fixture
def tmp_storage(tmp_path):
    """Provide a temp directory for rule storage."""
    return tmp_path / "corrections"


@pytest.fixture
def memory(tmp_storage):
    """Provide a CorrectionMemory instance with temp storage."""
    return CorrectionMemory(
        storage_dir=tmp_storage,
        org_id="test_org",
        db_fingerprint="test_db_fp",
    )


class TestCorrectionRuleMatching:
    """Test rule matching logic."""

    def test_exact_match_scores_highest(self):
        rule = CorrectionRule(
            rule_id="r1",
            original_question="What is the total revenue?",
            original_sql="SELECT SUM(amount) FROM orders",
            corrected_sql="SELECT SUM(total) FROM orders WHERE status = 'completed'",
            correction_description="Use 'total' column, only completed orders",
            keywords=["total", "revenue"],
        )
        score = rule.matches("What is the total revenue?")
        assert score == 1.0

    def test_similar_question_scores_above_threshold(self):
        rule = CorrectionRule(
            rule_id="r2",
            original_question="What is the total revenue?",
            original_sql="SELECT SUM(amount) FROM orders",
            corrected_sql="SELECT SUM(total) FROM orders WHERE status = 'completed'",
            correction_description="Use 'total' column",
            keywords=["total", "revenue"],
        )
        score = rule.matches("Show me total revenue by region")
        assert score > 0.0

    def test_unrelated_question_scores_low(self):
        rule = CorrectionRule(
            rule_id="r3",
            original_question="What is the total revenue?",
            original_sql="SELECT SUM(amount) FROM orders",
            corrected_sql="SELECT SUM(total) FROM orders",
            correction_description="Use 'total' column",
            keywords=["total", "revenue"],
        )
        score = rule.matches("How many users signed up yesterday?")
        assert score < 0.3

    def test_keyword_matching(self):
        rule = CorrectionRule(
            rule_id="r4",
            original_question="active users count",
            original_sql="SELECT COUNT(*) FROM users",
            corrected_sql="SELECT COUNT(*) FROM users WHERE last_login > NOW() - INTERVAL '30 days'",
            correction_description="Active means logged in within 30 days",
            keywords=["active", "users", "count"],
        )
        score = rule.matches("count of active users this month")
        assert score > 0.2

    def test_prompt_rule_format(self):
        rule = CorrectionRule(
            rule_id="r5",
            original_question="revenue by region",
            original_sql="SELECT region, SUM(amount) FROM orders GROUP BY region",
            corrected_sql="SELECT region, SUM(total) FROM orders WHERE status = 'completed' GROUP BY region",
            correction_description="Use total not amount, only completed orders",
        )
        prompt = rule.to_prompt_rule()
        assert "CORRECTION" in prompt
        assert "revenue by region" in prompt
        assert "total" in prompt


class TestCorrectionMemory:
    """Test CorrectionMemory storage and retrieval."""

    def test_add_and_retrieve(self, memory):
        memory.add_correction(
            original_question="What is the revenue?",
            original_sql="SELECT SUM(amount) FROM orders",
            corrected_sql="SELECT SUM(total) FROM orders WHERE status = 'completed'",
            correction_description="Use total column, filter completed only",
        )
        matches = memory.find_matching_rules("What is the revenue?")
        assert len(matches) >= 1
        rule, score = matches[0]
        assert score == 1.0
        assert "total" in rule.corrected_sql

    def test_no_match_for_unrelated(self, memory):
        memory.add_correction(
            original_question="What is the revenue?",
            original_sql="SELECT SUM(amount) FROM orders",
            corrected_sql="SELECT SUM(total) FROM orders",
            correction_description="Use total column",
        )
        matches = memory.find_matching_rules(
            "How many products are in stock?",
            threshold=0.4,
        )
        assert len(matches) == 0

    def test_get_prompt_rules(self, memory):
        memory.add_correction(
            original_question="Show me active users",
            original_sql="SELECT * FROM users",
            corrected_sql="SELECT * FROM users WHERE last_login > NOW() - INTERVAL '30 days'",
            correction_description="Active means logged in within 30 days",
        )
        rules = memory.get_prompt_rules("Show me active users")
        assert len(rules) >= 1
        assert "CORRECTION" in rules[0]
        assert "active" in rules[0].lower() or "30 days" in rules[0]

    def test_multiple_rules_ranked(self, memory):
        memory.add_correction(
            original_question="What is the revenue?",
            original_sql="SELECT SUM(amount) FROM orders",
            corrected_sql="SELECT SUM(total) FROM orders",
            correction_description="Use total column",
        )
        memory.add_correction(
            original_question="Revenue breakdown by region",
            original_sql="SELECT region, SUM(amount) FROM orders GROUP BY region",
            corrected_sql="SELECT region, SUM(total) FROM orders WHERE status = 'completed' GROUP BY region",
            correction_description="Use total, filter completed",
        )
        matches = memory.find_matching_rules("What is the revenue?")
        assert len(matches) >= 1
        # First match should be the exact question match
        assert matches[0][1] >= matches[-1][1]

    def test_remove_rule(self, memory):
        rule = memory.add_correction(
            original_question="test",
            original_sql="SELECT 1",
            corrected_sql="SELECT 2",
            correction_description="test fix",
        )
        assert len(memory.get_all_rules()) == 1
        memory.remove_rule(rule.rule_id)
        assert len(memory.get_all_rules()) == 0

    def test_clear_all_rules(self, memory):
        for i in range(5):
            memory.add_correction(
                original_question=f"question {i}",
                original_sql=f"SELECT {i}",
                corrected_sql=f"SELECT {i + 1}",
                correction_description=f"fix {i}",
            )
        assert len(memory.get_all_rules()) == 5
        memory.clear()
        assert len(memory.get_all_rules()) == 0


class TestPersistence:
    """Test that rules persist to disk and reload."""

    def test_persist_and_reload(self, tmp_storage):
        # Create memory, add rule
        mem1 = CorrectionMemory(
            storage_dir=tmp_storage,
            org_id="org1",
            db_fingerprint="db1",
        )
        mem1.add_correction(
            original_question="revenue query",
            original_sql="SELECT SUM(amount) FROM orders",
            corrected_sql="SELECT SUM(total) FROM orders",
            correction_description="Use total column",
        )

        # Create new memory instance, same scope — should load
        mem2 = CorrectionMemory(
            storage_dir=tmp_storage,
            org_id="org1",
            db_fingerprint="db1",
        )
        assert len(mem2.get_all_rules()) == 1
        assert mem2.get_all_rules()[0].corrected_sql == "SELECT SUM(total) FROM orders"

    def test_scope_isolation(self, tmp_storage):
        """Rules from org1 must NOT leak into org2."""
        mem1 = CorrectionMemory(
            storage_dir=tmp_storage,
            org_id="org1",
            db_fingerprint="db1",
        )
        mem1.add_correction(
            original_question="org1 query",
            original_sql="SELECT 1",
            corrected_sql="SELECT 2",
            correction_description="org1 fix",
        )

        mem2 = CorrectionMemory(
            storage_dir=tmp_storage,
            org_id="org2",
            db_fingerprint="db2",
        )
        # org2 should have zero rules — no leakage
        assert len(mem2.get_all_rules()) == 0

    def test_no_storage_dir_works_in_memory(self):
        """Memory-only mode (no persistence) should work without error."""
        mem = CorrectionMemory(storage_dir=None)
        mem.add_correction(
            original_question="test",
            original_sql="SELECT 1",
            corrected_sql="SELECT 2",
            correction_description="test",
        )
        assert len(mem.get_all_rules()) == 1


class TestCorrectionDoesNotRecur:
    """
    Phase 8 exit criteria: a corrected mistake does not recur on a
    repeat/near-repeat question.

    This tests the full correction flow: store a correction, then
    verify it's retrieved when the same or similar question is asked.
    """

    def test_exact_repeat_returns_correction(self, memory):
        """Same question asked again → correction rule surfaces."""
        memory.add_correction(
            original_question="What is the total revenue last quarter?",
            original_sql="SELECT SUM(amount) FROM orders WHERE date > '2024-01-01'",
            corrected_sql="SELECT SUM(total) FROM orders WHERE status = 'completed' AND date > '2024-01-01'",
            correction_description="Use 'total' not 'amount', filter by completed status",
        )
        # Repeat the exact same question
        rules = memory.get_prompt_rules("What is the total revenue last quarter?")
        assert len(rules) >= 1
        assert "total" in rules[0].lower()
        assert "completed" in rules[0].lower()

    def test_near_repeat_returns_correction(self, memory):
        """Similar (not identical) question → correction rule still surfaces."""
        memory.add_correction(
            original_question="What is the total revenue last quarter?",
            original_sql="SELECT SUM(amount) FROM orders",
            corrected_sql="SELECT SUM(total) FROM orders WHERE status = 'completed'",
            correction_description="Use total, filter completed",
            keywords=["total", "revenue", "quarter"],
        )
        # Ask a similar but not identical question
        rules = memory.get_prompt_rules(
            "Show me the revenue for last quarter",
            threshold=0.2,
        )
        assert len(rules) >= 1
        assert "total" in rules[0].lower()

    def test_correction_applied_count_tracked(self, memory):
        """Applied count increases each time a rule is used."""
        rule = memory.add_correction(
            original_question="user count",
            original_sql="SELECT COUNT(*) FROM users",
            corrected_sql="SELECT COUNT(*) FROM users WHERE active = true",
            correction_description="Count only active users",
            keywords=["user", "count"],
        )
        assert rule.applied_count == 0

        memory.get_prompt_rules("user count")
        # After applying, count should increase
        updated_rules = memory.get_all_rules()
        assert updated_rules[0].applied_count == 1

    def test_scope_check(self, memory):
        """Scope matching works correctly."""
        assert memory.rules_scope_matches("test_org", "test_db_fp") is True
        assert memory.rules_scope_matches("other_org", "test_db_fp") is False


class TestKeywordExtraction:
    """Test automatic keyword extraction."""

    def test_stopwords_removed(self, memory):
        keywords = memory._extract_keywords(
            "What is the total revenue for all users in the system?"
        )
        assert "the" not in keywords
        assert "is" not in keywords
        assert "for" not in keywords
        assert "revenue" in keywords
        assert "total" in keywords

    def test_short_words_filtered(self, memory):
        keywords = memory._extract_keywords("a I x the")
        assert len(keywords) == 0

    def test_meaningful_words_kept(self, memory):
        keywords = memory._extract_keywords(
            "Show me revenue by product category"
        )
        assert "revenue" in keywords
        assert "product" in keywords
        assert "category" in keywords
