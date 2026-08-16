"""
Phase 9 — Key Resolver unit tests.

Per Tests.md §1:
- key_resolver.py: candidate key matches ranked correctly; ambiguous cases
  flagged, not auto-joined

Per Phases.md Phase 9 exit criteria:
- Correctly flags an ambiguous key match instead of silently joining;
  remembers confirmation
"""

import json
import pytest
from pathlib import Path
from src.agent.key_resolver import (
    KeyResolver,
    MatchConfidenceTier,
    CandidateJoin,
)


@pytest.fixture
def tmp_storage(tmp_path):
    return tmp_path / "key_memory"


@pytest.fixture
def resolver(tmp_storage):
    return KeyResolver(
        storage_dir=tmp_storage,
        org_id="test_org",
        db_fingerprint="test_db",
    )


class TestCandidateKeyRanking:
    """Test candidate key scoring and ranking."""

    def test_exact_fk_pattern_ranked_highest(self, resolver):
        # orders.user_id -> users.id
        result = resolver.resolve_join(
            left_table="orders",
            left_columns=["id", "user_id", "total_amount", "created_at"],
            right_table="users",
            right_columns=["id", "email", "name", "created_at"],
        )
        assert result.selected_join is not None
        assert result.selected_join.left_column == "user_id"
        assert result.selected_join.right_column == "id"
        assert result.tier == MatchConfidenceTier.HIGH
        assert result.requires_confirmation is False

    def test_shared_column_name_match(self, resolver):
        # transactions.account_id -> balances.account_id
        result = resolver.resolve_join(
            left_table="transactions",
            left_columns=["tx_id", "account_id", "amount"],
            right_table="balances",
            right_columns=["balance_id", "account_id", "current_balance"],
        )
        assert result.selected_join is not None
        assert result.selected_join.left_column == "account_id"
        assert result.selected_join.right_column == "account_id"
        assert result.tier == MatchConfidenceTier.HIGH

    def test_dirty_abbreviation_matched(self, resolver):
        # orders.cust_id -> customers.id
        result = resolver.resolve_join(
            left_table="orders",
            left_columns=["id", "cust_id", "amount"],
            right_table="customers",
            right_columns=["id", "customer_name"],
        )
        assert len(result.candidates) > 0
        top = result.candidates[0]
        assert top.left_column == "cust_id"
        assert top.right_column == "id"


class TestAmbiguityFlagging:
    """Test that ambiguous candidate matches are flagged and never auto-joined."""

    def test_multiple_competing_fk_columns_flagged_as_ambiguous(self, resolver):
        # orders has buyer_id, seller_id, and created_by_id -> users.id
        # All 3 are user IDs with close scores
        result = resolver.resolve_join(
            left_table="orders",
            left_columns=["id", "buyer_user_id", "seller_user_id", "created_by_user_id"],
            right_table="users",
            right_columns=["id", "name"],
        )
        assert result.tier == MatchConfidenceTier.AMBIGUOUS
        assert result.requires_confirmation is True
        assert result.selected_join is None
        assert result.is_safe_to_join is False
        assert len(result.candidates) >= 2

    def test_generic_ids_not_auto_joined(self, resolver):
        # orders.id and products.id should not silently join
        result = resolver.resolve_join(
            left_table="orders",
            left_columns=["id", "amount"],
            right_table="products",
            right_columns=["id", "price"],
        )
        # Should be NO_MATCH or AMBIGUOUS, never silently joined
        assert result.is_safe_to_join is False
        assert result.requires_confirmation is True

    def test_unrelated_tables_return_no_match(self, resolver):
        result = resolver.resolve_join(
            left_table="weather_logs",
            left_columns=["temperature", "humidity", "recorded_at"],
            right_table="inventory",
            right_columns=["sku", "quantity", "warehouse_shelf"],
        )
        assert result.tier == MatchConfidenceTier.NO_MATCH
        assert result.requires_confirmation is True
        assert result.selected_join is None


class TestConfirmOnceRememberFlow:
    """Test user confirmation and persistence in key memory."""

    def test_remember_and_reuse_confirmed_join(self, resolver):
        # Suppose orders.buyer_user_id vs users.id was ambiguous
        initial_result = resolver.resolve_join(
            left_table="orders",
            left_columns=["id", "buyer_user_id", "seller_user_id"],
            right_table="users",
            right_columns=["id", "name"],
        )
        assert initial_result.requires_confirmation is True

        # User confirms buyer_user_id
        resolver.remember_join(
            left_table="orders",
            left_column="buyer_user_id",
            right_table="users",
            right_column="id",
            notes="Confirmed for buyer queries",
        )

        # Subsequent query on the same pair of tables now resolves immediately as CONFIRMED
        second_result = resolver.resolve_join(
            left_table="orders",
            left_columns=["id", "buyer_user_id", "seller_user_id"],
            right_table="users",
            right_columns=["id", "name"],
        )
        assert second_result.tier == MatchConfidenceTier.CONFIRMED
        assert second_result.requires_confirmation is False
        assert second_result.selected_join is not None
        assert second_result.selected_join.left_column == "buyer_user_id"
        assert second_result.selected_join.right_column == "id"
        assert second_result.is_safe_to_join is True

    def test_persistence_across_instances(self, tmp_storage):
        # Create first resolver and remember join
        res1 = KeyResolver(storage_dir=tmp_storage, org_id="orgA", db_fingerprint="db1")
        res1.remember_join(
            left_table="legacy_orders",
            left_column="legacy_uid",
            right_table="users",
            right_column="id",
        )

        # Create new resolver with same storage & scope
        res2 = KeyResolver(storage_dir=tmp_storage, org_id="orgA", db_fingerprint="db1")
        result = res2.resolve_join(
            left_table="legacy_orders",
            left_columns=["legacy_uid", "order_date"],
            right_table="users",
            right_columns=["id", "username"],
        )
        assert result.tier == MatchConfidenceTier.CONFIRMED
        assert result.selected_join.left_column == "legacy_uid"
        assert result.selected_join.right_column == "id"

    def test_scope_isolation_in_key_memory(self, tmp_storage):
        # Remember for orgA
        resA = KeyResolver(storage_dir=tmp_storage, org_id="orgA", db_fingerprint="db1")
        resA.remember_join(
            left_table="orders",
            left_column="buyer_id",
            right_table="users",
            right_column="id",
        )

        # orgB should not have this mapping
        resB = KeyResolver(storage_dir=tmp_storage, org_id="orgB", db_fingerprint="db1")
        assert len(resB.get_confirmed_joins()) == 0
