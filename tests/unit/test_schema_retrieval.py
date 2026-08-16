"""
Phase 2 — Schema Retrieval tests.

Per Tests.md §1:
- Correct tables/columns retrieved for a given question
- Drift detection catches renamed/removed columns

Per Phases.md Phase 2 exit criteria:
- Index correctly reflects schema after a simulated column rename/migration
"""

import pytest
from src.agent.schema_retrieval import (
    SchemaIndex,
    TableInfo,
    ColumnInfo,
    RelationshipInfo,
    SchemaDrift,
)


def _make_users_table() -> TableInfo:
    return TableInfo(
        name="users",
        columns=[
            ColumnInfo(name="id", data_type="integer", is_primary_key=True),
            ColumnInfo(name="name", data_type="varchar"),
            ColumnInfo(name="email", data_type="varchar"),
            ColumnInfo(name="created_at", data_type="timestamp"),
        ],
    )


def _make_orders_table() -> TableInfo:
    return TableInfo(
        name="orders",
        columns=[
            ColumnInfo(name="id", data_type="integer", is_primary_key=True),
            ColumnInfo(name="user_id", data_type="integer", is_foreign_key=True,
                       foreign_key_target="users.id"),
            ColumnInfo(name="total", data_type="decimal"),
            ColumnInfo(name="status", data_type="varchar"),
        ],
    )


def _make_products_table() -> TableInfo:
    return TableInfo(
        name="products",
        columns=[
            ColumnInfo(name="id", data_type="integer", is_primary_key=True),
            ColumnInfo(name="name", data_type="varchar"),
            ColumnInfo(name="price", data_type="decimal"),
            ColumnInfo(name="category", data_type="varchar"),
        ],
    )


def _build_index() -> SchemaIndex:
    idx = SchemaIndex()
    idx.index_table(_make_users_table())
    idx.index_table(_make_orders_table())
    idx.index_table(_make_products_table())
    idx.index_relationship(RelationshipInfo(
        source_table="orders",
        source_column="user_id",
        target_table="users",
        target_column="id",
    ))
    return idx


class TestSchemaIndexing:
    """Test basic indexing operations."""

    def test_index_table(self):
        idx = SchemaIndex()
        idx.index_table(_make_users_table())
        assert "users" in idx.tables

    def test_index_multiple_tables(self):
        idx = _build_index()
        assert set(idx.get_all_table_names()) == {"users", "orders", "products"}

    def test_get_table(self):
        idx = _build_index()
        users = idx.get_table("users")
        assert users is not None
        assert users.name == "users"
        assert len(users.columns) == 4

    def test_get_nonexistent_table(self):
        idx = _build_index()
        assert idx.get_table("nonexistent") is None

    def test_index_relationship(self):
        idx = _build_index()
        assert len(idx.relationships) == 1
        assert idx.relationships[0].source_table == "orders"

    def test_no_duplicate_relationships(self):
        idx = _build_index()
        idx.index_relationship(RelationshipInfo(
            source_table="orders",
            source_column="user_id",
            target_table="users",
            target_column="id",
        ))
        assert len(idx.relationships) == 1  # Not duplicated

    def test_remove_table(self):
        idx = _build_index()
        idx.remove_table("products")
        assert "products" not in idx.tables
        assert len(idx.get_all_table_names()) == 2

    def test_remove_table_cleans_relationships(self):
        idx = _build_index()
        idx.remove_table("orders")
        assert len(idx.relationships) == 0  # orders had the FK

    def test_all_column_names(self):
        idx = _build_index()
        cols = idx.get_all_column_names()
        assert "users.id" in cols
        assert "orders.total" in cols
        assert "products.price" in cols


class TestSchemaRetrieval:
    """Test relevant schema retrieval for queries."""

    def test_retrieve_users_table(self):
        idx = _build_index()
        results = idx.retrieve_relevant("show me all users")
        table_names = [r.name for r in results if isinstance(r, TableInfo)]
        assert "users" in table_names

    def test_retrieve_orders_table(self):
        idx = _build_index()
        results = idx.retrieve_relevant("total orders by status")
        table_names = [r.name for r in results if isinstance(r, TableInfo)]
        assert "orders" in table_names

    def test_retrieve_includes_relationships(self):
        idx = _build_index()
        results = idx.retrieve_relevant("orders for each user")
        has_relationship = any(isinstance(r, RelationshipInfo) for r in results)
        assert has_relationship

    def test_retrieve_limits_to_top_k(self):
        idx = _build_index()
        results = idx.retrieve_relevant("show everything", top_k=1)
        tables = [r for r in results if isinstance(r, TableInfo)]
        assert len(tables) <= 1

    def test_schema_context_format(self):
        idx = _build_index()
        ctx = idx.get_schema_context("user orders")
        assert "table" in ctx.lower()
        # Should contain relevant info but not everything
        assert len(ctx) > 0

    def test_column_name_match_boosted(self):
        """A query mentioning a specific column name should retrieve the right table."""
        idx = _build_index()
        results = idx.retrieve_relevant("what is the price of each product")
        table_names = [r.name for r in results if isinstance(r, TableInfo)]
        assert "products" in table_names


class TestDriftDetection:
    """Test schema drift detection — the core of Phase 2 exit criteria."""

    def test_detect_added_table(self):
        idx = _build_index()
        live_tables = [
            _make_users_table(),
            _make_orders_table(),
            _make_products_table(),
            TableInfo(name="reviews", columns=[
                ColumnInfo(name="id", data_type="integer", is_primary_key=True),
                ColumnInfo(name="text", data_type="text"),
            ]),
        ]
        drifts = idx.detect_drift(live_tables)
        added = [d for d in drifts if d.drift_type == "added_table"]
        assert len(added) == 1
        assert added[0].table_name == "reviews"

    def test_detect_removed_table(self):
        idx = _build_index()
        live_tables = [_make_users_table(), _make_orders_table()]
        drifts = idx.detect_drift(live_tables)
        removed = [d for d in drifts if d.drift_type == "removed_table"]
        assert len(removed) == 1
        assert removed[0].table_name == "products"

    def test_detect_added_column(self):
        idx = _build_index()
        modified_users = _make_users_table()
        modified_users.columns.append(
            ColumnInfo(name="phone", data_type="varchar")
        )
        live_tables = [modified_users, _make_orders_table(), _make_products_table()]
        drifts = idx.detect_drift(live_tables)
        added = [d for d in drifts if d.drift_type == "added_column"]
        assert len(added) == 1
        assert added[0].column_name == "phone"
        assert added[0].table_name == "users"

    def test_detect_removed_column(self):
        idx = _build_index()
        modified_users = TableInfo(
            name="users",
            columns=[
                ColumnInfo(name="id", data_type="integer", is_primary_key=True),
                ColumnInfo(name="name", data_type="varchar"),
                # email column removed
                ColumnInfo(name="created_at", data_type="timestamp"),
            ],
        )
        live_tables = [modified_users, _make_orders_table(), _make_products_table()]
        drifts = idx.detect_drift(live_tables)
        removed = [d for d in drifts if d.drift_type == "removed_column"]
        assert len(removed) == 1
        assert removed[0].column_name == "email"

    def test_detect_renamed_column(self):
        """
        Phase 2 exit criteria: index correctly reflects schema after a
        simulated column rename.

        Simulates: users.email -> users.email_address
        The drift detector should flag this as a possible rename,
        not just an add+remove.
        """
        idx = _build_index()
        modified_users = TableInfo(
            name="users",
            columns=[
                ColumnInfo(name="id", data_type="integer", is_primary_key=True),
                ColumnInfo(name="name", data_type="varchar"),
                ColumnInfo(name="email_address", data_type="varchar"),  # was "email"
                ColumnInfo(name="created_at", data_type="timestamp"),
            ],
        )
        live_tables = [modified_users, _make_orders_table(), _make_products_table()]
        drifts = idx.detect_drift(live_tables)

        renamed = [d for d in drifts if d.drift_type == "renamed_column"]
        assert len(renamed) == 1
        assert renamed[0].old_value == "email"
        assert renamed[0].new_value == "email_address"

    def test_detect_type_change(self):
        idx = _build_index()
        modified_users = TableInfo(
            name="users",
            columns=[
                ColumnInfo(name="id", data_type="bigint", is_primary_key=True),  # was integer
                ColumnInfo(name="name", data_type="varchar"),
                ColumnInfo(name="email", data_type="varchar"),
                ColumnInfo(name="created_at", data_type="timestamp"),
            ],
        )
        live_tables = [modified_users, _make_orders_table(), _make_products_table()]
        drifts = idx.detect_drift(live_tables)
        type_changed = [d for d in drifts if d.drift_type == "type_changed"]
        assert len(type_changed) == 1
        assert type_changed[0].column_name == "id"
        assert type_changed[0].old_value == "integer"
        assert type_changed[0].new_value == "bigint"

    def test_no_drift_when_identical(self):
        idx = _build_index()
        live_tables = [_make_users_table(), _make_orders_table(), _make_products_table()]
        drifts = idx.detect_drift(live_tables)
        assert len(drifts) == 0

    def test_apply_drift_updates_index(self):
        """After applying drift, the index should reflect the new schema."""
        idx = _build_index()
        modified_users = TableInfo(
            name="users",
            columns=[
                ColumnInfo(name="id", data_type="integer", is_primary_key=True),
                ColumnInfo(name="name", data_type="varchar"),
                ColumnInfo(name="email_address", data_type="varchar"),  # renamed
                ColumnInfo(name="created_at", data_type="timestamp"),
                ColumnInfo(name="phone", data_type="varchar"),  # new
            ],
        )
        live_tables = [modified_users, _make_orders_table(), _make_products_table()]
        drifts = idx.detect_drift(live_tables)
        idx.apply_drift(live_tables, drifts)

        # Index should now reflect the updated schema
        users = idx.get_table("users")
        assert users is not None
        col_names = users.column_names()
        assert "email_address" in col_names
        assert "phone" in col_names
        assert "email" not in col_names  # old name gone

    def test_apply_drift_removes_table(self):
        idx = _build_index()
        live_tables = [_make_users_table(), _make_orders_table()]
        drifts = idx.detect_drift(live_tables)
        idx.apply_drift(live_tables, drifts)
        assert "products" not in idx.tables

    def test_apply_drift_adds_table(self):
        idx = _build_index()
        new_table = TableInfo(name="reviews", columns=[
            ColumnInfo(name="id", data_type="integer", is_primary_key=True),
        ])
        live_tables = [
            _make_users_table(), _make_orders_table(),
            _make_products_table(), new_table,
        ]
        drifts = idx.detect_drift(live_tables)
        idx.apply_drift(live_tables, drifts)
        assert "reviews" in idx.tables

    def test_drift_describe(self):
        """Ensure drift descriptions are human-readable."""
        drift = SchemaDrift(
            drift_type="renamed_column",
            table_name="users",
            old_value="email",
            new_value="email_address",
        )
        desc = drift.describe()
        assert "email" in desc
        assert "email_address" in desc
        assert "rename" in desc.lower()
