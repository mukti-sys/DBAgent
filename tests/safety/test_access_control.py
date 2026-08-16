"""
Phase 1 — Safety tests for Access Control & Safety Layer.

Per Tests.md §3 (Safety Tests — must never fail, ever):
- Natural-language request that implies a DROP/DELETE/UPDATE → blocked in read-only mode
- Crafted SQL-injection-style user input → parameterized/escaped, never interpolated
- DDL/DML blocking with write-mode flag

These tests do NOT require a live database — they test the access control
logic and query classification in isolation.
"""

import pytest
from src.db.access_control import (
    AccessController,
    AccessDeniedError,
    WriteConfirmationRequired,
    QueryType,
)


class TestQueryClassification:
    """Test that queries are classified correctly."""

    def test_select_classified(self):
        ac = AccessController()
        assert ac.classify_query("SELECT * FROM users") == QueryType.SELECT

    def test_select_with_where(self):
        ac = AccessController()
        assert ac.classify_query("SELECT id, name FROM users WHERE active = true") == QueryType.SELECT

    def test_select_with_join(self):
        ac = AccessController()
        sql = "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id"
        assert ac.classify_query(sql) == QueryType.SELECT

    def test_select_with_cte(self):
        ac = AccessController()
        sql = "WITH active_users AS (SELECT * FROM users WHERE active = true) SELECT * FROM active_users"
        assert ac.classify_query(sql) == QueryType.SELECT

    def test_drop_classified_as_ddl(self):
        ac = AccessController()
        assert ac.classify_query("DROP TABLE users") == QueryType.DDL

    def test_drop_database_classified(self):
        ac = AccessController()
        assert ac.classify_query("DROP DATABASE production") == QueryType.DDL

    def test_alter_classified_as_ddl(self):
        ac = AccessController()
        assert ac.classify_query("ALTER TABLE users ADD COLUMN email VARCHAR(255)") == QueryType.DDL

    def test_truncate_classified_as_ddl(self):
        ac = AccessController()
        assert ac.classify_query("TRUNCATE TABLE users") == QueryType.DDL

    def test_create_classified_as_ddl(self):
        ac = AccessController()
        assert ac.classify_query("CREATE TABLE test (id INT)") == QueryType.DDL

    def test_delete_classified_as_dml(self):
        ac = AccessController()
        assert ac.classify_query("DELETE FROM users WHERE id = 1") == QueryType.DML

    def test_update_classified_as_dml(self):
        ac = AccessController()
        assert ac.classify_query("UPDATE users SET name = 'test' WHERE id = 1") == QueryType.DML

    def test_insert_classified_as_dml(self):
        ac = AccessController()
        assert ac.classify_query("INSERT INTO users (name) VALUES ('test')") == QueryType.DML


class TestReadOnlyMode:
    """Test that read-only mode blocks dangerous operations."""

    def test_select_allowed_in_read_only(self):
        ac = AccessController(write_mode_enabled=False)
        allowed, reason = ac.check_query_allowed("SELECT * FROM users")
        assert allowed is True

    def test_drop_blocked_in_read_only(self):
        ac = AccessController(write_mode_enabled=False)
        allowed, reason = ac.check_query_allowed("DROP TABLE users")
        assert allowed is False
        assert "read-only" in reason.lower()

    def test_delete_blocked_in_read_only(self):
        ac = AccessController(write_mode_enabled=False)
        allowed, reason = ac.check_query_allowed("DELETE FROM users")
        assert allowed is False

    def test_update_blocked_in_read_only(self):
        ac = AccessController(write_mode_enabled=False)
        allowed, reason = ac.check_query_allowed("UPDATE users SET name = 'x'")
        assert allowed is False

    def test_insert_blocked_in_read_only(self):
        ac = AccessController(write_mode_enabled=False)
        allowed, reason = ac.check_query_allowed("INSERT INTO users (name) VALUES ('x')")
        assert allowed is False

    def test_truncate_blocked_in_read_only(self):
        ac = AccessController(write_mode_enabled=False)
        allowed, reason = ac.check_query_allowed("TRUNCATE TABLE users")
        assert allowed is False

    def test_alter_blocked_in_read_only(self):
        ac = AccessController(write_mode_enabled=False)
        allowed, reason = ac.check_query_allowed("ALTER TABLE users DROP COLUMN email")
        assert allowed is False

    def test_create_blocked_in_read_only(self):
        ac = AccessController(write_mode_enabled=False)
        allowed, reason = ac.check_query_allowed("CREATE TABLE evil (id INT)")
        assert allowed is False


class TestCraftedDangerousInputs:
    """
    Per Tests.md §3: Natural-language request that implies DROP/DELETE/UPDATE
    must be blocked. Test with SQL that an LLM might generate from crafted input.
    """

    def test_drop_via_crafted_nl_input(self):
        """Simulates: user says 'delete all records from users table'
        and the LLM generates a DELETE statement."""
        ac = AccessController(write_mode_enabled=False)
        sql = "DELETE FROM users"
        allowed, reason = ac.check_query_allowed(sql)
        assert allowed is False, "DELETE must be blocked in read-only mode"

    def test_drop_table_via_crafted_input(self):
        """Simulates: user says 'drop the users table'
        and the LLM generates a DROP TABLE statement."""
        ac = AccessController(write_mode_enabled=False)
        sql = "DROP TABLE users CASCADE"
        allowed, reason = ac.check_query_allowed(sql)
        assert allowed is False, "DROP TABLE must be blocked in read-only mode"

    def test_update_via_crafted_input(self):
        """Simulates: user says 'set all users to inactive'
        and the LLM generates an UPDATE."""
        ac = AccessController(write_mode_enabled=False)
        sql = "UPDATE users SET active = false"
        allowed, reason = ac.check_query_allowed(sql)
        assert allowed is False, "UPDATE must be blocked in read-only mode"

    def test_sneaky_delete_with_subquery(self):
        """DELETE with a subquery to make it look more complex."""
        ac = AccessController(write_mode_enabled=False)
        sql = "DELETE FROM orders WHERE user_id IN (SELECT id FROM users WHERE active = false)"
        allowed, reason = ac.check_query_allowed(sql)
        assert allowed is False, "Complex DELETE must still be blocked"

    def test_multiline_drop(self):
        """Multi-line SQL with a DROP."""
        ac = AccessController(write_mode_enabled=False)
        sql = """
        -- just cleaning up
        DROP TABLE IF EXISTS users;
        """
        allowed, reason = ac.check_query_allowed(sql)
        assert allowed is False, "Multi-line DROP must be blocked"


class TestSQLInjectionProtection:
    """
    Per Tests.md §3: Crafted SQL-injection-style user input must be
    parameterized/escaped, never interpolated.
    """

    def test_injection_via_single_quote(self):
        ac = AccessController()
        user_input = "'; DROP TABLE users; --"
        sql = f"SELECT * FROM users WHERE name = '{user_input}'"
        assert ac.validate_no_injection(sql, user_input) is False

    def test_injection_via_or_1_equals_1(self):
        ac = AccessController()
        user_input = "' OR 1=1 --"
        sql = f"SELECT * FROM users WHERE name = '{user_input}'"
        assert ac.validate_no_injection(sql, user_input) is False

    def test_injection_via_union(self):
        ac = AccessController()
        user_input = "' UNION SELECT password FROM admin_users --"
        sql = f"SELECT * FROM users WHERE name = '{user_input}'"
        assert ac.validate_no_injection(sql, user_input) is False

    def test_safe_parameterized_not_flagged(self):
        """A properly parameterized query should not be flagged."""
        ac = AccessController()
        user_input = "John"
        sql = "SELECT * FROM users WHERE name = :name"
        assert ac.validate_no_injection(sql, user_input) is True

    def test_safe_input_not_flagged(self):
        """Normal user input in a parameterized query."""
        ac = AccessController()
        user_input = "How many users signed up last month?"
        sql = "SELECT COUNT(*) FROM users WHERE created_at > :start_date"
        assert ac.validate_no_injection(sql, user_input) is True


class TestWriteMode:
    """Test write-mode enable/disable and confirmation flow."""

    def test_write_mode_default_off(self):
        ac = AccessController()
        assert ac.write_mode_enabled is False

    def test_enable_write_mode(self):
        ac = AccessController()
        msg = ac.enable_write_mode()
        assert ac.write_mode_enabled is True
        assert "WARNING" in msg

    def test_disable_write_mode(self):
        ac = AccessController()
        ac.enable_write_mode()
        msg = ac.disable_write_mode()
        assert ac.write_mode_enabled is False
        assert "read-only" in msg.lower()

    def test_write_mode_requires_confirmation(self):
        """Even with write mode on, DML/DDL requires confirmation."""
        ac = AccessController(write_mode_enabled=True)
        with pytest.raises(WriteConfirmationRequired):
            ac.check_query_allowed("DELETE FROM users WHERE id = 1")

    def test_write_mode_ddl_requires_confirmation(self):
        ac = AccessController(write_mode_enabled=True)
        with pytest.raises(WriteConfirmationRequired):
            ac.check_query_allowed("DROP TABLE users")

    def test_write_mode_select_no_confirmation(self):
        """SELECT should not require confirmation even in write mode."""
        ac = AccessController(write_mode_enabled=True)
        allowed, reason = ac.check_query_allowed("SELECT * FROM users")
        assert allowed is True

    def test_confirm_write_allowed(self):
        ac = AccessController(write_mode_enabled=True)
        allowed, reason = ac.confirm_write("DELETE FROM users WHERE id = 1")
        assert allowed is True

    def test_confirm_write_denied_without_write_mode(self):
        ac = AccessController(write_mode_enabled=False)
        allowed, reason = ac.confirm_write("DELETE FROM users WHERE id = 1")
        assert allowed is False
