"""
Unit tests for SessionManager (saving, listing, loading, restoring multi-turn sessions).
"""

import json
from pathlib import Path
import pytest

from src.interface.session_manager import SessionManager
from src.agent.conversation_state import ConversationState, QueryContext


@pytest.fixture
def temp_session_dir(tmp_path):
    return tmp_path / "sessions"


@pytest.fixture
def manager(temp_session_dir):
    return SessionManager(sessions_dir=temp_session_dir)


def test_save_and_load_session(manager):
    conv_state = ConversationState()
    conv_state.add_turn(
        QueryContext(
            question="total revenue by country?",
            sql="SELECT BillingCountry, SUM(Total) FROM Invoice GROUP BY BillingCountry",
            filters={"BillingCountry": "USA"},
            tables_used=["Invoice"],
            result_summary="10 rows",
        )
    )

    context = {
        "db_path": "test.db",
        "dialect": "sqlite",
        "model_name": "gpt-4o",
        "provider_name": "OpenAI",
        "provider_key": "openai",
        "base_url": "https://api.openai.com/v1",
        "history": [
            {
                "question": "total revenue by country?",
                "sql": "SELECT ...",
                "row_count": 10,
                "success": True,
            }
        ],
    }

    session_id = "test_session_001"
    saved_path = manager.save_session(
        session_id=session_id,
        context=context,
        name="Revenue Study",
        conversation_state=conv_state,
    )

    assert saved_path.exists()
    loaded = manager.load_session(session_id)
    assert loaded is not None
    assert loaded["session_id"] == session_id
    assert loaded["name"] == "Revenue Study"
    assert loaded["db_path"] == "test.db"
    assert loaded["model_name"] == "gpt-4o"
    assert len(loaded["history"]) == 1
    assert len(loaded["conversation_turns"]) == 1
    assert loaded["conversation_turns"][0]["filters"] == {"BillingCountry": "USA"}


def test_restore_into_context_restores_memory(manager):
    conv_state_orig = ConversationState()
    conv_state_orig.add_turn(
        QueryContext(
            question="customers in Germany",
            sql="SELECT * FROM Customer WHERE Country = 'Germany'",
            filters={"Country": "Germany"},
            tables_used=["Customer"],
            result_summary="4 rows",
        )
    )

    context_orig = {
        "db_path": "chinook.db",
        "dialect": "sqlite",
        "model_name": "qwen2.5-coder",
        "provider_key": "ollama",
        "history": [{"question": "customers in Germany", "row_count": 4, "success": True}],
    }

    session_id = "germany_cust"
    manager.save_session(
        session_id=session_id,
        context=context_orig,
        name="Germany Customers",
        conversation_state=conv_state_orig,
    )

    # Now simulate restoring into a fresh new App context
    fresh_context = {}
    fresh_conv_state = ConversationState()

    session_data = manager.load_session("germany_cust")
    manager.restore_into_context(session_data, fresh_context, fresh_conv_state)

    assert fresh_context["session_id"] == "germany_cust"
    assert fresh_context["session_name"] == "Germany Customers"
    assert fresh_context["model_name"] == "qwen2.5-coder"
    assert fresh_conv_state.turn_count == 1
    assert fresh_conv_state.last_context.question == "customers in Germany"
    assert fresh_conv_state.session_filters == {"Country": "Germany"}

    # Verify that follow-up resolution works on the restored state!
    inheritance = fresh_conv_state.resolve_context("now break that down by city")
    assert inheritance.is_follow_up is True
    assert inheritance.inherited_filters["Country"] == "Germany"


def test_list_sessions(manager):
    ctx1 = {"db_path": "sales.db", "history": [{"question": "q1"}]}
    ctx2 = {"db_path": "hr.db", "history": [{"question": "q2"}, {"question": "q3"}]}

    manager.save_session("session_1", ctx1, name="Sales Session")
    manager.save_session("session_2", ctx2, name="HR Session")

    sessions = manager.list_sessions()
    assert len(sessions) == 2
    names = {s["name"] for s in sessions}
    assert "Sales Session" in names
    assert "HR Session" in names


def test_load_session_by_prefix_and_name(manager):
    ctx = {"db_path": "analytics.db", "history": []}
    manager.save_session("20260909_140000", ctx, name="My Analytics")

    # Match by exact ID
    assert manager.load_session("20260909_140000") is not None
    # Match by prefix
    assert manager.load_session("20260909") is not None
    # Match by name (case-insensitive)
    assert manager.load_session("my analytics") is not None
    # Non-existent
    assert manager.load_session("non_existent") is None


def test_delete_session(manager):
    ctx = {"db_path": "test.db", "history": []}
    manager.save_session("to_delete", ctx, name="Temp Session")
    assert manager.load_session("to_delete") is not None

    deleted = manager.delete_session("to_delete")
    assert deleted is True
    assert manager.load_session("to_delete") is None
    assert manager.delete_session("non_existent") is False
