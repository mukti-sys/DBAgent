"""
Unit tests for terminal interface commands (system, database, provider, model, session, export, query).
"""

import json
import sqlite3
from pathlib import Path
import pytest

from src.interface.commands import CommandRegistry
from src.interface.commands import system as system_cmds
from src.interface.commands import database as database_cmds
from src.interface.commands import provider as provider_cmds
from src.interface.commands import model as model_cmds
from src.interface.commands import session as session_cmds
from src.interface.commands import export as export_cmds
from src.interface.commands import query as query_cmds

from src.interface.session_manager import SessionManager
from src.agent.conversation_state import ConversationState
from src.agent.pipeline import PipelineResult


@pytest.fixture
def sqlite_test_db(tmp_path):
    db_file = tmp_path / "sample.db"
    conn = sqlite3.connect(db_file)
    conn.execute("CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price REAL)")
    conn.execute("INSERT INTO products VALUES (1, 'Widget', 19.99)")
    conn.execute("INSERT INTO products VALUES (2, 'Gadget', 29.99)")
    conn.commit()
    conn.close()
    return str(db_file)


@pytest.fixture
def base_context(tmp_path):
    manager = SessionManager(sessions_dir=tmp_path / "sessions")
    conv_state = ConversationState()
    return {
        "db_path": None,
        "db_connected": False,
        "dialect": "sqlite",
        "model_name": None,
        "provider_name": None,
        "provider_key": None,
        "api_key": None,
        "base_url": None,
        "session_id": "test_sess_001",
        "session_name": "test_sess_001",
        "should_quit": False,
        "history": [],
        "last_result": None,
        "table_info": {},
        "table_names": [],
        "known_columns": {},
        "session_manager": manager,
        "conversation_state": conv_state,
    }


def test_command_registry():
    reg = CommandRegistry()
    ran = []

    def sample_handler(args, ctx):
        ran.append((args, ctx))

    reg.register("testcmd", sample_handler, "Test command description", "/testcmd", "Custom")

    assert reg.get("testcmd") is not None
    assert "/testcmd" in reg.get_command_names()
    assert reg.execute("testcmd", ["arg1", "arg2"], {"key": "val"}) is True
    assert ran == [(["arg1", "arg2"], {"key": "val"})]
    assert reg.execute("unknown", [], {}) is False


def test_database_connect_and_sample(sqlite_test_db, base_context):
    database_cmds.cmd_connect([sqlite_test_db], base_context)
    assert base_context["db_connected"] is True
    assert "products" in base_context["table_names"]
    assert "name" in base_context["known_columns"]["products"]

    # Test /sample command
    database_cmds.cmd_sample(["products", "2"], base_context)

    # Test /tables and /schema
    database_cmds.cmd_tables([], base_context)
    database_cmds.cmd_schema(["products"], base_context)

    # Test /disconnect
    database_cmds.cmd_disconnect([], base_context)
    assert base_context["db_connected"] is False


def test_provider_baseurl_key_local(base_context, tmp_path, monkeypatch):
    monkeypatch.setattr("src.interface.commands.provider.Path", lambda p: tmp_path / p if isinstance(p, str) and p.startswith("config") else Path(p))

    # /baseurl command
    provider_cmds.cmd_baseurl(["http://localhost:11434/v1"], base_context)
    assert base_context["base_url"] == "http://localhost:11434/v1"

    # /key command
    provider_cmds.cmd_key(["test-api-key-1234"], base_context)
    assert base_context["api_key"] == "test-api-key-1234"

    # /local command
    provider_cmds.cmd_local(["qwen2.5-coder"], base_context)
    assert base_context["provider_key"] == "ollama"
    assert base_context["model_name"] == "qwen2.5-coder"
    assert base_context["base_url"] == "http://localhost:11434"


def test_model_command(base_context):
    model_cmds.cmd_model(["gpt-4o-mini"], base_context)
    assert base_context["model_name"] == "gpt-4o-mini"


def test_session_lifecycle_commands(sqlite_test_db, base_context):
    # Connect DB & add history
    database_cmds.cmd_connect([sqlite_test_db], base_context)
    base_context["history"].append({
        "question": "test question",
        "sql": "SELECT 1",
        "row_count": 1,
        "success": True,
    })

    # Save session
    session_cmds.cmd_session(["save", "TestRun"], base_context)

    # Check list
    session_cmds.cmd_session(["list"], base_context)

    # Create new session
    session_cmds.cmd_session(["new", "BrandNew"], base_context)
    assert base_context["session_name"] == "BrandNew"
    assert len(base_context["history"]) == 0

    # Load back the previous session
    session_cmds.cmd_session(["load", "TestRun"], base_context)
    assert base_context["session_name"] == "TestRun"
    assert len(base_context["history"]) == 1
    assert base_context["db_connected"] is True


def test_export_command(tmp_path, base_context):
    # Prepare mock query result
    res = PipelineResult(
        question="Select all products",
        sql="SELECT * FROM products",
        rows=[{"id": 1, "name": "Widget", "price": 19.99}],
        row_count=1,
        success=True,
    )
    base_context["last_result"] = res

    # Export to CSV
    csv_file = tmp_path / "test_out.csv"
    export_cmds.cmd_export(["csv", str(csv_file)], base_context)
    assert csv_file.exists()
    content = csv_file.read_text(encoding="utf-8")
    assert "Widget" in content

    # Export to JSON
    json_file = tmp_path / "test_out.json"
    export_cmds.cmd_export(["json", str(json_file)], base_context)
    assert json_file.exists()
    data = json.loads(json_file.read_text(encoding="utf-8"))
    assert data[0]["name"] == "Widget"

    # Export to Markdown
    md_file = tmp_path / "test_out.md"
    export_cmds.cmd_export(["md", str(md_file)], base_context)
    assert md_file.exists()
    md_content = md_file.read_text(encoding="utf-8")
    assert "| Widget |" in md_content


def test_query_explain_and_sql(sqlite_test_db, base_context):
    database_cmds.cmd_connect([sqlite_test_db], base_context)

    # Direct raw SQL execution
    query_cmds.cmd_sql(["SELECT", "name", "FROM", "products", "WHERE", "id = 1"], base_context)
    assert base_context["last_result"] is not None
    assert base_context["last_result"].rows[0]["name"] == "Widget"

    # Query explanation
    query_cmds.cmd_explain([], base_context)


def test_system_doctor_and_status(sqlite_test_db, base_context):
    database_cmds.cmd_connect([sqlite_test_db], base_context)
    base_context["model_name"] = "qwen2.5-coder"
    base_context["provider_key"] = "ollama"

    # Doctor check
    system_cmds.cmd_doctor([], base_context)
    # Status check
    system_cmds.cmd_status([], base_context)
    # History check
    system_cmds.cmd_history([], base_context)
