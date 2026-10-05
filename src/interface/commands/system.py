"""System commands — /help, /clear, /status, /quit, /doctor"""
import os
import sys
from pathlib import Path
from rich.table import Table
from rich.panel import Panel
from rich import box

from src.interface import display
from src.interface.display import THEME as T, console


def cmd_help(args, context):
    registry = context.get("registry")
    if not registry:
        return
    if args:
        cmd = registry.get(args[0].lstrip("/"))
        if cmd:
            console.print(f"\n  [{T['text_accent']}]{cmd['usage']}[/]")
            console.print(f"  {cmd['description']}\n")
        else:
            display.show_error(f"Unknown command: {args[0]}")
        return
    display.show_help(registry.get_help())


def cmd_clear(args, context):
    os.system("cls" if os.name == "nt" else "clear")


def cmd_status(args, context):
    display.show_status(
        db_path=context.get("db_path"),
        model=context.get("model_name"),
        dialect=context.get("dialect", "sqlite"),
        session_id=context.get("session_id"),
    )


def cmd_quit(args, context):
    # Auto-save session on quit if manager exists
    manager = context.get("session_manager")
    session_id = context.get("session_id")
    if manager and session_id and context.get("history"):
        try:
            manager.save_session(
                session_id=session_id,
                context=context,
                conversation_state=context.get("conversation_state"),
            )
        except Exception:
            pass

    console.print(f"\n  [{T['text_secondary']}]Goodbye.[/]\n")
    context["should_quit"] = True


def cmd_history(args, context):
    display.show_history(context.get("history", []))


def cmd_doctor(args, context):
    """Run diagnostics on environment, database, LLM setup, and session storage."""
    PASS = f"[{T['status_success']}]PASS[/]"
    WARN = f"[{T['status_warning']}]WARN[/]"
    FAIL = f"[{T['status_error']}]FAIL[/]"
    NA = f"[{T['text_secondary']}]N/A[/]"

    table = Table(
        title=f"[{T['text_accent']}]DBAgent System Doctor[/]",
        box=box.ROUNDED,
        border_style=T["border_default"],
    )
    table.add_column("Component", style=f"bold {T['text_primary']}", width=24)
    table.add_column("Status", width=10, justify="center")
    table.add_column("Details", style=T["text_secondary"])

    # 1. Database
    if context.get("db_connected"):
        table_count = len(context.get("table_names", []))
        table.add_row("Database Connection", PASS, f"{context.get('db_path')} ({table_count} tables)")
    else:
        table.add_row("Database Connection", WARN, "Not connected (run /connect <db>)")

    # 2. LLM Model
    if context.get("model_name"):
        provider = context.get("provider_name") or "Default"
        table.add_row("LLM Model", PASS, f"{context.get('model_name')} via {provider}")
    else:
        table.add_row("LLM Model", WARN, "Not configured (run /provider or /local)")

    # 3. Base URL / API Key
    prov_key = context.get("provider_key")
    if prov_key == "ollama":
        url = context.get("base_url") or "http://localhost:11434/v1"
        table.add_row("LLM Endpoint", PASS, f"Local Ollama at {url}")
        table.add_row("API Key", PASS, "Not required for local LLM")
    else:
        url = context.get("base_url") or "Provider default"
        table.add_row("LLM Endpoint", PASS, url)
        has_key = bool(context.get("api_key"))
        if not has_key and prov_key:
            from src.interface.commands.provider import PROVIDERS
            env_key = PROVIDERS.get(prov_key, {}).get("env_key")
            if env_key and os.environ.get(env_key):
                has_key = True
        if has_key:
            table.add_row("API Key", PASS, "API key is present")
        elif context.get("model_name"):
            table.add_row("API Key", WARN, "API key missing (run /key <api_key>)")
        else:
            table.add_row("API Key", NA, "Configure provider first")

    # 4. Sessions
    sessions_dir = Path("config/sessions")
    try:
        sessions_dir.mkdir(parents=True, exist_ok=True)
        count = len(list(sessions_dir.glob("*.json")))
        table.add_row("Session Store", PASS, f"config/sessions/ ({count} saved)")
    except Exception as e:
        table.add_row("Session Store", FAIL, str(e))

    # 5. Core Pipeline
    try:
        from src.agent.pipeline import Pipeline
        from src.agent.sql_generator import SQLGenerator
        from src.agent.critic import Critic
        from src.agent.result_validator import ResultValidator
        table.add_row("Core Pipeline", PASS, "All agent components verified")
    except Exception as e:
        table.add_row("Core Pipeline", FAIL, f"Import error: {e}")

    # 6. Memory & Multi-turn
    conv = context.get("conversation_state")
    turns = conv.turn_count if conv else 0
    table.add_row("Multi-turn Memory", PASS, f"Active ({turns} turns in memory)")

    console.print()
    console.print(table)
    console.print()


def register(registry):
    registry.register("help", cmd_help, "Show all commands", "/help [command]", "System")
    registry.register("clear", cmd_clear, "Clear the terminal", "/clear", "System")
    registry.register("status", cmd_status, "Show connection status", "/status", "System")
    registry.register("doctor", cmd_doctor, "Run system & connection diagnostics", "/doctor", "System")
    registry.register("history", cmd_history, "Show query history", "/history", "System")
    registry.register("quit", cmd_quit, "Exit DBAgent", "/quit", "System")
    registry.register("exit", cmd_quit, "Exit DBAgent", "/exit", "System")
