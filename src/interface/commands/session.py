"""
Session commands — /session list, /session load, /session save, /session new, /session info.
Allows users to switch sessions and seamlessly continue previous work.
"""

from rich.table import Table
from rich.panel import Panel
from rich import box

from src.interface import display
from src.interface.display import THEME as T, console
from src.interface.session_manager import SessionManager


def cmd_session(args, context):
    """Router for /session subcommands."""
    subcmd = args[0].lower() if args else "list"
    subargs = args[1:] if len(args) > 1 else []

    if subcmd in ("list", "ls"):
        _list_sessions(context)
    elif subcmd in ("new", "create"):
        _new_session(subargs, context)
    elif subcmd in ("save",):
        _save_session(subargs, context)
    elif subcmd in ("load", "resume", "open"):
        _load_session(subargs, context)
    elif subcmd in ("info", "current"):
        _show_session_info(context)
    elif subcmd in ("delete", "rm"):
        _delete_session(subargs, context)
    else:
        # If user typed `/session <session_name_or_id>` directly, treat as load
        _load_session([subcmd] + subargs, context)


def _list_sessions(context):
    manager = context.get("session_manager") or SessionManager()
    sessions = manager.list_sessions()
    current_id = context.get("session_id")

    if not sessions:
        console.print(f"  [{T['text_secondary']}]No saved sessions found. Use /session save [name] to save current session.[/]\n")
        return

    table = Table(title=f"[{T['text_accent']}]Saved Sessions[/]", box=box.ROUNDED, border_style=T["border_default"])
    table.add_column("Status", width=3, justify="center")
    table.add_column("ID / Name", style=f"bold {T['text_primary']}")
    table.add_column("Last Active", style=T["text_secondary"])
    table.add_column("Database", style=T["ui_symbol"])
    table.add_column("Queries", justify="right", style=T["status_success"])
    table.add_column("Last Question", style=T["text_secondary"], max_width=40)

    for s in sessions:
        sid = s["session_id"]
        is_current = (sid == current_id or s.get("name") == context.get("session_name"))
        indicator = f"[{T['status_success']}]*[/]" if is_current else " "
        display_name = s.get("name") or sid
        if display_name != sid:
            display_title = f"{display_name}\n[{T['text_secondary']}]({sid})[/]"
        else:
            display_title = sid

        db_str = s.get("db_path") or f"[{T['text_secondary']}]None[/]"
        if db_str != f"[{T['text_secondary']}]None[/]":
            from pathlib import Path
            db_str = Path(db_str).name

        updated = s.get("updated_at", "")
        if "T" in updated:
            updated = updated.split("T")[0] + " " + updated.split("T")[1][:5]

        table.add_row(
            indicator,
            display_title,
            updated,
            db_str,
            str(s.get("query_count", 0)),
            s.get("last_question", "")[:40] or f"[{T['text_secondary']}]-[/]",
        )

    console.print(table)
    console.print(
        f"  [{T['text_secondary']}]Commands: [{T['text_accent']}]/session load <id>[/] to continue, "
        f"[{T['text_accent']}]/session new [name][/] for fresh session, "
        f"[{T['text_accent']}]/session save [name][/] to save.[/]\n"
    )


def _new_session(args, context):
    manager = context.get("session_manager") or SessionManager()
    # Save current if has history
    current_id = context.get("session_id")
    if current_id and context.get("history"):
        manager.save_session(
            session_id=current_id,
            context=context,
            conversation_state=context.get("conversation_state"),
        )

    new_id = manager.generate_session_id()
    name = args[0] if args else None

    context["session_id"] = new_id
    context["session_name"] = name or new_id
    context["history"] = []
    context["last_result"] = None

    conv_state = context.get("conversation_state")
    if conv_state is not None:
        conv_state._history.clear()
        conv_state._session_filters.clear()

    manager.save_session(
        session_id=new_id,
        context=context,
        name=name,
        conversation_state=conv_state,
    )

    display.show_success(f"Started new session: {name or new_id}")
    console.print(f"  [{T['text_secondary']}]Session ID: {new_id}[/]\n")


def _save_session(args, context):
    manager = context.get("session_manager") or SessionManager()
    session_id = context.get("session_id") or manager.generate_session_id()
    context["session_id"] = session_id
    name = args[0] if args else context.get("session_name")

    path = manager.save_session(
        session_id=session_id,
        context=context,
        name=name,
        conversation_state=context.get("conversation_state"),
    )
    display.show_success(f"Session saved: {name or session_id}")
    console.print(f"  [{T['text_secondary']}]Saved to {path}[/]\n")


def _load_session(args, context):
    if not args:
        display.show_error("Usage: /session load <id_or_name>", "Example: /session load 20260909_120000")
        return

    identifier = args[0]
    manager = context.get("session_manager") or SessionManager()
    session_data = manager.load_session(identifier)

    if not session_data:
        display.show_error(f"Session not found: {identifier}", "Use /session list to see saved sessions.")
        return

    # Auto-save current session first if it has unsaved queries
    current_id = context.get("session_id")
    if current_id and current_id != session_data["session_id"] and context.get("history"):
        manager.save_session(
            session_id=current_id,
            context=context,
            conversation_state=context.get("conversation_state"),
        )

    manager.restore_into_context(
        session_data=session_data,
        context=context,
        conversation_state=context.get("conversation_state"),
    )

    # Reconnect to DB if session has database configured
    db_path = session_data.get("db_path")
    db_reconnected = False
    if db_path:
        from src.interface.commands.database import cmd_connect
        try:
            cmd_connect([db_path], context)
            db_reconnected = context.get("db_connected", False)
        except Exception:
            db_reconnected = False

    name = session_data.get("name") or session_data["session_id"]
    query_count = len(session_data.get("history", []))

    display.show_success(f"Resumed session: {name}")
    info_parts = [f"[{T['ui_symbol']}]{query_count} prior queries[/]"]
    if db_reconnected:
        info_parts.append(f"[{T['status_success']}]Connected to {db_path}[/]")
    elif db_path:
        info_parts.append(f"[{T['status_warning']}]Database {db_path} could not be auto-reconnected[/]")

    console.print(f"  {' | '.join(info_parts)}\n")


def _show_session_info(context):
    session_id = context.get("session_id", "None")
    session_name = context.get("session_name") or session_id
    db_path = context.get("db_path") or f"[{T['text_secondary']}]Not connected[/]"
    model = context.get("model_name") or f"[{T['text_secondary']}]Not set[/]"
    history_count = len(context.get("history", []))

    conv_state = context.get("conversation_state")
    memory_turns = conv_state.turn_count if conv_state else 0

    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    table.add_column("Key", style=T["text_secondary"])
    table.add_column("Value")
    table.add_row("Session Name", session_name)
    table.add_row("Session ID", session_id)
    table.add_row("Database", db_path)
    table.add_row("Model", model)
    table.add_row("Queries Run", str(history_count))
    table.add_row("Conversation Memory Turns", str(memory_turns))

    console.print(Panel(table, title=f"[{T['text_accent']}]Current Session Info[/]", border_style=T["border_default"]))


def _delete_session(args, context):
    if not args:
        display.show_error("Usage: /session delete <id_or_name>")
        return
    identifier = args[0]
    manager = context.get("session_manager") or SessionManager()
    if manager.delete_session(identifier):
        display.show_success(f"Deleted session: {identifier}")
    else:
        display.show_error(f"Could not find or delete session: {identifier}")


def register(registry):
    registry.register(
        "session", cmd_session, "Manage sessions (list, load, save, new, info)",
        "/session [list|load|save|new|info|delete]", "Session",
    )
    registry.register(
        "sessions", lambda args, ctx: _list_sessions(ctx), "List all saved sessions",
        "/sessions", "Session",
    )
    registry.register(
        "resume", lambda args, ctx: _load_session(args, ctx), "Resume a previous session",
        "/resume <id_or_name>", "Session",
    )
