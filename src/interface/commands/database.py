"""Database commands — /connect, /tables, /schema, /dialect"""
import sqlite3
from pathlib import Path
from src.interface import display


def cmd_connect(args, context):
    if not args:
        display.show_error("Usage: /connect <path-to-db>", "Example: /connect chinook.db")
        return
    db_path = args[0]
    if not db_path.startswith("postgres") and not Path(db_path).exists():
        display.show_error(f"File not found: {db_path}")
        return
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = cursor.fetchall()
        # Index schema
        table_info = {}
        table_names = []
        known_columns = {}
        for (table_name,) in tables:
            table_names.append(table_name)
            cursor.execute(f'PRAGMA table_info("{table_name}")')
            cols = cursor.fetchall()
            table_info[table_name] = [
                {"name": c[1], "type": c[2] or "TEXT", "nullable": not c[3], "is_pk": bool(c[5])}
                for c in cols
            ]
            known_columns[table_name] = [c[1] for c in cols]
        conn.close()
        context.update({
            "db_path": db_path, "dialect": "sqlite", "db_connected": True,
            "table_info": table_info, "table_names": table_names,
            "known_columns": known_columns,
        })
        display.show_success(f"Connected to {db_path}")
        display.console.print(f"  [dim]Found {len(tables)} table(s). Type /tables to see them.[/]\n")
    except Exception as e:
        display.show_error(f"Connection failed: {e}")


def cmd_disconnect(args, context):
    if not context.get("db_connected"):
        display.show_warning("Not connected to any database.")
        return
    context.update({"db_path": None, "db_connected": False, "table_info": {}, "table_names": []})
    display.show_success("Disconnected.")


def cmd_tables(args, context):
    if not context.get("db_connected"):
        display.show_error("Not connected.", "Use /connect <path> first.")
        return
    try:
        conn = sqlite3.connect(context["db_path"])
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        table_data = []
        for (name,) in cursor.fetchall():
            cursor.execute(f'SELECT COUNT(*) FROM "{name}"')
            rows = cursor.fetchone()[0]
            cursor.execute(f'PRAGMA table_info("{name}")')
            cols = len(cursor.fetchall())
            table_data.append({"name": name, "columns": cols, "rows": rows})
        conn.close()
        display.show_tables(table_data)
    except Exception as e:
        display.show_error(f"Failed: {e}")


def cmd_schema(args, context):
    if not context.get("db_connected"):
        display.show_error("Not connected.", "Use /connect <path> first.")
        return
    if not args:
        cmd_tables([], context)
        display.console.print("  [dim]Use /schema <table_name> for details.[/]\n")
        return
    table_name = args[0]
    try:
        conn = sqlite3.connect(context["db_path"])
        cursor = conn.cursor()
        cursor.execute(f'PRAGMA table_info("{table_name}")')
        raw_cols = cursor.fetchall()
        if not raw_cols:
            display.show_error(f"Table '{table_name}' not found.")
            conn.close()
            return
        cursor.execute(f'PRAGMA foreign_key_list("{table_name}")')
        raw_fks = cursor.fetchall()
        conn.close()
        fk_cols = {fk[3]: (fk[2], fk[4]) for fk in raw_fks}
        columns = [
            {"name": c[1], "type": c[2] or "TEXT", "nullable": not c[3],
             "is_pk": bool(c[5]), "is_fk": c[1] in fk_cols}
            for c in raw_cols
        ]
        foreign_keys = [
            {"from_col": fc, "to_table": tt, "to_col": tc}
            for fc, (tt, tc) in fk_cols.items()
        ]
        display.show_schema(table_name, columns, foreign_keys)
    except Exception as e:
        display.show_error(f"Failed: {e}")


def cmd_dialect(args, context):
    if not args:
        display.console.print(f"  Current dialect: [bold]{context.get('dialect', 'sqlite')}[/]")
        display.console.print("  [dim]Usage: /dialect sqlite|postgres|mysql[/]\n")
        return
    dialect = args[0].lower()
    if dialect not in ("sqlite", "postgres", "postgresql", "mysql"):
        display.show_error(f"Unknown dialect: {dialect}", "Supported: sqlite, postgres, mysql")
        return
    context["dialect"] = dialect
    display.show_success(f"Dialect set to {dialect}")


def register(registry):
    registry.register("connect", cmd_connect, "Connect to a database", "/connect <path>", "Database")
    registry.register("disconnect", cmd_disconnect, "Disconnect", "/disconnect", "Database")
    registry.register("tables", cmd_tables, "Show all tables", "/tables", "Database")
    registry.register("schema", cmd_schema, "Show table schema", "/schema [table]", "Database")
    registry.register("dialect", cmd_dialect, "Set SQL dialect", "/dialect sqlite|postgres|mysql", "Database")
