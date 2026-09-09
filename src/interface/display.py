"""
Display system — Rich output formatting for the terminal UI.
"""

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich import box
from typing import Any

console = Console()


def show_banner():
    """Display the welcome banner."""
    banner = Text()
    banner.append("╔══════════════════════════════════════════════════╗\n", style="bold cyan")
    banner.append("║", style="bold cyan")
    banner.append("          🗄️  DBAgent — Database Agent           ", style="bold white")
    banner.append("║\n", style="bold cyan")
    banner.append("║", style="bold cyan")
    banner.append("     Ask questions in plain English. Get SQL.     ", style="dim white")
    banner.append("║\n", style="bold cyan")
    banner.append("╚══════════════════════════════════════════════════╝", style="bold cyan")
    console.print(banner)
    console.print("  Type [bold green]/help[/] for commands, or just ask a question.\n")


def show_status(db_path=None, model=None, dialect="sqlite", session_id=None):
    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    table.add_column("Key", style="dim")
    table.add_column("Value")
    table.add_row("Database", db_path or "[red]Not connected[/]")
    table.add_row("Model", model or "[red]Not configured[/]")
    table.add_row("Dialect", dialect)
    if session_id:
        table.add_row("Session", session_id[:8])
    console.print(Panel(table, title="[bold]Status[/]", border_style="cyan"))


def show_query_result(question, sql=None, rows=None, row_count=0,
                      confidence_level="", confidence_score=0.0,
                      validation_findings=None, assumptions=None,
                      attempts=1, execution_time_ms=0.0, recalled_facts=None):
    rows = rows or []
    if sql:
        syntax = Syntax(sql, "sql", theme="monokai", line_numbers=False, word_wrap=True)
        console.print(Panel(syntax, title="[bold blue]SQL[/]", border_style="blue", padding=(0, 1)))
    if rows:
        _show_result_table(rows, row_count)
    elif row_count == 0 and sql:
        console.print("  [yellow]⚠ Query returned 0 rows[/]\n")

    meta_parts = []
    if confidence_level:
        color = "green" if "high" in confidence_level.lower() else "yellow"
        meta_parts.append(f"[{color}]📊 {confidence_level}[/]")
    if attempts > 1:
        meta_parts.append(f"[cyan]🔄 {attempts} attempts[/]")
    if execution_time_ms > 0:
        meta_parts.append(f"[dim]⏱ {execution_time_ms:.0f}ms[/]")
    if meta_parts:
        console.print("  " + "  │  ".join(meta_parts))

    if validation_findings:
        console.print()
        for f in validation_findings:
            console.print(f"  [yellow]{f}[/]")
    if assumptions:
        console.print()
        for a in assumptions:
            console.print(f"  [cyan]📋 {a}[/]")
    console.print()


def _show_result_table(rows, total_count, max_display=20):
    if not rows:
        return
    columns = list(rows[0].keys())
    table = Table(box=box.ROUNDED)
    for col in columns:
        table.add_column(col, style="white")
    for row in rows[:max_display]:
        table.add_row(*[_format_cell(row.get(col)) for col in columns])
    if total_count > max_display:
        table.add_row(*["..." for _ in columns])
    console.print(table)
    if total_count > max_display:
        console.print(f"  [dim]Showing {max_display} of {total_count} rows.[/]")
    else:
        console.print(f"  [dim]{total_count} row(s)[/]")


def _format_cell(value):
    if value is None:
        return "[dim]NULL[/]"
    if isinstance(value, float):
        return f"{value:,.2f}" if value != int(value) else str(int(value))
    return str(value)


def show_tables(tables):
    table = Table(title="[bold]Database Tables[/]", box=box.ROUNDED)
    table.add_column("#", style="dim", width=4)
    table.add_column("Table", style="bold white")
    table.add_column("Columns", justify="right")
    table.add_column("Rows", justify="right", style="cyan")
    for i, t in enumerate(tables, 1):
        table.add_row(str(i), t.get("name", ""), str(t.get("columns", "")), str(t.get("rows", "")))
    console.print(table)


def show_schema(table_name, columns, foreign_keys=None):
    table = Table(title=f"[bold]{table_name}[/]", box=box.ROUNDED)
    table.add_column("Column", style="bold white")
    table.add_column("Type", style="cyan")
    table.add_column("PK", justify="center", width=4)
    table.add_column("FK", justify="center", width=4)
    table.add_column("Nullable", justify="center", width=8)
    for col in columns:
        pk = "🔑" if col.get("is_pk") else ""
        fk = "🔗" if col.get("is_fk") else ""
        nullable = "✓" if col.get("nullable", True) else "✗"
        table.add_row(col["name"], col.get("type", ""), pk, fk, nullable)
    console.print(table)
    if foreign_keys:
        console.print("\n  [bold]Foreign Keys:[/]")
        for fk in foreign_keys:
            console.print(f"    [dim]→[/] {fk.get('from_col', '')} → {fk.get('to_table', '')}.{fk.get('to_col', '')}")
    console.print()


def show_error(message, suggestion=None):
    console.print(f"\n  [bold red]❌ {message}[/]")
    if suggestion:
        console.print(f"  [dim]{suggestion}[/]")
    console.print()


def show_warning(message):
    console.print(f"  [yellow]⚠ {message}[/]")


def show_success(message):
    console.print(f"  [bold green]✅ {message}[/]")


def show_info(message):
    console.print(f"  [cyan]ℹ {message}[/]")


def show_help(commands):
    console.print()
    for category, cmds in commands.items():
        console.print(f"  [bold cyan]{category}[/]")
        for cmd, description in cmds:
            console.print(f"    [bold green]{cmd:<28}[/] {description}")
        console.print()


def show_history(entries):
    if not entries:
        console.print("  [dim]No queries in this session yet.[/]\n")
        return
    table = Table(title="[bold]Query History[/]", box=box.SIMPLE)
    table.add_column("#", style="dim", width=4)
    table.add_column("Question", style="white")
    table.add_column("Rows", justify="right", style="cyan")
    table.add_column("Status", width=6)
    for i, entry in enumerate(entries, 1):
        status = "[green]✓[/]" if entry.get("success") else "[red]✗[/]"
        table.add_row(str(i), entry.get("question", "")[:60], str(entry.get("row_count", "")), status)
    console.print(table)
