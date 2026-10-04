"""
Display system -- Qwen Code-style terminal UI.

Matches Qwen Code's layout from actual source:
- Header.tsx: bordered info panel, gradient logo, tildeifyPath
- Help.tsx: bordered panel, categorized commands, shortcuts
- AuthDialog.tsx: bordered dialog, view stack, step breadcrumbs
- theme.ts: semantic color tokens (dark theme)
- Footer.tsx: status line
"""

import os
import sys
import time
import random
import threading
import itertools
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich import box

console = Console()

VERSION = "0.1.0"

# -- Semantic color tokens (from Qwen Code themes/theme.ts darkTheme) ------
# Every color below is traced to the exact field in darkTheme (line 112-130)
# and the SemanticColors mapping in Theme constructor (line 176-208).
THEME = {
    # text.*
    "text_primary": "white",           # Foreground (terminal default)
    "text_secondary": "#6C7086",       # Gray
    "text_link": "#89B4FA",            # AccentBlue
    "text_accent": "#CBA6F7",          # AccentPurple
    "text_code": "#ADD8E6",            # LightBlue
    # background.*
    "bg_primary": "#1E1E2E",           # Background
    # border.*
    "border_default": "#6C7086",       # Gray
    "border_focused": "#89B4FA",       # AccentBlue
    # ui.*
    "ui_comment": "#6C7086",           # Gray
    "ui_symbol": "#89DCEB",            # AccentCyan
    # status.*
    "status_error": "#F38BA8",         # AccentRed
    "status_success": "#A6E3A1",       # AccentGreen
    "status_warning": "#F9E2AF",       # AccentYellow
    "status_error_dim": "#8B3A4A",     # AccentRedDim
    "status_warning_dim": "#8B7530",   # AccentYellowDim
    # gradient (from GradientColors, line 129)
    "gradient": ["#4796E4", "#847ACE", "#C3677F"],
}

T = THEME  # shorthand


def _gradient_text(text, colors=None):
    """Apply a horizontal gradient across characters (like ink-gradient)."""
    colors = colors or T["gradient"]
    result = Text()
    if not text:
        return result
    for i, char in enumerate(text):
        color_idx = int(i / len(text) * len(colors))
        color_idx = min(color_idx, len(colors) - 1)
        result.append(char, style=colors[color_idx])
    return result


def _tildeify_path(path_str):
    """Replace home directory with ~ (like Qwen's tildeifyPath)."""
    home = os.path.expanduser("~")
    if path_str.startswith(home):
        return "~" + path_str[len(home):]
    return path_str


def _shorten_path(path_str, max_length):
    """Shorten path to fit max_length (like Qwen's shortenPath)."""
    if len(path_str) <= max_length:
        return path_str
    parts = path_str.replace("\\", "/").split("/")
    if len(parts) <= 2:
        return path_str[:max_length]
    # Keep first and last, shorten middle
    shortened = parts[0] + "/.../" + parts[-1]
    if len(shortened) <= max_length:
        return shortened
    return path_str[:max_length - 1] + "..."


# -- ASCII Art Banner (like Qwen's AsciiArt.ts shortAsciiLogo) -------------
_BANNER_LINES = [
    " ____  ____    _                    _   ",
    "|  _ \\| __ )  / \\   __ _  ___ _ __ | |_ ",
    "| | | |  _ \\ / _ \\ / _` |/ _ \\ '_ \\| __|",
    "| |_| | |_) / ___ \\ (_| |  __/ | | | |_ ",
    "|____/|____/_/   \\_\\__, |\\___|_| |_|\\__|",
    "                   |___/                 ",
]


def show_banner(model=None, db_path=None, cwd=None):
    r"""
    Qwen-style Header (from Header.tsx lines 178-237):
    ASCII art on left, bordered info panel on right.

    Layout:
      [gradient logo]  [gap]  +----------------------------+
                               | >_ DBAgent (v0.1.0)       |
                               |                           |
                               | API Key | model-name      |
                               | ~/Documents/agent         |
                               +----------------------------+
    """
    console.print()

    # -- Build info panel content (from Header.tsx lines 213-235) --
    auth_type = "API Key" if model else "No Provider"
    model_text = model or "Not configured"
    working_dir = _tildeify_path(cwd or os.getcwd())

    # Title line: ">_ DBAgent" in accent + version in secondary
    title_line = Text()
    title_line.append(">_ DBAgent", style=f"bold {T['text_accent']}")
    title_line.append(f" (v{VERSION})", style=T["text_secondary"])

    # Blank spacer (like Header.tsx line 225)
    spacer_line = Text(" ")

    # Auth + model line (like Header.tsx line 228-233)
    auth_model_line = Text()
    auth_model_text = f"{auth_type} | {model_text}"
    auth_model_line.append(auth_model_text, style=T["text_secondary"])

    # Directory line (like Header.tsx line 235)
    dir_line = Text()
    dir_line.append(working_dir, style=T["text_secondary"])

    # If db_path, show it in auth line
    if db_path:
        db_name = Path(db_path).name
        auth_model_text = f"{auth_type} | {model_text} | {db_name}"
        auth_model_line = Text()
        auth_model_line.append(auth_model_text, style=T["text_secondary"])

    # Build the bordered info panel (from Header.tsx borderStyle="single")
    panel_content = Text()
    panel_content.append_text(title_line)
    panel_content.append("\n")
    panel_content.append_text(spacer_line)
    panel_content.append("\n")
    panel_content.append_text(auth_model_line)
    panel_content.append("\n")
    panel_content.append_text(dir_line)

    info_panel = Panel(
        panel_content,
        border_style=T["border_default"],
        box=box.ROUNDED,
        padding=(0, 1),
        expand=False,
        width=min(50, (console.width or 80) - 50),
    )

    # -- Compute layout (from Header.tsx lines 107-136) --
    art_width = max(len(line) for line in _BANNER_LINES)
    logo_gap = 2          # Header.tsx line 100
    container_margin = 2  # Header.tsx line 99
    min_info_panel_width = 40  # Header.tsx line 104-105
    available_width = (console.width or 80) - container_margin * 2
    show_logo = (
        available_width >= art_width + logo_gap + min_info_panel_width
    )

    if show_logo:
        # Side-by-side: gradient art + gap + bordered panel
        # Render art lines
        art_text = Text()
        for i, art_line in enumerate(_BANNER_LINES):
            art_text.append_text(_gradient_text(art_line))
            if i < len(_BANNER_LINES) - 1:
                art_text.append("\n")

        # Use Columns for side-by-side
        console.print(
            Columns(
                [art_text, info_panel],
                padding=(0, logo_gap),
                expand=False,
            )
        )
    else:
        # Stacked layout for narrow terminals
        for art_line in _BANNER_LINES:
            console.print(_gradient_text(art_line))
        console.print()
        console.print(info_panel)

    console.print()


def show_status(db_path=None, model=None, dialect="sqlite", session_id=None):
    """Show connection status in a clean, compact format."""
    console.print()
    items = [
        ("Database", db_path or "Not connected", T["text_link"] if db_path else T["status_error"]),
        ("Model", model or "Not configured", T["text_primary"] if model else T["status_error"]),
        ("Dialect", dialect, T["text_secondary"]),
    ]
    if session_id:
        items.append(("Session", session_id[:12], T["text_secondary"]))

    for label, value, color in items:
        console.print(f"  [{T['text_secondary']}]{label:<12}[/] [{color}]{value}[/]")
    console.print()


def show_query_result(question, sql=None, rows=None, row_count=0,
                      confidence_level="", confidence_score=0.0,
                      validation_findings=None, assumptions=None,
                      attempts=1, execution_time_ms=0.0, recalled_facts=None):
    """Display query results with clean formatting."""
    rows = rows or []
    console.print()

    if sql:
        syntax = Syntax(sql, "sql", theme="monokai", line_numbers=False, word_wrap=True)
        console.print(Panel(
            syntax,
            title="[bold]SQL[/]",
            border_style=T["text_accent"],
            padding=(0, 1),
            expand=False,
        ))

    if rows:
        _show_result_table(rows, row_count)
    elif row_count == 0 and sql:
        console.print(f"  [{T['text_secondary']}]Query returned 0 rows[/]")

    meta_parts = []
    if confidence_level:
        color = T["status_success"] if "high" in confidence_level.lower() else T["status_warning"]
        meta_parts.append(f"[{color}]{confidence_level}[/]")
    if attempts > 1:
        meta_parts.append(f"[{T['ui_symbol']}]{attempts} attempts[/]")
    if execution_time_ms > 0:
        meta_parts.append(f"[{T['text_secondary']}]{execution_time_ms:.0f}ms[/]")
    if meta_parts:
        dot = f"  [{T['text_secondary']}].[/]  "
        console.print("  " + dot.join(meta_parts))

    if validation_findings:
        console.print()
        for f in validation_findings:
            console.print(f"  [{T['status_warning']}]! {f}[/]")

    if assumptions:
        console.print()
        for a in assumptions:
            console.print(f"  [{T['text_secondary']}]assumption:[/] {a}")

    console.print()


def _show_result_table(rows, total_count, max_display=20):
    """Render results as a clean table."""
    if not rows:
        return
    columns = list(rows[0].keys())
    table = Table(box=box.SIMPLE_HEAVY, show_edge=False, padding=(0, 1))
    for col in columns:
        table.add_column(col, style="white", header_style=f"bold {T['text_accent']}")
    for row in rows[:max_display]:
        table.add_row(*[_format_cell(row.get(col)) for col in columns])
    if total_count > max_display:
        table.add_row(*[f"[{T['text_secondary']}]...[/]" for _ in columns])
    console.print(table)
    if total_count > max_display:
        console.print(f"  [{T['text_secondary']}]Showing {max_display} of {total_count} rows[/]")
    else:
        console.print(f"  [{T['text_secondary']}]{total_count} row(s)[/]")


def _format_cell(value):
    if value is None:
        return f"[{T['text_secondary']}]NULL[/]"
    if isinstance(value, float):
        return f"{value:,.2f}" if value != int(value) else str(int(value))
    return str(value)


def show_tables(tables):
    """Show database tables in a clean list."""
    console.print()
    table = Table(box=box.SIMPLE_HEAVY, show_edge=False, padding=(0, 1))
    table.add_column("#", style=T["text_secondary"], width=4)
    table.add_column("Table", style=f"bold {T['text_primary']}")
    table.add_column("Columns", justify="right", style=T["text_secondary"])
    table.add_column("Rows", justify="right", style=T["text_accent"])
    for i, t in enumerate(tables, 1):
        table.add_row(str(i), t.get("name", ""), str(t.get("columns", "")), str(t.get("rows", "")))
    console.print(table)
    console.print()


def show_schema(table_name, columns, foreign_keys=None):
    """Show table schema details."""
    console.print()
    table = Table(
        title=f"[bold {T['text_primary']}]{table_name}[/]",
        box=box.SIMPLE_HEAVY, show_edge=False, padding=(0, 1),
    )
    table.add_column("Column", style=f"bold {T['text_primary']}")
    table.add_column("Type", style=T["text_accent"])
    table.add_column("PK", justify="center", width=4, style=T["status_warning"])
    table.add_column("FK", justify="center", width=4, style=T["ui_symbol"])
    table.add_column("Nullable", justify="center", width=8, style=T["text_secondary"])
    for col in columns:
        pk = "*" if col.get("is_pk") else ""
        fk = ">" if col.get("is_fk") else ""
        nullable = "yes" if col.get("nullable", True) else "no"
        table.add_row(col["name"], col.get("type", ""), pk, fk, nullable)
    console.print(table)
    if foreign_keys:
        console.print()
        for fk in foreign_keys:
            console.print(
                f"  [{T['text_secondary']}]->[/] {fk.get('from_col', '')} "
                f"[{T['text_secondary']}]->[/] {fk.get('to_table', '')}.{fk.get('to_col', '')}"
            )
    console.print()


def show_error(message, suggestion=None):
    """Show error with X prefix (like Qwen Code status.error)."""
    console.print(f"\n  [{T['status_error']}]X {message}[/]")
    if suggestion:
        console.print(f"  [{T['text_secondary']}]{suggestion}[/]")
    console.print()


def show_warning(message):
    console.print(f"  [{T['status_warning']}]{message}[/]")


def show_success(message):
    """Show success message (like AuthDialog success notification)."""
    console.print(f"  [{T['status_success']}]{message}[/]")


def show_info(message):
    console.print(f"  [{T['text_secondary']}]{message}[/]")


def show_help(commands):
    """
    Qwen-style Help (from Help.tsx lines 84-131):
    Bordered panel with category headers, signature in accent, description below.

    Layout:
      +---------------------------------------------+
      |  DBAgent Help                               |
      |                                             |
      |  Provider & Model (3)                       |
      |   /provider [list]                          |
      |      Interactive LLM provider setup wizard  |
      |   /auth                                     |
      |      Alias for /provider                    |
      |  ...                                        |
      |                                             |
      |  Type /help <command> for details            |
      +---------------------------------------------+
    """
    console.print()

    help_lines = Text()

    # Title (like HelpTabs in Help.tsx line 133-137)
    help_lines.append("DBAgent", style=f"bold {T['text_accent']}")
    help_lines.append(" Help\n\n", style=T["text_secondary"])

    for category, cmds in commands.items():
        # Group header: "Category (count)" (Help.tsx line 350-355)
        help_lines.append(f"{category}", style=f"bold {T['text_primary']}")
        help_lines.append(f" ({len(cmds)})\n", style=T["text_secondary"])

        for cmd, description in cmds:
            # Command signature in accent (Help.tsx line 358-361)
            help_lines.append(f"  {cmd}\n", style=T["text_accent"])
            # Description indented below (Help.tsx line 364-369)
            help_lines.append(f"    {description}\n", style=T["text_primary"])

        help_lines.append("\n")

    # Footer hint (Help.tsx line 118-121)
    help_lines.append("Type ", style=T["text_secondary"])
    help_lines.append("/help <command>", style=T["text_accent"])
    help_lines.append(" for details on a specific command", style=T["text_secondary"])

    # Bordered panel (Help.tsx borderStyle="single", borderColor=theme.border.default)
    panel = Panel(
        help_lines,
        border_style=T["border_default"],
        box=box.ROUNDED,
        padding=(1, 2),
        expand=False,
        width=min(70, console.width or 80),
    )
    console.print(panel)
    console.print()


def show_history(entries):
    """Show query history."""
    if not entries:
        console.print(f"  [{T['text_secondary']}]No queries in this session yet.[/]\n")
        return
    console.print()
    for i, entry in enumerate(entries, 1):
        status = f"[{T['status_success']}]+[/]" if entry.get("success") else f"[{T['status_error']}]x[/]"
        q = entry.get("question", "")[:60]
        rows = entry.get("row_count", "")
        console.print(f"  {status} [{T['text_secondary']}]{i:>2}.[/] {q}  [{T['text_secondary']}]({rows} rows)[/]")
    console.print()


# -- Animated Spinner (from LoadingIndicator.tsx + RespondingSpinner.tsx) ---
# Qwen Code uses ink-spinner with 'dots' type. These are the braille dot
# spinner frames (matching cli-spinners 'dots').
_SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
# Windows fallback (no braille support in some terminals)
_SPINNER_FRAMES_ASCII = ["/", "-", "\\", "|"]


class Spinner:
    """
    Animated spinner with elapsed timer (from LoadingIndicator.tsx).

    Shows: ⠋ Generating SQL... (3s · esc to cancel)

    Usage:
        with Spinner("Generating SQL..."):
            result = slow_function()
    """

    def __init__(self, message="Thinking...", show_cancel_hint=True):
        self.message = message
        self.show_cancel_hint = show_cancel_hint
        self._stop_event = threading.Event()
        self._thread = None
        self._start_time = 0
        # Use ASCII frames on Windows if terminal doesn't support unicode
        self._frames = _SPINNER_FRAMES if os.name != 'nt' else _SPINNER_FRAMES_ASCII

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def start(self):
        self._start_time = time.time()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1)
        # Clear the spinner line
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()

    def update_message(self, message):
        """Update the spinner message mid-flight."""
        self.message = message

    def _animate(self):
        frames = itertools.cycle(self._frames)
        while not self._stop_event.is_set():
            frame = next(frames)
            elapsed = int(time.time() - self._start_time)
            time_str = f"{elapsed}s" if elapsed < 60 else f"{elapsed // 60}m{elapsed % 60}s"

            # Format like LoadingIndicator.tsx line 113:
            # "(3s · esc to cancel)"
            cancel_hint = " · esc to cancel" if self.show_cancel_hint else ""
            status = f"({time_str}{cancel_hint})"

            line = f"\r  {frame} {self.message} {status}"
            sys.stdout.write(line)
            sys.stdout.flush()
            self._stop_event.wait(0.08)  # ~12fps like cli-spinners dots


# -- Tips System (from Tips.tsx) -------------------------------------------
# Tips are shown once at startup, randomly selected.
# From Tips.tsx line 39: fallback is "Type / to see all available commands."
_TIPS = [
    "Type / to see all available commands.",
    "Use /connect <path> to connect to a database.",
    "Use /explain to understand the last generated SQL.",
    "Use /session save <name> to save your work.",
    "Use /doctor to check your setup.",
    "Use /export csv to export query results.",
    "Ask follow-up questions -- DBAgent remembers context.",
    "Use /model to switch between LLM models.",
    "Use /sample <table> to preview data.",
    "Press Up/Down arrows to cycle through prompt history.",
]


def show_tips():
    """Show a random startup tip (from Tips.tsx)."""
    tip = random.choice(_TIPS)
    console.print(f"  [{T['text_secondary']}]Tips: {tip}[/]")


# -- Footer Status Line (from Footer.tsx lines 212-260) --------------------
def show_footer_hint():
    """Show the idle footer hint: '? for shortcuts' (from Footer.tsx line 157)."""
    console.print(f"  [{T['text_secondary']}]? for shortcuts[/]")


def show_footer_status(model=None, db_path=None, table_count=0):
    """
    Show persistent status info below prompt (from Footer.tsx right items).
    Format: model | db_name (N tables)
    """
    parts = []
    if model:
        parts.append(model)
    if db_path:
        db_name = Path(db_path).name
        if table_count:
            parts.append(f"{db_name} ({table_count} tables)")
        else:
            parts.append(db_name)
    if parts:
        status = " | ".join(parts)
        console.print(f"  [{T['text_secondary']}]{status}[/]")
