"""
Export commands — /export to save query results to CSV, JSON, or Markdown.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from src.interface import display


def cmd_export(args, context):
    """Export the most recent query results to a file."""
    last_result = context.get("last_result")
    if not last_result or not getattr(last_result, "rows", None):
        display.show_warning("No query results available to export. Run a query first.")
        return

    rows = last_result.rows
    if not rows:
        display.show_warning("Last query returned 0 rows. Nothing to export.")
        return

    fmt = "csv"
    target_path = None

    if args:
        first_arg = args[0].lower()
        if first_arg in ("csv", "json", "md", "markdown"):
            fmt = "md" if first_arg == "markdown" else first_arg
            if len(args) > 1:
                target_path = Path(args[1])
        else:
            # First argument is a filepath
            target_path = Path(args[0])
            ext = target_path.suffix.lower().lstrip(".")
            if ext in ("csv", "json", "md"):
                fmt = ext

    if target_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = Path("exports")
        export_dir.mkdir(parents=True, exist_ok=True)
        target_path = export_dir / f"result_{timestamp}.{fmt}"
    else:
        target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if fmt == "csv":
            _export_csv(rows, target_path)
        elif fmt == "json":
            _export_json(rows, target_path)
        elif fmt == "md":
            _export_markdown(rows, target_path, last_result.sql)
        else:
            display.show_error(f"Unsupported format: {fmt}", "Supported formats: csv, json, md")
            return

        display.show_success(f"Exported {len(rows)} row(s) to [bold]{target_path}[/]")
    except Exception as e:
        display.show_error(f"Failed to export data: {e}")


def _export_csv(rows, path):
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _export_json(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False, default=str)


def _export_markdown(rows, path, sql=None):
    fields = list(rows[0].keys())
    lines = []
    if sql:
        lines.append("```sql")
        lines.append(sql)
        lines.append("```\n")

    lines.append("| " + " | ".join(fields) + " |")
    lines.append("| " + " | ".join(["---"] * len(fields)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(f, "")) for f in fields) + " |")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def register(registry):
    registry.register(
        "export", cmd_export, "Export last query results to file (csv, json, md)",
        "/export [csv|json|md] [filepath]", "Data & Analysis",
    )
