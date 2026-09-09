"""
Query commands — /explain to understand the last query, /sql to run raw SQL directly.
"""

from rich.panel import Panel
from rich.syntax import Syntax
from src.interface import display
from src.agent.executor import QueryExecutor


def cmd_explain(args, context):
    """Explain the last generated SQL query in clear, plain English."""
    last_result = context.get("last_result")
    if not last_result or not getattr(last_result, "sql", None):
        display.show_warning("No recent SQL query to explain. Run a query first.")
        return

    sql = last_result.sql
    question = getattr(last_result, "question", "")

    display.console.print(f"\n  [bold cyan]Original Question:[/] {question}")
    display.console.print(Panel(Syntax(sql, "sql", theme="monokai"), title="[bold blue]SQL Query[/]", border_style="blue"))

    # Break down SQL clauses
    explanation_parts = []
    lower_sql = sql.lower()

    # Tables
    if "from" in lower_sql:
        from_idx = lower_sql.find("from")
        clause = sql[from_idx + 4:].split("where")[0].split("group")[0].split("order")[0].split(";")[0].strip()
        explanation_parts.append(f"[bold]• Sources:[/] Queries data from [cyan]{clause}[/]")

    # Joins
    if "join" in lower_sql:
        explanation_parts.append("[bold]• Relationships:[/] Combines related records across tables using foreign key matches.")

    # Filters
    if "where" in lower_sql:
        where_idx = lower_sql.find("where")
        clause = sql[where_idx + 5:].split("group")[0].split("order")[0].split("limit")[0].split(";")[0].strip()
        explanation_parts.append(f"[bold]• Filters:[/] Only includes rows matching condition(s): [yellow]{clause}[/]")

    # Aggregations & Groupings
    if "group by" in lower_sql:
        grp_idx = lower_sql.find("group by")
        clause = sql[grp_idx + 8:].split("having")[0].split("order")[0].split("limit")[0].split(";")[0].strip()
        explanation_parts.append(f"[bold]• Grouping:[/] Summarizes metrics grouped by [cyan]{clause}[/]")

    if any(agg in lower_sql for agg in ("sum(", "count(", "avg(", "max(", "min(")):
        aggs = [agg.upper()[:-1] for agg in ("sum(", "count(", "avg(", "max(", "min(") if agg in lower_sql]
        explanation_parts.append(f"[bold]• Calculations:[/] Computes mathematical aggregates ({', '.join(aggs)})")

    # Sorting
    if "order by" in lower_sql:
        ord_idx = lower_sql.find("order by")
        clause = sql[ord_idx + 8:].split("limit")[0].split(";")[0].strip()
        desc = "descending (highest first)" if "desc" in clause.lower() else "ascending (lowest first)"
        explanation_parts.append(f"[bold]• Sorting:[/] Orders output {desc} by [cyan]{clause}[/]")

    # Limit
    if "limit" in lower_sql:
        lim_idx = lower_sql.find("limit")
        clause = sql[lim_idx + 5:].split(";")[0].strip()
        explanation_parts.append(f"[bold]• Restriction:[/] Limits output to top [green]{clause}[/] rows")

    display.console.print("\n  [bold green]Query Breakdown:[/]")
    for part in explanation_parts:
        display.console.print(f"  {part}")

    if getattr(last_result, "assumptions", None):
        display.console.print("\n  [bold yellow]Assumptions Made by Agent:[/]")
        for a in last_result.assumptions:
            display.console.print(f"  📋 {a}")

    display.console.print()


def cmd_sql(args, context):
    """Execute raw SQL directly on the connected database."""
    if not context.get("db_connected"):
        display.show_error("Not connected to a database.", "Use /connect <path> first.")
        return
    if not args:
        display.show_error("Usage: /sql <raw-sql-query>", "Example: /sql SELECT * FROM Album LIMIT 3")
        return
    sql_query = " ".join(args).strip()
    db_path = context.get("db_path")

    try:
        def _sqlite_exec(sql, params=None):
            import sqlite3
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            try:
                cur = conn.cursor()
                cur.execute(sql, params or {})
                return [dict(r) for r in cur.fetchall()]
            finally:
                conn.close()

        executor = QueryExecutor(execute_fn=_sqlite_exec)
        exec_result = executor.execute(sql_query)

        if exec_result.error:
            display.show_error(f"SQL execution error: {exec_result.error}")
            return

        display.show_query_result(
            question="Raw SQL execution",
            sql=sql_query,
            rows=exec_result.rows,
            row_count=exec_result.row_count,
            execution_time_ms=exec_result.execution_time_ms,
        )

        from src.agent.pipeline import PipelineResult
        result = PipelineResult(
            question="Raw SQL execution",
            sql=sql_query,
            rows=exec_result.rows,
            row_count=exec_result.row_count,
            column_names=exec_result.column_names,
            success=True,
            execution_time_ms=exec_result.execution_time_ms,
        )
        context["last_result"] = result
        context["history"].append({
            "question": sql_query,
            "sql": sql_query,
            "row_count": exec_result.row_count,
            "success": True,
            "timestamp": "now",
        })
    except Exception as e:
        display.show_error(f"Execution failed: {e}")


def register(registry):
    registry.register("explain", cmd_explain, "Explain the last query in plain English", "/explain", "Data & Analysis")
    registry.register("sql", cmd_sql, "Execute raw read-only SQL directly", "/sql <query>", "Data & Analysis")
