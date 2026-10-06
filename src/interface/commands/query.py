"""
Query commands — /explain to understand the last query, /sql to run raw SQL directly.
"""

from rich.panel import Panel
from rich.syntax import Syntax
from src.interface import display
from src.interface.display import THEME as T, console
from src.agent.executor import QueryExecutor


def cmd_explain(args, context):
    """Explain the last generated SQL query in clear, plain English."""
    last_result = context.get("last_result")
    if not last_result or not getattr(last_result, "sql", None):
        display.show_warning("No recent SQL query to explain. Run a query first.")
        return

    sql = last_result.sql
    question = getattr(last_result, "question", "")

    display.console.print(f"\n  [{T['text_link']}]Original Question:[/] {question}")
    display.console.print(Panel(Syntax(sql, "sql", theme="monokai"), title=f"[bold {T['text_accent']}]SQL Query[/]", border_style=T["text_accent"]))

    # Break down SQL clauses
    explanation_parts = []
    lower_sql = sql.lower()

    # Tables
    if "from" in lower_sql:
        from_idx = lower_sql.find("from")
        clause = sql[from_idx + 4:].split("where")[0].split("group")[0].split("order")[0].split(";")[0].strip()
        explanation_parts.append(f"[bold]* Sources:[/] Queries data from [{T['ui_symbol']}]{clause}[/]")

    # Joins
    if "join" in lower_sql:
        explanation_parts.append("[bold]• Relationships:[/] Combines related records across tables using foreign key matches.")

    # Filters
    if "where" in lower_sql:
        where_idx = lower_sql.find("where")
        clause = sql[where_idx + 5:].split("group")[0].split("order")[0].split("limit")[0].split(";")[0].strip()
        explanation_parts.append(f"[bold]* Filters:[/] Only includes rows matching: [{T['status_warning']}]{clause}[/]")

    # Aggregations & Groupings
    if "group by" in lower_sql:
        grp_idx = lower_sql.find("group by")
        clause = sql[grp_idx + 8:].split("having")[0].split("order")[0].split("limit")[0].split(";")[0].strip()
        explanation_parts.append(f"[bold]* Grouping:[/] Summarizes metrics grouped by [{T['ui_symbol']}]{clause}[/]")

    if any(agg in lower_sql for agg in ("sum(", "count(", "avg(", "max(", "min(")):
        aggs = [agg.upper()[:-1] for agg in ("sum(", "count(", "avg(", "max(", "min(") if agg in lower_sql]
        explanation_parts.append(f"[bold]• Calculations:[/] Computes mathematical aggregates ({', '.join(aggs)})")

    # Sorting
    if "order by" in lower_sql:
        ord_idx = lower_sql.find("order by")
        clause = sql[ord_idx + 8:].split("limit")[0].split(";")[0].strip()
        desc = "descending (highest first)" if "desc" in clause.lower() else "ascending (lowest first)"
        explanation_parts.append(f"[bold]* Sorting:[/] Orders output {desc} by [{T['ui_symbol']}]{clause}[/]")

    # Limit
    if "limit" in lower_sql:
        lim_idx = lower_sql.find("limit")
        clause = sql[lim_idx + 5:].split(";")[0].strip()
        explanation_parts.append(f"[bold]* Restriction:[/] Limits output to top [{T['status_success']}]{clause}[/] rows")

    console.print(f"\n  [{T['status_success']}]Query Breakdown:[/]")
    for part in explanation_parts:
        console.print(f"  {part}")

    # Uncertainty Decomposition
    unc = getattr(last_result, "uncertainty", None)
    if unc:
        console.print(f"\n  [{T['text_accent']}]Uncertainty Decomposition:[/] Overall: {unc.composite_uncertainty:.0%} ({unc.dominant_dimension})")
        dims = [
            ("Schema Linking", unc.schema_linking.uncertainty, unc.schema_linking.reasons),
            ("Join Path", unc.join_path.uncertainty, unc.join_path.reasons),
            ("Aggregation", unc.aggregation.uncertainty, unc.aggregation.reasons),
            ("Value Grounding", unc.value_grounding.uncertainty, unc.value_grounding.reasons),
        ]
        for name, score, reasons in dims:
            status_color = T['status_success'] if score < 0.25 else (T['status_warning'] if score < 0.5 else T['status_error'])
            reason_txt = f" - {reasons[0]}" if reasons else ""
            console.print(f"    • {name:<17}: [{status_color}]{score:.0%}[/]{reason_txt}")

    # Counterfactual Verification
    cf = getattr(last_result, "counterfactual_result", None)
    if cf and cf.executed:
        status_color = T['status_success'] if cf.passed else T['status_warning']
        status_label = "PASSED" if cf.passed else "FLAGGED"
        console.print(f"\n  [{T['text_link']}]Counterfactual Invariant Checks:[/] [{status_color}]{status_label}[/]")
        for chk in cf.checks:
            mark = f"[{T['status_success']}]✓[/]" if chk.passed else f"[{T['status_error']}]✗[/]"
            console.print(f"    {mark} {chk.description}")
            if not chk.passed and chk.reason:
                console.print(f"      [{T['status_error']}]{chk.reason}[/]")

    if getattr(last_result, "assumptions", None):
        console.print(f"\n  [{T['status_warning']}]Assumptions Made by Agent:[/]")
        for a in last_result.assumptions:
            console.print(f"  {a}")

    console.print()


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
