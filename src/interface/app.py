"""
Main application loop — wires everything together.
"""

import logging
from datetime import datetime
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.formatted_text import HTML
from pathlib import Path

from src.interface import display
from src.interface.commands import CommandRegistry
from src.interface.commands import system as system_cmds
from src.interface.commands import database as database_cmds
from src.interface.commands import provider as provider_cmds
from src.interface.commands import model as model_cmds

logger = logging.getLogger(__name__)


class App:
    """Main DBAgent terminal application."""

    def __init__(self, db_path=None, model=None):
        self.context = {
            "db_path": db_path, "db_connected": False, "dialect": "sqlite",
            "model_name": model, "provider_name": None, "provider_key": None,
            "api_key": None, "base_url": None,
            "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "should_quit": False, "history": [], "last_result": None,
            "table_info": {}, "table_names": [], "known_columns": {},
        }
        self.registry = CommandRegistry()
        self.context["registry"] = self.registry
        system_cmds.register(self.registry)
        database_cmds.register(self.registry)
        provider_cmds.register(self.registry)
        model_cmds.register(self.registry)

        history_path = Path("config/.prompt_history")
        history_path.parent.mkdir(parents=True, exist_ok=True)
        self.prompt_session = PromptSession(
            history=FileHistory(str(history_path)),
            auto_suggest=AutoSuggestFromHistory(),
        )
        self._load_saved_config()

    def run(self):
        display.show_banner()
        if self.context.get("db_path"):
            database_cmds.cmd_connect([self.context["db_path"]], self.context)
        if not self.context.get("model_name"):
            display.console.print(
                "  [yellow]No LLM provider configured.[/]\n"
                "  [dim]Run [bold green]/provider[/] to set up, "
                "or [bold green]/connect <db>[/] to explore.[/]\n"
            )

        while not self.context.get("should_quit"):
            try:
                user_input = self.prompt_session.prompt(
                    HTML("<ansigreen><b>DBAgent</b></ansigreen> <ansigray>›</ansigray> "),
                ).strip()
                if not user_input:
                    continue
                self._handle_input(user_input)
            except KeyboardInterrupt:
                display.console.print()
                continue
            except EOFError:
                display.console.print("\n  [dim]Goodbye! 👋[/]\n")
                break

    def _handle_input(self, user_input):
        if user_input.startswith("/"):
            parts = user_input[1:].split(maxsplit=1)
            cmd_name = parts[0].lower() if parts else ""
            cmd_args = parts[1].split() if len(parts) > 1 else []
            if not self.registry.execute(cmd_name, cmd_args, self.context):
                display.show_error(f"Unknown command: /{cmd_name}", "Type /help for commands.")
        else:
            self._handle_question(user_input)

    def _handle_question(self, question):
        if not self.context.get("db_connected"):
            display.show_error("Not connected to a database.", "Use /connect <path> first.")
            return
        if not self.context.get("model_name"):
            display.show_error("No LLM model configured.", "Use /provider to set up.")
            return

        display.console.print("  [dim]Thinking...[/]", end="\r")
        try:
            result = self._run_pipeline(question)
            display.console.print("              ", end="\r")
            if result.refused:
                display.show_error(f"Refused: {result.refusal_reason}")
            elif result.error:
                display.show_error(result.error)
            else:
                display.show_query_result(
                    question=question, sql=result.sql, rows=result.rows,
                    row_count=result.row_count, confidence_level=result.confidence_level,
                    confidence_score=result.confidence_score,
                    validation_findings=result.validation_findings,
                    assumptions=result.assumptions, attempts=result.attempts,
                    execution_time_ms=result.execution_time_ms,
                    recalled_facts=result.recalled_facts,
                )
            self.context["last_result"] = result
            self.context["history"].append({
                "question": question, "sql": result.sql,
                "row_count": result.row_count, "success": result.success,
                "timestamp": datetime.now().isoformat(),
            })
        except Exception as e:
            display.console.print("              ", end="\r")
            display.show_error(f"Pipeline error: {e}")
            logger.exception("Pipeline error")

    def _run_pipeline(self, question):
        from src.agent.pipeline import Pipeline, PipelineResult
        from src.agent.sql_generator import SQLGenerator
        from src.agent.executor import QueryExecutor
        from src.agent.critic import Critic

        dialect = self.context.get("dialect", "sqlite")
        db_path = self.context["db_path"]

        try:
            from src.agent.llm_client import LLMClient
            llm = LLMClient(model=self.context.get("model_name"),
                            provider=self.context.get("provider_key"))
        except Exception as e:
            return PipelineResult(question=question, error=f"LLM client error: {e}")

        sql_generator = SQLGenerator(llm_client=llm, dialect=dialect)
        executor = QueryExecutor(db_path=db_path, read_only=True)
        critic = Critic(
            known_tables=self.context.get("table_names", []),
            known_columns=self.context.get("known_columns", {}),
            dialect=dialect,
        )

        result_validator = None
        try:
            from src.agent.result_validator import ResultValidator
            result_validator = ResultValidator()
        except Exception:
            pass

        pipeline = Pipeline(
            sql_generator=sql_generator, executor=executor, critic=critic,
            result_validator=result_validator,
        )
        return pipeline.run(question)

    def _load_saved_config(self):
        config_path = Path("config/providers.yaml")
        if not config_path.exists():
            return
        try:
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
            active_model = config.get("active_model")
            if active_model and not self.context.get("model_name"):
                self.context["model_name"] = active_model
            for p in config.get("providers", []):
                if p.get("model") == active_model:
                    self.context.update({
                        "provider_name": p.get("name"),
                        "provider_key": p.get("key"),
                        "base_url": p.get("base_url"),
                    })
                    break
        except Exception:
            pass
