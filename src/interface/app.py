"""
Main application loop — wires UI, commands, pipeline, sessions, and multi-turn conversational memory.
"""

import logging
from datetime import datetime
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from pathlib import Path

from src.interface import display
from src.interface.commands import CommandRegistry
from src.interface.commands import system as system_cmds
from src.interface.commands import database as database_cmds
from src.interface.commands import provider as provider_cmds
from src.interface.commands import model as model_cmds
from src.interface.commands import session as session_cmds
from src.interface.commands import export as export_cmds
from src.interface.commands import query as query_cmds

from src.interface.session_manager import SessionManager
from src.agent.conversation_state import ConversationState, QueryContext

logger = logging.getLogger(__name__)


class App:
    """Main DBAgent terminal application."""

    def __init__(self, db_path=None, model=None):
        self.session_manager = SessionManager()
        self.conversation_state = ConversationState()

        session_id = self.session_manager.generate_session_id()

        self.context = {
            "db_path": db_path,
            "db_connected": False,
            "dialect": "sqlite",
            "model_name": model,
            "provider_name": None,
            "provider_key": None,
            "api_key": None,
            "base_url": None,
            "session_id": session_id,
            "session_name": session_id,
            "should_quit": False,
            "history": [],
            "last_result": None,
            "table_info": {},
            "table_names": [],
            "known_columns": {},
            "conversation_state": self.conversation_state,
            "session_manager": self.session_manager,
        }

        self.registry = CommandRegistry()
        self.context["registry"] = self.registry

        # Register all command categories
        system_cmds.register(self.registry)
        database_cmds.register(self.registry)
        provider_cmds.register(self.registry)
        model_cmds.register(self.registry)
        session_cmds.register(self.registry)
        export_cmds.register(self.registry)
        query_cmds.register(self.registry)

        history_path = Path("config/.prompt_history")
        history_path.parent.mkdir(parents=True, exist_ok=True)

        # Autocompletion for all slash commands
        completer = WordCompleter(
            self.registry.get_command_names(),
            ignore_case=True,
            sentence=False,
        )

        try:
            self.prompt_session = PromptSession(
                history=FileHistory(str(history_path)),
                auto_suggest=AutoSuggestFromHistory(),
                completer=completer,
            )
        except Exception:
            from prompt_toolkit.output import DummyOutput
            from prompt_toolkit.input import DummyInput
            self.prompt_session = PromptSession(
                history=FileHistory(str(history_path)),
                input=DummyInput(),
                output=DummyOutput(),
            )

        self._load_saved_config()

    def run(self):
        display.show_banner()
        if self.context.get("db_path"):
            database_cmds.cmd_connect([self.context["db_path"]], self.context)
        if not self.context.get("model_name"):
            display.console.print(
                "  [yellow]No LLM provider configured.[/]\n"
                "  [dim]Run [bold green]/provider[/] for interactive wizard, "
                "[bold green]/local[/] for local Ollama, "
                "or [bold green]/connect <db>[/] to explore tables.[/]\n"
            )

        while not self.context.get("should_quit"):
            try:
                # Dynamic prompt showing active db & session indicator
                prompt_html = self._build_prompt_html()
                user_input = self.prompt_session.prompt(HTML(prompt_html)).strip()
                if not user_input:
                    continue
                self._handle_input(user_input)
            except KeyboardInterrupt:
                display.console.print()
                continue
            except EOFError:
                self._cleanup_and_exit()
                break

    def _build_prompt_html(self) -> str:
        db_name = ""
        if self.context.get("db_connected") and self.context.get("db_path"):
            p = Path(self.context["db_path"]).name
            db_name = f"<ansicyan>({p})</ansicyan> "

        return f"<ansigreen><b>DBAgent</b></ansigreen> {db_name}<ansigray>›</ansigray> "

    def _cleanup_and_exit(self):
        # Auto-save session on exit
        if self.context.get("session_id") and self.context.get("history"):
            try:
                self.session_manager.save_session(
                    session_id=self.context["session_id"],
                    context=self.context,
                    conversation_state=self.conversation_state,
                )
            except Exception:
                pass
        display.console.print("\n  [dim]Goodbye! 👋[/]\n")

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
            display.show_error("No LLM model configured.", "Use /provider or /local to set up.")
            return

        # Multi-turn context resolution
        inheritance = self.conversation_state.resolve_context(question)

        prompt_question = question
        if inheritance.is_ambiguous and inheritance.clarifying_question:
            display.console.print(f"\n  [yellow]❓ {inheritance.clarifying_question}[/]\n")

        if inheritance.is_follow_up:
            display.console.print("  [dim]🔗 Following up from previous query...[/]")
            prompt_question = self.conversation_state.build_context_prompt(question, inheritance)

        display.console.print("  [dim]Thinking...[/]", end="\r")
        try:
            result = self._run_pipeline(prompt_question)
            display.console.print("              ", end="\r")

            if result.refused:
                display.show_error(f"Refused: {result.refusal_reason}")
            elif result.error:
                display.show_error(result.error)
            else:
                display.show_query_result(
                    question=question,
                    sql=result.sql,
                    rows=result.rows,
                    row_count=result.row_count,
                    confidence_level=result.confidence_level,
                    confidence_score=result.confidence_score,
                    validation_findings=result.validation_findings,
                    assumptions=result.assumptions,
                    attempts=result.attempts,
                    execution_time_ms=result.execution_time_ms,
                    recalled_facts=result.recalled_facts,
                )

            self.context["last_result"] = result
            self.context["history"].append({
                "question": question,
                "sql": result.sql,
                "row_count": result.row_count,
                "success": result.success,
                "timestamp": datetime.now().isoformat(),
            })

            # Record turn in multi-turn conversational memory
            if result.success and result.sql:
                turn_context = QueryContext(
                    question=question,
                    sql=result.sql,
                    filters=inheritance.inherited_filters if inheritance.is_follow_up else {},
                    tables_used=result.column_names,
                    result_summary=f"{result.row_count} rows",
                )
                self.conversation_state.add_turn(turn_context)

            # Auto-save session snapshot
            self.session_manager.save_session(
                session_id=self.context["session_id"],
                context=self.context,
                conversation_state=self.conversation_state,
            )

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
            from src.agent.llm_client import LLMClient, ProviderConfig, ModelEntry
            llm = LLMClient()
            model_name = self.context.get("model_name", "gpt-4o")
            provider_key = self.context.get("provider_key") or "openai"
            base_url = self.context.get("base_url")

            llm._providers[provider_key] = ProviderConfig(
                name=provider_key,
                type=provider_key,
                credentials_env=f"{provider_key.upper()}_API_KEY",
                base_url=base_url,
            )
            llm._models.append(ModelEntry(id=model_name, provider=provider_key))
            llm._role_map["generation"] = model_name
        except Exception as e:
            return PipelineResult(question=question, error=f"LLM client error: {e}")

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

        sql_generator = SQLGenerator(llm_client=llm, dialect=dialect)
        executor = QueryExecutor(execute_fn=_sqlite_exec)
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
            sql_generator=sql_generator,
            executor=executor,
            critic=critic,
            result_validator=result_validator,
            dialect=dialect,
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
