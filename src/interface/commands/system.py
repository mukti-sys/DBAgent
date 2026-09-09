"""System commands — /help, /clear, /status, /quit"""
import os
from src.interface import display


def cmd_help(args, context):
    registry = context.get("registry")
    if not registry:
        return
    if args:
        cmd = registry.get(args[0].lstrip("/"))
        if cmd:
            display.console.print(f"\n  [bold green]{cmd['usage']}[/]")
            display.console.print(f"  {cmd['description']}\n")
        else:
            display.show_error(f"Unknown command: {args[0]}")
        return
    display.show_help(registry.get_help())


def cmd_clear(args, context):
    os.system("cls" if os.name == "nt" else "clear")


def cmd_status(args, context):
    display.show_status(
        db_path=context.get("db_path"), model=context.get("model_name"),
        dialect=context.get("dialect", "sqlite"), session_id=context.get("session_id"),
    )


def cmd_quit(args, context):
    display.console.print("\n  [dim]Goodbye! 👋[/]\n")
    context["should_quit"] = True


def cmd_history(args, context):
    display.show_history(context.get("history", []))


def register(registry):
    registry.register("help", cmd_help, "Show all commands", "/help [command]", "System")
    registry.register("clear", cmd_clear, "Clear the terminal", "/clear", "System")
    registry.register("status", cmd_status, "Show connection status", "/status", "System")
    registry.register("quit", cmd_quit, "Exit DBAgent", "/quit", "System")
    registry.register("exit", cmd_quit, "Exit DBAgent", "/exit", "System")
    registry.register("history", cmd_history, "Show query history", "/history", "System")
