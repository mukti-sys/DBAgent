"""Model commands — /model to switch models."""
from InquirerPy import inquirer
from src.interface import display
from src.interface.display import THEME as T, console
from src.interface.commands.provider import PROVIDERS, _save_provider_config


def cmd_model(args, context):
    if args:
        model_name = args[0]
        context["model_name"] = model_name
        _save_provider_config(context)
        display.show_success(f"Model switched to {model_name}")
        return

    provider_key = context.get("provider_key")
    if not provider_key:
        display.show_error("No provider configured.", "Run /provider first.")
        return

    provider = PROVIDERS.get(provider_key, {})
    suggestions = provider.get("models", [])
    if suggestions:
        console.print()
        console.print(f"  [{T['text_secondary']}]Suggestions:[/]")
        for s in suggestions:
            console.print(f"    [{T['text_secondary']}]{s}[/]")
        console.print()

    current = context.get("model_name", "")
    try:
        model_name = inquirer.text(
            message="Model name:",
            default=current,
            qmark="  ",
            amark="  ",
        ).execute()
    except (KeyboardInterrupt, EOFError):
        return

    if model_name:
        context["model_name"] = model_name
        _save_provider_config(context)
        display.show_success(f"Model switched to {model_name}")


def register(registry):
    registry.register("model", cmd_model, "Switch LLM model", "/model [name]", "Provider & Model")
