"""Model commands — /model to switch models."""
from InquirerPy import inquirer
from src.interface import display
from src.interface.commands.provider import PROVIDERS


def cmd_model(args, context):
    if args:
        context["model_name"] = args[0]
        display.show_success(f"Model switched to {args[0]}")
        return
    provider_key = context.get("provider_key")
    if not provider_key:
        display.show_error("No provider configured.", "Run /provider first.")
        return
    provider = PROVIDERS.get(provider_key, {})
    models = provider.get("models", [])
    if not models:
        try:
            model_name = inquirer.text(message="Enter model name:").execute()
        except (KeyboardInterrupt, EOFError):
            return
    else:
        try:
            model_name = inquirer.select(
                message="Select a model:", choices=models + ["Enter custom model name"], pointer="›",
            ).execute()
        except (KeyboardInterrupt, EOFError):
            return
        if model_name == "Enter custom model name":
            try:
                model_name = inquirer.text(message="Model name:").execute()
            except (KeyboardInterrupt, EOFError):
                return
    if model_name:
        context["model_name"] = model_name
        display.show_success(f"Model switched to {model_name}")


def register(registry):
    registry.register("model", cmd_model, "Switch LLM model", "/model [name]", "Provider & Model")
