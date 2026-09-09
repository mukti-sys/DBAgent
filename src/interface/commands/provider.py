"""Provider commands — /provider interactive wizard for LLM setup."""
import os
import yaml
from pathlib import Path
from InquirerPy import inquirer
from InquirerPy.separator import Separator
from src.interface import display

PROVIDERS = {
    "openai": {"name": "OpenAI", "description": "GPT-4o, GPT-4o-mini, o3",
               "base_url": "https://api.openai.com/v1", "env_key": "OPENAI_API_KEY",
               "models": ["gpt-4o", "gpt-4o-mini", "o3-mini"]},
    "google": {"name": "Google Gemini", "description": "Gemini Flash, Gemini Pro",
               "base_url": "https://generativelanguage.googleapis.com/v1beta", "env_key": "GOOGLE_API_KEY",
               "models": ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"]},
    "nvidia": {"name": "NVIDIA NIM", "description": "Llama-3.1-70B, Mixtral",
               "base_url": "https://integrate.api.nvidia.com/v1", "env_key": "NVIDIA_API_KEY",
               "models": ["meta/llama-3.1-70b-instruct"]},
    "deepseek": {"name": "DeepSeek", "description": "DeepSeek V4 Flash, V4 Pro",
                 "base_url": "https://api.deepseek.com/v1", "env_key": "DEEPSEEK_API_KEY",
                 "models": ["deepseek-chat", "deepseek-coder"]},
    "openrouter": {"name": "OpenRouter", "description": "Access 100+ models",
                   "base_url": "https://openrouter.ai/api/v1", "env_key": "OPENROUTER_API_KEY",
                   "models": ["meta-llama/llama-3.1-70b-instruct"]},
    "ollama": {"name": "Ollama (Local)", "description": "Run models locally — no API key",
               "base_url": "http://localhost:11434/v1", "env_key": "",
               "models": ["llama3.1", "qwen2.5-coder", "mistral"]},
    "custom": {"name": "Custom Provider", "description": "Any OpenAI-compatible endpoint",
               "base_url": "", "env_key": "", "models": []},
}


def cmd_provider(args, context):
    if args and args[0] == "list":
        _show_providers(context)
        return
    display.console.print()
    choices = []
    for key, info in PROVIDERS.items():
        if key == "custom":
            choices.append(Separator())
        choices.append({"name": f"{info['name']}\n    {info['description']}", "value": key})
    try:
        provider_key = inquirer.select(
            message="Select a Provider", choices=choices, pointer="›",
            instruction="(↑↓ to navigate, Enter to select, Esc to cancel)",
        ).execute()
    except (KeyboardInterrupt, EOFError):
        display.console.print("  [dim]Cancelled.[/]\n")
        return
    provider = PROVIDERS[provider_key]

    api_key = ""
    if provider_key != "ollama":
        try:
            api_key = inquirer.secret(message=f"Enter your {provider['name']} API key:").execute()
        except (KeyboardInterrupt, EOFError):
            return
        if not api_key:
            display.show_error("API key is required.")
            return

    default_url = provider["base_url"]
    try:
        if provider_key == "custom":
            base_url = inquirer.text(message="Enter the base URL:").execute()
        else:
            base_url = inquirer.text(message="Base URL:", default=default_url).execute()
    except (KeyboardInterrupt, EOFError):
        return

    if provider.get("models"):
        try:
            model_name = inquirer.select(
                message="Select a model:", choices=provider["models"] + ["Enter custom model name"],
                pointer="›",
            ).execute()
        except (KeyboardInterrupt, EOFError):
            return
        if model_name == "Enter custom model name":
            try:
                model_name = inquirer.text(message="Model name:").execute()
            except (KeyboardInterrupt, EOFError):
                return
    else:
        try:
            model_name = inquirer.text(message="Model name:").execute()
        except (KeyboardInterrupt, EOFError):
            return

    context.update({
        "provider_name": provider["name"], "provider_key": provider_key,
        "api_key": api_key, "base_url": base_url, "model_name": model_name,
    })
    if api_key and provider.get("env_key"):
        os.environ[provider["env_key"]] = api_key
    _save_provider_config(context)
    display.show_success(f"Provider configured: {provider['name']} → {model_name}")
    display.console.print(f"  [dim]Base URL: {base_url}[/]\n")


def _show_providers(context):
    config_path = Path("config/providers.yaml")
    if not config_path.exists():
        display.console.print("  [dim]No providers configured. Run /provider to set up.[/]\n")
        return
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
    providers = config.get("providers", [])
    if not providers:
        display.console.print("  [dim]No providers configured.[/]\n")
        return
    from rich.table import Table
    from rich import box
    table = Table(title="[bold]Configured Providers[/]", box=box.ROUNDED)
    table.add_column("Provider", style="bold")
    table.add_column("Model", style="cyan")
    table.add_column("Base URL", style="dim")
    for p in providers:
        table.add_row(p.get("name", ""), p.get("model", ""), p.get("base_url", "")[:40])
    display.console.print(table)


def _save_provider_config(context):
    config_path = Path("config/providers.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    providers = config.get("providers", [])
    entry = {
        "name": context.get("provider_name", ""), "key": context.get("provider_key", ""),
        "model": context.get("model_name", ""), "base_url": context.get("base_url", ""),
    }
    found = False
    for i, p in enumerate(providers):
        if p.get("key") == entry["key"]:
            providers[i] = entry
            found = True
            break
    if not found:
        providers.append(entry)
    config["providers"] = providers
    config["active_model"] = context.get("model_name", "")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def register(registry):
    registry.register("provider", cmd_provider, "Interactive LLM provider setup wizard",
                      "/provider", "Provider & Model")
