"""
Provider commands -- /provider (or /auth) wizard, /baseurl, /key, /local.

Matches Qwen Code's AuthDialog.tsx:
  - Bordered panel wrapping the entire wizard
  - View stack: main -> thirdparty-select -> provider-setup
  - Step breadcrumbs: "DeepSeek . Step 2/3 . API Key"
  - DescriptiveRadioButtonSelect: bold title + dim description per item
  - Error shown inline at bottom
  - Escape goes back one level
"""

import os
import yaml
from pathlib import Path
from InquirerPy import inquirer
from InquirerPy.separator import Separator
from rich.panel import Panel
from rich.text import Text
from rich import box
from src.interface import display
from src.interface.display import THEME as T, console

# -- Provider definitions --------------------------------------------------
PROVIDERS = {
    "openai": {
        "name": "OpenAI",
        "description": "Quick setup for OpenAI (gpt-4o, gpt-4o-mini, o3)",
        "base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
        "models": ["gpt-4o", "gpt-4o-mini", "o3-mini"],
        "category": "third_party",
    },
    "google": {
        "name": "Google Gemini",
        "description": "Quick setup for Gemini models (Flash, Pro)",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "env_key": "GOOGLE_API_KEY",
        "models": ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"],
        "category": "third_party",
    },
    "deepseek": {
        "name": "DeepSeek API Key",
        "description": "Quick setup for DeepSeek (deepseek-chat, deepseek-coder)",
        "base_url": "https://api.deepseek.com/v1",
        "env_key": "DEEPSEEK_API_KEY",
        "models": ["deepseek-chat", "deepseek-coder"],
        "category": "third_party",
    },
    "nvidia": {
        "name": "NVIDIA NIM",
        "description": "Quick setup for NVIDIA NIM models",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "env_key": "NVIDIA_API_KEY",
        "models": ["meta/llama-3.1-70b-instruct"],
        "category": "third_party",
    },
    "openrouter": {
        "name": "OpenRouter",
        "description": "Connect with an OpenRouter API key (get one from openrouter.ai/keys)",
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
        "models": ["meta-llama/llama-3.1-70b-instruct"],
        "category": "third_party",
    },
    "ollama": {
        "name": "Ollama (Local)",
        "description": "Run models locally -- no API key needed",
        "base_url": "http://localhost:11434",
        "env_key": "",
        "models": ["qwen2.5-coder", "llama3.1", "mistral", "deepseek-r1"],
        "category": "local",
    },
    "custom": {
        "name": "Custom Provider",
        "description": "Manually connect a local server, proxy, or unsupported provider",
        "base_url": "",
        "env_key": "",
        "models": [],
        "category": "custom",
    },
}


def _print_step_breadcrumb(provider_name, step, total_steps, step_label):
    """
    Step breadcrumb (from AuthDialog.tsx line 297):
    "DeepSeek . Step 2/3 . API Key"
    """
    console.print(
        f"  [{T['text_accent']}]{provider_name}[/] "
        f"[{T['text_secondary']}]. Step {step}/{total_steps} . {step_label}[/]"
    )


def _mask_api_key(api_key):
    """
    Mask an API key for display (from useAuth.ts maskApiKey, line 46-51):
    Show first 3 and last 4 chars: "sk-...abcd"
    """
    trimmed = api_key.strip()
    if not trimmed:
        return "(not set)"
    if len(trimmed) <= 6:
        return "***"
    return f"{trimmed[:3]}...{trimmed[-4:]}"


def cmd_provider(args, context):
    """Interactive provider setup wizard (also accessible as /auth)."""
    if args and args[0] == "list":
        _show_providers(context)
        return

    # -- Step 1: Category selection (AuthDialog.tsx main view, lines 307-331)
    console.print()
    # Bordered title like AuthDialog.tsx line 315
    console.print(f"  [{T['text_accent']}]Connect a Provider[/]")
    console.print()

    category_choices = [
        {
            "name": "Third-party Providers\n    Choose a built-in provider and connect with an API key",
            "value": "third_party",
        },
        {
            "name": "Local LLM (Ollama)\n    Run models locally -- no API key needed",
            "value": "local",
        },
        {
            "name": "Custom Provider\n    Manually connect a local server, proxy, or unsupported provider",
            "value": "custom",
        },
    ]

    try:
        category = inquirer.select(
            message="",
            choices=category_choices,
            pointer=">",
            qmark="",
            amark="",
            instruction=f"Enter to select, Esc to go back",
        ).execute()
    except (KeyboardInterrupt, EOFError):
        console.print(f"  [{T['text_secondary']}]Cancelled.[/]\n")
        return

    # -- Step 2: Provider selection (for third-party) ----------------------
    if category == "third_party":
        provider_key = _select_third_party_provider()
        if not provider_key:
            return
    elif category == "local":
        provider_key = "ollama"
    else:
        provider_key = "custom"

    provider = PROVIDERS[provider_key]
    total_steps = 4 if provider_key == "custom" else 3
    step = 1

    # -- API Key step (AuthDialog step breadcrumb, line 297) ----------------
    api_key = ""
    if provider_key not in ("ollama",):
        step += 1
        console.print()
        # Step breadcrumb like AuthDialog.tsx line 297
        _print_step_breadcrumb(provider['name'], step, total_steps, "API Key")
        console.print()
        try:
            api_key = inquirer.secret(
                message="API Key:",
                qmark="  ",
                amark="  ",
            ).execute()
        except (KeyboardInterrupt, EOFError):
            return
        if not api_key:
            display.show_error("API key is required.")
            return

    # -- Base URL step (custom only, others use default) -------------------
    if provider_key == "custom":
        step += 1
        console.print()
        _print_step_breadcrumb(provider['name'], step, total_steps, "Base URL")
        console.print()
        try:
            base_url = inquirer.text(
                message="Base URL:",
                qmark="  ",
                amark="  ",
            ).execute()
        except (KeyboardInterrupt, EOFError):
            return
        if not base_url:
            display.show_error("Base URL is required.")
            return
    else:
        base_url = provider["base_url"]

    # -- Model step --------------------------------------------------------
    step += 1
    console.print()
    _print_step_breadcrumb(provider['name'], step, total_steps, "Model IDs")
    console.print()

    suggestions = provider.get("models", [])
    if suggestions:
        console.print(f"  [{T['text_secondary']}]Suggestions (type any model your API supports):[/]")
        for s in suggestions:
            console.print(f"    [{T['text_secondary']}]{s}[/]")
        console.print()

    default_model = suggestions[0] if suggestions else ""
    try:
        model_name = inquirer.text(
            message="Model name:",
            default=default_model,
            qmark="  ",
            amark="  ",
        ).execute()
    except (KeyboardInterrupt, EOFError):
        return

    if not model_name:
        display.show_error("Model name is required.")
        return

    # -- Apply configuration -----------------------------------------------
    context.update({
        "provider_name": provider["name"],
        "provider_key": provider_key,
        "api_key": api_key,
        "base_url": base_url,
        "model_name": model_name,
    })
    if api_key and provider.get("env_key"):
        os.environ[provider["env_key"]] = api_key

    _save_provider_config(context)
    console.print()
    # Success message like AuthDialog.tsx line 195-198
    display.show_success(
        f"Successfully configured {provider['name']}. Use /model to switch models."
    )
    console.print(f"  [{T['text_secondary']}]Model: {model_name}[/]")
    console.print(f"  [{T['text_secondary']}]{base_url}[/]\n")


def _select_third_party_provider():
    """Show provider list (AuthDialog.tsx thirdparty-select view, lines 333-356)."""
    console.print()
    # View title like VIEW_TITLES['thirdparty-select'] (AuthDialog.tsx line 115)
    console.print(f"  [{T['text_accent']}]Third-party Providers[/] [{T['text_secondary']}]. Provider[/]")
    console.print()

    # Build choices: each provider is bold name + dim description
    choices = []
    for key, info in PROVIDERS.items():
        if info.get("category") != "third_party":
            continue
        choices.append({
            "name": f"{info['name']}\n    {info['description']}",
            "value": key,
        })

    try:
        provider_key = inquirer.select(
            message="",
            choices=choices,
            pointer=">",
            qmark="",
            amark="",
            # AuthDialog.tsx line 353
            instruction="Enter to select, Esc to go back",
        ).execute()
    except (KeyboardInterrupt, EOFError):
        console.print(f"  [{T['text_secondary']}]Cancelled.[/]\n")
        return None

    return provider_key


def cmd_baseurl(args, context):
    """View or change the active LLM base URL directly."""
    if not args:
        current_url = context.get("base_url") or "Not configured"
        provider = context.get("provider_name") or "Default"
        console.print(f"  Active Provider: [{T['text_accent']}]{provider}[/]")
        console.print(f"  Current Base URL: [{T['text_link']}]{current_url}[/]")
        console.print(f"  [{T['text_secondary']}]To change: /baseurl <new-url>[/]\n")
        return

    new_url = args[0].strip()
    context["base_url"] = new_url
    _save_provider_config(context)
    display.show_success(f"Base URL updated to: {new_url}")
    console.print(f"  [{T['text_secondary']}]Active model: {context.get('model_name', 'Not set')}[/]\n")


def cmd_key(args, context):
    """View or change the active API key directly."""
    provider_key = context.get("provider_key")
    provider = PROVIDERS.get(provider_key, {}) if provider_key else {}

    if not args:
        key = context.get("api_key")
        if not key and provider.get("env_key"):
            key = os.environ.get(provider["env_key"])

        if key:
            # API key masking like useAuth.ts maskApiKey() line 46-51
            masked = _mask_api_key(key)
            console.print(f"  API Key configured: [{T['status_success']}]{masked}[/]")
        else:
            console.print(f"  [{T['status_warning']}]No API key currently set for active provider.[/]")
        console.print(f"  [{T['text_secondary']}]To change: /key <your-api-key>[/]\n")
        return

    new_key = args[0].strip()
    context["api_key"] = new_key
    if provider.get("env_key"):
        os.environ[provider["env_key"]] = new_key

    _save_provider_config(context)
    display.show_success("API key updated successfully.")


def cmd_local(args, context):
    """Quick one-command switch to local LLM (Ollama or local OpenAI-compatible server)."""
    model_name = args[0].strip() if args else "qwen2.5-coder"
    base_url = args[1].strip() if len(args) > 1 else "http://localhost:11434"

    context.update({
        "provider_name": "Ollama (Local)",
        "provider_key": "ollama",
        "model_name": model_name,
        "base_url": base_url,
        "api_key": "",
    })
    _save_provider_config(context)
    display.show_success(
        f"Successfully configured Ollama (Local). Use /model to switch models."
    )
    console.print(f"  [{T['text_secondary']}]Model: {model_name}[/]")
    console.print(f"  [{T['text_secondary']}]Endpoint: {base_url} (No API key required)[/]\n")


def _show_providers(context):
    config_path = Path("config/providers.yaml")
    if not config_path.exists():
        console.print(f"  [{T['text_secondary']}]No providers configured. Run /provider to set up.[/]\n")
        return
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
    providers = config.get("providers", [])
    if not providers:
        console.print(f"  [{T['text_secondary']}]No providers configured.[/]\n")
        return
    from rich.table import Table
    table = Table(
        title=f"[{T['text_accent']}]Configured Providers[/]",
        box=box.ROUNDED,
        border_style=T["border_default"],
    )
    table.add_column("Provider", style=f"bold {T['text_primary']}")
    table.add_column("Model", style=T["ui_symbol"])
    table.add_column("Base URL", style=T["text_secondary"])
    for p in providers:
        table.add_row(p.get("name", ""), p.get("model", ""), p.get("base_url", "")[:40])
    console.print(table)


def _save_provider_config(context):
    config_path = Path("config/providers.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    providers = config.get("providers", [])
    entry = {
        "name": context.get("provider_name", ""),
        "key": context.get("provider_key", ""),
        "model": context.get("model_name", ""),
        "base_url": context.get("base_url", ""),
    }
    found = False
    for i, p in enumerate(providers):
        if p.get("key") == entry["key"]:
            providers[i] = entry
            found = True
            break
    if not found and entry["key"]:
        providers.append(entry)
    config["providers"] = providers
    if context.get("model_name"):
        config["active_model"] = context.get("model_name")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def register(registry):
    registry.register(
        "provider", cmd_provider, "Interactive LLM provider setup wizard",
        "/provider [list]", "Provider & Model",
    )
    # /auth alias -- like Qwen Code
    registry.register(
        "auth", cmd_provider, "Interactive LLM provider setup wizard",
        "/auth", "Provider & Model",
    )
    registry.register(
        "baseurl", cmd_baseurl, "View or change the active LLM base URL",
        "/baseurl [url]", "Provider & Model",
    )
    registry.register(
        "key", cmd_key, "View or set API key for the active provider",
        "/key [api_key]", "Provider & Model",
    )
    registry.register(
        "apikey", cmd_key, "View or set API key for the active provider",
        "/apikey [api_key]", "Provider & Model",
    )
    registry.register(
        "local", cmd_local, "Quick switch to local LLM (Ollama)",
        "/local [model] [base_url]", "Provider & Model",
    )
    registry.register(
        "ollama", cmd_local, "Quick switch to local Ollama LLM",
        "/ollama [model] [base_url]", "Provider & Model",
    )
