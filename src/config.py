"""
Config loader for DBAgent.

Loads settings from config/settings.yaml and config/glossary.yaml.
All thresholds and connection settings live in config/, not hardcoded (Rules.md §3).
"""

import os
from pathlib import Path
from typing import Any

import yaml


def _find_config_dir() -> Path:
    """Locate the config directory relative to the project root."""
    # Walk up from this file to find the project root (where config/ lives)
    current = Path(__file__).resolve().parent
    while current != current.parent:
        candidate = current / "config"
        if candidate.is_dir():
            return candidate
        current = current.parent
    raise FileNotFoundError("Could not locate config/ directory")


def load_settings() -> dict[str, Any]:
    """Load settings from config/settings.yaml, with env var overrides."""
    config_dir = _find_config_dir()
    settings_path = config_dir / "settings.yaml"

    if not settings_path.exists():
        raise FileNotFoundError(f"Settings file not found: {settings_path}")

    with open(settings_path, "r") as f:
        settings = yaml.safe_load(f) or {}

    # Environment variable overrides
    if db_url := os.environ.get("DB_URL"):
        settings.setdefault("database", {})["url"] = db_url

    if anthropic_model := os.environ.get("ANTHROPIC_MODEL"):
        settings.setdefault("llm", {})["model"] = anthropic_model

    return settings


def load_glossary() -> dict[str, Any]:
    """Load glossary from config/glossary.yaml."""
    config_dir = _find_config_dir()
    glossary_path = config_dir / "glossary.yaml"

    if not glossary_path.exists():
        return {"terms": []}

    with open(glossary_path, "r") as f:
        glossary = yaml.safe_load(f) or {"terms": []}

    return glossary
