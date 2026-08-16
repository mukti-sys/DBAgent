"""
Phase 0 — Scaffolding test.
Verifies that the project structure is correct and pytest runs.
"""

from pathlib import Path


def test_project_structure_exists():
    """Verify all expected directories and key files exist."""
    project_root = Path(__file__).resolve().parent.parent.parent

    # Directories per Architecture.md §4
    expected_dirs = [
        "src/agent",
        "src/db",
        "src/interface",
        "tests/unit",
        "tests/integration",
        "tests/safety",
        "tests/eval",
        "config",
    ]
    for d in expected_dirs:
        assert (project_root / d).is_dir(), f"Missing directory: {d}"

    # Key files
    expected_files = [
        "requirements.txt",
        "config/glossary.yaml",
        "config/settings.yaml",
        "src/agent/glossary.py",
        "src/agent/intent.py",
        "src/agent/schema_retrieval.py",
        "src/agent/sql_generator.py",
        "src/agent/verifier.py",
        "src/agent/confidence.py",
        "src/agent/executor.py",
        "src/agent/narrator.py",
        "src/agent/correction_memory.py",
        "src/agent/key_resolver.py",
        "src/agent/concurrency_guard.py",
        "src/db/connector.py",
        "src/db/access_control.py",
        "src/interface/cli.py",
    ]
    for f in expected_files:
        assert (project_root / f).is_file(), f"Missing file: {f}"


def test_config_loads():
    """Verify config loader can find and read settings."""
    from src.config import load_settings, load_glossary

    settings = load_settings()
    assert "database" in settings
    assert "llm" in settings
    assert "confidence" in settings

    glossary = load_glossary()
    assert "terms" in glossary


def test_pytest_runs():
    """Trivial test to confirm pytest itself is working."""
    assert True
