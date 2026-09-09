"""
CLI entry point for DBAgent.

Usage:
    python -m src.interface.cli                       # Interactive setup
    python -m src.interface.cli --db chinook.db        # Connect directly
    python -m src.interface.cli --db chinook.db --model gemini-2.0-flash
"""

import argparse
import logging
import sys


def main():
    parser = argparse.ArgumentParser(
        description="DBAgent — Ask your database questions in plain English.",
    )
    parser.add_argument("--db", type=str, default=None, help="Path to database file")
    parser.add_argument("--model", type=str, default=None, help="LLM model name")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stderr)])

    from src.interface.app import App
    app = App(db_path=args.db, model=args.model)
    app.run()


if __name__ == "__main__":
    main()
