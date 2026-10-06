"""
CLI entry point for DBAgent.

Usage:
    dbagent                                   # Interactive setup
    dbagent --db chinook.db                   # Connect directly
    dbagent --db chinook.db --model gpt-4o    # Connect with a specific model
    dbagent --db mydata.db --model gemini-2.5-flash -v
"""

import argparse
import logging
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="dbagent",
        description="DBAgent -- Ask your database questions in plain English.",
    )
    parser.add_argument("--db", type=str, default=None, help="Path to database file")
    parser.add_argument("--model", type=str, default=None, help="LLM model name (any model your API supports)")
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
