"""
Glossary store — org-specific term definitions.

Loads terms from config/glossary.yaml. Each term maps a business
concept (e.g. "active user", "revenue") to its SQL-level meaning.

Per PRD.md §4: DBAgent surfaces conflicts between definitions;
it does not adjudicate them (that's an org data-governance problem).
"""

from dataclasses import dataclass, field
from typing import Any

from src.config import load_glossary


@dataclass
class GlossaryTerm:
    """A single business term definition."""
    term: str
    definition: str
    sql_expression: str = ""
    notes: str = ""
    alternatives: list[str] = field(default_factory=list)

    def matches(self, text: str) -> bool:
        """Check if this term appears in the given text (case-insensitive)."""
        text_lower = text.lower()
        if self.term.lower() in text_lower:
            return True
        return any(alt.lower() in text_lower for alt in self.alternatives)


class GlossaryStore:
    """
    Manages org-specific term definitions.

    Loads from config/glossary.yaml and provides lookup for the intent
    reconstruction step. Surfaces conflicts — never silently picks one
    definition (Rules.md §4, PRD.md §4).
    """

    def __init__(self, terms: list[GlossaryTerm] | None = None):
        self._terms: list[GlossaryTerm] = terms or []

    @classmethod
    def from_config(cls) -> "GlossaryStore":
        """Load glossary from config/glossary.yaml."""
        raw = load_glossary()
        terms = []
        for entry in raw.get("terms", []):
            if isinstance(entry, dict):
                terms.append(GlossaryTerm(
                    term=entry.get("term", ""),
                    definition=entry.get("definition", ""),
                    sql_expression=entry.get("sql_expression", ""),
                    notes=entry.get("notes", ""),
                    alternatives=entry.get("alternatives", []),
                ))
        return cls(terms=terms)

    def lookup(self, text: str) -> list[GlossaryTerm]:
        """Find all glossary terms that appear in the given text."""
        return [t for t in self._terms if t.matches(text)]

    def add_term(self, term: GlossaryTerm) -> None:
        """Add a term to the glossary."""
        self._terms.append(term)

    def get_all_terms(self) -> list[GlossaryTerm]:
        """Return all terms."""
        return list(self._terms)

    def find_conflicts(self, term_name: str) -> list[GlossaryTerm]:
        """
        Find conflicting definitions for a term.

        Per PRD.md §4: surface the conflict, don't adjudicate it.
        """
        matches = [
            t for t in self._terms
            if t.term.lower() == term_name.lower()
        ]
        if len(matches) > 1:
            return matches
        return []

    def format_for_prompt(self, terms: list[GlossaryTerm]) -> str:
        """Format matched terms for inclusion in an LLM prompt."""
        if not terms:
            return ""

        lines = ["Glossary terms relevant to this question:"]
        for t in terms:
            line = f"- {t.term}: {t.definition}"
            if t.sql_expression:
                line += f" (SQL: {t.sql_expression})"
            if t.notes:
                line += f" [Note: {t.notes}]"
            lines.append(line)
        return "\n".join(lines)
