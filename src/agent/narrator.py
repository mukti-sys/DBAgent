"""
Narrator — turns query results into prose, computing figures
in code rather than letting the LLM restate numbers from memory.

Per Architecture.md §5: Narration never lets the LLM recompute a
number that's already in the query result — it substitutes the real
value into a template instead.

Per Architecture.md §3: Routes through llm_client for phrasing;
numbers are always computed in code (Rules.md §2).

Per Design.md §1: Answer output includes:
1. Interpretation
2. SQL
3. Confidence
4. Answer (narrated result with real computed numbers)
5. Notes (optional)
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from src.agent.llm_client import LLMClient as UnifiedLLMClient

logger = logging.getLogger(__name__)


@dataclass
class NarrationResult:
    """Complete narrated response per Design.md §1."""
    interpretation: str      # Plain-language restatement
    sql: str                 # The exact generated query
    confidence: str          # "high" / "flagged: ..." / "refused: ..."
    answer: str              # Narrated result with real numbers
    notes: list[str] = field(default_factory=list)

    def format_output(self) -> str:
        """Format the complete output per Design.md §1."""
        parts = [
            f"**Interpretation:** {self.interpretation}",
            f"\n**SQL:**\n```sql\n{self.sql}\n```",
            f"\n**Confidence:** {self.confidence}",
            f"\n**Answer:** {self.answer}",
        ]
        if self.notes:
            parts.append("\n**Notes:**")
            for note in self.notes:
                parts.append(f"- {note}")
        return "\n".join(parts)


class ResultNarrator:
    """
    Narrates query results using real computed values.

    Key principle: numbers in the narration are computed from the actual
    result set in code, NOT restated by the LLM from memory.

    Routes through llm_client for phrasing (Architecture.md §3).
    """

    def __init__(self, llm_client: UnifiedLLMClient | None = None):
        self._llm_client = llm_client  # For LLM-based phrasing when available

    def narrate(
        self,
        rows: list[dict[str, Any]],
        column_names: list[str],
        question: str,
        sql: str,
        confidence_display: str,
        interpretation: str = "",
        notes: list[str] | None = None,
    ) -> NarrationResult:
        """
        Build a narrated response from query results.

        All numbers are computed from the actual rows, never from
        LLM memory/generation.
        """
        # Compute summary statistics from real data
        computed = self._compute_stats(rows, column_names)

        # Build the answer text with real values
        answer = self._build_answer(rows, column_names, computed, question)

        return NarrationResult(
            interpretation=interpretation or f"Reading this as: {question}",
            sql=sql,
            confidence=confidence_display,
            answer=answer,
            notes=notes or [],
        )

    def _compute_stats(
        self,
        rows: list[dict[str, Any]],
        column_names: list[str],
    ) -> dict[str, Any]:
        """
        Compute summary statistics from actual result rows.

        All values are computed in code from real data — this is the
        function that prevents narration hallucination.
        """
        stats: dict[str, Any] = {
            "row_count": len(rows),
        }

        if not rows or not column_names:
            return stats

        # For each numeric column, compute sum/avg/min/max
        for col in column_names:
            values = []
            for row in rows:
                val = row.get(col)
                if val is not None:
                    try:
                        values.append(float(val))
                    except (ValueError, TypeError):
                        pass  # Non-numeric, skip

            if values:
                stats[f"{col}_sum"] = sum(values)
                stats[f"{col}_avg"] = sum(values) / len(values)
                stats[f"{col}_min"] = min(values)
                stats[f"{col}_max"] = max(values)
                stats[f"{col}_count"] = len(values)

        return stats

    def _build_answer(
        self,
        rows: list[dict[str, Any]],
        column_names: list[str],
        computed: dict[str, Any],
        question: str,
    ) -> str:
        """
        Build a narration answer using computed values only.

        Never references values that aren't in `computed` or `rows`.
        """
        row_count = computed["row_count"]

        if row_count == 0:
            return "The query returned no results."

        if row_count == 1 and len(column_names) == 1:
            # Single scalar result
            col = column_names[0]
            value = rows[0].get(col)
            return f"The result is {self._format_value(value)}."

        if row_count == 1:
            # Single row, multiple columns
            parts = []
            for col in column_names:
                value = rows[0].get(col)
                parts.append(f"{col}: {self._format_value(value)}")
            return "Result: " + ", ".join(parts) + "."

        # Multiple rows — summarize
        parts = [f"The query returned {row_count} rows."]

        # Add per-column summaries for numeric columns
        for col in column_names:
            sum_key = f"{col}_sum"
            if sum_key in computed:
                avg = computed.get(f"{col}_avg", 0)
                total = computed[sum_key]
                parts.append(
                    f"  {col}: total={self._format_value(total)}, "
                    f"avg={self._format_value(avg)}"
                )

        return " ".join(parts)

    def _format_value(self, value: Any) -> str:
        """Format a value for display — numbers with proper formatting."""
        if value is None:
            return "NULL"
        if isinstance(value, float):
            if value == int(value):
                return f"{int(value):,}"
            return f"{value:,.2f}"
        if isinstance(value, int):
            return f"{value:,}"
        return str(value)
