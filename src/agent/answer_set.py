"""
Honest Answer Set mode — surfaces a calibrated set of plausible interpretations
instead of forcing a single, potentially wrong SQL query on ambiguous or medium-confidence questions.

Per Conformal Prediction & calibrated selective classification principles:
When epistemic or aleatoric uncertainty is non-negligible, presenting the bounded
hypothesis set preserves user trust and enables rapid human-in-the-loop selection.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from src.agent.uncertainty import UncertaintyDecomposition

logger = logging.getLogger(__name__)


@dataclass
class SQLCandidate:
    """A distinct valid interpretation and SQL query candidate."""
    title: str
    sql: str
    description: str
    probability: float  # 0.0 - 1.0
    assumptions: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    column_names: list[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    error: str | None = None

    @property
    def is_successful(self) -> bool:
        return self.error is None and len(self.rows) >= 0


@dataclass
class HonestAnswerSet:
    """A collection of candidate interpretations with probabilities."""
    question: str
    is_ambiguous: bool
    candidates: list[SQLCandidate] = field(default_factory=list)
    recommended_index: int = 0
    ambiguity_reason: str = ""

    @property
    def has_alternatives(self) -> bool:
        return len(self.candidates) > 1

    def best_candidate(self) -> SQLCandidate | None:
        if not self.candidates:
            return None
        if 0 <= self.recommended_index < len(self.candidates):
            return self.candidates[self.recommended_index]
        return self.candidates[0]

    def format_display(self) -> str:
        """Render a clean, formatted comparison of interpretations."""
        if not self.candidates:
            return "No candidates generated."

        lines = []
        if self.is_ambiguous:
            lines.append(f"⚠️ Ambiguity detected: {self.ambiguity_reason}")
            lines.append(f"Found {len(self.candidates)} plausible interpretations:\n")

        for idx, cand in enumerate(self.candidates, 1):
            marker = " (Recommended)" if idx - 1 == self.recommended_index else ""
            lines.append(f"Option {idx}: {cand.title} {marker}".strip())
            lines.append(f"  • Explanation: {cand.description}")
            lines.append(f"  • Probability: {cand.probability:.0%}")
            if cand.assumptions:
                lines.append(f"  • Assumptions: {', '.join(cand.assumptions)}")
            if cand.rows:
                lines.append(f"  • Returned: {cand.row_count} row(s)")
            lines.append(f"  • SQL: {cand.sql}\n")

        return "\n".join(lines).strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "is_ambiguous": self.is_ambiguous,
            "ambiguity_reason": self.ambiguity_reason,
            "recommended_index": self.recommended_index,
            "candidates": [
                {
                    "title": c.title,
                    "sql": c.sql,
                    "description": c.description,
                    "probability": round(c.probability, 3),
                    "assumptions": c.assumptions,
                    "row_count": c.row_count,
                    "error": c.error,
                }
                for c in self.candidates
            ],
        }


class AnswerSetGenerator:
    """
    Evaluates whether an honest answer set is warranted and synthesizes candidates.
    """

    def __init__(
        self,
        medium_confidence_min: float = 0.35,
        medium_confidence_max: float = 0.75,
    ):
        self.medium_confidence_min = medium_confidence_min
        self.medium_confidence_max = medium_confidence_max

    def should_generate_answer_set(
        self,
        confidence_score: float,
        uncertainty: UncertaintyDecomposition | None = None,
        is_ambiguous: bool = False,
    ) -> bool:
        """
        Determine if honest answer set mode should be triggered.
        Triggers when:
        1. Query is explicitly flagged as ambiguous, OR
        2. Confidence score falls in the medium/uncertainty band (e.g. 0.35 - 0.75), OR
        3. Uncertainty decomposition has multiple candidate options.
        """
        if is_ambiguous:
            return True

        if self.medium_confidence_min <= confidence_score <= self.medium_confidence_max:
            return True

        if uncertainty:
            # If dominant dimension has multiple candidate options
            dims = [
                uncertainty.schema_linking,
                uncertainty.join_path,
                uncertainty.aggregation,
                uncertainty.value_grounding,
            ]
            for d in dims:
                if len(d.candidate_options) >= 2 and d.uncertainty >= 0.35:
                    return True

        return False

    def build_answer_set(
        self,
        question: str,
        primary_sql: str,
        uncertainty: UncertaintyDecomposition | None = None,
        alternative_sqls: list[dict[str, Any]] | None = None,
        executor=None,
    ) -> HonestAnswerSet:
        """
        Construct an HonestAnswerSet from the primary query and alternatives.
        Executes alternatives safely if executor is provided.
        """
        candidates: list[SQLCandidate] = []

        # 1. Primary candidate
        primary_prob = uncertainty.composite_confidence if uncertainty else 0.70
        primary_assumptions = []
        if uncertainty and uncertainty.dominant_dimension != "schema_linking":
            primary_assumptions.append(f"Assumed {uncertainty.dominant_dimension} resolution")

        primary_cand = SQLCandidate(
            title="Primary Interpretation",
            sql=primary_sql,
            description="Direct translation matching strongest schema signals",
            probability=max(0.5, min(0.9, primary_prob)),
            assumptions=primary_assumptions,
        )
        candidates.append(primary_cand)

        # 2. Alternative candidates if provided
        if alternative_sqls:
            remaining_prob = 1.0 - primary_cand.probability
            per_alt_prob = round(remaining_prob / max(len(alternative_sqls), 1), 2)
            for alt in alternative_sqls:
                cand = SQLCandidate(
                    title=alt.get("title", "Alternative Interpretation"),
                    sql=alt.get("sql", ""),
                    description=alt.get("description", "Alternative query interpretation"),
                    probability=alt.get("probability", per_alt_prob),
                    assumptions=alt.get("assumptions", []),
                )
                if cand.sql and cand.sql != primary_sql:
                    candidates.append(cand)

        # 3. Rule-derived alternative if uncertainty flagged specific dimension alternatives
        if len(candidates) == 1 and uncertainty:
            alt_cand = self._derive_alternative_from_uncertainty(
                question=question,
                primary_sql=primary_sql,
                uncertainty=uncertainty,
            )
            if alt_cand:
                # Adjust probabilities so sum <= 1.0
                primary_cand.probability = 0.60
                alt_cand.probability = 0.40
                candidates.append(alt_cand)

        # 4. Optional Execution of candidates (Read-Only)
        if executor and len(candidates) > 1:
            for c in candidates:
                if not c.rows and c.sql:
                    try:
                        res = executor.execute(c.sql)
                        if res.error:
                            c.error = res.error
                        else:
                            c.rows = res.rows
                            c.row_count = res.row_count
                            c.column_names = res.column_names
                            c.execution_time_ms = res.execution_time_ms
                    except Exception as e:
                        c.error = str(e)

        ambiguity_reason = ""
        if uncertainty and uncertainty.needs_clarification and uncertainty.clarification:
            ambiguity_reason = uncertainty.clarification.rationale or uncertainty.clarification.question

        return HonestAnswerSet(
            question=question,
            is_ambiguous=len(candidates) > 1,
            candidates=candidates,
            recommended_index=0,
            ambiguity_reason=ambiguity_reason,
        )

    def _derive_alternative_from_uncertainty(
        self,
        question: str,
        primary_sql: str,
        uncertainty: UncertaintyDecomposition,
    ) -> SQLCandidate | None:
        """Heuristically derive alternative candidate based on dominant uncertainty."""
        dominant = uncertainty.dominant_dimension

        if dominant == "aggregation":
            # If primary used COUNT(*), alternative might be SUM or detailed list
            if "COUNT(" in primary_sql.upper() and ("sum" in question.lower() or "amount" in question.lower() or "total" in question.lower()):
                alt_sql = primary_sql.upper().replace("COUNT(*)", "SUM(Total)").replace("COUNT(1)", "SUM(Total)")
                return SQLCandidate(
                    title="By Monetary Total (SUM)",
                    sql=alt_sql,
                    description="Calculates sum of values instead of transaction count",
                    probability=0.40,
                    assumptions=["User asked for sum/total rather than raw count"],
                )
        elif dominant == "join_path":
            if "INNER JOIN" in primary_sql.upper():
                alt_sql = primary_sql.replace("INNER JOIN", "LEFT JOIN").replace("inner join", "left join")
                return SQLCandidate(
                    title="Inclusive Join (LEFT JOIN)",
                    sql=alt_sql,
                    description="Includes records even if related joined rows are missing",
                    probability=0.35,
                    assumptions=["Preserve records with null relations"],
                )

        return None
