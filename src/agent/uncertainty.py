"""
Uncertainty decomposition — decomposes confidence/uncertainty into 4 orthogonal
relational dimensions:
1. Schema linking uncertainty (ambiguous entities, fuzzy matches)
2. Join path uncertainty (multiple valid paths, synthetic joins, missing FKs)
3. Aggregation uncertainty (grouping ambiguity, metric vs dimension confusion)
4. Value grounding uncertainty (unverified literal filter values, status codes)

Also generates targeted, high-information-gain clarifying questions when
uncertainty exceeds actionable thresholds.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import sqlglot
from sqlglot import exp

logger = logging.getLogger(__name__)


class UncertaintyDimension(Enum):
    SCHEMA_LINKING = "schema_linking"
    JOIN_PATH = "join_path"
    AGGREGATION = "aggregation"
    VALUE_GROUNDING = "value_grounding"


@dataclass
class DimensionUncertainty:
    """Uncertainty analysis for a single orthogonal dimension."""
    dimension: str
    uncertainty: float  # 0.0 (completely certain) to 1.0 (completely uncertain)
    confidence: float   # 1.0 - uncertainty
    reasons: list[str] = field(default_factory=list)
    candidate_options: list[str] = field(default_factory=list)


@dataclass
class TargetedClarification:
    """Actionable clarifying question designed to maximize information gain."""
    dimension: str
    question: str
    options: list[str] = field(default_factory=list)
    recommended_option: str | None = None
    rationale: str = ""


@dataclass
class UncertaintyDecomposition:
    """Full decomposed uncertainty across all relational dimensions."""
    schema_linking: DimensionUncertainty
    join_path: DimensionUncertainty
    aggregation: DimensionUncertainty
    value_grounding: DimensionUncertainty
    composite_uncertainty: float
    composite_confidence: float
    dominant_dimension: str
    needs_clarification: bool = False
    clarification: TargetedClarification | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "composite_uncertainty": round(self.composite_uncertainty, 3),
            "composite_confidence": round(self.composite_confidence, 3),
            "dominant_dimension": self.dominant_dimension,
            "dimensions": {
                "schema_linking": {
                    "uncertainty": round(self.schema_linking.uncertainty, 3),
                    "reasons": self.schema_linking.reasons,
                },
                "join_path": {
                    "uncertainty": round(self.join_path.uncertainty, 3),
                    "reasons": self.join_path.reasons,
                },
                "aggregation": {
                    "uncertainty": round(self.aggregation.uncertainty, 3),
                    "reasons": self.aggregation.reasons,
                },
                "value_grounding": {
                    "uncertainty": round(self.value_grounding.uncertainty, 3),
                    "reasons": self.value_grounding.reasons,
                },
            },
            "needs_clarification": self.needs_clarification,
            "clarification": {
                "question": self.clarification.question,
                "options": self.clarification.options,
            } if self.clarification else None,
        }


class UncertaintyDecomposer:
    """
    Decomposes uncertainty for a Text-to-SQL generation pair.
    
    Examines:
    - User natural language question
    - Generated SQL query AST
    - Retrieved schema context & known database schema
    """

    def __init__(
        self,
        clarification_threshold: float = 0.45,
        dimension_weights: dict[str, float] | None = None,
    ):
        self.clarification_threshold = clarification_threshold
        self.dimension_weights = dimension_weights or {
            "schema_linking": 0.35,
            "join_path": 0.25,
            "aggregation": 0.20,
            "value_grounding": 0.20,
        }

    def decompose(
        self,
        question: str,
        sql: str | None,
        schema_tables: dict[str, list[str]] | None = None,
        foreign_keys: list[dict[str, Any]] | None = None,
        retrieved_tables: list[str] | None = None,
    ) -> UncertaintyDecomposition:
        """
        Decompose uncertainty into the 4 dimensions and propose targeted clarification if needed.
        """
        schema_tables = schema_tables or {}
        foreign_keys = foreign_keys or []
        retrieved_tables = retrieved_tables or []

        if not sql or not sql.strip():
            # Total uncertainty across all dimensions
            empty_dim = lambda name: DimensionUncertainty(
                dimension=name,
                uncertainty=1.0,
                confidence=0.0,
                reasons=["No SQL query generated"],
            )
            return UncertaintyDecomposition(
                schema_linking=empty_dim("schema_linking"),
                join_path=empty_dim("join_path"),
                aggregation=empty_dim("aggregation"),
                value_grounding=empty_dim("value_grounding"),
                composite_uncertainty=1.0,
                composite_confidence=0.0,
                dominant_dimension="schema_linking",
                needs_clarification=True,
                clarification=TargetedClarification(
                    dimension="schema_linking",
                    question="Could you please specify which table or data you would like to query?",
                    options=list(schema_tables.keys())[:5],
                ),
            )

        # Parse AST
        try:
            parsed = sqlglot.parse_one(sql)
        except Exception as e:
            logger.warning(f"Could not parse SQL for uncertainty analysis: {e}")
            parsed = None

        # 1. Schema Linking Uncertainty
        schema_linking = self._evaluate_schema_linking(
            question=question,
            sql=sql,
            parsed=parsed,
            schema_tables=schema_tables,
            retrieved_tables=retrieved_tables,
        )

        # 2. Join Path Uncertainty
        join_path = self._evaluate_join_path(
            parsed=parsed,
            foreign_keys=foreign_keys,
            schema_tables=schema_tables,
        )

        # 3. Aggregation Uncertainty
        aggregation = self._evaluate_aggregation(
            question=question,
            parsed=parsed,
        )

        # 4. Value Grounding Uncertainty
        value_grounding = self._evaluate_value_grounding(
            question=question,
            parsed=parsed,
        )

        # Calculate composite score
        weights = self.dimension_weights
        composite_unc = (
            schema_linking.uncertainty * weights.get("schema_linking", 0.35)
            + join_path.uncertainty * weights.get("join_path", 0.25)
            + aggregation.uncertainty * weights.get("aggregation", 0.20)
            + value_grounding.uncertainty * weights.get("value_grounding", 0.20)
        )
        composite_unc = max(0.0, min(1.0, composite_unc))
        composite_conf = round(1.0 - composite_unc, 3)

        # Find dominant uncertainty dimension
        dims = [schema_linking, join_path, aggregation, value_grounding]
        dominant = max(dims, key=lambda d: d.uncertainty)

        # Formulate Targeted Clarification if dominant uncertainty > threshold
        needs_clarification = dominant.uncertainty >= self.clarification_threshold
        clarification = None
        if needs_clarification:
            clarification = self._build_clarification(dominant, question, schema_tables)

        return UncertaintyDecomposition(
            schema_linking=schema_linking,
            join_path=join_path,
            aggregation=aggregation,
            value_grounding=value_grounding,
            composite_uncertainty=round(composite_unc, 3),
            composite_confidence=composite_conf,
            dominant_dimension=dominant.dimension,
            needs_clarification=needs_clarification,
            clarification=clarification,
        )

    def _evaluate_schema_linking(
        self,
        question: str,
        sql: str,
        parsed: exp.Expression | None,
        schema_tables: dict[str, list[str]],
        retrieved_tables: list[str],
    ) -> DimensionUncertainty:
        reasons = []
        candidates = []
        uncertainty = 0.0

        if not parsed:
            return DimensionUncertainty(
                dimension="schema_linking",
                uncertainty=0.6,
                confidence=0.4,
                reasons=["SQL syntax unparseable for schema linking verification"],
            )

        tables_in_sql = [t.name for t in parsed.find_all(exp.Table) if t.name]
        
        # Check for unknown / ungrounded tables
        if schema_tables:
            unknown_tables = [t for t in tables_in_sql if t.lower() not in [k.lower() for k in schema_tables]]
            if unknown_tables:
                uncertainty += 0.5 * len(unknown_tables)
                reasons.append(f"Tables not found in known schema: {', '.join(unknown_tables)}")

        # Check for ambiguous question concepts mapping to multiple tables
        q_lower = question.lower()
        ambiguous_matches = []
        for tbl in schema_tables:
            if tbl.lower() in q_lower and tbl not in tables_in_sql:
                ambiguous_matches.append(tbl)
        
        if ambiguous_matches:
            uncertainty += 0.25
            reasons.append(f"Question mentions concepts matching alternative tables: {', '.join(ambiguous_matches)}")
            candidates.extend(ambiguous_matches)

        # Column-level check: unqualified column names when multiple tables are joined
        if len(tables_in_sql) > 1:
            columns = list(parsed.find_all(exp.Column))
            unqualified = [c.name for c in columns if not c.table]
            if len(unqualified) > 2:
                uncertainty += 0.2
                reasons.append(f"Multiple tables joined with unqualified column references ({len(unqualified)} columns)")

        uncertainty = min(1.0, max(0.0, uncertainty))
        return DimensionUncertainty(
            dimension="schema_linking",
            uncertainty=round(uncertainty, 3),
            confidence=round(1.0 - uncertainty, 3),
            reasons=reasons,
            candidate_options=candidates,
        )

    def _evaluate_join_path(
        self,
        parsed: exp.Expression | None,
        foreign_keys: list[dict[str, Any]],
        schema_tables: dict[str, list[str]],
    ) -> DimensionUncertainty:
        reasons = []
        candidates = []
        uncertainty = 0.0

        if not parsed:
            return DimensionUncertainty(dimension="join_path", uncertainty=0.2, confidence=0.8)

        tables = [t.name for t in parsed.find_all(exp.Table) if t.name]
        joins = list(parsed.find_all(exp.Join))
        where_clause = parsed.find(exp.Where)

        if len(tables) > 1 and not joins and not where_clause:
            # Potential cartesian product
            uncertainty = 0.95
            reasons.append("Multiple tables referenced without explicit JOIN or WHERE conditions (Cartesian join)")
            return DimensionUncertainty(dimension="join_path", uncertainty=uncertainty, confidence=0.05, reasons=reasons)

        if joins:
            # Check for join predicates
            for j in joins:
                on_clause = j.args.get("on")
                if not on_clause and j.kind != "CROSS":
                    if not where_clause:
                        uncertainty += 0.8
                        reasons.append("Join without explicit ON or WHERE condition detected (Cartesian join)")
                    else:
                        uncertainty += 0.4
                        reasons.append("Join without explicit ON condition detected (relies on WHERE)")

            # Check if joins match known foreign keys
            if foreign_keys:
                known_pairs = set()
                for fk in foreign_keys:
                    from_tbl = fk.get("from_table", "").lower()
                    to_tbl = fk.get("to_table", "").lower()
                    if from_tbl and to_tbl:
                        known_pairs.add((from_tbl, to_tbl))
                        known_pairs.add((to_tbl, from_tbl))

                for j in joins:
                    join_tbl = j.this.name if hasattr(j.this, "name") else ""
                    if join_tbl and tables:
                        primary_tbl = tables[0].lower()
                        j_tbl_lower = join_tbl.lower()
                        if (primary_tbl, j_tbl_lower) not in known_pairs and len(tables) > 2:
                            uncertainty += 0.25
                            reasons.append(f"Join between '{primary_tbl}' and '{join_tbl}' is not a declared foreign key path")

        uncertainty = min(1.0, max(0.0, uncertainty))
        return DimensionUncertainty(
            dimension="join_path",
            uncertainty=round(uncertainty, 3),
            confidence=round(1.0 - uncertainty, 3),
            reasons=reasons,
            candidate_options=candidates,
        )

    def _evaluate_aggregation(
        self,
        question: str,
        parsed: exp.Expression | None,
    ) -> DimensionUncertainty:
        reasons = []
        candidates = []
        uncertainty = 0.0

        if not parsed:
            return DimensionUncertainty(dimension="aggregation", uncertainty=0.1, confidence=0.9)

        q_lower = question.lower()
        has_agg_sql = bool(list(parsed.find_all(exp.AggFunc)))
        has_group_by = bool(parsed.find(exp.Group))
        has_order_by = bool(parsed.find(exp.Order))
        has_limit = bool(parsed.find(exp.Limit))

        # Words implying aggregation
        agg_words = ["total", "sum", "average", "avg", "count", "how many", "number of", "most", "least", "top", "highest", "lowest"]
        implied_agg = any(w in q_lower for w in agg_words)

        if implied_agg and not has_agg_sql:
            uncertainty += 0.45
            reasons.append("Question implies aggregation ('total', 'how many', 'top') but SQL does not use aggregate functions")
            candidates.extend(["COUNT(*)", "SUM(...)", "AVG(...)"])

        # Top-N / Extreme ambiguity: "top" usually implies ORDER BY + LIMIT
        top_words = ["top", "highest", "best", "most", "lowest", "least"]
        if any(w in q_lower for w in top_words):
            if not has_order_by:
                uncertainty += 0.35
                reasons.append("Ranking question missing ORDER BY clause")
            if not has_limit and not has_agg_sql:
                uncertainty += 0.25
                reasons.append("Superlative question missing LIMIT constraint")

        # Group By check: aggregate + non-aggregate in SELECT without GROUP BY
        selects = list(parsed.find_all(exp.Select))
        if selects:
            for s in selects:
                exprs = s.expressions
                has_agg_expr = any(list(e.find_all(exp.AggFunc)) for e in exprs)
                has_non_agg_col = any(isinstance(e, exp.Column) for e in exprs)
                if has_agg_expr and has_non_agg_col and not has_group_by:
                    uncertainty += 0.5
                    reasons.append("Mix of aggregate and scalar columns without explicit GROUP BY")

        uncertainty = min(1.0, max(0.0, uncertainty))
        return DimensionUncertainty(
            dimension="aggregation",
            uncertainty=round(uncertainty, 3),
            confidence=round(1.0 - uncertainty, 3),
            reasons=reasons,
            candidate_options=candidates,
        )

    def _evaluate_value_grounding(
        self,
        question: str,
        parsed: exp.Expression | None,
    ) -> DimensionUncertainty:
        reasons = []
        candidates = []
        uncertainty = 0.0

        if not parsed:
            return DimensionUncertainty(dimension="value_grounding", uncertainty=0.1, confidence=0.9)

        # Extract literals in SQL WHERE / HAVING clauses
        literals = list(parsed.find_all(exp.Literal))
        q_lower = question.lower()

        for lit in literals:
            val_str = str(lit.this).strip("'\"")
            # If literal is a string and does not appear verbatim in question
            if lit.is_string:
                if val_str.lower() not in q_lower:
                    uncertainty += 0.3
                    reasons.append(f"Filter literal '{val_str}' was inferred and does not appear verbatim in question")
                    candidates.append(val_str)

        # Check for status codes / boolean numeric guesses (e.g. status = 1 vs 'active')
        for eq in parsed.find_all(exp.EQ):
            right = eq.right
            if isinstance(right, exp.Literal) and right.is_number:
                left_col = eq.left.name if hasattr(eq.left, "name") else ""
                if left_col and any(term in left_col.lower() for term in ["status", "state", "type", "flag"]):
                    uncertainty += 0.2
                    reasons.append(f"Numeric code {right.this} used for column '{left_col}' without explicit enum confirmation")

        uncertainty = min(1.0, max(0.0, uncertainty))
        return DimensionUncertainty(
            dimension="value_grounding",
            uncertainty=round(uncertainty, 3),
            confidence=round(1.0 - uncertainty, 3),
            reasons=reasons,
            candidate_options=candidates,
        )

    def _build_clarification(
        self,
        dominant_dim: DimensionUncertainty,
        question: str,
        schema_tables: dict[str, list[str]],
    ) -> TargetedClarification:
        dim = dominant_dim.dimension

        if dim == "schema_linking":
            options = dominant_dim.candidate_options or list(schema_tables.keys())[:4]
            return TargetedClarification(
                dimension=dim,
                question="Which entity or table should this query focus on?",
                options=options,
                rationale="; ".join(dominant_dim.reasons),
            )
        elif dim == "join_path":
            return TargetedClarification(
                dimension=dim,
                question="How should these records be related or combined?",
                options=["Direct association only (INNER JOIN)", "Include unmatched records (LEFT JOIN)"],
                rationale="; ".join(dominant_dim.reasons),
            )
        elif dim == "aggregation":
            return TargetedClarification(
                dimension=dim,
                question="What summary metric would you like to compute?",
                options=["Total sum", "Average", "Distinct count", "Individual row list"],
                recommended_option="Total sum",
                rationale="; ".join(dominant_dim.reasons),
            )
        else:  # value_grounding
            options = dominant_dim.candidate_options or ["Exact match", "Partial match", "All active records"]
            return TargetedClarification(
                dimension=dim,
                question=f"Which specific value or status should be filtered?",
                options=options,
                rationale="; ".join(dominant_dim.reasons),
            )
