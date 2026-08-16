"""
Intent reconstructor — restates the user's question in plain language
against the glossary, decides if clarification is needed.

Per PRD.md G1: Reconstruct user intent before running anything; ask a
clarifying question when the request is genuinely ambiguous.

Per Architecture.md §3: Routes through llm_client when LLM-based
restatement is needed. Ambiguity detection is rule-based.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.agent.glossary import GlossaryStore, GlossaryTerm
from src.agent.llm_client import LLMClient as UnifiedLLMClient


class AmbiguityType(Enum):
    """Types of ambiguity detected in a user question."""
    UNCLEAR_METRIC = "unclear_metric"         # "revenue" without specifying which definition
    MISSING_FILTER = "missing_filter"         # "top users" without time range or criteria
    AMBIGUOUS_ENTITY = "ambiguous_entity"     # "orders" when multiple order-like tables exist
    CONFLICTING_TERMS = "conflicting_terms"   # Multiple glossary definitions for same term
    VAGUE_AGGREGATION = "vague_aggregation"   # "how many" without specifying what to count
    UNCLEAR_TIMEFRAME = "unclear_timeframe"   # Question implies time but doesn't specify range


@dataclass
class AmbiguityFlag:
    """A detected ambiguity in a user question."""
    ambiguity_type: AmbiguityType
    description: str
    suggested_clarification: str


@dataclass
class IntentResult:
    """Result of intent reconstruction."""
    original_question: str
    is_ambiguous: bool
    ambiguities: list[AmbiguityFlag] = field(default_factory=list)
    matched_glossary_terms: list[GlossaryTerm] = field(default_factory=list)
    clarifying_question: str | None = None
    restatement: str | None = None  # Plain-language restatement

    def needs_clarification(self) -> bool:
        """Whether the agent should ask the user before proceeding."""
        return self.is_ambiguous and len(self.ambiguities) > 0


class IntentReconstructor:
    """
    Analyzes user questions for ambiguity and reconstructs intent.

    Ambiguity detection is rule-based for common patterns;
    the full intent restatement uses the glossary for term resolution.
    """

    # Vague quantifier patterns that need clarification
    _VAGUE_PATTERNS = [
        (r'\btop\b(?!\s+\d)', AmbiguityType.MISSING_FILTER, "top by what criteria?"),
        (r'\bbest\b', AmbiguityType.MISSING_FILTER, "best by what metric?"),
        (r'\brecent\b', AmbiguityType.UNCLEAR_TIMEFRAME, "how recent? (last week, month, year?)"),
        (r'\blately\b', AmbiguityType.UNCLEAR_TIMEFRAME, "what time range?"),
        (r'\ba lot\b', AmbiguityType.VAGUE_AGGREGATION, "what threshold counts as 'a lot'?"),
        (r'\bhow many\b\s*$', AmbiguityType.VAGUE_AGGREGATION, "count of what exactly?"),
    ]

    def __init__(
        self,
        glossary: GlossaryStore | None = None,
        llm_client: UnifiedLLMClient | None = None,
    ):
        self._glossary = glossary or GlossaryStore()
        self._llm_client = llm_client  # Used for LLM-based restatement when available

    def analyze(self, question: str) -> IntentResult:
        """
        Analyze a user question for ambiguity and resolve glossary terms.

        Returns an IntentResult with ambiguity flags and matched terms.
        """
        ambiguities: list[AmbiguityFlag] = []

        # 1. Check for vague patterns
        for pattern, amb_type, suggestion in self._VAGUE_PATTERNS:
            if re.search(pattern, question, re.IGNORECASE):
                ambiguities.append(AmbiguityFlag(
                    ambiguity_type=amb_type,
                    description=f"Detected vague term matching: {pattern}",
                    suggested_clarification=suggestion,
                ))

        # 2. Resolve glossary terms
        matched_terms = self._glossary.lookup(question)

        # 3. Check for conflicting glossary definitions
        seen_terms: dict[str, list[GlossaryTerm]] = {}
        for term in matched_terms:
            seen_terms.setdefault(term.term.lower(), []).append(term)

        for term_name, definitions in seen_terms.items():
            if len(definitions) > 1:
                ambiguities.append(AmbiguityFlag(
                    ambiguity_type=AmbiguityType.CONFLICTING_TERMS,
                    description=f"Multiple definitions for '{term_name}'",
                    suggested_clarification=(
                        f"'{term_name}' has {len(definitions)} definitions. "
                        f"Which do you mean? "
                        + " vs. ".join(d.definition for d in definitions)
                    ),
                ))

        # 4. Detect ambiguous metrics
        metric_words = ["revenue", "profit", "sales", "growth", "churn",
                        "retention", "engagement", "conversion"]
        for word in metric_words:
            if word in question.lower():
                # If no glossary term matches, the metric is undefined
                if not any(t.term.lower() == word for t in matched_terms):
                    ambiguities.append(AmbiguityFlag(
                        ambiguity_type=AmbiguityType.UNCLEAR_METRIC,
                        description=f"'{word}' used but not defined in glossary",
                        suggested_clarification=(
                            f"How is '{word}' defined for your organization? "
                            f"(No glossary entry found.)"
                        ),
                    ))

        is_ambiguous = len(ambiguities) > 0

        # Build clarifying question if needed
        clarifying_question = None
        if is_ambiguous:
            clarifying_question = self._build_clarifying_question(
                question, ambiguities
            )

        return IntentResult(
            original_question=question,
            is_ambiguous=is_ambiguous,
            ambiguities=ambiguities,
            matched_glossary_terms=matched_terms,
            clarifying_question=clarifying_question,
        )

    def _build_clarifying_question(
        self,
        original: str,
        ambiguities: list[AmbiguityFlag],
    ) -> str:
        """Build a clarifying question from detected ambiguities."""
        parts = [f"Before I answer \"{original}\", I need to clarify:"]
        for i, amb in enumerate(ambiguities, 1):
            parts.append(f"{i}. {amb.suggested_clarification}")
        return "\n".join(parts)

    def build_restatement_prompt(
        self,
        question: str,
        glossary_context: str,
        schema_context: str,
    ) -> str:
        """
        Build the LLM prompt for intent restatement.

        The LLM will restate the question in plain language,
        incorporating glossary definitions and schema context.
        """
        return f"""Restate the following user question in precise, unambiguous language
that maps directly to database concepts. Use the glossary definitions and schema
information provided.

User question: {question}

{glossary_context}

Available schema:
{schema_context}

Instructions:
- Restate what the user is asking in one clear sentence
- Reference specific table and column names from the schema
- Apply any relevant glossary definitions
- If the question is still ambiguous after applying glossary terms, say so explicitly
- Do NOT generate SQL — just restate the intent

Restatement:"""
