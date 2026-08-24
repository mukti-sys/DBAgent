"""
Multi-turn conversational state — tracks session-scoped filters,
context, and references across sequential user queries.

Per Phases.md Phase 11:
- Session-scoped filter/context tracking
- Graceful degradation (ask, don't assume) when state is uncertain
- Exit criteria: "now break that down by region" correctly inherits
  prior filters in eval test cases

Per Rules.md §4: Ask-don't-assume for ambiguous context.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class QueryContext:
    """Context from a single query in the conversation."""
    question: str
    sql: str | None = None
    filters: dict[str, Any] = field(default_factory=dict)
    tables_used: list[str] = field(default_factory=list)
    columns_referenced: list[str] = field(default_factory=list)
    aggregation: str | None = None  # "count", "sum", "avg", etc.
    time_range: str | None = None
    result_summary: str | None = None


@dataclass
class ContextInheritanceResult:
    """Result of attempting to inherit context from prior turns."""
    inherited_filters: dict[str, Any]
    inherited_tables: list[str]
    inherited_aggregation: str | None
    inherited_time_range: str | None
    is_follow_up: bool
    is_ambiguous: bool
    ambiguity_reason: str | None = None
    clarifying_question: str | None = None


class ConversationState:
    """
    Manages multi-turn conversation context with filter/context inheritance.

    Key principle: inherit context from prior turns when the current question
    is clearly a follow-up, but ASK (never assume) when the reference is
    ambiguous or context is uncertain.
    """

    # Patterns that indicate a follow-up question
    _FOLLOW_UP_PATTERNS = [
        r'\bnow\b',
        r'\balso\b',
        r'\band\b\s+(?:also|what|how|show|break|group|filter|sort)',
        r'\bbreak\s+(?:that|this|it)\s+down\b',
        r'\bgroup\s+(?:that|this|it)\s+by\b',
        r'\bfilter\s+(?:that|this|it|those)\b',
        r'\bsort\s+(?:that|this|it|those)\b',
        r'\bwhat\s+about\b',
        r'\bhow\s+about\b',
        r'\bsame\s+(?:thing|query|question)\b',
        r'\bbut\s+(?:for|with|only|just)\b',
        r'\binstead\b',
        r'\bnarrow\s+(?:that|this|it)\s+down\b',
        r'\bexclude\b',
        r'\binclude\s+only\b',
    ]

    # Patterns indicating a fresh/independent question
    _FRESH_QUESTION_PATTERNS = [
        r'^(?:what|how\s+many|show\s+me|list|find|get)\s',
        r'\btotal\s+(?:number|count|amount)\b',
    ]

    # Breakdown patterns: "break it down by X"
    _BREAKDOWN_PATTERN = re.compile(
        r'break\s+(?:that|this|it)\s+down\s+by\s+(\w+)',
        re.IGNORECASE,
    )

    _GROUP_BY_PATTERN = re.compile(
        r'group\s+(?:that|this|it)\s+by\s+(\w+)',
        re.IGNORECASE,
    )

    def __init__(self, max_history: int = 10):
        self._history: list[QueryContext] = []
        self._max_history = max_history
        self._session_filters: dict[str, Any] = {}

    @property
    def turn_count(self) -> int:
        return len(self._history)

    @property
    def last_context(self) -> QueryContext | None:
        return self._history[-1] if self._history else None

    @property
    def session_filters(self) -> dict[str, Any]:
        return dict(self._session_filters)

    def add_turn(self, context: QueryContext) -> None:
        """Record a completed query turn."""
        self._history.append(context)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        # Update session filters from this turn's filters
        for key, value in context.filters.items():
            self._session_filters[key] = value

    def resolve_context(self, question: str) -> ContextInheritanceResult:
        """
        Determine whether the current question is a follow-up and what
        context to inherit from prior turns.

        Returns ContextInheritanceResult with inherited context or
        a clarifying question if the reference is ambiguous.
        """
        if not self._history:
            return ContextInheritanceResult(
                inherited_filters={},
                inherited_tables=[],
                inherited_aggregation=None,
                inherited_time_range=None,
                is_follow_up=False,
                is_ambiguous=False,
            )

        is_follow_up = self._detect_follow_up(question)

        if not is_follow_up:
            return ContextInheritanceResult(
                inherited_filters={},
                inherited_tables=[],
                inherited_aggregation=None,
                inherited_time_range=None,
                is_follow_up=False,
                is_ambiguous=False,
            )

        # It's a follow-up — try to inherit context
        last = self._history[-1]

        # Check for breakdown request
        breakdown_col = self._extract_breakdown_column(question)

        # Check for ambiguity: multiple prior contexts that could be referenced
        is_ambiguous = False
        ambiguity_reason = None
        clarifying_question = None

        if len(self._history) > 1 and self._is_ambiguous_reference(question):
            is_ambiguous = True
            ambiguity_reason = (
                f"Multiple prior queries could be referenced. "
                f"Last query was about '{last.question}', but there are "
                f"{len(self._history)} prior turns."
            )
            clarifying_question = (
                f"I want to make sure I'm building on the right query. "
                f"Are you referring to your last question "
                f"(\"{self._truncate(last.question, 60)}\")?"
            )

        # Build inherited context from the most recent turn
        inherited_filters = dict(last.filters)
        # Also include accumulated session filters
        for key, value in self._session_filters.items():
            if key not in inherited_filters:
                inherited_filters[key] = value

        # Parse any new filters from the current question
        new_filters = self._extract_filters_from_question(question)
        inherited_filters.update(new_filters)

        return ContextInheritanceResult(
            inherited_filters=inherited_filters,
            inherited_tables=list(last.tables_used),
            inherited_aggregation=last.aggregation,
            inherited_time_range=last.time_range,
            is_follow_up=True,
            is_ambiguous=is_ambiguous,
            ambiguity_reason=ambiguity_reason,
            clarifying_question=clarifying_question,
        )

    def build_context_prompt(
        self,
        question: str,
        inheritance: ContextInheritanceResult,
    ) -> str:
        """Build a context-aware prompt incorporating inherited state."""
        parts = []

        if inheritance.is_follow_up and inheritance.inherited_filters:
            parts.append("Context from prior conversation:")
            for key, value in inheritance.inherited_filters.items():
                parts.append(f"  - {key}: {value}")
            parts.append("")

        if inheritance.inherited_tables:
            parts.append(f"Previously referenced tables: {', '.join(inheritance.inherited_tables)}")
            parts.append("")

        if inheritance.inherited_time_range:
            parts.append(f"Active time range: {inheritance.inherited_time_range}")
            parts.append("")

        parts.append(f"Current question: {question}")

        if inheritance.is_follow_up:
            parts.append("")
            parts.append(
                "Note: This is a follow-up question. Apply the inherited "
                "filters and context unless the user explicitly overrides them."
            )

        return "\n".join(parts)

    def clear(self) -> None:
        """Reset conversation state."""
        self._history = []
        self._session_filters = {}

    def _detect_follow_up(self, question: str) -> bool:
        """Detect whether the question is a follow-up to prior turns."""
        q_lower = question.lower().strip()

        # Check for explicit follow-up patterns
        for pattern in self._FOLLOW_UP_PATTERNS:
            if re.search(pattern, q_lower):
                return True

        # Check if it's clearly a fresh question
        for pattern in self._FRESH_QUESTION_PATTERNS:
            if re.search(pattern, q_lower):
                # Could still be follow-up if it also matches follow-up patterns
                return False

        # Short questions without clear subject often reference prior context
        if len(q_lower.split()) <= 4 and not q_lower.endswith("?"):
            return True

        return False

    def _extract_breakdown_column(self, question: str) -> str | None:
        """Extract the column name from 'break that down by X' patterns."""
        match = self._BREAKDOWN_PATTERN.search(question)
        if match:
            return match.group(1)
        match = self._GROUP_BY_PATTERN.search(question)
        if match:
            return match.group(1)
        return None

    def _is_ambiguous_reference(self, question: str) -> bool:
        """Check if the follow-up reference is ambiguous."""
        q_lower = question.lower()

        # Pronouns without clear antecedent in a long conversation
        pronoun_patterns = [r'\bthat\b', r'\bthose\b', r'\bthem\b', r'\bit\b']
        has_pronoun = any(re.search(p, q_lower) for p in pronoun_patterns)

        if has_pronoun and len(self._history) >= 3:
            # Multiple recent turns with different tables
            recent_tables = set()
            for ctx in self._history[-3:]:
                recent_tables.update(ctx.tables_used)
            if len(recent_tables) > 2:
                return True

        return False

    def _extract_filters_from_question(self, question: str) -> dict[str, Any]:
        """Extract explicit filter values from a question."""
        filters: dict[str, Any] = {}
        q_lower = question.lower()

        # "for <value>" patterns
        for_match = re.search(r'for\s+(\w+(?:\s+\w+)?)\s*$', q_lower)
        if for_match:
            filters["explicit_filter"] = for_match.group(1)

        # "in <region/location>" patterns
        in_match = re.search(r'\bin\s+([\w\s]+?)(?:\s*$|\s+and\b)', q_lower)
        if in_match:
            filters["location_filter"] = in_match.group(1).strip()

        # "by <column>" for grouping
        breakdown_col = self._extract_breakdown_column(question)
        if breakdown_col:
            filters["group_by"] = breakdown_col

        return filters

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        if len(text) <= max_len:
            return text
        return text[:max_len - 3] + "..."
