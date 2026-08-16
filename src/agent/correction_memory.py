"""
Correction memory — persists user corrections as rules,
scoped per (org_id, db_fingerprint). Applied on subsequent matching queries.

Per Architecture.md §5: Correction memory is keyed per (org_id, db_fingerprint)
so rules never leak across unrelated databases.

Per PRD.md §3 G6: Persist user corrections as org-specific rules so the
same mistake isn't repeated.

Per Rules.md §2: This is a best-effort mitigation — corrections improve
accuracy over time but don't eliminate errors.
"""

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CorrectionRule:
    """A single user-provided correction rule."""
    rule_id: str
    original_question: str        # The question that triggered the wrong answer
    original_sql: str             # The SQL that was generated (wrong)
    corrected_sql: str            # The corrected SQL the user provided
    correction_description: str   # What the user said was wrong
    keywords: list[str] = field(default_factory=list)  # For retrieval matching
    org_id: str = "default"
    db_fingerprint: str = "default"
    created_at: str = ""
    applied_count: int = 0

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.rule_id:
            self.rule_id = self._generate_id()

    def _generate_id(self) -> str:
        content = f"{self.original_question}:{self.corrected_sql}:{self.org_id}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def matches(self, question: str, threshold: float = 0.3) -> float:
        """
        Score how well this rule matches a given question.

        Uses keyword overlap as a simple matching heuristic.
        Returns a score between 0.0 and 1.0.
        """
        question_lower = question.lower()
        question_words = set(re.findall(r'\w+', question_lower))

        if not self.keywords:
            # Fall back to matching against original question words
            rule_words = set(re.findall(r'\w+', self.original_question.lower()))
        else:
            rule_words = {k.lower() for k in self.keywords}

        if not rule_words:
            return 0.0

        # Jaccard-style overlap
        overlap = question_words & rule_words
        union = question_words | rule_words

        if not union:
            return 0.0

        score = len(overlap) / len(union)

        # Boost if the original question is very similar
        if self.original_question.lower().strip() == question_lower.strip():
            score = 1.0

        return score

    def to_prompt_rule(self) -> str:
        """Format this rule for inclusion in SQL generation prompt."""
        return (
            f"CORRECTION: When asked \"{self.original_question}\", "
            f"do NOT generate SQL like: {self.original_sql}. "
            f"Instead: {self.correction_description}. "
            f"Correct SQL: {self.corrected_sql}"
        )


class CorrectionMemory:
    """
    Manages correction rules scoped per (org_id, db_fingerprint).

    Rules are persisted to disk as JSON and loaded on startup.
    Per Architecture.md §5, rules never leak across unrelated databases.

    Storage is file-based for simplicity; upgrade to a DB-backed store
    if the rule set grows large.
    """

    def __init__(
        self,
        storage_dir: str | Path | None = None,
        org_id: str = "default",
        db_fingerprint: str = "default",
    ):
        self._org_id = org_id
        self._db_fingerprint = db_fingerprint
        self._rules: list[CorrectionRule] = []

        if storage_dir:
            self._storage_dir = Path(storage_dir)
        else:
            self._storage_dir = None

        # Load existing rules
        if self._storage_dir:
            self._load()

    def _storage_path(self) -> Path | None:
        """Get the storage file path for this org/db scope."""
        if not self._storage_dir:
            return None
        scope_key = f"{self._org_id}_{self._db_fingerprint}"
        safe_name = re.sub(r'[^\w\-]', '_', scope_key)
        return self._storage_dir / f"corrections_{safe_name}.json"

    def _load(self) -> None:
        """Load rules from disk."""
        path = self._storage_path()
        if not path or not path.exists():
            return

        try:
            with open(path, "r") as f:
                data = json.load(f)
            self._rules = [CorrectionRule(**rule) for rule in data]
            logger.info(f"Loaded {len(self._rules)} correction rules from {path}")
        except Exception as e:
            logger.error(f"Failed to load correction rules: {e}")
            self._rules = []

    def _save(self) -> None:
        """Persist rules to disk."""
        path = self._storage_path()
        if not path:
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "w") as f:
                json.dump([asdict(r) for r in self._rules], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save correction rules: {e}")

    def add_correction(
        self,
        original_question: str,
        original_sql: str,
        corrected_sql: str,
        correction_description: str,
        keywords: list[str] | None = None,
    ) -> CorrectionRule:
        """
        Add a user correction as a new rule.

        Keywords are auto-extracted from the question if not provided.
        """
        if keywords is None:
            keywords = self._extract_keywords(original_question)

        rule = CorrectionRule(
            rule_id="",
            original_question=original_question,
            original_sql=original_sql,
            corrected_sql=corrected_sql,
            correction_description=correction_description,
            keywords=keywords,
            org_id=self._org_id,
            db_fingerprint=self._db_fingerprint,
        )

        self._rules.append(rule)
        self._save()

        logger.info(f"Added correction rule {rule.rule_id}: {correction_description}")
        return rule

    def find_matching_rules(
        self,
        question: str,
        threshold: float = 0.3,
        max_rules: int = 5,
    ) -> list[tuple[CorrectionRule, float]]:
        """
        Find correction rules that match a given question.

        Returns list of (rule, score) tuples, sorted by score descending.
        Only returns rules above the threshold.
        """
        scored: list[tuple[CorrectionRule, float]] = []

        for rule in self._rules:
            score = rule.matches(question, threshold)
            if score >= threshold:
                scored.append((rule, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:max_rules]

    def get_prompt_rules(
        self,
        question: str,
        threshold: float = 0.3,
    ) -> list[str]:
        """
        Get formatted correction rules for inclusion in the SQL generation prompt.

        This is what the sql_generator calls to apply corrections.
        """
        matches = self.find_matching_rules(question, threshold)
        rules = []
        for rule, score in matches:
            rules.append(rule.to_prompt_rule())
            rule.applied_count += 1
        if rules:
            self._save()  # Persist updated applied_count
        return rules

    def get_all_rules(self) -> list[CorrectionRule]:
        """Return all stored rules."""
        return list(self._rules)

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule by ID."""
        before = len(self._rules)
        self._rules = [r for r in self._rules if r.rule_id != rule_id]
        if len(self._rules) < before:
            self._save()
            return True
        return False

    def clear(self) -> None:
        """Remove all rules."""
        self._rules = []
        self._save()

    def _extract_keywords(self, question: str) -> list[str]:
        """Extract meaningful keywords from a question for matching."""
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "shall",
            "of", "in", "to", "for", "with", "on", "at", "from", "by",
            "as", "into", "through", "during", "before", "after", "above",
            "below", "between", "out", "off", "over", "under", "again",
            "further", "then", "once", "here", "there", "when", "where",
            "why", "how", "all", "both", "each", "few", "more", "most",
            "other", "some", "such", "no", "nor", "not", "only", "own",
            "same", "so", "than", "too", "very", "just", "because",
            "but", "and", "or", "if", "while", "about", "up", "down",
            "it", "its", "this", "that", "these", "those", "i", "me",
            "my", "we", "our", "you", "your", "he", "him", "his", "she",
            "her", "they", "them", "their", "what", "which", "who",
            "show", "tell", "give", "get", "find", "list", "many",
        }
        words = re.findall(r'\w+', question.lower())
        return [w for w in words if w not in stopwords and len(w) > 1]

    def rules_scope_matches(self, org_id: str, db_fingerprint: str) -> bool:
        """Check if this memory's scope matches the given org/db."""
        return self._org_id == org_id and self._db_fingerprint == db_fingerprint
