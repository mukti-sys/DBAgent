"""
Key resolver — fuzzy-matches candidate join keys across dirty/multi-DB schemas.
Asks once, remembers. Never silently guesses a join across inconsistent keys.

Per PRD.md §3 G7: Handle multi-DB / inconsistent join keys defensively: propose
candidate matches with confidence, confirm once, remember.

Per Rules.md §2 & §4: Genuinely unresolved — no universal resolver exists.
Best-effort mitigation: propose candidate joins with a confidence score;
confirm ambiguous ones with the user once; remember the answer. Never
silently guess a join across inconsistent keys.
"""

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MatchConfidenceTier(Enum):
    """Confidence tiers for proposed join keys."""
    CONFIRMED = "confirmed"      # Previously confirmed by user & remembered
    HIGH = "high"                # High-confidence single clear match (>= threshold)
    AMBIGUOUS = "ambiguous"      # Multiple candidates or moderate confidence; requires confirmation
    LOW = "low"                  # Weak signal; must not auto-join
    NO_MATCH = "no_match"        # No viable join candidate found


@dataclass
class CandidateJoin:
    """A proposed join between two columns across tables."""
    left_table: str
    left_column: str
    right_table: str
    right_column: str
    confidence: float            # 0.0 to 1.0
    match_reason: str
    is_confirmed: bool = False
    notes: str = ""

    @property
    def join_key(self) -> str:
        """Normalized identifier for the join pair."""
        t1, c1, t2, c2 = self.left_table, self.left_column, self.right_table, self.right_column
        if (t1, c1) > (t2, c2):
            t1, c1, t2, c2 = t2, c2, t1, c1
        return f"{t1}.{c1} = {t2}.{c2}"

    def to_sql_on_clause(self) -> str:
        return f"{self.left_table}.{self.left_column} = {self.right_table}.{self.right_column}"


@dataclass
class KeyResolutionResult:
    """Result of attempting to resolve join keys between two tables."""
    left_table: str
    right_table: str
    tier: MatchConfidenceTier
    candidates: list[CandidateJoin] = field(default_factory=list)
    selected_join: CandidateJoin | None = None
    requires_confirmation: bool = False
    message: str = ""

    @property
    def is_safe_to_join(self) -> bool:
        """True only if confirmed or unequivocally high confidence without ambiguity."""
        return not self.requires_confirmation and self.selected_join is not None


class KeyResolver:
    """
    Defensive join key matcher and memory for cross-table and multi-DB joins.

    Heuristics:
    1. Check persisted/confirmed memory first (1.0 confidence)
    2. Exact name match (e.g. customer_id == customer_id)
    3. Prefix/suffix foreign-key naming patterns (e.g. order.user_id == users.id, order.uid == user.id)
    4. Fuzzy string similarity across column names with tokenization
    5. Disambiguation: if top 2 candidates have close scores, flag as AMBIGUOUS
    """

    def __init__(
        self,
        storage_dir: str | Path | None = None,
        org_id: str = "default",
        db_fingerprint: str = "default",
        high_confidence_threshold: float = 0.85,
        ambiguity_delta_threshold: float = 0.15,
    ):
        self._org_id = org_id
        self._db_fingerprint = db_fingerprint
        self._high_threshold = high_confidence_threshold
        self._ambiguity_delta = ambiguity_delta_threshold
        self._confirmed_joins: dict[str, dict[str, Any]] = {}
        self._storage_dir = Path(storage_dir) if storage_dir else None

        if self._storage_dir:
            self._load()

    def _storage_path(self) -> Path | None:
        if not self._storage_dir:
            return None
        safe_scope = re.sub(r'[^\w\-]', '_', f"{self._org_id}_{self._db_fingerprint}")
        return self._storage_dir / f"key_memory_{safe_scope}.json"

    def _load(self) -> None:
        path = self._storage_path()
        if not path or not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                self._confirmed_joins = json.load(f)
            logger.info(f"Loaded {len(self._confirmed_joins)} confirmed join keys from {path}")
        except Exception as e:
            logger.error(f"Failed to load confirmed join keys: {e}")
            self._confirmed_joins = {}

    def _save(self) -> None:
        path = self._storage_path()
        if not path:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._confirmed_joins, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save confirmed join keys: {e}")

    def remember_join(
        self,
        left_table: str,
        left_column: str,
        right_table: str,
        right_column: str,
        notes: str = "User confirmed",
    ) -> CandidateJoin:
        """Remember a confirmed join key mapping so future queries don't ask again."""
        join = CandidateJoin(
            left_table=left_table,
            left_column=left_column,
            right_table=right_table,
            right_column=right_column,
            confidence=1.0,
            match_reason="Previously confirmed by user (persisted in key memory)",
            is_confirmed=True,
            notes=notes,
        )
        self._confirmed_joins[join.join_key] = asdict(join)
        self._save()
        logger.info(f"Remembered join mapping: {join.join_key}")
        return join

    def get_confirmed_joins(self) -> list[CandidateJoin]:
        """Return all remembered join keys."""
        return [CandidateJoin(**data) for data in self._confirmed_joins.values()]

    def clear_memory(self) -> None:
        """Clear remembered join keys."""
        self._confirmed_joins = {}
        self._save()

    def resolve_join(
        self,
        left_table: str,
        left_columns: list[str],
        right_table: str,
        right_columns: list[str],
    ) -> KeyResolutionResult:
        """
        Propose or retrieve candidate join keys between two tables.

        Returns KeyResolutionResult with confidence tier and candidates.
        """
        # 1. Check remembered confirmed joins
        for raw_data in self._confirmed_joins.values():
            cj = CandidateJoin(**raw_data)
            if (
                (cj.left_table.lower() == left_table.lower() and cj.right_table.lower() == right_table.lower()) or
                (cj.left_table.lower() == right_table.lower() and cj.right_table.lower() == left_table.lower())
            ):
                return KeyResolutionResult(
                    left_table=left_table,
                    right_table=right_table,
                    tier=MatchConfidenceTier.CONFIRMED,
                    candidates=[cj],
                    selected_join=cj,
                    requires_confirmation=False,
                    message=f"Using confirmed join key: {cj.to_sql_on_clause()}",
                )

        # 2. Score candidate column pairs
        candidates: list[CandidateJoin] = []
        for l_col in left_columns:
            for r_col in right_columns:
                score, reason = self._score_column_pair(
                    left_table, l_col, right_table, r_col
                )
                if score > 0.3:
                    candidates.append(CandidateJoin(
                        left_table=left_table,
                        left_column=l_col,
                        right_table=right_table,
                        right_column=r_col,
                        confidence=round(score, 3),
                        match_reason=reason,
                        is_confirmed=False,
                    ))

        # Sort candidates descending by confidence
        candidates.sort(key=lambda c: c.confidence, reverse=True)

        if not candidates:
            return KeyResolutionResult(
                left_table=left_table,
                right_table=right_table,
                tier=MatchConfidenceTier.NO_MATCH,
                candidates=[],
                selected_join=None,
                requires_confirmation=True,
                message=f"No viable join key found between {left_table} and {right_table}.",
            )

        top_candidate = candidates[0]

        # 3. Check for ambiguity (multiple candidates with close confidence)
        is_ambiguous = False
        if len(candidates) > 1:
            second_candidate = candidates[1]
            diff = top_candidate.confidence - second_candidate.confidence
            if diff < self._ambiguity_delta and top_candidate.confidence < 0.95:
                is_ambiguous = True

        if top_candidate.confidence < self._high_threshold:
            is_ambiguous = True

        if is_ambiguous:
            return KeyResolutionResult(
                left_table=left_table,
                right_table=right_table,
                tier=MatchConfidenceTier.AMBIGUOUS,
                candidates=candidates[:5],
                selected_join=None,
                requires_confirmation=True,
                message=(
                    f"Ambiguous join between {left_table} and {right_table}. "
                    f"Top candidate: {top_candidate.to_sql_on_clause()} (confidence {top_candidate.confidence:.2f}). "
                    f"Please confirm or select the correct join key."
                ),
            )

        # 4. Unambiguous high-confidence match
        return KeyResolutionResult(
            left_table=left_table,
            right_table=right_table,
            tier=MatchConfidenceTier.HIGH,
            candidates=candidates[:5],
            selected_join=top_candidate,
            requires_confirmation=False,
            message=f"High-confidence join found: {top_candidate.to_sql_on_clause()} (confidence {top_candidate.confidence:.2f})",
        )

    def _score_column_pair(
        self,
        left_table: str,
        left_col: str,
        right_table: str,
        right_col: str,
    ) -> tuple[float, str]:
        """Compute matching score and rationale for a pair of columns."""
        l_norm = self._normalize_name(left_col)
        r_norm = self._normalize_name(right_col)
        lt_norm = self._normalize_table(left_table)
        rt_norm = self._normalize_table(right_table)

        # Exact column name match
        if left_col.lower() == right_col.lower():
            # If both are generic "id", this is not a valid join key across different tables
            if l_norm in ("id", "uuid", "pk", "key"):
                return 0.2, "Both are generic 'id', not foreign keys"
            # Key-like columns (_id/_uuid/_key/_pk) are strong join signals
            if l_norm.endswith(("_id", "_uuid", "_key", "_pk")):
                return 0.95, f"Exact key column match '{left_col}'"
            # Shared non-key columns (timestamps, audit fields) are usually
            # coincidence, not a relationship — keep them below the FK-pattern
            # score and below the auto-join threshold so they never silently join
            return 0.4, f"Shared non-key column '{left_col}' — weak join signal"

        # Standard FK pattern 1: left table has '<right_table>_id' and right table has 'id'
        if l_norm in (f"{rt_norm}_id", f"{rt_norm}id", f"{rt_norm}_pk", f"{rt_norm}_key") and r_norm in ("id", "pk", "uuid", f"{rt_norm}_id"):
            return 0.92, f"FK pattern: {left_table}.{left_col} -> {right_table}.{right_col}"

        # Standard FK pattern 2: right table has '<left_table>_id' and left table has 'id'
        if r_norm in (f"{lt_norm}_id", f"{lt_norm}id", f"{lt_norm}_pk", f"{lt_norm}_key") and l_norm in ("id", "pk", "uuid", f"{lt_norm}_id"):
            return 0.92, f"FK pattern: {right_table}.{right_col} -> {left_table}.{left_col}"

        # Prefix / substring matching (e.g. cust_id vs customer_id)
        token_score = self._token_similarity(left_col, right_col, left_table, right_table)
        if token_score > 0.75:
            return token_score, f"Token and semantic similarity ({token_score:.2f})"

        # String similarity fallback
        seq_ratio = SequenceMatcher(None, l_norm, r_norm).ratio()
        if seq_ratio > 0.7:
            return seq_ratio * 0.75, f"Fuzzy name similarity ({seq_ratio:.2f})"

        return 0.0, "No strong match"

    def _normalize_name(self, name: str) -> str:
        """Lowercase and normalize column names."""
        clean = re.sub(r'[^a-zA-Z0-9_]', '', name.lower())
        return clean

    def _normalize_table(self, name: str) -> str:
        """Singularize and normalize table names (e.g. 'users' -> 'user')."""
        t = re.sub(r'[^a-zA-Z0-9_]', '', name.lower())
        if t.endswith("ies"):
            t = t[:-3] + "y"
        elif t.endswith("es") and not t.endswith("ses"):
            t = t[:-2]
        elif t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        return t

    def _token_similarity(self, col1: str, col2: str, t1: str, t2: str) -> float:
        """Calculate token-based overlap score between columns and table names."""
        tokens1 = set(re.split(r'[_]+', col1.lower()))
        tokens2 = set(re.split(r'[_]+', col2.lower()))
        
        # Add table context
        t1_norm = self._normalize_table(t1)
        t2_norm = self._normalize_table(t2)

        # Check for dirty abbreviation matches (e.g., acc_id vs account_id, cust_id vs customer_id)
        abbreviations = {
            "cust": "customer", "acc": "account", "org": "organization",
            "usr": "user", "prod": "product", "tx": "transaction",
            "cat": "category", "loc": "location", "addr": "address"
        }
        
        expanded1 = set()
        for tok in tokens1:
            expanded1.add(tok)
            if tok in abbreviations:
                expanded1.add(abbreviations[tok])
                
        expanded2 = set()
        for tok in tokens2:
            expanded2.add(tok)
            if tok in abbreviations:
                expanded2.add(abbreviations[tok])

        overlap = expanded1 & expanded2
        if not overlap:
            return 0.0

        if "id" in overlap and len(overlap) == 1:
            # Only "id" matched - check if one column has the other table's root
            if t1_norm in expanded2 or t2_norm in expanded1:
                return 0.85
            return 0.3

        score = len(overlap) / max(len(expanded1), len(expanded2))
        return min(score + 0.2, 0.9)
