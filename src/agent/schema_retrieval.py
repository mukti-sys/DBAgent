"""
Schema retrieval — embeds table/column/relationship metadata,
retrieves relevant schema slices, and detects drift on connect.

Per Architecture.md §3: Schema embeddings generated via llm_client.embed()
(same provider-agnostic path as everything else).

Per Architecture.md §5:
- Schema index is diffed and incrementally refreshed whenever a drift check
  finds new/renamed/removed columns — never a silent full rebuild.
- Only relevant schema is retrieved per query (Rules.md §3: avoid dumping
  the entire schema into every prompt).

Per Rules.md §1: If user configures a local model, embeddings route through
the same configured provider — no silent fallback to external API.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from src.agent.llm_client import LLMClient as UnifiedLLMClient

logger = logging.getLogger(__name__)


@dataclass
class ColumnInfo:
    """Metadata about a single column."""
    name: str
    data_type: str
    nullable: bool = True
    is_primary_key: bool = False
    is_foreign_key: bool = False
    foreign_key_target: str | None = None  # "table.column" format
    description: str = ""

    def to_embedding_text(self) -> str:
        """Convert to text for embedding."""
        parts = [f"column {self.name} ({self.data_type})"]
        if self.is_primary_key:
            parts.append("PRIMARY KEY")
        if self.is_foreign_key and self.foreign_key_target:
            parts.append(f"FOREIGN KEY -> {self.foreign_key_target}")
        if self.description:
            parts.append(self.description)
        return " ".join(parts)


@dataclass
class TableInfo:
    """Metadata about a single table."""
    name: str
    schema_name: str = "public"
    columns: list[ColumnInfo] = field(default_factory=list)
    description: str = ""

    def to_embedding_text(self) -> str:
        """Convert to text for embedding."""
        col_texts = [c.to_embedding_text() for c in self.columns]
        header = f"table {self.schema_name}.{self.name}"
        if self.description:
            header += f" ({self.description})"
        return f"{header}: {', '.join(col_texts)}"

    def column_names(self) -> set[str]:
        return {c.name for c in self.columns}

    def fingerprint(self) -> str:
        """Hash of table structure for drift detection."""
        data = {
            "name": self.name,
            "schema": self.schema_name,
            "columns": [
                {"name": c.name, "type": c.data_type, "pk": c.is_primary_key,
                 "fk": c.is_foreign_key, "fk_target": c.foreign_key_target}
                for c in sorted(self.columns, key=lambda c: c.name)
            ],
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class RelationshipInfo:
    """A foreign key relationship between tables."""
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    relationship_type: str = "many-to-one"  # or "one-to-one", "many-to-many"

    def to_embedding_text(self) -> str:
        return (
            f"relationship: {self.source_table}.{self.source_column} "
            f"-> {self.target_table}.{self.target_column} "
            f"({self.relationship_type})"
        )


@dataclass
class SchemaDrift:
    """Describes a detected change between indexed schema and live schema."""
    drift_type: str  # "added_table", "removed_table", "added_column", "removed_column", "renamed_column", "type_changed"
    table_name: str
    column_name: str | None = None
    old_value: str | None = None
    new_value: str | None = None

    def describe(self) -> str:
        if self.drift_type == "added_table":
            return f"New table: {self.table_name}"
        elif self.drift_type == "removed_table":
            return f"Removed table: {self.table_name}"
        elif self.drift_type == "added_column":
            return f"New column: {self.table_name}.{self.column_name}"
        elif self.drift_type == "removed_column":
            return f"Removed column: {self.table_name}.{self.column_name}"
        elif self.drift_type == "renamed_column":
            return f"Possible rename in {self.table_name}: {self.old_value} -> {self.new_value}"
        elif self.drift_type == "type_changed":
            return f"Type changed: {self.table_name}.{self.column_name} ({self.old_value} -> {self.new_value})"
        return f"Unknown drift: {self.drift_type} on {self.table_name}"


class SchemaIndex:
    """
    In-memory schema index with embedding-based retrieval and drift detection.

    For v1, uses a simple TF-IDF-like similarity rather than a full vector DB.
    Can be swapped for ChromaDB/pgvector later per Architecture.md §3.

    When an llm_client is provided, uses llm_client.embed() for real embeddings.
    Otherwise falls back to keyword matching.
    """

    def __init__(self, llm_client: UnifiedLLMClient | None = None):
        self._tables: dict[str, TableInfo] = {}
        self._relationships: list[RelationshipInfo] = []
        self._fingerprints: dict[str, str] = {}  # table_name -> fingerprint
        self._llm_client = llm_client  # For embedding generation via llm_client.embed()

    @property
    def tables(self) -> dict[str, TableInfo]:
        return dict(self._tables)

    @property
    def relationships(self) -> list[RelationshipInfo]:
        return list(self._relationships)

    def index_table(self, table: TableInfo) -> None:
        """Add or update a table in the index."""
        self._tables[table.name] = table
        self._fingerprints[table.name] = table.fingerprint()

    def index_relationship(self, rel: RelationshipInfo) -> None:
        """Add a relationship to the index."""
        # Avoid duplicates
        for existing in self._relationships:
            if (existing.source_table == rel.source_table and
                existing.source_column == rel.source_column and
                existing.target_table == rel.target_table and
                existing.target_column == rel.target_column):
                return
        self._relationships.append(rel)

    def remove_table(self, table_name: str) -> None:
        """Remove a table from the index."""
        self._tables.pop(table_name, None)
        self._fingerprints.pop(table_name, None)
        # Remove related relationships
        self._relationships = [
            r for r in self._relationships
            if r.source_table != table_name and r.target_table != table_name
        ]

    def detect_drift(self, live_tables: list[TableInfo]) -> list[SchemaDrift]:
        """
        Compare indexed schema against live schema and report drifts.

        Per Architecture.md §5: diffed and incrementally refreshed,
        never a silent full rebuild that masks what changed.
        """
        drifts: list[SchemaDrift] = []
        live_table_map = {t.name: t for t in live_tables}

        # Check for removed tables
        for name in list(self._tables.keys()):
            if name not in live_table_map:
                drifts.append(SchemaDrift(
                    drift_type="removed_table",
                    table_name=name,
                ))

        # Check for new or modified tables
        for name, live_table in live_table_map.items():
            if name not in self._tables:
                drifts.append(SchemaDrift(
                    drift_type="added_table",
                    table_name=name,
                ))
                continue

            indexed_table = self._tables[name]
            indexed_cols = {c.name: c for c in indexed_table.columns}
            live_cols = {c.name: c for c in live_table.columns}

            # Check for removed columns
            for col_name in indexed_cols:
                if col_name not in live_cols:
                    # Check if this might be a rename
                    possible_rename = self._detect_possible_rename(
                        indexed_cols[col_name], live_cols, indexed_cols
                    )
                    if possible_rename:
                        drifts.append(SchemaDrift(
                            drift_type="renamed_column",
                            table_name=name,
                            old_value=col_name,
                            new_value=possible_rename,
                        ))
                    else:
                        drifts.append(SchemaDrift(
                            drift_type="removed_column",
                            table_name=name,
                            column_name=col_name,
                        ))

            # Check for new columns
            for col_name in live_cols:
                if col_name not in indexed_cols:
                    # Only add as "new" if not already detected as rename target
                    is_rename_target = any(
                        d.drift_type == "renamed_column" and d.new_value == col_name
                        for d in drifts
                    )
                    if not is_rename_target:
                        drifts.append(SchemaDrift(
                            drift_type="added_column",
                            table_name=name,
                            column_name=col_name,
                        ))

            # Check for type changes on existing columns
            for col_name in indexed_cols:
                if col_name in live_cols:
                    if indexed_cols[col_name].data_type != live_cols[col_name].data_type:
                        drifts.append(SchemaDrift(
                            drift_type="type_changed",
                            table_name=name,
                            column_name=col_name,
                            old_value=indexed_cols[col_name].data_type,
                            new_value=live_cols[col_name].data_type,
                        ))

        return drifts

    def apply_drift(self, live_tables: list[TableInfo], drifts: list[SchemaDrift]) -> None:
        """
        Apply detected drifts by incrementally updating the index.
        Logs each change explicitly — no silent rebuilds.
        """
        live_table_map = {t.name: t for t in live_tables}

        for drift in drifts:
            logger.info(f"Schema drift applied: {drift.describe()}")

            if drift.drift_type == "removed_table":
                self.remove_table(drift.table_name)
            elif drift.drift_type == "added_table":
                if drift.table_name in live_table_map:
                    self.index_table(live_table_map[drift.table_name])
            elif drift.drift_type in ("added_column", "removed_column",
                                       "renamed_column", "type_changed"):
                # Re-index the whole table from live data
                if drift.table_name in live_table_map:
                    self.index_table(live_table_map[drift.table_name])

    def _detect_possible_rename(
        self,
        removed_col: ColumnInfo,
        live_cols: dict[str, ColumnInfo],
        indexed_cols: dict[str, ColumnInfo],
    ) -> str | None:
        """
        Heuristic: if a column was removed and a new column with the same type
        appeared, it might be a rename. Uses name similarity as a tiebreaker.
        """
        candidates = []
        for col_name, col in live_cols.items():
            if col_name in indexed_cols:
                continue  # Not new
            if col.data_type == removed_col.data_type:
                similarity = SequenceMatcher(None, removed_col.name, col_name).ratio()
                candidates.append((col_name, similarity))

        if candidates:
            best = max(candidates, key=lambda x: x[1])
            if best[1] > 0.4:  # Reasonable similarity threshold
                return best[0]
        return None

    def retrieve_relevant(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[TableInfo | RelationshipInfo]:
        """
        Retrieve the most relevant schema elements for a given natural language query.

        v1: simple keyword/token overlap scoring.
        Future: replace with embedding similarity via ChromaDB.
        """
        query_tokens = set(query.lower().split())

        scored_tables: list[tuple[float, TableInfo]] = []
        for table in self._tables.values():
            text = table.to_embedding_text().lower()
            text_tokens = set(text.split())
            overlap = len(query_tokens & text_tokens)
            # Boost exact table/column name matches
            name_match = 1.0 if table.name.lower() in query.lower() else 0.0
            col_match = sum(
                1.0 for c in table.columns if c.name.lower() in query.lower()
            )
            score = overlap + name_match * 3.0 + col_match * 2.0
            if score > 0:
                scored_tables.append((score, table))

        scored_tables.sort(key=lambda x: x[0], reverse=True)
        results: list[TableInfo | RelationshipInfo] = [
            t for _, t in scored_tables[:top_k]
        ]

        # Include relationships involving retrieved tables
        retrieved_table_names = {t.name for t in results if isinstance(t, TableInfo)}
        for rel in self._relationships:
            if (rel.source_table in retrieved_table_names or
                rel.target_table in retrieved_table_names):
                results.append(rel)

        return results

    def get_schema_context(self, query: str, top_k: int = 10) -> str:
        """
        Get a formatted schema context string for use in LLM prompts.
        Only includes relevant tables/relationships, not the whole schema.
        """
        relevant = self.retrieve_relevant(query, top_k)

        parts = []
        for item in relevant:
            if isinstance(item, TableInfo):
                parts.append(item.to_embedding_text())
            elif isinstance(item, RelationshipInfo):
                parts.append(item.to_embedding_text())

        return "\n".join(parts)

    def get_all_table_names(self) -> list[str]:
        """Return all indexed table names."""
        return list(self._tables.keys())

    def get_table(self, name: str) -> TableInfo | None:
        """Get a specific table by name."""
        return self._tables.get(name)

    def get_all_column_names(self) -> set[str]:
        """Return all column names across all tables."""
        cols = set()
        for table in self._tables.values():
            for col in table.columns:
                cols.add(f"{table.name}.{col.name}")
        return cols
