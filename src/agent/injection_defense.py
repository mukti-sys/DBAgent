"""
Prompt injection defense — sanitizes and flags suspicious free-text field
content before it re-enters the agent context.

Per Phases.md Phase 13:
- Sanitize/flag suspicious free-text field content before it re-enters
  agent context
- Exit criteria: a crafted injection payload in a test "comments" field
  is neutralized/flagged, not obeyed

Per Rules.md §4: Never trust data from the DB as instructions.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Threat level for detected injection attempts."""
    SAFE = "safe"
    SUSPICIOUS = "suspicious"
    BLOCKED = "blocked"


@dataclass
class ScanResult:
    """Result of scanning a single field for injection attempts."""
    field_name: str
    original_value: str
    sanitized_value: str
    threat_level: ThreatLevel
    detections: list[str] = field(default_factory=list)
    was_modified: bool = False


@dataclass
class InjectionScanReport:
    """Aggregate scan report across all fields in a row."""
    total_fields: int
    safe_fields: int
    suspicious_fields: int
    blocked_fields: int
    results: list[ScanResult] = field(default_factory=list)

    @property
    def has_threats(self) -> bool:
        return self.suspicious_fields > 0 or self.blocked_fields > 0

    @property
    def is_safe(self) -> bool:
        return not self.has_threats


class PromptInjectionDefense:
    """
    Scans and sanitizes DB field values that will be included in LLM context.

    Defense layers:
    1. Pattern matching for known injection signatures
    2. Structural analysis (unusual control characters, role markers)
    3. Sanitization (neutralize without destroying data)
    4. Flagging for human review when uncertain

    Key principle: data from the DB is NEVER trusted as instructions.
    All free-text fields are treated as potentially adversarial.
    """

    # Known injection patterns (case-insensitive)
    _INJECTION_PATTERNS = [
        # Direct instruction injection
        (r'(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions?|prompts?|rules?)',
         "instruction_override", ThreatLevel.BLOCKED),
        (r'you\s+are\s+now\s+(?:a|an)\s+',
         "role_reassignment", ThreatLevel.BLOCKED),
        (r'(?:system|assistant|user)\s*:\s*',
         "role_marker_injection", ThreatLevel.BLOCKED),
        (r'(?:new\s+)?(?:system\s+)?prompt\s*:',
         "prompt_injection", ThreatLevel.BLOCKED),

        # SQL injection via data fields
        (r"(?:';?\s*(?:DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE)\s)",
         "sql_injection_in_data", ThreatLevel.BLOCKED),
        (r'(?:UNION\s+(?:ALL\s+)?SELECT)',
         "union_injection", ThreatLevel.BLOCKED),

        # Prompt leaking attempts
        (r'(?:repeat|show|print|output|display)\s+(?:your|the|all)\s+(?:system\s+)?(?:prompt|instructions?|rules?)',
         "prompt_leak_attempt", ThreatLevel.SUSPICIOUS),

        # Jailbreak patterns
        (r'(?:DAN|do\s+anything\s+now)',
         "jailbreak_attempt", ThreatLevel.BLOCKED),
        (r'(?:pretend|act\s+as\s+if)\s+(?:you|there)\s+(?:are|is)\s+no\s+(?:rules?|restrictions?|limits?)',
         "restriction_bypass", ThreatLevel.BLOCKED),

        # Data exfiltration attempts
        (r'(?:send|post|transmit|exfiltrate)\s+(?:\w+\s+){0,3}(?:data|information|content)\s+to',
         "data_exfiltration", ThreatLevel.BLOCKED),

        # Encoding tricks
        (r'(?:base64|hex|rot13)\s*(?:decode|encode)',
         "encoding_trick", ThreatLevel.SUSPICIOUS),
    ]

    # Structural red flags
    _STRUCTURAL_PATTERNS = [
        # Excessive special characters suggesting encoded payloads
        (r'[\x00-\x08\x0b\x0c\x0e-\x1f]{3,}', "control_characters", ThreatLevel.SUSPICIOUS),
        # Markdown/formatting that could manipulate LLM context
        (r'```(?:system|python|bash|sh)\s', "code_block_injection", ThreatLevel.SUSPICIOUS),
        # XML/HTML tags that mimic system messages
        (r'<(?:system|instruction|prompt|rule)[^>]*>', "xml_tag_injection", ThreatLevel.BLOCKED),
    ]

    def __init__(self, strict_mode: bool = True):
        """
        Args:
            strict_mode: If True, BLOCKED items are fully sanitized.
                If False, only flagged (for review pipelines).
        """
        self._strict = strict_mode
        self._compiled_patterns = [
            (re.compile(p, re.IGNORECASE), name, level)
            for p, name, level in self._INJECTION_PATTERNS
        ]
        self._compiled_structural = [
            (re.compile(p, re.IGNORECASE), name, level)
            for p, name, level in self._STRUCTURAL_PATTERNS
        ]

    def scan_value(self, field_name: str, value: str) -> ScanResult:
        """Scan a single field value for injection attempts."""
        if not isinstance(value, str) or not value.strip():
            return ScanResult(
                field_name=field_name,
                original_value=str(value),
                sanitized_value=str(value),
                threat_level=ThreatLevel.SAFE,
            )

        detections: list[str] = []
        max_threat = ThreatLevel.SAFE

        # Check injection patterns
        for pattern, name, level in self._compiled_patterns:
            if pattern.search(value):
                detections.append(f"{name} ({level.value})")
                if level.value == "blocked" or (level == ThreatLevel.BLOCKED):
                    max_threat = ThreatLevel.BLOCKED
                elif max_threat != ThreatLevel.BLOCKED:
                    max_threat = ThreatLevel.SUSPICIOUS

        # Check structural patterns
        for pattern, name, level in self._compiled_structural:
            if pattern.search(value):
                detections.append(f"{name} ({level.value})")
                if level == ThreatLevel.BLOCKED:
                    max_threat = ThreatLevel.BLOCKED
                elif max_threat != ThreatLevel.BLOCKED:
                    max_threat = ThreatLevel.SUSPICIOUS

        # Sanitize if needed
        sanitized = value
        was_modified = False

        if max_threat == ThreatLevel.BLOCKED and self._strict:
            sanitized = self._sanitize(value)
            was_modified = sanitized != value
        elif max_threat == ThreatLevel.SUSPICIOUS:
            sanitized = self._soft_sanitize(value)
            was_modified = sanitized != value

        if detections:
            logger.warning(
                f"Injection scan for field '{field_name}': "
                f"threat={max_threat.value}, detections={detections}"
            )

        return ScanResult(
            field_name=field_name,
            original_value=value,
            sanitized_value=sanitized,
            threat_level=max_threat,
            detections=detections,
            was_modified=was_modified,
        )

    def scan_row(self, row: dict[str, Any]) -> InjectionScanReport:
        """Scan all string fields in a database row."""
        results: list[ScanResult] = []

        for field_name, value in row.items():
            if isinstance(value, str):
                result = self.scan_value(field_name, value)
                results.append(result)

        safe = sum(1 for r in results if r.threat_level == ThreatLevel.SAFE)
        suspicious = sum(1 for r in results if r.threat_level == ThreatLevel.SUSPICIOUS)
        blocked = sum(1 for r in results if r.threat_level == ThreatLevel.BLOCKED)

        return InjectionScanReport(
            total_fields=len(results),
            safe_fields=safe,
            suspicious_fields=suspicious,
            blocked_fields=blocked,
            results=results,
        )

    def sanitize_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """Return a copy of the row with all string fields sanitized."""
        sanitized_row = dict(row)
        for field_name, value in row.items():
            if isinstance(value, str):
                result = self.scan_value(field_name, value)
                sanitized_row[field_name] = result.sanitized_value
        return sanitized_row

    def _sanitize(self, value: str) -> str:
        """Hard sanitize: wrap value in data markers so LLM treats it as data."""
        # Remove any role markers
        cleaned = re.sub(r'(?:system|assistant|user)\s*:', '[ROLE_MARKER_REMOVED]:', value)
        # Wrap in data delimiters
        return f"[DB_DATA_START]{cleaned}[DB_DATA_END]"

    def _soft_sanitize(self, value: str) -> str:
        """Soft sanitize: flag but preserve original content."""
        return f"[FLAGGED_CONTENT]{value}[/FLAGGED_CONTENT]"
