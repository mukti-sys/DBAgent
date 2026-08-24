"""
Concurrency guard — query timeout + circuit breaker under concurrent load.

Per Phases.md Phase 10: Query timeout + circuit breaker under simulated
concurrent load. Expensive/concurrent queries are throttled or flagged,
not left to degrade the DB.

Per Architecture.md §5: All queries run with a configurable timeout;
circuit breaker trips after repeated failures to protect the DB.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation — requests pass through
    OPEN = "open"          # Tripped — all requests rejected immediately
    HALF_OPEN = "half_open"  # Testing recovery — allow one probe request


@dataclass
class QueryMetrics:
    """Tracking metrics for a single query execution."""
    query_id: str
    start_time: float
    end_time: float | None = None
    timed_out: bool = False
    error: str | None = None
    duration_ms: float = 0.0

    def finish(self, error: str | None = None, timed_out: bool = False):
        self.end_time = time.monotonic()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.error = error
        self.timed_out = timed_out


class QueryTimeoutError(Exception):
    """Raised when a query exceeds the configured timeout."""
    pass


class CircuitBreakerOpenError(Exception):
    """Raised when the circuit breaker is open and rejecting requests."""
    pass


class ConcurrencyLimitError(Exception):
    """Raised when too many concurrent queries are in flight."""
    pass


@dataclass
class GuardConfig:
    """Configuration for the concurrency guard."""
    query_timeout_seconds: float = 30.0
    max_concurrent_queries: int = 10
    circuit_breaker_threshold: int = 5      # failures before tripping
    circuit_breaker_reset_seconds: float = 60.0
    circuit_breaker_half_open_max: int = 1  # probes allowed in half-open


class ConcurrencyGuard:
    """
    Protects the database from expensive or runaway queries.

    Features:
    1. Query timeout — kills queries exceeding the configured limit
    2. Concurrency limiter — caps simultaneous in-flight queries
    3. Circuit breaker — trips after repeated failures, auto-resets after cooldown

    Thread-safe: uses locks for all mutable state.
    """

    def __init__(self, config: GuardConfig | None = None):
        self._config = config or GuardConfig()
        self._lock = threading.Lock()

        # Concurrency tracking
        self._active_queries: int = 0
        self._total_executed: int = 0

        # Circuit breaker state
        self._circuit_state: CircuitState = CircuitState.CLOSED
        self._consecutive_failures: int = 0
        self._last_failure_time: float = 0.0
        self._half_open_probes: int = 0

        # Metrics history (bounded)
        self._recent_metrics: list[QueryMetrics] = []
        self._max_history: int = 100

    @property
    def circuit_state(self) -> CircuitState:
        with self._lock:
            self._maybe_transition_to_half_open()
            return self._circuit_state

    @property
    def active_queries(self) -> int:
        with self._lock:
            return self._active_queries

    @property
    def consecutive_failures(self) -> int:
        with self._lock:
            return self._consecutive_failures

    def execute_with_guard(
        self,
        query_fn: Callable[[], Any],
        query_id: str = "unnamed",
        timeout_override: float | None = None,
    ) -> Any:
        """
        Execute a query function with timeout, concurrency limit, and circuit breaker.

        Args:
            query_fn: The callable to execute (e.g., DB query).
            query_id: Identifier for logging/metrics.
            timeout_override: Override the default timeout for this query.

        Returns:
            The result of query_fn.

        Raises:
            CircuitBreakerOpenError: If circuit breaker is open.
            ConcurrencyLimitError: If too many concurrent queries.
            QueryTimeoutError: If query exceeds timeout.
        """
        timeout = timeout_override or self._config.query_timeout_seconds

        # 1. Check circuit breaker
        self._check_circuit_breaker()

        # 2. Check concurrency limit
        self._acquire_slot()

        metrics = QueryMetrics(
            query_id=query_id,
            start_time=time.monotonic(),
        )

        try:
            # 3. Execute with timeout
            result = self._execute_with_timeout(query_fn, timeout, query_id)

            # 4. Record success
            metrics.finish()
            self._record_success(metrics)
            return result

        except QueryTimeoutError:
            metrics.finish(error="timeout", timed_out=True)
            self._record_failure(metrics)
            raise

        except Exception as e:
            metrics.finish(error=str(e))
            self._record_failure(metrics)
            raise

        finally:
            self._release_slot()

    def _check_circuit_breaker(self):
        """Check if the circuit breaker allows the request."""
        with self._lock:
            self._maybe_transition_to_half_open()

            if self._circuit_state == CircuitState.OPEN:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is OPEN after {self._consecutive_failures} consecutive failures. "
                    f"Queries are being rejected to protect the database. "
                    f"Will attempt recovery in {self._config.circuit_breaker_reset_seconds}s."
                )

            if self._circuit_state == CircuitState.HALF_OPEN:
                if self._half_open_probes >= self._config.circuit_breaker_half_open_max:
                    raise CircuitBreakerOpenError(
                        "Circuit breaker is HALF_OPEN — max probe requests reached. "
                        "Waiting for probe result before allowing more."
                    )
                self._half_open_probes += 1

    def _acquire_slot(self):
        """Acquire a concurrency slot or raise if limit exceeded."""
        with self._lock:
            if self._active_queries >= self._config.max_concurrent_queries:
                raise ConcurrencyLimitError(
                    f"Concurrency limit reached: {self._active_queries}/{self._config.max_concurrent_queries} "
                    f"queries in flight. Try again later."
                )
            self._active_queries += 1
            self._total_executed += 1

    def _release_slot(self):
        """Release a concurrency slot."""
        with self._lock:
            self._active_queries = max(0, self._active_queries - 1)

    def _execute_with_timeout(
        self,
        query_fn: Callable[[], Any],
        timeout: float,
        query_id: str,
    ) -> Any:
        """Execute a function with a timeout using a thread."""
        result_container: dict[str, Any] = {}
        error_container: dict[str, Any] = {}

        def target():
            try:
                result_container["result"] = query_fn()
            except Exception as e:
                error_container["error"] = e

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            logger.warning(f"Query '{query_id}' exceeded timeout of {timeout}s")
            raise QueryTimeoutError(
                f"Query '{query_id}' timed out after {timeout}s"
            )

        if "error" in error_container:
            raise error_container["error"]

        return result_container.get("result")

    def _record_success(self, metrics: QueryMetrics):
        """Record a successful query and potentially close the circuit."""
        with self._lock:
            self._recent_metrics.append(metrics)
            if len(self._recent_metrics) > self._max_history:
                self._recent_metrics = self._recent_metrics[-self._max_history:]

            # Reset failure counter on success
            self._consecutive_failures = 0

            if self._circuit_state == CircuitState.HALF_OPEN:
                logger.info("Circuit breaker transitioning from HALF_OPEN -> CLOSED")
                self._circuit_state = CircuitState.CLOSED
                self._half_open_probes = 0

    def _record_failure(self, metrics: QueryMetrics):
        """Record a failed query and potentially trip the circuit breaker."""
        with self._lock:
            self._recent_metrics.append(metrics)
            if len(self._recent_metrics) > self._max_history:
                self._recent_metrics = self._recent_metrics[-self._max_history:]

            self._consecutive_failures += 1
            self._last_failure_time = time.monotonic()

            if self._circuit_state == CircuitState.HALF_OPEN:
                # Probe failed — back to OPEN
                logger.warning("Circuit breaker probe failed — remaining OPEN")
                self._circuit_state = CircuitState.OPEN
                self._half_open_probes = 0

            elif (self._circuit_state == CircuitState.CLOSED and
                  self._consecutive_failures >= self._config.circuit_breaker_threshold):
                logger.warning(
                    f"Circuit breaker TRIPPED after {self._consecutive_failures} "
                    f"consecutive failures — transitioning to OPEN"
                )
                self._circuit_state = CircuitState.OPEN

    def _maybe_transition_to_half_open(self):
        """Check if the circuit breaker should transition from OPEN to HALF_OPEN."""
        if self._circuit_state != CircuitState.OPEN:
            return

        elapsed = time.monotonic() - self._last_failure_time
        if elapsed >= self._config.circuit_breaker_reset_seconds:
            logger.info(
                f"Circuit breaker cooldown ({self._config.circuit_breaker_reset_seconds}s) "
                f"elapsed — transitioning OPEN -> HALF_OPEN"
            )
            self._circuit_state = CircuitState.HALF_OPEN
            self._half_open_probes = 0

    def get_status(self) -> dict[str, Any]:
        """Return current guard status for diagnostics."""
        with self._lock:
            self._maybe_transition_to_half_open()
            return {
                "circuit_state": self._circuit_state.value,
                "active_queries": self._active_queries,
                "total_executed": self._total_executed,
                "consecutive_failures": self._consecutive_failures,
                "config": {
                    "timeout_seconds": self._config.query_timeout_seconds,
                    "max_concurrent": self._config.max_concurrent_queries,
                    "breaker_threshold": self._config.circuit_breaker_threshold,
                    "breaker_reset_seconds": self._config.circuit_breaker_reset_seconds,
                },
            }

    def reset(self):
        """Reset guard to initial state (for testing)."""
        with self._lock:
            self._circuit_state = CircuitState.CLOSED
            self._consecutive_failures = 0
            self._active_queries = 0
            self._half_open_probes = 0
            self._recent_metrics = []
