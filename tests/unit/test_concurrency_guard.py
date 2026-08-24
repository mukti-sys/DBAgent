"""
Phase 10 — Concurrency Guard unit tests.

Per Tests.md §1:
- concurrency_guard.py: timeout triggers correctly; circuit breaker opens
  under simulated load

Per Phases.md Phase 10 exit criteria:
- Expensive/concurrent queries are throttled or flagged, not left to
  degrade the DB
"""

import time
import threading
import pytest

from src.agent.concurrency_guard import (
    ConcurrencyGuard,
    GuardConfig,
    CircuitState,
    QueryTimeoutError,
    CircuitBreakerOpenError,
    ConcurrencyLimitError,
)


@pytest.fixture
def guard():
    return ConcurrencyGuard(GuardConfig(
        query_timeout_seconds=1.0,
        max_concurrent_queries=3,
        circuit_breaker_threshold=3,
        circuit_breaker_reset_seconds=0.5,
    ))


class TestQueryTimeout:
    """Test that query timeout triggers correctly."""

    def test_fast_query_succeeds(self, guard):
        result = guard.execute_with_guard(lambda: 42, query_id="fast")
        assert result == 42

    def test_slow_query_times_out(self, guard):
        def slow():
            time.sleep(5)
            return "too late"

        with pytest.raises(QueryTimeoutError, match="timed out"):
            guard.execute_with_guard(slow, query_id="slow", timeout_override=0.2)

    def test_timeout_override_respected(self, guard):
        # Default is 1s, override to 0.1s
        def medium():
            time.sleep(0.5)
            return "done"

        with pytest.raises(QueryTimeoutError):
            guard.execute_with_guard(medium, query_id="medium", timeout_override=0.1)

    def test_query_error_propagated(self, guard):
        def bad_query():
            raise ValueError("bad SQL")

        with pytest.raises(ValueError, match="bad SQL"):
            guard.execute_with_guard(bad_query, query_id="bad")


class TestConcurrencyLimit:
    """Test that concurrent query limiting works."""

    def test_concurrent_queries_within_limit_succeed(self, guard):
        results = []
        barrier = threading.Barrier(2, timeout=5)

        def query():
            barrier.wait()
            time.sleep(0.1)
            return "ok"

        threads = []
        for i in range(2):
            t = threading.Thread(
                target=lambda: results.append(
                    guard.execute_with_guard(query, query_id=f"q{i}")
                )
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=5)

        assert len(results) == 2
        assert all(r == "ok" for r in results)

    def test_exceeding_concurrency_limit_raises(self, guard):
        # Fill up all 3 slots with blocking queries
        barrier = threading.Event()
        errors = []

        def blocking_query():
            barrier.wait()
            return "done"

        threads = []
        for i in range(3):
            t = threading.Thread(
                target=lambda: guard.execute_with_guard(blocking_query, query_id=f"blocking{i}")
            )
            threads.append(t)
            t.start()

        # Wait for slots to fill
        time.sleep(0.2)

        # 4th query should be rejected
        try:
            guard.execute_with_guard(lambda: "overflow", query_id="overflow")
        except ConcurrencyLimitError:
            errors.append("limit_hit")

        # Release blocking queries
        barrier.set()
        for t in threads:
            t.join(timeout=5)

        assert "limit_hit" in errors


class TestCircuitBreaker:
    """Test circuit breaker opens under simulated load failures."""

    def test_circuit_starts_closed(self, guard):
        assert guard.circuit_state == CircuitState.CLOSED

    def test_circuit_trips_after_threshold_failures(self, guard):
        # 3 failures should trip it (threshold=3)
        for i in range(3):
            with pytest.raises(ValueError):
                guard.execute_with_guard(
                    lambda: (_ for _ in ()).throw(ValueError("fail")),
                    query_id=f"fail{i}",
                )

        assert guard.circuit_state == CircuitState.OPEN

    def test_open_circuit_rejects_requests(self, guard):
        # Trip the breaker
        for i in range(3):
            try:
                guard.execute_with_guard(
                    lambda: (_ for _ in ()).throw(ValueError("fail")),
                    query_id=f"trip{i}",
                )
            except ValueError:
                pass

        # Now it should reject
        with pytest.raises(CircuitBreakerOpenError):
            guard.execute_with_guard(lambda: "should_not_run", query_id="rejected")

    def test_circuit_transitions_to_half_open_after_cooldown(self, guard):
        # Trip the breaker
        for i in range(3):
            try:
                guard.execute_with_guard(
                    lambda: (_ for _ in ()).throw(ValueError("fail")),
                    query_id=f"trip{i}",
                )
            except ValueError:
                pass

        assert guard.circuit_state == CircuitState.OPEN

        # Wait for cooldown (0.5s)
        time.sleep(0.6)

        assert guard.circuit_state == CircuitState.HALF_OPEN

    def test_successful_probe_closes_circuit(self, guard):
        # Trip the breaker
        for i in range(3):
            try:
                guard.execute_with_guard(
                    lambda: (_ for _ in ()).throw(ValueError("fail")),
                    query_id=f"trip{i}",
                )
            except ValueError:
                pass

        # Wait for cooldown
        time.sleep(0.6)
        assert guard.circuit_state == CircuitState.HALF_OPEN

        # Successful probe should close it
        result = guard.execute_with_guard(lambda: "recovered", query_id="probe")
        assert result == "recovered"
        assert guard.circuit_state == CircuitState.CLOSED

    def test_failed_probe_reopens_circuit(self, guard):
        # Trip the breaker
        for i in range(3):
            try:
                guard.execute_with_guard(
                    lambda: (_ for _ in ()).throw(ValueError("fail")),
                    query_id=f"trip{i}",
                )
            except ValueError:
                pass

        # Wait for cooldown
        time.sleep(0.6)
        assert guard.circuit_state == CircuitState.HALF_OPEN

        # Failed probe should reopen
        try:
            guard.execute_with_guard(
                lambda: (_ for _ in ()).throw(ValueError("still broken")),
                query_id="bad_probe",
            )
        except ValueError:
            pass

        assert guard.circuit_state == CircuitState.OPEN

    def test_success_resets_failure_counter(self, guard):
        # 2 failures (not enough to trip)
        for i in range(2):
            try:
                guard.execute_with_guard(
                    lambda: (_ for _ in ()).throw(ValueError("fail")),
                    query_id=f"fail{i}",
                )
            except ValueError:
                pass

        assert guard.consecutive_failures == 2

        # 1 success resets counter
        guard.execute_with_guard(lambda: "ok", query_id="success")
        assert guard.consecutive_failures == 0

        # Now 2 more failures should still not trip (counter was reset)
        for i in range(2):
            try:
                guard.execute_with_guard(
                    lambda: (_ for _ in ()).throw(ValueError("fail")),
                    query_id=f"fail_again{i}",
                )
            except ValueError:
                pass

        assert guard.circuit_state == CircuitState.CLOSED


class TestGuardStatus:
    """Test guard status reporting."""

    def test_status_returns_config(self, guard):
        status = guard.get_status()
        assert status["circuit_state"] == "closed"
        assert status["active_queries"] == 0
        assert status["config"]["timeout_seconds"] == 1.0
        assert status["config"]["max_concurrent"] == 3
        assert status["config"]["breaker_threshold"] == 3

    def test_reset_clears_state(self, guard):
        # Trip the breaker
        for i in range(3):
            try:
                guard.execute_with_guard(
                    lambda: (_ for _ in ()).throw(ValueError("fail")),
                    query_id=f"trip{i}",
                )
            except ValueError:
                pass

        assert guard.circuit_state == CircuitState.OPEN

        guard.reset()
        assert guard.circuit_state == CircuitState.CLOSED
        assert guard.consecutive_failures == 0
