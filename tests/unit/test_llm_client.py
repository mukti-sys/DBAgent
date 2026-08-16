"""
Phase 8.5 — LLM Client tests.

Per Tests.md §1:
- llm_client.py: routes to the configured provider/model per role;
  capability self-test correctly grades a compliant model as pass and a
  malformed/non-compliant stub as degraded or fail; simulated timeout/rate-limit
  triggers retry-with-backoff, then a clean 'refused: model unavailable' —
  never a silent hang or a fabricated answer.
"""

import json
import pytest
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.agent.llm_client import (
    LLMClient,
    ModelConfig,
    ModelUnavailableError,
    ModelCapabilityError,
    SelfTestResult,
    MAX_RETRIES,
)


@pytest.fixture
def models_yaml(tmp_path):
    """Create a test models.yaml."""
    config = tmp_path / "models.yaml"
    config.write_text("""
default_provider: "test_provider"

roles:
  intent:
    provider: "test_provider"
    model: "test-cheap"
    credentials_env: "TEST_API_KEY"

  generation:
    provider: "test_provider"
    model: "test-mid"
    credentials_env: "TEST_API_KEY"

  narration:
    provider: "test_provider"
    model: "test-cheap"
    credentials_env: "TEST_API_KEY"

  verification_explanation:
    provider: "test_provider"
    model: "test-cheap"
    credentials_env: "TEST_API_KEY"

  embeddings:
    provider: "test_provider"
    model: "test-embed"
    credentials_env: "TEST_API_KEY"

  escalation:
    provider: "test_provider"
    model: "test-strong"
    credentials_env: "TEST_API_KEY"
""")
    return config


def make_completion_fn(response_text: str):
    """Create a simple completion function that returns fixed text."""
    def fn(model, messages, **kwargs):
        return response_text
    return fn


def make_failing_completion_fn(error_class=Exception, message="API error"):
    """Create a completion function that always fails."""
    def fn(model, messages, **kwargs):
        raise error_class(message)
    return fn


def make_embedding_fn(dim: int = 3):
    """Create an embedding function that returns fixed vectors."""
    def fn(model, texts):
        return [[0.1, 0.2, 0.3]] * len(texts)
    return fn


class TestModelRouting:
    """Test that LLM client routes to the correct provider/model per role."""

    def test_routes_to_intent_model(self, models_yaml):
        client = LLMClient(
            models_config_path=models_yaml,
            completion_fn=make_completion_fn("intent result"),
        )
        config = client.get_model_config("intent")
        assert config.model == "test-cheap"
        assert config.provider == "test_provider"

    def test_routes_to_generation_model(self, models_yaml):
        client = LLMClient(
            models_config_path=models_yaml,
            completion_fn=make_completion_fn("gen result"),
        )
        config = client.get_model_config("generation")
        assert config.model == "test-mid"

    def test_routes_to_narration_model(self, models_yaml):
        client = LLMClient(
            models_config_path=models_yaml,
            completion_fn=make_completion_fn("narration result"),
        )
        config = client.get_model_config("narration")
        assert config.model == "test-cheap"

    def test_routes_to_embeddings_model(self, models_yaml):
        client = LLMClient(
            models_config_path=models_yaml,
            embedding_fn=make_embedding_fn(),
        )
        config = client.get_model_config("embeddings")
        assert config.model == "test-embed"

    def test_routes_to_escalation_model(self, models_yaml):
        client = LLMClient(
            models_config_path=models_yaml,
            completion_fn=make_completion_fn("escalation result"),
        )
        config = client.get_model_config("escalation")
        assert config.model == "test-strong"

    def test_unknown_role_falls_back_to_generation(self, models_yaml):
        client = LLMClient(
            models_config_path=models_yaml,
            completion_fn=make_completion_fn("fallback result"),
        )
        config = client.get_model_config("unknown_role")
        assert config.model == "test-mid"  # Falls back to generation

    def test_litellm_model_id_format(self):
        config = ModelConfig(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            credentials_env="ANTHROPIC_API_KEY",
        )
        assert config.get_litellm_model_id() == "anthropic/claude-sonnet-4-20250514"

    def test_local_model_id_format(self):
        config = ModelConfig(
            provider="local",
            model="llama-3-8b",
            credentials_env="",
        )
        # Local models don't use provider/ prefix
        assert config.get_litellm_model_id() == "llama-3-8b"


class TestGenerate:
    """Test text completion generation."""

    def test_generate_returns_text(self, models_yaml):
        client = LLMClient(
            models_config_path=models_yaml,
            completion_fn=make_completion_fn("Hello world"),
        )
        result = client.generate(
            role="intent",
            messages=[{"role": "user", "content": "Say hello"}],
        )
        assert result == "Hello world"

    def test_generate_passes_model_to_fn(self, models_yaml):
        captured = {}
        def capture_fn(model, messages, **kwargs):
            captured["model"] = model
            return "ok"

        client = LLMClient(
            models_config_path=models_yaml,
            completion_fn=capture_fn,
        )
        client.generate(
            role="generation",
            messages=[{"role": "user", "content": "test"}],
        )
        assert captured["model"] == "test_provider/test-mid"


class TestGenerateStructured:
    """Test structured/JSON output generation."""

    def test_structured_output_parsed(self, models_yaml):
        client = LLMClient(
            models_config_path=models_yaml,
            completion_fn=make_completion_fn('{"status": "ok", "count": 42}'),
        )
        result = client.generate_structured(
            role="intent",
            messages=[{"role": "user", "content": "Return JSON"}],
        )
        assert result["status"] == "ok"
        assert result["count"] == 42

    def test_structured_output_strips_markdown_fences(self, models_yaml):
        client = LLMClient(
            models_config_path=models_yaml,
            completion_fn=make_completion_fn('```json\n{"key": "value"}\n```'),
        )
        result = client.generate_structured(
            role="intent",
            messages=[{"role": "user", "content": "Return JSON"}],
        )
        assert result["key"] == "value"

    def test_invalid_json_raises(self, models_yaml):
        client = LLMClient(
            models_config_path=models_yaml,
            completion_fn=make_completion_fn("not valid json"),
        )
        with pytest.raises(json.JSONDecodeError):
            client.generate_structured(
                role="intent",
                messages=[{"role": "user", "content": "Return JSON"}],
            )


class TestEmbed:
    """Test embedding generation."""

    def test_embed_returns_vectors(self, models_yaml):
        client = LLMClient(
            models_config_path=models_yaml,
            embedding_fn=make_embedding_fn(),
        )
        result = client.embed(["hello", "world"])
        assert len(result) == 2
        assert len(result[0]) == 3

    def test_embed_uses_embeddings_role(self, models_yaml):
        captured = {}
        def capture_fn(model, texts):
            captured["model"] = model
            return [[0.1]] * len(texts)

        client = LLMClient(
            models_config_path=models_yaml,
            embedding_fn=capture_fn,
        )
        client.embed(["test"])
        assert captured["model"] == "test_provider/test-embed"


class TestRetryWithBackoff:
    """Test retry and backoff behavior."""

    def test_retries_on_transient_failure(self, models_yaml):
        call_count = 0

        def flaky_fn(model, messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Transient error")
            return "success"

        client = LLMClient(
            models_config_path=models_yaml,
            completion_fn=flaky_fn,
        )
        # Patch time.sleep to avoid actual waits
        with patch("src.agent.llm_client.time.sleep"):
            result = client.generate(
                role="intent",
                messages=[{"role": "user", "content": "test"}],
            )
        assert result == "success"
        assert call_count == 3

    def test_raises_model_unavailable_after_max_retries(self, models_yaml):
        client = LLMClient(
            models_config_path=models_yaml,
            completion_fn=make_failing_completion_fn(),
        )
        with patch("src.agent.llm_client.time.sleep"):
            with pytest.raises(ModelUnavailableError) as exc_info:
                client.generate(
                    role="intent",
                    messages=[{"role": "user", "content": "test"}],
                )
        assert "refused: model unavailable" in str(exc_info.value)

    def test_no_silent_hang_on_failure(self, models_yaml):
        """Per Rules.md §4: never a silent hang or fabricated answer."""
        client = LLMClient(
            models_config_path=models_yaml,
            completion_fn=make_failing_completion_fn(),
        )
        with patch("src.agent.llm_client.time.sleep"):
            with pytest.raises(ModelUnavailableError):
                client.generate(
                    role="intent",
                    messages=[{"role": "user", "content": "test"}],
                )
        # If we get here, the call raised properly — no hang

    def test_permanent_error_not_retried(self, models_yaml):
        call_count = 0

        def permanent_fn(model, messages, **kwargs):
            nonlocal call_count
            call_count += 1
            raise ModelUnavailableError("permanently gone")

        client = LLMClient(
            models_config_path=models_yaml,
            completion_fn=permanent_fn,
        )
        with pytest.raises(ModelUnavailableError):
            client.generate(
                role="intent",
                messages=[{"role": "user", "content": "test"}],
            )
        assert call_count == 1  # Not retried


class TestCapabilitySelfTest:
    """Test capability self-test grading."""

    def test_compliant_model_passes(self, models_yaml):
        """Self-test correctly grades a compliant model as pass."""
        def smart_fn(model, messages, **kwargs):
            content = messages[-1]["content"]
            if "status" in content and "42" in content:
                return '{"status": "ok", "number": 42}'
            if "CANNOT_ANSWER" in content:
                return "CANNOT_ANSWER - this question is not related to the database schema."
            return "some response"

        client = LLMClient(
            models_config_path=models_yaml,
            completion_fn=smart_fn,
        )
        result = client.self_test("intent")
        assert result.grade == "pass"
        assert result.is_usable is True

    def test_malformed_stub_fails(self, models_yaml):
        """Self-test correctly fails a stub model that returns malformed output."""
        def bad_fn(model, messages, **kwargs):
            # Returns garbage for everything
            return "not json, not a decline, just garbage"

        client = LLMClient(
            models_config_path=models_yaml,
            completion_fn=bad_fn,
        )
        result = client.self_test("intent")
        # Should be degraded or fail since structured output fails
        assert result.grade in ("degraded", "fail")

    def test_model_that_crashes_fails(self, models_yaml):
        """A model that errors out on self-test gets grade 'fail'."""
        client = LLMClient(
            models_config_path=models_yaml,
            completion_fn=make_failing_completion_fn(),
        )
        with patch("src.agent.llm_client.time.sleep"):
            result = client.self_test("intent")
        assert result.grade == "fail"
        assert result.is_usable is False

    def test_degraded_model_still_usable(self, models_yaml):
        """A degraded model is usable but with warnings."""
        def partial_fn(model, messages, **kwargs):
            content = messages[-1]["content"]
            if "status" in content and "42" in content:
                return '{"status": "ok", "number": 42}'
            # Doesn't properly decline unanswerable questions
            return "The airspeed velocity is 42 km/h"

        client = LLMClient(
            models_config_path=models_yaml,
            completion_fn=partial_fn,
        )
        result = client.self_test("intent")
        assert result.grade == "degraded"
        assert result.is_usable is True

    def test_self_test_result_cached(self, models_yaml):
        client = LLMClient(
            models_config_path=models_yaml,
            completion_fn=make_completion_fn('{"status": "ok", "number": 42}'),
        )
        client.self_test("intent")
        cached = client.get_self_test_result("intent")
        assert cached is not None
        assert cached.role == "intent"


class TestConfigLoading:
    """Test models.yaml config loading."""

    def test_loads_from_yaml(self, models_yaml):
        client = LLMClient(models_config_path=models_yaml)
        config = client.get_model_config("intent")
        assert config.provider == "test_provider"
        assert config.model == "test-cheap"

    def test_missing_config_uses_defaults(self, tmp_path):
        nonexistent = tmp_path / "nonexistent.yaml"
        client = LLMClient(models_config_path=nonexistent)
        config = client.get_model_config("generation")
        # Should fall back to hardcoded defaults
        assert config.provider is not None

    def test_no_config_path_works(self):
        client = LLMClient(completion_fn=make_completion_fn("ok"))
        result = client.generate(
            role="intent",
            messages=[{"role": "user", "content": "test"}],
        )
        assert result == "ok"


class TestProviderAbstraction:
    """Test that the abstraction supports switching providers."""

    def test_two_different_providers(self, tmp_path):
        """Verify same interface works with different provider configs."""
        # Config A: provider_a
        config_a = tmp_path / "models_a.yaml"
        config_a.write_text("""
default_provider: "provider_a"
roles:
  generation:
    provider: "provider_a"
    model: "model-a"
    credentials_env: "KEY_A"
""")

        # Config B: provider_b
        config_b = tmp_path / "models_b.yaml"
        config_b.write_text("""
default_provider: "provider_b"
roles:
  generation:
    provider: "provider_b"
    model: "model-b"
    credentials_env: "KEY_B"
""")

        captured_models = []

        def capture_fn(model, messages, **kwargs):
            captured_models.append(model)
            return "response"

        # Client A
        client_a = LLMClient(
            models_config_path=config_a,
            completion_fn=capture_fn,
        )
        client_a.generate(
            role="generation",
            messages=[{"role": "user", "content": "test"}],
        )

        # Client B
        client_b = LLMClient(
            models_config_path=config_b,
            completion_fn=capture_fn,
        )
        client_b.generate(
            role="generation",
            messages=[{"role": "user", "content": "test"}],
        )

        # They should route to different models
        assert captured_models[0] == "provider_a/model-a"
        assert captured_models[1] == "provider_b/model-b"
