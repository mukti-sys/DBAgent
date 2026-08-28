"""
LLM Client tests — updated for multi-provider architecture.

Tests:
- Model routing per role
- Custom provider with base_url
- Local model (Ollama) support
- Provider/model listing
- Structured output parsing
- Retry with backoff
- Capability self-test grading
"""

import json
import pytest
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.agent.llm_client import (
    LLMClient,
    ModelConfig,
    ProviderConfig,
    ModelEntry,
    ModelUnavailableError,
    ModelCapabilityError,
    SelfTestResult,
    MAX_RETRIES,
)


@pytest.fixture
def models_yaml(tmp_path):
    """Create a test models.yaml with new 3-section format."""
    config = tmp_path / "models.yaml"
    config.write_text("""
providers:
  test_provider:
    type: "test_provider"
    credentials_env: "TEST_API_KEY"
  test_local:
    type: "ollama"
    base_url: "http://localhost:11434"
    credentials_env: ""

models:
  - id: "test-cheap"
    provider: "test_provider"
    tier: "cheap"
    capabilities: ["completion", "structured_output"]
  - id: "test-mid"
    provider: "test_provider"
    tier: "mid"
    capabilities: ["completion", "structured_output"]
  - id: "test-embed"
    provider: "test_provider"
    tier: "cheap"
    capabilities: ["embeddings"]
  - id: "test-strong"
    provider: "test_provider"
    tier: "expensive"
    capabilities: ["completion", "structured_output"]
  - id: "local-model"
    provider: "test_local"
    tier: "free"
    capabilities: ["completion"]

roles:
  intent:
    model: "test-cheap"
  generation:
    model: "test-mid"
  narration:
    model: "test-cheap"
  verification:
    model: "test-cheap"
  embeddings:
    model: "test-embed"
  escalation:
    model: "test-strong"
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


class TestLiteLLMModelId:
    """Test model ID formatting for LiteLLM."""

    def test_standard_provider_format(self):
        prov = ProviderConfig(name="anthropic", type="anthropic", credentials_env="KEY")
        entry = ModelEntry(id="claude-sonnet-4-20250514", provider="anthropic")
        config = ModelConfig(provider_config=prov, model_entry=entry)
        assert config.get_litellm_model_id() == "anthropic/claude-sonnet-4-20250514"

    def test_ollama_format(self):
        prov = ProviderConfig(name="ollama", type="ollama", base_url="http://localhost:11434")
        entry = ModelEntry(id="qwen3:8b", provider="ollama")
        config = ModelConfig(provider_config=prov, model_entry=entry)
        assert config.get_litellm_model_id() == "ollama/qwen3:8b"

    def test_custom_openai_compatible(self):
        prov = ProviderConfig(
            name="custom", type="openai",
            credentials_env="CUSTOM_KEY",
            base_url="https://my-server.com/v1",
        )
        entry = ModelEntry(id="my-model", provider="custom")
        config = ModelConfig(provider_config=prov, model_entry=entry)
        assert config.get_litellm_model_id() == "openai/my-model"

    def test_gemini_format(self):
        prov = ProviderConfig(name="google", type="gemini", credentials_env="KEY")
        entry = ModelEntry(id="gemini-2.5-flash", provider="google")
        config = ModelConfig(provider_config=prov, model_entry=entry)
        assert config.get_litellm_model_id() == "gemini/gemini-2.5-flash"


class TestProviderConfig:
    """Test provider configuration and credential detection."""

    def test_local_provider_detected(self):
        prov = ProviderConfig(name="ollama", type="ollama", base_url="http://localhost:11434")
        assert prov.is_local is True

    def test_cloud_provider_not_local(self):
        prov = ProviderConfig(name="anthropic", type="anthropic", credentials_env="KEY")
        assert prov.is_local is False

    def test_localhost_base_url_is_local(self):
        prov = ProviderConfig(
            name="custom", type="openai",
            base_url="http://localhost:1234/v1",
        )
        assert prov.is_local is True

    def test_local_no_credentials_needed(self):
        prov = ProviderConfig(name="ollama", type="ollama", credentials_env="")
        assert prov.has_credentials is True

    def test_cloud_needs_credentials(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        prov = ProviderConfig(name="anthropic", type="anthropic", credentials_env="ANTHROPIC_API_KEY")
        assert prov.has_credentials is False

    def test_cloud_with_credentials(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")
        prov = ProviderConfig(name="anthropic", type="anthropic", credentials_env="ANTHROPIC_API_KEY")
        assert prov.has_credentials is True


class TestLiteLLMKwargs:
    """Test that extra kwargs for LiteLLM calls are built correctly."""

    def test_base_url_passed(self):
        prov = ProviderConfig(
            name="custom", type="openai",
            base_url="https://my-server.com/v1",
            credentials_env="",
        )
        entry = ModelEntry(id="model", provider="custom")
        config = ModelConfig(provider_config=prov, model_entry=entry)
        kwargs = config.get_litellm_kwargs()
        assert kwargs["api_base"] == "https://my-server.com/v1"

    def test_api_version_passed(self):
        prov = ProviderConfig(
            name="azure", type="azure",
            credentials_env="AZURE_KEY",
            api_version="2024-02-01",
        )
        entry = ModelEntry(id="gpt-4o", provider="azure")
        config = ModelConfig(provider_config=prov, model_entry=entry)
        kwargs = config.get_litellm_kwargs()
        assert kwargs["api_version"] == "2024-02-01"

    def test_local_gets_dummy_key(self):
        prov = ProviderConfig(name="ollama", type="ollama", credentials_env="")
        entry = ModelEntry(id="qwen3:8b", provider="ollama")
        config = ModelConfig(provider_config=prov, model_entry=entry)
        kwargs = config.get_litellm_kwargs()
        assert kwargs["api_key"] == "local"

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("MY_KEY", "sk-real-key")
        prov = ProviderConfig(name="openai", type="openai", credentials_env="MY_KEY")
        entry = ModelEntry(id="gpt-4o", provider="openai")
        config = ModelConfig(provider_config=prov, model_entry=entry)
        kwargs = config.get_litellm_kwargs()
        assert kwargs["api_key"] == "sk-real-key"


class TestListProviders:
    """Test provider and model listing."""

    def test_list_providers(self, models_yaml):
        client = LLMClient(models_config_path=models_yaml)
        providers = client.list_providers()
        names = [p["name"] for p in providers]
        assert "test_provider" in names
        assert "test_local" in names

    def test_list_models(self, models_yaml):
        client = LLMClient(models_config_path=models_yaml)
        models = client.list_available_models()
        ids = [m["id"] for m in models]
        assert "test-cheap" in ids
        assert "test-mid" in ids
        assert "local-model" in ids

    def test_list_role_assignments(self, models_yaml):
        client = LLMClient(models_config_path=models_yaml)
        roles = client.list_role_assignments()
        assert roles["intent"] == "test-cheap"
        assert roles["generation"] == "test-mid"

    def test_model_availability_check(self, models_yaml):
        client = LLMClient(models_config_path=models_yaml)
        models = client.list_available_models()
        # Local model should be available (no creds needed)
        local = next(m for m in models if m["id"] == "local-model")
        assert local["available"] is True


class TestCustomProvider:
    """Test custom/third-party provider with base_url."""

    def test_custom_provider_config(self, tmp_path):
        config = tmp_path / "models.yaml"
        config.write_text("""
providers:
  openrouter:
    type: "openrouter"
    credentials_env: "OPENROUTER_API_KEY"
    base_url: "https://openrouter.ai/api/v1"

models:
  - id: "google/gemma-2-9b-it"
    provider: "openrouter"
    tier: "cheap"
    capabilities: ["completion"]

roles:
  generation:
    model: "google/gemma-2-9b-it"
""")
        client = LLMClient(models_config_path=config)
        mc = client.get_model_config("generation")
        assert mc.model == "google/gemma-2-9b-it"
        assert mc.provider_config.base_url == "https://openrouter.ai/api/v1"
        kwargs = mc.get_litellm_kwargs()
        assert kwargs["api_base"] == "https://openrouter.ai/api/v1"


class TestLocalModel:
    """Test local model (Ollama) configuration."""

    def test_local_model_routing(self, tmp_path):
        config = tmp_path / "models.yaml"
        config.write_text("""
providers:
  ollama:
    type: "ollama"
    base_url: "http://localhost:11434"
    credentials_env: ""

models:
  - id: "qwen3:8b"
    provider: "ollama"
    tier: "free"
    capabilities: ["completion"]

roles:
  generation:
    model: "qwen3:8b"
""")
        client = LLMClient(models_config_path=config)
        mc = client.get_model_config("generation")
        assert mc.model == "qwen3:8b"
        assert mc.provider_config.is_local is True
        assert mc.get_litellm_model_id() == "ollama/qwen3:8b"
        assert mc.has_credentials() is True  # Local doesn't need creds


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
        assert call_count == 1


class TestCapabilitySelfTest:
    """Test capability self-test grading."""

    def test_compliant_model_passes(self, models_yaml):
        def smart_fn(model, messages, **kwargs):
            content = messages[-1]["content"]
            if "status" in content and "42" in content:
                return '{"status": "ok", "number": 42}'
            if "CANNOT_ANSWER" in content:
                return "CANNOT_ANSWER - not a database question."
            return "some response"

        client = LLMClient(
            models_config_path=models_yaml,
            completion_fn=smart_fn,
        )
        result = client.self_test("intent")
        assert result.grade == "pass"
        assert result.is_usable is True

    def test_malformed_stub_fails(self, models_yaml):
        client = LLMClient(
            models_config_path=models_yaml,
            completion_fn=make_completion_fn("not json, not a decline, garbage"),
        )
        result = client.self_test("intent")
        assert result.grade in ("degraded", "fail")

    def test_model_that_crashes_fails(self, models_yaml):
        client = LLMClient(
            models_config_path=models_yaml,
            completion_fn=make_failing_completion_fn(),
        )
        with patch("src.agent.llm_client.time.sleep"):
            result = client.self_test("intent")
        assert result.grade == "fail"
        assert result.is_usable is False

    def test_degraded_model_still_usable(self, models_yaml):
        def partial_fn(model, messages, **kwargs):
            content = messages[-1]["content"]
            if "status" in content and "42" in content:
                return '{"status": "ok", "number": 42}'
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

    def test_loads_providers(self, models_yaml):
        client = LLMClient(models_config_path=models_yaml)
        providers = client.list_providers()
        assert len(providers) >= 2

    def test_loads_models(self, models_yaml):
        client = LLMClient(models_config_path=models_yaml)
        models = client.list_available_models()
        assert len(models) >= 4

    def test_loads_roles(self, models_yaml):
        client = LLMClient(models_config_path=models_yaml)
        roles = client.list_role_assignments()
        assert "intent" in roles
        assert "generation" in roles

    def test_missing_config_uses_defaults(self, tmp_path):
        nonexistent = tmp_path / "nonexistent.yaml"
        client = LLMClient(models_config_path=nonexistent)
        config = client.get_model_config("generation")
        assert config.model is not None

    def test_no_config_path_works(self):
        client = LLMClient(completion_fn=make_completion_fn("ok"))
        result = client.generate(
            role="intent",
            messages=[{"role": "user", "content": "test"}],
        )
        assert result == "ok"


class TestProviderAbstraction:
    """Test multi-provider switching."""

    def test_two_different_providers(self, tmp_path):
        config_a = tmp_path / "models_a.yaml"
        config_a.write_text("""
providers:
  prov_a:
    type: "prov_a"
    credentials_env: "KEY_A"
models:
  - id: "model-a"
    provider: "prov_a"
roles:
  generation:
    model: "model-a"
""")

        config_b = tmp_path / "models_b.yaml"
        config_b.write_text("""
providers:
  prov_b:
    type: "prov_b"
    credentials_env: "KEY_B"
models:
  - id: "model-b"
    provider: "prov_b"
roles:
  generation:
    model: "model-b"
""")

        captured_models = []
        def capture_fn(model, messages, **kwargs):
            captured_models.append(model)
            return "response"

        client_a = LLMClient(models_config_path=config_a, completion_fn=capture_fn)
        client_a.generate(role="generation", messages=[{"role": "user", "content": "test"}])

        client_b = LLMClient(models_config_path=config_b, completion_fn=capture_fn)
        client_b.generate(role="generation", messages=[{"role": "user", "content": "test"}])

        assert captured_models[0] == "prov_a/model-a"
        assert captured_models[1] == "prov_b/model-b"
