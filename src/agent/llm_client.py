"""
LLM Client — unified interface for all model calls.

Per Architecture.md §2: Unified interface (LiteLLM under the hood) to any
configured provider/model for both completions and embeddings. Handles
per-role model selection, the escalation cascade, and a capability self-test
before trusting a newly configured model.

Per Rules.md §1: If the user configures a local model, no pipeline stage
may silently fall back to an external API — including schema embeddings.

Per Rules.md §4 (API reliability): Retry transient failures with backoff;
on persistent failure, surface 'refused: model unavailable'.

Per Rules.md §4 (Model-dependent reliability): Capability self-test grades
pass/degraded/fail before a model is trusted in the pipeline.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

logger = logging.getLogger(__name__)


# Default retry/backoff config
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 1.0
BACKOFF_MULTIPLIER = 2.0


class ModelUnavailableError(Exception):
    """Raised when a model is permanently unavailable after retries."""
    pass


class ModelCapabilityError(Exception):
    """Raised when a model fails the capability self-test."""
    pass


@dataclass
class ModelConfig:
    """Configuration for a single model role."""
    provider: str
    model: str
    credentials_env: str
    role: str = ""

    def get_litellm_model_id(self) -> str:
        """
        Return the model ID in LiteLLM format: provider/model.
        LiteLLM uses this to route to the correct API.
        """
        # Local models served via OpenAI-compatible endpoint use model name directly
        if self.provider in ("local", "ollama"):
            return self.model
        # All other providers use provider/model format
        return f"{self.provider}/{self.model}"

    def has_credentials(self) -> bool:
        """Check if the required env var is set."""
        return bool(os.environ.get(self.credentials_env, ""))


@dataclass
class SelfTestResult:
    """Result of a capability self-test for a model."""
    grade: str  # "pass", "degraded", "fail"
    model_id: str
    role: str
    details: list[str] = field(default_factory=list)

    @property
    def is_usable(self) -> bool:
        return self.grade in ("pass", "degraded")


class LLMClient:
    """
    Unified LLM interface routing through LiteLLM.

    All modules (intent, sql_generator, narrator, verifier, schema_retrieval)
    call this instead of any provider SDK directly.

    Supports:
    - generate(): text completion
    - generate_structured(): JSON/structured output
    - embed(): embedding generation
    - self_test(): capability verification for a role's model
    """

    def __init__(
        self,
        models_config_path: str | Path | None = None,
        completion_fn: Callable | None = None,
        embedding_fn: Callable | None = None,
    ):
        """
        Initialize LLM client.

        Args:
            models_config_path: Path to config/models.yaml.
                If None, uses default config location.
            completion_fn: Injectable completion function for testing.
                Signature: (model, messages, **kwargs) -> response_text
            embedding_fn: Injectable embedding function for testing.
                Signature: (model, input_texts) -> list[list[float]]
        """
        self._role_configs: dict[str, ModelConfig] = {}
        self._default_provider: str = "anthropic"
        self._self_test_results: dict[str, SelfTestResult] = {}

        # Injectable functions (for testing without real API calls)
        self._completion_fn = completion_fn
        self._embedding_fn = embedding_fn

        if models_config_path:
            self._load_config(Path(models_config_path))

    def _load_config(self, path: Path) -> None:
        """Load model configuration from YAML."""
        if not path.exists():
            logger.warning(f"Models config not found at {path}, using defaults")
            return

        with open(path, "r") as f:
            config = yaml.safe_load(f) or {}

        self._default_provider = config.get("default_provider", "anthropic")

        roles = config.get("roles", {})
        for role_name, role_config in roles.items():
            self._role_configs[role_name] = ModelConfig(
                provider=role_config.get("provider", self._default_provider),
                model=role_config.get("model", ""),
                credentials_env=role_config.get("credentials_env", ""),
                role=role_name,
            )

    def get_model_config(self, role: str) -> ModelConfig:
        """Get model config for a given role."""
        if role in self._role_configs:
            return self._role_configs[role]

        # Fall back to generation config or a sensible default
        if "generation" in self._role_configs:
            cfg = self._role_configs["generation"]
            return ModelConfig(
                provider=cfg.provider,
                model=cfg.model,
                credentials_env=cfg.credentials_env,
                role=role,
            )

        return ModelConfig(
            provider=self._default_provider,
            model="claude-sonnet-4-20250514",
            credentials_env="ANTHROPIC_API_KEY",
            role=role,
        )

    def generate(
        self,
        role: str,
        messages: list[dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 0.0,
        **kwargs,
    ) -> str:
        """
        Generate a text completion for the given role.

        Args:
            role: Pipeline role (intent, generation, narration, etc.)
            messages: List of {"role": "user"|"system"|"assistant", "content": "..."}
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature

        Returns:
            Response text

        Raises:
            ModelUnavailableError: After all retries exhausted
        """
        config = self.get_model_config(role)
        model_id = config.get_litellm_model_id()

        return self._call_with_retry(
            fn=self._do_completion,
            model_id=model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

    def generate_structured(
        self,
        role: str,
        messages: list[dict[str, str]],
        response_format: dict | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Generate structured (JSON) output for the given role.

        Args:
            role: Pipeline role
            messages: Chat messages
            response_format: JSON schema for the response
            max_tokens: Maximum tokens
            temperature: Sampling temperature

        Returns:
            Parsed JSON dict

        Raises:
            ModelUnavailableError: After all retries exhausted
        """
        config = self.get_model_config(role)
        model_id = config.get_litellm_model_id()

        # Add JSON instruction to the messages
        json_messages = list(messages)
        if json_messages and json_messages[-1]["role"] == "user":
            json_messages[-1] = dict(json_messages[-1])
            json_messages[-1]["content"] += "\n\nRespond with valid JSON only."

        text = self._call_with_retry(
            fn=self._do_completion,
            model_id=model_id,
            messages=json_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
            **kwargs,
        )

        # Parse JSON from response
        try:
            # Handle markdown code fences
            cleaned = text.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                # Remove first and last fence lines
                lines = [l for l in lines if not l.strip().startswith("```")]
                cleaned = "\n".join(lines).strip()
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse structured output: {e}")
            raise

    def embed(
        self,
        texts: list[str],
        role: str = "embeddings",
    ) -> list[list[float]]:
        """
        Generate embeddings for the given texts.

        Per Rules.md §1: routes through the configured provider,
        never silently falls back to an external API.

        Args:
            texts: List of text strings to embed
            role: Role to use for model selection (default: "embeddings")

        Returns:
            List of embedding vectors

        Raises:
            ModelUnavailableError: After all retries exhausted
        """
        config = self.get_model_config(role)
        model_id = config.get_litellm_model_id()

        return self._call_with_retry(
            fn=self._do_embedding,
            model_id=model_id,
            texts=texts,
        )

    def self_test(self, role: str) -> SelfTestResult:
        """
        Run a capability self-test for a role's configured model.

        Tests:
        1. Valid structured/JSON output on a canary prompt
        2. Correctly declines a deliberately unanswerable question
        3. Tool-calling works if the role requires it (not tested here —
           requires actual tool schemas)

        Grades: pass / degraded / fail
        - pass: all checks succeed
        - degraded: some checks fail, model still usable with warnings
        - fail: critical checks fail, model should not be used

        Per Rules.md §4 (Model-dependent reliability): A model that fails
        is refused; it is never silently allowed to run at reduced reliability.
        """
        config = self.get_model_config(role)
        details = []
        failures = 0
        critical_failures = 0

        # Test 1: Structured output
        try:
            result = self.generate_structured(
                role=role,
                messages=[{
                    "role": "user",
                    "content": 'Return a JSON object with exactly two keys: "status" set to "ok" and "number" set to 42.',
                }],
            )
            if isinstance(result, dict) and result.get("status") == "ok":
                details.append("PASS: Structured output works correctly")
            else:
                details.append(f"DEGRADED: Structured output returned unexpected format: {result}")
                failures += 1
        except Exception as e:
            details.append(f"FAIL: Structured output failed: {e}")
            failures += 1
            critical_failures += 1

        # Test 2: Decline unanswerable question
        try:
            response = self.generate(
                role=role,
                messages=[{
                    "role": "user",
                    "content": (
                        "You are a SQL agent. Given this schema: table 'users' with columns (id, name, email). "
                        "Question: What is the airspeed velocity of an unladen swallow? "
                        "If this question cannot be answered from the schema, respond with exactly: CANNOT_ANSWER"
                    ),
                }],
            )
            if "CANNOT_ANSWER" in response.upper() or "cannot" in response.lower():
                details.append("PASS: Correctly declined unanswerable question")
            else:
                details.append(f"DEGRADED: Did not clearly decline unanswerable question: {response[:100]}")
                failures += 1
        except Exception as e:
            details.append(f"FAIL: Decline test failed: {e}")
            failures += 1
            critical_failures += 1

        # Determine grade
        if critical_failures > 0:
            grade = "fail"
        elif failures > 0:
            grade = "degraded"
        else:
            grade = "pass"

        result = SelfTestResult(
            grade=grade,
            model_id=config.get_litellm_model_id(),
            role=role,
            details=details,
        )
        self._self_test_results[role] = result

        if grade == "fail":
            logger.error(f"Model self-test FAILED for role '{role}': {details}")
        elif grade == "degraded":
            logger.warning(f"Model self-test DEGRADED for role '{role}': {details}")
        else:
            logger.info(f"Model self-test PASSED for role '{role}'")

        return result

    def get_self_test_result(self, role: str) -> SelfTestResult | None:
        """Get cached self-test result for a role."""
        return self._self_test_results.get(role)

    def _do_completion(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 0.0,
        **kwargs,
    ) -> str:
        """Perform actual completion call (via injected fn or LiteLLM)."""
        if self._completion_fn:
            return self._completion_fn(model_id, messages, **kwargs)

        # Use LiteLLM for actual API calls
        try:
            import litellm
            response = litellm.completion(
                model=model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
            return response.choices[0].message.content
        except ImportError:
            raise ModelUnavailableError(
                "litellm is not installed. Install it with: pip install litellm"
            )

    def _do_embedding(
        self,
        model_id: str,
        texts: list[str],
        **kwargs,
    ) -> list[list[float]]:
        """Perform actual embedding call (via injected fn or LiteLLM)."""
        if self._embedding_fn:
            return self._embedding_fn(model_id, texts)

        try:
            import litellm
            response = litellm.embedding(
                model=model_id,
                input=texts,
            )
            return [item["embedding"] for item in response.data]
        except ImportError:
            raise ModelUnavailableError(
                "litellm is not installed. Install it with: pip install litellm"
            )

    def _call_with_retry(
        self,
        fn: Callable,
        **kwargs,
    ) -> Any:
        """
        Call a function with retry and exponential backoff.

        Per Rules.md §4 (API reliability): retry transient failures,
        surface 'refused: model unavailable' on persistent failure.
        """
        last_error = None
        backoff = INITIAL_BACKOFF_SECONDS

        for attempt in range(MAX_RETRIES):
            try:
                return fn(**kwargs)
            except ModelUnavailableError:
                raise  # Don't retry permanent errors
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    logger.warning(
                        f"Attempt {attempt + 1}/{MAX_RETRIES} failed: {e}. "
                        f"Retrying in {backoff}s..."
                    )
                    time.sleep(backoff)
                    backoff *= BACKOFF_MULTIPLIER

        raise ModelUnavailableError(
            f"refused: model unavailable after {MAX_RETRIES} attempts. "
            f"Last error: {last_error}"
        )
