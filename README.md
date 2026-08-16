# DBAgent

A text-to-SQL agent that prioritizes honesty over accuracy — showing its work, flagging uncertainty, and refusing rather than guessing when it lacks signal to answer safely.

## Features

- **Intent Reconstruction** — Restates questions against a glossary; asks clarifying questions when ambiguous
- **Schema Retrieval** — Embedding-based retrieval of relevant schema slices with drift detection
- **SQL Generation** — LLM-powered SQL generation with sqlglot validation
- **Verification** — EXPLAIN integration, join-cardinality checks, hallucinated reference detection
- **Confidence Gate** — Combines schema-match, join-confidence, and result-sanity into a refusal/flag decision
- **Execution & Narration** — Read-only execution with programmatic value computation (no LLM-recalled numbers)
- **Correction Memory** — Persists user corrections as scoped rules (org/db-keyed)
- **Provider Abstraction** — LiteLLM-based unified interface for any LLM provider (Anthropic, OpenAI, Google, local models)
- **Multi-DB Key Handling** — Fuzzy key matching with confirm-once-remember flow
- **Concurrency Guard** — Query timeout + circuit breaker under load

## Tech Stack

- **Python 3.11+**
- **LLM:** Provider-agnostic via LiteLLM (`config/models.yaml`)
- **SQL parsing:** sqlglot
- **DB connectivity:** SQLAlchemy (read-only enforced)
- **Testing:** pytest

## Setup

```bash
# Clone
git clone https://github.com/mukti-sys/DBAgent.git
cd DBAgent

# Install dependencies
pip install -r requirements.txt

# Configure
# Set your API key
export ANTHROPIC_API_KEY="your-key-here"
# Or configure any provider in config/models.yaml

# Run tests
pytest tests/unit
```

## Project Structure

```
├── src/
│   ├── agent/          # Core agent modules
│   │   ├── llm_client.py        # Unified LLM interface
│   │   ├── intent.py            # Intent reconstruction
│   │   ├── schema_retrieval.py  # Schema indexing & retrieval
│   │   ├── sql_generator.py     # SQL generation
│   │   ├── verifier.py          # SQL verification
│   │   ├── confidence.py        # Confidence scoring
│   │   ├── executor.py          # Query execution
│   │   ├── narrator.py          # Result narration
│   │   ├── correction_memory.py # Correction persistence
│   │   ├── glossary.py          # Glossary store
│   │   ├── key_resolver.py      # Multi-DB key resolution
│   │   └── concurrency_guard.py # Concurrency protection
│   ├── db/             # Database connectivity
│   └── interface/      # CLI interface
├── tests/
│   ├── unit/           # Unit tests per module
│   ├── integration/    # Full pipeline tests
│   ├── safety/         # Security tests
│   └── eval/           # Accuracy/calibration eval
├── config/
│   ├── settings.yaml   # Thresholds & connection settings
│   ├── glossary.yaml   # Business term definitions
│   └── models.yaml     # Per-role model configuration
└── requirements.txt
```

## Running Tests

```bash
pytest tests/unit                # Unit tests
pytest tests/integration         # Integration tests (requires test DB)
pytest tests/safety              # Safety tests
pytest tests/eval                # Eval harness
```

## License

MIT
