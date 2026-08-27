# DBAgent

A text-to-SQL agent that prioritizes **honesty over accuracy** — showing its work, flagging uncertainty, and refusing rather than guessing when it lacks signal to answer safely.

## Features

| Module | What It Does |
|--------|-------------|
| **Intent Reconstruction** | Restates questions against a glossary; asks clarifying questions when ambiguous |
| **Schema Retrieval** | Embedding-based retrieval of relevant schema slices with drift detection |
| **SQL Generation** | LLM-powered SQL generation with sqlglot validation |
| **Verification** | EXPLAIN integration, join-cardinality checks, hallucinated reference detection |
| **Confidence Gate** | Combines schema-match, join-confidence, and result-sanity into a refusal/flag decision |
| **Execution & Narration** | Read-only execution with programmatic value computation (no LLM-recalled numbers) |
| **Correction Memory** | Persists user corrections as scoped rules (org/db-keyed) |
| **Provider Abstraction** | LiteLLM-based unified interface for any LLM provider |
| **Multi-DB Key Handling** | Fuzzy key matching with confirm-once-remember flow |
| **Concurrency Guard** | Query timeout + circuit breaker under load |
| **Multi-Turn State** | Session-scoped filter/context tracking across follow-up questions |
| **Eval Harness** | Execution-match + semantic-match scoring with accuracy & calibration reports |
| **Injection Defense** | Detects and neutralizes prompt injection payloads in DB data fields |

## Tech Stack

- **Python 3.11+**
- **LLM:** Provider-agnostic via LiteLLM (`config/models.yaml`)
- **SQL parsing:** sqlglot
- **DB connectivity:** SQLAlchemy (read-only enforced)
- **Testing:** pytest

## Quick Start

```bash
# Clone
git clone https://github.com/mukti-sys/DBAgent.git
cd DBAgent

# Create virtual environment
python -m venv venv
source venv/bin/activate     # Linux/Mac
# venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Configure your LLM provider
# Option 1: Set API key
export ANTHROPIC_API_KEY="your-key-here"
# Option 2: Edit config/models.yaml for any provider

# Verify installation
pytest tests/unit
```

## Configuration

### LLM Provider (`config/models.yaml`)

Route different agent roles to different models:

```yaml
roles:
  intent:
    provider: anthropic
    model: claude-3-haiku-20240307
  generation:
    provider: anthropic
    model: claude-sonnet-4-20250514
  narration:
    provider: anthropic
    model: claude-3-haiku-20240307
  verification:
    provider: anthropic
    model: claude-sonnet-4-20250514
  embeddings:
    provider: openai
    model: text-embedding-3-small
```

### Glossary (`config/glossary.yaml`)

Map business terms to schema concepts:

```yaml
terms:
  revenue: orders.amount
  active user: users where last_login > 30 days ago
  churn rate: percentage of users who cancelled in period
```

### Settings (`config/settings.yaml`)

Tune thresholds for confidence scoring, timeouts, and circuit breaker behavior.

## Project Structure

```
├── src/
│   ├── agent/                    # Core agent modules
│   │   ├── llm_client.py         # Unified LLM interface (LiteLLM)
│   │   ├── intent.py             # Intent reconstruction & glossary resolution
│   │   ├── schema_retrieval.py   # Schema indexing, retrieval & drift detection
│   │   ├── sql_generator.py      # SQL generation with sqlglot validation
│   │   ├── verifier.py           # SQL verification (EXPLAIN, joins, hallucinations)
│   │   ├── confidence.py         # Confidence scoring & refusal gate
│   │   ├── executor.py           # Read-only query execution
│   │   ├── narrator.py           # Result narration (template + real values)
│   │   ├── correction_memory.py  # Correction persistence (org/db-scoped)
│   │   ├── glossary.py           # Glossary store
│   │   ├── key_resolver.py       # Multi-DB fuzzy key resolution
│   │   ├── concurrency_guard.py  # Query timeout & circuit breaker
│   │   ├── conversation_state.py # Multi-turn session context
│   │   ├── eval_harness.py       # Eval scoring & calibration reporting
│   │   └── injection_defense.py  # Prompt injection detection & sanitization
│   ├── db/                       # Database connectivity
│   │   ├── access_control.py     # Read-only enforcement & DDL/DML blocking
│   │   └── connector.py          # SQLAlchemy connection management
│   └── interface/
│       └── cli.py                # CLI interface
├── tests/
│   ├── unit/                     # Unit tests (one per module)
│   ├── integration/              # Full pipeline tests (requires test DB)
│   ├── safety/                   # Security & access control tests
│   └── eval/                     # Eval harness runner
├── config/
│   ├── settings.yaml             # Thresholds & connection settings
│   ├── glossary.yaml             # Business term definitions
│   └── models.yaml               # Per-role LLM model configuration
└── requirements.txt
```

## Running Tests

```bash
# All unit tests
pytest tests/unit

# Specific module
pytest tests/unit/test_sql_generator.py -v

# Safety tests (must never fail)
pytest tests/safety

# Integration tests (requires seeded test DB)
pytest tests/integration

# Eval harness (produces accuracy/calibration report)
pytest tests/eval

# Full suite
pytest
```

## Design Principles

1. **Honesty over accuracy** — The agent refuses or flags when uncertain rather than guessing
2. **Show your work** — Every answer traces back to actual query results, never LLM-recalled numbers
3. **Read-only by default** — No writes to the database unless explicitly enabled
4. **Data is never instructions** — DB field content is sanitized before entering LLM context
5. **Confirm, don't assume** — Ambiguous joins, unclear references, and vague questions trigger clarification
6. **Provider-agnostic** — Swap LLM providers via config without code changes

## Architecture

```
User Question
     │
     ▼
┌─────────────┐    ┌──────────────┐
│ Intent      │───▶│ Schema       │
│ Reconstruct │    │ Retrieval    │
└─────────────┘    └──────────────┘
     │                    │
     ▼                    ▼
┌─────────────┐    ┌──────────────┐
│ SQL         │───▶│ Verification │
│ Generation  │    │ (EXPLAIN)    │
└─────────────┘    └──────────────┘
                         │
                         ▼
                  ┌──────────────┐
                  │ Confidence   │
                  │ Gate         │
                  └──────────────┘
                    │         │
              ┌─────┘         └──────┐
              ▼                      ▼
        ┌──────────┐          ┌────────────┐
        │ Execute  │          │ Refuse /   │
        │ & Narrate│          │ Flag       │
        └──────────┘          └────────────┘
```

## License

MIT
