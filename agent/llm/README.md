# Agent LLM Layer

Unified LLM interface for the agent so you can switch providers (OpenAI, Cerebras, etc.) via configuration without code changes.

## Quick start

```python
from agent.llm import get_llm, LLMClient

# From RAG config (uses RAG_LLM_PROVIDER and API keys from env)
config = ...  # your RAG Config
llm = get_llm(config, openai_api_key=os.getenv("OPENAI_API_KEY"))

# Single call
text = llm.complete(
    [{"role": "user", "content": "What was revenue in Q1?"}],
    temperature=0.3,
    max_tokens=2000,
    stream=False,
)
```

## Switching providers

Set **one** of:

| Env var | Values | Effect |
|--------|--------|--------|
| `RAG_LLM_PROVIDER` | `openai` | Use OpenAI only |
| `RAG_LLM_PROVIDER` | `cerebras` | Use Cerebras only |
| `RAG_LLM_PROVIDER` | `auto` (default) | Use Cerebras if `CEREBRAS_API_KEY` is set, else OpenAI |

Optional overrides:

- `RAG_OPENAI_MODEL` – default OpenAI model (e.g. `gpt-4.1-mini-2025-04-14`)
- `RAG_CEREBRAS_MODEL` – default Cerebras model (e.g. `qwen-3-235b-a22b-instruct-2507`)
- `OPENAI_API_KEY`, `CEREBRAS_API_KEY` – API keys (required for the provider you use)

## Using a specific model per call

```python
# Evaluation often uses a different model (from config)
eval_text = llm.complete(
    messages,
    model=config.get("evaluation_model"),  # override default
    temperature=0.05,
    max_tokens=3000,
    stream=False,
)
```

## Streaming

```python
stream = llm.complete(messages, stream=True, max_tokens=8000)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Adding a new provider

1. Implement `LLMClient` in `agent/llm/` (see `openai_client.py`, `cerebras_client.py`).
2. Register it in `factory.py` and add a new `RAG_LLM_PROVIDER` option (and env for API key / model if needed).
