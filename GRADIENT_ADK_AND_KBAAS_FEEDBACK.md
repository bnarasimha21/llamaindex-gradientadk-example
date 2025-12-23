# Gradient ADK and KBAas Feedback

**Version:** 0.1.9 | **Date:** December 23, 2025

---

# Table of Contents

1. [ADK (Agent Development Kit) Feedback](#gradient-adk-feedback)
2. [Knowledge Base as a Service (KBAas) Feedback](#knowledge-base-as-a-service-kbaas-feedback)

---

# Gradient ADK Feedback

## 🐛 Bugs

### 1. Agent Name Validation Not Enforced During `init`

`gradient agent init` accepts invalid names (spaces, apostrophes) but `deploy` rejects them.

```
# init accepts this:
Agent workspace name: Narsi's Agent Workspace
✅ Project created successfully

# deploy rejects it:
❌ Invalid agent workspace name: 'Narsi's Agent Workspace'
```

**Fix:** Validate name during `init`, not `deploy`.

---

### 2. README/Template Input Field Mismatch

README uses `input.get("query")` but the generated template uses `input.get("prompt")`.

| Source | Code |
|--------|------|
| README | `input.get("query")` |
| Template | `input.get("prompt")` |

Copy-pasting from README causes silent failures.

---

### 3. Deploy Doesn't Load `.env` File

`gradient agent run` loads `.env` automatically, but `deploy` does not.

```bash
# This fails even with DIGITALOCEAN_API_TOKEN in .env:
gradient agent deploy
❌ DigitalOcean API token is required

# Workaround:
source .env && export DIGITALOCEAN_API_TOKEN && gradient agent deploy
```

---

## 💡 Feature Requests

### 1. Example Files in `agents/` and `tools/`

The scaffolded directories are empty. Add starter examples:

- `agents/example_agent.py` - Simple agent with state management
- `tools/example_tool.py` - Sample tool with `@trace_tool` decorator

This helps new users understand the expected patterns.

---

### 2. `gradient agent test` Command

Add a local testing command:

```bash
gradient agent test --input '{"prompt": "Hello"}'
```

This would validate the agent works before deployment without needing curl.

---

### 3. Config File Reference

README mentions `config.yaml` but the generated project uses `.gradient/agent.yml`.

Please clarify:
- Which file name is canonical?
- Are both supported?
- What's the migration path if one is deprecated?

---

*Tested on macOS, Python 3.11*

---

# Knowledge Base as a Service (KBAas) Feedback

**SDK:** `gradient` Python package | **LlamaIndex Integration:** `llama-index-llms-digitalocean-gradientai`

---

## 💡 Feature Requests

### 1. Relevance Scores in Retrieval Results

Results don't currently include similarity/relevance scores:

```python
for result in response.results:
    print(result.text_content)
    print(result.score)  # ❌ AttributeError - not available
```

Scores would enable:
- Setting relevance thresholds to filter low-quality results
- Debugging retrieval quality

---

### 2. Batch Retrieval API

For multi-query RAG (e.g., HyDE, query decomposition):

```python
# ❌ Currently requires N separate API calls
queries = ["beaches in Europe", "best time to visit", "budget tips"]

# ✅ Desired: Single batch call
responses = gradient_client.retrieve.batch_documents(
    knowledge_base_id=kb_id,
    queries=queries,
    num_results_per_query=3
)
```

### 3. Source Attribution / Citations

Source URL is available via `metadata.item_name`, but additional attribution would help:

```python
for result in response.results:
    print(result.text_content)
    print(result.metadata.get('item_name'))  # ✅ Source URL available
    print(result.document_name)              # ❌ Not available
    print(result.chunk_index)                # ❌ Not available
```

Having `document_name` and `chunk_index` would enable better citation formatting in RAG responses.

---

## 🚧 Planned Work: LlamaIndex Retriever Integration

Currently, there's no native `GradientRetriever` class for LlamaIndex. Developers must manually call the SDK, extract `text_content` from results, and convert to LlamaIndex format.

**I plan to create a native LlamaIndex retriever package**, similar to how I built [`llama-index-llms-digitalocean-gradientai`](https://pypi.org/project/llama-index-llms-digitalocean-gradientai/) for LLM integration.

**Proposed Package:** `llama-index-retrievers-digitalocean-gradient`

```python
from llama_index.retrievers.digitalocean.gradient import GradientKBRetriever

retriever = GradientKBRetriever(
    knowledge_base_id="kb-uuid",
    api_token="...",
    num_results=5
)

# Works seamlessly with LlamaIndex query engines
query_engine = RetrieverQueryEngine(retriever=retriever)
```

This will provide the same developer experience as the LLM package - making KBAas a first-class citizen in the LlamaIndex ecosystem.

---

*Tested on macOS, Python 3.11, gradient SDK, llama-index-llms-digitalocean-gradientai*

