# Gradient ADK Feedback

**Version:** 0.1.9 | **Date:** December 23, 2025

---

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

