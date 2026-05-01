# Support RAG Agent

Terminal-based multi-domain support agent for the HackerRank Orchestrate challenge. It reads support tickets, retrieves only from the local `data/` corpus, decides whether to reply or escalate, and writes evaluator-compatible output.

## Quickstart

From the repository root:

```bash
sh code/install.sh
sh code/setup_ollama.sh
. .venv/bin/activate
python code/check_ollama.py
python code/main.py --retriever qdrant --rebuild-index
```

The default run reads:

```text
support_tickets/support_tickets.csv
```

and writes:

```text
support_tickets/output.csv
```

## Requirements

- Python 3.9+
- The files in this repository, especially `data/` and `support_tickets/`
- Ollama for the recommended local LLM mode

`code/install.sh` creates `.venv`, installs Python dependencies from `code/requirements.txt`, and creates `.env` from `.env.example` if `.env` does not already exist.

`code/setup_ollama.sh` pulls the default local model:

```text
qwen2.5:1.5b
```

If Ollama is not installed, install it from the Ollama website, then rerun:

```bash
sh code/setup_ollama.sh
```

## Environment

Copy the example file:

```bash
cp .env.example .env
```

Recommended local mode:

```bash
SUPPORT_AGENT_LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:1.5b
SUPPORT_AGENT_ALLOW_HEURISTIC_FALLBACK=true
```

Optional OpenAI mode, if you have credits:

```bash
SUPPORT_AGENT_LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4.1-mini
SUPPORT_AGENT_ALLOW_HEURISTIC_FALLBACK=true
```

Do not commit `.env`. The repository `.gitignore` excludes it.

## Run Modes

Run the full evaluator CSV:

```bash
python code/main.py --retriever qdrant --rebuild-index
```

Run without rebuilding the existing local Qdrant index:

```bash
python code/main.py --retriever qdrant
```

Run a single ticket directly from the terminal:

```bash
python code/main.py \
  --issue "How do I dispute a charge?" \
  --subject "Dispute charge" \
  --company "Visa"
```

Single-ticket mode prints JSON with the same fields as the output CSV:

```text
status, product_area, response, justification, request_type
```

Use the dependency-free retriever fallback if Qdrant is unavailable:

```bash
python code/main.py --retriever bm25
```

## High-Level Flow

1. Normalize each ticket into a stable internal format.
2. Detect obvious risk and request type using deterministic gates.
3. Route the ticket to HackerRank, Claude, or Visa using company hints, corpus retrieval, and routing hints.
4. Retrieve support evidence from the real local corpus with Qdrant local BM25-style sparse search.
5. Use routing hints only for classification and query expansion, never as final answer evidence.
6. Ask the configured LLM judge to refine routing/answerability decisions, with deterministic fallback enabled when configured.
7. Generate a grounded response from retrieved corpus chunks, or escalate risky/unsupported cases.
8. Validate the output schema and write the required CSV columns.

## Output Contract

The output CSV must contain exactly:

```text
status, product_area, response, justification, request_type
```

Allowed values:

```text
status: replied, escalated
request_type: product_issue, feature_request, bug, invalid
```

Product/support claims must come from `data/`. If the docs do not support a safe answer, the agent should escalate or reply as invalid/out of scope.

## Tests

```bash
python -m unittest discover -s code/tests -p 'test_*.py'
```
