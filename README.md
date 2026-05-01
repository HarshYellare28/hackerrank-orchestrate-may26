# HackerRank Orchestrate Support Agent

This repository contains a terminal-based multi-domain support agent for the HackerRank Orchestrate challenge. It handles tickets for HackerRank, Claude, and Visa using only the local support corpus in `data/`.

The main implementation is in `code/`.

## Quickstart

From the repository root:

```bash
sh code/install.sh
sh code/setup_ollama.sh
. .venv/bin/activate
python code/check_ollama.py
python code/main.py --retriever qdrant --rebuild-index
```

This reads:

```text
support_tickets/support_tickets.csv
```

and writes:

```text
support_tickets/output.csv
```

## Environment

`code/install.sh` creates `.venv`, installs Python dependencies, and creates `.env` from `.env.example` if one does not already exist.

Recommended local mode:

```bash
SUPPORT_AGENT_LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:1.5b
SUPPORT_AGENT_ALLOW_HEURISTIC_FALLBACK=true
```

Optional OpenAI mode:

```bash
SUPPORT_AGENT_LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4.1-mini
SUPPORT_AGENT_ALLOW_HEURISTIC_FALLBACK=true
```

Do not commit `.env`.

## Running

Full CSV run:

```bash
python code/main.py --retriever qdrant --rebuild-index
```

Single-ticket terminal mode:

```bash
python code/main.py \
  --issue "How do I dispute a charge?" \
  --subject "Dispute charge" \
  --company "Visa"
```

BM25 fallback:

```bash
python code/main.py --retriever bm25
```

## High-Level Flow

1. Normalize CSV or terminal ticket input.
2. Detect obvious risk, invalid requests, and request type.
3. Route the ticket to the best domain using company hints, corpus retrieval, and routing hints.
4. Retrieve evidence from the real local corpus with Qdrant local search.
5. Keep generated routing hints separate from answer evidence.
6. Use the configured LLM judge for final routing/refinement, with deterministic fallback when enabled.
7. Generate a grounded answer from retrieved support docs, or escalate high-risk/unsupported cases.
8. Validate the required schema and write output CSV rows.

## Output Contract

The output CSV contains exactly:

```text
status, product_area, response, justification, request_type
```

Allowed values:

```text
status: replied, escalated
request_type: product_issue, feature_request, bug, invalid
```

## Tests

```bash
python -m unittest discover -s code/tests -p 'test_*.py'
```

## Submission Files

Submit:

1. The `code/` directory.
2. `support_tickets/output.csv`.
3. The shared chat transcript log required by `AGENTS.md`.

See `problem_statement.md` and `evalutation_criteria.md` for the challenge contract and scoring criteria.
