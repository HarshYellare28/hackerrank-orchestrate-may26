#!/usr/bin/env sh
set -eu

MODEL="${OLLAMA_MODEL:-qwen2.5:1.5b}"

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama is not installed. Install it from https://ollama.com/download, then rerun this script." >&2
  exit 2
fi

echo "Using Ollama model: $MODEL"
echo "Tip: qwen2.5:1.5b is the lightweight default for local reproducibility."
ollama pull "$MODEL"
echo "Model is available. If the Ollama app/server is not running, start it with: ollama serve"
