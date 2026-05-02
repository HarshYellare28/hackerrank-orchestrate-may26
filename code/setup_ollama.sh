#!/usr/bin/env sh
set -eu

MODEL="${OLLAMA_MODEL:-qwen2.5:1.5b}"

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama is not installed. Attempting to install Ollama..."
  OS="$(uname -s)"
  if [ "$OS" = "Darwin" ] && command -v brew >/dev/null 2>&1; then
    brew install --cask ollama
  elif [ "$OS" = "Linux" ] && command -v curl >/dev/null 2>&1; then
    curl -fsSL https://ollama.com/install.sh | sh
  else
    echo "Automatic Ollama installation is not available on this machine." >&2
    echo "Install Ollama from https://ollama.com/download, then rerun: sh code/setup_ollama.sh" >&2
    exit 2
  fi
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama still was not found on PATH after installation." >&2
  echo "Open the Ollama app or restart your shell, then rerun: sh code/setup_ollama.sh" >&2
  exit 2
fi

echo "Using Ollama model: $MODEL"
echo "Tip: qwen2.5:1.5b is the lightweight default for local reproducibility."
ollama pull "$MODEL"
echo "Model is available. If the Ollama app/server is not running, start it with: ollama serve"
