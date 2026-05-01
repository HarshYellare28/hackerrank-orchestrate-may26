#!/usr/bin/env sh
set -eu

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r code/requirements.txt

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example."
else
  echo ".env already exists; leaving it unchanged."
fi

echo "Installed dependencies."
echo "Recommended local model setup: sh code/setup_ollama.sh"
echo "Then run: . .venv/bin/activate && python code/main.py --retriever qdrant --rebuild-index"
