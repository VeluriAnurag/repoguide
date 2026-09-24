#!/usr/bin/env bash
# Starts Ollama in the background, waits until it answers, then runs the app.
set -euo pipefail

ollama serve > /tmp/ollama.log 2>&1 &
until curl -sf http://127.0.0.1:11434/api/version > /dev/null; do sleep 1; done

# Load Llama 3.2 into memory now so the first visitor's answer isn't extra slow.
curl -sf http://127.0.0.1:11434/api/generate -d '{"model": "llama3.2", "prompt": "hi", "stream": false}' > /dev/null &

exec python -m streamlit run app/ui.py --server.port 7860 --server.address 0.0.0.0
