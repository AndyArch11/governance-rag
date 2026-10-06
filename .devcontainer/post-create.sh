#!/usr/bin/env bash
# Provisions the dev container: system libraries, project venv, dependencies and directories.
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${WORKSPACE_DIR}"

# Named volumes are created root-owned
sudo chown -R "$(id -u):$(id -g)" .venv "${HOME}/.cache"

# OpenCV/EasyOCR (Docling OCR) runtime libraries
sudo apt-get update
sudo apt-get install -y --no-install-recommends libgl1 libglib2.0-0
sudo rm -rf /var/lib/apt/lists/*

if ! .venv/bin/python3 --version > /dev/null 2>&1; then
    python3 -m venv .venv
fi

.venv/bin/python3 -m pip install --upgrade pip
.venv/bin/python3 -m pip install -r requirements-dev.txt -r requirements-academic.txt
.venv/bin/python3 -m pip install -e .

make setup-dirs

if [[ ! -f .env ]]; then
    cp .env.example .env
    echo "Created .env from .env.example"
fi

echo "Dev container ready: $(.venv/bin/python3 --version), OLLAMA_HOST=${OLLAMA_HOST:-unset}"
