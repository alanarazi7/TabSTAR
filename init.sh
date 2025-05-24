#!/bin/bash

# Ensure uv is installed
if ! command -v uv &>/dev/null; then
    echo "📦 uv not found. Installing uv..."
    pip install uv || echo "⚠️ Failed to install uv; continuing..."
else
    echo "📦 uv is already installed. Proceeding..."
fi

# Create uv venv with Python 3.11 if it doesn't already exist
if [ ! -d ".venv" ]; then
    echo "🌀 Creating uv venv with Python 3.11"
    uv venv -p python3.11 || echo "⚠️ Failed to create uv venv; continuing..."
else
    echo "🌀 uv venv already exists. Skipping creation."
fi

# Activate the environment
if [ -f ".venv/bin/activate" ]; then
    echo "🚀 Activating uv venv"
    # shellcheck disable=SC1091
    source ".venv/bin/activate" || echo "⚠️ Failed to activate venv; continuing..."
else
    echo "⚠️ Activation script not found; did venv creation fail?"
fi

# Install dependencies using uv
if [ -f "requirements.txt" ]; then
    echo "📄 Installing dependencies from requirements.txt using uv"
    uv pip install -r requirements.txt || echo "⚠️ Dependency install failed; continuing..."
else
    echo "⚠️ requirements.txt not found; skipping dependency install."
fi

echo "🎉 Setup completed! To activate in a new shell, run: source .venv/bin/activate"