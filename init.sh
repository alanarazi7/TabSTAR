#!/bin/bash

ENV_DIR=".TabSTAR-env"

# Ensure uv is installed
if ! command -v uv &>/dev/null; then
    echo "📦 uv not found. Installing uv..."
    pip install uv || echo "⚠️ Failed to install uv; continuing..."
else
    echo "📦 uv is already installed. Proceeding..."
fi

# Create uv venv with Python 3.11 if it doesn't already exist
if [ ! -d "$ENV_DIR" ]; then
    echo "🌀 Creating uv venv ($ENV_DIR) with Python 3.11"
    uv venv -p python3.11 "$ENV_DIR" || echo "⚠️ Failed to create uv venv; continuing..."
else
    echo "🌀 uv venv ($ENV_DIR) already exists. Skipping creation."
fi

# Activate the environment
if [ -f "$ENV_DIR/bin/activate" ]; then
    echo "🚀 Activating uv venv ($ENV_DIR)"
    # shellcheck disable=SC1091
    source "$ENV_DIR/bin/activate" || echo "⚠️ Failed to activate venv; continuing..."
else
    echo "⚠️ Activation script not found in $ENV_DIR; did venv creation fail?"
fi

# Install dependencies using uv
if [ -f "requirements.txt" ]; then
    echo "📄 Installing dependencies from requirements.txt using uv"
    uv pip install -r requirements.txt || echo "⚠️ Dependency install failed; continuing..."
else
    echo "⚠️ requirements.txt not found; skipping dependency install."
fi

# Install the tabstar package itself in editable mode (finds src/tabstar)
echo "📦 Installing tabstar package (editable)"
uv pip install -e . || echo "⚠️ Failed to install tabstar package; continuing..."

PROJECT_ROOT="$(pwd)"
ACTIVATE="$ENV_DIR/bin/activate"
PY_PATH_LINE="export PYTHONPATH=\"\$PYTHONPATH:$PROJECT_ROOT\""

if ! grep -Fq "$PY_PATH_LINE" "$ACTIVATE"; then
    echo "🛠 Adding repo root ($PROJECT_ROOT) to PYTHONPATH in venv activate script"
    echo "$PY_PATH_LINE" >> "$ACTIVATE"
else
    echo "🛠 Repo root already in PYTHONPATH for venv activate script"
fi

echo "🧪 Verifying imports..."
python -c "import tabular, tabstar_paper; print('✅ Imports work!')" || echo "❌ Import check failed!"

echo "🎉 Setup completed! To activate in a new shell, run: source $ENV_DIR/bin/activate"
