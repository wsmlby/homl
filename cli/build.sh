#!/bin/bash
set -e

# This script builds the homl CLI into a single binary using PyInstaller.
# It should be run from the 'cli/' directory.
apt list --installed |grep binutils || {
    echo "binutils is not installed. Installing..."
    apt update && apt install -y binutils
}
# --- Dependencies ---
echo "Installing build dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller


# --- Build ---
echo "Building the binary with PyInstaller..."
# The entry point is the main function in the homl_cli package
ENTRY_POINT="homl_cli/main.py"
# We need to bundle the docker-compose template
DATA_FILE="homl_cli/data/*"
# The binary will be named 'homl'
BINARY_NAME="homl"
if [ -z "$CLI_VERSION" ]; then
    # If CLI_VERSION is not set, default to 'dev'
    CLI_VERSION="dev"
fi
echo "$CLI_VERSION" > homl_cli/__version.txt

pyinstaller \
    --name "$BINARY_NAME" \
    --onefile \
    --console \
    --add-data "homl_cli/__version.txt:." \
    --add-data "$DATA_FILE:data/" \
    "$ENTRY_POINT"

echo "---"
echo "✅ Build complete!"
echo "The binary is located at: dist/$BINARY_NAME"
echo "---"
