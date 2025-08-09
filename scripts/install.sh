#!/bin/bash

# Installation script for HoML.
# This script detects the user's environment, verifies it's a supported
# platform, and downloads the appropriate pre-built binary from GitHub releases.

set -e

# --- Configuration ---
GITHUB_REPO="wsmlby/homl"
INSTALL_DIR="/usr/local/bin"
BINARY_NAME="homl"
SUPPORTED_OS="linux"
SUPPORTED_ARCH="amd64"

# --- Environment Detection ---
# Get the directory of this script to source the detection helper
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source "$SCRIPT_DIR/detect_env.sh" # This sets the OS and ARCH variables

echo "Detected Environment:"
echo "  OS: $OS"
echo "  Architecture: $ARCH"
echo ""

# --- Platform Check ---
if [ "$OS" != "$SUPPORTED_OS" ] || [ "$ARCH" != "$SUPPORTED_ARCH" ]; then
    echo "❌ Your platform ($OS-$ARCH) is not currently supported."
    echo "HoML currently only supports Linux (amd64)."
    echo ""
    echo "We welcome contributions for other platforms!"
    echo "Please open an issue on our GitHub repository to request support for your platform:"
    echo "https://github.com/${GITHUB_REPO}/issues"
    exit 1
fi

# --- Temporary Directory and Cleanup ---
TMP_DIR=$(mktemp -d)
cleanup() {
    if [ -d "$TMP_DIR" ]; then
        rm -rf "$TMP_DIR"
    fi
}
trap cleanup EXIT

# --- Download Logic ---
RELEASE_ASSET_NAME="homl-${OS}-${ARCH}"

# Get the latest release tag from GitHub API
LATEST_RELEASE=$(curl -s "https://api.github.com/repos/${GITHUB_REPO}/releases/latest")
LATEST_TAG=$(echo "$LATEST_RELEASE" | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/')

if [ -z "$LATEST_TAG" ]; then
    echo "Could not determine the latest release tag. Aborting."
    exit 1
fi

echo "Latest release tag is: $LATEST_TAG"

DOWNLOAD_URL="https://github.com/${GITHUB_REPO}/releases/download/${LATEST_TAG}/${RELEASE_ASSET_NAME}"
TMP_BINARY_PATH="$TMP_DIR/$BINARY_NAME"

echo "This script will now attempt to download and install HoML."
echo "  Binary:  $RELEASE_ASSET_NAME"
echo "  From:    $DOWNLOAD_URL"
echo "  To:      $INSTALL_DIR/$BINARY_NAME"
echo ""

# --- Installation ---
echo "1. Downloading the binary..."
if ! curl -L -f -o "$TMP_BINARY_PATH" "$DOWNLOAD_URL"; then
    echo ""
    echo "Error: Download failed."
    echo "Please check the releases page for available binaries: https://github.com/${GITHUB_REPO}/releases"
    exit 1
fi

echo ""
echo "2. Making it executable..."
chmod +x "$TMP_BINARY_PATH"

echo ""
echo "3. Moving it to a directory in your PATH (requires sudo)..."
if [ -w "$INSTALL_DIR" ]; then
    mv "$TMP_BINARY_PATH" "$INSTALL_DIR/$BINARY_NAME"
else
    echo "Sudo privileges are required to move the binary to $INSTALL_DIR."
    sudo mv "$TMP_BINARY_PATH" "$INSTALL_DIR/$BINARY_NAME"
fi

echo "---"
echo "✅ HoML CLI installed successfully!"
echo "You can now use the 'homl' command."
echo "---"

