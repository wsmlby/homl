#!/bin/sh

# Installation script for HoML.
# This script detects the user's environment, verifies it's a supported
# platform, and downloads the appropriate pre-built binary from GitHub releases.
# It finds a writable directory in the user's PATH or uses ~/.local/bin,
# ensuring the installation does not require root privileges.

set -e

# --- Configuration ---
GITHUB_REPO="wsmlby/homl"
BINARY_NAME="homl"
SUPPORTED_OS="linux"
SUPPORTED_ARCH="amd64"

# --- Environment Detection ---
OS=""
ARCH=""

echo "Detecting your OS and architecture..."

# Detect OS using uname
case "$(uname -s)" in
    Linux*) 
        OS='linux'
        ;;
    *)
        OS='unsupported'
        ;;
esac

# Detect architecture using uname
case "$(uname -m)" in
    x86_64)
        ARCH='amd64'
        ;;
    aarch64 | arm64)
        ARCH='arm64'
        ;;
    *)
        ARCH='unsupported'
        ;;
esac

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

# --- Find Installation Directory ---
INSTALL_DIR=""
# Find a writable, user-owned directory in PATH
echo "Searching for a writable directory in your PATH..."
for dir in $(echo "$PATH" | tr ":" "\n"); do
    if [ -d "$dir" ] && [ -w "$dir" ] && [ -O "$dir" ]; then
        INSTALL_DIR="$dir"
        echo "Found writable directory: $INSTALL_DIR"
        break
    fi
done

# If no suitable directory is found in PATH, default to ~/.local/bin
if [ -z "$INSTALL_DIR" ]; then
    echo "No user-writable directory found in PATH. Defaulting to ~/.local/bin."
    INSTALL_DIR="$HOME/.local/bin"
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
echo "Fetching latest release information from GitHub..."
LATEST_RELEASE_URL="https://api.github.com/repos/${GITHUB_REPO}/releases/latest"
LATEST_RELEASE=$(curl -sL "$LATEST_RELEASE_URL")
LATEST_TAG=$(echo "$LATEST_RELEASE" | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/')

if [ -z "$LATEST_TAG" ]; then
    echo "Could not determine the latest release tag from $LATEST_RELEASE_URL. Aborting."
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
echo "1. Creating installation directory (if it doesn't exist)..."
mkdir -p "$INSTALL_DIR"

echo ""
echo "2. Downloading the binary..."
if ! curl -L -f -o "$TMP_BINARY_PATH" "$DOWNLOAD_URL"; then
    echo ""
    echo "Error: Download failed."
    echo "Please check the releases page for available binaries: https://github.com/${GITHUB_REPO}/releases"
    exit 1
fi

echo ""
echo "3. Making it executable..."
chmod +x "$TMP_BINARY_PATH"

echo ""
echo "4. Moving it to the installation directory..."
mv "$TMP_BINARY_PATH" "$INSTALL_DIR/$BINARY_NAME"

echo "---"
echo "✅ HoML CLI installed successfully to $INSTALL_DIR/$BINARY_NAME"
echo ""

# --- PATH Check ---
# If we installed to ~/.local/bin, check if it's in the PATH
if [ "$INSTALL_DIR" = "$HOME/.local/bin" ]; then
    case ":$PATH:" in
        *":$INSTALL_DIR:"*) 
            echo "The 'homl' command is now available in your shell."
            ;;
        *)
            echo "⚠️  IMPORTANT:"
            echo "The directory $INSTALL_DIR is not in your PATH."
            echo "To use the 'homl' command directly, you need to add it."
            echo ""
            echo "Please add the following line to your shell's startup file (e.g., ~/.bashrc, ~/.zshrc, or ~/.profile):"
            echo "export PATH=\"$INSTALL_DIR:$PATH\""
            echo ""
            echo "After adding it, restart your shell or run 'source <your_shell_file>' to apply the changes."
            ;;
    esac
else
    echo "The 'homl' command should be available in your shell, as it was installed in a directory in your PATH."
fi


echo "---"
