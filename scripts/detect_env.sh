#!/bin/bash

# A helper script to detect the current environment for HoLM builds.

set -e

detect_os() {
    case "$(uname -s)" in
        Linux*)     OS='linux';;
        Darwin*)    OS='macos';;
        *)          OS='unknown';;
    esac
    export OS
    echo "OS: $OS"
}

detect_arch() {
    case "$(uname -m)" in
        x86_64)     ARCH='x86_64';;
        arm64)      ARCH='arm64';;
        aarch64)    ARCH='arm64';; # Common alias for arm64
        *)          ARCH='unknown';;
    esac
    export ARCH
    echo "Architecture: $ARCH"
}

detect_accelerator() {
    # This is a placeholder. A real implementation would be more robust.
    export CUDA_VERSION='none' # Default to none

    if command -v nvidia-smi &> /dev/null; then
        ACCELERATOR='cuda'
        # Detect CUDA version dynamically
        CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9.]+')
        echo "Accelerator: CUDA"
        echo "Version: $CUDA_VERSION"
    elif command -v rocm-smi &> /dev/null; then
        ACCELERATOR='rocm'
        echo "Accelerator: AMD ROCm"
    elif command -v xpu-smi &> /dev/null; then
        ACCELERATOR='xpu'
        echo "Accelerator: Intel XPU"
    else
        ACCELERATOR='cpu'
        echo "Accelerator: CPU"
    fi
    export ACCELERATOR
}

# Run all detection functions
detect_os
detect_arch
detect_accelerator

echo "Environment detection complete."
