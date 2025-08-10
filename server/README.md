# HoML Server

The HoML server is designed to be run as a Docker container. It provides an OpenAI-compatible API for running inference on various models and a gRPC server for control by the `homl` CLI.

## Installation and Management

The server container is intended to be managed directly by the `homl` command-line tool. To install and run the server, please use the following command:

```bash
homl server install
```

This command will handle the creation of a `docker-compose.yml` file, configure the necessary volumes for the model cache and gRPC socket, and start the server.

### Advanced Configuration

For advanced users who wish to manage the server manually, the `homl server install` command generates a `docker-compose.yml` file in `~/.homl/`. This file can be inspected and modified to suit your needs.

Key configuration points managed by the installer include:
-   **User Permissions:** The container is run with the host user's UID/GID to ensure correct ownership of the socket and cache files.
-   **Volume Mounts:** The socket is shared at `~/.homl/run` and the model cache is persisted at `~/.homl/models`.
-   **Insecure Socket:** An `--insecure-socket` flag on the `install` command allows for a world-writable socket as a fallback, which is controlled by the `HOML_INSECURE_SOCKET` environment variable passed to the container.

## Building from Source

For developers, the platform-specific HoML server images can be built from source.

### CUDA
```bash
docker build -f Dockerfile.cuda -t homl/server:latest-cuda .
```

### CPU
Building the CPU image is a two-step process:
1.  **Build the vLLM CPU base image:** This requires a clone of the vLLM repository. See the comments in `Dockerfile.cpu` for detailed instructions on building the `homl/vllm-cpu:latest` base image.
2.  **Build the HoML server image:**
    ```bash
    docker build -f Dockerfile.cpu.app -t homl/server:latest-cpu .
    ```
