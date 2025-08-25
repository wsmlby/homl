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
1.  **Build the vLLM CPU base image:** This requires a clone of the vLLM repository. See the comments in `Dockerfile.cpu.base` for detailed instructions on building the `homl/vllm-cpu:latest` base image.
2.  **Build the HoML server image:**
    ```bash
    docker build -f Dockerfile.cpu -t homl/server:latest-cpu .
    ```

### Other platforms
To test this on other platforms

1. find/build the vLLM base image. You can find them [here](https://hub.docker.com/search?q=vllm). For example :

    1. ROCm:   rocm/vllm:latest
    2. Intel GPU: intel/vllm:latest
    3. TPU: vllm/vllm-tpu:nightly
    4. You can also build base images from here https://github.com/vllm-project/vllm/tree/main/docker
    5. Or, of course, create a new image for vLLM on that platform.

2. create a Dockerfile for your platform similar to [this](Dockerfile.cpu)
    1. make sure you use the correct base image from step 1
    2. set the `ENV ACCELERATOR=<your-accelerator-name>`
3. make the proper modification for [some platform specific commands](homl_server/model_manager.py) where ACCELERATOR is used.
4. Build the Docker image using the Dockerfile you created.
5. To use the image, use HOML_DOCKER_IMAGE_OVERRIDE environment variable to specify the image when running the `homl server install` command. For example
```bash
docker build -f Dockerfile.xpu -t homl/server:latest-xpu .
HOML_DOCKER_IMAGE_OVERRIDE=homl/server:latest-xpu homl server install
```
6. use `homl server log` to see logs


