# HoML: The best of Ollama and vLLM

HoML aims to combine the ease of use of Ollama with the high-performance inference capabilities of vLLM. This project provides a simple and intuitive command-line interface (CLI) to run and manage large language models, powered by the speed and compatibility of vLLM.

## Project Status

**Beta:** This project is in beta. All planned features are implemented, but there may still be some bugs. We welcome feedback and contributions!

## Why HoML?

[Ollama](https://ollama.ai/) has set a new standard for ease of use in running local LLMs. Its simple CLI and user-friendly approach have made it incredibly popular. On the other hand, [vLLM](https://github.com/vllm-project/vllm) is a state-of-the-art inference engine known for its high throughput and broad compatibility with models from the Hugging Face Hub.

HoML brings the best of both worlds together, offering:

*   **An Ollama-like experience:** A simple, intuitive CLI that just works.
*   **High-performance inference:** Powered by vLLM for maximum speed.
*   **Broad model compatibility:** Access to a vast range of models from the Hugging Face Hub.

## Features

*   **One-Line Installation:** A simple, one-line script for easy installation and upgrades across a wide range of machines.
*   **Simple CLI:** An intuitive command-line interface for managing and running models.
*   **Easy Model Management:** A `pull` command to download models from the Hugging Face Hub.
*   **Automatic Model Loading/Unloading:** Models are loaded and unloaded from memory as needed, with an option to disable this feature.
*   **Interactive Chat:** A `run` command to start an interactive chat session with a model.
*   **OpenAI-Compatible API:** A built-in server that exposes an OpenAI-compatible API for seamless integration with existing tools.
*   **Curated Model List:** A website and a curated list of tested and verified models, with clear version compatibility.

## Components

### HoML Server
HoML Server manage all states and serve OpenAI-Compatible API. It runs in a container environment like Docker. An unix socket based gRPC server also runs and expose the control interface to the cli.

HoML Server can be installed with the cli but can also be installed manually if people are comfortable with docker etc.

This lives under server/

This will be a python project that build into a few supported Dockerfile based on corresponding vLLM dockers, add on ourcode to manage models lifecycles and the gRPC server.

### HoML CLI
HoML CLI is the interface to communicate with the HoML server.

It is a python package that can be installed with the one line installcation script.

`curl -sSL https://homl.dev/install.sh | sh`

This lives under cli/

## CLI Usage

### Managing the Server

**Install and start the HoML server:**
```bash
homl server install
```
This command automates the process of setting up and running the HoML server using Docker Compose.

**Stop the HoML server:**
```bash
homl server stop
```

**Restart the HoML server:**
```bash
homl server restart
```

**View server logs:**
```bash
homl server log
```

### Managing Models

**Pull a model from the Hugging Face Hub:**
```bash
homl pull google/gemma-3-4b-it
```

**Run a model:**
```bash
homl run gemma-3-4b-it
```

**Chat with a model:**
```bash
homl chat gemma-3-4b-it
```

**List running models:**
```bash
homl ps
```

**List locally available models:**
```bash
homl list
```

**Stop a running model:**
```bash
homl stop gemma-3-4b-it
```

### Authentication

**Authenticate with Hugging Face:**
```bash
homl auth hugging-face <your-hf-token>
```
You can also use `homl auth hugging-face --auto` to automatically load the token from `~/.cache/huggingface/token`.

## TODO

*   Enable multiple models running at the same time (`server/homl_server/main.py`)
*   Add support for ROCm and XPU (`cli/homl_cli/main.py`)

## Contributing

We welcome contributions! If you're interested in helping out, please check out the issues section and feel free to open a pull request.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.