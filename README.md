# [HoML](https://homl.dev): The best of Ollama and vLLM

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
*   **Automatic GPU Memory Management:** HoML intelligently manages your GPU memory. Models are automatically loaded when requested via the OpenAI-compatible API and unloaded when another model is requested. To free up resources for other applications, models are also automatically unloaded after a configurable idle period (defaulting to 10 minutes).
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

## Documentation

For detailed information on how to use the HoML CLI, please refer to our official documentation:

[**HoML Documentation**](https://homl.dev/docs/cli.html)

## TODO / Roadmap
*   Improve vLLM startup time to support faster switching between models.
*   MultiGPU support: Enable multiple models running at the same time on different GPUs.
*   Enable multiple models running at the same time on the same GPU, this means we need to be able to estimate the vRAM usage of each model and manage the memory accordingly.
*   Add support for ROCm, Apple Silicon, and other architectures.
*   Add support for loading adapter layers.
*   Add support for endpoints other than chatcompletion, such as embeddings and text generation.

## Contributing

We welcome contributions! If you're interested in helping out, please check out the issues section and feel free to open a pull request.

We are particularly looking for help with:
*   Help to host CI for ROCm and Apple Silicon.
*   Testing and verifying models for the curated list.
*   Improving the CLI experience.

## Credits

HoML stands on the shoulders of giants. We would like to extend our heartfelt gratitude to the developers and communities behind these incredible open-source projects:

*   **[vLLM](https://github.com/vllm-project/vllm):** For the high-performance inference engine that powers HoML.
*   **[Open WebUI](https://github.com/open-webui/open-webui):** For the user-friendly web interface that enhances the HoML experience.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.