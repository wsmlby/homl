from homl_cli.utils import install_utils
from homl_cli.utils.chat import chat_with_model, complete_with_model
from homl_cli.utils import get_resource_path
from homl_cli.utils.model import start_model, get_client_stub

import click
import grpc

from homl_cli import config
import subprocess

from homl_cli import daemon_pb2
from pathlib import Path


@click.group()
def main():
    """
    HoML CLI: A tool to combine the ease of use of Ollama with the speed of vLLM.
    """
    pass


@click.group(help="Manage HoML server.")
def server():
    """Manage server."""
    pass


@click.group(help="Manage authentication.")
def auth():
    """Manage authentication."""
    pass


main.add_command(auth)
main.add_command(server)


# Config CLI group
@click.group(help="Manage HoML configuration. Use 'homl config list' to see all available keys.")
def config_cli():
    pass


@config_cli.command("get")
@click.argument("key", required=True)
def config_get(key):
    """Get a config value."""
    value = config.get_config_value(key)
    if value is None:
        click.echo(f"Key '{key}' not set.")
    else:
        click.echo(f"{key}: {value}")


@config_cli.command("set")
@click.argument("key", required=True)
@click.argument("value", required=True)
def config_set(key, value):
    """Set a config value."""
    if key not in config.CONFIG_KEYS:
        click.secho(
            f"Unknown key '{key}'. Use 'homl config list' to see available keys.", fg="red")
        return
    config.set_config_value(key, value)
    click.echo(f"Set '{key}' to '{value}'")
    click.secho(
        "Note: Changes to config will only take effect after you restart the HoML server.", fg="yellow")


@config_cli.command("list")
def config_list():
    """List all available config keys and their descriptions."""
    click.echo("Available config keys:")
    for key, desc in config.CONFIG_KEYS.items():
        value = config.get_config_value(key)
        click.echo(f"- {key}: {desc}")
        if value is not None:
            # Do not print credentials or sensitive values
            click.echo(f"    Current value: {value}")


main.add_command(config_cli, name="config")


@main.command()
def version():
    """Show CLI and server version."""
    version_file = get_resource_path("__version.txt")
    cli_version = "dev"
    if version_file.exists():
        with open(version_file, 'r') as f:
            cli_version = f.read().strip()
    click.echo(f"HoML CLI version: {cli_version}")
    # Try to get server version via gRPC
    stub = get_client_stub()
    if stub:
        try:
            resp = stub.Version(daemon_pb2.VersionRequest())
            click.echo(f"HoML Server version: {resp.version}")
        except Exception:
            click.echo("HoML Server version: unavailable (gRPC error)")
    else:
        click.echo("HoML Server version: unavailable (daemon not running)")


@server.command()
def log():
    # run docker logs
    """Displays the logs of the HoML server."""
    subprocess.run(["docker", "logs", "-f", "homl-homl-server-1"], check=True)


@server.command()
def stop():
    """Stops the HoML server."""
    subprocess.run(["docker", "rm", "-f", "homl-homl-server-1"], check=True)


@server.command()
def restart():
    """Restarts the HoML server."""
    subprocess.run(["docker", "restart", "homl-homl-server-1"], check=True)


@server.command()
@click.option('--insecure-socket', is_flag=True, help="Use a world-writable socket (less secure).")
@click.option('--upgrade', is_flag=True, help="Force reinstallation even if the server is already running.")
@click.option('--gptoss', is_flag=True, help="Use the GPTOSS image instead of the default.")
@click.option('--webui', is_flag=True, help="Install the Open WebUI alongside the server.")
def install(insecure_socket, upgrade, gptoss, webui):
    install_utils.install(insecure_socket=insecure_socket, upgrade=upgrade, gptoss=gptoss, install_webui=webui)

@main.command()
@click.argument('model_name')
@click.option('--eager', is_flag=True, help="Start the model in eager mode, faster startup but slower latency, similar throughput.")
def run(model_name, eager):
    """Starts a model with the vLLM server."""
    start_model(model_name, eager=eager)


@main.command()
def ps():
    """Lists running models."""
    stub = get_client_stub()
    if stub:
        response = stub.ListRunningModels(
            daemon_pb2.ListRunningModelsRequest())
        if len(response.vram_total) > 0:
            for vram in response.vram_total:
                click.echo(
                    f"Device available VRAM {vram.device_id}: {vram.vram_mb} MB")
        if not response.models:
            click.echo("No models are currently running.")
            return

        # Basic table output
        click.echo(f"{'MODEL':<40} {'PID':<10} {'STATUS':<10}")
        for model in response.models:
            vram_info = ""
            if len(model.vram_usage) > 0:
                if len(model.vram_usage) == 1:
                    vram_info = f"{model.vram_usage[0].vram_mb}MB"
                else:
                    # If there are multiple VRAM usages, format them accordingly
                    vram_info = ", ".join(
                        f"{vram.device_id}: {vram.vram_mb}MB" for vram in model.vram_usage)
            click.echo(
                f"{model.model_name:<40} {model.pid:<10} {model.status:<10} {model.ram_mb}MB RAM, VRAM: {vram_info}")


@main.command()
@click.argument('model_name')
def chat(model_name):
    """Starts a chat session with a model using the OpenAI-compatible API."""
    if start_model(model_name, True) == 0:
        return
    port = config.get_config_value("port", 7456)
    api_url = f"http://localhost:{port}/v1/chat/completions"
    click.echo(
        f"Starting chat with model '{model_name}'. Type 'exit' to quit.")
    chat_with_model(model_name, api_url)


@main.command()
@click.argument('model_name')
@click.argument('prompt')
@click.option('--limit')
def complete(model_name, prompt, limit):
    """Starts a chat session with a model using the OpenAI-compatible API."""
    if start_model(model_name, True) == 0:
        return
    port = config.get_config_value("port", 7456)
    api_url = f"http://localhost:{port}/v1/completions"
    click.echo(prompt)
    complete_with_model(model_name, api_url, prompt, limit=int(limit) if limit else 1024)



@main.command()
@click.argument('model_name')
def stop(model_name):
    """Stops a running model."""
    stub = get_client_stub()
    if stub:
        response = stub.StopModel(
            daemon_pb2.StopModelRequest(model_name=model_name))
        click.echo(response.message)


@main.command()
@click.argument('model_name')
def pull(model_name):
    """Pulls a model from a registry."""
    stub = get_client_stub()
    if "gptoss" in model_name.lower():
        # check if running container homl-homl-server-1 is the GPTOSS image, if not, ask the user to use the gptoss flag and reinstall
        subprocess.run(["docker", "ps", "-f", "name=homl-homl-server-1",
                       "--format", "{{.Image}}"], check=True, capture_output=True)
        image_name = subprocess.run(["docker", "ps", "-f", "name=homl-homl-server-1", "--format",
                                    "{{.Image}}"], check=True, capture_output=True, text=True).stdout.strip()
        if "gptoss" not in image_name.lower():
            click.secho(
                "You are trying to pull a GPTOSS model, but the server is not running with the GPTOSS image.", fg="red")
            click.secho(
                "Please use the --gptoss flag when to reinstall the server with 'homl server install --gptoss'.", fg="yellow")
            click.secho(
                "GPTOSS support is experimental, if you encounter issues, you can revert by reinstall using 'homl server install'.", fg="yellow")
            return

    # get the Hugging Face token from config
    hf_token = config.get_config_value("hugging_face_token", "")
    if stub:
        click.echo(f"Pulling model '{model_name}'...")
        try:
            responses = stub.PullModel(daemon_pb2.PullModelRequest(
                model_name=model_name, hf_token=hf_token))
            for resp in responses:
                msg = resp.message
                percent = resp.percent
                done = resp.done
                success = resp.success
                if percent >= 0 and percent < 100:
                    click.echo(f"[{percent}%] {msg}")
                elif percent == -1:
                    click.echo(msg)
                else:
                    if success:
                        click.secho(msg, fg="green")
                    else:
                        if "authenticated" in msg.lower():
                            if hf_token == "":
                                click.secho(
                                    "Failed to pull model. Hugging Face token is not set.", fg="red")
                                click.secho(
                                    "Please set your Hugging Face token with 'homl auth hugging-face <token>'", fg="yellow")
                                return
                            else:
                                click.secho(
                                    "Hugging Face token is set, but authentication failed. Please check your token.", fg="red")
                        click.secho(f"Error: {msg}", fg="red")
                if done:
                    break
        except grpc.RpcError as e:
            click.secho(f"An RPC error occurred: {e.details()}", fg="red")


@main.command()
@click.option('--with-size', is_flag=True, help="Include model sizes in the output.")
def list(with_size):
    """Lists all locally available models."""
    stub = get_client_stub()
    if stub:
        response = stub.ListLocalModels(daemon_pb2.ListLocalModelsRequest(
            with_size=with_size
        ))
        if not response.models:
            click.echo("No models are available locally.")
            return
        click.echo("Locally available models:")
        for model in response.models:
            if with_size:
                click.echo(f"- {model.model} ({model.size_mb} MB)")
            else:
                click.echo(f"- {model.model}")


@main.command()
def info():
    """(Placeholder) Prints version and other debug information."""
    click.echo("Placeholder for info command.")


@auth.command(name="hugging-face")
@click.argument("token", required=False)
@click.option("--auto", is_flag=True, help="Load token from ~/.cache/huggingface/token")
def hugging_face(token: str = None, auto: bool = False):
    """Saves the Hugging Face token."""
    if not token and not auto:
        click.secho(
            "Please provide a Hugging Face token or use --auto to set it automatically from your local ~/.cache/huggingface/token", fg="red")
        return
    if token:
        config.set_config_value("hugging_face_token", token)
        click.echo("Hugging Face token saved successfully.")
        return
    # Attempt to read the token from the default Hugging Face cache location
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if not token_path.exists():
        click.secho(
            "Hugging Face token file not found at ~/.cache/huggingface/token", fg="red")
        return
    with open(token_path, 'r') as f:
        token = f.read().strip()
    if token:
        config.set_config_value("hugging_face_token", token)
        click.echo(
            "Hugging Face token loaded successfully from ~/.cache/huggingface/token.")

    if not token:
        click.secho("Hugging Face token file is empty.", fg="red")


if __name__ == "__main__":
    main()
