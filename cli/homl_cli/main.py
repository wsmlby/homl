import json
import platform as platform_module
import sys
import click
import grpc

from homl_cli import config
# These modules are generated from daemon.proto
import os
import subprocess
import shutil
from string import Template
from homl_cli import daemon_pb2
from homl_cli import daemon_pb2_grpc
from pathlib import Path
from typing import Dict, Any
import requests

# Define the default socket path within the user's homl config directory
DEFAULT_SOCKET_PATH = config.CONFIG_DIR / "run" / "homl.sock"

def get_socket_path() -> str:
    """Gets the socket path from config, or returns the default."""
    # In the future, a `homl config` command could change this value
    path = config.get_config_value("socket_path", str(DEFAULT_SOCKET_PATH))
    return f"unix://{path}"


def get_client_stub():
    """Creates and returns a gRPC client stub."""
    socket_path = get_socket_path()
    try:
        # Ensure the parent directory for the socket exists for the client
        socket_p = Path(socket_path.replace("unix://", ""))
        socket_p.parent.mkdir(parents=True, exist_ok=True)

        channel = grpc.insecure_channel(socket_path)
        # Check if the server is available
        grpc.channel_ready_future(channel).result(timeout=1)
        return daemon_pb2_grpc.DaemonStub(channel)
    except grpc.FutureTimeoutError:
        click.echo("Error: The HoML daemon is not running.")
        click.echo("Please make sure the server is installed and running, e.g. with 'homl install'.")
        return None

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

def check_and_install_docker():
    """Checks for Docker and Docker Compose and asks to install if missing."""
    # Check for 'docker' and 'docker compose' (not 'docker-compose')
    docker_exists = shutil.which("docker") is not None
    # Check if 'docker compose' is available
    try:
        result = subprocess.run(["docker", "compose", "version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        compose_exists = result.returncode == 0
    except Exception:
        compose_exists = False
    if docker_exists and compose_exists:
        return True

    if docker_exists and not compose_exists:
        click.secho("🔥 Docker Compose (plugin) not found.", fg="yellow")
        # Try to install docker compose via local package manager
        distro = None
        distro_like = None
        try:
            with open("/etc/os-release") as f:
                for line in f:
                    if line.startswith("ID="):
                        distro = line.strip().split("=")[1].strip('"')
                    elif line.startswith("ID_LIKE="):
                        distro_like = line.strip().split("=")[1].strip('"')
        except Exception:
            pass
        installed = False
        if distro in ["ubuntu", "debian"]:
            click.echo("Attempting to install docker-compose-plugin via apt...")
            try:
                subprocess.run(["sudo", "apt-get", "update"], check=True)
                subprocess.run(["sudo", "apt-get", "install", "-y", "docker-compose-plugin"], check=True)
                # Re-check if compose is now available
                result = subprocess.run(["docker", "compose", "version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if result.returncode == 0:
                    click.secho("✅ Docker Compose plugin installed successfully.", fg="green")
                    return True
                else:
                    click.secho("❌ Docker Compose plugin installation failed.", fg="red")
            except Exception as e:
                click.secho(f"❌ Failed to install docker-compose-plugin: {e}", fg="red")
        elif distro in ["fedora", "centos", "rhel", "amzn"] or (distro_like and "fedora" in distro_like):
            click.echo("Attempting to install docker-compose-plugin via dnf...")
            try:
                subprocess.run(["sudo", "dnf", "install", "-y", "docker-compose-plugin"], check=True)
                result = subprocess.run(["docker", "compose", "version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if result.returncode == 0:
                    click.secho("✅ Docker Compose plugin installed successfully.", fg="green")
                    return True
                else:
                    click.secho("❌ Docker Compose plugin installation failed.", fg="red")
            except Exception as e:
                click.secho(f"❌ Failed to install docker-compose-plugin: {e}", fg="red")
        elif distro in ["arch", "manjaro"]:
            click.echo("Attempting to install docker-compose via pacman...")
            try:
                subprocess.run(["sudo", "pacman", "-Sy", "docker-compose"], check=True)
                result = subprocess.run(["docker", "compose", "version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if result.returncode == 0:
                    click.secho("✅ Docker Compose installed successfully.", fg="green")
                    return True
                else:
                    click.secho("❌ Docker Compose installation failed.", fg="red")
            except Exception as e:
                click.secho(f"❌ Failed to install docker-compose: {e}", fg="red")
        # If not installed, fallback to official script
        click.secho("Could not install Docker Compose plugin via package manager.", fg="yellow")

    click.secho("🔥 Docker or Docker Compose not found.", fg="yellow")
    if click.confirm("May I attempt to install them using the official script? (Requires sudo)"):
        try:
            click.echo("Downloading Docker installation script...")
            script_path = "/tmp/get-docker.sh"
            subprocess.run(
                ["curl", "-fsSL", "https://get.docker.com", "-o", script_path],
                check=True
            )
            click.echo("Running installation script with sudo...")
            subprocess.run(["sudo", "sh", script_path], check=True)
            click.secho("✅ Docker installed successfully.", fg="green")
            # Add current user to docker group to avoid sudo for docker commands
            click.echo("Adding current user to the 'docker' group...")
            subprocess.run(["sudo", "usermod", "-aG", "docker", os.getlogin()], check=True)
            click.secho("Please log out and log back in for the group changes to take effect.", fg="yellow")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            click.secho(f"❌ Docker installation failed: {e}", fg="red")
            return False
    else:
        click.echo("Installation cancelled.")
        return False

def get_platform_config(accelerator: str) -> Dict[str, Any]:
    """Returns the docker image and other config for a given platform."""
    # In the future, these images would be hosted on a public registry.
    # For now, they are conceptual names.
    if accelerator == "cuda":
        return {
            "image": "ghcr.io/wsmlby/homl/server:latest-cuda",
            "deploy_resources": """
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
""",
        }
    #TODO: Add support for ROCm and XPU in the future
    else:  # cpu
        return {
            "image": "ghcr.io/wsmlby/homl/server:latest-cpu",
            "deploy_resources": "",
        }

def check_and_install_nvidia_runtime():
    """Checks for nvidia-container-runtime and asks to install if missing."""
    # This is a placeholder for a real check. A real check would be OS-specific.
    # For now, we'll assume it's installed if `nvidia-smi` exists.
    if shutil.which("nvidia-smi"):
        return True

    click.secho("🔥 NVIDIA container runtime prerequisites not met.", fg="yellow")
    if click.confirm("May I attempt to install the NVIDIA container toolkit? (Requires sudo)"):
        try:
            # This is a simplified script. A robust version would handle different distros.
            subprocess.run(["sudo", "apt-get", "update"], check=True)
            subprocess.run(["sudo", "apt-get", "install", "-y", "nvidia-container-toolkit"], check=True)
            subprocess.run(["sudo", "nvidia-ctk", "runtime", "configure", "--runtime=docker"], check=True)
            subprocess.run(["sudo", "systemctl", "restart", "docker"], check=True)
            click.secho("✅ NVIDIA Container Toolkit installed successfully.", fg="green")
            return True
        except Exception as e:
            click.secho(f"❌ NVIDIA runtime installation failed: {e}", fg="red")
            return False
    return False


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

def get_resource_path(relative_path: str) -> Path:
    """
    Get the absolute path to a resource, handling running from source and from
    a PyInstaller bundle.
    """
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # Running in a PyInstaller bundle
        base_path = Path(sys._MEIPASS)
    else:
        # Running in a normal Python environment
        base_path = Path(__file__).parent.absolute()
    return base_path / relative_path

def detect_platform() -> Dict[str, str]:
    """Detects the current platform (OS, architecture, accelerator)."""
    # Detect OS
    os_map = {"linux": "linux", "darwin": "macos", "win32": "windows"}
    os_name = os_map.get(sys.platform, "unknown")

    # Detect architecture
    arch_map = {"x86_64": "amd64", "aarch64": "arm64"}
    arch = arch_map.get(platform_module.machine(), "unknown")

    # Detect accelerator
    accelerator = "cpu"
    if shutil.which("nvidia-smi"):
        accelerator = "cuda"
    # TODO: Add future accelerator detections here (e.g., rocm)

    return {"os": os_name, "arch": arch, "accelerator": accelerator}

@server.command()
@click.option('--insecure-socket', is_flag=True, help="Use a world-writable socket (less secure).")
@click.option('--upgrade', is_flag=True, help="Force reinstallation even if the server is already running.")
def install(insecure_socket: bool, upgrade: bool):
    """Installs and starts the HoML server using Docker Compose."""
    click.echo("🚀 Starting HoML server installation...")

    # 1. Check for Docker
    if not check_and_install_docker():
        return

    # 2. Detect hardware platform
    click.echo("🔬 Detecting hardware platform...")
    platform = detect_platform()
    click.echo(f"Detected Platform: {platform.get('accelerator', 'cpu')}")

    accelerator = platform.get("accelerator", "cpu")
    platform_config = get_platform_config(accelerator)

    # 3. Check for platform-specific dependencies
    if accelerator == "cuda":
        if not check_and_install_nvidia_runtime():
            return
    else:
        click.echo(f"Currently only CUDA is supported for GPU acceleration. Using CPU mode instead.")

    # 4. Define paths
    homl_dir = config.CONFIG_DIR
    socket_dir = homl_dir / "run"
    model_dir = homl_dir / "models"
    compose_path = homl_dir / "docker-compose.yml"
    template_path = get_resource_path("docker-compose.yml.template")
    user_etc_path = homl_dir / "etc_pwd_tmp"

    socket_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # 5. Get UID/GID
    uid = os.getuid()
    gid = os.getgid()

    # 6. Populate and write docker-compose.yml
    with open(template_path, 'r') as f:
        template = Template(f.read())

    # Create user in etc/passwd format
    user_etc_path = homl_dir / "etc_pwd_tmp"
    user_etc_path.write_text(f"myuser:x:{uid}:{gid}::/models/home:/bin/sh")

    compose_content = template.substitute(
        UID=uid,
        GID=gid,
        HOML_DOCKER_IMAGE=platform_config["image"],
        HOML_DEPLOY_RESOURCES=platform_config["deploy_resources"],
        HOML_INSECURE_SOCKET=str(insecure_socket).lower() if insecure_socket else "false",
        PWD_PATH=str(user_etc_path.resolve()),
        SOCKET_VOLUME_PATH=str(socket_dir.resolve()),
        MODEL_CACHE_PATH=str(model_dir.resolve())
    )

    with open(compose_path, 'w') as f:
        f.write(compose_content)

    click.echo(f"📝 Wrote docker-compose configuration to {compose_path}")


    # 7. Run docker compose up
    click.echo("🐳 Starting server with Docker Compose... (This may take a moment)")

    def print_post_install_message():
        click.secho("✅ HoML server started successfully!", fg="green")
        click.echo("\nNext steps:")
        click.echo("  1. Pull a model, e.g., 'homl pull qwen3:0.6b'")
        click.echo("  2. Run the model: 'homl run qwen3:0.6b'")
        click.echo("  3. Chat with it: 'homl chat qwen3:0.6b'")
        click.echo("\nYour OpenAI-compatible API is available at:")
        click.secho("  http://0.0.0.0:7456", fg="cyan")

    try:
        if upgrade:
            subprocess.run(["docker", "compose", "-f", str(compose_path), "pull"], check=True)
        subprocess.run(["docker", "compose", "-f", str(compose_path), "up", "-d"], check=True)
        print_post_install_message()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        click.echo("Could not run 'docker compose' directly. Trying with 'sudo'.")
        try:
            if upgrade:
                subprocess.run(["sudo", "docker", "compose", "-f", str(compose_path), "pull"], check=True)
            subprocess.run(["sudo", "docker", "compose", "-f", str(compose_path), "up", "-d"], check=True)
            print_post_install_message()
        except Exception as sudo_e:
            click.secho(f"❌ Failed to start server with sudo: {sudo_e}", fg="red")
            click.secho("Please ensure Docker and Docker Compose are installed and you have permissions to use them.", fg="red")



@main.command()
@click.argument('model_name')
def run(model_name):
    """Starts a model with the vLLM server."""
    stub = get_client_stub()
    if stub:
        response = stub.StartModel(daemon_pb2.StartModelRequest(model_name=model_name))
        click.echo(response.message)

@main.command()
def ps():
    """Lists running models."""
    stub = get_client_stub()
    if stub:
        response = stub.ListRunningModels(daemon_pb2.ListRunningModelsRequest())
        if len(response.vram_total) > 0:
            for vram in response.vram_total:
                click.echo(f"Device available VRAM {vram.device_id}: {vram.vram_mb} MB")
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
                    vram_info = ", ".join(f"{vram.device_id}: {vram.vram_mb}MB" for vram in model.vram_usage)
            click.echo(f"{model.model_name:<40} {model.pid:<10} {model.status:<10} {model.ram_mb}MB RAM, VRAM: {vram_info}")



@main.command()
@click.argument('model_name')
def chat(model_name):
    """Starts a chat session with a model using the OpenAI-compatible API."""
    api_url = f"http://localhost:7456/v1/chat/completions"
    # api_url = f"http://172.23.0.2:8100/v1/chat/completions"
    click.echo(f"Starting chat with model '{model_name}'. Type 'exit' to quit.")
    history = []
    while True:
        user_input = click.prompt("You")
        if user_input.strip().lower() in ["exit", "quit"]:
            click.echo("Exiting chat.")
            break
        history.append({"role": "user", "content": user_input})
        payload = {
            "model": model_name,
            "messages": history,
            "stream": True
        }
        try:
            with requests.post(api_url, json=payload, stream=True) as resp:
                if resp.status_code == 500:
                    click.secho("Error: The model is not running or the server is not available.", fg="red")
                    click.secho(resp.content.decode(errors="ignore"), fg="red")
                    return
                resp.raise_for_status()
                click.echo("Model:", nl=False)
                response_text = ""
                for chunk in resp.iter_content(chunk_size=None):
                    if chunk:
                        text = chunk.decode(errors="ignore")
                        if text.startswith("data: [DONE]"):
                            break
                        if text.startswith("data: "):
                            text = text[6:]
                        try:
                            json_data = json.loads(text)
                        except json.JSONDecodeError:
                            click.secho(f"Error decoding JSON: {text}", fg="red")
                            continue
                        if "choices" in json_data and len(json_data["choices"]) > 0:
                            text = json_data["choices"][0].get("delta", {}).get("content", "")
                        click.echo(text, nl=False)
                        response_text += text
                click.echo("")
                history.append({"role": "assistant", "content": response_text})
        except Exception as e:
            click.secho(f"Error communicating with model: {e}", fg="red")


@main.command()
@click.argument('model_name')
def stop(model_name):
    """Stops a running model."""
    stub = get_client_stub()
    if stub:
        response = stub.StopModel(daemon_pb2.StopModelRequest(model_name=model_name))
        click.echo(response.message)

@main.command()
@click.argument('model_name')
def pull(model_name):
    """Pulls a model from a registry."""
    stub = get_client_stub()
    # get the Hugging Face token from config
    hf_token = config.get_config_value("hugging_face_token", "")
    if stub:
        click.echo(f"Pulling model '{model_name}'...")
        try:
            responses = stub.PullModel(daemon_pb2.PullModelRequest(model_name=model_name, hf_token=hf_token))
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
                                click.secho("Failed to pull model. Hugging Face token is not set.", fg="red")
                                click.secho("Please set your Hugging Face token with 'homl auth hugging-face <token>'", fg="yellow")
                                return
                            else:
                                click.secho("Hugging Face token is set, but authentication failed. Please check your token.", fg="red")
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
        click.secho("Please provide a Hugging Face token or use --auto to set it automatically from your local ~/.cache/huggingface/token", fg="red")
        return
    if token:
        config.set_config_value("hugging_face_token", token)
        click.echo("Hugging Face token saved successfully.")
        return
    # Attempt to read the token from the default Hugging Face cache location
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if not token_path.exists():
        click.secho("Hugging Face token file not found at ~/.cache/huggingface/token", fg="red")
        return
    with open(token_path, 'r') as f:
        token = f.read().strip()
    if token:
        config.set_config_value("hugging_face_token", token)
        click.echo("Hugging Face token loaded successfully from ~/.cache/huggingface/token.")

    if not token:
        click.secho("Hugging Face token file is empty.", fg="red")

if __name__ == "__main__":
    main()
