
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Dict
from string import Template
import platform as platform_module
import click

from homl_cli import config
from homl_cli.utils import get_resource_path


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


def check_and_install_docker():
    """Checks for Docker and Docker Compose and asks to install if missing."""
    # Check for 'docker' and 'docker compose' (not 'docker-compose')
    docker_exists = shutil.which("docker") is not None
    # Check if 'docker compose' is available
    try:
        result = subprocess.run(
            ["docker", "compose", "version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        compose_exists = result.returncode == 0
    except Exception:
        compose_exists = False
    if docker_exists and compose_exists:
        return True

    # Offer to install via official Docker script
    click.secho("🔥 Docker or Docker Compose not found.", fg="red")
    if click.confirm("May I attempt to install Docker using the official script? (Requires sudo)"):
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
            subprocess.run(["sudo", "usermod", "-aG", "docker",
                           os.getlogin()], check=True)
            click.secho(
                "Please log out and log back in for the group changes to take effect.", fg="yellow")
            # Re-check for docker and compose
            docker_exists = shutil.which("docker") is not None
            try:
                result = subprocess.run(
                    ["docker", "compose", "version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                compose_exists = result.returncode == 0
            except Exception:
                compose_exists = False
            if docker_exists and compose_exists:
                return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            click.secho(f"❌ Docker installation failed: {e}", fg="red")

    click.secho("Please install both Docker and Docker Compose (plugin) manually using your package manager or official instructions, then try again.", fg="yellow")
    click.secho(
        "See: https://docs.docker.com/engine/install/ and https://docs.docker.com/compose/install/", fg="yellow")
    return False


def get_platform_config(accelerator: str, gptoss: bool) -> Dict[str, Any]:
    """Returns the docker image and other config for a given platform."""   
    if accelerator == "cuda":
        cfg = {
            "image": "ghcr.io/wsmlby/homl/server:latest-cuda" if not gptoss else "ghcr.io/wsmlby/homl/server:latest-cuda-gptoss",
            "deploy_resources": """
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
""",
        }
    # TODO: Add support for ROCm and XPU in the future
    else:  # cpu
        cfg = {
            "image": "ghcr.io/wsmlby/homl/server:latest-cpu",
            "deploy_resources": "",
        }
    
    if os.environ.get("HOML_DOCKER_IMAGE_OVERRIDE"):
        cfg["image"] = os.environ["HOML_DOCKER_IMAGE_OVERRIDE"]

    return cfg


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
            subprocess.run(["sudo", "apt-get", "install", "-y",
                           "nvidia-container-toolkit"], check=True)
            subprocess.run(["sudo", "nvidia-ctk", "runtime",
                           "configure", "--runtime=docker"], check=True)
            subprocess.run(
                ["sudo", "systemctl", "restart", "docker"], check=True)
            click.secho(
                "✅ NVIDIA Container Toolkit installed successfully.", fg="green")
            return True
        except Exception as e:
            click.secho(f"❌ NVIDIA runtime installation failed: {e}", fg="red")
            return False
    return False


def install(insecure_socket: bool, upgrade: bool, gptoss: bool, install_webui: bool = False):
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
    platform_config = get_platform_config(accelerator, gptoss)

    # 3. Check for platform-specific dependencies
    if accelerator == "cuda":
        if not check_and_install_nvidia_runtime():
            return
    # add other platform checks here
    else:
        click.secho("No NVIDIA runtime found. Currently only support NVIDIA GPU. Abort.", fg="red")
        return

    # 4. Define paths

    # Read config values
    port = int(config.get_config_value("port", 7456))
    model_home = config.get_config_value(
        "model_home", str(config.CONFIG_DIR / "models"))
    model_load_timeout = int(
        config.get_config_value("model_load_timeout", 180))
    model_unload_idle_time = int(
        config.get_config_value("model_unload_idle_time", 600))

    homl_dir = config.CONFIG_DIR
    socket_dir = homl_dir / "run"
    model_dir = Path(model_home)
    module_info_cache_path = model_dir / "module_info_cache"
    if module_info_cache_path.exists():
        click.echo(
            f"Removing old module info cache at {module_info_cache_path}")
        shutil.rmtree(module_info_cache_path)
    compose_path = homl_dir / "docker-compose.yml"
    template_path = get_resource_path("data/docker-compose.yml.template")
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
        HOML_INSECURE_SOCKET=str(insecure_socket).lower(
        ) if insecure_socket else "false",
        PWD_PATH=str(user_etc_path.resolve()),
        SOCKET_VOLUME_PATH=str(socket_dir.resolve()),
        MODEL_CACHE_PATH=str(model_dir.resolve()),
        HOML_PORT=str(port),
        HOML_MODEL_LOAD_TIMEOUT=str(model_load_timeout),
        HOML_MODEL_UNLOAD_IDLE_TIME=str(model_unload_idle_time)
    )

    if install_webui:
        template_path = get_resource_path("data/openui.template")
        with open(template_path, 'r') as f:
            openui_template = Template(f.read())
        compose_content += openui_template.substitute(
            WEBUI_PORT=str(7457)
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
        click.secho(f"  http://0.0.0.0:{port}", fg="cyan")
        if install_webui:
            click.echo("\nYour Open WebUI is available at:")
            click.secho(f"  http://0.0.0.0:7457", fg="cyan")

    try:
        if upgrade:
            subprocess.run(["docker", "compose", "-f",
                           str(compose_path), "pull"], check=True)
        subprocess.run(["docker", "compose", "-f",
                       str(compose_path), "up", "-d"], check=True)
        print_post_install_message()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        click.echo("Could not run 'docker compose' directly. Trying with 'sudo'.")
        try:
            if upgrade:
                subprocess.run(["sudo", "docker", "compose",
                               "-f", str(compose_path), "pull"], check=True)
            subprocess.run(["sudo", "docker", "compose", "-f",
                           str(compose_path), "up", "-d"], check=True)
            print_post_install_message()
        except Exception as sudo_e:
            click.secho(
                f"❌ Failed to start server with sudo: {sudo_e}", fg="red")
            click.secho(
                "Please ensure Docker and Docker Compose are installed and you have permissions to use them.", fg="red")
