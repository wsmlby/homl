
import hashlib
import multiprocessing
import os
from pathlib import Path
import json
import threading
import time
import uvloop
import daemon_pb2
from typing import List
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.entrypoints.openai.tool_parsers import ToolParserManager
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.cli_args import FrontendArgs
from vllm.entrypoints.openai.api_server import build_app, init_app_state, build_async_engine_client, run_server, make_arg_parser, validate_parsed_serve_args
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_HOME = os.environ.get("HOML_MODEL_HOME", "/models")
MODEL_LIB = os.path.join(MODEL_HOME, "lib")
TORCH_CACHE = os.path.join(MODEL_LIB, "torch_cache")
MODEL_CONFIG_HOME = os.path.join(MODEL_HOME, "config")
MODEL_LOAD_TIMEOUT = int(os.environ.get(
    "HOML_MODEL_LOAD_TIMEOUT", 180))  # seconds
# # This is the time after which a model will be unloaded if it is idle
MODEL_UNLOAD_IDLE_TIME = int(os.environ.get(
    "HOML_MODEL_UNLOAD_IDLE_TIME", 600))  # 10 minutes default
module_info_cache = os.path.join(MODEL_HOME, "module_info_cache")

os.makedirs(os.path.join(MODEL_HOME, "home"), exist_ok=True)
os.makedirs(MODEL_LIB, exist_ok=True)
os.makedirs(TORCH_CACHE, exist_ok=True)
os.makedirs(MODEL_CONFIG_HOME, exist_ok=True)
os.makedirs(module_info_cache, exist_ok=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = TORCH_CACHE
os.environ["VLLM_LAZY_LOAD_MODULE_INFO_CACHE"] = module_info_cache
# Ensure cache and lib directories exist


def start_server(port: int, model_name: str, model_path: str, eager, params: List[str] = []):
    """
    Starts the vLLM server in a separate thread.
    This is a workaround to avoid starting a new process for each request.
    """
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args(
        [
            "--model", model_path, "--port", str(port)
        ] + (["--enforce-eager"] if eager else []) + params)
    logger.info(f"Starting vLLM server with args: {args}")
    validate_parsed_serve_args(args)

    uvloop.run(run_server(args))


class ModelManager:
    def load_alias(self):
        # Load alias.json if exists
        alias_path = Path(MODEL_HOME) / "alias.json"
        if alias_path.exists():
            try:
                import json
                with open(alias_path) as f:
                    # Assuming alias.json is a simple key-value mapping
                    alias = json.load(f)
                    # Reverse mapping
                    return alias, {v: k for k, v in alias.items()}
            except Exception as e:
                logger.error(f"Error loading alias: {e}")
        return {}, {}
    def get_config(self, model_path):
        config_path = Path(MODEL_CONFIG_HOME) / (model_path+".json")
        settings = {}
        if config_path.exists():
            config_json = json.load(open(config_path))
            settings = config_json.get("settings", {})
            if not settings:
                settings = {}
        return settings
    def save_config(self, model_path, settings):
        config_path = Path(MODEL_CONFIG_HOME) / (model_path+".json")
        config_json = {"settings": settings}
        with open(config_path, "w") as f:
            json.dump(config_json, f)
    

    def load_manifest(self):
        # Load manifest.json if exists
        manifest_path = Path(MODEL_HOME) / "manifest.json"
        if manifest_path.exists():
            try:
                import json
                with open(manifest_path) as f:
                    mapping = json.load(f)
                    # Reverse mapping
                    return mapping, {v: k for k, v in mapping.items()}
            except Exception as e:
                logger.error(f"Error loading manifest: {e}")
        return {}, {}

    def add_manifest_entry(self, model_id, model_path):
        manifest, reverse_manifest = self.load_manifest()
        manifest[model_id] = model_path
        # Write back to manifest.json
        manifest_path = Path(MODEL_HOME) / "manifest.json"
        try:
            with open(manifest_path, "w") as f:
                import json
                json.dump(manifest, f)
        except Exception as e:
            logger.error(f"Error writing manifest: {e}")

    def add_alias_entry(self, alias_name, model_id):
        alias, reverse_alias = self.load_alias()
        alias[alias_name] = model_id
        # Write back to alias.json
        alias_path = Path(MODEL_HOME) / "alias.json"
        try:
            with open(alias_path, "w") as f:
                import json
                json.dump(alias, f)
        except Exception as e:
            logger.error(f"Error writing alias: {e}")

    def list_model_paths(self):
        # List model directories in the cache
        if not os.path.exists(MODEL_LIB):
            return []
        return [model_dir.name for model_dir in Path(MODEL_LIB).iterdir() if model_dir.is_dir()]

    def list_local(self, with_size: bool = False):
        """Lists locally available models, showing alias if available (from manifest)."""
        _, manifest = self.load_manifest()
        _, reverse_alias = self.load_alias()
        model_paths = self.list_model_paths()
        logger.info(
            f"Local model paths: {model_paths}, manifest: {manifest}, reverse_alias: {reverse_alias}")
        model_ids = [(manifest.get(model), model)
                     for model in model_paths if model in manifest]
        models = [(reverse_alias.get(model, model), self.get_model_size(
            model_path) if with_size else -1, model) for model, model_path in model_ids]
        return [daemon_pb2.LocalModelInfo(model=model, size_mb=size, model_id=model_id) for model, size, model_id in models]
    """
    Manages model lifecycle: running, loading, unloading, etc.
    For each model, starts a vLLM server on a unique port.
    """
    BASE_PORT = 8100
    MAX_PORT = 9000

    def __init__(self):
        self.running_models = {}  # model_name: {pid, port, process, status}
        self.used_ports = set()
        self.last_access = {}  # model_name: last access timestamp
        self.lock = threading.Lock()
        self._start_idle_unload_thread()
        self.mp_context = multiprocessing.get_context("fork")

    def _start_idle_unload_thread(self):
        def idle_unload_loop():
            while True:
                now = time.time()
                models_to_unload = []
                with self.lock:
                    for model_id in list(self.running_models.keys()):
                        if self.running_models[model_id]["status"] == "starting":
                            continue
                        last = self.last_access.get(model_id, now)
                        if now - last > MODEL_UNLOAD_IDLE_TIME:
                            logger.info(
                                f"Unloading idle model {model_id} after {now - last:.1f}s of inactivity")
                            models_to_unload.append(model_id)
                # Do it outside the lock to avoid deadlocks
                for model_id in models_to_unload:
                    self.stop_model(model_id)
                    self.last_access.pop(model_id, None)
                time.sleep(10)
        t = threading.Thread(target=idle_unload_loop, daemon=True)
        t.start()

    # def _scan_local_models(self):
    #     # Scan cache dir for models
    #     cache_dir = os.environ.get("HOML_MODEL_CACHE", "/models")
    #     self.local_models = set()
    #     if not os.path.exists(cache_dir):
    #         return
    #     for model_dir in Path(cache_dir).iterdir():
    #         if model_dir.is_dir():
    #             self.local_models.add(model_dir.name)

    def _find_free_port(self):
        for port in range(self.BASE_PORT, self.MAX_PORT):
            if port not in self.used_ports:
                return port
        raise RuntimeError("No free ports available for vLLM models")

    def is_running(self, model_id):
        return model_id in self.running_models and self.running_models[model_id]['status'] == "running" and self.running_models[model_id]["process"].is_alive()

    def is_local(self, model_name):
        alias, _ = self.load_alias()
        model_paths, _ = self.load_manifest()
        if model_name in alias:
            model_id = alias[model_name]
        else:
            model_id = model_name
        model_path = model_paths.get(model_id)
        if not model_path:
            return False, model_id, None
        return model_path in self.list_model_paths(), model_id, model_path

    def start_model(self, model_name, eager, params: List[str] = [], use_model_default_param: bool = True):
        _, model_id, model_path = self.is_local(model_name)
        if not model_path:
            return False, "Model not available locally", None, None
        if model_id in self.running_models:
            if self.running_models[model_id].get("status") == "starting":
                # Model is already starting
                return True, "Model starting", self.get_port(model_id), self.running_models[model_id]["pid"]

        if self.is_running(model_id):
            self.last_access[model_id] = time.time()
            return True, "Model already running", self.get_port(model_id), self.running_models[model_id]["pid"]
        # if there is model running that is not this model, stop it
        # TODO: enable multiple models running at the same time
        if self.running_models:
            for running_model_id in list(self.running_models.keys()):
                if running_model_id != model_id and self.is_running(running_model_id):
                    logger.info(
                        f"Stopping running model {running_model_id} before starting {model_id}")
                    self.stop_model(running_model_id)
        with self.lock:
            port = self._find_free_port()
            self.running_models[model_id] = {
                "pid": None,
                "port": port,
                "process": None,
                "status": "starting",
                "server": None
            }
            local_dir = os.path.join(MODEL_LIB, model_path)
            env = os.environ.copy()
            if "gpt-oss" in model_id:
                os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN_VLLM_V1"
            logger.info(
                f"Starting model in a separate process {model_id} on port {port} with local dir {local_dir} model_name: {model_name}")
            try:
                cmd = [
                    "python3", "-m", "vllm.entrypoints.openai.api_server",
                    "--model", local_dir,
                    "--port", str(port)
                ]
                # proc = subprocess.Popen(cmd, env=env)
                settings = self.get_config(model_path)
                params = settings.get("params", []) if use_model_default_param else params
                logger.info(
                    f"Starting vLLM server with args: {params}"
                )
                proc = self.mp_context.Process(
                    target=start_server, args=(port, model_name, local_dir, eager, params))
                proc.start()

                if "gpt-oss" in model_id:
                    if "VLLM_ATTENTION_BACKEND" in env:
                        os.environ["VLLM_ATTENTION_BACKEND"] = env["VLLM_ATTENTION_BACKEND"]
                    else:
                        del os.environ["VLLM_ATTENTION_BACKEND"]
                self.running_models[model_id] = {
                    "pid": proc.pid,
                    "port": port,
                    "process": proc,
                    "status": "running"
                }
                self.used_ports.add(port)
                self.last_access[model_id] = time.time()
                return True, f"Model started on port {port}", port, proc.pid
            except Exception as e:
                port = self.running_models[model_id]["port"]
                self.used_ports.discard(port)
                del self.running_models[model_id]
                return False, f"Failed to start model: {str(e)}", None, None

    def stop_model(self, model_name):
        _, model_id, _ = self.is_local(model_name)
        with self.lock:
            if model_id in self.running_models:
                if not self.is_running(model_id):
                    return False, "Model not running"
                proc = self.running_models[model_id]["process"]
                proc.terminate()
                port = self.running_models[model_id]["port"]
                self.used_ports.discard(port)
                del self.running_models[model_id]
                self.last_access.pop(model_id, None)
                return True, "Model stopped"
            return False, "Model not running"

    def get_rammb(self, pid):
        """Get RAM usage in MB for a given process ID."""
        try:
            import psutil
            # Convert bytes to MB
            return sum([psutil.Process(cpid).memory_info().rss for cpid in self.get_pids_from_pid(pid)]) // (1024 * 1024)
        except Exception as e:
            logger.error(f"Failed to get RAM usage for PID {pid}: {str(e)}")
            return None

    def get_pids_from_pid(self, pid):
        """Get all child PIDs from a given PID."""
        try:
            import psutil
            process = psutil.Process(pid)
            # Include the parent PID
            return set([p.pid for p in process.children(recursive=True)] + [pid])
        except Exception as e:
            logger.error(f"Failed to get child PIDs for PID {pid}: {str(e)}")
            return {pid}

    def get_vram_mb(self, pid):
        pids = self.get_pids_from_pid(pid)
        logger.info(
            f"Checking VRAM for PID {pid} on and all its child processes {pids}")
        rst = []
        if os.environ.get("ACCELERATOR") == "CUDA":
            try:

                import pynvml
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    usage = 0
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    try:
                        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(
                            handle)
                    except pynvml.NVMLError:
                        procs = []
                    logger.info(
                        f"Device {i} has {len(procs)} running processes: {[p.pid for p in procs]}")

                    for p in procs:
                        if p.pid in pids:  # Check if the process is in the list of PIDs
                            usage += p.usedGpuMemory
                    # Convert bytes to MB
                    rst.append(daemon_pb2.VRAMInfo(
                        device_id=i, vram_mb=usage // (1024 * 1024)))
                pynvml.nvmlShutdown()
                return rst
            except Exception as e:
                logger.error(
                    f"Failed to get VRAM usage for PID {pid}: {str(e)}")
        return []

    def get_running_info(self, name, data):
        if data["status"] != "running":
            return daemon_pb2.RunningModel(model_name=name, pid=0, status=data["status"], ram_mb=0, vram_usage=[])
        pid = data["pid"]
        if pid is None:
            raise ValueError(
                f"Process ID for model '{name}' is None, cannot retrieve running info")
        ram = self.get_rammb(pid)
        vram = self.get_vram_mb(pid)
        return daemon_pb2.RunningModel(model_name=name, pid=data["pid"], status=data["status"], ram_mb=ram, vram_usage=vram)

    def list_running(self):
        _, alias = self.load_alias()
        return [(alias.get(name, name), name) for name in self.running_models]

    def get_model_size(self, model_path):
        """Get the size of a model in MB."""
        model_dir = os.path.join(MODEL_LIB, model_path)
        total_size = 0
        for root, _, files in os.walk(model_dir):
            for file in files:
                total_size += os.path.getsize(os.path.join(root, file))
        return total_size // (1024 * 1024)

    def get_port(self, model_name):
        return self.running_models.get(model_name, {}).get("port", None)

    def resolve_alias(self, model_name):
        if ":" in model_name and "/" in model_name:
            raise ValueError(
                f"Model ID should not contain both ':' and '/': {model_name}")
        if ":" in model_name and "/" not in model_name:
            import json
            import requests
            alias, variant = model_name.split(":", 1)
            # Fetch model info from homl.dev
            url = f"https://homl.dev/model-configs/{alias}.json"
            try:
                resp = requests.get(url, timeout=10)
                info = resp.json()
                variant_info = info.get("variants", {}).get(variant)
                if not variant_info:
                    raise ValueError(
                        f"Variant '{variant}' not found for alias '{alias}'")
                hf_id = variant_info.get("model_id")
                if not hf_id:
                    raise ValueError(
                        f"No model_id found for alias '{alias}' variant '{variant}, available variants: {list(info.get('variants', {}).keys())}'")
                
                params = info.get("params", [])
                variant_params = variant_info.get("params", []) if variant_info else []
                if not variant_params:
                    variant_info["params"] = params
                self.add_alias_entry(model_name, hf_id)
                return hf_id, variant_info
            except Exception as e:
                raise ValueError(
                    f"Failed to resolve alias: {str(e)} from server, please check if https://homl.dev/models/{alias}.html exists")
        return model_name, {}

    def model_info(self, model_id):
        from huggingface_hub import model_info
        try:
            info = model_info(model_id, files_metadata=True)
            total_size = 0
            for file in info.siblings:
                if hasattr(file, "size") and file.size:
                    total_size += file.size
            size_mb = total_size / (1024 * 1024)
            info.size = f"{size_mb:.2f} MB"
            return info
        except Exception as e:
            raise ValueError(
                f"Failed to fetch model info for '{model_id}': {str(e)}")

    def wait_for_model(self, port, timeout=MODEL_LOAD_TIMEOUT):
        """Wait for a model to be ready by checking if its port is available."""
        start_time = time.time()
        print(f"Waiting for model server on port {port} to be ready...")
        while not self.is_ready(port):
            print(
                f"[Not Ready]Waiting for model server on port {port} to be ready...")
            if time.time() - start_time > timeout:
                return False
            time.sleep(1)
        return True

    def is_ready(self, port):
        """Check if the model server is ready by making a simple HTTP request."""
        import httpx
        try:
            url = f"http://localhost:{port}/v1/models"
            response = httpx.get(url, timeout=5)
            return response.status_code == 200
        except httpx.RequestError:
            return False

    def download_model(self, model_id, hf_token=None):
        from huggingface_hub import snapshot_download

        model_path = hashlib.sha256(model_id.encode()).hexdigest()
        # Download to a local cache directory
        local_dir = os.path.join(MODEL_LIB, model_path)
        self.add_manifest_entry(model_id, model_path)
        try:
            snapshot_download(repo_id=model_id, local_dir=local_dir,
                              token=hf_token if hf_token else None)
            return model_path
        except Exception as e:
            import shutil
            shutil.rmtree(local_dir)
            raise ValueError(f"Failed to download model: {str(e)}")

    def get_total_rammb(self):
        """Get total RAM available in MB."""
        try:
            import psutil
            return psutil.virtual_memory().total // (1024 * 1024)  # Convert bytes to MB
        except Exception as e:
            logger.error(f"Failed to get total RAM: {str(e)}")
            return 0

    def get_total_vram_available(self):
        rst = {}
        if os.environ.get("ACCELERATOR") == "CUDA":
            try:
                import pynvml
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                    rst[i] = (info.total // (1024 * 1024),  info.used //
                              (1024 * 1024))  # Convert bytes to MB
                pynvml.nvmlShutdown()
                return rst
            except Exception as e:
                logger.error(f"Failed to get total VRAM: {str(e)}")
        return {}

