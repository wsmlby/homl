import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

# gRPC imports
import grpc
from concurrent import futures
import daemon_pb2
import daemon_pb2_grpc

# FastAPI imports
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import httpx
from fastapi.responses import StreamingResponse
import logging
import multiprocessing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Server version (set at build time or fallback to dev)
SERVER_VERSION = os.environ.get("HOML_SERVER_VERSION", "dev")

import subprocess
import random
MODEL_HOME = os.environ.get("HOML_MODEL_HOME", "/models")
MODEL_LIB = os.path.join(MODEL_HOME, "lib")
TORCH_CACHE = os.path.join(MODEL_LIB, "torch_cache")
MODEL_LOAD_TIMEOUT = int(os.environ.get("HOML_MODEL_LOAD_TIMEOUT", 180))  # seconds
# # This is the time after which a model will be unloaded if it is idle
MODEL_UNLOAD_IDLE_TIME = int(os.environ.get("HOML_MODEL_UNLOAD_IDLE_TIME", 600))  # 10 minutes default
module_info_cache = os.path.join(MODEL_HOME, "module_info_cache")

os.makedirs(os.path.join(MODEL_HOME, "home"), exist_ok=True)
os.makedirs(MODEL_LIB, exist_ok=True)
os.makedirs(TORCH_CACHE, exist_ok=True)
os.makedirs(module_info_cache, exist_ok=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = TORCH_CACHE
os.environ["VLLM_LAZY_LOAD_MODULE_INFO_CACHE"] = module_info_cache
# Ensure cache and lib directories exist

import uvloop
from vllm.utils import (FlexibleArgumentParser)
from vllm.entrypoints.openai.tool_parsers import ToolParserManager
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.cli_args import FrontendArgs
from vllm.entrypoints.openai.api_server import build_app, init_app_state, maybe_register_tokenizer_info_endpoint, build_async_engine_client, run_server, make_arg_parser, validate_parsed_serve_args


def start_server(port: int, model_name: str, model_path: str, eager):
    """
    Starts the vLLM server in a separate thread.
    This is a workaround to avoid starting a new process for each request.
    """
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args(["--model", model_path, "--port", str(port)] + (["--enforce-eager"] if eager else []))
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
                    return alias, {v:k for k, v in alias.items()}  # Reverse mapping
            except Exception as e:
                logger.error(f"Error loading alias: {e}")
        return {}, {}
    def load_manifest(self):
        # Load manifest.json if exists
        manifest_path = Path(MODEL_HOME) / "manifest.json"
        if manifest_path.exists():
            try:
                import json
                with open(manifest_path) as f:
                    mapping = json.load(f)
                    return mapping, {v:k for k, v in mapping.items()}  # Reverse mapping
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
        logger.info(f"Local model paths: {model_paths}, manifest: {manifest}, reverse_alias: {reverse_alias}")
        model_ids = [(manifest.get(model), model) for model in model_paths if model in manifest]
        models = [(reverse_alias.get(model, model), self.get_model_size(model_path) if with_size else -1, model) for model, model_path in model_ids]
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
                            logger.info(f"Unloading idle model {model_id} after {now - last:.1f}s of inactivity")
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

    def start_model(self, model_name, eager):
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
                    logger.info(f"Stopping running model {running_model_id} before starting {model_id}")
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
            logger.info(f"Starting model in a separate process {model_id} on port {port} with local dir {local_dir} model_name: {model_name}")
            try:
                cmd = [
                    "python3", "-m", "vllm.entrypoints.openai.api_server",
                    "--model", local_dir,
                    "--port", str(port)
                ]
                # proc = subprocess.Popen(cmd, env=env)
                proc = self.mp_context.Process(target=start_server, args=(port, model_name, local_dir, eager))
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
            return sum([psutil.Process(cpid).memory_info().rss  for cpid in self.get_pids_from_pid(pid)]) // (1024 * 1024)  # Convert bytes to MB
        except Exception as e:
            logger.error(f"Failed to get RAM usage for PID {pid}: {str(e)}")
            return None
    def get_pids_from_pid(self, pid):
        """Get all child PIDs from a given PID."""
        try:
            import psutil
            process = psutil.Process(pid)
            return set([p.pid for p in process.children(recursive=True)] + [pid])  # Include the parent PID
        except Exception as e:
            logger.error(f"Failed to get child PIDs for PID {pid}: {str(e)}")
            return {pid}
    def get_vram_mb(self, pid):
        pids = self.get_pids_from_pid(pid)
        logger.info(f"Checking VRAM for PID {pid} on and all its child processes {pids}")
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
                        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    except pynvml.NVMLError:
                        procs = []
                    logger.info(f"Device {i} has {len(procs)} running processes: {[p.pid for p in procs]}")

                    for p in procs:
                        if p.pid in pids:  # Check if the process is in the list of PIDs
                            usage += p.usedGpuMemory
                    rst.append(daemon_pb2.VRAMInfo(device_id=i, vram_mb=usage // (1024 * 1024)))  # Convert bytes to MB
                pynvml.nvmlShutdown()
                return rst
            except Exception as e:
                logger.error(f"Failed to get VRAM usage for PID {pid}: {str(e)}")
        return []
        

    def get_running_info(self, name, data):
        if data["status"] != "running":
            return daemon_pb2.RunningModel(model_name=name, pid=0, status=data["status"], ram_mb=0, vram_usage=[])
        pid = data["pid"]
        if pid is None:
            raise ValueError(f"Process ID for model '{name}' is None, cannot retrieve running info")
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
            raise ValueError(f"Model ID should not contain both ':' and '/': {model_name}")
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
                    raise ValueError(f"Variant '{variant}' not found for alias '{alias}'")
                hf_id = variant_info.get("model_id")
                if not hf_id:
                    raise ValueError(f"No model_id found for alias '{alias}' variant '{variant}, available variants: {list(info.get('variants', {}).keys())}'")
                self.add_alias_entry(model_name, hf_id)
                return hf_id
            except Exception as e:
                raise ValueError(f"Failed to resolve alias: {str(e)} from server, please check if https://homl.dev/models/{alias}.html exists")
        return model_name
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
            raise ValueError(f"Failed to fetch model info for '{model_id}': {str(e)}")
    def wait_for_model(self, port, timeout=MODEL_LOAD_TIMEOUT):
        """Wait for a model to be ready by checking if its port is available."""
        start_time = time.time()
        print(f"Waiting for model server on port {port} to be ready...")
        while not self.is_ready(port):
            print(f"[Not Ready]Waiting for model server on port {port} to be ready...")
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
            snapshot_download(repo_id=model_id, local_dir=local_dir, token=hf_token if hf_token else None)
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

                    rst[i] = (info.total // (1024 * 1024),  info.used // (1024 * 1024))  # Convert bytes to MB
                pynvml.nvmlShutdown()
                return rst
            except Exception as e:
                logger.error(f"Failed to get total VRAM: {str(e)}")
        return {}


# gRPC Servicer
class DaemonServicer(daemon_pb2_grpc.DaemonServicer):
    def Version(self, request, context):
        return daemon_pb2.VersionResponse(version=SERVER_VERSION)
    def ListLocalModels(self, request, context):
        logger.info("Listing local models")
        models = self.model_manager.list_local(request.with_size)
        return daemon_pb2.ListLocalModelsResponse(models=models)
    def __init__(self, model_manager):
        self.model_manager = model_manager

    def StartModel(self, request, context):
        logger.info(f"Starting model: {request.model_name}")
        ok, msg, port, pid = self.model_manager.start_model(request.model_name, request.eager_mode)
        if not ok:
            logger.error(f"Failed to start model {request.model_name}: {msg}")
            return daemon_pb2.StartModelResponse(message=msg, pid=0)
        if self.model_manager.wait_for_model(port):
            logger.info(f"Model {request.model_name} started successfully on port {port}")
            return daemon_pb2.StartModelResponse(message=msg, pid=pid)
        else:
            logger.error(f"Model {request.model_name} failed to start within timeout")
            return daemon_pb2.StartModelResponse(message="Model failed to start", pid=pid)

    def StopModel(self, request, context):
        logger.info(f"Stopping model: {request.model_name}")
        ok, msg = self.model_manager.stop_model(request.model_name)
        return daemon_pb2.StopModelResponse(message=msg)
    

    def ListRunningModels(self, request, context):
        logger.info("Listing running models")
        models = []
        for name, model_id in self.model_manager.list_running():
            data = self.model_manager.running_models[model_id]
            models.append(self.model_manager.get_running_info(name, data))
        
        return daemon_pb2.ListRunningModelsResponse(models=models, total_ram_mb=self.model_manager.get_total_rammb(),
                                                    vram_total=[
            daemon_pb2.VRAMInfo(device_id=device_id, vram_mb=vram[0]) for device_id, vram in self.model_manager.get_total_vram_available().items()
        ])

    def PullModel(self, request, context):
        logger.info(f"Pulling model: {request.model_name}")
        model_name = request.model_name
        yield daemon_pb2.PullModelProgress(message=f"Pulling model '{model_name}'...", percent=0, done=False, success=False)
        # Check if model is already local
        local, model_id, model_path = self.model_manager.is_local(model_name)
        if local:
            yield daemon_pb2.PullModelProgress(message=f"Model '{model_name}' is already available locally.", percent=100, done=True, success=True)
            return

        if ":" in model_name:
            yield daemon_pb2.PullModelProgress(message=f"Resolving alias for '{model_name}'...", percent=1, done=False, success=False)
        # Resolve alias if needed
        try:
            model_id = self.model_manager.resolve_alias(model_name)
            if not model_id:
                yield daemon_pb2.PullModelProgress(message=f"Failed to resolve alias for '{model_name}'", percent=0, done=True, success=False)
                return
        except ValueError as e:
            yield daemon_pb2.PullModelProgress(message=str(e), percent=0, done=True, success=False)
            return
        local, model_id, model_path = self.model_manager.is_local(model_id)
        if local:
            yield daemon_pb2.PullModelProgress(message=f"Model '{model_id}' is already available locally.", percent=100, done=True, success=True)
            return

        yield daemon_pb2.PullModelProgress(message=f"Pulling model '{model_id}'...", percent=2, done=False, success=False)

        try:
            model_info =self.model_manager.model_info(model_id)
            if not model_info:
                yield daemon_pb2.PullModelProgress(message=f"Model '{model_id}' not found", percent=100, done=True, success=False)
                return
            yield daemon_pb2.PullModelProgress(message=f"Model '{model_id}' found, size: {model_info.size}, starting download...\n(HF doesn't support progress, but we are downloading, I promise)", percent=3, done=False, success=True)
            self.model_manager.download_model(model_id, hf_token=request.hf_token)
            yield daemon_pb2.PullModelProgress(message=f"Model '{model_id}' pulled", percent=100, done=True, success=True)
        except ValueError as e:
            yield daemon_pb2.PullModelProgress(message=f"Failed to pull model: {str(e)}", percent=100, done=True, success=False)
            return

async def proxy_streaming_request(request, path, port, model_name, model_id, model_manager):
    stream = request.get("stream", False)
    logger.info(f"Proxying request for model {model_name} on port {port}, stream={stream} for request: {request}")
    url = f"http://localhost:{port}{path}"
    
    try:   
        if stream:
            async def generate_chunks():
                async with httpx.AsyncClient() as client:
                    async with client.stream('POST', url, json=request, timeout=None) as upstream_response:
                        async for chunk in upstream_response.aiter_text():
                            if chunk.startswith("data: "):
                                try:
                                    data0 = json.loads(chunk[6:].strip())
                                    data0["model"] = model_name
                                    chunk = f"data: {json.dumps(data0)}\n\n"
                                    model_manager.last_access[model_id] = time.time()
                                except json.JSONDecodeError:
                                    pass
                            yield chunk
            return StreamingResponse(generate_chunks(), media_type="text/event-stream")
        else:
            async with httpx.AsyncClient() as client:
                vllm_response = await client.post(url, json=request, timeout=30.0)
                rst = vllm_response.json()
                rst['model'] = model_name
                return JSONResponse(rst, status_code=vllm_response.status_code)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"vLLM API error: {str(e)}")

def get_model_ready(model_name, model_manager, eager):
    local, model_id, model_path = model_manager.is_local(model_name)
    if not model_id:
        raise HTTPException(status_code=400, detail="Missing model name")
    logger.info(f"Checking if model {model_id} is running")
    # Update last access time for model
    model_manager.last_access[model_id] = time.time()
    if not model_manager.is_running(model_id):
        logger.info(f"Model {model_id} is not running, starting it")
        if local:
            ok, msg, port, pid = model_manager.start_model(model_id, eager)
            logger.info(f"Model start response: {msg}")
            if not ok:
                raise HTTPException(status_code=500, detail=msg)
        else:
            raise HTTPException(status_code=404, detail="Model not available locally")
    port = model_manager.get_port(model_id)
    logger.info(f"Model {model_id} is running on port {port}")
    if model_manager.wait_for_model(port):
        logger.info(f"Model {model_id} is ready on port {port}")
    if not port:
        raise HTTPException(status_code=500, detail="Model port not found")
    return port, model_id


async def proxied_api(request, path, model_manager):
    model_name = request.get("model")
    logger.info(f"Received request@ path: {path} for model: {model_name}")
    del request["model"]
    port, model_id = get_model_ready(model_name, model_manager, True)
    return await proxy_streaming_request(request, path, port, model_name, model_id, model_manager)

# FastAPI OpenAI-Compatible API
def create_api_app(model_manager):
    app = FastAPI()
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


    
    @app.post("/v1/chat/completions")
    async def chat_completions(request: dict):
        return await proxied_api(request, "/v1/chat/completions", model_manager)
    
    @app.post("/v1/completions")
    async def completions(request: dict):
        return await proxied_api(request, "/v1/completions", model_manager)

    @app.post("/v1/responses")
    async def responses(request: dict):
        return await proxied_api(request, "/v1/responses", model_manager)

    @app.get("/v1/models")
    async def list_models():
        logger.info("Listing available models")
        _, alias = model_manager.load_alias()
        _, manifest = model_manager.load_manifest()
        models = []
        for model_path in model_manager.list_model_paths():
            model_id = manifest.get(model_path)
            if not model_id:
                continue
            alias_name = alias.get(model_id, model_id)
            models.append({
                "id": alias_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "homl",
                "root": model_id,
                "parent": None,
            })
        return  JSONResponse({"data": models})

    return app

# Entrypoint
def serve():
    SOCKET_DIR = Path("/var/run/homl")
    SOCKET_PATH = SOCKET_DIR / "homl.sock"
    UNIX_SOCKET_PATH = f"unix://{SOCKET_PATH}"
    SOCKET_DIR.mkdir(parents=True, exist_ok=True)

    model_manager = ModelManager()

    # Start gRPC server (blocking)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    daemon_pb2_grpc.add_DaemonServicer_to_server(DaemonServicer(model_manager), server)
    server.add_insecure_port(UNIX_SOCKET_PATH)
    server.start()
    logger.info(f"gRPC server started on {UNIX_SOCKET_PATH}")
    if os.environ.get("HOML_INSECURE_SOCKET", "false").lower() == "true":
        try:
            os.chmod(SOCKET_PATH, 0o777)
            logger.info(f"Socket permissions set to world-writable for {SOCKET_PATH}")
        except OSError as e:
            logger.info(f"Error setting socket permissions: {e}")

    # Start FastAPI server (blocking)
    app = create_api_app(model_manager)
    logger.info("Starting OpenAI-compatible API server on 0.0.0.0:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080)

if __name__ == "__main__":
    serve()
