# --- New Implementation ---
import argparse
import hashlib
import json
import multiprocessing
import os
import cloudpickle as pickle
import sys
import threading
import time
from pathlib import Path

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.multiprocessing.client import MQLLMEngineClient
from vllm.engine.multiprocessing.engine import run_mp_engine
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.utils import get_open_zmq_ipc_path, random_uuid

# gRPC imports
import grpc
from concurrent import futures
import daemon_pb2
import daemon_pb2_grpc

# FastAPI imports
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
import logging

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

os.makedirs(os.path.join(MODEL_HOME, "home"), exist_ok=True)
os.makedirs(MODEL_LIB, exist_ok=True)
os.makedirs(TORCH_CACHE, exist_ok=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = TORCH_CACHE
# Ensure cache and lib directories exist


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
                print(f"Error loading alias: {e}")
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
                print(f"Error loading manifest: {e}")
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
            print(f"Error writing manifest: {e}")
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
            print(f"Error writing alias: {e}")

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

    def __init__(self):
        self.running_models = {}  # model_id: {process, client, ipc_path, status, pid}
        self.last_access = {}  # model_name: last access timestamp
        self.lock = threading.Lock()
        self._start_idle_unload_thread()

    def _start_idle_unload_thread(self):
        def idle_unload_loop():
            while True:
                now = time.time()
                models_to_unload = []
                with self.lock:
                    for model_id in list(self.running_models.keys()):
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

    def is_running(self, model_id):
        return model_id in self.running_models and self.running_models[model_id]["process"].poll() is None

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

    def cache_model_config(self, model_id: str, model_path: str):
        logger.info(f"Caching vLLM config for model {model_id}")
        parser = AsyncEngineArgs.add_cli_args(argparse.ArgumentParser())
        # We need to provide a model arg to parse_args, otherwise it will fail
        # with "the following arguments are required: --model"
        # The value does not matter as we override it later.
        cli_args = parser.parse_args(["--model", "dummy"])
        cli_args.model = os.path.join(MODEL_LIB, model_path)
        # some other defaults that are not in EngineArgs but are in the openai server
        cli_args.served_model_name = model_id

        engine_args = AsyncEngineArgs.from_cli_args(cli_args)

        # This will create the config and also download the model weights if they are not present.
        # In our case, the model is already downloaded, so it should be fast.
        vllm_config = engine_args.create_engine_config()

        config_path = os.path.join(MODEL_LIB, model_path, "vllm_config.pkl")
        with open(config_path, "wb") as f:
            pickle.dump(vllm_config, f)

        logger.info(f"Saved vLLM config for model {model_id} to {config_path}")

    def start_model(self, model_name):
        _, model_id, model_path = self.is_local(model_name)
        if not model_path:
            return False, "Model not available locally", None, None
        
        if self.is_running(model_id):
            self.last_access[model_id] = time.time()
            # ZMQ doesn't use a port in the same way, but we need to return something
            # for the client to connect to. We will handle this in the client.
            # For now, returning None for the port.
            return True, "Model already running", None, self.running_models[model_id]["pid"]

        if self.running_models:
            for running_model_id in list(self.running_models.keys()):
                if running_model_id != model_id and self.is_running(running_model_id):
                    logger.info(f"Stopping running model {running_model_id} before starting {model_id}")
                    self.stop_model(running_model_id)

        with self.lock:
            ipc_path = get_open_zmq_ipc_path()

            config_path = os.path.join(MODEL_LIB, model_path, "vllm_config.pkl")
            if not os.path.exists(config_path):
                self.cache_model_config(model_id, model_path)

            with open(config_path, 'rb') as f:
                vllm_config = pickle.load(f)

            engine_alive = multiprocessing.Value('b', True, lock=False)
            ctx = multiprocessing.get_context("spawn")
            proc = ctx.Process(
                target=run_mp_engine,
                args=(vllm_config, UsageContext.OPENAI_API_SERVER, ipc_path,
                      True, # disable_log_stats
                      False, # enable_log_requests
                      engine_alive))
            proc.start()

            client = MQLLMEngineClient(ipc_path, vllm_config, proc.pid)

            self.running_models[model_id] = {
                "process": proc,
                "client": client,
                "ipc_path": ipc_path,
                "status": "running",
                "pid": proc.pid
            }
            self.last_access[model_id] = time.time()
            return True, f"Model engine started for {model_id}", None, proc.pid

    def stop_model(self, model_name):
        _, model_id, _ = self.is_local(model_name)
        with self.lock:
            if model_id in self.running_models:
                proc = self.running_models[model_id]["process"]
                proc.terminate()
                client = self.running_models[model_id]["client"]
                client.close()
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
        pid = data["pid"]
        if pid is None:
            raise ValueError(f"Process ID for model '{name}' is None, cannot retrieve running info")
        ram = self.get_rammb(pid)
        vram = self.get_vram_mb(pid)
        return daemon_pb2.RunningModel(model_name=name, pid=data["pid"], status=data["status"], ram_mb=ram, vram_usage=vram)
    def list_running(self):
        _, alias = self.load_alias()
        return [(alias.get(name, name), name) for name in self.running_models if self.is_running(name)]
    
    def get_model_size(self, model_path):
        """Get the size of a model in MB."""
        model_dir = os.path.join(MODEL_LIB, model_path)
        total_size = 0
        for root, _, files in os.walk(model_dir):
            for file in files:
                total_size += os.path.getsize(os.path.join(root, file))
        return total_size // (1024 * 1024)

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

    def wait_for_model(self, model_id, timeout=MODEL_LOAD_TIMEOUT):
        """Wait for a model to be ready by checking its engine client."""
        start_time = time.time()
        client = self.running_models[model_id]["client"]
        while True:
            if time.time() - start_time > timeout:
                return False
            try:
                # setup() will block until the engine is ready.
                client.setup()
                return True
            except TimeoutError:
                # The process might have died.
                proc = self.running_models[model_id]["process"]
                if not proc.is_alive():
                    return False
                continue
            except Exception as e:
                logger.error(f"Error setting up engine client for {model_id}: {e}")
                return False

    def download_model(self, model_id, hf_token=None):
        from huggingface_hub import snapshot_download

        model_path = hashlib.sha256(model_id.encode()).hexdigest()
        # Download to a local cache directory
        local_dir = os.path.join(MODEL_LIB, model_path)
        self.add_manifest_entry(model_id, model_path)
        try:
            snapshot_download(repo_id=model_id,
                              local_dir=local_dir,
                              token=hf_token if hf_token else None)
            self.cache_model_config(model_id, model_path)
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
        ok, msg, _, pid = self.model_manager.start_model(request.model_name)
        if ok and self.model_manager.wait_for_model(request.model_name):
            logger.info(f"Model {request.model_name} started successfully")
            return daemon_pb2.StartModelResponse(message=msg, pid=pid)
        else:
            logger.error(f"Model {request.model_name} failed to start within timeout")
            self.model_manager.stop_model(request.model_name)
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

    from fastapi import BackgroundTasks

    @app.post("/v1/chat/completions")
    async def chat_completions(request: dict, background_tasks: BackgroundTasks):
        model_name = request.get("model")
        if not model_name:
            raise HTTPException(status_code=400, detail="Missing model name")

        logger.info(f"Received chat completion request for model: {model_name}")
        local, model_id, model_path = model_manager.is_local(model_name)
        if not local:
            raise HTTPException(status_code=404,
                                detail=f"Model '{model_name}' not found locally."
                                )

        # Ensure model is running
        if not model_manager.is_running(model_id):
            logger.info(f"Model {model_id} is not running, starting it...")
            ok, msg, _, pid = model_manager.start_model(model_name)
            if not ok:
                raise HTTPException(status_code=500, detail=msg)
            if not model_manager.wait_for_model(model_id):
                model_manager.stop_model(model_id)
                raise HTTPException(status_code=500,
                                    detail="Model failed to start.")

        # Update last access time
        model_manager.last_access[model_id] = time.time()

        client = model_manager.running_models[model_id]["client"]

        try:
            sampling_params = SamplingParams(
                n=request.get("n", 1),
                presence_penalty=request.get("presence_penalty", 0.0),
                frequency_penalty=request.get("frequency_penalty", 0.0),
                temperature=request.get("temperature", 1.0),
                top_p=request.get("top_p", 1.0),
                max_tokens=request.get("max_tokens", 256),
                stop=request.get("stop", []),
            )

            messages = request.get("messages", [])
            tokenizer = await client.get_tokenizer()
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

            request_id = f"homl-{random_uuid()}"
            stream = request.get("stream", False)

            if stream:
                async def generate_stream():
                    previous_texts = [""] * sampling_params.n
                    generator = client.generate(prompt, sampling_params, request_id)
                    async for request_output in generator:
                        for i, output in enumerate(request_output.outputs):
                            delta_text = output.text[len(previous_texts[i]):]
                            previous_texts[i] = output.text
                            if delta_text:
                                yield f"data: {json.dumps({'choices': [{'delta': {'content': delta_text}}]})}\n\n"
                    yield "data: [DONE]\n\n"
                return StreamingResponse(generate_stream(), media_type="text/event-stream")

            else:
                generator = client.generate(prompt, sampling_params, request_id)
                final_output = None
                async for request_output in generator:
                    final_output = request_output

                choices = []
                for i, output in enumerate(final_output.outputs):
                    choices.append({
                        "index": i,
                        "message": {
                            "role": "assistant",
                            "content": output.text,
                        },
                        "finish_reason": output.finish_reason,
                    })

                response = {
                    "id": request_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": choices,
                    "usage": {
                        "prompt_tokens": len(final_output.prompt_token_ids),
                        "completion_tokens": sum(len(output.token_ids) for output in final_output.outputs),
                        "total_tokens": len(final_output.prompt_token_ids) + sum(len(output.token_ids) for output in final_output.outputs),
                    }
                }
                return JSONResponse(response)
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            raise HTTPException(status_code=500, detail=str(e))

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
>>>>>>> REPLACE
