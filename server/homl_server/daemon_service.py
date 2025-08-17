
from logging import config
import daemon_pb2_grpc
import daemon_pb2
import os
import subprocess
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Server version (set at build time or fallback to dev)
SERVER_VERSION = os.environ.get("HOML_SERVER_VERSION", "dev")


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
        ok, msg, port, pid = self.model_manager.start_model(
            request.model_name, request.eager_mode, use_model_default_param=not request.use_params, params=request.params)
        if not ok:
            logger.error(f"Failed to start model {request.model_name}: {msg}")
            return daemon_pb2.StartModelResponse(message=msg, pid=0)
        if self.model_manager.wait_for_model(port):
            logger.info(
                f"Model {request.model_name} started successfully on port {port}")
            return daemon_pb2.StartModelResponse(message=msg, pid=pid)
        else:
            logger.error(
                f"Model {request.model_name} failed to start within timeout")
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
        if local and not request.refresh_config:
            yield daemon_pb2.PullModelProgress(message=f"Model '{model_name}' is already available locally.", percent=100, done=True, success=True)
            return

        if ":" in model_name:
            yield daemon_pb2.PullModelProgress(message=f"Resolving alias for '{model_name}'...", percent=1, done=False, success=False)
        # Resolve alias if needed
        config = {}
        try:
            model_id, config = self.model_manager.resolve_alias(model_name)
            if not model_id:
                yield daemon_pb2.PullModelProgress(message=f"Failed to resolve alias for '{model_name}'", percent=0, done=True, success=False)
                return
        except ValueError as e:
            yield daemon_pb2.PullModelProgress(message=str(e), percent=0, done=True, success=False)
            return
        local, model_id, model_path = self.model_manager.is_local(model_id)
        if local:
            yield daemon_pb2.PullModelProgress(message=f"Model '{model_id}' is already available locally.", percent=100, done=True, success=True)
            settings = self.model_manager.get_config(model_path)
            settings.update(config)
            self.model_manager.save_config(model_path, settings)
            return

        yield daemon_pb2.PullModelProgress(message=f"Pulling model '{model_id}'...", percent=2, done=False, success=False)

        try:
            model_info = self.model_manager.model_info(model_id)
            if not model_info:
                yield daemon_pb2.PullModelProgress(message=f"Model '{model_id}' not found", percent=100, done=True, success=False)
                return
            yield daemon_pb2.PullModelProgress(message=f"Model '{model_id}' found, size: {model_info.size}, starting download...\n(HF doesn't support progress, but we are downloading, I promise)", percent=3, done=False, success=True)
            model_path = self.model_manager.download_model(
                model_id, hf_token=request.hf_token)
            settings = self.model_manager.get_config(model_path)
            settings.update(config)
            self.model_manager.save_config(model_path, settings)
            yield daemon_pb2.PullModelProgress(message=f"Model '{model_id}' pulled", percent=100, done=True, success=True)
        except ValueError as e:
            yield daemon_pb2.PullModelProgress(message=f"Failed to pull model: {str(e)}", percent=100, done=True, success=False)
            return

    def ConfigModelParam(self, request, context):
        logger.info(f"Configuring model: {request.model_name}")
        model_name = request.model_name
        params = request.params.params

        local, model_id, model_path = self.model_manager.is_local(model_name)

        # Validate and apply the configuration
        if not local:
            raise ValueError(f"Model '{model_name}' is not available locally.")
        settings = self.model_manager.get_config(model_path)
        settings['params'] = list(params)
        self.model_manager.save_config(model_path, settings)
        return daemon_pb2.ModelConfigResponse(settings=daemon_pb2.ModelSettings(settings={}), params=daemon_pb2.ModelParam(params=params))

    def GetModelConfig(self, request, context):
        logger.info(f"Getting model config: {request.model_name}")
        model_name = request.model_name

        local, model_id, model_path = self.model_manager.is_local(model_name)

        if not local:
            raise ValueError(f"Model '{model_name}' is not available locally.")

        settings = self.model_manager.get_config(model_path)
        return daemon_pb2.ModelConfigResponse(settings=daemon_pb2.ModelSettings(settings={}), params=daemon_pb2.ModelParam(params=settings.get('params', [])))