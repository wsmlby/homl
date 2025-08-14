import os
from pathlib import Path

import grpc
from concurrent import futures
from api import create_api_app
from model_manager import ModelManager

import daemon_pb2_grpc
from daemon_service import DaemonServicer
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def serve():
    SOCKET_DIR = Path("/var/run/homl")
    SOCKET_PATH = SOCKET_DIR / "homl.sock"
    UNIX_SOCKET_PATH = f"unix://{SOCKET_PATH}"
    SOCKET_DIR.mkdir(parents=True, exist_ok=True)

    model_manager = ModelManager()

    # Start gRPC server (blocking)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    daemon_pb2_grpc.add_DaemonServicer_to_server(
        DaemonServicer(model_manager), server)
    server.add_insecure_port(UNIX_SOCKET_PATH)
    server.start()
    logger.info(f"gRPC server started on {UNIX_SOCKET_PATH}")
    if os.environ.get("HOML_INSECURE_SOCKET", "false").lower() == "true":
        try:
            os.chmod(SOCKET_PATH, 0o777)
            logger.info(
                f"Socket permissions set to world-writable for {SOCKET_PATH}")
        except OSError as e:
            logger.info(f"Error setting socket permissions: {e}")

    # Start FastAPI server (blocking)
    app = create_api_app(model_manager)
    logger.info("Starting OpenAI-compatible API server on 0.0.0.0:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    serve()
