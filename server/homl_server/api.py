import json
import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import httpx
from fastapi.responses import StreamingResponse
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def proxy_streaming_request(request, path, port, model_name, model_id, model_manager):
    stream = request.get("stream", False)
    logger.info(
        f"Proxying request for model {model_name} on port {port}, stream={stream} for request: {request}")
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
                                    model_manager.last_access[model_id] = time.time(
                                    )
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
        raise HTTPException(
            status_code=500, detail=f"vLLM API error: {str(e)}")


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
            ok, msg, port, pid = model_manager.start_model(model_id, eager, [], use_model_default_param=True)
            logger.info(f"Model start response: {msg}")
            if not ok:
                raise HTTPException(status_code=500, detail=msg)
        else:
            raise HTTPException(
                status_code=404, detail="Model not available locally")
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
        return JSONResponse({"data": models})

    return app
