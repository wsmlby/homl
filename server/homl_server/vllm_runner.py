import argparse
import asyncio
import pickle
import uvicorn
from argparse import Namespace
from fastapi import FastAPI

from vllm.config import VllmConfig
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.api_server import (build_app, init_app_state,
                                                 lifespan)
from vllm.usage.usage_lib import UsageContext


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        type=str,
                        required=True,
                        help="Path to the VllmConfig pickle file.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on.")
    parser.add_argument("--port",
                        type=int,
                        required=True,
                        help="Port to run the server on.")
    parser.add_argument("--served-model-name",
                        type=str,
                        required=True,
                        help="The name of the model served")
    # Add other necessary args for init_app_state with their defaults
    parser.add_argument("--response-role", type=str, default="assistant")
    parser.add_argument("--chat-template", type=str, default=None)
    parser.add_argument("--enable-auto-tool-choice", action="store_true", default=False)
    parser.add_argument("--tool-call-parser", type=str, default=None)
    parser.add_argument("--reasoning-parser", type=str, default=None)
    parser.add_argument("--lora-modules", type=str, default=None)
    parser.add_argument("--disable-log-requests", action="store_true", default=True)
    parser.add_argument("--max-log-len", type=int, default=None)

    args = parser.parse_args()

    with open(args.config, 'rb') as f:
        vllm_config: VllmConfig = pickle.load(f)

    # Create the AsyncLLMEngine
    engine = AsyncLLMEngine.from_vllm_config(
        vllm_config,
        usage_context=UsageContext.OPENAI_API_SERVER,
    )

    # Create a minimal namespace for init_app_state
    # We need to provide values that would normally come from EngineArgs
    app_args = Namespace(
        served_model_name=[args.served_model_name],
        model=vllm_config.model_config.model,
        disable_log_requests=args.disable_log_requests,
        max_log_len=args.max_log_len,
        response_role=args.response_role,
        chat_template=args.chat_template,
        chat_template_content_format=None,
        return_tokens_as_token_ids=False,
        enable_auto_tool_choice=args.enable_auto_tool_choice,
        tool_call_parser=args.tool_call_parser,
        reasoning_parser=args.reasoning_parser,
        enable_prompt_tokens_details=False,
        enable_force_include_usage=False,
        lora_modules=args.lora_modules,
        disable_fastapi_docs=False,
        root_path="",
        middleware=[],
        allowed_origins=["*"],
        allow_credentials=True,
        allowed_methods=["*"],
        allowed_headers=["*"],
        api_key=None,
        enable_request_id_headers=False,
        enable_server_load_tracking=False,
    )

    app = build_app(app_args)

    # The lifespan function requires the event loop to be running.
    # We'll run it in a separate task.
    async def run_app():
        await init_app_state(engine, vllm_config, app.state, app_args)

        config = uvicorn.Config(app,
                                host=args.host,
                                port=args.port,
                                log_level="info",
                                lifespan="on")
        server = uvicorn.Server(config)
        await server.serve()

    asyncio.run(run_app())


if __name__ == "__main__":
    main()
