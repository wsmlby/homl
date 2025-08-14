import grpc
from homl_cli import daemon_pb2, daemon_pb2_grpc
import click
from homl_cli.utils.spinner import Spinner
from homl_cli import config

def get_client_stub():
    """Creates and returns a gRPC client stub."""
    socket_path = config.get_socket_path()
    try:
        channel = grpc.insecure_channel(socket_path)
        grpc.channel_ready_future(channel).result(timeout=1)
        return daemon_pb2_grpc.DaemonStub(channel)
    except grpc.FutureTimeoutError:
        click.echo("Error: The HoML daemon is not running.")
        click.echo("Please make sure the server is installed and running, e.g. with 'homl server install'.")
        return None

def start_model(model_name, eager):
    """Starts a model with the vLLM server. Used by both run and chat commands."""
    stub = get_client_stub()
    if stub:
        spinner = Spinner(f"Starting model '{model_name}' (vLLM is a bit slow to start)...")
        spinner.start()
        try:
            response = stub.StartModel(daemon_pb2.StartModelRequest(model_name=model_name, eager_mode=eager))
        finally:
            spinner.stop()
        click.echo(response.message)
        return response.pid
    return 0
