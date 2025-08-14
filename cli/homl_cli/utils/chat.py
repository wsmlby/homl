import json
import click
import requests


def chat_with_model(model_name: str, api_url: str):
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
                    click.secho(
                        "Error: The model is not running or the server is not available.", fg="red")
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
                            click.secho(
                                f"Error decoding JSON: {text}", fg="red")
                            continue
                        if "choices" in json_data and len(json_data["choices"]) > 0:
                            text = json_data["choices"][0].get(
                                "delta", {}).get("content", "")
                        click.echo(text, nl=False)
                        response_text += text
                click.echo("")
                history.append({"role": "assistant", "content": response_text})
        except Exception as e:
            click.secho(f"Error communicating with model: {e}", fg="red")