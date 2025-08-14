import time
import threading
import itertools
import click

class Spinner:
    def __init__(self, message="Starting..."):
        self.message = message
        self.done = threading.Event()
        self.spinner_cycle = itertools.cycle(['|', '/', '-', '\\'])
        self.thread = threading.Thread(target=self._spin)

    def _spin(self):
        while not self.done.is_set():
            click.echo(f"\r{self.message} {next(self.spinner_cycle)}", nl=False)
            time.sleep(0.1)

    def start(self):
        self.thread.start()

    def stop(self, ok_message="✅        "):
        self.done.set()
        self.thread.join()
        click.echo(f"\r{self.message} {ok_message}")
