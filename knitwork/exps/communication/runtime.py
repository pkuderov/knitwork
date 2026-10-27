"""Cooperative session pauses; never interrupt an optimizer update halfway."""

import signal
import time


class PauseRequested(Exception):
    pass


class StopController:
    def __init__(self, deadline=None):
        self.deadline = deadline
        self.reason = None
        self.handlers = {}

    def request(self, reason):
        if self.reason is None:
            self.reason = reason

    def check(self):
        if self.deadline is not None and time.time() >= self.deadline:
            self.request('daily_deadline')
        if self.reason is not None:
            raise PauseRequested(self.reason)

    def install(self):
        def handler(number, frame):
            self.request(signal.Signals(number).name)
        for number in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
            self.handlers[number] = signal.signal(number, handler)

    def restore(self):
        for number, handler in self.handlers.items():
            signal.signal(number, handler)
