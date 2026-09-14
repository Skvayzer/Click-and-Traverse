"""Cooperative manual stopping at a completed PPO/checkpoint boundary."""
from pathlib import Path
import signal


class StopRequest:
    def __init__(self, stop_file=None):
        self.stop_file = Path(stop_file) if stop_file is not None else None
        self.signal_reason = None
        self._previous = {}

    def _handle_signal(self, signum, frame):
        del frame
        self.signal_reason = signal.Signals(signum).name

    def __enter__(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._previous[sig] = signal.signal(sig, self._handle_signal)
        return self

    def __exit__(self, *exc):
        for sig, previous in self._previous.items():
            signal.signal(sig, previous)

    @property
    def reason(self):
        if self.signal_reason is not None:
            return self.signal_reason
        if self.stop_file is not None and self.stop_file.exists():
            return "stop_file"
        return None

    def requested(self):
        return self.reason is not None
