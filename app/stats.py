"""Telemetry: rolling FPS counter + tegrastats parser for GPU load / temp."""
import re
import subprocess
import threading
import time
from collections import deque


class FpsTracker:
    def __init__(self, window=30):
        self._times = deque(maxlen=window)
        self._lock = threading.Lock()

    def tick(self):
        with self._lock:
            self._times.append(time.time())

    def fps(self) -> float:
        with self._lock:
            if len(self._times) < 2:
                return 0.0
            span = self._times[-1] - self._times[0]
            return (len(self._times) - 1) / span if span > 0 else 0.0


# tegrastats sample fields we care about.
_GPU_RE = re.compile(r"GR3D_FREQ (\d+)%")
_TEMP_RE = re.compile(r"(?:gpu|GPU)@([\d.]+)C")
_RAM_RE = re.compile(r"RAM (\d+)/(\d+)MB")


class TegraStats:
    """Reads `tegrastats` in the background; exposes latest GPU%, temp, RAM."""

    def __init__(self, interval_ms=1000):
        self.interval_ms = interval_ms
        self._proc = None
        self._thread = None
        self._running = False
        self._lock = threading.Lock()
        self._data = {"gpu": None, "temp": None, "ram_used": None, "ram_total": None}

    def start(self):
        try:
            self._proc = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
        except FileNotFoundError:
            print("tegrastats not found — GPU/temp telemetry disabled")
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        for line in self._proc.stdout:
            if not self._running:
                break
            gpu = _GPU_RE.search(line)
            temp = _TEMP_RE.search(line)
            ram = _RAM_RE.search(line)
            with self._lock:
                if gpu:
                    self._data["gpu"] = int(gpu.group(1))
                if temp:
                    self._data["temp"] = float(temp.group(1))
                if ram:
                    self._data["ram_used"] = int(ram.group(1))
                    self._data["ram_total"] = int(ram.group(2))

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._data)

    def stop(self):
        self._running = False
        if self._proc:
            self._proc.terminate()
