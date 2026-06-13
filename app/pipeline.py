"""Inference worker: capture -> detect -> annotate -> JPEG buffer.

Decoupled from both the camera (pulls newest frame) and the HTTP clients
(they read the latest encoded JPEG), so neither blocks the other.
"""
import threading
import time

import cv2

from .config import state
from .stats import FpsTracker


class InferenceWorker:
    def __init__(self, camera, detector, jpeg_quality=80, encode_jpeg=True):
        self.camera = camera
        self.detector = detector
        self.jpeg_quality = jpeg_quality
        self.encode_jpeg = encode_jpeg   # web needs JPEG; local Tk viewer reads ndarray
        self.fps = FpsTracker()

        self._jpeg = None            # latest encoded annotated frame (bytes)
        self._annotated = None       # latest annotated frame (BGR ndarray)
        self._counts = {}
        self._lock = threading.Lock()
        self._running = False
        self._paused = False
        self._thread = None
        self._new_frame = threading.Event()

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        enc = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        while self._running:
            if self._paused:        # GPU handed to training; idle the loop
                time.sleep(0.1)
                continue
            frame = self.camera.read_latest()
            if frame is None:
                time.sleep(0.005)
                continue

            with self._lock:
                detector = self.detector
            cfg = state.snapshot()
            try:
                if cfg["detect_enabled"]:
                    annotated, counts = detector.infer(frame)
                else:
                    annotated, counts = frame, {}
            except Exception as e:  # noqa: BLE001 - one bad frame/model must not kill the loop
                print(f"inference error (skipping frame): {type(e).__name__}: {e}")
                time.sleep(0.05)
                continue

            jpeg = None
            if self.encode_jpeg:
                ok, buf = cv2.imencode(".jpg", annotated, enc)
                if not ok:
                    continue
                jpeg = buf.tobytes()

            with self._lock:
                self._annotated = annotated
                if jpeg is not None:
                    self._jpeg = jpeg
                self._counts = counts
            self.fps.tick()
            self._new_frame.set()

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def set_detector(self, detector):
        with self._lock:
            self.detector = detector

    def get_jpeg(self):
        with self._lock:
            return self._jpeg

    def get_annotated(self):
        """Latest annotated BGR frame as ndarray (for the native local viewer)."""
        with self._lock:
            return None if self._annotated is None else self._annotated

    def wait_for_frame(self, timeout=1.0):
        """Block until a new frame is encoded (for paced MJPEG streaming)."""
        triggered = self._new_frame.wait(timeout)
        if triggered:
            self._new_frame.clear()
        return triggered

    def stats(self) -> dict:
        with self._lock:
            counts = dict(self._counts)
        return {"fps": round(self.fps.fps(), 1), "counts": counts}

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
