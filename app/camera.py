"""Latest-frame-wins USB camera capture thread.

Continuously reads frames and keeps only the newest one, so downstream
inference never builds up latency from a backlog of stale frames.

Exclusive: only one process may own a given camera device at a time. Two
processes grabbing the same USB cam (e.g. web + local viewer at once) makes
the camera hand out black/empty frames, which silently breaks detection.
A per-device file lock turns that into a clear, early error instead.
"""
import fcntl
import os
import tempfile
import threading
import time

import cv2

# Frames whose mean pixel value is below this are treated as black/empty
# (lens cap, camera not ready, or device contention).
BLACK_FRAME_MEAN = 5.0


class CameraBusyError(RuntimeError):
    """Raised when the camera device is already owned by another process."""


class CaptureThread:
    def __init__(self, device=0, width=1280, height=720):
        self.device = device
        self.width = width
        self.height = height
        self._cap = None
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self.frames_read = 0
        self._lock_fd = None
        self._black_warned = False
        self._black_streak = 0

    def _acquire_device_lock(self):
        """Exclusive per-device lock so two instances can't share one camera."""
        path = os.path.join(tempfile.gettempdir(), f"orin-yolo-cam-{self.device}.lock")
        fd = open(path, "w")
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fd.close()
            raise CameraBusyError(
                f"camera device {self.device} is already in use by another "
                f"instance. Run one mode at a time (web OR local), not both."
            )
        self._lock_fd = fd

    def _open(self):
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # Small internal buffer => fresher frames.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def start(self):
        if self._running:
            return
        self._acquire_device_lock()
        self._cap = self._open()
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open camera device {self.device}")
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        fail = 0
        while self._running:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                fail += 1
                if fail > 30:  # reconnect after sustained failure
                    self._cap.release()
                    time.sleep(0.5)
                    self._cap = self._open()
                    fail = 0
                else:
                    time.sleep(0.01)
                continue
            fail = 0
            self._check_black(frame)
            with self._lock:
                self._frame = frame
                self.frames_read += 1

    def _check_black(self, frame):
        """Warn once if frames stay near-black (cam not ready / contention)."""
        if float(frame.mean()) < BLACK_FRAME_MEAN:
            self._black_streak += 1
            if self._black_streak >= 30 and not self._black_warned:
                self._black_warned = True
                print(f"WARNING: camera {self.device} returning black frames — "
                      f"detection will be wrong. Check the lens cap, that no "
                      f"other process holds the camera, and the device index.")
        else:
            self._black_streak = 0
            self._black_warned = False

    def read_latest(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap:
            self._cap.release()
        if self._lock_fd:
            self._lock_fd.close()  # releases the flock
            self._lock_fd = None
