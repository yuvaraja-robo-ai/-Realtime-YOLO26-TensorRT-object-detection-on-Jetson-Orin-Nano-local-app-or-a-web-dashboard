"""Thread-safe runtime state shared between web layer and inference worker."""
import threading
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Camera defaults (override via env in main if desired).
CAM_DEVICE = 0
CAM_WIDTH = 1280
CAM_HEIGHT = 720

# Inference defaults.
# imgsz matches the exported TensorRT engine (re-export to change — see
# scripts/export_engine.py). Higher = better small-object detection, lower FPS.
DEFAULT_IMGSZ = 1280
DEFAULT_CONF = 0.15  # lower recovers small/low-confidence objects (more recall)

# Base (pretrained) model and custom (fine-tuned) model basenames in MODELS_DIR.
# yolo26s (small) > yolo26n (nano) on accuracy; at imgsz 1280 it's memory-bound
# so costs ~no FPS on the Orin. Re-export if you change this (export_engine.py).
BASE_MODEL = "yolo26s"
CUSTOM_MODEL = "custom"

# Dataset + training (on-device fine-tune).
DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"
RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
TRAIN_EPOCHS = 50
TRAIN_IMGSZ = 640
TRAIN_BATCH = 4
SUGGEST_CONF = 0.10  # low conf so auto-assist offers generic relabel candidates


class AppState:
    """Mutable settings the UI can change live, guarded by a lock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.conf = DEFAULT_CONF
        self.classes = None          # None => all COCO classes; else list[int]
        self.detect_enabled = True
        self.preprocess_clahe = False  # CLAHE for dim/uneven lighting
        self.track = True              # ByteTrack temporal smoothing (anti-flicker)
        self.model_name = "unknown"  # set by detector on load

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "conf": self.conf,
                "classes": list(self.classes) if self.classes else None,
                "detect_enabled": self.detect_enabled,
                "preprocess_clahe": self.preprocess_clahe,
                "track": self.track,
                "model_name": self.model_name,
            }

    def update(self, conf=None, classes=None, detect_enabled=None,
               preprocess_clahe=None, track=None) -> None:
        with self._lock:
            if conf is not None:
                self.conf = max(0.01, min(0.99, float(conf)))
            if classes is not None:
                # Empty list / None both mean "all classes".
                self.classes = list(classes) if classes else None
            if detect_enabled is not None:
                self.detect_enabled = bool(detect_enabled)
            if preprocess_clahe is not None:
                self.preprocess_clahe = bool(preprocess_clahe)
            if track is not None:
                self.track = bool(track)

    def set_model_name(self, name: str) -> None:
        with self._lock:
            self.model_name = name


# Single shared instance.
state = AppState()
