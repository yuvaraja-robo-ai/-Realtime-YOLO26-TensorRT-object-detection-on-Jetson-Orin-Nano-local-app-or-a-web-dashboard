"""SAHI deep scan: slice the current frame into tiles, detect per tile, merge.

Standard inference resizes the whole 1280x720 frame down to the model input,
so small/distant objects shrink below what the model can resolve. SAHI
(Slicing Aided Hyper Inference) instead runs detection on full-resolution
tiles and stitches the results, catching the small stuff a single pass misses.

This is heavy (several inferences per frame) so it is on-demand only — a
"deep scan" of one still frame, never the live 30 fps stream.
"""
import threading

import cv2

from .config import BASE_MODEL, MODELS_DIR

# Tile geometry. 640 tiles at full camera resolution + 20% overlap so objects
# straddling a slice boundary still land whole inside at least one tile.
SLICE = 640
OVERLAP = 0.2


class DeepScanner:
    def __init__(self, model_basename: str = BASE_MODEL, device: str = "cuda:0"):
        self.model_basename = model_basename
        self.device = device
        self._model = None
        self._lock = threading.Lock()

    def _ensure_model(self):
        """Load the SAHI-wrapped detector lazily (first scan only)."""
        if self._model is not None:
            return
        from sahi import AutoDetectionModel
        # SAHI runs on the .pt (TensorRT engines are fixed-size, no slicing).
        pt = MODELS_DIR / f"{self.model_basename}.pt"
        path = str(pt) if pt.exists() else f"{self.model_basename}.pt"
        last = None
        for mtype in ("ultralytics", "yolov8"):  # name changed across SAHI versions
            try:
                self._model = AutoDetectionModel.from_pretrained(
                    model_type=mtype, model_path=path, device=self.device,
                )
                return
            except Exception as e:  # noqa: BLE001 - try the next type string
                last = e
        raise RuntimeError(f"could not load SAHI model: {last}")

    def scan(self, frame, conf: float = 0.2):
        """Run sliced inference on a BGR frame.

        Returns (annotated_bgr, counts_dict, num_dets).
        """
        from sahi.predict import get_sliced_prediction
        with self._lock:
            self._ensure_model()
            self._model.confidence_threshold = conf
            # SAHI wants RGB.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = get_sliced_prediction(
                rgb, self._model,
                slice_height=SLICE, slice_width=SLICE,
                overlap_height_ratio=OVERLAP, overlap_width_ratio=OVERLAP,
                verbose=0,
            )

        annotated = frame.copy()
        counts = {}
        preds = result.object_prediction_list
        for p in preds:
            name = p.category.name
            counts[name] = counts.get(name, 0) + 1
            box = p.bbox
            x1, y1, x2, y2 = (int(box.minx), int(box.miny),
                              int(box.maxx), int(box.maxy))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (40, 200, 70), 2)
            label = f"{name} {p.score.value:.2f}"
            cv2.putText(annotated, label, (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 200, 70), 1,
                        cv2.LINE_AA)
        return annotated, counts, len(preds)
