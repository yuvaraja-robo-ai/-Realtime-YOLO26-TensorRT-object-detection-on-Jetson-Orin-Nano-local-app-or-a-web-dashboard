"""YOLO detector: loads TensorRT engine (or .pt fallback) and annotates frames."""
from pathlib import Path

import cv2
from ultralytics import YOLO

from .config import DEFAULT_IMGSZ, MODELS_DIR, SUGGEST_CONF, state


def apply_clahe(frame):
    """Contrast-limited adaptive histogram equalization on the L channel.

    Evens out exposure in dim/uneven/backlit scenes so the model sees the
    object instead of shadow — applied to luminance only so colors are kept.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


class Detector:
    def __init__(self, model_basename="yolo26n", imgsz=DEFAULT_IMGSZ):
        self.imgsz = imgsz
        engine = MODELS_DIR / f"{model_basename}.engine"
        pt = MODELS_DIR / f"{model_basename}.pt"

        if engine.exists():
            path, kind = engine, "TensorRT"
        elif pt.exists():
            path, kind = pt, "PyTorch"
        else:
            # Let ultralytics download the .pt by bare name.
            path, kind = Path(f"{model_basename}.pt"), "PyTorch(download)"

        # TensorRT engines are built for a fixed input size — don't override
        # imgsz on predict or it asserts. .pt models accept any imgsz.
        self.is_engine = (kind == "TensorRT")
        self.model = YOLO(str(path), task="detect")
        self.names = self.model.names
        self._track_failed = False  # fall back to predict if tracking errors
        label = f"{model_basename} [{kind}]"
        state.set_model_name(label)
        print(f"detector loaded: {label} from {path}")

    def _imgsz_kw(self):
        return {} if self.is_engine else {"imgsz": self.imgsz}

    def infer(self, frame):
        """Run detection on a BGR frame.

        Returns (annotated_frame, counts_dict) where counts_dict maps
        class name -> number of detections.
        """
        cfg = state.snapshot()
        if cfg["preprocess_clahe"]:
            frame = apply_clahe(frame)

        kw = dict(conf=cfg["conf"], classes=cfg["classes"], verbose=False,
                  **self._imgsz_kw())
        # Tracking (ByteTrack, persist across frames) smooths detections and
        # kills the in/out flicker of bare per-frame predict.
        if cfg["track"] and not self._track_failed:
            try:
                results = self.model.track(frame, persist=True,
                                           tracker="bytetrack.yaml", **kw)
            except Exception as e:  # noqa: BLE001 - fall back, don't kill the loop
                print(f"tracking unavailable, using predict: {type(e).__name__}: {e}")
                self._track_failed = True
                results = self.model.predict(frame, **kw)
        else:
            results = self.model.predict(frame, **kw)
        r = results[0]
        annotated = r.plot()  # BGR ndarray with boxes + labels

        counts = {}
        if r.boxes is not None:
            for c in r.boxes.cls.tolist():
                name = self.names[int(c)]
                counts[name] = counts.get(name, 0) + 1
        return annotated, counts

    def suggest(self, frame):
        """Auto-assist: low-conf pass returning normalized candidate boxes.

        Returns [{cls_name, cx, cy, w, h, conf}] (xywh normalized 0..1).
        Low conf surfaces generic candidates for objects the model is unsure
        about, including ones the user will relabel as a new class.
        """
        h, w = frame.shape[:2]
        r = self.model.predict(frame, conf=SUGGEST_CONF, verbose=False,
                               **self._imgsz_kw())[0]
        out = []
        if r.boxes is None:
            return out
        for box, cls, conf in zip(r.boxes.xywh.tolist(),
                                  r.boxes.cls.tolist(),
                                  r.boxes.conf.tolist()):
            cx, cy, bw, bh = box
            out.append({
                "cls_name": self.names[int(cls)],
                "cx": cx / w, "cy": cy / h, "w": bw / w, "h": bh / h,
                "conf": round(float(conf), 3),
            })
        return out
