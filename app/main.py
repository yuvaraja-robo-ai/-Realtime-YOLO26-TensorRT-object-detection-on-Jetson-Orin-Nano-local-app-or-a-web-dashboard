"""FastAPI app: dashboard, MJPEG stream, WebSocket stats, config + snapshot."""
import asyncio
import time
from pathlib import Path

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .camera import BLACK_FRAME_MEAN, CaptureThread
from .config import (BASE_MODEL, CAM_DEVICE, CAM_HEIGHT, CAM_WIDTH,
                     CUSTOM_MODEL, MODELS_DIR, TRAIN_BATCH, TRAIN_EPOCHS,
                     TRAIN_IMGSZ, state)
from .dataset import DatasetManager
from .deepscan import DeepScanner
from .detector import Detector
from .pipeline import InferenceWorker
from .stats import TegraStats
from .trainer import Trainer

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
SNAP_DIR = Path(__file__).resolve().parent.parent / "snapshots"

app = FastAPI(title="Orin YOLO Detector")

camera: CaptureThread = None
worker: InferenceWorker = None
tegra: TegraStats = None
dataset: DatasetManager = None
trainer: Trainer = None
deepscanner: DeepScanner = None
COCO_NAMES: dict = {}
active_model: str = BASE_MODEL


def _swap_model(basename: str):
    """Hot-swap the inference worker's detector to a different model."""
    global active_model
    worker.set_detector(Detector(basename))
    active_model = basename


@app.on_event("startup")
def _startup():
    global camera, worker, tegra, dataset, trainer, deepscanner, COCO_NAMES, active_model
    camera = CaptureThread(CAM_DEVICE, CAM_WIDTH, CAM_HEIGHT)
    camera.start()
    detector = Detector(BASE_MODEL)
    COCO_NAMES = dict(detector.names)
    active_model = BASE_MODEL
    worker = InferenceWorker(camera, detector)
    worker.start()
    tegra = TegraStats()
    tegra.start()
    dataset = DatasetManager()
    trainer = Trainer(dataset, worker.pause, worker.resume, _swap_model)
    deepscanner = DeepScanner(active_model)


@app.on_event("shutdown")
def _shutdown():
    for c in (worker, tegra, camera):
        if c:
            c.stop()


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text()


@app.get("/classes")
def classes():
    """COCO id -> name map for the class-filter UI."""
    return {"names": COCO_NAMES}


def _mjpeg_generator():
    boundary = b"--frame"
    while True:
        worker.wait_for_frame(timeout=1.0)
        jpeg = worker.get_jpeg()
        if jpeg is None:
            time.sleep(0.05)
            continue
        yield (boundary + b"\r\n"
               b"Content-Type: image/jpeg\r\n"
               b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
               + jpeg + b"\r\n")


@app.get("/stream")
def stream():
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/snapshot")
def snapshot():
    jpeg = worker.get_jpeg()
    if jpeg is None:
        return Response(status_code=503, content="no frame yet")
    SNAP_DIR.mkdir(exist_ok=True)
    fname = SNAP_DIR / f"snap_{int(time.time())}.jpg"
    fname.write_bytes(jpeg)
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={"Content-Disposition": f'attachment; filename="{fname.name}"',
                 "X-Saved-Path": str(fname)},
    )


@app.post("/deepscan")
def deepscan():
    """SAHI sliced inference on the current frame — catches small objects the
    live single-pass stream misses. Heavy; on-demand only. Returns annotated
    JPEG with detection counts in the X-Counts header (JSON)."""
    import json as _json
    frame = camera.read_latest()
    if frame is None:
        return Response(status_code=503, content="no frame yet")
    if float(frame.mean()) < BLACK_FRAME_MEAN:
        return Response(status_code=409, content="frame is black")
    conf = state.snapshot()["conf"]
    annotated, counts, n = deepscanner.scan(frame, conf=conf)
    ok, buf = cv2.imencode(".jpg", annotated)
    if not ok:
        return Response(status_code=500, content="encode failed")
    return Response(
        content=buf.tobytes(),
        media_type="image/jpeg",
        headers={"X-Counts": _json.dumps(counts), "X-Det-Count": str(n)},
    )


class ConfigIn(BaseModel):
    conf: float | None = None
    classes: list[int] | None = None
    detect_enabled: bool | None = None
    preprocess_clahe: bool | None = None
    track: bool | None = None


@app.post("/config")
def set_config(cfg: ConfigIn):
    state.update(conf=cfg.conf, classes=cfg.classes,
                 detect_enabled=cfg.detect_enabled,
                 preprocess_clahe=cfg.preprocess_clahe, track=cfg.track)
    return state.snapshot()


# ---- training: capture / label / dataset / train ------------------------

class BoxIn(BaseModel):
    cls_name: str
    cx: float
    cy: float
    w: float
    h: float


class LabelIn(BaseModel):
    id: str
    boxes: list[BoxIn]


class ClassIn(BaseModel):
    name: str


class TrainIn(BaseModel):
    epochs: int = TRAIN_EPOCHS
    imgsz: int = TRAIN_IMGSZ
    batch: int = TRAIN_BATCH


class ModelIn(BaseModel):
    model: str  # "base" or "custom"


@app.post("/capture")
def capture():
    """Save the current RAW (un-annotated) camera frame for training."""
    frame = camera.read_latest()
    if frame is None:
        return Response(status_code=503, content="no frame yet")
    if float(frame.mean()) < BLACK_FRAME_MEAN:
        return Response(status_code=409,
                        content="frame is black — camera not ready or in use "
                                "elsewhere; not saving to dataset")
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        return Response(status_code=500, content="encode failed")
    return {"id": dataset.save_capture(buf.tobytes())}


@app.get("/captures")
def captures():
    return {"captures": dataset.list_captures(), "classes": dataset.classes()}


@app.get("/capture/{cap_id}")
def capture_image(cap_id: str):
    p = dataset.image_path(cap_id)
    if not p.exists():
        return Response(status_code=404)
    return Response(content=p.read_bytes(), media_type="image/jpeg")


@app.post("/label/suggest")
def label_suggest(body: LabelIn):
    """Auto-assist: return low-conf candidate boxes for a captured frame."""
    p = dataset.image_path(body.id)
    if not p.exists():
        return Response(status_code=404)
    frame = cv2.imread(str(p))
    return {"boxes": worker.detector.suggest(frame)}


@app.post("/label")
def save_label(body: LabelIn):
    return dataset.save_label(body.id, [b.model_dump() for b in body.boxes])


@app.post("/class")
def add_class(body: ClassIn):
    idx = dataset.add_class(body.name)
    return {"index": idx, "classes": dataset.classes()}


@app.get("/dataset/stats")
def dataset_stats():
    return dataset.stats()


@app.post("/train")
def train(body: TrainIn):
    if trainer.is_running():
        return {"started": False, "reason": "already training"}
    started = trainer.start(epochs=body.epochs, imgsz=body.imgsz, batch=body.batch)
    return {"started": started}


@app.get("/train/status")
def train_status():
    return trainer.status()


@app.post("/model/select")
def model_select(body: ModelIn):
    basename = CUSTOM_MODEL if body.model == "custom" else BASE_MODEL
    if body.model == "custom" and not (MODELS_DIR / f"{CUSTOM_MODEL}.engine").exists() \
            and not (MODELS_DIR / f"{CUSTOM_MODEL}.pt").exists():
        return Response(status_code=409, content="no custom model trained yet")
    _swap_model(basename)
    return {"active_model": active_model}


@app.websocket("/ws")
async def ws_stats(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            payload = worker.stats()
            payload.update(tegra.snapshot())
            snap = state.snapshot()
            payload["model_name"] = snap["model_name"]
            payload["detect_enabled"] = snap["detect_enabled"]
            payload["preprocess_clahe"] = snap["preprocess_clahe"]
            payload["track"] = snap["track"]
            payload["active_model"] = active_model
            payload["training"] = trainer.status()
            await ws.send_json(payload)
            await asyncio.sleep(0.2)  # ~5 Hz
    except WebSocketDisconnect:
        return


# Serve static assets (app.js, style.css) at /static.
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
