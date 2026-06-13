"""On-device fine-tune: pause stream -> train yolo26n -> export engine -> hot-swap.

Single GPU, so training pauses live inference first and resumes after the new
custom model is exported and swapped in. Progress is exposed for the WebSocket.
"""
import shutil
import threading
import traceback
from pathlib import Path

from ultralytics import YOLO

from .config import (BASE_MODEL, CUSTOM_MODEL, MODELS_DIR, RUNS_DIR,
                     TRAIN_BATCH, TRAIN_EPOCHS, TRAIN_IMGSZ)


class Trainer:
    """Runs one fine-tune job at a time. Reports a status dict for the UI."""

    def __init__(self, dataset, pause_fn, resume_fn, swap_fn):
        self.dataset = dataset
        self.pause_fn = pause_fn      # () -> pause inference worker
        self.resume_fn = resume_fn    # () -> resume inference worker
        self.swap_fn = swap_fn        # (basename) -> hot-swap active model
        self._lock = threading.Lock()
        self._thread = None
        self._status = {"state": "idle", "epoch": 0, "total": 0, "msg": ""}

    def status(self) -> dict:
        with self._lock:
            return dict(self._status)

    def _set(self, **kw):
        with self._lock:
            self._status.update(kw)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, epochs=TRAIN_EPOCHS, imgsz=TRAIN_IMGSZ, batch=TRAIN_BATCH) -> bool:
        if self.is_running():
            return False
        self._thread = threading.Thread(
            target=self._run, args=(epochs, imgsz, batch), daemon=True
        )
        self._thread.start()
        return True

    def _run(self, epochs, imgsz, batch):
        try:
            self._set(state="preparing", epoch=0, total=epochs, msg="building dataset")
            data_yaml = self.dataset.build_data_yaml()

            self._set(state="preparing", msg="pausing live stream")
            self.pause_fn()

            base_pt = MODELS_DIR / f"{BASE_MODEL}.pt"
            model = YOLO(str(base_pt) if base_pt.exists() else f"{BASE_MODEL}.pt")

            def on_epoch_end(trainer):
                self._set(state="training", epoch=int(trainer.epoch) + 1,
                          total=epochs, msg="training")
            model.add_callback("on_train_epoch_end", on_epoch_end)

            self._set(state="training", msg="training")
            model.train(
                data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch,
                device=0, workers=2, cache=False, project=str(RUNS_DIR),
                name="custom", exist_ok=True, verbose=False, plots=False,
            )

            best = RUNS_DIR / "custom" / "weights" / "best.pt"
            if not best.exists():
                raise FileNotFoundError("training produced no best.pt")
            custom_pt = MODELS_DIR / f"{CUSTOM_MODEL}.pt"
            shutil.copy(str(best), str(custom_pt))

            self._set(state="exporting", msg="exporting TensorRT engine")
            eng = YOLO(str(custom_pt)).export(
                format="engine", half=True, imgsz=imgsz, device=0)
            target = MODELS_DIR / f"{CUSTOM_MODEL}.engine"
            if Path(eng).resolve() != target.resolve():
                shutil.move(str(eng), str(target))

            self._set(state="swapping", msg="loading custom model")
            self.swap_fn(CUSTOM_MODEL)

            self._set(state="done", msg="custom model live")
        except Exception as e:  # noqa: BLE001 - surface any failure to UI
            self._set(state="error", msg=f"{type(e).__name__}: {e}")
            traceback.print_exc()
        finally:
            self.resume_fn()
