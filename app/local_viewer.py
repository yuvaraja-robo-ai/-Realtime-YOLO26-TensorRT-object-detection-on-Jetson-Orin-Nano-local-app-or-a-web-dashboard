"""Native local viewer — Tkinter window, no web server.

Lowest-latency path on the Orin itself: camera -> inference worker (annotated
ndarray) -> Tk canvas. Skips JPEG encode + HTTP + browser entirely, so the
local stream has none of the MJPEG/network lag. Controls (confidence, detect
toggle, model switch, snapshot) reuse the same shared `state` as the web app.

Run:  python3 -m app.local_viewer        (or ./run.sh local)
"""
import argparse
import time
import tkinter as tk
from pathlib import Path

import cv2
from PIL import Image, ImageTk

from .camera import CaptureThread
from .config import (BASE_MODEL, CAM_DEVICE, CAM_HEIGHT, CAM_WIDTH,
                     CUSTOM_MODEL, DEFAULT_CONF, MODELS_DIR, state)
from .detector import Detector
from .pipeline import InferenceWorker
from .stats import TegraStats

SNAP_DIR = Path(__file__).resolve().parent.parent / "snapshots"
REFRESH_MS = 15  # UI poll interval (~66 Hz cap); worker rate is the real limit


def _custom_exists() -> bool:
    return ((MODELS_DIR / f"{CUSTOM_MODEL}.engine").exists()
            or (MODELS_DIR / f"{CUSTOM_MODEL}.pt").exists())


class LocalViewer:
    def __init__(self, device, width, height):
        self.camera = CaptureThread(device, width, height)
        self.camera.start()
        # encode_jpeg=False: native viewer reads the ndarray, no wasted JPEG.
        self.worker = InferenceWorker(self.camera, Detector(BASE_MODEL),
                                      encode_jpeg=False)
        self.worker.start()
        self.tegra = TegraStats()
        self.tegra.start()
        self.active_model = BASE_MODEL
        self._imgtk = None  # keep a ref so Tk doesn't GC the image

        self._build_ui()

    # ---- UI -------------------------------------------------------------
    def _build_ui(self):
        self.root = tk.Tk()
        self.root.title("Orin YOLO — Local Viewer")
        self.root.configure(bg="#111")
        self.root.protocol("WM_DELETE_WINDOW", self._quit)
        self.root.bind("<q>", lambda e: self._quit())

        self.video = tk.Label(self.root, bg="#000")
        self.video.pack(fill=tk.BOTH, expand=True)

        bar = tk.Frame(self.root, bg="#111")
        bar.pack(fill=tk.X, side=tk.BOTTOM)

        self.status = tk.Label(bar, bg="#111", fg="#0f0", anchor="w",
                               font=("monospace", 11))
        self.status.pack(side=tk.LEFT, padx=8, pady=4)

        tk.Button(bar, text="Snapshot", command=self._snapshot).pack(
            side=tk.RIGHT, padx=4)

        self.detect_var = tk.IntVar(value=1)
        tk.Checkbutton(bar, text="Detect", variable=self.detect_var, bg="#111",
                       fg="#ddd", selectcolor="#333", activebackground="#111",
                       command=self._toggle_detect).pack(side=tk.RIGHT, padx=4)

        self.track_var = tk.IntVar(value=1)
        tk.Checkbutton(bar, text="Track", variable=self.track_var, bg="#111",
                       fg="#ddd", selectcolor="#333", activebackground="#111",
                       command=lambda: state.update(track=bool(self.track_var.get()))
                       ).pack(side=tk.RIGHT, padx=4)

        self.clahe_var = tk.IntVar(value=0)
        tk.Checkbutton(bar, text="CLAHE", variable=self.clahe_var, bg="#111",
                       fg="#ddd", selectcolor="#333", activebackground="#111",
                       command=lambda: state.update(preprocess_clahe=bool(self.clahe_var.get()))
                       ).pack(side=tk.RIGHT, padx=4)

        self.model_var = tk.StringVar(value="base")
        tk.Radiobutton(bar, text="Base", variable=self.model_var, value="base",
                       bg="#111", fg="#ddd", selectcolor="#333",
                       command=self._switch_model).pack(side=tk.RIGHT)
        rb = tk.Radiobutton(bar, text="Custom", variable=self.model_var,
                            value="custom", bg="#111", fg="#ddd",
                            selectcolor="#333", command=self._switch_model)
        rb.pack(side=tk.RIGHT)
        if not _custom_exists():
            rb.configure(state=tk.DISABLED)

        tk.Label(bar, text="conf", bg="#111", fg="#ddd").pack(
            side=tk.RIGHT, padx=(8, 0))
        self.conf_var = tk.DoubleVar(value=DEFAULT_CONF)
        tk.Scale(bar, from_=0.01, to=0.99, resolution=0.01,
                 orient=tk.HORIZONTAL, variable=self.conf_var, length=140,
                 bg="#111", fg="#ddd", troughcolor="#333", highlightthickness=0,
                 command=self._set_conf).pack(side=tk.RIGHT)

    # ---- control callbacks ---------------------------------------------
    def _set_conf(self, _val):
        state.update(conf=self.conf_var.get())

    def _toggle_detect(self):
        state.update(detect_enabled=bool(self.detect_var.get()))

    def _switch_model(self):
        name = CUSTOM_MODEL if self.model_var.get() == "custom" else BASE_MODEL
        if name == self.active_model:
            return
        self.worker.set_detector(Detector(name))
        self.active_model = name

    def _snapshot(self):
        frame = self.worker.get_annotated()
        if frame is None:
            return
        SNAP_DIR.mkdir(exist_ok=True)
        path = SNAP_DIR / f"snap_{int(time.time())}.jpg"
        cv2.imwrite(str(path), frame)
        print(f"saved {path}")

    # ---- render loop ----------------------------------------------------
    def _tick(self):
        frame = self.worker.get_annotated()
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.video.configure(image=self._imgtk)
            self._update_status()
        self.root.after(REFRESH_MS, self._tick)

    def _update_status(self):
        s = self.worker.stats()
        t = self.tegra.snapshot()
        counts = " ".join(f"{k}:{v}" for k, v in sorted(s["counts"].items()))
        parts = [f"{s['fps']:.1f} FPS", f"model={self.active_model}"]
        if t.get("gpu") is not None:
            parts.append(f"GPU {t['gpu']}%")
        if t.get("temp") is not None:
            parts.append(f"{t['temp']:.0f}C")
        if counts:
            parts.append("| " + counts)
        self.status.configure(text="  ".join(parts))

    # ---- lifecycle ------------------------------------------------------
    def run(self):
        self.root.after(REFRESH_MS, self._tick)
        self.root.mainloop()

    def _quit(self):
        for c in (self.worker, self.tegra, self.camera):
            c.stop()
        self.root.destroy()


def main():
    ap = argparse.ArgumentParser(description="Orin YOLO native local viewer")
    ap.add_argument("--device", type=int, default=CAM_DEVICE)
    ap.add_argument("--width", type=int, default=CAM_WIDTH)
    ap.add_argument("--height", type=int, default=CAM_HEIGHT)
    args = ap.parse_args()
    LocalViewer(args.device, args.width, args.height).run()


if __name__ == "__main__":
    main()
