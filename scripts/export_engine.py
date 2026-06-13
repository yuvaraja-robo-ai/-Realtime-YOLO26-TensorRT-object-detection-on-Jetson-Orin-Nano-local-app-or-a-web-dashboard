#!/usr/bin/env python3
"""One-time: export a YOLO .pt -> TensorRT FP16 engine for fast inference.

    python3 scripts/export_engine.py yolo26s 1280   # app default (app/config.py)
    python3 scripts/export_engine.py            # yolo26n, imgsz 1280
    python3 scripts/export_engine.py yolo26n 960   # smaller = more FPS

First run downloads the .pt weights. TRT build can take several minutes on Orin.
Output engine is placed in models/ next to the .pt.
"""
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "yolo26n"
    imgsz = int(sys.argv[2]) if len(sys.argv) > 2 else 1280
    MODELS_DIR.mkdir(exist_ok=True)

    pt_path = MODELS_DIR / f"{name}.pt"
    print(f"loading {name} (downloads to {pt_path} if missing)")
    model = YOLO(str(pt_path) if pt_path.exists() else f"{name}.pt")

    print(f"exporting TensorRT FP16 engine  imgsz={imgsz} ...")
    engine_path = model.export(format="engine", half=True, imgsz=imgsz, device=0)
    engine_path = Path(engine_path)

    # Ensure both .pt and .engine live in models/.
    target = MODELS_DIR / f"{name}.engine"
    if engine_path.resolve() != target.resolve():
        shutil.move(str(engine_path), str(target))
    src_pt = Path(f"{name}.pt")
    if src_pt.exists() and not pt_path.exists():
        shutil.move(str(src_pt), str(pt_path))

    print(f"OK: engine at {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
