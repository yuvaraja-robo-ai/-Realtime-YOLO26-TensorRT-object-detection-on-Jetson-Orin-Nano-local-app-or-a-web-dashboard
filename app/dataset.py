"""Dataset manager: captures, YOLO-format labels, custom class list, data.yaml.

Layout (DATASET_DIR):
    images/<id>.jpg
    labels/<id>.txt       # YOLO: "<cls> <cx> <cy> <w> <h>" normalized, one per box
    classes.json          # {"names": ["wrench", "gizmo", ...]} index == class id
    data.yaml             # generated at train time
"""
import json
import threading
import time
from pathlib import Path

from .config import DATASET_DIR


class DatasetManager:
    def __init__(self, root: Path = DATASET_DIR):
        self.root = root
        self.images = root / "images"
        self.labels = root / "labels"
        self.classes_file = root / "classes.json"
        self._lock = threading.Lock()
        for d in (self.images, self.labels):
            d.mkdir(parents=True, exist_ok=True)
        if not self.classes_file.exists():
            self._write_classes([])

    # ---- classes ----------------------------------------------------------
    def _read_classes(self):
        return json.loads(self.classes_file.read_text()).get("names", [])

    def _write_classes(self, names):
        self.classes_file.write_text(json.dumps({"names": names}, indent=2))

    def classes(self):
        with self._lock:
            return self._read_classes()

    def add_class(self, name: str) -> int:
        name = name.strip()
        if not name:
            raise ValueError("empty class name")
        with self._lock:
            names = self._read_classes()
            if name not in names:
                names.append(name)
                self._write_classes(names)
            return names.index(name)

    # ---- captures ---------------------------------------------------------
    def save_capture(self, jpeg: bytes) -> str:
        cap_id = f"cap_{int(time.time() * 1000)}"
        (self.images / f"{cap_id}.jpg").write_bytes(jpeg)
        return cap_id

    def image_path(self, cap_id: str) -> Path:
        return self.images / f"{cap_id}.jpg"

    def list_captures(self):
        out = []
        for img in sorted(self.images.glob("*.jpg")):
            cap_id = img.stem
            lbl = self.labels / f"{cap_id}.txt"
            n = 0
            if lbl.exists():
                n = len([ln for ln in lbl.read_text().splitlines() if ln.strip()])
            out.append({"id": cap_id, "labeled": lbl.exists(), "boxes": n})
        return out

    def save_label(self, cap_id: str, boxes: list[dict]) -> dict:
        """boxes: [{cls_name, cx, cy, w, h}] all normalized 0..1.

        Creates any new class names on the fly. Empty list writes an empty
        label file (valid YOLO negative/background sample).
        """
        if not self.image_path(cap_id).exists():
            raise FileNotFoundError(cap_id)
        lines = []
        for b in boxes:
            cls = self.add_class(b["cls_name"])
            lines.append(
                f"{cls} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}"
            )
        (self.labels / f"{cap_id}.txt").write_text("\n".join(lines))
        return {"id": cap_id, "boxes": len(lines)}

    # ---- stats + training yaml -------------------------------------------
    def stats(self):
        names = self.classes()
        counts = {n: 0 for n in names}
        labeled = 0
        for lbl in self.labels.glob("*.txt"):
            rows = [ln for ln in lbl.read_text().splitlines() if ln.strip()]
            if rows:
                labeled += 1
            for r in rows:
                idx = int(r.split()[0])
                if 0 <= idx < len(names):
                    counts[names[idx]] += 1
        return {
            "num_images": len(list(self.images.glob("*.jpg"))),
            "num_labeled": labeled,
            "classes": [{"name": n, "count": counts[n]} for n in names],
        }

    def build_data_yaml(self) -> Path:
        """Write data.yaml. Small datasets reuse all images for train+val."""
        names = self._read_classes()
        if not names:
            raise ValueError("no classes labeled yet")
        yaml_path = self.root / "data.yaml"
        names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(names))
        yaml_path.write_text(
            f"path: {self.root.resolve()}\n"
            f"train: images\n"
            f"val: images\n"
            f"names:\n{names_block}\n"
        )
        return yaml_path
