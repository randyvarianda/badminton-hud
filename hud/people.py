"""Step 3: YOLO11m + ByteTrack on every frame, plus a torso colour sample per box."""
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

MODEL = Path(__file__).resolve().parent.parent / "models" / "yolo11m.pt"


def device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def torso_color(frame, box):
    """Shirt colour as a point in the HSV cone: (S cos H, S sin H, V), or None."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    crop = frame[int(y0 + 0.2 * h): int(y0 + 0.45 * h), int(x0 + 0.3 * w): int(x1 - 0.3 * w)]
    if crop.size < 30:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    hsv = hsv[hsv[:, 2] > 40]  # drop deep shadow
    if len(hsv) < 15:
        return None
    # Judge the shirt by its coloured pixels: white or black panels would make a red
    # jersey look pale (pink) or dark. Plain white/black/grey shirts use every pixel.
    chroma = hsv[hsv[:, 1] > 70]
    if len(chroma) > 0.2 * len(hsv):
        hsv = chroma
    return cone(hsv[:, 0] * 2, hsv[:, 1] / 255, hsv[:, 2] / 255).mean(0)


def cone(h_deg, s, v):
    h = np.radians(h_deg)
    return np.stack([s * np.cos(h), s * np.sin(h), v], -1)


def track(video: Path, progress):
    """Returns per-frame lists of dicts {id, box, conf, color}."""
    model = YOLO(str(MODEL))
    out = []
    stream = model.track(source=str(video), stream=True, imgsz=1280, classes=[0], conf=0.15,
                         tracker="bytetrack.yaml", device=device(), verbose=False, persist=True)
    for r in stream:
        dets = []
        b = r.boxes
        if b is not None and len(b):
            ids = b.id.int().tolist() if b.id is not None else [-1] * len(b)
            for tid, xyxy, c in zip(ids, b.xyxy.tolist(), b.conf.tolist()):
                dets.append({"id": tid, "box": xyxy, "conf": c,
                             "color": torso_color(r.orig_img, xyxy)})
        out.append(dets)
        progress(len(out))
    return out
