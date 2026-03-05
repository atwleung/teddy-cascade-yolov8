#!/usr/bin/env python3
"""
stage2_api.py

Stage-2 YOLOv8 inference as an HTTP API (FastAPI).

- Loads YOLO model once at startup.
- Accepts a batch of ROI images as base64-encoded JPEG/PNG.
- Returns top-1 class per ROI (cls_name + conf).

Environment variables:
  MODEL_PATH   : path to best.pt (default: "best.pt")
  DEVICE       : ultralytics device string (default: "mps") e.g. "cpu", "mps", "0" (cuda:0)
  MAX_DET      : max detections per ROI (default: 10)
  API_KEY      : if set, require header "X-API-Key: <API_KEY>"

Run:
  pip install -r requirements.txt
  export MODEL_PATH=best.pt
  export DEVICE=mps
  uvicorn stage2_api:app --host 127.0.0.1 --port 8000

Later on GPU:
  export DEVICE=0
  uvicorn stage2_api:app --host 0.0.0.0 --port 8000
"""

import os
import base64
import io
from typing import List, Optional

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from ultralytics import YOLO


MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
DEVICE = os.getenv("DEVICE", "mps")
API_KEY = os.getenv("API_KEY", "")
DEFAULT_MAX_DET = int(os.getenv("MAX_DET", "10"))


app = FastAPI(title="Stage2 Teddy Fine API")


class PredictRequest(BaseModel):
    images_b64: List[str]          # base64 JPG/PNG
    conf: float = 0.25
    iou: float = 0.6
    imgsz: int = 320               # 320/416 are typical for ROI
    max_det: Optional[int] = None  # override DEFAULT_MAX_DET if provided


def _decode_b64_image(b64: str) -> np.ndarray:
    """
    Decode base64 image into HxWx3 RGB uint8 numpy array.
    """
    try:
        raw = base64.b64decode(b64)
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        return np.array(im)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}") from e


# Load model once
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    # Fail fast if model can't load; uvicorn will show this in logs.
    raise RuntimeError(f"Failed to load model at MODEL_PATH={MODEL_PATH}: {e}") from e


@app.get("/health")
def health():
    return {"ok": True, "device": DEVICE, "model_path": MODEL_PATH, "max_det": DEFAULT_MAX_DET}


@app.post("/predict")
def predict(req: PredictRequest, x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    # Optional API key auth (recommended if exposed beyond localhost)
    if API_KEY:
        if not x_api_key or x_api_key != API_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized")

    if not req.images_b64:
        raise HTTPException(status_code=400, detail="images_b64 is empty")

    imgs = [_decode_b64_image(s) for s in req.images_b64]
    max_det = int(req.max_det) if req.max_det is not None else DEFAULT_MAX_DET

    # Batch inference
    results = model.predict(
        imgs,
        device=DEVICE,
        conf=req.conf,
        iou=req.iou,
        imgsz=req.imgsz,
        max_det=max_det,
        verbose=False,
    )

    out = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            out.append({"top1": None})
            continue

        # top-1 by confidence
        confs = r.boxes.conf.cpu().numpy()
        j = int(np.argmax(confs))
        cls_id = int(r.boxes.cls[j].item())
        out.append(
            {
                "top1": {
                    "cls_id": cls_id,
                    "cls_name": str(r.names[cls_id]),
                    "conf": float(r.boxes.conf[j].item()),
                }
            }
        )

    return {"results": out}