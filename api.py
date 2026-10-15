"""FastAPI inference server.

Usage:
    uvicorn api:app --reload

Environment variables:
    MODEL_PATH  Path to a saved Keras model (default: saved_models/custom_cnn.keras)
    MODEL_TYPE  'fashion_mnist' or 'flowers'  (default: fashion_mnist)
"""
import io
import os
from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from src.data.fashion_mnist import CLASS_NAMES as FASHION_NAMES

_model = None
_class_names: list[str] = []
_norm_mean: np.ndarray | None = None
_norm_std: np.ndarray | None = None


def _load_norm_stats(model_path: str):
    """Load per-pixel normalization stats saved alongside the model, if present."""
    stats_path = model_path.replace(".keras", "_norm_stats.npz")
    if os.path.exists(stats_path):
        data = np.load(stats_path)
        return data["X_mean"], data["X_std"]
    return None, None


@asynccontextmanager
async def _lifespan(app: FastAPI):
    import tensorflow as tf

    global _model, _class_names, _norm_mean, _norm_std

    model_path = os.environ.get("MODEL_PATH", "saved_models/custom_cnn.keras")
    model_type = os.environ.get("MODEL_TYPE", "fashion_mnist")

    _model = tf.keras.models.load_model(model_path)
    _norm_mean, _norm_std = _load_norm_stats(model_path)

    if model_type == "fashion_mnist":
        _class_names = FASHION_NAMES
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {model_type!r}")

    yield


app = FastAPI(title="CNN Image Classifier", version="1.0", lifespan=_lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict")
async def predict(file: Annotated[UploadFile, File()]):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("L").resize((28, 28))
    arr = np.array(img, dtype="float32")[..., np.newaxis]

    if _norm_mean is not None and _norm_std is not None:
        arr = (arr - _norm_mean) / _norm_std
    else:
        arr = arr / 255.0

    arr = np.expand_dims(arr, axis=0)

    probs = _model.predict(arr, verbose=0)[0]
    top_idx = int(np.argmax(probs))

    return {
        "class": _class_names[top_idx],
        "confidence": round(float(probs[top_idx]), 4),
        "scores": {name: round(float(p), 4) for name, p in zip(_class_names, probs)},
    }
