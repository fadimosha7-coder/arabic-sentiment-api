import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from .model import SentimentModel
from .schemas import PredictRequest, PredictResponse

MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).resolve().parents[2] / "models" / "sentiment_model"))
state = {}


@asynccontextmanager
async def lifespan(app):
    if MODEL_DIR.exists():
        state["model"] = SentimentModel(MODEL_DIR)     # مرة وحدة عند التشغيل
    yield
    state.clear()


app = FastAPI(title="Arabic sentiment API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok" if "model" in state else "model_not_loaded"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model = state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    t0 = time.perf_counter()
    results = model.predict(req.texts)
    return {"results": results, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}