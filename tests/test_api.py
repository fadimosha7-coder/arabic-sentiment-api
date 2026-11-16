import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from sentiment_api.main import app


MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).resolve().parents[1] / "models" / "sentiment_model"))
needs_model = pytest.mark.skipif(not MODEL_DIR.exists(), reason="model folder not found")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:        # بيشغّل lifespan ويحمّل الموديل
        yield c

@needs_model
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@needs_model
def test_predict_ok(client):
    r = client.post("/predict", json={"texts": ["المبارة كانت روعة", "الوضع صار صعب"]})
    assert r.status_code == 200
    body = r.json()
    assert [x["label"] for x in body["results"]] == ["pos", "neg"]
    assert body["latency_ms"] > 0


@needs_model
@pytest.mark.parametrize("payload", [
    {"texts": []},
    {"texts": [""]},
    {"texts": ["   "]},
    {"texts": ["@user https://t.co/abc"]},
    {"texts": ["x"] * 17},
    {"texts": ["x" * 1001]},
    {"wrong_key": ["hi"]},
    {"texts": "not a list"},
])
def test_invalid_inputs_rejected(client, payload):
    assert client.post("/predict", json=payload).status_code == 422