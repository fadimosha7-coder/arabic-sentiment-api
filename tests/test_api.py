import pytest
from fastapi.testclient import TestClient

from sentiment_api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:        # بيشغّل lifespan ويحمّل الموديل
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_ok(client):
    r = client.post("/predict", json={"texts": ["المبارة كانت روعة", "الوضع صار صعب"]})
    assert r.status_code == 200
    body = r.json()
    assert [x["label"] for x in body["results"]] == ["pos", "neg"]
    assert body["latency_ms"] > 0


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