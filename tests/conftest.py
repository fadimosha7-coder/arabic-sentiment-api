import os
from pathlib import Path

import pytest

from sentiment_api.model import SentimentModel

MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).resolve().parents[1] / "models" / "sentiment_model"))


@pytest.fixture(scope="session")
def svc():
    if not MODEL_DIR.exists():
        pytest.skip(f"model folder not found: {MODEL_DIR}")
    return SentimentModel(MODEL_DIR)