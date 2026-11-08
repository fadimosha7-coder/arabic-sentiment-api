import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")


def clean_text(s: str) -> str:
    s = URL_RE.sub(" ", s)
    s = MENTION_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


class SentimentModel:
    def __init__(self, path, device="cpu"):
        path = Path(path)
        self.cfg = json.loads((path / "serving_config.json").read_text(encoding="utf-8"))
        self.tok = AutoTokenizer.from_pretrained(str(path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(path)).to(device).eval()
        self.device = device

    @torch.no_grad()
    def predict(self, texts, batch_size=32):
        out = []
        for i in range(0, len(texts), batch_size):
            batch = [clean_text(x) for x in texts[i:i + batch_size]]
            enc = self.tok(batch, truncation=True, max_length=self.cfg["max_len"],
                           padding=True, return_tensors="pt").to(self.device)
            probs = torch.softmax(self.model(**enc).logits, dim=-1).cpu()
            for p in probs:
                k = int(p.argmax())
                out.append({"label": self.cfg["labels"][str(k)], "confidence": round(float(p[k]), 4)})
        return out