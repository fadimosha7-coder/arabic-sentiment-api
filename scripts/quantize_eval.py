import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch
from sentiment_api.model import SentimentModel

MODEL_DIR = ROOT / "models" / "sentiment_model"
va = pd.read_csv(ROOT / "data" / "arsentd_split.csv")
va = va[va.split == "val"]
texts, labels = list(va["Tweet"]), list(va["label"])
print(f"val tweets: {len(texts)} | torch {torch.__version__} | threads {torch.get_num_threads()} | engine {torch.backends.quantized.engine}")


def size_mb(model):
    tmp = ROOT / "tmp_state.pt"
    torch.save(model.state_dict(), tmp)
    mb = tmp.stat().st_size / 1e6
    tmp.unlink()
    return mb


def param_breakdown(model):
    lin = sum(p.numel() for m in model.modules() if isinstance(m, torch.nn.Linear) for p in m.parameters())
    tot = sum(p.numel() for p in model.parameters())
    return tot, lin, tot - lin


def bench(svc, n_single=20, n_batch=3):
    svc.predict(texts[:1])                                   # تسخين
    t0 = time.perf_counter()
    for i in range(n_single):
        svc.predict([texts[i]])
    single = (time.perf_counter() - t0) / n_single * 1000
    t0 = time.perf_counter()
    for _ in range(n_batch):
        svc.predict(texts[:16], batch_size=16)
    b16 = (time.perf_counter() - t0) / n_batch * 1000
    return single, b16


def evaluate(svc):
    res = svc.predict(texts, batch_size=32)
    preds = [r["label"] for r in res]
    acc = float(np.mean([p == l for p, l in zip(preds, labels)]))
    return acc, preds, np.array([r["confidence"] for r in res])


fp = SentimentModel(MODEL_DIR)
tot, lin, other = param_breakdown(fp.model)
print(f"params: total {tot / 1e6:.1f}M | Linear {lin / 1e6:.1f}M | other (embeddings, norms...) {other / 1e6:.1f}M")

q = SentimentModel(MODEL_DIR)
q.model = torch.ao.quantization.quantize_dynamic(q.model, {torch.nn.Linear}, dtype=torch.qint8)

rows = {}
for name, svc in [("fp32", fp), ("int8", q)]:
    print(f"measuring {name} ...")
    s, b = bench(svc)
    acc, preds, conf = evaluate(svc)
    rows[name] = dict(size=size_mb(svc.model), single=s, b16=b, acc=acc, preds=preds, conf=conf)

print()
for name, r in rows.items():
    print(f"{name}: size {r['size']:.0f} MB | single {r['single']:.0f} ms | 16 texts {r['b16']:.0f} ms | val acc {r['acc']:.1%}")

changed = sum(a != b for a, b in zip(rows["fp32"]["preds"], rows["int8"]["preds"]))
dconf = np.abs(rows["fp32"]["conf"] - rows["int8"]["conf"]).mean()
print(f"labels changed: {changed} of {len(texts)} | mean |confidence diff| = {dconf:.3f}")

drop = rows["fp32"]["acc"] - rows["int8"]["acc"]
ratio = rows["int8"]["single"] / rows["fp32"]["single"]
print(f"accuracy drop = {drop * 100:+.1f} points | single-text time ratio (int8/fp32) = {ratio:.2f}")
print("Rule (drop <= 1.0 point AND time reduced by >= 25%):", "ACCEPT" if drop <= 0.01 and ratio <= 0.75 else "REJECT")

import mlflow
mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
mlflow.set_experiment("arabic-sentiment")
for name, r in rows.items():
    with mlflow.start_run(run_name=f"serving_{name}"):
        mlflow.set_tags({"provenance": "measured now (local CPU)", "split": "val",
                         "model": "UBC-NLP/MARBERTv2 seed42", "precision": name})
        mlflow.log_params({"threads": torch.get_num_threads(), "torch": torch.__version__})
        mlflow.log_metrics({"val_acc": r["acc"], "size_mb": r["size"],
                            "single_ms": r["single"], "b16_ms": r["b16"]})