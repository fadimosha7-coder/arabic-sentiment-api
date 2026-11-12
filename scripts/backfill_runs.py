from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parents[1]
mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
mlflow.set_experiment("arabic-sentiment")
TAG = "backfilled from my notes (not re-run, not verified by code)"

RUNS = []


def add(name, model, split, seed, acc, f1=None, **extra):
    RUNS.append((name, model, split, seed, acc, f1, extra))


hp = dict(epochs=4, lr=2e-5, batch=16, max_len=96)
add("majority", "majority", "val", None, 0.595, 0.373)
for seed, a, f in [(42, .896, .892), (43, .914, .910), (44, .905, .901)]:
    add("marbertv2", "UBC-NLP/MARBERTv2", "val", seed, a, f, **hp)
for seed, a, f in [(42, .896, .891), (43, .890, .885), (44, .908, .904)]:
    add("marbert", "UBC-NLP/MARBERT", "val", seed, a, f, **hp)
for seed, a in [(42, .906), (43, .937), (44, .898)]:
    add("topic_shift_no_politics", "UBC-NLP/MARBERTv2", "val_politics", seed, a, train_set="no politics (947)")
for seed, a in [(42, .937), (43, .929), (44, .937)]:
    add("topic_shift_control", "UBC-NLP/MARBERTv2", "val_politics", seed, a, train_set="random same size (947)")
for seed, a, f in [(42, .866, .861), (43, .891, .885), (44, .884, .879)]:
    add("marbertv2_final", "UBC-NLP/MARBERTv2", "test", seed, a, f, **hp)
for seed, a, f in [(42, .913, .913), (43, .903, .903), (44, .916, .916)]:
    add("marbertv2_final", "UBC-NLP/MARBERTv2", "ajgt", seed, a, f, **hp)
add("tfidf_final", "tfidf+logreg", "test", None, .745, .735)
add("tfidf_final", "tfidf+logreg", "ajgt", None, .748, .748)

for name, model, split, seed, acc, f1, extra in RUNS:
    run_name = f"{name}_{split}" + (f"_s{seed}" if seed is not None else "")
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({"provenance": TAG, "split": split, "model": model})
        params = dict(extra)
        if seed is not None:
            params["seed"] = seed
        if params:
            mlflow.log_params(params)
        mlflow.log_metric("acc", acc)
        if f1 is not None:
            mlflow.log_metric("macro_f1", f1)
print(len(RUNS), "runs logged")      # 21