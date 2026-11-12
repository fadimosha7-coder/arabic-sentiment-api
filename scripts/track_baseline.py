import re
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline, make_union

ROOT = Path(__file__).resolve().parents[1]
mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
mlflow.set_experiment("arabic-sentiment")

t = pd.read_csv(ROOT / "data" / "arsentd_split.csv")
tr, va = t[t.split == "train"], t[t.split == "val"]


def clean(s):
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"@\w+", " ", s)
    s = re.sub(r"[\u064B-\u0652\u0640]", "", s)
    s = re.sub(r"[إأآ]", "ا", s)
    return re.sub(r"\s+", " ", s).strip()


vec = make_union(
    TfidfVectorizer(preprocessor=clean, ngram_range=(1, 2), min_df=2, sublinear_tf=True),
    TfidfVectorizer(preprocessor=clean, analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
)
clf = make_pipeline(vec, LogisticRegression(max_iter=2000, C=3.0, class_weight="balanced"))

with mlflow.start_run(run_name="tfidf_lr_baseline_val"):
    mlflow.set_tags({"provenance": "measured now (re-run locally)", "split": "val", "model": "tfidf+logreg"})
    mlflow.log_params({"word_ngrams": "1-2", "char_ngrams": "2-5", "min_df": 2, "C": 3.0, "class_weight": "balanced"})
    clf.fit(tr["Tweet"], tr["label"])
    pred = clf.predict(va["Tweet"])
    acc, f1 = accuracy_score(va["label"], pred), f1_score(va["label"], pred, average="macro")
    mlflow.log_metrics({"val_acc": acc, "val_macro_f1": f1, "n_val": len(va)})
    print(f"val acc {acc:.1%} | macro-F1 {f1:.3f}   (يوم 2: 74.5% | 0.736)")