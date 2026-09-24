"""
Train, evaluate and serve the flow classifier.

Model choice
------------
Tree ensembles over hand-crafted metadata features, not deep learning:
* the features are heavy-tailed and on wildly different scales (bytes vs
  microseconds); trees do not care, so no scaling pipeline is needed;
* a few hundred shallow trees evaluate in microseconds per flow and fit in a few
  hundred KB - the budget of a home-router CPU;
* feature importances tell us *why* a flow was classified, which matters when
  the output is a QoS decision the user might question.

Evaluation protocol
-------------------
Windows from one flow are near-duplicates, so a random train/test split leaks
and reports inflated accuracy. All splits here are by *capture session*
(StratifiedGroupKFold on `session`), and a fixed set of HELD_OUT_SESSIONS is
excluded from training entirely to serve as a final test set and as the source
of the demo pcap.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from .feature_extraction import FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
DEFAULT_MODEL = MODELS_DIR / "metaqos_classifier.joblib"

# One or two sessions per class, never used for training. Chosen for volume
# (so the held-out metrics are not computed on a handful of windows) and, where
# possible, for a different application than the bulk of the training data so
# the test also probes generalisation across apps of the same class.
HELD_OUT_SESSIONS = [
    "facebook_video2",   # video_call  (training keeps facebook_video1, hangouts_video1, skype_video1/2)
    "hangouts_audio3",   # voip_audio
    "skype_audio2",      # voip_audio
    "netflix2",          # streaming
    "youtube4",          # streaming
    "vimeo3",            # streaming
    "sftp_down_3",       # file_transfer
    "scpDown3",          # file_transfer
    "skype_file3",       # file_transfer
    "facebook_chat_4",   # chat
    "gmailchat2",        # chat
    "email2",            # email
]

MAX_WINDOWS_PER_FLOW = 300


def make_models(seed: int = 0) -> dict[str, object]:
    return {
        "decision_tree": DecisionTreeClassifier(max_depth=10, min_samples_leaf=20, class_weight="balanced", random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=14, min_samples_leaf=5, max_features=0.5,
                                                class_weight="balanced_subsample", n_jobs=-1, random_state=seed),
        "hist_gbt": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                                   l2_regularization=1.0, random_state=seed),
    }


# ---------------------------------------------------------------------------
# data preparation
# ---------------------------------------------------------------------------


def cap_windows_per_flow(df: pd.DataFrame, max_per_flow: int = MAX_WINDOWS_PER_FLOW) -> pd.DataFrame:
    """Keep at most `max_per_flow` windows per flow, evenly spaced in time, so a
    20-minute call does not outvote every chat session in the training set."""
    def _pick(g: pd.DataFrame) -> pd.DataFrame:
        if len(g) <= max_per_flow:
            return g
        idx = np.linspace(0, len(g) - 1, max_per_flow).round().astype(int)
        return g.sort_values("window_idx").iloc[idx]
    return df.groupby("flow_id", group_keys=False, sort=False).apply(_pick).reset_index(drop=True)


def split_held_out(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = df["session"].isin(HELD_OUT_SESSIONS)
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)


def xy(df: pd.DataFrame, features: list[str] | None = None):
    features = features or FEATURE_NAMES
    return df[features].to_numpy(dtype=np.float64), df["label"].to_numpy(), df["session"].to_numpy()


def _fit(model, X, y):
    if isinstance(model, HistGradientBoostingClassifier):
        model.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    else:
        model.fit(X, y)
    if hasattr(model, "n_jobs"):
        # parallel fit is fine, but a joblib pool per single-flow predict call
        # costs tens of ms; serving is one row at a time, so go single-threaded
        model.set_params(n_jobs=1)
    return model


# ---------------------------------------------------------------------------
# cross-validation
# ---------------------------------------------------------------------------


def cross_validate(model, X, y, groups, n_splits: int = 5, seed: int = 0) -> dict:
    """Leave-sessions-out CV. Returns pooled out-of-fold predictions + summary."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.empty(len(y), dtype=object)
    fold_f1 = []
    for tr, te in cv.split(X, y, groups):
        m = _fit(_clone(model), X[tr], y[tr])
        oof[te] = m.predict(X[te])
        fold_f1.append(f1_score(y[te], oof[te], average="macro"))
    return {
        "oof_pred": oof,
        "macro_f1": float(f1_score(y, oof, average="macro")),
        "accuracy": float((oof == y).mean()),
        "fold_macro_f1": [float(f) for f in fold_f1],
    }


def _clone(model):
    from sklearn.base import clone
    return clone(model)


# ---------------------------------------------------------------------------
# serving
# ---------------------------------------------------------------------------


@dataclass
class Prediction:
    label: str
    confidence: float
    proba: dict[str, float]


class FlowClassifier:
    """Thin wrapper around the serialised sklearn model with the metadata needed
    to use it correctly (feature order, class names, window size)."""

    def __init__(self, model, meta: dict):
        self.model = model
        self.meta = meta
        self.features: list[str] = meta["features"]
        self.classes: list[str] = [str(c) for c in model.classes_]
        self.window: int = meta["window"]

    @classmethod
    def load(cls, path: Path | str = DEFAULT_MODEL) -> "FlowClassifier":
        bundle = joblib.load(path)
        return cls(bundle["model"], bundle["meta"])

    def save(self, path: Path | str = DEFAULT_MODEL) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "meta": self.meta}, path, compress=3)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        return self.model.predict_proba(X)

    def predict(self, x: np.ndarray) -> Prediction:
        p = self.predict_proba(x)[0]
        i = int(p.argmax())
        return Prediction(self.classes[i], float(p[i]), dict(zip(self.classes, p.tolist())))

    def latency(self, X: np.ndarray, n: int = 500) -> dict[str, float]:
        """Median single-flow and batched per-flow inference time in microseconds."""
        X = np.asarray(X, dtype=np.float64)
        single = []
        for i in range(min(n, len(X))):
            t = time.perf_counter()
            self.model.predict_proba(X[i : i + 1])
            single.append(time.perf_counter() - t)
        t = time.perf_counter()
        self.model.predict_proba(X[:n])
        batched = (time.perf_counter() - t) / min(n, len(X))
        return {"single_flow_us_median": float(np.median(single) * 1e6),
                "single_flow_us_p95": float(np.percentile(single, 95) * 1e6),
                "batched_per_flow_us": float(batched * 1e6)}


# ---------------------------------------------------------------------------
# training entry point
# ---------------------------------------------------------------------------


def train(data_path: Path, out: Path = DEFAULT_MODEL, model_name: str = "hist_gbt",
          features: list[str] | None = None, seed: int = 0) -> tuple[FlowClassifier, dict]:
    features = features or FEATURE_NAMES
    df = pd.read_parquet(data_path)
    window = int(df["n_pkts"].max())
    train_df, test_df = split_held_out(df)
    train_df = cap_windows_per_flow(train_df)
    X, y, g = xy(train_df, features)

    model = make_models(seed)[model_name]
    cv = cross_validate(model, X, y, g, seed=seed)
    model = _fit(model, X, y)

    Xt, yt, _ = xy(test_df, features)
    held = {}
    if len(yt):
        pred = model.predict(Xt)
        held = {"accuracy": float((pred == yt).mean()), "macro_f1": float(f1_score(yt, pred, average="macro")),
                "report": classification_report(yt, pred, output_dict=True, zero_division=0),
                "confusion": confusion_matrix(yt, pred, labels=list(model.classes_)).tolist()}

    meta = {
        "model": model_name,
        "features": features,
        "window": window,
        "classes": list(model.classes_),
        "training_windows": int(len(y)),
        "training_sessions": sorted(set(g)),
        "held_out_sessions": HELD_OUT_SESSIONS,
        "cv_macro_f1": cv["macro_f1"],
        "cv_accuracy": cv["accuracy"],
        "cv_fold_macro_f1": cv["fold_macro_f1"],
        "held_out": held,
        "data_path": str(data_path),
        "seed": seed,
    }
    clf = FlowClassifier(model, meta)
    clf.save(out)
    meta["latency"] = clf.latency(Xt if len(Xt) else X)
    clf.meta["latency"] = meta["latency"]
    clf.save(out)
    return clf, meta


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data/processed/windows_w32.parquet")
    ap.add_argument("--model", default="hist_gbt", choices=list(make_models()))
    ap.add_argument("--out", type=Path, default=DEFAULT_MODEL)
    a = ap.parse_args()
    clf, meta = train(a.data, a.out, a.model)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("held_out", "training_sessions")}, indent=1))
    if meta["held_out"]:
        print("held-out accuracy %.3f macro-F1 %.3f" % (meta["held_out"]["accuracy"], meta["held_out"]["macro_f1"]))
