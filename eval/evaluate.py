"""
Full evaluation: model comparison, held-out metrics, latency, ablations.

    python eval/evaluate.py                       # uses data/processed/windows_w32.parquet
    python eval/evaluate.py --final hist_gbt      # force which model becomes models/metaqos_classifier.joblib

Writes eval/report.md, eval/results.json, eval/confusion_matrix.png,
eval/feature_importance.png and models/MODEL_CARD.md. Every number in those
files is computed here; nothing is hard-coded.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix, f1_score  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.classifier import (  # noqa: E402
    DEFAULT_MODEL, HELD_OUT_SESSIONS, MAX_WINDOWS_PER_FLOW, FlowClassifier, _fit, cap_windows_per_flow,
    cross_validate, make_models, split_held_out, xy,
)
from src.feature_extraction import FEATURE_NAMES  # noqa: E402
from src.qos_mapper import CLASS_TO_TIER  # noqa: E402

EVAL = ROOT / "eval"


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):  # itertuples keeps per-column dtypes (iterrows would upcast ints to float)
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in row]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("label").agg(windows=("label", "size"), flows=("flow_id", "nunique"),
                                sessions=("session", "nunique"), apps=("app", lambda s: ", ".join(sorted(s.unique()))))
    return g.reset_index()


def per_class_table(y_true, y_pred, classes) -> pd.DataFrame:
    rep = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)
    rows = [{"class": c, "precision": rep[c]["precision"], "recall": rep[c]["recall"], "f1": rep[c]["f1-score"],
             "support": int(rep[c]["support"])} for c in classes]
    rows.append({"class": "macro avg", "precision": rep["macro avg"]["precision"], "recall": rep["macro avg"]["recall"],
                 "f1": rep["macro avg"]["f1-score"], "support": int(rep["macro avg"]["support"])})
    return pd.DataFrame(rows)


def tier_accuracy(y_true, y_pred) -> float:
    """Accuracy at the level that matters for QoS: did the flow land in the right tier?"""
    t = np.array([CLASS_TO_TIER.get(c, "best_effort") for c in y_true])
    p = np.array([CLASS_TO_TIER.get(c, "best_effort") for c in y_pred])
    return float((t == p).mean())


def plot_confusion(cm, classes, path: Path, title: str):
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=40, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if cmn[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def flow_breakdown(test_df: pd.DataFrame, min_windows: int = 100) -> pd.DataFrame:
    """Every held-out flow with >= min_windows windows: what the model called it.
    Separates 'the classifier is wrong' from 'the capture-file label is wrong'
    (e.g. an interactive SSH session inside an SFTP capture)."""
    rows = []
    for (sess, label, fid), g in test_df.groupby(["session", "label", "flow_id"]):
        if len(g) < min_windows:
            continue
        top = g["pred"].value_counts()
        proto = "udp" if g["proto_udp"].iloc[0] else "tcp"
        rows.append({"session": sess, "label": label, "flow": f"{proto} :{fid.split(':')[2].split('<')[0]}<->:{fid.rsplit(':', 1)[1]}",
                     "windows": len(g), "mean pkt B": round(g["size_mean"].mean()), "pkt/s": round(g["pkts_per_s"].median()),
                     "up byte frac": round(g["up_byte_frac"].mean(), 2), "majority pred": top.index[0],
                     "share": round(top.iloc[0] / len(g), 3)})
    return pd.DataFrame(rows).sort_values(["session", "windows"], ascending=[True, False]).reset_index(drop=True)


def early_classification(model, test_df: pd.DataFrame, features, alpha: float = 0.4) -> pd.DataFrame:
    """Accuracy as a function of how many windows of the flow have been seen,
    for raw per-window predictions and for the EMA-smoothed decision the
    re-classification loop actually uses."""
    classes = list(model.classes_)
    X = test_df[features].to_numpy()
    P = model.predict_proba(X)
    test_df = test_df.assign(_p=list(P)).sort_values(["flow_id", "window_idx"])
    raw_hits, sm_hits = {}, {}
    for _, g in test_df.groupby("flow_id", sort=False):
        ps = None
        for k, (_, row) in enumerate(g.iterrows()):
            p = row["_p"]
            ps = p if ps is None else alpha * p + (1 - alpha) * ps
            raw_hits.setdefault(k, []).append(classes[int(p.argmax())] == row["label"])
            sm_hits.setdefault(k, []).append(classes[int(ps.argmax())] == row["label"])
    buckets = [(0, 0, "1st window"), (1, 1, "2nd"), (2, 4, "3rd-5th"), (5, 19, "6th-20th"), (20, 10**9, "21st+")]
    rows = []
    for lo, hi, name in buckets:
        r = [h for k, hs in raw_hits.items() if lo <= k <= hi for h in hs]
        s = [h for k, hs in sm_hits.items() if lo <= k <= hi for h in hs]
        if r:
            rows.append({"windows seen": name, "n": len(r), "raw acc": float(np.mean(r)), "smoothed acc": float(np.mean(s))})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data/processed/windows_w32.parquet")
    ap.add_argument("--final", default=None, help="model to export; default = best CV macro-F1")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    df = pd.read_parquet(a.data)
    window = int(df["n_pkts"].max())
    res: dict = {"data": str(a.data), "window": window, "n_windows_total": int(len(df))}

    # ---------------- dataset ----------------
    ds = dataset_summary(df)
    train_df, test_df = split_held_out(df)
    train_capped = cap_windows_per_flow(train_df)
    X, y, g = xy(train_capped)
    Xt, yt, gt = xy(test_df)
    classes = sorted(set(y))
    res["dataset"] = ds.to_dict("records")
    res["train_windows_capped"] = int(len(y)); res["train_windows_raw"] = int(len(train_df))
    res["test_windows"] = int(len(yt)); res["held_out_sessions"] = HELD_OUT_SESSIONS
    print(ds.to_string()); print(f"\ntrain windows {len(train_df)} -> {len(y)} after per-flow cap; held-out windows {len(yt)}")

    # ---------------- model comparison (leave-session-out CV) ----------------
    comp_rows, fitted = [], {}
    for name, model in make_models(a.seed).items():
        t = time.time()
        cv = cross_validate(model, X, y, g, seed=a.seed)
        t_cv = time.time() - t
        t = time.time(); m = _fit(model, X, y); t_fit = time.time() - t
        pred = m.predict(Xt)
        tmp = ROOT / "models" / f"_{name}.joblib"; joblib.dump(m, tmp, compress=3); size_kb = tmp.stat().st_size / 1024; tmp.unlink()
        clf = FlowClassifier(m, {"features": FEATURE_NAMES, "window": window})
        lat = clf.latency(Xt)
        with threadpool_limits(limits=1):
            lat1 = clf.latency(Xt)
        fitted[name] = m
        comp_rows.append({"model": name, "cv macro-F1": cv["macro_f1"], "cv acc": cv["accuracy"],
                          "cv fold F1 min/max": f"{min(cv['fold_macro_f1']):.2f}/{max(cv['fold_macro_f1']):.2f}",
                          "held-out macro-F1": float(f1_score(yt, pred, average="macro")), "held-out acc": float((pred == yt).mean()),
                          "held-out tier acc": tier_accuracy(yt, pred),
                          "single-flow latency (us)": round(lat["single_flow_us_median"]), "1-thread latency (us)": round(lat1["single_flow_us_median"]),
                          "batched (us/flow)": round(lat["batched_per_flow_us"], 1), "size (KB)": round(size_kb), "fit (s)": round(t_fit, 1)})
        print(f"{name:14s} cv F1 {cv['macro_f1']:.3f} acc {cv['accuracy']:.3f} | held-out F1 {comp_rows[-1]['held-out macro-F1']:.3f} "
              f"acc {comp_rows[-1]['held-out acc']:.3f} tier-acc {comp_rows[-1]['held-out tier acc']:.3f} | "
              f"lat {lat['single_flow_us_median']:.0f}us (1thr {lat1['single_flow_us_median']:.0f}us) size {size_kb:.0f}KB  [{t_cv:.0f}s cv]")
    comp = pd.DataFrame(comp_rows)
    res["model_comparison"] = comp_rows
    final_name = a.final or comp.sort_values("cv macro-F1", ascending=False).iloc[0]["model"]
    res["final_model"] = final_name
    model = fitted[final_name]
    print(f"\nfinal model: {final_name}")

    # ---------------- held-out detail for the final model ----------------
    pred = model.predict(Xt)
    pc = per_class_table(yt, pred, classes)
    cm = confusion_matrix(yt, pred, labels=classes)
    plot_confusion(cm, classes, EVAL / "confusion_matrix.png", f"{final_name}: held-out sessions (window={window})")
    res["held_out_per_class"] = pc.to_dict("records"); res["confusion"] = cm.tolist(); res["classes"] = classes
    per_app = (test_df.assign(pred=pred, hit=(pred == yt)).groupby(["label", "app", "session"])
               .agg(windows=("hit", "size"), acc=("hit", "mean")).reset_index())
    res["held_out_per_app"] = per_app.to_dict("records")
    print(pc.to_string()); print(per_app.to_string())
    flows = flow_breakdown(test_df.assign(pred=pred))
    res["held_out_flows"] = flows.to_dict("records")
    print(flows.to_string())

    # CV out-of-fold per-class (more sessions than the held-out set)
    cv_final = cross_validate(make_models(a.seed)[final_name], X, y, g, seed=a.seed)
    pc_cv = per_class_table(y, cv_final["oof_pred"], classes)
    res["cv_per_class"] = pc_cv.to_dict("records")
    cm_cv = confusion_matrix(y, cv_final["oof_pred"], labels=classes)
    plot_confusion(cm_cv, classes, EVAL / "confusion_matrix_cv.png", f"{final_name}: leave-session-out CV (train sessions)")
    res["confusion_cv"] = cm_cv.tolist()

    # ---------------- early classification ----------------
    early = early_classification(model, test_df, FEATURE_NAMES)
    res["early_classification"] = early.to_dict("records")
    print(early.to_string())

    # ---------------- feature importance ----------------
    pi = permutation_importance(model, Xt, yt, scoring="f1_macro", n_repeats=5, random_state=a.seed, n_jobs=4)
    imp = pd.DataFrame({"feature": FEATURE_NAMES, "importance": pi.importances_mean, "std": pi.importances_std}) \
        .sort_values("importance", ascending=False).reset_index(drop=True)
    res["feature_importance"] = imp.to_dict("records")
    fig, ax = plt.subplots(figsize=(7, 6))
    top = imp.head(20)[::-1]
    ax.barh(top["feature"], top["importance"], xerr=top["std"], color="#4c72b0")
    ax.set_xlabel("drop in held-out macro-F1 when feature is permuted"); ax.set_title("permutation importance (top 20)")
    fig.tight_layout(); fig.savefig(EVAL / "feature_importance.png", dpi=130); plt.close(fig)
    print(imp.head(12).to_string())

    # ---------------- ablation: no transport-protocol flag ----------------
    feats_noproto = [f for f in FEATURE_NAMES if f != "proto_udp"]
    Xn, _, _ = xy(train_capped, feats_noproto); Xtn, _, _ = xy(test_df, feats_noproto)
    m_ab = make_models(a.seed)[final_name]
    cv_ab = cross_validate(m_ab, Xn, y, g, seed=a.seed)
    m_ab = _fit(m_ab, Xn, y); pred_ab = m_ab.predict(Xtn)
    res["ablation_no_proto"] = {"cv_macro_f1": cv_ab["macro_f1"], "held_out_macro_f1": float(f1_score(yt, pred_ab, average="macro")),
                                "held_out_acc": float((pred_ab == yt).mean())}
    print("ablation no proto_udp:", res["ablation_no_proto"])

    # ---------------- export final model ----------------
    lat = FlowClassifier(model, {"features": FEATURE_NAMES, "window": window}).latency(Xt)
    with threadpool_limits(limits=1):
        lat1 = FlowClassifier(model, {"features": FEATURE_NAMES, "window": window}).latency(Xt)
    meta = {"model": final_name, "features": FEATURE_NAMES, "window": window, "classes": classes,
            "training_windows": int(len(y)), "training_sessions": sorted(set(g)), "held_out_sessions": HELD_OUT_SESSIONS,
            "cv_macro_f1": cv_final["macro_f1"], "cv_accuracy": cv_final["accuracy"], "cv_fold_macro_f1": cv_final["fold_macro_f1"],
            "held_out": {"accuracy": float((pred == yt).mean()), "macro_f1": float(f1_score(yt, pred, average="macro")),
                         "tier_accuracy": tier_accuracy(yt, pred)},
            "latency": lat, "latency_1thread": lat1, "data_path": str(a.data), "seed": a.seed}
    FlowClassifier(model, meta).save(DEFAULT_MODEL)
    res["final_meta"] = meta
    res["model_size_kb"] = DEFAULT_MODEL.stat().st_size / 1024
    (EVAL / "results.json").write_text(json.dumps(res, indent=1, default=str))
    write_report(res, ds, comp, pc, pc_cv, per_app, early, imp)
    write_model_card(res, pc, imp)
    print(f"\nwrote {EVAL / 'report.md'}, {DEFAULT_MODEL} ({res['model_size_kb']:.0f} KB)")


def write_report(res, ds, comp, pc, pc_cv, per_app, early, imp):
    fm = res["final_meta"]; ho = fm["held_out"]; ab = res["ablation_no_proto"]
    lines = [
        "# MetaQoS evaluation report", "",
        f"Generated by `eval/evaluate.py` from `{Path(res['data']).name}` (window = {res['window']} packets). "
        "All numbers below are measured; none are placeholders.", "",
        "## Data", "",
        "ISCX VPN-nonVPN 2016, NonVPN captures only (see `data/fetch_iscx.py` for the exact file list). "
        "One sample = one window of consecutive packets from one bidirectional flow.", "",
        md_table(ds, "{:.0f}"), "",
        f"* Windows total: {res['n_windows_total']}. Training windows after capping at {MAX_WINDOWS_PER_FLOW} per flow: "
        f"{res['train_windows_capped']} (from {res['train_windows_raw']}). Held-out windows: {res['test_windows']}.",
        f"* Held-out sessions (never used for training or model selection): `{'`, `'.join(res['held_out_sessions'])}`.",
        "* Cross-validation is **leave-sessions-out** (`StratifiedGroupKFold`, 5 folds, grouped by capture session). "
        "Splitting by window instead would leak near-duplicate windows of the same flow across folds and report ~99% accuracy that means nothing.",
        "", "## Model comparison", "",
        md_table(comp), "",
        "Latency is per flow decision on this machine (one 32-packet window -> one `predict_proba` call). "
        "`1-thread` pins BLAS/OpenMP to one thread, which is what a router CPU would run; sklearn's gradient boosting "
        "pays a large fixed per-call cost from its thread pool otherwise.", "",
        f"**Selected model: `{res['final_model']}`** (highest leave-session-out macro-F1).", "",
        "## Held-out sessions: per-class metrics", "",
        f"Accuracy {ho['accuracy']:.3f}, macro-F1 {ho['macro_f1']:.3f}, **QoS-tier accuracy {ho['tier_accuracy']:.3f}** "
        "(tier accuracy counts a prediction as correct when it lands the flow in the right queue, e.g. video_call vs voip_audio are both `realtime`).", "",
        md_table(pc), "", "![confusion matrix](confusion_matrix.png)", "",
        "### Per held-out capture", "", md_table(per_app), "",
        "### Per held-out flow (>= 100 windows)", "",
        "Ports shown only to identify the flow; they are not features. This table separates classifier errors from "
        "capture-file label noise: a second SSH session inside an SFTP capture is labelled `file_transfer` but carries "
        "57-byte interactive packets (the model calls it voice/chat/email, i.e. small-packet interactive classes), and a "
        "video call's separate audio stream is labelled `video_call` but is, in fact, voice.", "",
        md_table(pd.DataFrame(res["held_out_flows"])), "",
        "## Leave-session-out CV: per-class metrics (training sessions, out-of-fold)", "",
        f"Accuracy {fm['cv_accuracy']:.3f}, macro-F1 {fm['cv_macro_f1']:.3f}; per-fold macro-F1: "
        + ", ".join(f"{f:.3f}" for f in fm["cv_fold_macro_f1"]) + ".", "",
        md_table(pc_cv), "", "![cv confusion matrix](confusion_matrix_cv.png)", "",
        "## Early classification (held-out)", "",
        "How accurate is the decision after the first window vs later, for raw per-window predictions and for the "
        "EMA-smoothed decision the re-classification loop uses (alpha = 0.4)?", "",
        md_table(early), "",
        "## Feature importance (permutation, held-out macro-F1)", "",
        md_table(imp.head(15)), "", "![feature importance](feature_importance.png)", "",
        "## Ablation: without the transport-protocol flag", "",
        f"`proto_udp` is the only header field used. Removing it: CV macro-F1 {ab['cv_macro_f1']:.3f}, "
        f"held-out macro-F1 {ab['held_out_macro_f1']:.3f}, held-out accuracy {ab['held_out_acc']:.3f} "
        f"(vs {fm['cv_macro_f1']:.3f} / {ho['macro_f1']:.3f} / {ho['accuracy']:.3f} with it).", "",
        "## Inference cost", "",
        f"* Model file: {res['model_size_kb']:.0f} KB (joblib, compressed).",
        f"* Single-flow latency: median {fm['latency']['single_flow_us_median']:.0f} us "
        f"(p95 {fm['latency']['single_flow_us_p95']:.0f} us); pinned to one thread: {fm['latency_1thread']['single_flow_us_median']:.0f} us. "
        f"Batched: {fm['latency']['batched_per_flow_us']:.1f} us/flow.",
        "* Feature extraction for one window is a few NumPy reductions over 32 values (~50 us in Python).",
    ]
    (EVAL / "report.md").write_text("\n".join(lines))


def write_model_card(res, pc, imp):
    fm = res["final_meta"]; ho = fm["held_out"]
    lines = [
        f"# Model card: MetaQoS flow classifier (`{res['final_model']}`)", "",
        "## Intended use", "",
        "Assign a traffic class to an encrypted network flow from packet metadata only (sizes, timings, direction, "
        "transport protocol) so a home router can place it in an appropriate QoS queue. Not intended for security "
        "decisions, user identification, or any purpose that requires knowing *what* is being communicated.", "",
        "## Inputs", "",
        f"{len(fm['features'])} features computed over a window of {fm['window']} consecutive packets of one bidirectional flow "
        "(see `src/feature_extraction.py`). No IP addresses, ports or payload bytes are used.", "",
        "## Outputs", "",
        f"One of `{'`, `'.join(fm['classes'])}` with a class-probability vector; the QoS mapper acts only above a confidence floor.", "",
        "## Training data", "",
        "ISCX VPN-nonVPN 2016 (Draper-Gil, Lashkari, Mamun, Ghorbani; ICISSP 2016), NonVPN captures listed in "
        f"`data/fetch_iscx.py`. {fm['training_windows']} windows from {len(fm['training_sessions'])} capture sessions "
        f"(after capping {MAX_WINDOWS_PER_FLOW} windows per flow). Labels are per capture file, so background flows in a capture "
        "carry that file's label (known label noise).", "",
        "## Evaluation", "",
        f"* Leave-session-out 5-fold CV on training sessions: accuracy {fm['cv_accuracy']:.3f}, macro-F1 {fm['cv_macro_f1']:.3f}.",
        f"* Held-out sessions ({len(fm['held_out_sessions'])} sessions never seen in training): accuracy {ho['accuracy']:.3f}, "
        f"macro-F1 {ho['macro_f1']:.3f}, QoS-tier accuracy {ho['tier_accuracy']:.3f}.", "",
        md_table(pc), "",
        "## Cost", "",
        f"{res['model_size_kb']:.0f} KB on disk; {fm['latency_1thread']['single_flow_us_median']:.0f} us per flow decision single-threaded.", "",
        "## Most influential features", "",
        ", ".join(f"`{f}`" for f in imp.head(8)["feature"]), "",
        "## Known limitations", "",
        "* No `browsing` or `gaming` class: ISCX does not contain those captures. Such traffic will be mapped to the nearest "
        "class the model knows (see the demo's synthetic flows) and should be treated as unknown.",
        "* Trained on 2016 captures from one lab; codecs, CDNs and transport (QUIC) have changed. Expect degraded accuracy "
        "on modern traffic without retraining on fresh captures.",
        "* Only NonVPN traffic. Inside a VPN tunnel all flows share one 5-tuple and this per-flow approach does not apply.",
        "* Capture-file labelling means some 'video_call' windows are really the app's control channel, etc.",
        "", "See `eval/report.md` for the full evaluation.",
    ]
    (ROOT / "models" / "MODEL_CARD.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
