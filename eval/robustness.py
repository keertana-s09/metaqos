"""
Robustness of the non-VPN-trained model on VPN-tunnelled traffic.

    python eval/robustness.py     # -> eval/robustness.json, appends to eval/report.md

This is a real measurement, not a simulation: ISCX captured the same activities
a second time inside an OpenVPN tunnel. The model trained purely on non-VPN
traffic is applied to those captures unchanged.

What the number means
---------------------
Inside a tunnel every application shares ONE 5-tuple (client <-> VPN server), so
per-flow classification cannot separate concurrent applications. Each ISCX VPN
capture happens to contain a single activity, so the tunnel flow's behaviour
still reflects that activity - which makes these captures a fair test of
"does tunnelling destroy the timing/size signal the model relies on?" but NOT
evidence that the system works on a mixed real-world tunnel. It cannot; that
limitation is structural, not a matter of accuracy.

`video_call` has no VPN capture in the release, so only 5 of 6 classes appear.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.evaluate import md_table, per_class_table, plot_confusion, tier_accuracy  # noqa: E402
from src.classifier import FlowClassifier, split_held_out  # noqa: E402

EVAL = ROOT / "eval"
VPN_PARQUET = ROOT / "data/processed/windows_vpn_w32.parquet"
NONVPN_PARQUET = ROOT / "data/processed/windows_w32.parquet"


def main():
    if not VPN_PARQUET.exists():
        sys.exit(f"{VPN_PARQUET} missing. Run:\n"
                 f"  python data/fetch_iscx.py --vpn\n"
                 f"  python data/build_dataset.py --window 32 --vpn")
    clf = FlowClassifier.load()
    vpn = pd.read_parquet(VPN_PARQUET)
    classes = clf.classes

    Xv = vpn[clf.features].to_numpy(dtype=np.float64)
    pred_v = clf.model.predict(Xv)
    yv = vpn["label"].to_numpy()

    # non-VPN baseline restricted to the same 5 classes, so the comparison is like-for-like
    nonvpn = pd.read_parquet(NONVPN_PARQUET)
    _, held = split_held_out(nonvpn)
    shared = sorted(set(yv))
    held_s = held[held["label"].isin(shared)]
    Xb = held_s[clf.features].to_numpy(dtype=np.float64)
    yb = held_s["label"].to_numpy()
    pred_b = clf.model.predict(Xb)

    res = {
        "classes_present": shared,
        "vpn_windows": int(len(yv)), "vpn_flows": int(vpn["flow_id"].nunique()),
        "baseline": {"windows": int(len(yb)), "accuracy": float((pred_b == yb).mean()),
                     "macro_f1": float(f1_score(yb, pred_b, average="macro")),
                     "tier_accuracy": tier_accuracy(yb, pred_b)},
        "vpn": {"windows": int(len(yv)), "accuracy": float((pred_v == yv).mean()),
                "macro_f1": float(f1_score(yv, pred_v, average="macro")),
                "tier_accuracy": tier_accuracy(yv, pred_v)},
        "per_class": per_class_table(yv, pred_v, shared).to_dict("records"),
        "confusion": confusion_matrix(yv, pred_v, labels=classes).tolist(),
        "confusion_labels": classes,
    }
    b, v = res["baseline"], res["vpn"]
    res["delta"] = {k: v[k] - b[k] for k in ("accuracy", "macro_f1", "tier_accuracy")}

    # per-capture detail: one row per VPN capture file
    per_file = (vpn.assign(pred=pred_v, hit=(pred_v == yv)).groupby(["label", "file"])
                .agg(windows=("hit", "size"), acc=("hit", "mean"),
                     majority=("pred", lambda s: s.value_counts().index[0])).reset_index())
    res["per_file"] = per_file.to_dict("records")

    plot_confusion(np.array(res["confusion"]), classes, EVAL / "confusion_matrix_vpn.png",
                   "held-out model applied to VPN-tunnelled traffic")

    print(f"non-VPN baseline ({b['windows']} windows, same {len(shared)} classes): "
          f"acc {b['accuracy']:.3f}  macro-F1 {b['macro_f1']:.3f}  tier {b['tier_accuracy']:.3f}")
    print(f"VPN-tunnelled    ({v['windows']} windows): "
          f"acc {v['accuracy']:.3f}  macro-F1 {v['macro_f1']:.3f}  tier {v['tier_accuracy']:.3f}")
    print(f"delta: acc {res['delta']['accuracy']:+.3f}  macro-F1 {res['delta']['macro_f1']:+.3f}  "
          f"tier {res['delta']['tier_accuracy']:+.3f}")
    print(per_file.to_string())

    (EVAL / "robustness.json").write_text(json.dumps(res, indent=1, default=str))
    _append_report(res, per_file)
    print(f"\nwrote {EVAL / 'robustness.json'} and the VPN section of report.md")


def _append_report(res, per_file):
    b, v, d = res["baseline"], res["vpn"], res["delta"]
    sec = [
        "## Robustness: VPN-tunnelled traffic (measured, not simulated)",
        "",
        "The model trained only on non-VPN traffic, applied unchanged to the ISCX OpenVPN captures of the same "
        f"activities. `video_call` has no VPN capture in the release, so this covers {len(res['classes_present'])} "
        f"classes: `{'`, `'.join(res['classes_present'])}`. The non-VPN baseline below is the held-out set restricted "
        "to those same classes, so the comparison is like-for-like.",
        "",
        "| set | windows | accuracy | macro-F1 | QoS-tier accuracy |",
        "|---|---|---|---|---|",
        f"| non-VPN held-out | {b['windows']} | {b['accuracy']:.3f} | {b['macro_f1']:.3f} | {b['tier_accuracy']:.3f} |",
        f"| VPN-tunnelled | {v['windows']} | {v['accuracy']:.3f} | {v['macro_f1']:.3f} | {v['tier_accuracy']:.3f} |",
        f"| **change** | | **{d['accuracy']:+.3f}** | **{d['macro_f1']:+.3f}** | **{d['tier_accuracy']:+.3f}** |",
        "",
        "**This number must not be read as 'MetaQoS works through a VPN.'** Inside a tunnel every application shares "
        "one 5-tuple, so per-flow classification cannot separate concurrent applications at all. Each ISCX VPN capture "
        "contains a single activity, so the tunnel flow still carries that activity's timing and size signature; the "
        "measurement answers only the narrower question of whether tunnelling destroys that signature.",
        "",
        md_table(per_class_table_from(res)), "",
        "![VPN confusion matrix](confusion_matrix_vpn.png)", "",
        "### Per VPN capture", "", md_table(per_file), "",
    ]
    report = EVAL / "report.md"
    txt = report.read_text().split("\n## Robustness: VPN")[0].rstrip()
    report.write_text(txt + "\n\n" + "\n".join(sec))


def per_class_table_from(res) -> pd.DataFrame:
    return pd.DataFrame(res["per_class"])


if __name__ == "__main__":
    main()
