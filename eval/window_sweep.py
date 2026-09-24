"""
How many packets does the router need to see before deciding?

Builds feature tables for several window sizes (if not already built) and
reports leave-session-out CV and held-out macro-F1 for the chosen model type.
Smaller windows decide sooner but see less; the sweep quantifies the trade-off.

    python eval/window_sweep.py --windows 8 16 32 64 --model hist_gbt
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.classifier import _fit, cap_windows_per_flow, cross_validate, make_models, split_held_out, xy  # noqa: E402

PROCESSED = ROOT / "data" / "processed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--model", default="hist_gbt")
    a = ap.parse_args()

    rows = []
    for w in a.windows:
        path = PROCESSED / f"windows_w{w}.parquet"
        if not path.exists():
            subprocess.run([sys.executable, str(ROOT / "data/build_dataset.py"), "--window", str(w)], check=True)
        df = pd.read_parquet(path)
        tr, te = split_held_out(df)
        X, y, g = xy(cap_windows_per_flow(tr))
        Xt, yt, _ = xy(te)
        cv = cross_validate(make_models()[a.model], X, y, g)
        m = _fit(make_models()[a.model], X, y)
        ho = float(f1_score(yt, m.predict(Xt), average="macro"))
        rows.append({"window (pkts)": int(w), "train windows": int(len(y)), "cv macro-F1": cv["macro_f1"], "cv acc": cv["accuracy"],
                     "held-out macro-F1": ho, "held-out acc": float((m.predict(Xt) == yt).mean())})
        print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    (ROOT / "eval" / "window_sweep.json").write_text(json.dumps(rows, indent=1))
    print(out.to_string(index=False))
    # append to the report (replacing any previous sweep section)
    from eval.evaluate import md_table
    report = ROOT / "eval" / "report.md"
    if report.exists():
        txt = report.read_text().split("\n## Window size")[0].rstrip()
        txt += ("\n\n## Window size sweep\n\nSame model type (`" + a.model + "`), same session split, different observation "
                "window. A smaller window decides sooner (8 packets is ~80 ms of a VoIP call) but sees less.\n\n"
                + md_table(out) + "\n")
        report.write_text(txt)


if __name__ == "__main__":
    main()
