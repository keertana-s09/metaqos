"""
Turn the raw ISCX pcaps into a feature table: one row per observation window.

    python data/build_dataset.py --window 32 --out data/processed/windows_w32.parquet

Labeling caveat (important for interpreting results)
----------------------------------------------------
ISCX labels are per *capture file*: every flow in `skype_video1a.pcap` is
labelled "video_call", including background chatter the OS produced during the
call (OCSP checks, telemetry, Skype's own signalling channel, ...). We drop the
worst of it (DNS/NTP/mDNS/SSDP in flows.py, and flows that never reach half a
window) but some label noise remains. It depresses accuracy slightly and is a
property of the dataset, not something we fix by hand-curating flows, which
would be un-reproducible.

Provenance columns (never used as features)
-------------------------------------------
file        the pcap the window came from
session     capture session; `xxx1a` and `xxx1b` are the two ends of one call
            and share a session id so they never straddle a train/test split
app         application name (skype_video, netflix, ...) for per-app analysis
flow_id     5-tuple string
window_idx  0 for the first window of the flow, 1 for the second, ...
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.fetch_iscx import MANIFEST, RAW_DIR, VPN_MANIFEST  # noqa: E402
from src.feature_extraction import FEATURE_NAMES, extract_features  # noqa: E402
from src.flows import FlowTable, guess_local_host  # noqa: E402

PROCESSED = ROOT / "data" / "processed"


def session_id(stem: str) -> str:
    """facebook_video1a -> facebook_video1 ; voipbuster_4b -> voipbuster_4 ; netflix3 -> netflix3"""
    return re.sub(r"(\d)[ab]$", r"\1", stem)


def app_name(stem: str) -> str:
    """facebook_video1a -> facebook_video ; AIMchat1 -> aimchat ; youtubeHTML5_1 -> youtubehtml5"""
    s = re.sub(r"_?\d+[ab]?$", "", stem)
    return s.lower()


def process_file(path: Path, label: str, window: int) -> pd.DataFrame:
    local = guess_local_host(str(path))
    ft = FlowTable(window=window, local_nets=[f"{local}/32"] if local else None)
    rows, meta = [], []
    for w in ft.windows_from_pcap(str(path)):
        rows.append(extract_features(w.packets, w.proto))
        meta.append((w.flow_id, w.index))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(np.stack(rows), columns=FEATURE_NAMES)
    df["flow_id"] = [m[0] for m in meta]
    df["window_idx"] = [m[1] for m in meta]
    stem = path.stem
    df["file"] = stem
    df["session"] = session_id(stem)
    df["app"] = app_name(stem)
    df["label"] = label
    df["local_host"] = local
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--vpn", action="store_true", help="build from the VPN-tunnelled captures instead")
    args = ap.parse_args()
    out = args.out or PROCESSED / (f"windows_vpn_w{args.window}.parquet" if args.vpn
                                   else f"windows_w{args.window}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)

    jobs = []
    for label, files in (VPN_MANIFEST if args.vpn else MANIFEST).items():
        for rel in files:
            p = RAW_DIR / Path(rel).name
            if p.exists():
                jobs.append((p, label))
            else:
                print(f"missing (skipped): {p.name}")

    t0 = time.time()
    parts, summary = [], []
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(process_file, p, lab, args.window): (p, lab) for p, lab in jobs}
        for fut in as_completed(futs):
            p, lab = futs[fut]
            df = fut.result()
            parts.append(df)
            summary.append({"file": p.name, "label": lab, "windows": len(df),
                            "flows": int(df["flow_id"].nunique()) if len(df) else 0,
                            "local_host": df["local_host"].iloc[0] if len(df) else None})
            print(f"{lab:14s} {p.name:28s} {len(df):7d} windows  {summary[-1]['flows']:5d} flows", flush=True)

    full = pd.concat(parts, ignore_index=True)
    full.to_parquet(out, index=False)
    (out.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {out}: {len(full)} windows, {full['flow_id'].nunique()} flows, "
          f"{full['session'].nunique()} sessions in {time.time() - t0:.0f}s")
    print(full.groupby("label").agg(windows=("label", "size"), flows=("flow_id", "nunique"),
                                    sessions=("session", "nunique")).to_string())


if __name__ == "__main__":
    main()
