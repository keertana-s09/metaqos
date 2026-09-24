"""
Transport-property ablation: which properties of a modern transport break the
classifier, measured one factor at a time on real held-out traffic.

    python eval/transport_ablation.py

Why this exists
---------------
We cannot measure QUIC/HTTP-3 robustness directly: ISCX was captured in 2016 and
contains essentially no QUIC. What we CAN do is take the real held-out captures
and remove, one at a time, the transport properties that QUIC changes, then
re-run the unchanged model. That attributes the damage to a specific property
instead of inventing a single "simulated QUIC accuracy" number.

The transforms (each applied to real packets, at the packet level):

  mtu1360   cap packets at 1360 B, re-segmenting larger ones.
            Matches the MTU of the ISCX VPN captures - this is the cross-check
            that the measured VPN drop really is an MTU effect.
  mtu1200   cap at 1200 B: QUIC's mandatory minimum, the conservative floor
            some implementations and tunnelled paths sit at.
  noacks    drop pure-ACK packets (TCP, <= 66 B). QUIC carries ACK frames inside
            encrypted packets, so a router never sees them as separate packets.
  asudp     relabel the transport as UDP, as QUIC would appear.
  quiclike  all of the above together (1350 B cap + no visible ACKs + UDP).

IMPORTANT: `quiclike` is NOT a QUIC measurement. It is real traffic with QUIC's
packet-level *consequences* imposed on it. It does not reproduce QUIC's pacing,
congestion control, stream multiplexing or 0-RTT. Report it as a sensitivity
ablation, never as "accuracy on QUIC".
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.evaluate import md_table, per_class_table, tier_accuracy  # noqa: E402
from src.classifier import FlowClassifier  # noqa: E402
from src.feature_extraction import extract_features  # noqa: E402
from src.flows import TCP, UDP, FlowTable, Packet, guess_local_host, read_pcap  # noqa: E402

EVAL = ROOT / "eval"
RAW = ROOT / "data" / "raw"

from eval.dpi_baseline import HELD_OUT_FILES  # noqa: E402

PURE_ACK_MAX = 66  # IPv4+TCP header with common options; no payload
SPLIT_GAP = 1.2e-5

TRANSFORMS = {
    "baseline": {},
    "mtu1360": {"mtu": 1360},
    "mtu1200": {"mtu": 1200},
    "noacks": {"drop_acks": True},
    "asudp": {"as_udp": True},
    "quiclike": {"mtu": 1350, "drop_acks": True, "as_udp": True},
}


def transform_packets(pkts, mtu=None, drop_acks=False, as_udp=False):
    for p in pkts:
        if drop_acks and p.proto == TCP and p.length <= PURE_ACK_MAX:
            continue
        proto = UDP if as_udp else p.proto
        if mtu is None or p.length <= mtu:
            yield Packet(p.ts, p.src, p.dst, p.sport, p.dport, proto, p.length)
            continue
        n_full, rem = divmod(p.length, mtu)
        for i in range(n_full):
            yield Packet(p.ts + i * SPLIT_GAP, p.src, p.dst, p.sport, p.dport, proto, mtu)
        if rem:
            yield Packet(p.ts + n_full * SPLIT_GAP, p.src, p.dst, p.sport, p.dport, proto, rem)


def process(args):
    stem, label, name, kw, window = args
    path = next((p for p in (RAW / f"{stem}.pcap", RAW / f"{stem}.pcapng") if p.exists()), None)
    if path is None:
        return name, label, stem, None
    local = guess_local_host(str(path))
    ft = FlowTable(window=window, local_nets=[f"{local}/32"] if local else None)
    rows = []
    for p in transform_packets(read_pcap(str(path)), **kw):
        for w in ft.add_packet(p):
            rows.append(extract_features(w.packets, w.proto))
    for w in ft.flush():
        rows.append(extract_features(w.packets, w.proto))
    return name, label, stem, (np.stack(rows) if rows else None)


def main():
    clf = FlowClassifier.load()
    jobs = [(stem, label, name, kw, clf.window)
            for name, kw in TRANSFORMS.items() for stem, label in HELD_OUT_FILES.items()]
    print(f"{len(jobs)} jobs ({len(TRANSFORMS)} transforms x {len(HELD_OUT_FILES)} captures)")

    buckets: dict[str, list] = {n: [] for n in TRANSFORMS}
    labels: dict[str, list] = {n: [] for n in TRANSFORMS}
    with ProcessPoolExecutor(10) as ex:
        futs = [ex.submit(process, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            name, label, stem, X = fut.result()
            if X is not None:
                buckets[name].append(X)
                labels[name] += [label] * len(X)
            if i % 16 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)

    rows, per_class = [], {}
    base_acc = None
    for name in TRANSFORMS:
        X = np.vstack(buckets[name])
        y = np.array(labels[name])
        pred = clf.model.predict(X)
        acc = float((pred == y).mean())
        row = {"transform": name, "windows": int(len(y)), "accuracy": acc,
               "macro_f1": float(f1_score(y, pred, average="macro")),
               "tier_accuracy": tier_accuracy(y, pred)}
        if base_acc is None:
            base_acc = acc
        row["delta_acc"] = acc - base_acc
        rows.append(row)
        per_class[name] = per_class_table(y, pred, clf.classes).to_dict("records")
        print(f"  {name:10s} n={len(y):7d}  acc {acc:.4f}  macroF1 {row['macro_f1']:.4f}  "
              f"tier {row['tier_accuracy']:.4f}  ({row['delta_acc']:+.4f})", flush=True)

    df = pd.DataFrame(rows)
    res = {"summary": rows, "per_class": per_class, "pure_ack_max_bytes": PURE_ACK_MAX,
           "transforms": {k: v for k, v in TRANSFORMS.items()}}
    (EVAL / "transport_ablation.json").write_text(json.dumps(res, indent=1))

    # per-class F1 for the most damaging transforms, to show WHICH classes break
    f1_tbl = pd.DataFrame({name: {c["class"]: c["f1"] for c in per_class[name]} for name in TRANSFORMS}).round(3)
    f1_tbl.index.name = "class"
    print("\nper-class F1:\n" + f1_tbl.to_string())

    _append_report(df, f1_tbl)
    print(f"\nwrote {EVAL / 'transport_ablation.json'} and the ablation section of report.md")


def _append_report(df, f1_tbl):
    sec = [
        "## Transport-property ablation (which property breaks the model)",
        "",
        "ISCX predates QUIC, so QUIC robustness cannot be measured directly. Instead each transport property that "
        "QUIC changes is removed from the **real** held-out captures one at a time, at the packet level, and the "
        "unchanged model is re-run. That attributes the damage to a specific property.",
        "",
        "| transform | what it does |",
        "|---|---|",
        "| `mtu1360` | cap packets at 1360 B (the ISCX VPN captures' tun MTU) |",
        "| `mtu1200` | cap at 1200 B, QUIC's mandatory minimum datagram |",
        "| `noacks` | drop pure-ACK packets (TCP, <= 66 B); QUIC hides ACK frames inside encrypted packets |",
        "| `asudp` | relabel the transport as UDP, as QUIC appears |",
        "| `quiclike` | all three together (1350 B + no visible ACKs + UDP) |",
        "",
        md_table(df), "",
        "Per-class F1 under each transform:", "",
        md_table(f1_tbl.reset_index()), "",
        "**`quiclike` is not a QUIC measurement.** It is real traffic carrying QUIC's packet-level consequences; it "
        "does not reproduce QUIC pacing, congestion control, stream multiplexing or 0-RTT. Quote it as a sensitivity "
        "ablation. The only way to fill the QUIC row properly is to capture real HTTP-3 traffic and re-run "
        "`eval/evaluate.py` against it.",
        "",
    ]
    report = EVAL / "report.md"
    txt = report.read_text().split("\n## Transport-property ablation")[0].rstrip()
    report.write_text(txt + "\n\n" + "\n".join(sec))


if __name__ == "__main__":
    main()
