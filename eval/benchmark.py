"""
Resource and throughput benchmarks: what it costs to run MetaQoS, and whether
it keeps up under concurrent multi-flow load.

    python eval/benchmark.py                 # -> eval/benchmark.json + stdout tables

Measures, on THIS machine (a desktop x86 CPU - see the note in the report about
router-class SoCs):

1. Per-decision latency, split into its two real parts (feature extraction and
   model inference), at p50/p95/p99, pinned to one thread.
2. Model footprint: on disk and resident in memory.
3. End-to-end streaming throughput: packets/second through the full pipeline
   (flow table -> features -> classify -> QoS decision).
4. Concurrent multi-flow load: the same, with 1 ... N flows interleaved, plus
   flow-table memory per tracked flow, so the per-flow state cost is explicit.
"""
from __future__ import annotations

import gc
import json
import pickle
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.classifier import DEFAULT_MODEL, FlowClassifier, split_held_out  # noqa: E402
from src.feature_extraction import extract_features  # noqa: E402
from src.flows import UDP, FlowTable, Packet  # noqa: E402
from src.qos_mapper import QoSPolicy  # noqa: E402
from src.reclassify import FlowTracker  # noqa: E402

EVAL = ROOT / "eval"


def percentiles(samples: list[float]) -> dict[str, float]:
    a = np.array(samples) * 1e6  # -> microseconds
    return {"p50_us": float(np.percentile(a, 50)), "p95_us": float(np.percentile(a, 95)),
            "p99_us": float(np.percentile(a, 99)), "mean_us": float(a.mean())}


def bench_latency(clf: FlowClassifier, X: np.ndarray, windows: list[list], n: int = 2000) -> dict:
    """Feature extraction and inference timed separately, one thread."""
    out = {}
    with threadpool_limits(limits=1):
        # warm up (first call allocates sklearn's internal buffers)
        for i in range(20):
            clf.predict_proba(X[i])
        infer = []
        for i in range(min(n, len(X))):
            t = time.perf_counter()
            clf.predict_proba(X[i])
            infer.append(time.perf_counter() - t)
        out["inference"] = percentiles(infer)

        feat = []
        for w in windows[:n]:
            t = time.perf_counter()
            extract_features(w, UDP)
            feat.append(time.perf_counter() - t)
        out["feature_extraction"] = percentiles(feat)

        both = [a + b for a, b in zip(infer, feat)]
        out["per_decision_total"] = percentiles(both)

        t = time.perf_counter()
        clf.predict_proba(X[:1000])
        out["batched_1000_us_per_flow"] = (time.perf_counter() - t) / 1000 * 1e6
    return out


def bench_footprint(clf: FlowClassifier, disk_kb: float | None = None) -> dict:
    if disk_kb is None:
        disk_kb = Path(DEFAULT_MODEL).stat().st_size / 1024
    blob = pickle.dumps(clf.model)
    gc.collect()
    tracemalloc.start()
    m = pickle.loads(blob)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    n_nodes = None
    if hasattr(m, "estimators_"):
        n_nodes = int(sum(e.tree_.node_count for e in m.estimators_))
    elif hasattr(m, "tree_"):
        n_nodes = int(m.tree_.node_count)
    return {"on_disk_kb": disk_kb, "pickle_kb": len(blob) / 1024, "resident_kb": peak / 1024, "tree_nodes": n_nodes}


def synth_flow(flow_id: int, n_packets: int, t0: float = 0.0) -> list[Packet]:
    """A plausible bidirectional flow; content does not matter for a cost benchmark."""
    lan, wan = f"192.168.1.{flow_id % 250 + 2}", f"203.0.113.{flow_id % 250 + 2}"
    sport, dport = 40000 + flow_id % 20000, 443
    out, t = [], t0
    for i in range(n_packets):
        up = i % 3 == 0
        src, dst, sp, dp = (lan, wan, sport, dport) if up else (wan, lan, dport, sport)
        out.append(Packet(t, src, dst, sp, dp, 6, 80 if up else 1400))
        t += 0.0002
    return out


def bench_throughput(clf: FlowClassifier, label: str, n_flows_list=(1, 10, 50, 200, 1000), pkts_per_flow: int = 320) -> pd.DataFrame:
    """Interleave N concurrent flows through the full pipeline; measure packets/s,
    decisions/s, per-flow state memory and the decision latency seen under load."""
    rows = []
    for n_flows in n_flows_list:
        flows = [synth_flow(i, pkts_per_flow) for i in range(n_flows)]
        interleaved = [p for tup in zip(*flows) for p in tup]  # round-robin across flows
        # pass 1: timing, untraced. tracemalloc traces every allocation and
        # roughly halves throughput, so measuring both at once would report a
        # number that is an artefact of the measurement.
        tracker = FlowTracker(clf, QoSPolicy(), local_nets=["192.168.1.0/24"])
        gc.collect()
        with threadpool_limits(limits=1):
            t0 = time.perf_counter()
            n_dec = 0
            for p in interleaved:
                n_dec += len(tracker.feed(p))
            dt = time.perf_counter() - t0

        # pass 2: memory of the tracker state, traced
        tracker2 = FlowTracker(clf, QoSPolicy(), local_nets=["192.168.1.0/24"])
        gc.collect()
        tracemalloc.start()
        with threadpool_limits(limits=1):
            for p in interleaved:
                tracker2.feed(p)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        rows.append({
            "model": label,
            "concurrent flows": n_flows,
            "packets": len(interleaved),
            "decisions": n_dec,
            "packets/s": len(interleaved) / dt,
            "decisions/s": n_dec / dt,
            "us per packet": dt / len(interleaved) * 1e6,
            "us per decision": dt / max(n_dec, 1) * 1e6,
            "peak KB": peak / 1024,
            "KB per flow": peak / 1024 / n_flows,
        })
        print(f"  {label:16s} {n_flows:5d} flows: {rows[-1]['packets/s']:9.0f} pkt/s  {rows[-1]['decisions/s']:7.0f} dec/s  "
              f"{rows[-1]['us per packet']:5.1f} us/pkt  {rows[-1]['KB per flow']:6.1f} KB/flow", flush=True)
    return pd.DataFrame(rows)


def bench_flow_state_memory(n_flows: int = 5000) -> dict:
    """Memory held by the flow table alone (no classifier), at rest."""
    gc.collect()
    tracemalloc.start()
    ft = FlowTable(window=32, local_nets=["192.168.1.0/24"])
    for i in range(n_flows):
        for p in synth_flow(i, 16):  # half a window each -> all flows resident
            ft.add_packet(p)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    n = len(ft.flows)
    return {"flows_tracked": n, "peak_kb": peak / 1024, "bytes_per_flow": peak / max(n, 1)}


def main():
    clf = FlowClassifier.load()
    df = pd.read_parquet(ROOT / "data/processed/windows_w32.parquet")
    _, test_df = split_held_out(df)
    X = test_df[clf.features].to_numpy(dtype=np.float64)

    # real packet windows for the feature-extraction timing
    from src.flows import PacketObs
    rng = np.random.default_rng(0)
    windows = [[PacketObs(float(t), int(s), bool(u)) for t, s, u in
                zip(np.cumsum(rng.exponential(0.005, 32)), rng.integers(60, 1500, 32), rng.random(32) > 0.5)]
               for _ in range(2000)]

    # The deployed forest is the accuracy choice; the compact tree is the
    # router-realistic one. Both get benchmarked so the trade-off has numbers.
    import joblib

    from src.classifier import _fit, cap_windows_per_flow, make_models, xy
    Xtr, ytr, _ = xy(cap_windows_per_flow(split_held_out(df)[0]))
    tree = _fit(make_models(0)["decision_tree"], Xtr, ytr)
    tree_path = EVAL / "_tree_for_bench.joblib"
    joblib.dump(tree, tree_path, compress=3)
    models = {f"{clf.meta.get('model')} (deployed)": (clf, None),
              "decision_tree (compact)": (FlowClassifier(tree, {"features": clf.features, "window": clf.window}),
                                          tree_path.stat().st_size / 1024)}

    res: dict = {"cpu": _cpu_name(), "window": clf.window, "deployed_model": clf.meta.get("model")}
    print(f"window={clf.window} cpu={res['cpu']}\n")

    res["latency"], res["footprint"], tp_parts = {}, {}, []
    for label, (m, disk_kb) in models.items():
        print(f"latency, single thread - {label}:")
        lat = bench_latency(m, X, windows)
        res["latency"][label] = lat
        for k, v in lat.items():
            if isinstance(v, dict):
                print(f"  {k:20s} p50 {v['p50_us']:8.1f} us   p95 {v['p95_us']:8.1f}   p99 {v['p99_us']:8.1f}")
        print(f"  batched (1000 flows) {lat['batched_1000_us_per_flow']:8.1f} us/flow")
        fp = bench_footprint(m, disk_kb)
        res["footprint"][label] = fp
        print(f"  footprint: {fp['on_disk_kb']:.0f} KB disk, {fp['resident_kb']:.0f} KB resident, {fp['tree_nodes']} nodes\n")

    print("throughput under concurrent load (full pipeline, single thread):")
    for label, (m, _) in models.items():
        tp_parts.append(bench_throughput(m, label.split()[0]))
    tp = pd.concat(tp_parts, ignore_index=True)
    res["throughput"] = tp.to_dict("records")
    tree_path.unlink(missing_ok=True)

    res["flow_state"] = bench_flow_state_memory()
    print(f"\nflow table at rest: {res['flow_state']['flows_tracked']} flows, "
          f"{res['flow_state']['bytes_per_flow']:.0f} B/flow\n")

    (EVAL / "benchmark.json").write_text(json.dumps(res, indent=1))
    print(f"wrote {EVAL / 'benchmark.json'}")


def _cpu_name() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if "model name" in line:
            return line.split(":", 1)[1].strip()
    return "unknown"


if __name__ == "__main__":
    main()
