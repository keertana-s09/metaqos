"""
Streamlit dashboard: replays a pcap through the classifier and shows live
per-flow class / confidence / queue, the decision timeline and the tc commands.

    streamlit run src/dashboard.py
    streamlit run src/dashboard.py -- --pcap other.pcap
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.classifier import DEFAULT_MODEL, FlowClassifier  # noqa: E402
from src.demo import DEMO_PCAP, load_sidecar, truth_for  # noqa: E402
from src.flows import read_pcap  # noqa: E402
from src.qos_mapper import TIERS, QoSPolicy, policy_table  # noqa: E402
from src.reclassify import FlowTracker  # noqa: E402

TIER_ICON = {"realtime": "🔴", "interactive": "🟠", "streaming": "🔵", "bulk": "⚪", "best_effort": "🟢"}


def _args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap", type=Path, default=DEMO_PCAP)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    return ap.parse_known_args()[0]  # streamlit forwards everything after `--`


@st.cache_resource
def load_model(path: str) -> FlowClassifier:
    return FlowClassifier.load(path)


@st.cache_data
def load_packets(path: str):
    return list(read_pcap(path))


def main():
    st.set_page_config(page_title="MetaQoS", layout="wide")
    args = _args()
    st.title("MetaQoS: encrypted-traffic QoS from packet metadata")
    st.caption("No payload is read. Each flow is classified from packet sizes, timings, direction and burstiness, "
               "re-classified every window, and mapped to a simulated DSCP/queue.")

    with st.sidebar:
        pcap = Path(st.text_input("pcap", str(args.pcap)))
        speed = st.slider("replay speed (x real time)", 0.5, 20.0, 4.0, 0.5)
        min_conf = st.slider("confidence floor for acting", 0.3, 0.95, 0.6, 0.05)
        alpha = st.slider("EMA weight of newest window", 0.1, 1.0, 0.4, 0.1)
        st.markdown("**Policy**"); st.code(policy_table())
        run = st.button("Replay", type="primary")

    clf = load_model(str(args.model))
    m = clf.meta
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("model", m.get("model", "?"))
    c2.metric("window (pkts)", clf.window)
    c3.metric("held-out macro-F1", f"{m.get('held_out', {}).get('macro_f1', float('nan')):.3f}")
    c4.metric("latency / flow", f"{m.get('latency_1thread', m.get('latency', {})).get('single_flow_us_median', float('nan')):.0f} us")

    table_ph = st.empty()
    chart_ph = st.empty()
    log_ph = st.empty()
    cmd_ph = st.empty()
    if not run:
        st.info("Press **Replay** in the sidebar.")
        return

    local_nets, truth = load_sidecar(pcap)
    tracker = FlowTracker(clf, QoSPolicy(min_confidence=min_conf), alpha=alpha, local_nets=local_nets)
    pkts = load_packets(str(pcap))
    t0 = pkts[0].ts
    wall0 = time.time()
    timeline, log = [], []

    def refresh(t_rel: float):
        rows = tracker.snapshot()
        if rows:
            df = pd.DataFrame(rows)
            df.insert(3, "truth", [truth_for(f, truth) for f in df["flow_id"]])
            df = df.sort_values("flow_id")
            df["tier"] = [f"{TIER_ICON.get(t, '')} {t}" for t in df["tier"]]
            table_ph.dataframe(df, use_container_width=True, hide_index=True,
                               column_config={"confidence": st.column_config.ProgressColumn("confidence", min_value=0.0, max_value=1.0, format="%.2f")})
        if timeline:
            tl = pd.DataFrame(timeline)
            chart_ph.line_chart(tl.pivot_table(index="t", columns="flow", values="queue", aggfunc="last").ffill(),
                                height=220, y_label="queue (1 = highest priority)")
        log_ph.code("\n".join(log[-10:]) or "waiting for first window...", language="text")

    last_refresh = 0.0
    for i, p in enumerate(pkts):
        lag = (p.ts - t0) / speed - (time.time() - wall0)
        if lag > 0.002:
            time.sleep(lag)
        for ev in tracker.feed(p):
            t_rel = ev.ts - t0
            timeline.append({"t": round(t_rel, 1), "flow": ev.flow_id.split("<->")[0], "queue": ev.decision.tier.queue})
            if ev.changed or ev.window_idx == 0:
                log.append(f"t+{t_rel:6.1f}s {ev.flow_id:44s} win {ev.window_idx:3d} raw={ev.raw_label}({ev.raw_confidence:.2f}) "
                           f"-> {ev.label}({ev.confidence:.2f})  {ev.previous_tier or '-'} -> {ev.decision.tier.name}"
                           + ("  TIER CHANGE" if ev.changed else ""))
        if time.time() - last_refresh > 0.4:
            refresh(p.ts - t0); last_refresh = time.time()
    tracker.flush()
    refresh(pkts[-1].ts - t0)
    cmds = [st_.decision.tc_command() for st_ in tracker.states.values() if st_.decision and st_.decision.acted]
    cmd_ph.expander("simulated router commands (not executed)").code("\n\n".join(cmds), language="bash")
    st.success("replay finished")


main()
