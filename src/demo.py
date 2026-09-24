"""
CLI demo: replay a pcap through the classifier in (scaled) real time and show,
per flow, the current class, confidence and simulated QoS queue, updating as
re-classification happens.

    python -m src.demo                                   # replay data/samples/demo.pcap
    python -m src.demo --pcap some.pcap --speed 4        # 4x faster than real time
    python -m src.demo --speed 0                         # as fast as possible, then print summary
    python -m src.demo --live eth0                       # live capture (needs CAP_NET_RAW)

If a sidecar `<pcap>.json` exists (written by data/make_demo_pcap.py) the
ground-truth class of each LAN host is shown next to the prediction.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from .classifier import DEFAULT_MODEL, FlowClassifier
from .flows import Packet, read_pcap
from .qos_mapper import QoSPolicy
from .reclassify import FlowEvent, FlowTracker

ROOT = Path(__file__).resolve().parents[1]
DEMO_PCAP = ROOT / "data" / "samples" / "demo.pcap"

TIER_STYLE = {"realtime": "bold red", "interactive": "bold yellow", "streaming": "bold cyan",
              "bulk": "dim", "best_effort": "white"}


def load_sidecar(pcap: Path) -> tuple[list[str] | None, dict[str, str]]:
    sc = pcap.with_suffix(".json")
    if not sc.exists():
        return None, {}
    d = json.loads(sc.read_text())
    truth = {f["lan_ip"]: f["true_class"] + (" (syn)" if f["source"] == "synthetic" else "") for f in d["flows"]}
    return d.get("local_nets"), truth


def truth_for(flow_id: str, truth: dict[str, str]) -> str:
    for ip, label in truth.items():
        if f":{ip}:" in flow_id:
            return label
    return "?"


def render(tracker: FlowTracker, truth: dict[str, str], log: list[FlowEvent], t_rel: float, n_pkts: int) -> Group:
    tbl = Table(title=f"MetaQoS  t={t_rel:6.1f}s  packets={n_pkts}", expand=True)
    for col in ("flow", "pkts", "win", "truth", "class", "conf", "tier", "DSCP", "queue", "changes"):
        tbl.add_column(col, justify="right" if col in ("pkts", "win", "conf", "queue", "changes") else "left")
    for row in sorted(tracker.snapshot(), key=lambda r: r["flow_id"]):
        style = TIER_STYLE.get(row["tier"], "")
        tbl.add_row(row["flow_id"], str(row["packets"]), str(row["windows"]), truth_for(row["flow_id"], truth),
                    row["class"], f"{row['confidence']:.2f}", f"[{style}]{row['tier']}[/]", str(row["dscp"]),
                    str(row["queue"]), str(row["changes"]))
    lines = []
    for e in log[-8:]:
        tag = "[bold]TIER CHANGE[/]" if e.changed else "first decision"
        lines.append(f"t+{e.ts:6.1f}s  {e.flow_id:44s} win {e.window_idx:3d}  raw={e.raw_label}({e.raw_confidence:.2f})"
                     f"  -> {e.label}({e.confidence:.2f})  {e.previous_tier or '-'} -> {e.decision.tier.name}  {tag}")
    return Group(tbl, Panel("\n".join(lines) or "waiting for first window...", title="decision log"))


def replay(pcap: Path, clf: FlowClassifier, speed: float, min_conf: float, alpha: float, console: Console):
    local_nets, truth = load_sidecar(pcap)
    tracker = FlowTracker(clf, QoSPolicy(min_confidence=min_conf), alpha=alpha, local_nets=local_nets)
    log: list[FlowEvent] = []
    t0_pcap = t0_wall = None
    n = 0

    def handle(p: Packet):
        nonlocal t0_pcap, t0_wall, n
        if t0_pcap is None:
            t0_pcap, t0_wall = p.ts, time.time()
        if speed > 0:
            lag = (p.ts - t0_pcap) / speed - (time.time() - t0_wall)
            if lag > 0.002:
                time.sleep(lag)
        n += 1
        for ev in tracker.feed(p):
            ev.ts -= t0_pcap
            if ev.changed or ev.window_idx == 0:
                log.append(ev)

    with Live(console=console, refresh_per_second=8, transient=False) as live:
        for p in read_pcap(str(pcap)):
            handle(p)
            if n % 50 == 0:
                live.update(render(tracker, truth, log, p.ts - t0_pcap, n))
        for ev in tracker.flush():
            ev.ts -= t0_pcap
            if ev.changed or ev.window_idx == 0:
                log.append(ev)
        live.update(render(tracker, truth, log, (p.ts - t0_pcap) if t0_pcap else 0, n))
    return tracker, truth


def summary(tracker: FlowTracker, truth: dict[str, str], console: Console):
    rows = tracker.snapshot()
    console.print("\n[bold]Final decisions[/]")
    hits = total = 0
    for r in sorted(rows, key=lambda r: r["flow_id"]):
        t = truth_for(r["flow_id"], truth)
        ok = t.split(" ")[0] == r["class"]
        if t != "?" and "(syn)" not in t:
            total += 1
            hits += ok
        mark = "" if t == "?" else (" [green]OK[/]" if ok else " [red]MISS[/]")
        console.print(f"  {r['flow_id']:44s} truth={t:18s} pred={r['class']:14s} conf={r['confidence']:.2f} "
                      f"-> {r['tier']:12s} {r['dscp']:4s} q{r['queue']}  changes={r['changes']}{mark}")
    if total:
        console.print(f"\n  real held-out flows correct: {hits}/{total}")
    console.print("\n[bold]Simulated router actions for acted-on flows[/]")
    for st in tracker.states.values():
        if st.decision and st.decision.acted:
            console.print(f"[dim]{st.decision.tc_command()}[/]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap", type=Path, default=DEMO_PCAP)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed factor; 0 = no pacing")
    ap.add_argument("--min-conf", type=float, default=0.6)
    ap.add_argument("--alpha", type=float, default=0.4, help="EMA weight of the newest window")
    ap.add_argument("--live", metavar="IFACE", help="sniff a live interface instead of a pcap")
    a = ap.parse_args()

    console = Console()
    clf = FlowClassifier.load(a.model)
    console.print(f"model: {a.model.name}  ({clf.meta['model']}, window={clf.window} pkts, classes={clf.classes})")
    if a.live:
        live_capture(a.live, clf, a.min_conf, a.alpha, console)
        return
    tracker, truth = replay(a.pcap, clf, a.speed, a.min_conf, a.alpha, console)
    summary(tracker, truth, console)


def live_capture(iface: str, clf: FlowClassifier, min_conf: float, alpha: float, console: Console):
    """Live mode via scapy; requires root or CAP_NET_RAW on the Python binary."""
    from scapy.all import IP, TCP as sTCP, UDP as sUDP, sniff  # noqa: N811

    tracker = FlowTracker(clf, QoSPolicy(min_confidence=min_conf), alpha=alpha, local_nets=["192.168.0.0/16", "10.0.0.0/8"])
    log: list[FlowEvent] = []
    t0 = time.time()

    def on_pkt(pkt):
        if IP not in pkt:
            return
        l4 = pkt[sTCP] if sTCP in pkt else pkt[sUDP] if sUDP in pkt else None
        if l4 is None:
            return
        p = Packet(float(pkt.time), pkt[IP].src, pkt[IP].dst, l4.sport, l4.dport, 6 if sTCP in pkt else 17, pkt[IP].len)
        for ev in tracker.feed(p):
            ev.ts -= t0
            if ev.changed or ev.window_idx == 0:
                log.append(ev)

    with Live(console=console, refresh_per_second=4) as live:
        while True:
            sniff(iface=iface, prn=on_pkt, store=False, timeout=1)
            live.update(render(tracker, {}, log, time.time() - t0, sum(s.n_packets for s in tracker.states.values())))


if __name__ == "__main__":
    main()
