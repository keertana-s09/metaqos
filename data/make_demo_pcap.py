"""
Build the small demo pcap that the CLI/dashboard replays.

    python data/make_demo_pcap.py            # -> data/samples/demo.pcap + demo_flows.json

What goes in
------------
* One real flow per class, taken from HELD-OUT sessions only (the trained model
  has never seen them). We take the dominant flow of each capture, keep its
  first `--seconds` seconds, and rebuild every packet from metadata alone:
  same timestamps, same sizes, same direction, but IPs remapped to
  documentation ranges (192.168.1.x LAN, 203.0.113.x WAN) and payload replaced
  by zeros. The sample therefore contains none of the original bytes.
* Two synthetic flows for classes the training data lacks - an online-game-like
  UDP flow and a web-browsing-like TCP flow. They are labelled `synthetic` in
  the sidecar so the demo can show honestly how the model treats traffic it was
  never trained on.

All flows are shifted to start within the first few seconds so they overlap,
like concurrent devices behind one router.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from scapy.all import IP, TCP, UDP, Ether, wrpcap  # noqa: N811

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.flows import MTU, canonical_key, guess_local_host, read_pcap  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "samples"

# (held-out capture file, true class). Each must be in classifier.HELD_OUT_SESSIONS.
REAL_SOURCES = [
    ("facebook_video2a.pcap", "video_call"),
    ("hangouts_audio3.pcapng", "voip_audio"),
    ("netflix2.pcap", "streaming"),
    ("sftp_down_3a.pcap", "file_transfer"),
    ("facebook_chat_4a.pcap", "chat"),
    ("email2a.pcap", "email"),
]

LAN = "192.168.1.{}"
WAN = "203.0.113.{}"
HDR = {"tcp": 20 + 20, "udp": 20 + 8}  # IPv4 + transport header bytes


def dominant_flow(path: Path, local: str, max_packets: int = 400_000):
    """Pick the flow with the most packets in the first `max_packets`, then re-read
    only that flow's packets."""
    counts: Counter = Counter()
    for i, p in enumerate(read_pcap(str(path))):
        if p.sport in (53, 5353, 123) or p.dport in (53, 5353, 123):
            continue
        counts[canonical_key(p)[0]] += 1
        if i >= max_packets:
            break
    key = counts.most_common(1)[0][0]
    pkts = []
    for p in read_pcap(str(path)):
        k, _ = canonical_key(p)
        if k == key:
            pkts.append(p)
    return key, pkts


def real_flow_records(i: int, path: Path, seconds: float, max_pkts: int):
    local = guess_local_host(str(path))
    key, pkts = dominant_flow(path, local)
    # take the densest `seconds`-long span of the flow (sparse apps like email
    # may have only a handful of packets in their first 45 s)
    ts = [p.ts for p in pkts]
    best, j, start = 0, 0, 0
    for k in range(len(ts)):
        while ts[j] < ts[k] - seconds:
            j += 1
        if k - j + 1 > best:
            best, start = k - j + 1, j
    pkts = pkts[start:start + best][:max_pkts]
    t0 = pkts[0].ts
    lan, wan = LAN.format(10 + i), WAN.format(10 + i)
    recs = []
    for p in pkts:
        up = p.src == local
        src, dst = (lan, wan) if up else (wan, lan)
        recs.append((p.ts - t0, src, dst, p.sport, p.dport, p.proto, p.length))
    return recs, f"{path.name}:{key[0]}:{key[2]}<->{key[1]}:{key[3]}"


def synthetic_gaming(i: int, seconds: float, rng: random.Random):
    """Fast-paced online game: ~60 Hz client updates (small, constant), ~30 Hz
    server snapshots (larger, mildly variable), UDP, low jitter."""
    lan, wan = LAN.format(10 + i), WAN.format(10 + i)
    recs, t = [], 0.0
    while t < seconds:
        recs.append((t + rng.gauss(0, 0.0008), lan, wan, 51234, 27015, 17, rng.randint(60, 110) + 28))
        if int(t * 60) % 2 == 0:
            recs.append((t + 0.004 + rng.gauss(0, 0.002), wan, lan, 27015, 51234, 17, rng.randint(180, 420) + 28))
        t += 1 / 60
    return sorted(recs)


def synthetic_browsing(i: int, seconds: float, rng: random.Random):
    """Web browsing over TLS: a page load = a few small requests up, a burst of
    full-size responses down (with ACKs up), then a reading pause of 2-10 s."""
    lan, wan = LAN.format(10 + i), WAN.format(10 + i)
    recs, t, sport = [], 0.5, 44000
    while t < seconds:
        n_req = rng.randint(2, 6)
        for _ in range(n_req):
            recs.append((t, lan, wan, sport, 443, 6, rng.randint(200, 700) + 40))
            t += rng.uniform(0.005, 0.03)
            n_resp = rng.randint(3, 25)
            for k in range(n_resp):
                recs.append((t, wan, lan, 443, sport, 6, MTU if k < n_resp - 1 else rng.randint(300, 1400)))
                t += rng.uniform(0.0005, 0.003)
                if k % 2 == 1:
                    recs.append((t, lan, wan, sport, 443, 6, 52))
                    t += 0.0003
        t += rng.uniform(2, 10)
    return sorted(recs)


def to_packet(rec, base_ts: float):
    """Header-only packet whose IP length field and pcap wire length carry the
    real size, like `tcpdump -s 64`: no payload bytes are written at all."""
    ts, src, dst, sport, dport, proto, length = rec
    l4 = TCP(sport=sport, dport=dport, flags="PA") if proto == 6 else UDP(sport=sport, dport=dport)
    if proto == 17:
        l4.len = max(8, length - 20)
    pkt = Ether() / IP(src=src, dst=dst, len=max(length, HDR["tcp" if proto == 6 else "udp"])) / l4
    pkt.time = base_ts + ts
    pkt.wirelen = 14 + int(pkt[IP].len)
    return pkt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=45.0, help="length of each real flow excerpt")
    ap.add_argument("--max-pkts", type=int, default=6000, help="cap per real flow")
    ap.add_argument("--stagger", type=float, default=2.0, help="seconds between flow starts")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "demo.pcap")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)

    flows, all_pkts = [], []
    base = 1_700_000_000.0
    i = 0
    for fname, label in REAL_SOURCES:
        p = RAW / fname
        if not p.exists():
            print(f"missing {p.name}; run data/fetch_iscx.py first")
            continue
        recs, origin = real_flow_records(i, p, a.seconds, a.max_pkts)
        start = base + i * a.stagger
        all_pkts += [to_packet(r, start) for r in recs]
        flows.append({"lan_ip": LAN.format(10 + i), "true_class": label, "source": "iscx:" + origin,
                      "packets": len(recs), "seconds": round(recs[-1][0], 1)})
        print(f"{label:14s} {len(recs):6d} pkts {recs[-1][0]:5.1f}s  from {origin}")
        i += 1
    for label, gen in (("gaming", synthetic_gaming), ("browsing", synthetic_browsing)):
        recs = gen(i, a.seconds, rng)
        start = base + i * a.stagger
        all_pkts += [to_packet(r, start) for r in recs]
        flows.append({"lan_ip": LAN.format(10 + i), "true_class": label, "source": "synthetic",
                      "packets": len(recs), "seconds": round(recs[-1][0], 1)})
        print(f"{label:14s} {len(recs):6d} pkts {recs[-1][0]:5.1f}s  synthetic")
        i += 1

    all_pkts.sort(key=lambda p: p.time)
    wrpcap(str(a.out), all_pkts)
    sidecar = a.out.with_suffix(".json")
    sidecar.write_text(json.dumps({"local_nets": ["192.168.1.0/24"], "flows": flows}, indent=1))
    print(f"\nwrote {a.out} ({len(all_pkts)} packets, {a.out.stat().st_size / 1e6:.1f} MB) and {sidecar.name}")


if __name__ == "__main__":
    main()
