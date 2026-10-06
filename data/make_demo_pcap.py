"""
Build the small demo pcap that the CLI/dashboard replays (fast dpkt writer).

    python data/make_demo_pcap.py            # -> data/samples/demo.pcap + demo.json
"""
from __future__ import annotations

import argparse
import json
import random
import socket
import sys
from collections import Counter
from pathlib import Path

import dpkt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.flows import MTU, canonical_key, guess_local_host, read_pcap  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "samples"

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


def dominant_flow(path: Path, local: str, max_packets: int = 400_000):
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


def synthetic_video_call(i: int, seconds: float, rng: random.Random):
    lan, wan = LAN.format(10 + i), WAN.format(10 + i)
    recs, t = [], 0.0
    while t < seconds:
        recs.append((t + rng.gauss(0, 0.001), lan, wan, 50000, 3478, 17, 1120))
        recs.append((t + 0.002 + rng.gauss(0, 0.001), wan, lan, 3478, 50000, 17, 1150))
        if int(t * 50) % 2 == 0:
            recs.append((t + 0.005, lan, wan, 50002, 3478, 17, 120))
            recs.append((t + 0.007, wan, lan, 3478, 50002, 17, 130))
        t += 1 / 30
    return sorted(recs)


def synthetic_voip_audio(i: int, seconds: float, rng: random.Random):
    lan, wan = LAN.format(10 + i), WAN.format(10 + i)
    recs, t = [], 0.0
    while t < seconds:
        recs.append((t + rng.gauss(0, 0.001), lan, wan, 50004, 3478, 17, 140))
        recs.append((t + 0.004 + rng.gauss(0, 0.001), wan, lan, 3478, 50004, 17, 145))
        t += 0.02
    return sorted(recs)


def synthetic_streaming(i: int, seconds: float, rng: random.Random):
    lan, wan = LAN.format(10 + i), WAN.format(10 + i)
    recs, t = [], 0.0
    while t < seconds:
        recs.append((t, lan, wan, 49152, 443, 6, 120))
        t += 0.005
        for k in range(25):
            recs.append((t, wan, lan, 443, 49152, 6, MTU))
            t += 0.0004
            if k % 3 == 0:
                recs.append((t, lan, wan, 49152, 443, 6, 52))
                t += 0.0002
        t += rng.uniform(1.2, 2.5)
    return sorted(recs)


def synthetic_file_transfer(i: int, seconds: float, rng: random.Random):
    lan, wan = LAN.format(10 + i), WAN.format(10 + i)
    recs, t = [], 0.0
    while t < seconds:
        for k in range(30):
            recs.append((t, wan, lan, 22, 51000, 6, MTU))
            t += 0.0003
            if k % 2 == 0:
                recs.append((t, lan, wan, 51000, 22, 6, 52))
                t += 0.0001
    return sorted(recs)


def synthetic_chat(i: int, seconds: float, rng: random.Random):
    lan, wan = LAN.format(10 + i), WAN.format(10 + i)
    recs, t = [], 0.5
    while t < seconds:
        recs.append((t, lan, wan, 52000, 443, 6, 180))
        t += 0.05
        recs.append((t, wan, lan, 443, 52000, 6, 220))
        t += rng.uniform(3.0, 8.0)
    return sorted(recs)


def synthetic_email(i: int, seconds: float, rng: random.Random):
    lan, wan = LAN.format(10 + i), WAN.format(10 + i)
    recs, t = [], 1.0
    while t < seconds:
        recs.append((t, lan, wan, 53000, 993, 6, 320))
        t += 0.02
        recs.append((t, wan, lan, 993, 53000, 6, 850))
        t += rng.uniform(8.0, 15.0)
    return sorted(recs)


def synthetic_gaming(i: int, seconds: float, rng: random.Random):
    lan, wan = LAN.format(10 + i), WAN.format(10 + i)
    recs, t = [], 0.0
    while t < seconds:
        recs.append((t + rng.gauss(0, 0.0008), lan, wan, 51234, 27015, 17, rng.randint(60, 110) + 28))
        if int(t * 60) % 2 == 0:
            recs.append((t + 0.004 + rng.gauss(0, 0.002), wan, lan, 27015, 51234, 17, rng.randint(180, 420) + 28))
        t += 1 / 60
    return sorted(recs)


def synthetic_browsing(i: int, seconds: float, rng: random.Random):
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


SYNTH_GENERATORS = {
    "video_call": synthetic_video_call,
    "voip_audio": synthetic_voip_audio,
    "streaming": synthetic_streaming,
    "file_transfer": synthetic_file_transfer,
    "chat": synthetic_chat,
    "email": synthetic_email,
}


def to_dpkt_bytes(rec: tuple, rng: random.Random) -> bytes:
    ts, src, dst, sport, dport, proto, length = rec
    src_bytes = socket.inet_aton(src)
    dst_bytes = socket.inet_aton(dst)

    if proto == 6:
        l4 = dpkt.tcp.TCP(sport=sport, dport=dport, flags=dpkt.tcp.TH_ACK | dpkt.tcp.TH_PUSH)
    else:
        l4 = dpkt.udp.UDP(sport=sport, dport=dport, ulen=max(8, length - 20))

    ip = dpkt.ip.IP(
        src=src_bytes,
        dst=dst_bytes,
        p=proto,
        len=length,
        id=rng.randint(1, 65535),
        data=l4,
    )

    eth = dpkt.ethernet.Ethernet(
        src=b"\x00\x11\x22\x33\x44\x55",
        dst=b"\x66\x77\x88\x99\xaa\xbb",
        type=dpkt.ethernet.ETH_TYPE_IP,
        data=ip,
    )
    return bytes(eth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--max-pkts", type=int, default=3000)
    ap.add_argument("--stagger", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "demo.pcap")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)

    flows = []
    all_recs = []
    base_ts = 1_700_000_000.0
    i = 0

    for fname, label in REAL_SOURCES:
        p = RAW / fname
        if p.exists():
            recs, origin = real_flow_records(i, p, a.seconds, a.max_pkts)
            src_name = "iscx:" + origin
        else:
            gen = SYNTH_GENERATORS.get(label, synthetic_chat)
            recs = gen(i, a.seconds, rng)
            src_name = "synthetic"

        start = base_ts + i * a.stagger
        for r in recs:
            all_recs.append((start + r[0], r))

        flows.append({
            "lan_ip": LAN.format(10 + i),
            "true_class": label,
            "source": src_name,
            "packets": len(recs),
            "seconds": round(recs[-1][0], 1)
        })
        print(f"{label:14s} {len(recs):6d} pkts {recs[-1][0]:5.1f}s  ({src_name})")
        i += 1

    for label, gen in (("gaming", synthetic_gaming), ("browsing", synthetic_browsing)):
        recs = gen(i, a.seconds, rng)
        start = base_ts + i * a.stagger
        for r in recs:
            all_recs.append((start + r[0], r))

        flows.append({
            "lan_ip": LAN.format(10 + i),
            "true_class": label,
            "source": "synthetic",
            "packets": len(recs),
            "seconds": round(recs[-1][0], 1)
        })
        print(f"{label:14s} {len(recs):6d} pkts {recs[-1][0]:5.1f}s  synthetic")
        i += 1

    all_recs.sort(key=lambda item: item[0])

    with open(a.out, "wb") as f:
        writer = dpkt.pcap.Writer(f)
        for ts, rec in all_recs:
            raw_pkt = to_dpkt_bytes(rec, rng)
            writer.writepkt(raw_pkt, ts=ts)

    sidecar = a.out.with_suffix(".json")
    sidecar.write_text(json.dumps({"local_nets": ["192.168.1.0/24"], "flows": flows}, indent=1))
    print(f"\nwrote {a.out} ({len(all_recs)} packets, {a.out.stat().st_size / 1e6:.2f} MB) and {sidecar.name}")


if __name__ == "__main__":
    main()
