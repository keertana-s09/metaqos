"""
Flow reconstruction: turn a stream of packets into bidirectional flows keyed by
the 5-tuple (ip_a, ip_b, port_a, port_b, proto), and cut each flow into
fixed-size packet windows that become classifier samples.

Only packet *metadata* is ever touched: timestamp, IP addresses/ports (for
grouping only - they are never used as features), transport protocol and
on-the-wire length. Payload bytes are never read.

Design notes
------------
* Streaming, not batch. `FlowTable.add_packet` returns any windows completed by
  that packet, so the same code serves offline dataset building (millions of
  packets, bounded memory) and the live re-classification loop.
* Direction is "up" (LAN -> WAN) or "down". A real router knows which interface
  a packet arrived on; for pcaps we approximate with `local_nets` (CIDRs of the
  capture host) and fall back to "the flow initiator is the local side".
* Windows are counted in packets, not seconds, so a window always has the same
  number of observations regardless of rate; the window's duration is itself a
  feature that carries the rate information.
"""
from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass, field
from typing import Iterable, Iterator

import dpkt

TCP, UDP = 6, 17

# Captures taken on an end host often contain "super-packets" of 16-64 KB: the
# kernel hands the NIC one big segment and the NIC splits it (TSO/GSO/LRO) after
# the capture point. A router on the wire only ever sees MTU-sized packets, so
# we split oversize records into MTU-sized pieces to model the wire faithfully.
MTU = 1500
_SPLIT_GAP = 1.2e-5  # ~serialisation time of a 1500 B frame at 1 Gbit/s

# ---------------------------------------------------------------------------
# Packet records
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Packet:
    ts: float
    src: str
    dst: str
    sport: int
    dport: int
    proto: int  # 6 = TCP, 17 = UDP
    length: int  # bytes on the wire at the IP layer (header + payload)


def _decode_ip(ip) -> tuple[str, str, int, int, int, int] | None:
    if isinstance(ip, dpkt.ip.IP):
        src, dst = socket.inet_ntoa(ip.src), socket.inet_ntoa(ip.dst)
        length = ip.len
    elif isinstance(ip, dpkt.ip6.IP6):
        src = socket.inet_ntop(socket.AF_INET6, ip.src)
        dst = socket.inet_ntop(socket.AF_INET6, ip.dst)
        length = ip.plen + 40
    else:
        return None
    l4 = ip.data
    if isinstance(l4, dpkt.tcp.TCP):
        proto = TCP
    elif isinstance(l4, dpkt.udp.UDP):
        proto = UDP
    else:
        return None
    return src, dst, l4.sport, l4.dport, proto, length


def read_pcap(path: str) -> Iterator[Packet]:
    """Yield TCP/UDP packets from a .pcap or .pcapng file. Non-IP frames are skipped."""
    with open(path, "rb") as f:
        magic = f.read(4)
        f.seek(0)
        if magic == b"\x0a\x0d\x0d\x0a":
            reader = dpkt.pcapng.Reader(f)
        else:
            reader = dpkt.pcap.Reader(f)
        link = reader.datalink()
        if link == dpkt.pcap.DLT_EN10MB:
            frame_cls = dpkt.ethernet.Ethernet
        elif link == dpkt.pcap.DLT_LINUX_SLL:
            frame_cls = dpkt.sll.SLL
        elif link in (dpkt.pcap.DLT_RAW, 101, 228, 229):  # raw IP, DLT_IPV4, DLT_IPV6
            frame_cls = None
        else:
            raise ValueError(f"unsupported link type {link} in {path}")

        for ts, buf in reader:
            try:
                if frame_cls is None:
                    ip = dpkt.ip.IP(buf) if (buf[0] >> 4) == 4 else dpkt.ip6.IP6(buf)
                else:
                    ip = frame_cls(buf).data
            except (dpkt.UnpackError, IndexError):
                continue
            rec = _decode_ip(ip)
            if rec is None:
                continue
            src, dst, sport, dport, proto, length = rec
            if length <= MTU:
                yield Packet(float(ts), src, dst, sport, dport, proto, length)
                continue
            # split an offloaded super-packet into wire-sized segments
            n_full, rem = divmod(length, MTU)
            for i in range(n_full):
                yield Packet(float(ts) + i * _SPLIT_GAP, src, dst, sport, dport, proto, MTU)
            if rem:
                yield Packet(float(ts) + n_full * _SPLIT_GAP, src, dst, sport, dport, proto, rem)


def guess_local_host(path: str, n_packets: int = 20000) -> str | None:
    """Return the IP that takes part in the most packets among the first
    `n_packets`. On a single-host capture that is the capturing machine, which
    plays the role of the router's LAN side."""
    from collections import Counter

    counts: Counter[str] = Counter()
    for i, p in enumerate(read_pcap(path)):
        counts[p.src] += 1
        counts[p.dst] += 1
        if i >= n_packets:
            break
    return counts.most_common(1)[0][0] if counts else None


# ---------------------------------------------------------------------------
# Flow table
# ---------------------------------------------------------------------------

FlowKey = tuple[str, str, int, int, int]


def canonical_key(p: Packet) -> tuple[FlowKey, bool]:
    """Order the two endpoints so both directions map to the same key.
    Returns (key, forward) where forward=True means the packet goes a -> b."""
    a, b = (p.src, p.sport), (p.dst, p.dport)
    if a <= b:
        return (p.src, p.dst, p.sport, p.dport, p.proto), True
    return (p.dst, p.src, p.dport, p.sport, p.proto), False


@dataclass(slots=True)
class PacketObs:
    """What the classifier is allowed to see about one packet."""
    ts: float
    size: int
    up: bool  # True = local -> remote


@dataclass(slots=True)
class Flow:
    key: FlowKey
    first_ts: float
    last_ts: float
    a_is_local: bool  # is endpoint "a" of the canonical key the local host?
    n_packets: int = 0
    n_windows: int = 0
    buffer: list[PacketObs] = field(default_factory=list)

    @property
    def flow_id(self) -> str:
        a, b, pa, pb, proto = self.key
        name = "tcp" if proto == TCP else "udp"
        return f"{name}:{a}:{pa}<->{b}:{pb}"


@dataclass(slots=True)
class Window:
    """A completed observation window: `size` consecutive packets of one flow."""
    flow_id: str
    key: FlowKey
    index: int  # 0 = first window of the flow
    packets: list[PacketObs]
    proto: int


class FlowTable:
    """Streaming 5-tuple flow assembler that emits fixed-size packet windows.

    Parameters
    ----------
    window : packets per observation window.
    local_nets : CIDRs of the local (LAN) side, used to orient direction. If a
        flow touches none of them, the initiator is treated as local.
    idle_timeout : seconds of silence after which a flow is closed. A packet
        arriving later starts a *new* flow (new window index 0), which is what
        a router's connection tracker would also do.
    ignore_ports : transport ports whose flows are dropped entirely. Defaults to
        infrastructure chatter (DNS, NTP, mDNS, SSDP, DHCP) that appears in every
        capture regardless of application and would only add label noise.
    """

    DEFAULT_IGNORE_PORTS = frozenset({53, 5353, 123, 1900, 67, 68, 137, 138, 5355})

    def __init__(
        self,
        window: int = 32,
        local_nets: Iterable[str] | None = None,
        idle_timeout: float = 60.0,
        ignore_ports: Iterable[int] | None = None,
    ):
        self.window = window
        self.idle_timeout = idle_timeout
        self.local_nets = [ipaddress.ip_network(n) for n in (local_nets or [])]
        self.ignore_ports = frozenset(ignore_ports) if ignore_ports is not None else self.DEFAULT_IGNORE_PORTS
        self.flows: dict[FlowKey, Flow] = {}

    # -- helpers -----------------------------------------------------------
    def _is_local(self, ip: str) -> bool | None:
        if not self.local_nets:
            return None
        addr = ipaddress.ip_address(ip)
        return any(addr in n for n in self.local_nets)

    def _orient(self, p: Packet, forward: bool) -> bool:
        """Decide whether endpoint 'a' of the canonical key is the local side."""
        src_local = self._is_local(p.src)
        dst_local = self._is_local(p.dst)
        if src_local and not dst_local:
            src_is_local = True
        elif dst_local and not src_local:
            src_is_local = False
        else:
            src_is_local = True  # unknown: treat the initiator as local
        return src_is_local if forward else not src_is_local

    # -- main API ----------------------------------------------------------
    def add_packet(self, p: Packet) -> list[Window]:
        if p.sport in self.ignore_ports or p.dport in self.ignore_ports:
            return []
        key, forward = canonical_key(p)
        flow = self.flows.get(key)
        if flow is not None and p.ts - flow.last_ts > self.idle_timeout:
            flow = None  # expired: start over
        if flow is None:
            flow = Flow(key=key, first_ts=p.ts, last_ts=p.ts, a_is_local=self._orient(p, forward))
            self.flows[key] = flow

        src_is_a = forward
        up = src_is_a == flow.a_is_local
        flow.buffer.append(PacketObs(p.ts, p.length, up))
        flow.n_packets += 1
        flow.last_ts = p.ts

        out: list[Window] = []
        if len(flow.buffer) >= self.window:
            out.append(Window(flow.flow_id, key, flow.n_windows, flow.buffer, key[4]))
            flow.n_windows += 1
            flow.buffer = []
        return out

    def flush(self, min_packets: int | None = None) -> list[Window]:
        """Emit partial windows for all flows (end of capture). Windows shorter
        than `min_packets` are discarded (default: half a window)."""
        min_packets = self.window // 2 if min_packets is None else min_packets
        out = []
        for flow in self.flows.values():
            if len(flow.buffer) >= min_packets:
                out.append(Window(flow.flow_id, flow.key, flow.n_windows, flow.buffer, flow.key[4]))
                flow.n_windows += 1
                flow.buffer = []
        return out

    def windows_from_pcap(self, path: str, flush: bool = True) -> Iterator[Window]:
        for p in read_pcap(path):
            yield from self.add_packet(p)
        if flush:
            yield from self.flush()
