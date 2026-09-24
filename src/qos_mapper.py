"""
Map a predicted traffic class (+ confidence) to a simulated QoS action.

Tiers follow DiffServ conventions a home router (OpenWrt/tc, or a DSCP-honouring
Wi-Fi AP via WMM access categories) would act on:

    realtime     EF  (DSCP 46)  strict-priority / low-latency queue
    interactive  AF41(DSCP 34)  high weight, short queue
    streaming    AF31(DSCP 26)  medium weight, deep buffer tolerant
    bulk         CS1 (DSCP 8)   "lower effort": only gets leftover bandwidth
    best_effort  BE  (DSCP 0)   default

Safety rule: below `min_confidence` we do NOT act on the prediction and leave
the flow in best-effort. A mis-promotion (bulk download in the EF queue) is
worse for a household than no decision, and a mis-demotion (a video call in the
bulk queue) is worse still, so the low-confidence default is the neutral queue.

This module is deliberately pure: it produces a QoSDecision and the tc /
iptables command that *would* apply it. Nothing is executed.
"""
from __future__ import annotations

from dataclasses import dataclass

# WMM (Wi-Fi Multimedia) access category, which is how a DSCP mark actually
# buys priority on a home Wi-Fi link.
_WMM = {"realtime": "AC_VO", "interactive": "AC_VI", "streaming": "AC_VI", "bulk": "AC_BK", "best_effort": "AC_BE"}


@dataclass(frozen=True)
class Tier:
    name: str
    dscp_name: str
    dscp: int
    queue: int  # tc class minor id, 1 = highest priority
    description: str

    @property
    def wmm(self) -> str:
        return _WMM[self.name]


TIERS: dict[str, Tier] = {
    "realtime": Tier("realtime", "EF", 46, 1, "strict priority, minimal latency/jitter"),
    "interactive": Tier("interactive", "AF41", 34, 2, "high weight, short queue for snappy request/response"),
    "streaming": Tier("streaming", "AF31", 26, 3, "guaranteed share, tolerates buffering delay"),
    "best_effort": Tier("best_effort", "BE", 0, 4, "default queue"),
    "bulk": Tier("bulk", "CS1", 8, 5, "lower effort: yields to everything else"),
}

# traffic class -> tier. Classes the model does not know yet (gaming, browsing)
# are listed so the policy is complete once the model learns them.
CLASS_TO_TIER: dict[str, str] = {
    "video_call": "realtime",
    "voip_audio": "realtime",
    "gaming": "realtime",
    "chat": "interactive",
    "browsing": "interactive",
    "streaming": "streaming",
    "file_transfer": "bulk",
    "email": "bulk",
    "unknown": "best_effort",
}


@dataclass(frozen=True)
class QoSDecision:
    flow_id: str
    traffic_class: str
    confidence: float
    tier: Tier
    acted: bool  # False when confidence was too low and we fell back to best-effort
    reason: str

    def tc_command(self, iface: str = "eth0") -> str:
        """The iptables/tc commands a Linux router would run for this decision."""
        proto, rest = self.flow_id.split(":", 1)
        a, b = rest.split("<->")
        ip_a, port_a = a.rsplit(":", 1)
        ip_b, port_b = b.rsplit(":", 1)
        return (
            f"iptables -t mangle -A POSTROUTING -p {proto} -s {ip_a} --sport {port_a} -d {ip_b} --dport {port_b} "
            f"-j DSCP --set-dscp {self.tier.dscp}  # {self.tier.dscp_name}\n"
            f"tc filter add dev {iface} parent 1: protocol ip prio 1 u32 match ip dsfield {self.tier.dscp << 2:#x} 0xfc "
            f"flowid 1:{self.tier.queue}"
        )


class QoSPolicy:
    def __init__(self, min_confidence: float = 0.6, class_to_tier: dict[str, str] | None = None):
        self.min_confidence = min_confidence
        self.class_to_tier = class_to_tier or CLASS_TO_TIER

    def decide(self, flow_id: str, traffic_class: str, confidence: float) -> QoSDecision:
        tier_name = self.class_to_tier.get(traffic_class, "best_effort")
        if confidence < self.min_confidence:
            return QoSDecision(flow_id, traffic_class, confidence, TIERS["best_effort"], False,
                               f"confidence {confidence:.2f} < {self.min_confidence:.2f}; leaving in best-effort")
        return QoSDecision(flow_id, traffic_class, confidence, TIERS[tier_name], True,
                           f"{traffic_class} -> {tier_name} ({TIERS[tier_name].dscp_name})")


def policy_table() -> str:
    rows = ["class          tier          DSCP       queue  WMM"]
    for c, t in CLASS_TO_TIER.items():
        tier = TIERS[t]
        rows.append(f"{c:14s} {tier.name:13s} {tier.dscp_name:4s} ({tier.dscp:2d})  {tier.queue}      {tier.wmm}")
    return "\n".join(rows)


if __name__ == "__main__":
    print(policy_table())
    d = QoSPolicy().decide("udp:192.168.1.10:50000<->203.0.113.5:3478", "video_call", 0.93)
    print("\n", d.reason, "\n", d.tc_command())
