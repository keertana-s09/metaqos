"""QoS mapping and re-classification loop behaviour, using a stub classifier."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.classifier import FlowClassifier  # noqa: E402
from src.flows import Packet, TCP, UDP  # noqa: E402
from src.qos_mapper import CLASS_TO_TIER, TIERS, QoSPolicy  # noqa: E402
from src.reclassify import FlowTracker  # noqa: E402


class StubModel:
    """Predicts from mean packet size: small -> voip_audio, big -> file_transfer."""
    classes_ = np.array(["file_transfer", "voip_audio"])

    def predict_proba(self, X):
        size_mean = X[:, 9]  # FEATURE_NAMES.index("size_mean")
        p_voip = np.clip(1 - size_mean / 1500, 0.05, 0.95)
        return np.stack([1 - p_voip, p_voip], axis=1)


def stub_clf(window=4):
    return FlowClassifier(StubModel(), {"features": [], "window": window})


def test_policy_maps_every_known_class_and_respects_floor():
    pol = QoSPolicy(min_confidence=0.6)
    for cls, tier in CLASS_TO_TIER.items():
        d = pol.decide("udp:1.1.1.1:1<->2.2.2.2:2", cls, 0.9)
        assert d.tier is TIERS[tier] and d.acted
    low = pol.decide("udp:1.1.1.1:1<->2.2.2.2:2", "video_call", 0.5)
    assert low.tier.name == "best_effort" and not low.acted
    assert pol.decide("tcp:1.1.1.1:1<->2.2.2.2:2", "martian", 0.99).tier.name == "best_effort"


def test_tc_command_encodes_dscp_and_queue():
    d = QoSPolicy().decide("udp:192.168.1.10:50000<->203.0.113.5:3478", "voip_audio", 0.9)
    cmd = d.tc_command("wan0")
    assert "--set-dscp 46" in cmd and "flowid 1:1" in cmd and "-p udp" in cmd and "dev wan0" in cmd
    assert "dsfield 0xb8" in cmd  # 46 << 2


def _pkts(n, size, up_alt=True, t0=0.0, dt=0.01):
    out = []
    for i in range(n):
        up = (i % 2 == 0) if up_alt else True
        src, dst, sp, dp = ("10.0.0.1", "8.8.4.4", 5000, 6000) if up else ("8.8.4.4", "10.0.0.1", 6000, 5000)
        out.append(Packet(t0 + i * dt, src, dst, sp, dp, UDP, size))
    return out


def test_tracker_first_decision_then_smoothed_reclassification():
    tr = FlowTracker(stub_clf(window=4), QoSPolicy(min_confidence=0.6), alpha=0.3, local_nets=["10.0.0.0/8"])
    # 2 windows of tiny packets -> voip_audio / realtime
    evs = [e for p in _pkts(8, 100) for e in tr.feed(p)]
    assert [e.window_idx for e in evs] == [0, 1]
    assert evs[0].label == "voip_audio" and evs[0].decision.tier.name == "realtime" and not evs[0].changed
    # then the flow turns into a bulk transfer: big packets
    evs2 = [e for p in _pkts(12, 1500, t0=1.0) for e in tr.feed(p)]
    assert evs2[0].raw_label == "file_transfer"
    # smoothing: the first big window is not enough to flip the tier ...
    assert evs2[0].label == "voip_audio" and not evs2[0].changed
    # ... then it passes through best-effort while confidence is below the floor
    # (realtime -> best_effort -> bulk): never a direct jump on a shaky prediction
    tiers = [e.decision.tier.name for e in evs2]
    assert tiers == ["realtime", "best_effort", "bulk"]
    assert [e.changed for e in evs2] == [False, True, True]
    snap = tr.snapshot()[0]
    assert snap["class"] == "file_transfer" and snap["tier"] == "bulk" and snap["changes"] == 2
    assert snap["windows"] == 5 and snap["packets"] == 20


def test_tracker_low_confidence_stays_best_effort():
    tr = FlowTracker(stub_clf(window=4), QoSPolicy(min_confidence=0.9), local_nets=["10.0.0.0/8"])
    evs = [e for p in _pkts(4, 750) for e in tr.feed(p)]  # p_voip = 0.5
    assert evs[0].decision.tier.name == "best_effort" and not evs[0].decision.acted
