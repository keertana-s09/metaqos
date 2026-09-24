"""
Re-classification loop: feed packets in, get QoS decisions out, and keep
revising them as each flow reveals more of itself.

Every completed observation window produces a fresh prediction. Raw per-window
predictions flicker (a video call has silent stretches that look like audio; a
stream's buffer refill looks like a bulk download), and flapping a flow between
queues is worse for QoS than a slightly late decision. So each flow keeps an
exponential moving average of the class-probability vector, and the QoS tier
is re-evaluated from the *smoothed* distribution:

    p_smooth <- alpha * p_window + (1 - alpha) * p_smooth

The first window uses the raw prediction (nothing to average yet), so a flow
gets its first QoS decision after `window` packets - a few hundred ms for a
call, tens of ms for a download.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator

import numpy as np

from .classifier import FlowClassifier
from .feature_extraction import extract_features
from .flows import FlowTable, Packet, Window, read_pcap
from .qos_mapper import QoSDecision, QoSPolicy


@dataclass
class FlowState:
    flow_id: str
    proto: int
    n_windows: int = 0
    n_packets: int = 0
    first_ts: float = 0.0
    last_ts: float = 0.0
    p_smooth: np.ndarray | None = None
    decision: QoSDecision | None = None
    history: list[tuple[int, str, float]] = field(default_factory=list)  # (window_idx, raw label, raw conf)


@dataclass
class FlowEvent:
    """Emitted every time a window completes on some flow."""
    ts: float
    flow_id: str
    window_idx: int
    raw_label: str
    raw_confidence: float
    label: str  # smoothed
    confidence: float  # smoothed
    decision: QoSDecision
    changed: bool  # did the QoS tier change with this window?
    previous_tier: str | None


class FlowTracker:
    def __init__(self, classifier: FlowClassifier, policy: QoSPolicy | None = None,
                 alpha: float = 0.4, local_nets: Iterable[str] | None = None, **flowtable_kw):
        self.clf = classifier
        self.policy = policy or QoSPolicy()
        self.alpha = alpha
        self.table = FlowTable(window=classifier.window, local_nets=local_nets, **flowtable_kw)
        self.states: dict[str, FlowState] = {}
        self.events: list[FlowEvent] = []

    # -- core ----------------------------------------------------------------
    def _classify_window(self, w: Window) -> FlowEvent:
        x = extract_features(w.packets, w.proto)
        p_raw = self.clf.predict_proba(x)[0]
        st = self.states.get(w.flow_id)
        if st is None or w.index == 0:
            st = FlowState(w.flow_id, w.proto, first_ts=w.packets[0].ts)
            self.states[w.flow_id] = st
        st.n_windows += 1
        st.n_packets += len(w.packets)
        st.last_ts = w.packets[-1].ts

        st.p_smooth = p_raw if st.p_smooth is None else self.alpha * p_raw + (1 - self.alpha) * st.p_smooth
        i_raw, i_s = int(p_raw.argmax()), int(st.p_smooth.argmax())
        raw_label, raw_conf = self.clf.classes[i_raw], float(p_raw[i_raw])
        label, conf = self.clf.classes[i_s], float(st.p_smooth[i_s])
        st.history.append((w.index, raw_label, raw_conf))

        prev_tier = st.decision.tier.name if st.decision else None
        decision = self.policy.decide(w.flow_id, label, conf)
        changed = prev_tier is not None and decision.tier.name != prev_tier
        st.decision = decision
        ev = FlowEvent(st.last_ts, w.flow_id, w.index, raw_label, raw_conf, label, conf, decision, changed, prev_tier)
        self.events.append(ev)
        return ev

    def feed(self, p: Packet) -> list[FlowEvent]:
        return [self._classify_window(w) for w in self.table.add_packet(p)]

    def flush(self) -> list[FlowEvent]:
        return [self._classify_window(w) for w in self.table.flush()]

    def replay_pcap(self, path: str) -> Iterator[FlowEvent]:
        for p in read_pcap(path):
            yield from self.feed(p)
        yield from self.flush()

    # -- reporting -----------------------------------------------------------
    def snapshot(self) -> list[dict]:
        rows = []
        for st in self.states.values():
            d = st.decision
            rows.append({
                "flow_id": st.flow_id, "packets": st.n_packets, "windows": st.n_windows,
                "class": d.traffic_class if d else "-", "confidence": d.confidence if d else 0.0,
                "tier": d.tier.name if d else "-", "dscp": d.tier.dscp_name if d else "-",
                "queue": d.tier.queue if d else "-", "acted": d.acted if d else False,
                "changes": sum(1 for e in self.events if e.flow_id == st.flow_id and e.changed),
            })
        return rows
