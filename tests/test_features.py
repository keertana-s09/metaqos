"""Sanity checks for flow reconstruction and feature extraction on synthetic flows."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.flows import FlowTable, Packet, PacketObs, TCP, UDP, read_pcap  # noqa: E402
from src.feature_extraction import FEATURE_NAMES, N_FEATURES, extract_features, features_as_dict  # noqa: E402

LOCAL, REMOTE = "192.168.1.10", "203.0.113.5"


def mk(ts, size, up):
    return PacketObs(ts, size, up)


# --- feature extraction on constructed windows ------------------------------

def test_voip_like_window_is_periodic_symmetric_small():
    # 20 ms clocked audio, 160 B frames, alternating directions
    pk = [mk(i * 0.01, 160, i % 2 == 0) for i in range(40)]
    f = features_as_dict(extract_features(pk, UDP))
    assert f["proto_udp"] == 1
    assert f["n_pkts"] == 40
    assert f["size_std"] == 0
    assert abs(f["up_byte_frac"] - 0.5) < 1e-9
    assert f["burstiness"] == pytest.approx(-1.0)      # perfectly periodic
    assert f["iat_cv"] == pytest.approx(0.0, abs=1e-6)
    assert f["frac_small"] == 0 and f["frac_large"] == 0
    assert f["run_len_mean"] == 1 and f["dir_change_frac"] == 1  # strictly alternating
    assert f["pkts_per_s"] == pytest.approx(40 / 0.39)


def test_download_like_window_is_down_heavy_full_mtu():
    # 30 full-size data segments down, a small ACK up after every 2
    pk, t = [], 0.0
    for i in range(30):
        pk.append(mk(t, 1500, False)); t += 0.001
        if i % 2 == 1:
            pk.append(mk(t, 52, True)); t += 0.0005
    f = features_as_dict(extract_features(pk, TCP))
    assert f["proto_udp"] == 0
    assert f["up_byte_frac"] < 0.05
    assert f["down_size_mean"] == 1500 and f["up_size_mean"] == 52
    assert f["frac_large"] == pytest.approx(30 / 45)
    assert f["run_len_max"] == 2
    assert f["down_bytes"] == 45000


def test_chat_like_window_is_bursty_and_sparse():
    # messages every ~5 s, each a 3-packet burst 1 ms apart
    pk = []
    for m in range(8):
        base = m * 5.0
        up = m % 2 == 0
        pk += [mk(base, 120, up), mk(base + 0.001, 300, not up), mk(base + 0.002, 66, up)]
    f = features_as_dict(extract_features(pk, TCP))
    # B = (sigma-mu)/(sigma+mu): here sigma/mu ~ 1.5 -> B ~ 0.2. Anything > 0 is
    # burstier than Poisson; B -> 1 only for heavy-tailed gaps.
    assert f["burstiness"] > 0.1
    assert f["iat_cv"] > 1.0
    assert f["duration"] == pytest.approx(35.002)
    assert f["pkts_per_s"] < 1


def test_edge_cases_do_not_crash():
    assert extract_features([mk(0.0, 100, True)], TCP).shape == (N_FEATURES,)
    one_dir = [mk(i * 0.1, 500, True) for i in range(5)]
    f = features_as_dict(extract_features(one_dir, TCP))
    assert f["down_bytes"] == 0 and f["down_iat_mean"] == 0
    assert f["run_len_mean"] == 5 and f["dir_change_frac"] == 0
    assert len(FEATURE_NAMES) == N_FEATURES == len(set(FEATURE_NAMES))
    assert np.isfinite(extract_features(one_dir, TCP)).all()


# --- flow table -----------------------------------------------------------------

def _pkt(ts, src, dst, sport, dport, proto=TCP, length=100):
    return Packet(ts, src, dst, sport, dport, proto, length)


def test_flow_table_groups_both_directions_and_orients_by_local_net():
    ft = FlowTable(window=4, local_nets=["192.168.1.0/24"])
    out = []
    out += ft.add_packet(_pkt(0.0, LOCAL, REMOTE, 50000, 443, length=200))
    out += ft.add_packet(_pkt(0.1, REMOTE, LOCAL, 443, 50000, length=1400))
    out += ft.add_packet(_pkt(0.2, REMOTE, LOCAL, 443, 50000, length=1400))
    assert out == [] and len(ft.flows) == 1
    out += ft.add_packet(_pkt(0.3, LOCAL, REMOTE, 50000, 443, length=60))
    assert len(out) == 1
    w = out[0]
    assert w.index == 0 and len(w.packets) == 4
    assert [p.up for p in w.packets] == [True, False, False, True]
    assert [p.size for p in w.packets] == [200, 1400, 1400, 60]
    assert w.flow_id.startswith("tcp:")


def test_flow_table_initiator_fallback_and_windows_index():
    ft = FlowTable(window=2)  # no local nets: initiator = local
    a = [_pkt(t, REMOTE, LOCAL, 443, 50000) for t in (0.0, 0.1, 0.2, 0.3)]
    wins = [w for p in a for w in ft.add_packet(p)]
    assert [w.index for w in wins] == [0, 1]
    assert all(p.up for w in wins for p in w.packets)  # initiator (REMOTE here) counted as "up"


def test_flow_table_idle_timeout_and_ignored_ports():
    ft = FlowTable(window=2, idle_timeout=10)
    assert ft.add_packet(_pkt(0.0, LOCAL, "8.8.8.8", 4444, 53, UDP)) == []
    assert ft.flows == {}  # DNS dropped
    ft.add_packet(_pkt(0.0, LOCAL, REMOTE, 1, 2))
    ft.add_packet(_pkt(100.0, LOCAL, REMOTE, 1, 2))  # after timeout -> new flow, buffer restarts
    assert ft.flows[next(iter(ft.flows))].n_packets == 1
    assert ft.flush(min_packets=1)[0].index == 0


def test_flush_discards_short_partials():
    ft = FlowTable(window=8)
    for t in range(3):
        ft.add_packet(_pkt(float(t), LOCAL, REMOTE, 1, 2))
    assert ft.flush() == []              # 3 < 8/2
    assert len(ft.flush(min_packets=1)) == 1


# --- real pcap smoke test ----------------------------------------------------------

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "raw" / "youtube1.pcap"


@pytest.mark.skipif(not SAMPLE.exists(), reason="dataset not downloaded")
def test_real_pcap_parses_and_yields_windows():
    pkts = list(read_pcap(str(SAMPLE)))
    assert len(pkts) > 5000
    ft = FlowTable(window=32, local_nets=["131.202.0.0/16"])
    wins = list(ft.windows_from_pcap(str(SAMPLE)))
    assert len(wins) > 20
    X = np.stack([extract_features(w.packets, w.proto) for w in wins])
    assert X.shape[1] == N_FEATURES and np.isfinite(X).all()
    # a video stream should be download-dominated in at least some windows
    assert (X[:, FEATURE_NAMES.index("up_byte_frac")] < 0.2).any()


def test_oversize_packets_are_split_to_mtu(tmp_path):
    from scapy.all import IP, TCP as sTCP, Raw, wrpcap  # noqa: N811
    p = tmp_path / "gso.pcap"
    wrpcap(str(p), [IP(src=LOCAL, dst=REMOTE) / sTCP(sport=1, dport=2) / Raw(b"x" * 4000)])
    pk = list(read_pcap(str(p)))
    assert [x.length for x in pk] == [1500, 1500, 1040]
    assert pk[0].ts < pk[1].ts < pk[2].ts


@pytest.mark.skipif(not SAMPLE.exists(), reason="dataset not downloaded")
def test_guess_local_host_on_real_capture():
    from src.flows import guess_local_host
    assert guess_local_host(str(SAMPLE)) == "131.202.240.150"
