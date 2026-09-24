"""
Per-window metadata features.

Input is a list of PacketObs (timestamp, IP length, direction) - nothing else.
Output is a fixed-length numeric vector; FEATURE_NAMES documents every column.

Feature families and why they are here
--------------------------------------
Packet size   Media codecs emit characteristic sizes: VoIP audio frames are
              small (60-200 B) and constant; video call frames are mid-size and
              variable; bulk transfer and streaming saturate the MTU (~1500 B).
Inter-arrival Real-time media is clocked (20 ms audio, 33 ms video frames), so
time (IAT)    IAT variance is low. Streaming is bursty (fill buffer, pause).
              Chat/email are sparse and irregular.
Directionality Upload:download byte ratio separates symmetric calls from
              download-heavy streaming/file transfer and upload-heavy backups.
Burstiness    Goh-Barabasi index B = (sigma - mu) / (sigma + mu) of IATs, in
              [-1, 1]: -1 perfectly periodic, 0 Poisson, -> 1 very bursty.
              Direction run-lengths capture request/response (short alternating
              runs) vs one-way bulk flow (long runs).
Rate          Duration of the window and derived packets/s, bytes/s - the window
              is a fixed packet count, so its duration encodes throughput.

Ports and addresses are deliberately NOT features (see README). The transport
protocol flag is the only header field used.
"""
from __future__ import annotations

import numpy as np

from .flows import PacketObs, UDP

SMALL_PKT = 100  # bytes; below this ~ ACKs, keep-alives, audio frames
LARGE_PKT = 1200  # bytes; above this ~ full-MTU data segments

FEATURE_NAMES: list[str] = [
    # transport / window shape
    "proto_udp",
    "n_pkts",
    "duration",
    "pkts_per_s",
    "bytes_per_s",
    # directionality
    "up_pkt_frac",
    "up_byte_frac",
    "up_bytes",
    "down_bytes",
    # packet size, all packets
    "size_mean",
    "size_std",
    "size_min",
    "size_max",
    "frac_small",
    "frac_large",
    # packet size per direction
    "up_size_mean",
    "up_size_std",
    "up_size_max",
    "down_size_mean",
    "down_size_std",
    "down_size_max",
    # inter-arrival time, all packets
    "iat_mean",
    "iat_std",
    "iat_min",
    "iat_max",
    "iat_cv",
    "burstiness",
    # inter-arrival time per direction
    "up_iat_mean",
    "up_iat_std",
    "down_iat_mean",
    "down_iat_std",
    # direction pattern
    "run_len_mean",
    "run_len_max",
    "dir_change_frac",
]

N_FEATURES = len(FEATURE_NAMES)


def _stats(x: np.ndarray) -> tuple[float, float, float, float]:
    if x.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return float(x.mean()), float(x.std()), float(x.min()), float(x.max())


def _iat(ts: np.ndarray) -> np.ndarray:
    return np.diff(ts) if ts.size >= 2 else np.empty(0)


def extract_features(packets: list[PacketObs], proto: int) -> np.ndarray:
    """Compute the feature vector for one observation window."""
    n = len(packets)
    ts = np.fromiter((p.ts for p in packets), dtype=np.float64, count=n)
    size = np.fromiter((p.size for p in packets), dtype=np.float64, count=n)
    up = np.fromiter((p.up for p in packets), dtype=bool, count=n)

    duration = float(ts[-1] - ts[0]) if n >= 2 else 0.0
    safe_dur = max(duration, 1e-3)  # avoid inf rates on a 1-packet window
    total_bytes = float(size.sum())

    up_size, down_size = size[up], size[~up]
    up_bytes, down_bytes = float(up_size.sum()), float(down_size.sum())

    s_mean, s_std, s_min, s_max = _stats(size)
    us_mean, us_std, _, us_max = _stats(up_size)
    ds_mean, ds_std, _, ds_max = _stats(down_size)

    iat = _iat(ts)
    i_mean, i_std, i_min, i_max = _stats(iat)
    iat_cv = i_std / i_mean if i_mean > 0 else 0.0
    burst = (i_std - i_mean) / (i_std + i_mean) if (i_std + i_mean) > 0 else 0.0
    ui_mean, ui_std, _, _ = _stats(_iat(ts[up]))
    di_mean, di_std, _, _ = _stats(_iat(ts[~up]))

    # direction runs: lengths of maximal same-direction packet sequences
    if n >= 2:
        changes = np.flatnonzero(up[1:] != up[:-1])
        n_changes = int(changes.size)
        bounds = np.concatenate(([-1], changes, [n - 1]))
        runs = np.diff(bounds)
        run_mean, run_max = float(runs.mean()), float(runs.max())
    else:
        n_changes, run_mean, run_max = 0, float(n), float(n)

    return np.array(
        [
            1.0 if proto == UDP else 0.0,
            n,
            duration,
            n / safe_dur,
            total_bytes / safe_dur,
            float(up.mean()) if n else 0.0,
            up_bytes / total_bytes if total_bytes else 0.0,
            up_bytes,
            down_bytes,
            s_mean,
            s_std,
            s_min,
            s_max,
            float((size <= SMALL_PKT).mean()) if n else 0.0,
            float((size >= LARGE_PKT).mean()) if n else 0.0,
            us_mean,
            us_std,
            us_max,
            ds_mean,
            ds_std,
            ds_max,
            i_mean,
            i_std,
            i_min,
            i_max,
            iat_cv,
            burst,
            ui_mean,
            ui_std,
            di_mean,
            di_std,
            run_mean,
            run_max,
            n_changes / (n - 1) if n >= 2 else 0.0,
        ],
        dtype=np.float64,
    )


def features_as_dict(vec: np.ndarray) -> dict[str, float]:
    return dict(zip(FEATURE_NAMES, vec.tolist()))
