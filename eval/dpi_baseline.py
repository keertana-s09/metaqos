"""
DPI baseline: nDPI (via nfstream) vs MetaQoS on the same held-out flows.

    python eval/dpi_baseline.py     # -> eval/dpi_baseline.json, appends to eval/report.md

Why this comparison is not apples-to-apples in the obvious direction
-------------------------------------------------------------------
nDPI answers "which protocol/application is this?"; MetaQoS answers "which
traffic class is this, for queueing purposes?". Those coincide only when the
protocol implies the activity (SSH -> bulk transfer, RTP -> real-time media,
IMAPS -> email). When nDPI says `TLS.Google`, the flow could be YouTube
streaming, a Hangouts call, or Gmail - all different QoS classes.

So nDPI is scored the way a router vendor would actually use it, and
generously:

* Only labels whose protocol *determines* a class are mapped (see NDPI_TO_CLASS).
  Everything else (`TLS`, `HTTP`, `QUIC.Google`, `Unknown`, ...) counts as
  "no decision" - nDPI is never penalised for being wrong there, it simply
  abstains and the router falls back to best-effort.
* Scores are reported at class level AND at QoS-tier level, where nDPI gets
  full credit for `RTP`/`STUN` -> realtime without having to distinguish a
  video call from a voice call.
* Everything is reported per flow and weighted by packets, because a router
  cares about the flows that carry the traffic, not about the count of tiny
  background connections.

MetaQoS is scored on exactly the same flows, by majority vote over that flow's
windows.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.evaluate import md_table  # noqa: E402
from src.classifier import HELD_OUT_SESSIONS, FlowClassifier, split_held_out  # noqa: E402
from src.qos_mapper import CLASS_TO_TIER  # noqa: E402

EVAL = ROOT / "eval"
RAW = ROOT / "data" / "raw"

# nDPI label -> traffic class, ONLY where the protocol determines the activity.
# Absent = nDPI abstains (no QoS decision), which is never scored as an error.
NDPI_TO_CLASS: dict[str, str] = {
    "SSH": "file_transfer", "SCP": "file_transfer", "SFTP": "file_transfer",
    "FTP_CONTROL": "file_transfer", "FTP_DATA": "file_transfer", "FTPS": "file_transfer",
    "IMAPS": "email", "IMAP": "email", "POPS": "email", "POP3": "email",
    "SMTPS": "email", "SMTP": "email", "Mail_IMAPS": "email", "Mail_POPS": "email", "Mail_SMTPS": "email",
    "RTP": "voip_audio", "STUN.RTP": "voip_audio", "DTLS.SRTP": "voip_audio", "SRTP": "voip_audio",
    "TLS.YouTube": "streaming", "YouTube": "streaming", "TLS.NetFlix": "streaming", "NetFlix": "streaming",
    "Spotify": "streaming", "TLS.Spotify": "streaming", "Vimeo": "streaming", "TLS.Vimeo": "streaming",
    "YouTubeUpload": "streaming",
}
# Labels that pin the QoS tier even when the exact class is ambiguous.
NDPI_TO_TIER_ONLY: dict[str, str] = {"STUN": "realtime", "DTLS": "realtime", "Skype_TeamsCall": "realtime"}

HELD_OUT_FILES: dict[str, str] = {
    "facebook_video2a": "video_call", "hangouts_audio3": "voip_audio", "skype_audio2a": "voip_audio",
    "skype_audio2b": "voip_audio", "netflix2": "streaming", "youtube4": "streaming", "vimeo3": "streaming",
    "sftp_down_3a": "file_transfer", "sftp_down_3b": "file_transfer", "scpDown3": "file_transfer",
    "skype_file3": "file_transfer", "facebook_chat_4a": "chat", "facebook_chat_4b": "chat",
    "gmailchat2": "chat", "email2a": "email", "email2b": "email",
}

MIN_PACKETS = 32  # one MetaQoS window: the point at which either system could decide


def canonical(src_ip, src_port, dst_ip, dst_port, proto_num) -> str:
    proto = "tcp" if proto_num == 6 else "udp" if proto_num == 17 else str(proto_num)
    a, b = (src_ip, src_port), (dst_ip, dst_port)
    lo, hi = (a, b) if a <= b else (b, a)
    return f"{proto}:{lo[0]}:{lo[1]}<->{hi[0]}:{hi[1]}"


def run_ndpi() -> pd.DataFrame:
    from nfstream import NFStreamer

    rows = []
    for stem, label in HELD_OUT_FILES.items():
        path = next((p for p in (RAW / f"{stem}.pcap", RAW / f"{stem}.pcapng") if p.exists()), None)
        if path is None:
            print(f"missing {stem}; skipped")
            continue
        for fl in NFStreamer(source=str(path), statistical_analysis=False, n_dissections=20, idle_timeout=60):
            if fl.bidirectional_packets < MIN_PACKETS:
                continue
            rows.append({"file": stem, "label": label,
                         "flow_id": canonical(fl.src_ip, fl.src_port, fl.dst_ip, fl.dst_port, fl.protocol),
                         "packets": int(fl.bidirectional_packets), "bytes": int(fl.bidirectional_bytes),
                         "ndpi_app": fl.application_name, "ndpi_cat": fl.application_category_name})
        print(f"  nDPI {stem}: {sum(r['file'] == stem for r in rows)} flows >= {MIN_PACKETS} pkts", flush=True)
    return pd.DataFrame(rows)


def metaqos_per_flow(clf: FlowClassifier) -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data/processed/windows_w32.parquet")
    _, held = split_held_out(df)
    pred = clf.model.predict(held[clf.features].to_numpy(dtype=np.float64))
    held = held.assign(pred=pred)
    g = held.groupby("flow_id").agg(mq_pred=("pred", lambda s: s.value_counts().index[0]),
                                    mq_windows=("pred", "size"), label=("label", "first"))
    return g.reset_index()


def score(df: pd.DataFrame, pred_col: str, level: str) -> dict:
    """Accuracy over flows where a decision was made, plus coverage. `level` is
    'class' or 'tier'."""
    if level == "tier":
        truth = df["label"].map(lambda c: CLASS_TO_TIER.get(c, "best_effort"))
        pred = df[pred_col].map(lambda c: CLASS_TO_TIER.get(c, None) if isinstance(c, str) else None)
        pred = pred.where(df[pred_col].notna(), None)
    else:
        truth, pred = df["label"], df[pred_col]
    decided = pred.notna()
    hit = (pred == truth) & decided
    w = df["packets"]
    return {
        "flows_total": int(len(df)),
        "flows_decided": int(decided.sum()),
        "flow_coverage": float(decided.mean()),
        "packet_coverage": float(w[decided].sum() / w.sum()),
        "accuracy_when_decided": float(hit[decided].mean()) if decided.any() else 0.0,
        "packet_weighted_accuracy_when_decided": float(w[hit].sum() / w[decided].sum()) if decided.any() else 0.0,
        "accuracy_all_flows": float(hit.mean()),
        "packet_weighted_accuracy_all_flows": float(w[hit].sum() / w.sum()),
    }


def main():
    clf = FlowClassifier.load()
    print("running nDPI (nfstream) over held-out captures...")
    ndpi = run_ndpi()
    mq = metaqos_per_flow(clf)

    df = ndpi.merge(mq[["flow_id", "mq_pred", "mq_windows"]], on="flow_id", how="inner")
    print(f"\nnDPI flows: {len(ndpi)}   MetaQoS flows: {len(mq)}   joined on 5-tuple: {len(df)}")

    df["ndpi_class"] = df["ndpi_app"].map(NDPI_TO_CLASS)
    # tier-only labels: no class, but a tier
    df["ndpi_tier"] = df["ndpi_class"].map(lambda c: CLASS_TO_TIER.get(c) if isinstance(c, str) else None)
    tier_only = df["ndpi_tier"].isna() & df["ndpi_app"].isin(NDPI_TO_TIER_ONLY)
    df.loc[tier_only, "ndpi_tier"] = df.loc[tier_only, "ndpi_app"].map(NDPI_TO_TIER_ONLY)

    res = {
        "flows_compared": int(len(df)), "packets_compared": int(df["packets"].sum()),
        "held_out_sessions": HELD_OUT_SESSIONS, "min_packets": MIN_PACKETS,
        "ndpi_class": score(df, "ndpi_class", "class"),
        "metaqos_class": score(df, "mq_pred", "class"),
    }
    # tier scoring: nDPI uses its own tier column (includes tier-only labels)
    truth_tier = df["label"].map(lambda c: CLASS_TO_TIER.get(c, "best_effort"))
    nd_dec = df["ndpi_tier"].notna()
    nd_hit = (df["ndpi_tier"] == truth_tier) & nd_dec
    mq_tier = df["mq_pred"].map(lambda c: CLASS_TO_TIER.get(c, "best_effort"))
    w = df["packets"]
    res["ndpi_tier"] = {
        "flows_decided": int(nd_dec.sum()), "flow_coverage": float(nd_dec.mean()),
        "packet_coverage": float(w[nd_dec].sum() / w.sum()),
        "accuracy_when_decided": float(nd_hit[nd_dec].mean()) if nd_dec.any() else 0.0,
        "packet_weighted_accuracy_when_decided": float(w[nd_hit].sum() / w[nd_dec].sum()) if nd_dec.any() else 0.0,
        "accuracy_all_flows": float(nd_hit.mean()),
        "packet_weighted_accuracy_all_flows": float(w[nd_hit].sum() / w.sum()),
    }
    mq_hit = mq_tier == truth_tier
    res["metaqos_tier"] = {
        "flows_decided": int(len(df)), "flow_coverage": 1.0, "packet_coverage": 1.0,
        "accuracy_when_decided": float(mq_hit.mean()),
        "packet_weighted_accuracy_when_decided": float(w[mq_hit].sum() / w.sum()),
        "accuracy_all_flows": float(mq_hit.mean()),
        "packet_weighted_accuracy_all_flows": float(w[mq_hit].sum() / w.sum()),
    }

    app_tbl = (df.groupby("ndpi_app").agg(flows=("packets", "size"), packets=("packets", "sum"),
                                          mapped=("ndpi_class", lambda s: s.notna().sum()))
               .sort_values("packets", ascending=False).reset_index())
    app_tbl["% of packets"] = (app_tbl["packets"] / app_tbl["packets"].sum() * 100).round(1)
    res["ndpi_apps"] = app_tbl.to_dict("records")

    summary = pd.DataFrame([
        {"system": "nDPI (class)", **{k: res["ndpi_class"][k] for k in
            ("flow_coverage", "packet_coverage", "accuracy_when_decided", "packet_weighted_accuracy_all_flows")}},
        {"system": "nDPI (QoS tier)", **{k: res["ndpi_tier"][k] for k in
            ("flow_coverage", "packet_coverage", "accuracy_when_decided", "packet_weighted_accuracy_all_flows")}},
        {"system": "MetaQoS (class)", **{k: res["metaqos_class"][k] for k in
            ("flow_coverage", "packet_coverage", "accuracy_when_decided", "packet_weighted_accuracy_all_flows")}},
        {"system": "MetaQoS (QoS tier)", **{k: res["metaqos_tier"][k] for k in
            ("flow_coverage", "packet_coverage", "accuracy_when_decided", "packet_weighted_accuracy_all_flows")}},
    ]).rename(columns={"flow_coverage": "flows decided", "packet_coverage": "packets covered",
                       "accuracy_when_decided": "acc when decided",
                       "packet_weighted_accuracy_all_flows": "packet-weighted acc, all flows"})
    print("\n" + summary.to_string(index=False))
    print("\ntop nDPI labels by packet volume:\n" + app_tbl.head(12).to_string(index=False))

    (EVAL / "dpi_baseline.json").write_text(json.dumps(res, indent=1, default=str))
    _append_report(res, summary, app_tbl)
    print(f"\nwrote {EVAL / 'dpi_baseline.json'} and the DPI section of report.md")


def _append_report(res, summary, app_tbl):
    nd_c, nd_t, mq_t = res["ndpi_class"], res["ndpi_tier"], res["metaqos_tier"]
    sec = [
        "## DPI baseline: nDPI vs MetaQoS on the same flows",
        "",
        f"nDPI 4.x via nfstream, run over the same held-out captures, on the {res['flows_compared']} flows "
        f"(>= {res['min_packets']} packets) that both systems see, carrying {res['packets_compared']:,} packets.",
        "",
        "nDPI is scored generously: only labels whose protocol *determines* an activity are mapped "
        "(`SSH`->file transfer, `IMAPS`/`POPS`/`SMTPS`->email, `RTP`/`DTLS.SRTP`->real-time, "
        "`TLS.YouTube`/`TLS.NetFlix`->streaming). Generic labels (`TLS`, `HTTP`, `QUIC.Google`, `Unknown`) count as "
        "**abstentions, never as errors**. The tier row additionally gives nDPI full credit for `STUN`/`DTLS`->realtime "
        "without distinguishing a video call from a voice call.",
        "",
        md_table(summary), "",
        f"**The gap is coverage, not correctness.** When nDPI commits to a class it is right "
        f"{nd_c['accuracy_when_decided']:.1%} of the time - but it commits on only {nd_c['flow_coverage']:.1%} of flows, "
        f"carrying {nd_c['packet_coverage']:.1%} of the packets. Even at tier level, where it is scored most "
        f"favourably, it covers {nd_t['packet_coverage']:.1%} of packets. MetaQoS decides on every flow and gets "
        f"{mq_t['packet_weighted_accuracy_all_flows']:.1%} of packets into the right queue.",
        "",
        "The reason is visible in what nDPI actually returns on this traffic:",
        "",
        md_table(app_tbl.head(12)), "",
        "`TLS`, `TLS.Google` and `TLS.Facebook` identify the vendor, not the activity: the same label covers a "
        "Hangouts video call, a YouTube stream and a Gmail chat, which belong in three different queues. That is "
        "the structural limit metadata classification is designed around - and on 2016 traffic nDPI still has "
        "cleartext SNI and certificates to work with. On modern traffic with encrypted ClientHello its coverage "
        "would be lower, not higher.",
        "",
        "Caveats: nDPI is doing a harder and more general job than MetaQoS (it identifies hundreds of protocols "
        "with no training data, and it is right about the protocol far more often than this table's class mapping "
        "credits it for). This compares the two only as sources of a QoS queueing decision.",
        "",
    ]
    report = EVAL / "report.md"
    txt = report.read_text().split("\n## DPI baseline")[0].rstrip()
    report.write_text(txt + "\n\n" + "\n".join(sec))


if __name__ == "__main__":
    main()
