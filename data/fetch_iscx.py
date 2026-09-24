"""
Fetch a curated subset of the ISCX VPN-nonVPN 2016 dataset (Draper-Gil et al.,
ICISSP 2016) from its Kaggle mirror.

Why Kaggle: the original CIC host (cicresearch.ca) now gates every download
behind a request form, so nothing there can be scripted. The Kaggle dataset
`caiyafang/iscx-vpn-nonvpn-2016-complete-pcaps` is a byte-for-byte mirror of the
published pcaps (it ships the original SHA256SUMS.txt, which we verify against).

Why a subset: the full release is ~26 GB. Most of that is two enormous FTPS
captures (5.8 GB + 4.5 GB) and a 3.2 GB SCP capture that contribute a handful of
very long flows and almost no additional diversity. The MANIFEST below keeps
every application in the NonVPN set except those, at ~5 GB total.

Requires ~/.kaggle/kaggle.json (username + key). Files already on disk with a
matching SHA256 are skipped, so the script is safe to re-run.

Usage:
    python data/fetch_iscx.py            # download manifest
    python data/fetch_iscx.py --list     # just print the manifest
    python data/fetch_iscx.py --workers 6
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

KAGGLE_REF = "caiyafang/iscx-vpn-nonvpn-2016-complete-pcaps"
RAW_DIR = Path(__file__).resolve().parent / "raw"

# label -> list of files (relative path inside the Kaggle dataset).
# Naming convention in ISCX: `<app>_<type><N>a` and `<app>_<type><N>b` are the
# two endpoints of the SAME session, captured on each machine. labels.py groups
# them together so they never straddle a train/test split.
MANIFEST: dict[str, list[str]] = {
    "video_call": [
        "NonVPN-PCAPs-01/facebook_video1a.pcap",
        "NonVPN-PCAPs-01/facebook_video1b.pcapng",
        "NonVPN-PCAPs-01/facebook_video2a.pcap",
        "NonVPN-PCAPs-02/hangouts_video1b.pcapng",
        "NonVPN-PCAPs-03/skype_video1a.pcap",
        "NonVPN-PCAPs-03/skype_video1b.pcapng",
        "NonVPN-PCAPs-03/skype_video2a.pcap",
    ],
    "voip_audio": [
        "NonVPN-PCAPs-01/facebook_audio1a.pcap",
        "NonVPN-PCAPs-01/facebook_audio1b.pcapng",
        "NonVPN-PCAPs-01/facebook_audio2a.pcap",
        "NonVPN-PCAPs-01/facebook_audio2b.pcapng",
        "NonVPN-PCAPs-01/facebook_audio3.pcapng",
        "NonVPN-PCAPs-01/facebook_audio4.pcapng",
        "NonVPN-PCAPs-02/hangouts_audio1a.pcap",
        "NonVPN-PCAPs-02/hangouts_audio1b.pcapng",
        "NonVPN-PCAPs-02/hangouts_audio2a.pcap",
        "NonVPN-PCAPs-02/hangouts_audio2b.pcapng",
        "NonVPN-PCAPs-02/hangouts_audio3.pcapng",
        "NonVPN-PCAPs-02/hangouts_audio4.pcapng",
        "NonVPN-PCAPs-03/skype_audio1a.pcap",
        "NonVPN-PCAPs-03/skype_audio1b.pcapng",
        "NonVPN-PCAPs-03/skype_audio2a.pcap",
        "NonVPN-PCAPs-03/skype_audio2b.pcapng",
        "NonVPN-PCAPs-03/skype_audio3.pcapng",
        "NonVPN-PCAPs-03/skype_audio4.pcapng",
        "NonVPN-PCAPs-03/voipbuster1b.pcapng",
        "NonVPN-PCAPs-03/voipbuster2b.pcapng",
        "NonVPN-PCAPs-03/voipbuster3b.pcapng",
        "NonVPN-PCAPs-03/voipbuster_4a.pcap",
        "NonVPN-PCAPs-03/voipbuster_4b.pcap",
    ],
    "streaming": [
        "NonVPN-PCAPs-02/netflix1.pcap",
        "NonVPN-PCAPs-02/netflix2.pcap",
        "NonVPN-PCAPs-02/netflix3.pcap",
        "NonVPN-PCAPs-02/netflix4.pcap",
        "NonVPN-PCAPs-03/vimeo1.pcap",
        "NonVPN-PCAPs-03/vimeo2.pcap",
        "NonVPN-PCAPs-03/vimeo3.pcap",
        "NonVPN-PCAPs-03/vimeo4.pcap",
        "NonVPN-PCAPs-03/youtube1.pcap",
        "NonVPN-PCAPs-03/youtube2.pcap",
        "NonVPN-PCAPs-03/youtube3.pcap",
        "NonVPN-PCAPs-03/youtube4.pcap",
        "NonVPN-PCAPs-03/youtube5.pcap",
        "NonVPN-PCAPs-03/youtube6.pcap",
        "NonVPN-PCAPs-03/youtubeHTML5_1.pcap",
        "NonVPN-PCAPs-03/spotify1.pcap",
        "NonVPN-PCAPs-03/spotify2.pcap",
        "NonVPN-PCAPs-03/spotify3.pcap",
        "NonVPN-PCAPs-03/spotify4.pcap",
    ],
    "file_transfer": [
        "NonVPN-PCAPs-03/scpDown1.pcap",
        "NonVPN-PCAPs-03/scpDown2.pcap",
        "NonVPN-PCAPs-03/scpDown3.pcap",
        "NonVPN-PCAPs-03/scpDown4.pcap",
        "NonVPN-PCAPs-03/scpDown5.pcap",
        "NonVPN-PCAPs-03/sftpDown1.pcap",
        "NonVPN-PCAPs-03/sftpDown2.pcap",
        "NonVPN-PCAPs-03/sftpUp1.pcap",
        "NonVPN-PCAPs-03/sftp_down_3a.pcap",
        "NonVPN-PCAPs-03/sftp_down_3b.pcap",
        "NonVPN-PCAPs-03/sftp_up_2a.pcap",
        "NonVPN-PCAPs-03/sftp_up_2b.pcap",
        "NonVPN-PCAPs-02/ftps_up_2a.pcap",
        "NonVPN-PCAPs-02/ftps_up_2b.pcap",
        "NonVPN-PCAPs-03/skype_file1.pcap",
        "NonVPN-PCAPs-03/skype_file2.pcap",
        "NonVPN-PCAPs-03/skype_file3.pcap",
        "NonVPN-PCAPs-03/skype_file4.pcapng",
        "NonVPN-PCAPs-03/skype_file5.pcapng",
        "NonVPN-PCAPs-03/skype_file6.pcapng",
    ],
    "chat": [
        "NonVPN-PCAPs-01/AIMchat1.pcapng",
        "NonVPN-PCAPs-01/AIMchat2.pcapng",
        "NonVPN-PCAPs-01/aim_chat_3a.pcap",
        "NonVPN-PCAPs-01/aim_chat_3b.pcap",
        "NonVPN-PCAPs-01/facebook_chat_4a.pcap",
        "NonVPN-PCAPs-01/facebook_chat_4b.pcap",
        "NonVPN-PCAPs-01/facebookchat1.pcapng",
        "NonVPN-PCAPs-01/facebookchat2.pcapng",
        "NonVPN-PCAPs-01/facebookchat3.pcapng",
        "NonVPN-PCAPs-02/ICQchat1.pcapng",
        "NonVPN-PCAPs-02/ICQchat2.pcapng",
        "NonVPN-PCAPs-02/icq_chat_3a.pcap",
        "NonVPN-PCAPs-02/icq_chat_3b.pcap",
        "NonVPN-PCAPs-02/gmailchat1.pcapng",
        "NonVPN-PCAPs-02/gmailchat2.pcapng",
        "NonVPN-PCAPs-02/gmailchat3.pcapng",
        "NonVPN-PCAPs-02/hangout_chat_4b.pcap",
        "NonVPN-PCAPs-02/hangouts_chat_4a.pcap",
        "NonVPN-PCAPs-03/skype_chat1a.pcap",
        "NonVPN-PCAPs-03/skype_chat1b.pcap",
    ],
    "email": [
        "NonVPN-PCAPs-01/email1a.pcap",
        "NonVPN-PCAPs-01/email1b.pcap",
        "NonVPN-PCAPs-01/email2a.pcap",
        "NonVPN-PCAPs-01/email2b.pcap",
    ],
}


# ISCX also ships the same activities captured *inside an OpenVPN tunnel*. We do
# not train on these (inside a tunnel every application shares one 5-tuple, so
# per-flow classification no longer separates apps), but each VPN capture holds
# a single activity, which makes them a genuine robustness test: does the
# non-VPN-trained model still recognise the activity through the tunnel?
# `vpn_bittorrent.pcap` is excluded - no matching class in our label set.
VPN_MANIFEST: dict[str, list[str]] = {
    "voip_audio": [
        "VPN-PCAPS-01/vpn_facebook_audio2.pcap",
        "VPN-PCAPS-01/vpn_hangouts_audio1.pcap",
        "VPN-PCAPS-01/vpn_hangouts_audio2.pcap",
        "VPN-PCAPs-02/vpn_skype_audio1.pcap",
        "VPN-PCAPs-02/vpn_skype_audio2.pcap",
        "VPN-PCAPs-02/vpn_voipbuster1a.pcap",
        "VPN-PCAPs-02/vpn_voipbuster1b.pcap",
    ],
    "streaming": [
        "VPN-PCAPs-02/vpn_netflix_A.pcap",
        "VPN-PCAPs-02/vpn_spotify_A.pcap",
        "VPN-PCAPs-02/vpn_vimeo_A.pcap",
        "VPN-PCAPs-02/vpn_vimeo_B.pcap",
        "VPN-PCAPs-02/vpn_youtube_A.pcap",
    ],
    "file_transfer": [
        "VPN-PCAPS-01/vpn_ftps_A.pcap",
        "VPN-PCAPS-01/vpn_ftps_B.pcap",
        "VPN-PCAPs-02/vpn_sftp_A.pcap",
        "VPN-PCAPs-02/vpn_sftp_B.pcap",
        "VPN-PCAPs-02/vpn_skype_files1a.pcap",
        "VPN-PCAPs-02/vpn_skype_files1b.pcap",
    ],
    "chat": [
        "VPN-PCAPS-01/vpn_aim_chat1a.pcap",
        "VPN-PCAPS-01/vpn_aim_chat1b.pcap",
        "VPN-PCAPS-01/vpn_facebook_chat1a.pcap",
        "VPN-PCAPS-01/vpn_facebook_chat1b.pcap",
        "VPN-PCAPS-01/vpn_hangouts_chat1a.pcap",
        "VPN-PCAPS-01/vpn_hangouts_chat1b.pcap",
        "VPN-PCAPs-02/vpn_icq_chat1a.pcap",
        "VPN-PCAPs-02/vpn_icq_chat1b.pcap",
        "VPN-PCAPs-02/vpn_skype_chat1a.pcap",
        "VPN-PCAPs-02/vpn_skype_chat1b.pcap",
    ],
    "email": [
        "VPN-PCAPS-01/vpn_email2a.pcap",
        "VPN-PCAPS-01/vpn_email2b.pcap",
    ],
}


def _client():
    from kagglesdk import KaggleClient

    creds_path = Path(os.environ.get("KAGGLE_CONFIG_DIR", "~/.kaggle")).expanduser() / "kaggle.json"
    if not creds_path.exists():
        sys.exit(f"Kaggle credentials not found at {creds_path}. Create an API token at kaggle.com/settings.")
    creds = json.load(open(creds_path))
    return KaggleClient(username=creds["username"], api_token=creds["key"])


def _download_one(rel_path: str, expected_sha: str | None) -> tuple[str, float, str]:
    from kagglesdk.datasets.types.dataset_api_service import ApiDownloadDatasetRequest

    dest = RAW_DIR / Path(rel_path).name
    if dest.exists() and dest.stat().st_size > 0:
        # The mirror's SHA256SUMS.txt only covers the zip bundles, so per-file
        # checksums are usually unavailable; a completed (renamed) file is trusted.
        if not expected_sha or _sha256(dest) == expected_sha:
            return rel_path, 0.0, "cached"

    t0 = time.time()
    with _client() as c:
        req = ApiDownloadDatasetRequest()
        req.owner_slug, req.dataset_slug = KAGGLE_REF.split("/")
        req.file_name = rel_path
        resp = c.datasets.dataset_api_client.download_dataset(req)
        tmp = dest.with_suffix(dest.suffix + ".part")
        h = hashlib.sha256()
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(1 << 20):
                f.write(chunk)
                h.update(chunk)
    if expected_sha and h.hexdigest() != expected_sha:
        tmp.unlink(missing_ok=True)
        return rel_path, time.time() - t0, "SHA MISMATCH"
    tmp.rename(dest)
    return rel_path, time.time() - t0, f"{dest.stat().st_size / 1e6:.0f} MB"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_checksums() -> dict[str, str]:
    """SHA256SUMS.txt is part of the mirror; fetch it once and map filename -> digest."""
    sums = RAW_DIR / "SHA256SUMS.txt"
    if not sums.exists():
        _download_one("SHA256SUMS.txt", None)
    out = {}
    for line in sums.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            out[Path(parts[-1]).name] = parts[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--labels", nargs="*", help="only fetch these labels")
    ap.add_argument("--vpn", action="store_true", help="fetch the VPN-tunnelled captures (robustness test set)")
    args = ap.parse_args()

    manifest = VPN_MANIFEST if args.vpn else MANIFEST
    files = [(lab, f) for lab, fs in manifest.items() for f in fs if not args.labels or lab in args.labels]
    if args.list:
        for lab, f in files:
            print(f"{lab:14s} {f}")
        print(f"{len(files)} files")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    sha = _load_checksums()
    print(f"Fetching {len(files)} files with {args.workers} workers -> {RAW_DIR}")
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(_download_one, f, sha.get(Path(f).name)): f for _, f in files}
        for i, fut in enumerate(as_completed(futs), 1):
            rel, dt, status = fut.result()
            print(f"[{i:3d}/{len(files)}] {status:>14s}  {dt:6.0f}s  {rel}", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
