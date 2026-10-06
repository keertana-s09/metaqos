# MetaQoS — encrypted-traffic classification for home-router QoS, without DPI

MetaQoS classifies network flows into application categories (video call, VoIP
audio, streaming, file transfer, chat, email) using **only packet metadata**:
sizes, inter-arrival times, direction and burstiness. It never reads a payload
byte, so it works identically on TLS 1.3, QUIC/HTTP-3, or anything else that is
encrypted, and it cannot learn *what* you are saying, only *what kind* of
traffic it is. The predicted class and confidence are mapped to a simulated
QoS action (DSCP mark + priority queue), and the decision is revised as more
packets of the flow arrive.

```
packets ──► flow reconstruction ──► per-window features ──► classifier ──► QoS policy ──► queue / DSCP
              (5-tuple, direction)     (34 metadata stats)    (GBT/RF/tree)   (confidence floor,
                                                                              EMA re-classification)
```

## Results (measured, see `eval/report.md`)

Trained on 28058 windows from 63 ISCX capture sessions; tested on 48040 windows from 12 **held-out sessions** the model never saw. Six classes: video call, VoIP audio, streaming, file transfer, chat, email.

| model | CV macro-F1 | CV acc | held-out macro-F1 | held-out acc | held-out **QoS-tier acc** | latency / flow (1 thread) | size |
|---|---|---|---|---|---|---|---|
| `decision_tree` | 0.611 | 0.788 | 0.726 | 0.881 | 0.911 | 58 µs | 20 KB |
| `random_forest` | 0.671 | 0.850 | 0.817 | 0.934 | 0.970 | 3464 µs | 3661 KB |
| `hist_gbt` | 0.582 | 0.789 | 0.853 | 0.931 | 0.960 | 1596 µs | 562 KB |

Selected: **random forest** (best leave-session-out CV). On held-out sessions: accuracy **0.934**, macro-F1 **0.817**, and **0.970 QoS-tier accuracy** (the number that matters for the router: was the flow put in the right queue?). Per class on held-out (precision / recall / F1):

| class | P | R | F1 | windows |
|---|---|---|---|---|
| chat | 0.52 | 0.85 | 0.64 | 279 |
| email | 0.40 | 0.73 | 0.52 | 286 |
| file_transfer | 0.98 | 0.85 | 0.91 | 8701 |
| streaming | 0.99 | 0.99 | 0.99 | 8324 |
| video_call | 0.99 | 0.82 | 0.90 | 7620 |
| voip_audio | 0.91 | 0.99 | 0.95 | 22830 |

What the errors actually are (from the per-flow breakdown in the report):

* `video_call` recall 0.82: the misses are the Facebook call's *separate audio stream* (118 B packets, 108 pkt/s), classified as `voip_audio`. Same QoS tier, and arguably the right answer.
* `file_transfer` recall 0.85: the misses are a second, *interactive* SSH session inside the SFTP capture (57 B packets, 25 % upstream), which the capture-file labelling calls `file_transfer` and the model calls `voip_audio`/`email`/`chat` (small-packet interactive classes). The bulk SFTP flow itself is 99.9 % correct.
* `chat` / `email` precision is low because they are small classes (279 / 286 held-out windows) that absorb the label-noise flows above, and because email has only two capture sessions in the whole dataset (in CV it is unlearnable: 0 recall when one session is held out).
* Leave-session-out CV (0.67 macro-F1) is well below the held-out numbers because some folds hold out the *only* session of an application (e.g. all Hangouts video), which is a cross-application generalisation test with 4 training sessions per class.

**Early decisions are weak, re-classification fixes them.** Accuracy on a flow's first 32-packet window is 0.62 (handshakes look alike); it reaches 0.94 once a flow is established. EMA smoothing lifts the second decision from 0.61 to 0.73. Larger windows help: 64 packets gives 0.903 held-out macro-F1 vs 0.817 for 32, at twice the time-to-first-decision (8-packet windows: 0.755).

**Cost.** The forest is 3.6 MB and sklearn needs ~3.5 ms per single-flow call on one thread (the actual tree compute is 11 µs/flow when batched; the rest is sklearn's per-call overhead). The gradient-boosted model is 0.5 MB / ~1.6 ms with slightly better held-out F1 and worse CV; the 20 KB decision tree decides in 58 µs and gives up 5 points of accuracy. Pick by the target CPU.

Removing the only header feature (`proto_udp`) leaves held-out accuracy unchanged (0.934) and lowers macro-F1 from 0.817 to 0.789: the size/timing statistics carry the signal, the protocol flag only helps the small classes.

Full tables, confusion matrices, feature importances and the per-flow breakdown: [`eval/report.md`](eval/report.md). Model card: [`models/MODEL_CARD.md`](models/MODEL_CARD.md).

## Why metadata is enough

Encryption hides bytes, not behaviour. The transport layer still exposes:

| signal | what it reveals |
|---|---|
| packet sizes | codec frame sizes (audio ≈ 100–200 B, video frames split into ~1100 B chunks), full-MTU data segments for bulk transfer, tiny ACKs |
| inter-arrival times | real-time media is clocked (20 ms audio, ~33 ms video frames); streaming and downloads are bursty; chat is sparse |
| direction pattern | calls are symmetric; streaming/downloads are one-way with ACK back-channel; request/response alternates |
| burstiness | Goh–Barabási index of IATs (−1 periodic, 0 Poisson, →1 bursty) and direction run-lengths |

These are exactly the statistics that DPI vendors also fall back on for
encrypted traffic; the difference is that MetaQoS *only* uses them. Ports and IP
addresses are deliberately excluded from the feature set: they would let the
model cheat on the training data (SCP is always port 22 there) and they are
useless on modern traffic where everything hides behind 443/QUIC. The transport
protocol flag (TCP/UDP) is the only header field used, and the report includes
an ablation without it.

## Repository layout

```
data/
  fetch_iscx.py        download the curated ISCX subset from the Kaggle mirror
  build_dataset.py     pcaps -> one row per observation window (parquet)
  make_demo_pcap.py    build the demo pcap from held-out flows (+ synthetic gaming/browsing)
  samples/demo.pcap    demo capture (metadata-only rebuild, no original payload)
src/
  flows.py             pcap reader, 5-tuple flow table, packet windows, MTU re-segmentation
  feature_extraction.py  34 per-window features (documented in the module)
  classifier.py        train / cross-validate / serve; held-out session list
  qos_mapper.py        class -> DSCP/queue policy with confidence floor
  reclassify.py        streaming tracker: EMA-smoothed re-classification
  demo.py              CLI replay (rich live table)
  dashboard.py         Streamlit dashboard
models/
  metaqos_classifier.joblib   trained model + metadata
  MODEL_CARD.md
eval/
  evaluate.py          full evaluation -> report.md, results.json, plots
  window_sweep.py      accuracy vs observation-window size
  report.md            the evaluation report
tests/                 pytest unit tests (features, flows, QoS, re-classification)
```

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests -q        # 15 tests, ~2 s
```

## Run the User Interfaces

MetaQoS provides two complementary, modern web user interfaces:

### 1. Streamlit Interactive Dashboard
```bash
streamlit run src/dashboard.py
```
* **Features**: Dark-mode glassmorphic theme, interactive Plotly queue timeline, 34-feature flow inspector, held-out model telemetry, and Linux `tc` / OpenWrt `nftables` exporter.

### 2. Standalone Web Command Center (FastAPI + HTML5/WebSocket)
```bash
python -m src.web_server
# Open http://localhost:8000 in your browser
```
* **Features**: Real-time animated **Network Flow Topology Diagram**, WebSocket streaming PCAP replay, live search/filtering, and instant router rule copy.

### 3. CLI Live Replay
```bash
python -m src.demo                 # replay data/samples/demo.pcap in real time
python -m src.demo --speed 5       # 5x faster
python -m src.demo --speed 0       # instant summary + tc commands
```


The demo pcap contains one real flow per class taken from **held-out** capture
sessions (never seen in training) plus two synthetic flows (online gaming and
web browsing) for classes the training set does not contain. The table shows
ground truth next to the prediction, so you can see both what works and how
the model treats traffic it was never trained on.

Live capture (`--live eth0`) works with scapy if the interpreter has
`CAP_NET_RAW` (`sudo setcap cap_net_raw,cap_net_admin=eip .venv/bin/python3` or run as root).

## Reproduce the evaluation

```bash
# 1. data (~5 GB from the Kaggle mirror; needs ~/.kaggle/kaggle.json)
.venv/bin/python data/fetch_iscx.py --workers 6
# 2. feature table (about a minute on 8 cores)
.venv/bin/python data/build_dataset.py --window 32
# 3. train + evaluate all models, export the best, write eval/report.md and models/MODEL_CARD.md
.venv/bin/python eval/evaluate.py
# 4. (optional) window-size sweep
.venv/bin/python eval/window_sweep.py --windows 8 16 32 64
# 5. demo pcap from held-out sessions
.venv/bin/python data/make_demo_pcap.py
```

### Dataset

ISCX VPN-nonVPN 2016 (Draper-Gil, Lashkari, Mamun & Ghorbani, *Characterization
of Encrypted and VPN Traffic Using Time-Related Features*, ICISSP 2016). The
original CIC host now gates downloads behind a request form, so
`data/fetch_iscx.py` pulls individual files from the Kaggle mirror
`caiyafang/iscx-vpn-nonvpn-2016-complete-pcaps` (a copy of the published
pcaps). Only the NonVPN captures are used; the two multi-GB FTPS captures and
one 3 GB SCP capture are skipped as they add flows but no diversity.

Things to know about this dataset, because they shape the numbers:

* **Labels are per capture file.** Every flow in `skype_video1a.pcap` is
  labelled `video_call`, including the OS's background chatter and Skype's
  own signalling channel. Infrastructure ports (DNS, NTP, mDNS, SSDP) are
  dropped and flows shorter than half a window are discarded, but some label
  noise remains and caps achievable accuracy.
* **`…1a` / `…1b` are the two ends of the same session.** They are grouped
  so they never straddle a train/test split.
* **No browsing, no gaming captures exist in the release**, so the model has
  no such classes. The QoS policy already defines tiers for them; adding them
  needs new labelled captures (see *Limitations*).
* **Host captures contain TSO super-packets** (16–64 KB "packets" that the NIC
  segments after the capture point). `flows.py` re-segments them to 1500 B so
  features reflect what a router sees on the wire.

## Design decisions

**One sample = a window of 32 consecutive packets of one flow**, not the whole
flow. A router must decide early and keep deciding; 32 packets is ~0.3 s of a
VoIP call, ~60 ms of a video call, a few ms of a download. Windows are counted
in packets rather than seconds so every sample has the same number of
observations; the window's duration is itself a feature and carries the rate
information. `eval/window_sweep.py` quantifies the trade-off against 8/16/64.

**Tree ensembles, not deep learning.** The features span microseconds to
seconds and bytes to megabytes; trees are scale-free, so there is no
normalisation pipeline to ship. The final model is a few hundred KB and decides
a flow in well under a millisecond on one core, which is home-router
territory. Feature importances also explain decisions, which matters when the
output is a QoS action a user might question.

**Evaluation is leave-session-out.** Windows from one flow are near-duplicates.
A random window-level split leaks them across train and test and reports
~99 % accuracy that says nothing about a new call or a new download. Every
number in the report comes from cross-validation grouped by capture session,
plus a fixed set of held-out sessions the final model never sees. This is
also why the numbers are lower than in many papers on this dataset.

**Long flows are capped** at 300 windows each (evenly spaced) for training, so
a 20-minute call does not outvote every chat session, and classes are
re-weighted to balance.

**QoS acts only above a confidence floor** (0.6 by default). A wrong promotion
(a download in the EF queue) hurts real calls, and a wrong demotion (a call in
the bulk queue) is worse, so the low-confidence default is the neutral
best-effort queue.

**Re-classification is smoothed.** Per-window predictions flicker (a video
call's silent stretch looks like audio; a stream's buffer refill looks like a
download). Each flow keeps an exponential moving average of the class
probabilities (α = 0.4) and the queue is chosen from the smoothed
distribution, so a flow does not flap between queues on one odd window.
`eval/report.md` shows how much smoothing helps after the first few windows.

## Limitations and next steps

* **Classes.** Gaming and browsing are absent from ISCX. The demo's synthetic
  flows show what the model does with them today (it picks the nearest class
  it knows). The right fix is a small self-captured, labelled dataset:
  `tcpdump` a few minutes of each activity on your own machine, drop the pcaps
  in `data/raw/`, add them to the manifest with a label, rebuild, retrain.
* **Age and lab bias.** Captures are from 2016, one lab, one network. Codecs,
  CDNs and QUIC have changed since. Expect some degradation on today's traffic
  without fresh captures; the pipeline is built so that retraining is one
  command.
* **VPN traffic.** Inside a tunnel all applications share one 5-tuple; this
  per-flow approach does not apply (the ISCX VPN captures are intentionally
  not used).
* **Router integration** is simulated: the policy layer prints the
  `iptables … -j DSCP` / `tc filter` commands it would run. On OpenWrt the
  real thing is a few lines of nftables + CAKE with `diffserv4`.
