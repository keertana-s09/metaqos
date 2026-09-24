# Model card: MetaQoS flow classifier (`random_forest`)

## Intended use

Assign a traffic class to an encrypted network flow from packet metadata only (sizes, timings, direction, transport protocol) so a home router can place it in an appropriate QoS queue. Not intended for security decisions, user identification, or any purpose that requires knowing *what* is being communicated.

## Inputs

34 features computed over a window of 32 consecutive packets of one bidirectional flow (see `src/feature_extraction.py`). No IP addresses, ports or payload bytes are used.

## Outputs

One of `chat`, `email`, `file_transfer`, `streaming`, `video_call`, `voip_audio` with a class-probability vector; the QoS mapper acts only above a confidence floor.

## Training data

ISCX VPN-nonVPN 2016 (Draper-Gil, Lashkari, Mamun, Ghorbani; ICISSP 2016), NonVPN captures listed in `data/fetch_iscx.py`. 28058 windows from 63 capture sessions (after capping 300 windows per flow). Labels are per capture file, so background flows in a capture carry that file's label (known label noise).

## Evaluation

* Leave-session-out 5-fold CV on training sessions: accuracy 0.850, macro-F1 0.671.
* Held-out sessions (12 sessions never seen in training): accuracy 0.934, macro-F1 0.817, QoS-tier accuracy 0.970.

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| chat | 0.515 | 0.846 | 0.640 | 279 |
| email | 0.402 | 0.734 | 0.519 | 286 |
| file_transfer | 0.981 | 0.851 | 0.911 | 8701 |
| streaming | 0.995 | 0.985 | 0.990 | 8324 |
| video_call | 0.991 | 0.816 | 0.895 | 7620 |
| voip_audio | 0.905 | 0.991 | 0.946 | 22830 |
| macro avg | 0.798 | 0.871 | 0.817 | 48040 |

## Cost

3662 KB on disk; 3520 us per flow decision single-threaded.

## Most influential features

`size_max`, `iat_min`, `up_size_max`, `bytes_per_s`, `size_mean`, `run_len_mean`, `size_std`, `run_len_max`

## Known limitations

* No `browsing` or `gaming` class: ISCX does not contain those captures. Such traffic will be mapped to the nearest class the model knows (see the demo's synthetic flows) and should be treated as unknown.
* Trained on 2016 captures from one lab; codecs, CDNs and transport (QUIC) have changed. Expect degraded accuracy on modern traffic without retraining on fresh captures.
* Only NonVPN traffic. Inside a VPN tunnel all flows share one 5-tuple and this per-flow approach does not apply.
* Capture-file labelling means some 'video_call' windows are really the app's control channel, etc.

See `eval/report.md` for the full evaluation.