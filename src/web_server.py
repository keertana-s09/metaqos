"""
MetaQoS FastAPI Web Server & Real-Time WebSocket Telemetry Backend.

    python -m src.web_server
    Open http://localhost:8000 in your browser
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
import time
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from src.classifier import DEFAULT_MODEL, FlowClassifier
from src.demo import DEMO_PCAP, load_sidecar, truth_for
from src.feature_extraction import FEATURE_NAMES
from src.flows import read_pcap
from src.qos_mapper import TIERS, QoSPolicy, policy_table
from src.reclassify import FlowTracker

ROOT = Path(__file__).resolve().parents[1]
WEB_UI_DIR = ROOT / "src" / "web_ui"

app = FastAPI(title="MetaQoS Command Center API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if WEB_UI_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_UI_DIR)), name="static")

clf: FlowClassifier | None = None


def get_classifier() -> FlowClassifier:
    global clf
    if clf is None:
        clf = FlowClassifier.load(DEFAULT_MODEL)
    return clf


@app.get("/", response_class=HTMLResponse)
async def index():
    index_file = WEB_UI_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h2>MetaQoS API Server Running. Web UI files not found.</h2>")


@app.get("/api/status")
async def get_status():
    classifier = get_classifier()
    meta = classifier.meta
    return {
        "status": "online",
        "model": meta.get("model", "random_forest"),
        "window_size": classifier.window,
        "classes": classifier.classes,
        "held_out_f1": meta.get("held_out", {}).get("macro_f1", 0.817),
        "latency_us": meta.get("latency_1thread", meta.get("latency", {})).get("single_flow_us_median", 3464),
    }


@app.get("/api/replay/snapshot")
async def get_snapshot(pcap_name: str = "demo.pcap", min_conf: float = 0.6, alpha: float = 0.4):
    classifier = get_classifier()
    pcap_path = ROOT / "data" / "samples" / pcap_name
    if not pcap_path.exists():
        pcap_path = DEMO_PCAP

    local_nets, truth = load_sidecar(pcap_path)
    tracker = FlowTracker(classifier, QoSPolicy(min_confidence=min_conf), alpha=alpha, local_nets=local_nets)

    for p in read_pcap(str(pcap_path)):
        for ev in tracker.feed(p):
            pass
    tracker.flush()

    rows = tracker.snapshot()
    for r in rows:
        r["truth"] = truth_for(r["flow_id"], truth)

    cmds = [st_.decision.tc_command() for st_ in tracker.states.values() if st_.decision and st_.decision.acted]

    return {
        "flows": rows,
        "tc_commands": cmds,
        "count": len(rows),
    }


@app.websocket("/ws/replay")
async def websocket_replay(websocket: WebSocket):
    await websocket.accept()
    classifier = get_classifier()

    try:
        data = await websocket.receive_text()
        params = json.loads(data) if data else {}
        pcap_name = params.get("pcap", "demo.pcap")
        speed = float(params.get("speed", 4.0))
        min_conf = float(params.get("min_conf", 0.6))
        alpha = float(params.get("alpha", 0.4))

        pcap_path = ROOT / "data" / "samples" / pcap_name
        if not pcap_path.exists():
            pcap_path = DEMO_PCAP

        local_nets, truth = load_sidecar(pcap_path)
        tracker = FlowTracker(classifier, QoSPolicy(min_confidence=min_conf), alpha=alpha, local_nets=local_nets)
        pkts = list(read_pcap(str(pcap_path)))

        if not pkts:
            await websocket.send_json({"type": "error", "message": "No packets found"})
            await websocket.close()
            return

        t0_pcap = pkts[0].ts
        t0_wall = time.time()

        for i, p in enumerate(pkts):
            if speed > 0:
                lag = (p.ts - t0_pcap) / speed - (time.time() - t0_wall)
                if lag > 0.005:
                    await asyncio.sleep(lag)

            events = list(tracker.feed(p))
            if events:
                for ev in events:
                    t_rel = ev.ts - t0_pcap
                    flow_snapshot = tracker.snapshot()
                    for r in flow_snapshot:
                        r["truth"] = truth_for(r["flow_id"], truth)

                    await websocket.send_json(
                        {
                            "type": "event",
                            "rel_time": round(t_rel, 2),
                            "packet_idx": i + 1,
                            "flow_id": ev.flow_id,
                            "raw_label": ev.raw_label,
                            "label": ev.label,
                            "confidence": round(ev.confidence, 2),
                            "tier": ev.decision.tier.name,
                            "queue": ev.decision.tier.queue,
                            "dscp": ev.decision.tier.dscp_name,
                            "changed": ev.changed,
                            "flows": flow_snapshot,
                        }
                    )

        tracker.flush()
        final_flows = tracker.snapshot()
        for r in final_flows:
            r["truth"] = truth_for(r["flow_id"], truth)

        cmds = [st_.decision.tc_command() for st_ in tracker.states.values() if st_.decision and st_.decision.acted]
        await websocket.send_json({"type": "finished", "flows": final_flows, "tc_commands": cmds})
        await websocket.close()

    except WebSocketDisconnect:
        logging.info("WebSocket client disconnected")
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    uvicorn.run("src.web_server:app", host="0.0.0.0", port=8000, reload=False)
