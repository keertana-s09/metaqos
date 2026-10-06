// MetaQoS Web Command Center JavaScript Application

let socket = null;
let activeFlowsMap = new Map();
let totalProcessedPackets = 0;
let isReplaying = false;
let animationParticles = [];

const TIER_COLORS = {
  realtime: "#FF4D4D",
  interactive: "#FFA600",
  streaming: "#1F6FEB",
  best_effort: "#238636",
  bulk: "#8B949E"
};

const QUEUE_TIERS = {
  1: { name: "realtime", label: "Queue 1: Strict EF" },
  2: { name: "interactive", label: "Queue 2: High AF41" },
  3: { name: "streaming", label: "Queue 3: Med AF31" },
  4: { name: "best_effort", label: "Queue 4: Default BE" },
  5: { name: "bulk", label: "Queue 5: Bulk CS1" }
};

document.addEventListener("DOMContentLoaded", () => {
  initUIEvents();
  initCanvas();
  fetchAPIStatus();
});

function initUIEvents() {
  const speedSlider = document.getElementById("speed-slider");
  const speedVal = document.getElementById("speed-val");
  speedSlider.addEventListener("input", (e) => speedVal.textContent = parseFloat(e.target.value).toFixed(1) + "x");

  const confSlider = document.getElementById("conf-slider");
  const confVal = document.getElementById("conf-val");
  confSlider.addEventListener("input", (e) => confVal.textContent = parseFloat(e.target.value).toFixed(2));

  const alphaSlider = document.getElementById("alpha-slider");
  const alphaVal = document.getElementById("alpha-val");
  alphaSlider.addEventListener("input", (e) => alphaVal.textContent = parseFloat(e.target.value).toFixed(2));

  document.getElementById("btn-start").addEventListener("click", startReplay);
  document.getElementById("btn-pause").addEventListener("click", pauseReplay);
  document.getElementById("flow-search").addEventListener("input", filterFlowTable);
  document.getElementById("btn-copy-rules").addEventListener("click", copyRouterRules);
}

async function fetchAPIStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    if (data.status === "online") {
      document.getElementById("conn-status").textContent = "Connected Engine";
      document.getElementById("conn-status").classList.add("active");
    }
  } catch (e) {
    document.getElementById("conn-status").textContent = "Offline";
  }
}

function startReplay() {
  if (isReplaying) return;
  isReplaying = true;
  activeFlowsMap.clear();
  totalProcessedPackets = 0;
  
  document.getElementById("btn-start").disabled = true;
  document.getElementById("btn-pause").disabled = false;
  document.getElementById("flowTableBody").innerHTML = "";
  document.getElementById("logStream").innerHTML = "";

  const pcap = document.getElementById("pcap-select").value;
  const speed = parseFloat(document.getElementById("speed-slider").value);
  const min_conf = parseFloat(document.getElementById("conf-slider").value);
  const alpha = parseFloat(document.getElementById("alpha-slider").value);

  const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${wsProtocol}//${window.location.host}/ws/replay`;

  socket = new WebSocket(wsUrl);

  socket.onopen = () => {
    socket.send(JSON.stringify({ pcap, speed, min_conf, alpha }));
    appendLogEntry("WebSocket connection established. Starting pcap replay...");
  };

  socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "event") {
      handleReplayEvent(data);
    } else if (data.type === "finished") {
      handleReplayFinished(data);
    } else if (data.type === "error") {
      appendLogEntry("ERROR: " + data.message);
      pauseReplay();
    }
  };

  socket.onclose = () => {
    isReplaying = false;
    document.getElementById("btn-start").disabled = false;
    document.getElementById("btn-pause").disabled = true;
  };
}

function pauseReplay() {
  if (socket) {
    socket.close();
  }
  isReplaying = false;
  document.getElementById("btn-start").disabled = false;
  document.getElementById("btn-pause").disabled = true;
  appendLogEntry("Replay paused.");
}

function handleReplayEvent(data) {
  totalProcessedPackets = data.packet_idx;
  document.getElementById("stat-processed-pkts").textContent = totalProcessedPackets.toLocaleString();

  if (data.flows) {
    data.flows.forEach(flow => activeFlowsMap.set(flow.flow_id, flow));
    document.getElementById("stat-active-flows").textContent = activeFlowsMap.size;
    renderFlowTable(Array.from(activeFlowsMap.values()));
  }

  if (data.changed) {
    appendLogEntry(`t+${data.rel_time}s ${data.flow_id} -> ${data.label} (${data.confidence}) [Tier: ${data.tier}]`);
    spawnParticleForTier(data.tier);
  }
}

function handleReplayFinished(data) {
  appendLogEntry("🎉 Replay finished!");
  pauseReplay();
  if (data.tc_commands && data.tc_commands.length > 0) {
    document.getElementById("routerRulesCode").textContent = data.tc_commands.join("\n\n");
  } else {
    document.getElementById("routerRulesCode").textContent = "# No high-confidence flow rules were generated";
  }
}

function renderFlowTable(flows) {
  const tbody = document.getElementById("flowTableBody");
  const searchQuery = document.getElementById("flow-search").value.toLowerCase();
  
  tbody.innerHTML = "";
  if (flows.length === 0) {
    tbody.innerHTML = '<tr><td colspan="9" class="empty-state">No flows detected</td></tr>';
    return;
  }

  flows.sort((a, b) => a.flow_id.localeCompare(b.flow_id));

  flows.forEach(flow => {
    if (searchQuery && !flow.flow_id.toLowerCase().includes(searchQuery)) return;

    const tr = document.createElement("tr");
    const confPct = Math.round(flow.confidence * 100);
    const tier = flow.tier || "best_effort";

    tr.innerHTML = `
      <td class="flow-id">${escapeHtml(flow.flow_id)}</td>
      <td>${flow.packets}</td>
      <td>${flow.windows}</td>
      <td>${escapeHtml(flow.truth || "?")}</td>
      <td><strong>${escapeHtml(flow.class)}</strong></td>
      <td>
        <div class="conf-bar-bg"><div class="conf-bar-fill" style="width: ${confPct}%"></div></div>
        <span>${flow.confidence.toFixed(2)}</span>
      </td>
      <td><span class="tier-pill tier-${tier}">${tier}</span></td>
      <td>${escapeHtml(flow.dscp)}</td>
      <td><strong>Queue ${flow.queue}</strong></td>
    `;
    tbody.appendChild(tr);
  });
}

function filterFlowTable() {
  renderFlowTable(Array.from(activeFlowsMap.values()));
}

function appendLogEntry(text) {
  const logStream = document.getElementById("logStream");
  const div = document.createElement("div");
  div.className = "log-entry";
  div.textContent = text;
  logStream.appendChild(div);
  logStream.scrollTop = logStream.scrollHeight;
}

function copyRouterRules() {
  const code = document.getElementById("routerRulesCode").textContent;
  navigator.clipboard.writeText(code).then(() => {
    const btn = document.getElementById("btn-copy-rules");
    btn.textContent = "✅ Copied!";
    setTimeout(() => btn.textContent = "📋 Copy Rules", 2000);
  });
}

function escapeHtml(str) {
  return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/* Topology Canvas Animation */
function initCanvas() {
  const canvas = document.getElementById("topologyCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  function drawTopology() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Left Node: LAN Hosts
    ctx.fillStyle = "#161b22";
    ctx.strokeStyle = "#30363d";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(40, 50, 160, 140, 10);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = "#f0f6fc";
    ctx.font = "bold 13px Inter, sans-serif";
    ctx.fillText("🏠 Home LAN Hosts", 55, 80);
    ctx.font = "11px Inter, sans-serif";
    ctx.fillStyle = "#8b949e";
    ctx.fillText("192.168.1.x / 10.0.0.x", 55, 105);
    ctx.fillText("Encrypted Flows (TLS 1.3)", 55, 125);

    // Center Node: MetaQoS Classifier Engine
    ctx.fillStyle = "#161b22";
    ctx.strokeStyle = "#58a6ff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(450, 40, 240, 160, 12);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = "#58a6ff";
    ctx.font = "bold 15px Inter, sans-serif";
    ctx.fillText("⚡ MetaQoS Classifier", 475, 75);
    ctx.font = "11px Inter, sans-serif";
    ctx.fillStyle = "#f0f6fc";
    ctx.fillText("34 Metadata Stats / Window", 475, 100);
    ctx.fillText("Random Forest Model (0.817 F1)", 475, 120);
    ctx.fillStyle = "#3fb950";
    ctx.fillText("EMA Re-Classification (α = 0.4)", 475, 140);

    // Right Nodes: Queue 1 - 5
    const queueY = [30, 75, 120, 165, 205];
    const queueNames = ["Q1: EF Strict", "Q2: AF41 High", "Q3: AF31 Med", "Q4: BE Default", "Q5: CS1 Bulk"];
    const queueColors = ["#FF4D4D", "#FFA600", "#1F6FEB", "#238636", "#8B949E"];

    queueY.forEach((y, i) => {
      ctx.fillStyle = "#161b22";
      ctx.strokeStyle = queueColors[i];
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.roundRect(920, y, 180, 32, 6);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = queueColors[i];
      ctx.font = "bold 11px Inter, sans-serif";
      ctx.fillText(queueNames[i], 935, y + 20);
    });

    // Connecting Lines
    ctx.strokeStyle = "rgba(48, 54, 61, 0.6)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(200, 120);
    ctx.lineTo(450, 120);
    ctx.stroke();

    queueY.forEach(y => {
      ctx.beginPath();
      ctx.moveTo(690, 120);
      ctx.lineTo(920, y + 16);
      ctx.stroke();
    });

    // Update & draw particles
    animationParticles.forEach((p, idx) => {
      p.progress += p.speed;
      if (p.progress >= 1) {
        animationParticles.splice(idx, 1);
        return;
      }

      let px, py;
      if (p.stage === 1) {
        px = 200 + (450 - 200) * p.progress;
        py = 120;
      } else {
        px = 690 + (920 - 690) * p.progress;
        py = 120 + (p.targetY - 120) * p.progress;
      }

      ctx.fillStyle = p.color;
      ctx.shadowColor = p.color;
      ctx.shadowBlur = 8;
      ctx.beginPath();
      ctx.arc(px, py, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
    });

    requestAnimationFrame(drawTopology);
  }

  requestAnimationFrame(drawTopology);
}

function spawnParticleForTier(tier) {
  const queueYMap = {
    realtime: 46,
    interactive: 91,
    streaming: 136,
    best_effort: 181,
    bulk: 221
  };
  const color = TIER_COLORS[tier] || "#58a6ff";
  const targetY = queueYMap[tier] || 181;

  // Stage 1 particle (LAN -> Engine)
  animationParticles.push({
    stage: 1,
    progress: 0,
    speed: 0.03,
    color: "#58a6ff"
  });

  // Stage 2 particle (Engine -> Queue)
  setTimeout(() => {
    animationParticles.push({
      stage: 2,
      progress: 0,
      speed: 0.03,
      targetY: targetY,
      color: color
    });
  }, 400);
}
