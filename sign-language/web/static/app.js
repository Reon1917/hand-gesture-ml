const state = {
  cameraReady: false,
  inFlight: false,
  status: null,
  currentLabel: null,
};

const elements = {
  sourceVideo: document.getElementById("sourceVideo"),
  captureCanvas: document.getElementById("captureCanvas"),
  modelChip: document.getElementById("modelChip"),
  datasetChip: document.getElementById("datasetChip"),
  accuracyChip: document.getElementById("accuracyChip"),
  latencyBadge: document.getElementById("latencyBadge"),
  primaryLabel: document.getElementById("primaryLabel"),
  confidenceValue: document.getElementById("confidenceValue"),
  statusValue: document.getElementById("statusValue"),
  helperText: document.getElementById("helperText"),
  exampleGrid: document.getElementById("exampleGrid"),
  predictionList: document.getElementById("predictionList"),
  guidanceList: document.getElementById("guidanceList"),
  reloadModelButton: document.getElementById("reloadModelButton"),
};

function percent(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

function setStatusMessage(message) {
  elements.statusValue.textContent = message;
}

function renderGuidance(items) {
  elements.guidanceList.replaceChildren();
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    elements.guidanceList.appendChild(li);
  });
}

function renderExampleCards(cards, activeLabel) {
  elements.exampleGrid.replaceChildren();

  cards.forEach((card) => {
    const article = document.createElement("article");
    article.className = "example-card";
    if (activeLabel && card.label === activeLabel) {
      article.classList.add("active");
    }

    const media = card.image_url
      ? `<img src="${card.image_url}" alt="ASL example ${card.label}">`
      : `<div class="example-placeholder">${card.label}</div>`;

    article.innerHTML = `
      ${media}
      <div class="example-label-row">
        <strong>${card.label}</strong>
        <span>${card.image_url ? "dataset sample" : "placeholder"}</span>
      </div>
      <p>${card.hint || "Match the sample card and keep the pose steady."}</p>
    `;
    elements.exampleGrid.appendChild(article);
  });
}

function renderPredictionList(items) {
  elements.predictionList.replaceChildren();
  if (!items.length) {
    const empty = document.createElement("p");
    empty.className = "helper-text";
    empty.textContent = "Predictions appear here once the camera stream is running.";
    elements.predictionList.appendChild(empty);
    return;
  }

  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = "prediction-item";
    row.innerHTML = `
      <div class="row">
        <strong>${item.label}</strong>
        <span>${percent(item.confidence)}</span>
      </div>
      <div class="bar">
        <div class="bar-fill" style="width:${Math.max(4, Number(item.confidence || 0) * 100)}%"></div>
      </div>
    `;
    elements.predictionList.appendChild(row);
  });
}

function renderStatus(payload) {
  state.status = payload;
  const summary = payload.model_summary || {};

  elements.modelChip.textContent = payload.model_ready
    ? `Model ready · ${summary.model_name || "trained"}`
    : "Model missing";
  elements.modelChip.classList.toggle("chip-warn", !payload.model_ready);

  const datasetSize = Number(summary.dataset_size || 0);
  elements.datasetChip.textContent = datasetSize ? `${datasetSize.toLocaleString()} images` : "Awaiting training";

  if (typeof summary.validation_accuracy === "number") {
    elements.accuracyChip.textContent = `Val ${percent(summary.validation_accuracy)}`;
  } else {
    elements.accuracyChip.textContent = "No metrics yet";
  }

  renderGuidance(payload.guidance || []);
  renderExampleCards(payload.examples || [], state.currentLabel);
  setStatusMessage(payload.last_status || "Ready");
}

async function refreshStatus() {
  const response = await fetch("/api/status");
  const payload = await response.json();
  renderStatus(payload);
}

async function reloadModel() {
  elements.reloadModelButton.disabled = true;
  elements.reloadModelButton.textContent = "Reloading…";
  try {
    const response = await fetch("/api/reload-model", { method: "POST" });
    const payload = await response.json();
    renderStatus(payload);
  } finally {
    elements.reloadModelButton.disabled = false;
    elements.reloadModelButton.textContent = "Reload Model";
  }
}

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: "user",
      },
      audio: false,
    });
    elements.sourceVideo.srcObject = stream;
    await elements.sourceVideo.play();
    state.cameraReady = true;
    setStatusMessage("Camera connected. Hold a sign in frame.");
  } catch (error) {
    setStatusMessage("Camera access was denied or is unavailable.");
    elements.helperText.textContent = "The app needs webcam access to run live inference.";
  }
}

function captureFrameDataUrl() {
  const videoWidth = elements.sourceVideo.videoWidth;
  const videoHeight = elements.sourceVideo.videoHeight;
  if (!videoWidth || !videoHeight) {
    return null;
  }

  const canvas = elements.captureCanvas;
  const context = canvas.getContext("2d");
  canvas.width = 256;
  canvas.height = 256;
  context.drawImage(elements.sourceVideo, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.84);
}

async function predictOnce() {
  if (!state.cameraReady || state.inFlight || !state.status?.model_ready) {
    return;
  }

  const image = captureFrameDataUrl();
  if (!image) {
    return;
  }

  state.inFlight = true;
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image }),
    });
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.detail || "Prediction failed.");
    }

    state.currentLabel = payload.label || null;
    elements.primaryLabel.textContent = payload.label || "--";
    elements.confidenceValue.textContent = percent(payload.confidence);
    elements.latencyBadge.textContent = `Latency ${Math.round(payload.processing_ms || 0)} ms`;
    elements.helperText.textContent = payload.message || "Prediction updated.";
    renderPredictionList(payload.top_predictions || []);
    renderExampleCards(state.status.examples || [], state.currentLabel);
    setStatusMessage(`Live prediction running · ${payload.label || "no label"}`);
  } catch (error) {
    setStatusMessage(error.message);
  } finally {
    state.inFlight = false;
  }
}

async function boot() {
  await refreshStatus();
  renderPredictionList([]);
  await startCamera();
  window.setInterval(() => {
    void predictOnce();
  }, 480);
}

elements.reloadModelButton.addEventListener("click", () => {
  void reloadModel();
});

void boot();
