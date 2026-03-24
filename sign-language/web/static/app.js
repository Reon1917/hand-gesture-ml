const MEDIAPIPE_MODULE_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";
const MEDIAPIPE_WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";
const HAND_LANDMARKER_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

const state = {
  cameraReady: false,
  inFlight: false,
  status: null,
  currentLabel: null,
  handLandmarker: null,
  handConnections: HAND_CONNECTIONS,
  handTrackerReady: false,
  handTrackerError: null,
};

const elements = {
  sourceVideo: document.getElementById("sourceVideo"),
  overlayCanvas: document.getElementById("overlayCanvas"),
  captureCanvas: document.getElementById("captureCanvas"),
  handCropCanvas: document.getElementById("handCropCanvas"),
  cropPreviewCanvas: document.getElementById("cropPreviewCanvas"),
  modelChip: document.getElementById("modelChip"),
  datasetChip: document.getElementById("datasetChip"),
  accuracyChip: document.getElementById("accuracyChip"),
  latencyBadge: document.getElementById("latencyBadge"),
  handChip: document.getElementById("handChip"),
  trackerValue: document.getElementById("trackerValue"),
  trackerMeta: document.getElementById("trackerMeta"),
  handednessValue: document.getElementById("handednessValue"),
  cropValue: document.getElementById("cropValue"),
  primaryLabel: document.getElementById("primaryLabel"),
  confidenceValue: document.getElementById("confidenceValue"),
  statusValue: document.getElementById("statusValue"),
  helperText: document.getElementById("helperText"),
  exampleGrid: document.getElementById("exampleGrid"),
  predictionList: document.getElementById("predictionList"),
  guidanceList: document.getElementById("guidanceList"),
  reloadModelButton: document.getElementById("reloadModelButton"),
};

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function percent(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return "--";
  }
  return `${(numeric * 100).toFixed(1)}%`;
}

function setStatusMessage(message) {
  elements.statusValue.textContent = message;
}

function setChipTone(element, tone) {
  element.classList.remove("is-live", "is-warn", "is-idle");
  if (tone) {
    element.classList.add(`is-${tone}`);
  }
}

function setTrackerPanel(status, meta, tone, handLabel = "--", cropLabel = "Crop --") {
  elements.trackerValue.textContent = status;
  elements.trackerMeta.textContent = meta;
  elements.handednessValue.textContent = handLabel;
  elements.cropValue.textContent = cropLabel;
  elements.handChip.textContent = status;
  setChipTone(elements.handChip, tone);
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
    empty.textContent = "Predictions appear here once the hand tracker has a clear crop.";
    elements.predictionList.appendChild(empty);
    return;
  }

  items.forEach((item) => {
    const confidence = Number(item.confidence || 0);
    const row = document.createElement("div");
    row.className = "prediction-item";
    row.innerHTML = `
      <div class="row">
        <strong>${item.label}</strong>
        <span>${percent(confidence)}</span>
      </div>
      <div class="bar">
        <div class="bar-fill" style="width:${Math.max(4, confidence * 100)}%"></div>
      </div>
    `;
    elements.predictionList.appendChild(row);
  });
}

function clearCropPreview() {
  const ctx = elements.cropPreviewCanvas.getContext("2d");
  ctx.clearRect(0, 0, elements.cropPreviewCanvas.width, elements.cropPreviewCanvas.height);
}

function syncOverlayCanvas() {
  const width = elements.sourceVideo.videoWidth;
  const height = elements.sourceVideo.videoHeight;
  if (!width || !height) {
    return;
  }
  elements.overlayCanvas.width = width;
  elements.overlayCanvas.height = height;
}

function drawGuideFrame() {
  const ctx = elements.overlayCanvas.getContext("2d");
  const width = elements.overlayCanvas.width;
  const height = elements.overlayCanvas.height;
  if (!width || !height) {
    return;
  }

  ctx.clearRect(0, 0, width, height);
  const guideSize = Math.min(width, height) * 0.44;
  const x = (width - guideSize) / 2;
  const y = (height - guideSize) / 2;
  const corner = guideSize * 0.12;

  ctx.strokeStyle = "rgba(42, 103, 247, 0.42)";
  ctx.lineWidth = Math.max(3, width / 250);
  ctx.setLineDash([12, 10]);
  ctx.strokeRect(x, y, guideSize, guideSize);
  ctx.setLineDash([]);

  ctx.strokeStyle = "rgba(42, 103, 247, 0.88)";
  const corners = [
    [x, y, x + corner, y],
    [x, y, x, y + corner],
    [x + guideSize, y, x + guideSize - corner, y],
    [x + guideSize, y, x + guideSize, y + corner],
    [x, y + guideSize, x + corner, y + guideSize],
    [x, y + guideSize, x, y + guideSize - corner],
    [x + guideSize, y + guideSize, x + guideSize - corner, y + guideSize],
    [x + guideSize, y + guideSize, x + guideSize, y + guideSize - corner],
  ];

  for (const [x1, y1, x2, y2] of corners) {
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  }
}

function drawHandOverlay(hand) {
  syncOverlayCanvas();
  const ctx = elements.overlayCanvas.getContext("2d");
  const width = elements.overlayCanvas.width;
  const height = elements.overlayCanvas.height;
  ctx.clearRect(0, 0, width, height);

  if (!hand) {
    drawGuideFrame();
    return;
  }

  ctx.strokeStyle = "rgba(42, 103, 247, 0.95)";
  ctx.fillStyle = "rgba(42, 103, 247, 0.88)";
  ctx.lineWidth = Math.max(2.5, width / 260);

  for (const [start, end] of state.handConnections) {
    const a = hand.points[start];
    const b = hand.points[end];
    if (!a || !b) {
      continue;
    }
    ctx.beginPath();
    ctx.moveTo(a.x * width, a.y * height);
    ctx.lineTo(b.x * width, b.y * height);
    ctx.stroke();
  }

  for (const point of hand.points) {
    ctx.beginPath();
    ctx.arc(point.x * width, point.y * height, Math.max(3.5, width / 150), 0, Math.PI * 2);
    ctx.fill();
  }

  const crop = buildSquareCrop(hand.box, width, height);
  ctx.strokeStyle = "rgba(28, 138, 88, 0.96)";
  ctx.lineWidth = Math.max(3, width / 220);
  ctx.setLineDash([12, 8]);
  ctx.strokeRect(crop.left, crop.top, crop.size, crop.size);
  ctx.setLineDash([]);
}

function buildSquareCrop(box, videoWidth, videoHeight) {
  const xMin = box.xMin * videoWidth;
  const xMax = box.xMax * videoWidth;
  const yMin = box.yMin * videoHeight;
  const yMax = box.yMax * videoHeight;
  const width = Math.max(1, xMax - xMin);
  const height = Math.max(1, yMax - yMin);
  const centerX = (xMin + xMax) / 2;
  const centerY = (yMin + yMax) / 2;
  const size = Math.min(Math.max(width, height) * 1.7, Math.min(videoWidth, videoHeight));

  let left = centerX - size / 2;
  let top = centerY - size / 2;
  left = clamp(left, 0, Math.max(0, videoWidth - size));
  top = clamp(top, 0, Math.max(0, videoHeight - size));

  return { left, top, size };
}

function extractHandDetection(results) {
  if (!results?.landmarks?.length) {
    return null;
  }

  const points = results.landmarks[0].map((point) => ({
    x: clamp(point.x, 0, 1),
    y: clamp(point.y, 0, 1),
    z: point.z,
  }));
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const handedness = results.handednesses?.[0]?.[0];

  return {
    points,
    box: {
      xMin: Math.min(...xs),
      xMax: Math.max(...xs),
      yMin: Math.min(...ys),
      yMax: Math.max(...ys),
    },
    handednessLabel: handedness?.categoryName || handedness?.displayName || "Unknown",
    handednessScore: Number(handedness?.score || 0),
    coverage: Math.max(0, (Math.max(...xs) - Math.min(...xs)) * (Math.max(...ys) - Math.min(...ys))),
  };
}

function drawCropPreview() {
  const ctx = elements.cropPreviewCanvas.getContext("2d");
  ctx.clearRect(0, 0, elements.cropPreviewCanvas.width, elements.cropPreviewCanvas.height);
  ctx.drawImage(
    elements.handCropCanvas,
    0,
    0,
    elements.cropPreviewCanvas.width,
    elements.cropPreviewCanvas.height,
  );
}

function captureFullFrameDataUrl() {
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
  clearCropPreview();
  return canvas.toDataURL("image/jpeg", 0.84);
}

function captureHandCropDataUrl(hand) {
  const videoWidth = elements.sourceVideo.videoWidth;
  const videoHeight = elements.sourceVideo.videoHeight;
  if (!videoWidth || !videoHeight) {
    return null;
  }

  const crop = buildSquareCrop(hand.box, videoWidth, videoHeight);
  const canvas = elements.handCropCanvas;
  const context = canvas.getContext("2d");
  canvas.width = 256;
  canvas.height = 256;
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.drawImage(
    elements.sourceVideo,
    crop.left,
    crop.top,
    crop.size,
    crop.size,
    0,
    0,
    canvas.width,
    canvas.height,
  );
  drawCropPreview();
  return {
    image: canvas.toDataURL("image/jpeg", 0.9),
    crop,
  };
}

async function initHandTracker() {
  setTrackerPanel("Loading", "Preparing browser hand detection.", "idle");

  try {
    const mediapipe = await import(MEDIAPIPE_MODULE_URL);
    const { FilesetResolver, HandLandmarker } = mediapipe;
    const fileset = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_URL);
    state.handLandmarker = await HandLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath: HAND_LANDMARKER_MODEL_URL,
      },
      runningMode: "VIDEO",
      numHands: 1,
      minHandDetectionConfidence: 0.45,
      minHandPresenceConfidence: 0.45,
      minTrackingConfidence: 0.45,
    });
    state.handConnections = HandLandmarker.HAND_CONNECTIONS || HAND_CONNECTIONS;
    state.handTrackerReady = true;
    state.handTrackerError = null;
    setTrackerPanel(
      "Searching",
      "Tracker ready. Move one hand into the guide box and keep it clear of your face.",
      "idle",
    );
    drawGuideFrame();
  } catch (error) {
    state.handTrackerReady = false;
    state.handTrackerError = error;
    setTrackerPanel(
      "Tracker offline",
      "The browser hand detector could not load, so the app will fall back to the full webcam frame.",
      "warn",
    );
  }
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
        width: { ideal: 960 },
        height: { ideal: 720 },
        facingMode: "user",
      },
      audio: false,
    });
    elements.sourceVideo.srcObject = stream;
    await elements.sourceVideo.play();
    state.cameraReady = true;
    syncOverlayCanvas();
    drawGuideFrame();
    setStatusMessage("Camera connected. Hold a sign in frame.");
  } catch (error) {
    setStatusMessage("Camera access was denied or is unavailable.");
    elements.helperText.textContent = "The app needs webcam access to run live inference.";
  }
}

function resetPredictionView(message) {
  state.currentLabel = null;
  elements.primaryLabel.textContent = "--";
  elements.confidenceValue.textContent = "--";
  elements.latencyBadge.textContent = "Latency --";
  elements.helperText.textContent = message;
  renderPredictionList([]);
  renderExampleCards(state.status?.examples || [], null);
}

async function predictOnce() {
  if (!state.cameraReady || state.inFlight || !state.status?.model_ready) {
    return;
  }

  let hand = null;
  if (state.handTrackerReady && state.handLandmarker) {
    const results = state.handLandmarker.detectForVideo(elements.sourceVideo, performance.now());
    hand = extractHandDetection(results);
  }

  drawHandOverlay(hand);

  if (hand) {
    setTrackerPanel(
      "Hand locked",
      "Blue landmarks and the green crop box mean the browser is isolating your hand correctly.",
      "live",
      `${hand.handednessLabel} · ${percent(hand.handednessScore)}`,
      `Crop ${Math.round(hand.coverage * 100)}% of frame`,
    );
  } else if (state.handTrackerReady) {
    setTrackerPanel(
      "Searching",
      "No hand detected yet. Move one hand closer to the lens and keep it away from your face.",
      "idle",
    );
  }

  let imagePayload = null;
  if (hand) {
    imagePayload = captureHandCropDataUrl(hand)?.image || null;
  } else if (state.handTrackerError) {
    imagePayload = captureFullFrameDataUrl();
  } else {
    resetPredictionView("Waiting for a clear hand. Keep one hand inside the guide frame until the tracker locks.");
    setStatusMessage("Hand tracker is searching.");
    return;
  }

  if (!imagePayload) {
    return;
  }

  state.inFlight = true;
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: imagePayload }),
    });
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.detail || "Prediction failed.");
    }

    state.currentLabel = payload.label || null;
    elements.primaryLabel.textContent = payload.label || "--";
    elements.confidenceValue.textContent = percent(payload.confidence);
    elements.latencyBadge.textContent = `Latency ${Math.round(payload.processing_ms || 0)} ms`;
    elements.helperText.textContent = hand
      ? "The crop preview tile is the exact image sent to the classifier. If that tile looks right, retraining usually helps."
      : payload.message || "Prediction updated.";
    renderPredictionList(payload.top_predictions || []);
    renderExampleCards(state.status.examples || [], state.currentLabel);
    setStatusMessage(
      hand
        ? `Live prediction running · ${payload.label || "no label"}`
        : `Tracker fallback active · ${payload.label || "no label"}`,
    );
  } catch (error) {
    setStatusMessage(error.message);
  } finally {
    state.inFlight = false;
  }
}

async function boot() {
  await refreshStatus();
  renderPredictionList([]);
  clearCropPreview();
  await initHandTracker();
  await startCamera();
  window.addEventListener("resize", () => {
    syncOverlayCanvas();
    if (state.cameraReady) {
      drawGuideFrame();
    }
  });
  window.setInterval(() => {
    void predictOnce();
  }, 420);
}

elements.reloadModelButton.addEventListener("click", () => {
  void reloadModel();
});

void boot();
