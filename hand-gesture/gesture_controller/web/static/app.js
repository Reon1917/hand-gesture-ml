const state = {
  mode: "collect",
  labels: [],
  inFlight: false,
  cameraReady: false,
  currentStatus: null,
  handConnections: [],
};

const elements = {
  sourceVideo: document.getElementById("sourceVideo"),
  overlayCanvas: document.getElementById("overlayCanvas"),
  captureCanvas: document.getElementById("captureCanvas"),
  handModelChip: document.getElementById("handModelChip"),
  faceModelChip: document.getElementById("faceModelChip"),
  latencyChip: document.getElementById("latencyChip"),
  runtimeState: document.getElementById("runtimeState"),
  predictionValue: document.getElementById("predictionValue"),
  confidenceValue: document.getElementById("confidenceValue"),
  activeGestureValue: document.getElementById("activeGestureValue"),
  lastStatus: document.getElementById("lastStatus"),
  lastAction: document.getElementById("lastAction"),
  gestureButtons: document.getElementById("gestureButtons"),
  bindingsList: document.getElementById("bindingsList"),
  trainButton: document.getElementById("trainButton"),
  resetRuntimeButton: document.getElementById("resetRuntimeButton"),
  accuracyBadge: document.getElementById("accuracyBadge"),
  reportPreview: document.getElementById("reportPreview"),
  confusionMatrix: document.getElementById("confusionMatrix"),
  faceAccuracyBadge: document.getElementById("faceAccuracyBadge"),
  faceReportPreview: document.getElementById("faceReportPreview"),
  faceConfusionMatrix: document.getElementById("faceConfusionMatrix"),
  handFingerCount: document.getElementById("handFingerCount"),
  handPinchTarget: document.getElementById("handPinchTarget"),
  handPinchStrength: document.getElementById("handPinchStrength"),
  handPalmRotation: document.getElementById("handPalmRotation"),
  handOpenness: document.getElementById("handOpenness"),
  fingerStates: document.getElementById("fingerStates"),
  faceStateValue: document.getElementById("faceStateValue"),
  faceExpressionValue: document.getElementById("faceExpressionValue"),
  faceAttentionValue: document.getElementById("faceAttentionValue"),
  faceExpressionConfidenceValue: document.getElementById("faceExpressionConfidenceValue"),
  facePoseValue: document.getElementById("facePoseValue"),
  faceSmileValue: document.getElementById("faceSmileValue"),
  faceMouthValue: document.getElementById("faceMouthValue"),
  faceBlinkValue: document.getElementById("faceBlinkValue"),
  handDatasetHelp: document.getElementById("handDatasetHelp"),
  faceDatasetHelp: document.getElementById("faceDatasetHelp"),
  eventLog: document.getElementById("eventLog"),
};

function addLog(message) {
  const item = document.createElement("li");
  const stamp = new Date().toLocaleTimeString();
  item.innerHTML = `<span class="timestamp">${stamp}</span>${message}`;
  elements.eventLog.prepend(item);
  while (elements.eventLog.children.length > 14) {
    elements.eventLog.removeChild(elements.eventLog.lastChild);
  }
}

function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll(".mode-btn").forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === mode);
  });
  addLog(`Mode switched to <code>${mode}</code>.`);
}

function renderBindings(bindings) {
  elements.bindingsList.replaceChildren();
  Object.entries(bindings).forEach(([label, action]) => {
    const row = document.createElement("div");
    row.className = "binding-row";
    row.innerHTML = `<strong>${label}</strong><code>${action}</code>`;
    elements.bindingsList.appendChild(row);
  });
}

function renderGestureButtons(labels, counts) {
  elements.gestureButtons.replaceChildren();
  labels.forEach((label) => {
    const button = document.createElement("button");
    button.className = "gesture-button";
    button.dataset.label = label;
    button.innerHTML = `
      <span class="count-pill">${counts[label] ?? 0}</span>
      <strong>${label}</strong>
      <span>Save the latest detected landmarks</span>
    `;
    button.addEventListener("click", async () => {
      try {
        const response = await fetch("/api/save-sample", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ label }),
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.error || "Failed to save sample.");
        }
        elements.lastStatus.textContent = payload.last_status;
        addLog(`Saved sample for <code>${label}</code>.`);
        await refreshStatus();
      } catch (error) {
        elements.lastStatus.textContent = error.message;
        addLog(`Save failed: ${error.message}`);
      }
    });
    elements.gestureButtons.appendChild(button);
  });
}

function renderMetrics(metricsSummary, reportPreview, confusionMatrixUrl, badgeEl, reportEl, matrixEl, emptyLabel) {
  if (!metricsSummary) {
    badgeEl.textContent = emptyLabel;
    reportEl.textContent = reportPreview;
    matrixEl.removeAttribute("src");
    return;
  }

  if (typeof metricsSummary.accuracy === "number") {
    badgeEl.textContent = `Accuracy ${metricsSummary.accuracy.toFixed(3)}`;
  } else {
    badgeEl.textContent = "Insufficient holdout data";
  }
  reportEl.textContent = reportPreview;
  if (confusionMatrixUrl) {
    matrixEl.src = confusionMatrixUrl;
  } else {
    matrixEl.removeAttribute("src");
  }
}

function renderFingerStates(states) {
  elements.fingerStates.replaceChildren();
  Object.entries(states || {}).forEach(([finger, status]) => {
    const pill = document.createElement("span");
    pill.className = "finger-pill";
    pill.innerHTML = `<strong>${finger}</strong>${status}`;
    elements.fingerStates.appendChild(pill);
  });
}

function renderStatus(payload) {
  state.currentStatus = payload;
  state.labels = payload.labels;
  state.handConnections = payload.hand_connections || [];

  elements.handModelChip.textContent = payload.model_ready ? "Hand model ready" : "Hand model unavailable";
  elements.handModelChip.classList.toggle("chip-accent", payload.model_ready);
  elements.handModelChip.classList.toggle("chip-warn", !payload.model_ready);

  elements.faceModelChip.textContent = payload.face_model_ready ? "Face model ready" : "Face model optional";
  elements.faceModelChip.classList.toggle("chip-accent", payload.face_model_ready);
  elements.faceModelChip.classList.toggle("chip-warn", Boolean(payload.face_detector_error));

  elements.lastStatus.textContent = payload.last_status;
  if (payload.detector_error) {
    elements.lastStatus.textContent = payload.detector_error;
  }

  elements.handDatasetHelp.textContent = payload.hand_dataset_help || "";
  elements.faceDatasetHelp.textContent = payload.face_dataset_help || "";

  renderBindings(payload.bindings);
  renderGestureButtons(payload.labels, payload.counts);
  renderMetrics(
    payload.metrics_summary,
    payload.report_preview,
    payload.confusion_matrix_url,
    elements.accuracyBadge,
    elements.reportPreview,
    elements.confusionMatrix,
    "No hand metrics yet",
  );
  renderMetrics(
    payload.face_metrics_summary,
    payload.face_report_preview,
    payload.face_confusion_matrix_url,
    elements.faceAccuracyBadge,
    elements.faceReportPreview,
    elements.faceConfusionMatrix,
    "Optional face model",
  );

  const analysis = payload.analysis || {};
  const details = analysis.details || {};
  elements.runtimeState.textContent = analysis.state || "idle";
  elements.predictionValue.textContent = analysis.prediction || "-";
  elements.confidenceValue.textContent = Number(analysis.confidence || 0).toFixed(2);
  elements.activeGestureValue.textContent = analysis.active_gesture || "-";
  elements.lastAction.textContent = analysis.last_action || "Waiting for a stable gesture";
  elements.handFingerCount.textContent = `${details.finger_count ?? 0}`;
  elements.handPinchTarget.textContent = details.pinch_target || "-";
  elements.handPinchStrength.textContent = Number(details.pinch_strength || 0).toFixed(2);
  elements.handPalmRotation.textContent = `${Math.round(Number(details.palm_rotation_deg || 0))}°`;
  elements.handOpenness.textContent = Number(details.openness || 0).toFixed(2);
  renderFingerStates(details.finger_states || {});

  const face = payload.face_analysis || {};
  elements.faceStateValue.textContent = face.state || "no_face";
  elements.faceExpressionValue.textContent = face.expression || "unavailable";
  elements.faceAttentionValue.textContent = face.attention || "unknown";
  elements.faceExpressionConfidenceValue.textContent = Number(face.expression_confidence || 0).toFixed(2);
  elements.facePoseValue.textContent = `Y ${Math.round(Number(face.yaw_deg || 0))} / P ${Math.round(Number(face.pitch_deg || 0))} / R ${Math.round(Number(face.roll_deg || 0))}`;
  elements.faceSmileValue.textContent = Number(face.smile || 0).toFixed(2);
  elements.faceMouthValue.textContent = Number(face.mouth_open || 0).toFixed(2);
  elements.faceBlinkValue.textContent = Number(face.blink || 0).toFixed(2);

  if (typeof payload.processing_ms === "number") {
    elements.latencyChip.textContent = `Latency ${payload.processing_ms.toFixed(0)} ms`;
  }
}

async function refreshStatus() {
  const response = await fetch("/api/status");
  const payload = await response.json();
  renderStatus(payload);
}

async function trainModel() {
  elements.trainButton.disabled = true;
  elements.trainButton.textContent = "Retraining…";
  try {
    const response = await fetch("/api/train", { method: "POST" });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Training failed.");
    }
    addLog("Hand model retrained from the browser.");
    await refreshStatus();
    elements.lastStatus.textContent = payload.last_status;
  } catch (error) {
    elements.lastStatus.textContent = error.message;
    addLog(`Training failed: ${error.message}`);
  } finally {
    elements.trainButton.disabled = false;
    elements.trainButton.textContent = "Retrain Hand Model";
  }
}

async function resetRuntime() {
  await fetch("/api/reset-runtime", { method: "POST" });
  addLog("Runtime state reset.");
  await refreshStatus();
}

async function startCamera() {
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
  addLog("Browser camera connected.");
}

function resizeOverlayCanvas() {
  const rect = elements.sourceVideo.getBoundingClientRect();
  if (!rect.width || !rect.height) {
    return;
  }
  const dpr = window.devicePixelRatio || 1;
  elements.overlayCanvas.width = Math.round(rect.width * dpr);
  elements.overlayCanvas.height = Math.round(rect.height * dpr);
  elements.overlayCanvas.style.width = `${rect.width}px`;
  elements.overlayCanvas.style.height = `${rect.height}px`;
}

function drawOverlay(handPoints, facePoints) {
  resizeOverlayCanvas();
  const ctx = elements.overlayCanvas.getContext("2d");
  const width = elements.overlayCanvas.width;
  const height = elements.overlayCanvas.height;
  ctx.clearRect(0, 0, width, height);

  if (handPoints && handPoints.length) {
    ctx.lineWidth = Math.max(2, width / 320);
    ctx.strokeStyle = "rgba(200, 255, 99, 0.75)";
    ctx.fillStyle = "rgba(255, 244, 214, 0.95)";

    for (const [start, end] of state.handConnections) {
      const a = handPoints[start];
      const b = handPoints[end];
      if (!a || !b) {
        continue;
      }
      ctx.beginPath();
      ctx.moveTo(a.x * width, a.y * height);
      ctx.lineTo(b.x * width, b.y * height);
      ctx.stroke();
    }

    for (const point of handPoints) {
      ctx.beginPath();
      ctx.arc(point.x * width, point.y * height, Math.max(4, width / 130), 0, Math.PI * 2);
      ctx.fill();
    }
  }

  if (facePoints && facePoints.length) {
    ctx.fillStyle = "rgba(255, 184, 108, 0.88)";
    for (const point of facePoints) {
      ctx.beginPath();
      ctx.arc(point.x * width, point.y * height, Math.max(3, width / 170), 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

function captureBlob(canvas, quality = 0.55) {
  return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", quality));
}

async function analyzeOnce() {
  if (!state.cameraReady || state.inFlight) {
    return;
  }

  const width = elements.sourceVideo.videoWidth;
  const height = elements.sourceVideo.videoHeight;
  if (!width || !height) {
    return;
  }

  state.inFlight = true;
  const ctx = elements.captureCanvas.getContext("2d");
  const targetWidth = 384;
  const targetHeight = Math.round((height / width) * targetWidth);
  elements.captureCanvas.width = targetWidth;
  elements.captureCanvas.height = targetHeight;
  ctx.drawImage(elements.sourceVideo, 0, 0, targetWidth, targetHeight);

  try {
    const blob = await captureBlob(elements.captureCanvas);
    if (!blob) {
      throw new Error("Unable to capture frame.");
    }
    const response = await fetch(`/api/analyze?mode=${encodeURIComponent(state.mode)}`, {
      method: "POST",
      headers: { "Content-Type": "image/jpeg" },
      body: blob,
    });
    const payload = await response.json();
    if (response.ok) {
      renderStatus(payload);
      drawOverlay(payload.hand_landmarks || [], payload.face_landmarks || []);
    } else {
      elements.lastStatus.textContent = payload.error || "Analysis failed.";
      drawOverlay([], []);
    }
  } catch (error) {
    elements.lastStatus.textContent = error.message;
    drawOverlay([], []);
  } finally {
    state.inFlight = false;
  }
}

async function bootstrap() {
  await refreshStatus();
  document.querySelectorAll(".mode-btn").forEach((button) => {
    button.addEventListener("click", () => setMode(button.dataset.mode));
  });
  elements.trainButton.addEventListener("click", trainModel);
  elements.resetRuntimeButton.addEventListener("click", resetRuntime);
  window.addEventListener("resize", () => drawOverlay([], []));
  await startCamera();
  window.setInterval(analyzeOnce, 120);
}

bootstrap().catch((error) => {
  elements.lastStatus.textContent = error.message;
  addLog(`Startup failed: ${error.message}`);
});
