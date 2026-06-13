import {
  FaceLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";
import { saveRecording, getAllRecordings, deleteRecording } from "./db.js";

const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const stage = document.querySelector(".stage");
const statusEl = document.getElementById("status");

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const recordBtn = document.getElementById("recordBtn");
const recIndicator = document.getElementById("recIndicator");
const recTime = document.getElementById("recTime");
const meshToggle = document.getElementById("meshToggle");
const boxToggle = document.getElementById("boxToggle");
const mirrorToggle = document.getElementById("mirrorToggle");

const faceCountEl = document.getElementById("faceCount");
const fpsEl = document.getElementById("fps");
const smileBar = document.getElementById("smileBar");
const blinkLeftBar = document.getElementById("blinkLeftBar");
const blinkRightBar = document.getElementById("blinkRightBar");

const gallery = document.getElementById("gallery");
const galleryCount = document.getElementById("galleryCount");
const galleryEmpty = document.getElementById("galleryEmpty");
const playerModal = document.getElementById("playerModal");
const playerVideo = document.getElementById("playerVideo");
const playerMeta = document.getElementById("playerMeta");
const modalClose = document.getElementById("modalClose");
const modalBackdrop = document.getElementById("modalBackdrop");

let faceLandmarker = null;
let stream = null;
let running = false;
let rafId = null;
let lastVideoTime = -1;
let drawingUtils = null;

// FPS tracking
let frames = 0;
let fpsLastUpdate = performance.now();

// Recording state
let recordCanvas = null;
let recordCtx = null;
let mediaRecorder = null;
let recordedChunks = [];
let recording = false;
let recordStartTime = 0;
let recTimerId = null;
let activePlayerUrl = null;

function setStatus(msg, show = true) {
  statusEl.textContent = msg;
  statusEl.style.display = show ? "block" : "none";
}

async function createLandmarker() {
  setStatus("Loading model…");
  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
  );
  faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numFaces: 5,
    outputFaceBlendshapes: true,
  });
}

async function start() {
  startBtn.disabled = true;
  try {
    if (!faceLandmarker) await createLandmarker();

    setStatus("Requesting camera…");
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720, facingMode: "user" },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    drawingUtils = new DrawingUtils(ctx);

    // Offscreen canvas that composites video + overlay for recording
    recordCanvas = document.createElement("canvas");
    recordCanvas.width = video.videoWidth;
    recordCanvas.height = video.videoHeight;
    recordCtx = recordCanvas.getContext("2d");

    running = true;
    stopBtn.disabled = false;
    recordBtn.disabled = false;
    setStatus("", false);
    renderLoop();
  } catch (err) {
    console.error(err);
    setStatus("Error: " + err.message);
    startBtn.disabled = false;
  }
}

function stop() {
  if (recording) stopRecording();
  running = false;
  if (rafId) cancelAnimationFrame(rafId);
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  video.srcObject = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  startBtn.disabled = false;
  stopBtn.disabled = true;
  recordBtn.disabled = true;
  faceCountEl.textContent = "0";
  fpsEl.textContent = "0";
  setStatus("Camera stopped");
}

function blendshapeScore(blendshapes, name) {
  if (!blendshapes) return 0;
  const cat = blendshapes.categories.find((c) => c.categoryName === name);
  return cat ? cat.score : 0;
}

function renderLoop() {
  if (!running) return;

  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const results = faceLandmarker.detectForVideo(video, performance.now());

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const landmarksList = results.faceLandmarks || [];
    faceCountEl.textContent = String(landmarksList.length);

    for (const landmarks of landmarksList) {
      if (meshToggle.checked) {
        drawingUtils.drawConnectors(
          landmarks,
          FaceLandmarker.FACE_LANDMARKS_TESSELATION,
          { color: "rgba(45,212,191,0.25)", lineWidth: 0.5 }
        );
        drawingUtils.drawConnectors(
          landmarks,
          FaceLandmarker.FACE_LANDMARKS_FACE_OVAL,
          { color: "#2dd4bf", lineWidth: 2 }
        );
        drawingUtils.drawConnectors(
          landmarks,
          FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
          { color: "#fbbf24", lineWidth: 1.5 }
        );
        drawingUtils.drawConnectors(
          landmarks,
          FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
          { color: "#fbbf24", lineWidth: 1.5 }
        );
        drawingUtils.drawConnectors(
          landmarks,
          FaceLandmarker.FACE_LANDMARKS_LIPS,
          { color: "#f472b6", lineWidth: 1.5 }
        );
      }

      if (boxToggle.checked) {
        drawBoundingBox(landmarks);
      }
    }

    // Blendshapes from the first detected face
    const shapes = results.faceBlendshapes && results.faceBlendshapes[0];
    const smile =
      (blendshapeScore(shapes, "mouthSmileLeft") +
        blendshapeScore(shapes, "mouthSmileRight")) /
      2;
    smileBar.style.width = (smile * 100).toFixed(0) + "%";
    blinkLeftBar.style.width =
      (blendshapeScore(shapes, "eyeBlinkLeft") * 100).toFixed(0) + "%";
    blinkRightBar.style.width =
      (blendshapeScore(shapes, "eyeBlinkRight") * 100).toFixed(0) + "%";

    // Composite video + overlay onto the record canvas while recording
    if (recording && recordCtx) {
      recordCtx.save();
      recordCtx.clearRect(0, 0, recordCanvas.width, recordCanvas.height);
      if (mirrorToggle.checked) {
        recordCtx.translate(recordCanvas.width, 0);
        recordCtx.scale(-1, 1);
      }
      recordCtx.drawImage(video, 0, 0, recordCanvas.width, recordCanvas.height);
      recordCtx.drawImage(canvas, 0, 0, recordCanvas.width, recordCanvas.height);
      recordCtx.restore();
    }

    // FPS
    frames++;
    const now = performance.now();
    if (now - fpsLastUpdate >= 500) {
      fpsEl.textContent = Math.round((frames * 1000) / (now - fpsLastUpdate));
      frames = 0;
      fpsLastUpdate = now;
    }
  }

  rafId = requestAnimationFrame(renderLoop);
}

function drawBoundingBox(landmarks) {
  let minX = 1, minY = 1, maxX = 0, maxY = 0;
  for (const pt of landmarks) {
    minX = Math.min(minX, pt.x);
    minY = Math.min(minY, pt.y);
    maxX = Math.max(maxX, pt.x);
    maxY = Math.max(maxY, pt.y);
  }
  const x = minX * canvas.width;
  const y = minY * canvas.height;
  const w = (maxX - minX) * canvas.width;
  const h = (maxY - minY) * canvas.height;
  ctx.strokeStyle = "#2dd4bf";
  ctx.lineWidth = 2;
  ctx.strokeRect(x, y, w, h);
}

// ---- Recording ----

function pickMimeType() {
  const candidates = [
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
    "video/mp4",
  ];
  for (const t of candidates) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(t)) return t;
  }
  return "";
}

function formatDuration(seconds) {
  const s = Math.max(0, Math.round(seconds));
  const m = Math.floor(s / 60);
  return `${m}:${String(s % 60).padStart(2, "0")}`;
}

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function startRecording() {
  if (!recordCanvas) return;
  const mimeType = pickMimeType();
  const streamToRecord = recordCanvas.captureStream(30);
  recordedChunks = [];
  mediaRecorder = new MediaRecorder(
    streamToRecord,
    mimeType ? { mimeType, videoBitsPerSecond: 4_000_000 } : undefined
  );
  mediaRecorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) recordedChunks.push(e.data);
  };
  mediaRecorder.onstop = onRecordingStop;
  mediaRecorder.start();

  recording = true;
  recordStartTime = performance.now();
  recordBtn.textContent = "■ Stop recording";
  recordBtn.classList.add("recording");
  recIndicator.hidden = false;
  recTimerId = setInterval(() => {
    recTime.textContent = formatDuration((performance.now() - recordStartTime) / 1000);
  }, 250);
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
  recording = false;
  clearInterval(recTimerId);
  recIndicator.hidden = true;
  recTime.textContent = "0:00";
  recordBtn.textContent = "● Record";
  recordBtn.classList.remove("recording");
}

async function onRecordingStop() {
  const type = (mediaRecorder && mediaRecorder.mimeType) || "video/webm";
  const blob = new Blob(recordedChunks, { type });
  recordedChunks = [];
  if (blob.size === 0) return;

  const duration = (performance.now() - recordStartTime) / 1000;
  const thumbnail = recordCanvas.toDataURL("image/jpeg", 0.6);
  const id = Date.now();
  const record = {
    id,
    name: new Date(id).toLocaleString(),
    createdAt: id,
    duration,
    size: blob.size,
    type,
    thumbnail,
    blob,
  };
  await saveRecording(record);
  await refreshGallery();
}

function toggleRecording() {
  if (recording) stopRecording();
  else startRecording();
}

// ---- Gallery ----

async function refreshGallery() {
  const recordings = await getAllRecordings();
  galleryCount.textContent = String(recordings.length);
  galleryEmpty.style.display = recordings.length ? "none" : "block";
  gallery.innerHTML = "";
  for (const rec of recordings) gallery.appendChild(renderClip(rec));
}

function renderClip(rec) {
  const el = document.createElement("div");
  el.className = "clip";

  const ext = rec.type.includes("mp4") ? "mp4" : "webm";
  el.innerHTML = `
    <div class="clip-thumb">
      <img src="${rec.thumbnail}" alt="" />
      <div class="play-icon">▶</div>
      <span class="clip-dur">${formatDuration(rec.duration)}</span>
    </div>
    <div class="clip-info">
      <p class="clip-name">${rec.name}</p>
      <span class="clip-sub">${formatSize(rec.size)}</span>
    </div>
    <div class="clip-actions">
      <button data-act="play">Play</button>
      <button data-act="download">Download</button>
      <button data-act="delete" class="danger">Delete</button>
    </div>
  `;

  el.querySelector(".clip-thumb").addEventListener("click", () => openPlayer(rec));
  el.querySelector('[data-act="play"]').addEventListener("click", () => openPlayer(rec));
  el.querySelector('[data-act="download"]').addEventListener("click", () => {
    const url = URL.createObjectURL(rec.blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `face-tracker-${rec.id}.${ext}`;
    a.click();
    URL.revokeObjectURL(url);
  });
  el.querySelector('[data-act="delete"]').addEventListener("click", async () => {
    if (confirm("Delete this recording?")) {
      await deleteRecording(rec.id);
      await refreshGallery();
    }
  });

  return el;
}

function openPlayer(rec) {
  if (activePlayerUrl) URL.revokeObjectURL(activePlayerUrl);
  activePlayerUrl = URL.createObjectURL(rec.blob);
  playerVideo.src = activePlayerUrl;
  playerMeta.textContent = `${rec.name} · ${formatDuration(rec.duration)} · ${formatSize(rec.size)}`;
  playerModal.hidden = false;
  playerVideo.play().catch(() => {});
}

function closePlayer() {
  playerVideo.pause();
  playerVideo.removeAttribute("src");
  playerVideo.load();
  playerModal.hidden = true;
  if (activePlayerUrl) {
    URL.revokeObjectURL(activePlayerUrl);
    activePlayerUrl = null;
  }
}

// ---- Event wiring ----

mirrorToggle.addEventListener("change", () => {
  stage.classList.toggle("mirror", mirrorToggle.checked);
});
stage.classList.toggle("mirror", mirrorToggle.checked);

startBtn.addEventListener("click", start);
stopBtn.addEventListener("click", stop);
recordBtn.addEventListener("click", toggleRecording);
modalClose.addEventListener("click", closePlayer);
modalBackdrop.addEventListener("click", closePlayer);

refreshGallery();
