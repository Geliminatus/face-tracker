import {
  FaceLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";

const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const stage = document.querySelector(".stage");
const statusEl = document.getElementById("status");

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const meshToggle = document.getElementById("meshToggle");
const boxToggle = document.getElementById("boxToggle");
const mirrorToggle = document.getElementById("mirrorToggle");

const faceCountEl = document.getElementById("faceCount");
const fpsEl = document.getElementById("fps");
const smileBar = document.getElementById("smileBar");
const blinkLeftBar = document.getElementById("blinkLeftBar");
const blinkRightBar = document.getElementById("blinkRightBar");

let faceLandmarker = null;
let stream = null;
let running = false;
let rafId = null;
let lastVideoTime = -1;
let drawingUtils = null;

// FPS tracking
let frames = 0;
let fpsLastUpdate = performance.now();

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

    running = true;
    stopBtn.disabled = false;
    setStatus("", false);
    renderLoop();
  } catch (err) {
    console.error(err);
    setStatus("Error: " + err.message);
    startBtn.disabled = false;
  }
}

function stop() {
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

mirrorToggle.addEventListener("change", () => {
  stage.classList.toggle("mirror", mirrorToggle.checked);
});
stage.classList.toggle("mirror", mirrorToggle.checked);

startBtn.addEventListener("click", start);
stopBtn.addEventListener("click", stop);
