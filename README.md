# Face Tracker

Real-time, in-browser face landmark tracking using your webcam. Built on
[MediaPipe Tasks Vision](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) —
no backend, no install, no data leaves your machine.

## Features

- Tracks up to **5 faces** at once with a 478-point face mesh
- Eye, lip, and face-oval contours highlighted
- Bounding box per face
- Live **blendshape** readouts: smile and per-eye blink
- FPS + face-count stats
- Toggle mesh / box / mirror, all client-side

## Run it

The app uses ES modules and the webcam, so it must be served over `http://`
(not opened as a `file://` URL). Browsers also require a **secure context** —
`localhost` counts as secure, so a local server is all you need.

```bash
cd face-tracker

# pick any one:
python3 -m http.server 8080
# or
npx serve .
```

Then open <http://localhost:8080> and click **Start camera**. Grant the camera
permission when prompted.

## How it works

- `app.js` loads the MediaPipe `FaceLandmarker` (WASM + GPU delegate) from a CDN
  and the model `.task` file from Google's model store.
- Each animation frame, the current video frame is passed to
  `detectForVideo()`, which returns normalized landmarks and blendshape scores.
- `DrawingUtils` paints the mesh/contours onto a canvas overlaid on the video.

First load downloads the WASM runtime and model (~a few MB), then it's cached.

## Notes

- Needs a modern browser with WebGL/WebGPU (Chrome, Edge, Safari 17+, Firefox).
- All processing is local; the CDN is only used to fetch the library and model.
