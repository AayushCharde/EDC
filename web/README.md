# EDC Web

Browser version of EDC — iris tracking + blink-to-click rendered on `<canvas>`, no native cursor control (browsers can't move the OS cursor).

Uses [MediaPipe Tasks Vision](https://developers.google.com/mediapipe/solutions/vision/face_landmarker/web_js) (WebAssembly + WebGL) for the face landmark model.

## Run

```bash
cd web
npm install
npm run dev
```

Open the URL printed by Vite, allow camera access, and a virtual cursor will follow your iris. Closing your left eye flashes the frame and increments the blink counter.

## Files

- `src/EyeCursor.jsx` — the entire component (camera, MediaPipe, canvas draw loop).
- `src/main.jsx` — React 18 mount point.
- `index.html` — Vite entry.

## Tweak

Constants at the top of `EyeCursor.jsx`:

```js
const SMOOTHING = 0.3;          // 0 = no movement, 1 = no smoothing
const BLINK_THRESHOLD = 0.005;  // smaller = stricter
const CLICK_COOLDOWN_MS = 1000; // min ms between blinks
```
