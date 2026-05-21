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

## Limitations

- Blink threshold is in normalized image coords, so it's sensitive to face-to-camera distance — sit ~50–80 cm away or lower `BLINK_THRESHOLD`.
- No calibration: head movement also moves the cursor (same as the Python CLI).
- Requires HTTPS or localhost — `getUserMedia` is blocked on plain HTTP in production.

## Tweak

Constants at the top of `EyeCursor.jsx` (keep in sync with [`../main.py`](../main.py)):

```js
const SMOOTHING = 0.3;          // 0 = no movement, 1 = no smoothing
const BLINK_THRESHOLD = 0.005;  // smaller = stricter
const CLICK_COOLDOWN_MS = 1000; // min ms between blinks
```
