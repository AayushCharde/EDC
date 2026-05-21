import { useCallback, useEffect, useRef, useState } from "react";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

// Keep in sync with main.py — same landmark indices and tuning constants.
const RIGHT_IRIS = [474, 475, 476, 477];
const IRIS_TRACKING = 475;
const LEFT_EYE_UPPER = 159;
const LEFT_EYE_LOWER = 145;

const SMOOTHING = 0.3;
const BLINK_THRESHOLD = 0.005;
const CLICK_COOLDOWN_MS = 1000;
const FPS_UPDATE_INTERVAL_MS = 500;

async function createLandmarker(resolver) {
  const baseConfig = {
    modelAssetPath:
      "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
  };
  try {
    return await FaceLandmarker.createFromOptions(resolver, {
      baseOptions: { ...baseConfig, delegate: "GPU" },
      outputFaceBlendshapes: false,
      runningMode: "VIDEO",
      numFaces: 1,
    });
  } catch (err) {
    console.warn("GPU delegate unavailable, falling back to CPU:", err);
    return await FaceLandmarker.createFromOptions(resolver, {
      baseOptions: { ...baseConfig, delegate: "CPU" },
      outputFaceBlendshapes: false,
      runningMode: "VIDEO",
      numFaces: 1,
    });
  }
}

export default function EyeCursor() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const smoothed = useRef({ x: null, y: null });
  const lastClickAt = useRef(0);

  // Mutable handles so the Stop button (outside the effect) can tear things down.
  const handlesRef = useRef({
    landmarker: null,
    stream: null,
    rafId: 0,
    cancelled: false,
  });

  const [status, setStatus] = useState("idle");
  const [running, setRunning] = useState(false);
  const [clicks, setClicks] = useState(0);
  const [fps, setFps] = useState(0);

  const stop = useCallback(() => {
    const h = handlesRef.current;
    h.cancelled = true;
    cancelAnimationFrame(h.rafId);
    h.rafId = 0;
    if (h.stream) {
      h.stream.getTracks().forEach((t) => t.stop());
      h.stream = null;
    }
    if (h.landmarker) {
      try {
        h.landmarker.close();
      } catch {
        // ignore — already closed
      }
      h.landmarker = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext("2d");
      ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
    smoothed.current = { x: null, y: null };
    setRunning(false);
    setStatus("stopped");
    setFps(0);
  }, []);

  const start = useCallback(async () => {
    if (handlesRef.current.landmarker || handlesRef.current.stream) return;
    handlesRef.current.cancelled = false;
    setStatus("loading model…");
    setRunning(true);

    let lastFrameAt = performance.now();
    let smoothedFps = 0;
    let lastFpsPublishAt = 0;

    const isCancelled = () => handlesRef.current.cancelled;

    try {
      const resolver = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm",
      );
      if (isCancelled()) return;

      const landmarker = await createLandmarker(resolver);
      if (isCancelled()) {
        landmarker.close();
        return;
      }
      handlesRef.current.landmarker = landmarker;

      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      if (isCancelled()) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }
      handlesRef.current.stream = stream;

      const video = videoRef.current;
      if (!video) return;
      video.srcObject = stream;
      await video.play();
      if (isCancelled()) return;

      setStatus("tracking");

      const loop = () => {
        if (isCancelled()) return;
        const v = videoRef.current;
        const canvas = canvasRef.current;
        const lm = handlesRef.current.landmarker;
        if (!v || !canvas || !lm) return;

        const w = v.videoWidth;
        const h = v.videoHeight;
        if (w && h && (canvas.width !== w || canvas.height !== h)) {
          canvas.width = w;
          canvas.height = h;
        }

        const ctx = canvas.getContext("2d");
        ctx.save();
        ctx.scale(-1, 1);
        ctx.drawImage(v, -w, 0, w, h);
        ctx.restore();

        const result = lm.detectForVideo(v, performance.now());
        const landmarks = result.faceLandmarks?.[0];

        if (landmarks) {
          ctx.fillStyle = "#64ff32";
          for (const i of RIGHT_IRIS) {
            const p = landmarks[i];
            ctx.beginPath();
            ctx.arc((1 - p.x) * w, p.y * h, 3, 0, Math.PI * 2);
            ctx.fill();
          }

          const iris = landmarks[IRIS_TRACKING];
          const targetX = (1 - iris.x) * w;
          const targetY = iris.y * h;
          if (smoothed.current.x === null) {
            smoothed.current = { x: targetX, y: targetY };
          } else {
            smoothed.current.x += (targetX - smoothed.current.x) * SMOOTHING;
            smoothed.current.y += (targetY - smoothed.current.y) * SMOOTHING;
          }

          ctx.strokeStyle = "#ff5577";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(smoothed.current.x, smoothed.current.y, 14, 0, Math.PI * 2);
          ctx.stroke();
          ctx.beginPath();
          ctx.moveTo(smoothed.current.x - 8, smoothed.current.y);
          ctx.lineTo(smoothed.current.x + 8, smoothed.current.y);
          ctx.moveTo(smoothed.current.x, smoothed.current.y - 8);
          ctx.lineTo(smoothed.current.x, smoothed.current.y + 8);
          ctx.stroke();

          const upper = landmarks[LEFT_EYE_UPPER];
          const lower = landmarks[LEFT_EYE_LOWER];
          const eyeGap = Math.abs(upper.y - lower.y);
          const now = performance.now();
          if (eyeGap < BLINK_THRESHOLD && now - lastClickAt.current > CLICK_COOLDOWN_MS) {
            lastClickAt.current = now;
            setClicks((c) => c + 1);
          }
          if (eyeGap < BLINK_THRESHOLD) {
            ctx.fillStyle = "rgba(255, 85, 119, 0.25)";
            ctx.fillRect(0, 0, w, h);
          }
        }

        const now = performance.now();
        const dt = (now - lastFrameAt) / 1000;
        lastFrameAt = now;
        if (dt > 0) smoothedFps = 0.9 * smoothedFps + 0.1 * (1 / dt);
        if (now - lastFpsPublishAt > FPS_UPDATE_INTERVAL_MS) {
          setFps(smoothedFps);
          lastFpsPublishAt = now;
        }

        handlesRef.current.rafId = requestAnimationFrame(loop);
      };
      loop();
    } catch (err) {
      if (!isCancelled()) {
        console.error(err);
        setStatus(`error: ${err.message}`);
        stop();
      }
    }
  }, [stop]);

  // Auto-start on mount, and always tear down on unmount.
  useEffect(() => {
    start();
    return stop;
  }, [start, stop]);

  // Allow Esc to stop, matching the Python CLI.
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "Escape" && running) stop();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [running, stop]);

  return (
    <div style={{ padding: 24, textAlign: "center" }}>
      <h1 style={{ margin: "0 0 8px" }}>EDC — Web Demo</h1>
      <p style={{ margin: "0 0 16px", color: "#9aa4b2" }}>
        Iris tracking + blink-to-click, in the browser. Status: <b>{status}</b>
      </p>

      <div style={{ position: "relative", display: "inline-block" }}>
        <video ref={videoRef} playsInline muted style={{ display: "none" }} />
        <canvas
          ref={canvasRef}
          style={{
            borderRadius: 12,
            maxWidth: "min(90vw, 960px)",
            background: "#000",
          }}
        />
      </div>

      <div
        style={{
          marginTop: 16,
          display: "flex",
          gap: 12,
          justifyContent: "center",
          alignItems: "center",
          fontFamily: "ui-monospace, monospace",
        }}
      >
        <button
          onClick={running ? stop : start}
          style={{
            background: running ? "#ff5577" : "#6c63ff",
            color: "white",
            border: "none",
            borderRadius: 8,
            padding: "8px 16px",
            fontSize: 14,
            fontWeight: 600,
            cursor: "pointer",
          }}
        >
          {running ? "Stop" : "Start"}
        </button>
        <span>FPS: {fps.toFixed(1)}</span>
        <span>Blinks: {clicks}</span>
      </div>
      <p style={{ marginTop: 12, color: "#6b7280", fontSize: 12 }}>
        Press <kbd>Esc</kbd> or click <b>Stop</b> to release the camera.
      </p>
    </div>
  );
}
