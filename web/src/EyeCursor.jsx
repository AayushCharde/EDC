import { useEffect, useRef, useState } from "react";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

// Same landmark indices as the Python version.
const RIGHT_IRIS = [474, 475, 476, 477];
const IRIS_TRACKING = 475;
const LEFT_EYE_UPPER = 159;
const LEFT_EYE_LOWER = 145;

const SMOOTHING = 0.3;
const BLINK_THRESHOLD = 0.005;
const CLICK_COOLDOWN_MS = 1000;

export default function EyeCursor() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const smoothed = useRef({ x: null, y: null });
  const lastClickAt = useRef(0);
  const [status, setStatus] = useState("loading model…");
  const [clicks, setClicks] = useState(0);
  const [fps, setFps] = useState(0);

  useEffect(() => {
    let landmarker;
    let rafId;
    let stream;
    let lastFrameAt = performance.now();
    let smoothedFps = 0;

    async function start() {
      try {
        const resolver = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm",
        );
        landmarker = await FaceLandmarker.createFromOptions(resolver, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            delegate: "GPU",
          },
          outputFaceBlendshapes: false,
          runningMode: "VIDEO",
          numFaces: 1,
        });

        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        const video = videoRef.current;
        video.srcObject = stream;
        await video.play();
        setStatus("tracking");
        loop();
      } catch (err) {
        console.error(err);
        setStatus(`error: ${err.message}`);
      }
    }

    function loop() {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas || !landmarker) return;

      const w = video.videoWidth;
      const h = video.videoHeight;
      if (w && h && (canvas.width !== w || canvas.height !== h)) {
        canvas.width = w;
        canvas.height = h;
      }

      const ctx = canvas.getContext("2d");
      ctx.save();
      ctx.scale(-1, 1);
      ctx.drawImage(video, -w, 0, w, h);
      ctx.restore();

      const result = landmarker.detectForVideo(video, performance.now());
      const landmarks = result.faceLandmarks?.[0];

      if (landmarks) {
        ctx.fillStyle = "#64ff32";
        for (const i of RIGHT_IRIS) {
          const lm = landmarks[i];
          const x = (1 - lm.x) * w;
          const y = lm.y * h;
          ctx.beginPath();
          ctx.arc(x, y, 3, 0, Math.PI * 2);
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
      if (dt > 0) {
        smoothedFps = 0.9 * smoothedFps + 0.1 * (1 / dt);
        setFps(smoothedFps);
      }

      rafId = requestAnimationFrame(loop);
    }

    start();
    return () => {
      cancelAnimationFrame(rafId);
      landmarker?.close();
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, []);

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

      <div style={{ marginTop: 16, fontFamily: "ui-monospace, monospace" }}>
        FPS: {fps.toFixed(1)} &nbsp;·&nbsp; Blinks: {clicks}
      </div>
    </div>
  );
}
