# Eye Detection Cursor (EDC)

Hands-free mouse control using your webcam. EDC tracks your iris with MediaPipe FaceMesh to move the cursor and detects blinks to click — useful for accessibility, hands-busy demos, and as a real-time computer-vision learning project.

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0097A7?logo=google&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

## Demo

> _Add a GIF here — record with QuickTime / Peek / LICEcap and save as `docs/demo.gif`._

```
![demo](docs/demo.gif)
```

## How it works

| Stage | What happens |
|---|---|
| Capture | OpenCV reads frames from your webcam at native FPS |
| Detect | MediaPipe FaceMesh extracts 478 facial landmarks (with iris refinement) |
| Track | Iris landmark `475` is mapped from normalized image coords → screen coords |
| Smooth | Exponential moving average dampens cursor jitter |
| Click | Vertical gap between left-eye lid landmarks `145`/`159` shrinking past a threshold triggers a click |
| Debounce | Non-blocking timestamp check enforces a minimum gap between clicks |

## Features

- Real-time iris tracking — runs comfortably at 25–30 FPS on a laptop CPU
- Exponential cursor smoothing (configurable)
- Non-blocking click debounce (no UI freeze)
- Configurable sensitivity, cooldown, camera index, and mirror mode via CLI
- Debug overlay with live FPS and landmark visualization

## Install

```bash
git clone https://github.com/AayushCharde/EDC.git
cd EDC
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

Requires Python 3.9+.

## Run

```bash
python main.py                          # default settings
python main.py --debug                  # show FPS + landmark overlay
python main.py --smoothing 0.2          # smoother (slower) cursor
python main.py --blink-threshold 0.004  # stricter blink detection
python main.py --camera 1               # use external webcam
```

Press `q` in the preview window to quit.

### CLI options

| Flag | Default | Purpose |
|---|---|---|
| `--smoothing` | `0.3` | Cursor smoothing factor (0–1). Lower = smoother, higher = more responsive. |
| `--blink-threshold` | `0.005` | Eye-closure threshold in normalized coords. Lower = stricter. |
| `--click-cooldown` | `1.0` | Minimum seconds between clicks. |
| `--camera` | `0` | Camera device index. |
| `--no-mirror` | off | Disable horizontal mirroring of the camera feed. |
| `--debug` | off | Show FPS + landmark overlay. |

## Limitations

- Tracking accuracy depends on lighting and distance from the camera (~50–80 cm works best).
- No calibration step — cursor mapping is direct from iris position to screen, so head movement also moves the cursor. A 4-corner calibration is a planned improvement.
- Blink detection uses a single threshold, so users with smaller eye apertures may need to lower `--blink-threshold`.
- Built on `pyautogui`, which means it works on macOS, Linux, and Windows but may need accessibility permissions on macOS.

## Roadmap

- [ ] 4-corner calibration on startup
- [ ] Dwell-click mode (hover N ms instead of blink)
- [ ] Scroll gesture (look up/down past a threshold)
- [ ] Headless / config-file mode for accessibility deployment

## License

MIT — see [LICENSE](LICENSE).
