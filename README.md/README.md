# Real-Time Vision Pipeline Dashboard

A real-time vision pipeline that detects faces, extracts emotion + age + gender,
streams results via WebSocket, and renders a live dashboard with detection cards,
emotion charts, and system metrics.

---

## Project Structure

```
RealTime_Vision_Dashboard/
├── backend/
│   ├── server.py                  # FastAPI routes (no business logic)
│   ├── detection.py               # Detection API interface
│   ├── models/
│   │   └── schemas.py             # Pydantic schemas: BBox, Face, FaceWithImage, DetectionMetrics
│   ├── config/
│   │   └── settings.py            # DeviceConfig (GPU/CPU selection) + AppSettings (env vars)
│   ├── services/
│   │   ├── camera.py              # CameraService — camera I/O only
│   │   ├── analysis.py            # AnalysisService — DeepFace inference only
│   │   ├── metrics.py             # MetricsService — FPS/latency/counts only
│   │   └── detection_manager.py   # Detection pipeline orchestrator
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Dashboard.jsx
│   │   │   ├── DetectionCard.jsx
│   │   │   ├── EmotionChart.jsx
│   │   │   └── dashboard.css
│   │   └── App.jsx
│   └── package.json
└── README.md
```

---

## Backend Setup

### Install Dependencies

```bash
cd backend
python -m venv venv
source venv/bin/activate   # macOS/Linux
# OR
venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

Install DeepFace in editable mode (recommended):

```bash
git clone https://github.com/serengil/deepface.git
cd deepface
pip install -e .
```

### Run the Backend Server

```bash
uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

API available at: http://127.0.0.1:8000

---

## Frontend Setup (React)

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at: http://localhost:5173

---

## Features

### Backend (Python + FastAPI)
- ✅ Webcam capture via CameraService
- ✅ Face detection using DeepFace (hardware-aware backend selection)
- ✅ Emotion, age, and gender prediction
- ✅ WebSocket streaming for real-time updates
- ✅ REST endpoints for snapshots and health checks
- ✅ Fully async — asyncio.Lock, asyncio.create_task, thread-pool executor for inference
- ✅ Microservice package structure (models / config / services)

### Frontend (React)
- ✅ Live detection card feed (30 unique detections)
- ✅ Real-time emotion distribution chart
- ✅ System monitor panel (FPS, latency, face count, health)
- ✅ Responsive, scrollable UI
- ✅ Live frame display with bounding boxes

---

## Prerequisites
- Python 3.10+
- Node.js 16+
- Webcam

---

## API Endpoints

### REST
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | System status |
| GET | `/snapshot` | Latest inference JSON |
| GET | `/frame` | Live frame with detections |
| POST | `/detection/start` | Start pipeline |
| POST | `/detection/stop` | Stop pipeline |
| POST | `/detection/set-camera` | Change camera source |

### WebSocket
| Endpoint | Description |
|----------|-------------|
| `WS /stream` | Real-time detection JSON |
| `WS /detection/live` | Raw live frame with bounding boxes |
| `WS /detection/feed` | Cropped face images + metadata |

---

## Architecture

### Pipeline

```
Camera → CameraService → AnalysisService (thread-pool) → DetectionManager
                                  ↓                              ↓
                            DeepFace inference            asyncio.Lock
                                  ↓                              ↓
                          Face / FaceWithImage          WebSockets → React Dashboard
```

### Package structure

The backend is structured as a microservice-style package, not a monolith.
Each layer has one responsibility:

| Layer | File | Responsibility |
|-------|------|----------------|
| Models | `models/schemas.py` | Pydantic contracts for all data shapes |
| Config | `config/settings.py` | Hardware detection, env vars, all tunables |
| Service | `services/camera.py` | Camera I/O only |
| Service | `services/analysis.py` | DeepFace inference only |
| Service | `services/metrics.py` | FPS / latency / counts only |
| Orchestrator | `services/detection_manager.py` | Wires services, Detection pipeline orchestrator |
| Detection API Layer | `detection.py` |  Detection API interface |
| Routes | `server.py` | HTTP / WebSocket handlers only |

---

## CPU vs GPU Awareness

DeepFace supports several detector backends with very different performance
profiles. The right choice depends entirely on available hardware.

| Backend | CPU latency | GPU latency | Accuracy |
|---------|-------------|-------------|----------|
| opencv | ~50-100ms | N/A | Low |
| MTCNN | ~200-400ms | ~150-200ms* | High |
| RetinaFace | ~400-800ms | ~20-40ms | Highest |

*MPS (Apple Silicon Metal) — partial acceleration via TF/Keras Metal plugin.

### Backend selection ladder (`config/settings.py`)

```python
def _select_backend(self) -> str:
    if self.cuda_available:
        return "retinaface"   # GPU parallelism makes the heavy model worthwhile
    if self.mps_available:
        return "mtcnn"        # partial Metal acceleration, better accuracy than opencv
    return "opencv"           # CPU-only: speed over accuracy
```

**CUDA → RetinaFace:** Most accurate detector. On CPU it runs ~400-800ms —
not viable for live video. With CUDA it drops to ~20-40ms.

**MPS → MTCNN:** RetinaFace has unstable MPS support in DeepFace (falls back
to CPU internally). MTCNN gets partial Metal acceleration via TF/Keras,
giving better accuracy than OpenCV at ~150-200ms. Correct choice for
Apple Silicon.

**CPU → OpenCV:** Fastest pure-CPU detector (~50-100ms). Accuracy tradeoff
is acceptable for a live webcam feed where a missed detection is recovered
the next frame.

The selected backend and hardware probe are logged at startup:
```
Accelerator probe: torch=True  CUDA=False  MPS=True
DeviceConfig: CUDA=False  MPS=True  → backend=mtcnn
```

### Observed latency: 200-300ms on Apple Silicon (MPS + MTCNN)

This is expected and by design. MTCNN runs three sequential neural networks
internally (P-Net → R-Net → O-Net). DeepFace then runs three separate
attribute models for age, gender, and emotion (~50ms each).

This does not affect live feed smoothness because:
1. Inference runs every 5 frames (`detection_interval=5`), not every frame
2. DeepFace runs in a thread-pool executor — the asyncio event loop is never
   blocked, camera feed continues at full framerate
3. ~6 inference calls/sec at 200ms each is well within real-time bounds

---

## Design Choices

**Frame Interval = 5**
Running detection on every frame is too slow for CPU/MPS workloads.
Processing 1 out of every 5 frames keeps CPU usage low while maintaining
a smooth real-time experience. Configurable via `DETECTION_INTERVAL` env var.

**Async over threading**
FastAPI is built on asyncio. The original implementation mixed
`threading.Thread` with shared mutable state, which is not idiomatic and
introduced a race condition in signature tracking. The refactored version
uses `asyncio.Lock` and `asyncio.create_task` — the correct pattern for
FastAPI. DeepFace (CPU-bound) is offloaded to a thread-pool executor via
`loop.run_in_executor()` so the event loop is never blocked.

**Pydantic schemas**
All data shapes are defined as Pydantic models in `models/schemas.py`.
This gives automatic field validation, `.model_dump()` serialisation,
and FastAPI OpenAPI doc generation — none of which plain dataclasses provide.

**Signature-based deduplication**
Each detection generates a spatial signature quantised to 20px buckets
(configurable via `DEDUP_BUCKET_PX`). Small positional jitter between
frames doesn't create duplicate feed entries. Signature check and add
happen inside a single `async with state_lock` block — no race condition.

**Cropped face size = 80×80**
Small enough for fast WebSocket transfer, large enough to clearly show
emotion and identity in the detection card feed.

**Two WebSocket channels**
`/detection/live` → optimised for full video frames with bounding boxes
`/detection/feed` → optimised for face crops and metadata
Separation prevents bandwidth overload and keeps FPS stable.

**Environment-configurable tunables**
All hardcoded configuration values are defined in  `config/settings.py` and can be overridden via environment variables—no code changes are needed between environments:
```bash
DETECTION_INTERVAL=3 ANALYSIS_TARGET_WIDTH=480 uvicorn server:app
```

---

## Project Requirement Checklist

### A. Vision Service (Python + FastAPI)
- ✅ Captures video input (webcam)
- ✅ Runs face detection + emotion/age/gender prediction
- ✅ Sends detections as JSON over WebSocket and REST
- ✅ `/stream` WebSocket endpoint for real-time detections
- ✅ `/snapshot` endpoint returns latest inference result
- ✅ `/health` endpoint for health checks
- ✅ Uses DeepFace model
- ✅ Output JSON format matches assignment requirements

### B. Mini Dashboard (React)
- ✅ Live card feed
- ✅ Each card = one detection event
- ✅ Shows face cropped image
- ✅ Shows labels (emotion, age, gender)
- ✅ Real-time emotion distribution chart
- ✅ Updates live via WebSocket
- ✅ System Monitor Panel (FPS, latency, face count, health)

### C. Runtime Logic Requirements
- ✅ Async processing (asyncio, thread-pool executor for CPU-bound inference)
- ✅ Race condition eliminated (atomic check-and-add under asyncio.Lock)
- ✅ GPU/CPU awareness (CUDA → RetinaFace, MPS → MTCNN, CPU → OpenCV)
- ✅ Microservice package structure (models / config / services)
- ✅ Documentation (this README)
- ✅ Setup steps included

---

## License

© 2025. All rights reserved by Kimia Nahravanian.

This project and its contents are the intellectual property of the author
and may not be copied, modified, distributed, or used without explicit permission.