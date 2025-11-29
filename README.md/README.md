# Real-Time Vision Pipeline Dashboard

A mini real-time vision pipeline that detects faces, extracts emotion + age + gender, streams results to a lightweight backend, and renders a dashboard with live updates.


## Project Structure

```
RealTime_Vision_Dashboard/
├── backend/              # Python FastAPI microservice
│   ├── detection.py     # Face detection & analysis logic
│   ├── server.py        # FastAPI server with endpoints
│   └── requirements.txt # Python dependencies
├── frontend/            # React dashboard
│   ├── src/
│   │   ├── components/
│   │   │   ├── Dashboard.jsx    # Main dashboard component
│   │   │   ├── DetectionCard.jsx # Individual detection card
│   │   │   ├── EmotionChart.jsx # Emotion distribution chart
│   │   │   └── dashboard.css    # Styling
│   │   └── App.jsx
│   └── package.json
└── README.md

```
2️⃣ Backend Setup
✅ Install Dependencies

```bash
cd backend
python -m venv venv
venv\Scripts\activate     # Windows
# OR
source venv/bin/activate  # macOS/Linux

pip install -r requirements.txt

```

Install DeepFace in editable mode (recommended):

```bash

git clone https://github.com/serengil/deepface.git
cd deepface
pip install -e .
```

✅ Run the Backend Server

```bash
uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

The API will be available at:
👉 http://127.0.0.1:8000

3️⃣ Frontend Setup (React)

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install Node.js dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```
Frontend runs at:
👉 http://localhost:5173

## Features Overview

### Backend (Python + FastAPI)
- ✅ Captures video input (webcam)
- ✅ Face detection using DeepFace
- ✅ Emotion, age, and gender prediction
- ✅ WebSocket streaming for real-time updates
- ✅ REST endpoints for snapshots and health checks
- ✅ Async processing with threading

### Frontend (React)
- ✅ Live detection card feed (30 unique detections)
- ✅ Real-time emotion distribution chart
- ✅ System monitor panel (FPS, latency, face count, health)
- ✅ Responsive, scrollable UI
- ✅ Live frame display with bounding boxes

## Setup Instructions

### Prerequisites
- Python 3.10+
- Node.js 16+
- Webcam (for live detection)

📡 API Endpoints
REST

GET /health — system status

GET /snapshot — latest inference JSON

GET /frame — live frame with detections

POST /detection/start — start pipeline

POST /detection/stop — stop pipeline

POST /detection/set-camera — change camera

WebSocket

WS /stream — real-time detection JSON

WS /detection/live — raw live frame

WS /detection/feed — cropped face images

🧠 Simple Architecture

Camera → Detection Thread → DeepFace → JSON Events
         ↓                               ↑
         Frame Buffer  → WebSockets → React Dashboard

## Usage

1. Start the backend server (see Backend Setup)
2. Start the frontend development server (see Frontend Setup)
3. Open the frontend URL in your browser
4. Click "Start Detection" to begin face detection
5. The dashboard will show:
   - Live video feed with bounding boxes
   - Detection cards for each unique face detected
   - Emotion distribution chart
   - System metrics (FPS, latency, face count)

## Technical Details

### Models Used
- **DeepFace** - For face detection, emotion, age, and gender analysis
- **OpenCV** - For video capture and image processing

### Architecture
- **Backend**: FastAPI with async/await for concurrent processing
- **Frontend**: React with WebSocket connections for real-time updates
- **Threading**: Separate threads for frame capture and detection processing
- **Unique Detection Tracking**: Signature-based deduplication to show only unique detections

### Performance
- Detection runs every 5 frames (configurable)
- WebSocket updates at ~10-20 FPS
- Unique detections limited to 30 most recent
- Responsive UI with horizontal scrolling for detection feed

## Assignment Requirements Checklist

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
- ✅ System Monitor Panel:
  - ✅ FPS
  - ✅ Latency per frame
  - ✅ Count of faces detected
  - ✅ API health indicator (using /health)

### C. Runtime Logic Requirements
- ✅ Basic async processing (FastAPI async/await)
- ✅ Clean folder structure (backend/ and frontend/ separated)
- ✅ Documentation (this README)
- ✅ Setup steps included

📝 Design Choices

Frame Interval = 5
Running detection on every frame is too slow.
Processing 1 out of every 5 frames provides a smooth real-time experience while keeping CPU usage low.

Cropped Face Size = 80×80
Small enough for fast transfer over WebSocket, large enough to clearly identify emotion.

Two WebSocket Channels

/detection/live → optimized for video frames

/detection/feed → optimized for face crops & metadata
This separation prevents bandwidth overload and keeps FPS stable.

Threaded Detection
Detection and frame grabbing run on separate threads, ensuring:

UI stays smooth

Frame rate stays high

DeepFace doesn't block the webcam feed

Signature-Based Deduplication
Each detection generates a unique signature (crop + embeddings).
This ensures only unique faces appear in the detection feed.

Metrics Tracking (FPS + Latency)
Essential for debugging and performance optimization; displayed directly on the dashboard.

Fully Responsive Frontend
Built to scale from desktop → mobile with a flexible CSS grid layout and adaptive components.

## License

This project is created for assignment purposes.

