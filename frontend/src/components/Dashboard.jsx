import { useEffect, useState, useRef, useCallback } from "react";
import DetectionCard from "./DetectionCard";
import EmotionChart from "./EmotionChart";
import "./Dashboard.css";

const API_BASE = "http://localhost:8000";
const HEALTH_URL = `${API_BASE}/health`;
const FRAME_URL = `${API_BASE}/frame`;
const WS_LIVE_URL = `ws://localhost:8000/detection/live`;
const WS_FEED_URL = `ws://localhost:8000/detection/feed`;

export default function Dashboard() {
  const [health, setHealth] = useState(false);
  const [frameData, setFrameData] = useState(null);
  const [frameFaces, setFrameFaces] = useState([]);
  const [events, setEvents] = useState([]);
  const [metrics, setMetrics] = useState({});
  const [isStreamActive, setIsStreamActive] = useState(false);
  const [error, setError] = useState(null);

  const liveWsRef = useRef(null);
  const feedWsRef = useRef(null);
  const canvasRef = useRef(null);

 
  // Start Live Stream WebSocket
 
  const startLiveStream = useCallback(async () => {
    try {
      const resp = await fetch(`${API_BASE}/detection/start`, { method: "POST" });
      if (!resp.ok) throw new Error(await resp.text());
    } catch (err) {
      setError("Failed to start detection on server");
      console.error(err);
      return;
    }

    if (liveWsRef.current) liveWsRef.current.close();

    const ws = new WebSocket(WS_LIVE_URL);
    ws.onopen = () => setIsStreamActive(true);
    ws.onmessage = (msg) => {
      try {
        const data = JSON.parse(msg.data);
        if (data.frame) setFrameData(`data:image/jpeg;base64,${data.frame}`);
        if (data.metrics) setMetrics(data.metrics);
        if (Array.isArray(data.faces)) setFrameFaces(data.faces);
      } catch (err) {
        console.error("Live WebSocket parse error:", err);
      }
    };
    ws.onerror = () => setError("Live WebSocket error");
    ws.onclose = () => setIsStreamActive(false);
    liveWsRef.current = ws;
  }, []);

  
  // Start Detection Feed WebSocket
  
  const startFeedStream = useCallback(() => {
    if (feedWsRef.current) feedWsRef.current.close();
    const ws = new WebSocket(WS_FEED_URL);
    ws.onmessage = (msg) => {
      try {
        const data = JSON.parse(msg.data);
        if (Array.isArray(data.face_images)) {
          // Backend already provides unique detections limited to 30
          // Just update the events state with the new unique detections
          setEvents(prevEvents => {
            // Create a map of existing UIDs
            const existingUids = new Set(prevEvents.map(e => e._uid).filter(Boolean));
            
            // Filter out duplicates and merge with new detections
            const newUnique = data.face_images.filter(f => f._uid && !existingUids.has(f._uid));
            
            // Combine and limit to 30, sort by _uid (newest first)
            const combined = [...newUnique, ...prevEvents]
              .filter((e, idx, arr) => arr.findIndex(a => a._uid === e._uid) === idx) // Remove duplicates
              .sort((a, b) => (b._uid || '').localeCompare(a._uid || ''))
              .slice(0, 30);
            
            return combined;
          });
        }
      } catch (err) {
        console.error("Feed WebSocket parse error:", err);
      }
    };
    feedWsRef.current = ws;
  }, []);

  const startDetection = async () => {
    setError(null);
    await startLiveStream();
    startFeedStream();
  };

  const stopStreams = async () => {
    if (liveWsRef.current) liveWsRef.current.close();
    if (feedWsRef.current) feedWsRef.current.close();
    try {
      await fetch(`${API_BASE}/detection/stop`, { method: "POST" });
    } catch (e) { console.warn(e); }
    setIsStreamActive(false);
  };

 
  // Health polling
  
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const r = await fetch(HEALTH_URL);
        if (!r.ok) {
          setHealth(false);
          return;
        }
        const j = await r.json();
        setHealth(j.status === "ok");

        const f = await fetch(FRAME_URL);
        if (f.ok) {
          const jf = await f.json();
          if (jf.frame) setFrameData(`data:image/jpeg;base64,${jf.frame}`);
          if (Array.isArray(jf.faces)) setFrameFaces(jf.faces);
          if (jf.metrics) setMetrics(jf.metrics);
        } else if (f.status === 403) {
          console.error("403 Forbidden - CORS or authentication issue");
          setError("403 Forbidden - Check CORS configuration");
        }
      } catch (err) {
        console.error("Health check error:", err);
        setHealth(false);
      }
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  
  // Draw bounding boxes on live canvas
  
  useEffect(() => {
    if (!frameData) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const img = new Image();
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
      frameFaces.forEach((face) => {
        const { x = 0, y = 0, w = 0, h = 0 } = face.region || {};
        ctx.strokeStyle = "lime";
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);
        ctx.fillStyle = "lime";
        ctx.font = "16px Arial";
        const label = `${face.gender || "?"}, ${face.age ?? "?"}, ${face.emotion || "?"}`;
        ctx.fillText(label, x + 2, y > 20 ? y - 6 : y + 18);
      });
    };
    img.src = frameData;
  }, [frameData, frameFaces]);

  return (
    <div className="dashboard-container">
      <h2>Real-Time Vision Dashboard</h2>

      <div className="status-container">
        <div>
          <strong>System Status:</strong>
          <span style={{ color: health ? "green" : "red" }}>{health ? "Healthy" : "Offline"}</span>
        </div>
        <div>
          <strong>Stream Status:</strong>
          <span style={{ color: isStreamActive ? "green" : "red" }}>{isStreamActive ? "Active" : "Inactive"}</span>
        </div>
      </div>

      {error && <div style={{ color: "red" }}>{error}</div>}

      <div className="control-buttons">
        <button onClick={startDetection} disabled={isStreamActive}>Start Detection</button>
        <button onClick={stopStreams} disabled={!isStreamActive}>Stop Detection</button>
      </div>

      <div className="main-content">
        <div className="live-frame-container">
          <h3>Live Frame</h3>
          <canvas ref={canvasRef} />
        </div>

        <div className="emotion-chart-container">
          <EmotionChart events={events} />
        </div>
      </div>

      <div className="metrics-container">
        <div className="metric-item">
          <strong>FPS:</strong> {metrics.fps?.toFixed?.(1) ?? 0}
        </div>
        <div className="metric-item">
          <strong>Latency:</strong> {metrics.last_latency_ms?.toFixed?.(1) ?? 0} ms
        </div>
        <div className="metric-item">
          <strong>Faces (Latest):</strong> {metrics.face_count ?? frameFaces.length}
        </div>
        <div className="metric-item">
          <strong>Unique Detections:</strong> {events.length} / 30
        </div>
      </div>

      <div className="detection-feed-section">
        <h3>Detection Feed ({events.length} unique detections)</h3>
        <div className="detection-feed">
          {events.length > 0 ? (
            events.map((evt) => (
              <DetectionCard key={evt._uid || evt.id || Math.random()} event={evt} frame={evt.image} />
            ))
          ) : (
            <p className="no-detections">No detections yet. Start detection to see results.</p>
          )}
        </div>
      </div>
    </div>
  );
}






