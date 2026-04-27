"""
services/detection_manager.py
------------------------------
DetectionManager – the pipeline orchestrator.

This class does NOT implement any domain logic itself.  Its only job is to
wire CameraService → AnalysisService → MetricsService together and maintain
the shared state that server.py reads via the JSON-output methods.

Responsibilities (this file only):
  - Start / stop the async detection loop
  - Deduplicate detections (signature tracking)
  - Atomic state updates under asyncio.Lock
  - Serialise internal state to the dict shapes expected by server.py

All heavy lifting is delegated:
  Camera I/O   → CameraService
  Inference    → AnalysisService   (runs DeepFace in a thread-pool executor)
  Perf metrics → MetricsService
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from typing import List, Optional, Set

import cv2

from config.settings import DeviceConfig, settings
from models.schemas import Face, FaceWithImage
from services.camera import CameraService
from services.analysis import AnalysisService
from services.metrics import MetricsService

logger = logging.getLogger(__name__)


class DetectionManager:
    """
    Async-native orchestrator for the face-detection pipeline.

    Instantiate once at application startup (e.g. in server.py lifespan) and
    store on app.state so every route handler can reach it.

    Public interface used by server.py:
      set_camera()          – configure camera source
      start_detection()     – open camera, arm loop
      stop_detection()      – signal loop to stop
      stop_camera()         – stop + release + clear state
      run_detection_loop()  – the main async loop (await inside a Task)
      get_*_json()          – JSON-ready dicts for each endpoint
    """

    def __init__(self, detection_interval: Optional[int] = None) -> None:
        # --- services ---
        self._device_config = DeviceConfig()
        self.camera_service = CameraService()
        self.analysis_service = AnalysisService(self._device_config)
        self.metrics_service = MetricsService()

        # --- loop control ---
        self.is_active: bool = False
        self.stop_requested: bool = False
        self._detection_interval: int = (
            detection_interval
            if detection_interval is not None
            else settings.detection_interval
        )
        self._frame_count: int = 0

        # --- shared state (guarded by _lock) ---
        self._lock = asyncio.Lock()
        self.latest_frame = None            # numpy ndarray or None
        self.detection_results: List[Face] = []
        self.face_images: List[FaceWithImage] = []
        self.unique_detections: List[dict] = []   # max settings.max_unique_detections
        self._signatures: Set[str] = set()
        self.last_timestamp: Optional[str] = None

    # ------------------------------------------------------------------
    # Camera / pipeline control
    # ------------------------------------------------------------------

    def set_camera(self, camera_type: str, camera_url: Optional[str] = None) -> None:
        """Delegate camera configuration to CameraService."""
        self.camera_service.set_source(camera_type, camera_url)

    async def start_detection(self) -> None:
        
        if self.is_active:
            logger.info("Detection already active – ignoring start request")
            return
        try:
            self.camera_service.open()
            self.is_active = True
            self.stop_requested = False
            self._frame_count = 0
            logger.info("Detection pipeline started")
        except Exception:
            self.is_active = False
            logger.exception("Failed to start detection pipeline")
            raise

    async def stop_detection(self) -> None:
       
        self.stop_requested = True
        self.is_active = False
        logger.info("Detection pipeline stop requested")

    async def stop_camera(self) -> None:
        
        await self.stop_detection()
        self.camera_service.close()
        async with self._lock:
            self.unique_detections = []
            self._signatures = set()
            self.latest_frame = None
            self.detection_results = []
            self.face_images = []
        logger.info("Camera stopped and state cleared")

    # ------------------------------------------------------------------
    # Main async detection loop
    # ------------------------------------------------------------------

    async def run_detection_loop(self) -> None:
        """
        Async loop: read frame → analyse → update state.

        Designed to be launched as an asyncio.Task:
            task = asyncio.create_task(manager.run_detection_loop())

        The loop yields to the event loop on every iteration (asyncio.sleep)
        so FastAPI can continue serving HTTP requests while detection runs.
        """
        if not self.is_active:
            await self.start_detection()

        while not self.stop_requested:
            try:
                await self._process_one_frame()
            except Exception:
                logger.exception("Unhandled error in detection loop")
                await asyncio.sleep(0.1)

    async def _process_one_frame(self) -> None:
        # 1 – read
        ret, frame = self.camera_service.read_frame()
        if not ret or frame is None:
            await asyncio.sleep(0.05)
            return

        # 2 – fps tick (every frame)
        self.metrics_service.tick_fps()
        self._frame_count += 1

        # 3 – skip if not on detection cadence
        if self._frame_count % self._detection_interval != 0:
            await asyncio.sleep(0.01)
            return

        # 4 – inference (offloaded to thread pool inside AnalysisService)
        t0 = time.time()
        faces, face_images = await self.analysis_service.analyze_frame(frame)
        latency_ms = (time.time() - t0) * 1000
        self.metrics_service.record_inference(latency_ms)

        # 5 – draw bounding boxes
        annotated = self._draw_boxes(frame, faces)

        # 6 – compute unique-detection candidates OUTSIDE the lock
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        new_detections, new_sigs = self._compute_new_detections(
            face_images, timestamp
        )

        # 7 – atomic state write
        async with self._lock:
            self.latest_frame = annotated
            self.detection_results = faces
            self.face_images = face_images
            self.last_timestamp = timestamp

            if new_detections:
                self.unique_detections = (
                    new_detections + self.unique_detections
                )[: settings.max_unique_detections]
                self._signatures.update(new_sigs)

                if len(self._signatures) > settings.signature_prune_threshold:
                    self._prune_signatures()

            self.metrics_service.update_counts(
                face_count=len(faces),
                unique_count=len(self.unique_detections),
            )

        await asyncio.sleep(0.01)

    # ------------------------------------------------------------------
    # Deduplication helpers
    # ------------------------------------------------------------------

    def _compute_new_detections(
        self, face_images: List[FaceWithImage], timestamp: str
    ):
        """
        Identify face_images that haven't been seen before (by signature).
        Returns (list_of_new_detection_dicts, set_of_new_signatures).
        Intentionally runs outside the lock so lock hold-time is minimal.
        """
        new_detections = []
        new_sigs: Set[str] = set()
        now_ms = int(time.time() * 1000)

        for idx, fi in enumerate(face_images):
            sig = self._make_signature(fi)
            if sig not in self._signatures and sig not in new_sigs:
                new_detections.append(
                    {
                        "_uid": f"{now_ms}_{idx}",
                        "timestamp": timestamp,
                        "bbox": fi.bbox.to_list(),
                        "image": fi.image,
                        "age": fi.age,
                        "gender": fi.gender,
                        "emotion": fi.emotion,
                    }
                )
                new_sigs.add(sig)

        return new_detections, new_sigs

    @staticmethod
    def _make_signature(fi: FaceWithImage) -> str:
        
        b = settings.dedup_bucket_px
        sx = (fi.bbox.x1 // b) * b
        sy = (fi.bbox.y1 // b) * b
        return f"{sx}_{sy}_{fi.age}_{fi.gender}_{fi.emotion}"

    def _prune_signatures(self) -> None:
        """
        Drop signatures that are no longer represented in unique_detections.
        Call only while holding _lock.
        """
        b = settings.dedup_bucket_px
        active: Set[str] = set()
        for det in self.unique_detections:
            x, y = det["bbox"][0], det["bbox"][1]
            active.add(f"{(x // b) * b}_{(y // b) * b}_{det['age']}_{det['gender']}_{det['emotion']}")
        self._signatures = active

    # ------------------------------------------------------------------
    # Frame helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_boxes(frame, faces: List[Face]):
        
        out = frame.copy()
        for face in faces:
            x1, y1, x2, y2 = face.bbox.x1, face.bbox.y1, face.bbox.x2, face.bbox.y2
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{face.gender} | {face.age} | {face.emotion}"
            cv2.putText(
                out, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )
        return out

    @staticmethod
    def _encode_frame(frame) -> str:
       
        _, buf = cv2.imencode(".jpg", frame)
        return base64.b64encode(buf).decode("utf-8")

    # ------------------------------------------------------------------
    # JSON output methods (called by server.py route handlers)
    # ------------------------------------------------------------------

    async def get_health_status(self) -> dict:
        async with self._lock:
            return {
                "status": "ok" if self.is_active else "error",
                "is_active": self.is_active,
                "fps": self.metrics_service.fps,
                "has_frame": self.latest_frame is not None,
            }

    async def get_snapshot_json(self) -> Optional[dict]:
        async with self._lock:
            if self.latest_frame is None:
                return None
            return {
                "timestamp": self.last_timestamp or _now_iso(),
                "faces": [_face_to_dict(f) for f in self.detection_results],
            }

    async def get_stream_json(self) -> dict:
        async with self._lock:
            return {
                "timestamp": self.last_timestamp or _now_iso(),
                "faces": [_face_to_dict(f) for f in self.detection_results],
            }

    async def get_live_frame_json(self) -> Optional[dict]:
        async with self._lock:
            if self.latest_frame is None:
                return None
            metrics = self.metrics_service.get_metrics()
            return {
                "frame": self._encode_frame(self.latest_frame),
                "faces": [_face_to_region_dict(f) for f in self.detection_results],
                "timestamp": self.last_timestamp,
                "metrics": {
                    "fps": metrics.fps,
                    "last_latency_ms": metrics.inference_latency_ms,
                    "face_count": metrics.face_count,
                },
            }

    async def get_detection_feed_json(self) -> dict:
        async with self._lock:
            metrics = self.metrics_service.get_metrics()
            return {
                "face_images": list(self.unique_detections),
                "timestamp": self.last_timestamp,
                "metrics": {
                    "fps": metrics.fps,
                    "last_latency_ms": metrics.inference_latency_ms,
                    "face_count": metrics.face_count,
                    "unique_detections_count": metrics.unique_detections_count,
                },
            }

    # kept for backwards compatibility with any server.py calls
    async def get_frame(self) -> Optional[str]:
        async with self._lock:
            if self.latest_frame is None:
                return None
            return self._encode_frame(self.latest_frame)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


def _face_to_dict(face: Face) -> dict:
    return {
        "bbox": face.bbox.to_list(),
        "emotion": face.emotion,
        "age": face.age,
        "gender": face.gender,
    }


def _face_to_region_dict(face: Face) -> dict:
    return {
        "region": {
            "x": face.bbox.x1,
            "y": face.bbox.y1,
            "w": face.bbox.width,
            "h": face.bbox.height,
        },
        "age": face.age,
        "gender": face.gender,
        "emotion": face.emotion,
    }