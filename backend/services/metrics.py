"""
services/metrics.py
-------------------

Single responsibility: track FPS, inference latency, and detection counts.

The service is intentionally NOT async – all operations are simple arithmetic
that completes in microseconds.  The DetectionManager updates it synchronously
inside the async lock.

"""

from __future__ import annotations

import time
import logging

from models.schemas import DetectionMetrics

logger = logging.getLogger(__name__)


class MetricsService:
    """
    Tracks real-time performance metrics for the detection pipeline.

    Attributes are updated by DetectionManager on each processed frame and
    exposed as a typed DetectionMetrics snapshot via get_metrics().
    """

    def __init__(self) -> None:
        self.fps: float = 0.0
        self.inference_latency_ms: float = 0.0
        self.face_count: int = 0
        self.unique_detections_count: int = 0

        self._last_frame_time: float = time.time()

    # ------------------------------------------------------------------
    # Update methods (called by DetectionManager)
    # ------------------------------------------------------------------

    def tick_fps(self) -> None:
        """
        Record that a new frame has arrived and recalculate FPS.
        Call this once per frame, before the detection-interval guard.
        """
        now = time.time()
        delta = now - self._last_frame_time
        self.fps = 1.0 / delta if delta > 0 else 0.0
        self._last_frame_time = now

    def record_inference(self, latency_ms: float) -> None:
       
        self.inference_latency_ms = latency_ms

    def update_counts(self, face_count: int, unique_count: int) -> None:
        """Sync face / unique-detection counters after each detection run."""
        self.face_count = face_count
        self.unique_detections_count = unique_count

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def get_metrics(self) -> DetectionMetrics:
      
        return DetectionMetrics(
            fps=round(self.fps, 2),
            inference_latency_ms=round(self.inference_latency_ms, 2),
            face_count=self.face_count,
            unique_detections_count=self.unique_detections_count,
        )