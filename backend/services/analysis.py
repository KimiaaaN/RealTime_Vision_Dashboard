"""
services/analysis.py
--------------------
AnalysisService – owns all face-detection and attribute-analysis logic.

Single responsibility: take a raw BGR frame, return typed Face / FaceWithImage
objects. 

DeepFace is CPU-bound.  The service exposes an async entry-point
(analyze_frame) that offloads the blocking call to a thread-pool executor so
the FastAPI event loop is never stalled.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from typing import List, Tuple

import cv2
from deepface import DeepFace

from config.settings import DeviceConfig, settings
from models.schemas import BBox, Face, FaceWithImage

logger = logging.getLogger(__name__)


class AnalysisService:
    """
    Runs DeepFace inference and converts raw results into typed schema objects.

    Construction requires a DeviceConfig so the correct detector backend is
    used automatically (RetinaFace on CUDA, MTCNN on MPS, OpenCV on CPU).
    """

    def __init__(self, device_config: DeviceConfig) -> None:
        self._backend = device_config.detector_backend
        self._target_width = settings.analysis_target_width
        self._actions = settings.deepface_actions
        self._crop_size = settings.face_crop_size
        logger.info(
            "AnalysisService ready: backend=%s  target_width=%d",
            self._backend,
            self._target_width,
        )

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def analyze_frame(
        self, frame
    ) -> Tuple[List[Face], List[FaceWithImage]]:
        """
        Analyse *frame* asynchronously.

        The heavy DeepFace call is executed in a thread-pool executor so the
        event loop remains free to serve other requests during inference.

        Returns:
            faces       – list of Face (used for bbox overlay & stream JSON)
            face_images – list of FaceWithImage (used for the detection feed)
        """
        loop = asyncio.get_running_loop()
        faces, face_images = await loop.run_in_executor(
            None, self._run_deepface, frame
        )
        return faces, face_images

    # ------------------------------------------------------------------
    # Private sync implementation (runs in thread pool)
    # ------------------------------------------------------------------

    def _run_deepface(
        self, frame
    ) -> Tuple[List[Face], List[FaceWithImage]]:
        """Blocking DeepFace call – must NOT be awaited directly."""
        faces: List[Face] = []
        face_images: List[FaceWithImage] = []

        try:
            # --- downscale for speed ---
            downscaled, scale = self._downscale(frame)
            rgb = cv2.cvtColor(downscaled, cv2.COLOR_BGR2RGB)

            raw_results = DeepFace.analyze(
                rgb,
                actions=self._actions,
                enforce_detection=False,
                detector_backend=self._backend,
            )

            if not isinstance(raw_results, list):
                raw_results = [raw_results]

            h, w = frame.shape[:2]
            now_ms = int(time.time() * 1000)

            for idx, result in enumerate(raw_results):
                if "region" not in result:
                    continue

                bbox = self._scale_bbox(result["region"], scale, w, h)
                age = int(result["age"])
                gender: str = result["dominant_gender"]
                emotion: str = result["dominant_emotion"]

                face = Face(bbox=bbox, age=age, gender=gender, emotion=emotion)
                faces.append(face)

                image_b64 = self._crop_and_encode(frame, bbox)
                face_images.append(
                    FaceWithImage(
                        bbox=bbox,
                        age=age,
                        gender=gender,
                        emotion=emotion,
                        image=image_b64,
                    )
                )

        except Exception:
            logger.exception("DeepFace analysis error")

        return faces, face_images

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _downscale(self, frame) -> Tuple[object, float]:
        """Resize frame so its width ≤ target_width; return (frame, scale)."""
        h, w = frame.shape[:2]
        if w > self._target_width:
            scale = self._target_width / float(w)
            resized = cv2.resize(frame, (self._target_width, int(h * scale)))
            return resized, scale
        return frame, 1.0

    @staticmethod
    def _scale_bbox(region: dict, scale: float, frame_w: int, frame_h: int) -> BBox:
        """Map a region from the downscaled frame back to original dimensions."""
        rx, ry, rw, rh = region["x"], region["y"], region["w"], region["h"]
        x1 = max(0, min(int(rx / scale), frame_w - 1))
        y1 = max(0, min(int(ry / scale), frame_h - 1))
        x2 = max(0, min(int((rx + rw) / scale), frame_w))
        y2 = max(0, min(int((ry + rh) / scale), frame_h))
        return BBox(x1=x1, y1=y1, x2=x2, y2=y2)

    def _crop_and_encode(self, frame, bbox: BBox) -> str:
        """Crop face region, resize to crop_size, return base64 JPG string."""
        crop = frame[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2]
        if crop.size == 0:
            return ""
        resized = cv2.resize(crop, self._crop_size)
        _, buffer = cv2.imencode(".jpg", resized)
        return base64.b64encode(buffer).decode("utf-8")