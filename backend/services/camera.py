"""
services/camera.py
------------------
CameraService – owns every interaction with the physical (or IP) camera.

"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2

logger = logging.getLogger(__name__)


class CameraService:
    """
    Thin wrapper around cv2.VideoCapture.

    Supports:
      - Local webcam  (camera_type="local_camera")
      - IP / RTSP camera  (camera_type="ip_camera", camera_url="rtsp://…")

    Thread / async safety:
      This class is intentionally NOT async.  It is called from the async
      detection loop via asyncio.get_running_loop().run_in_executor() when
      frame reads need to be off the event loop, but the DetectionManager
      decides that not the  CameraService .
    """

    def __init__(self) -> None:
        self._capture: Optional[cv2.VideoCapture] = None
        self._source: int | str = 0   # default: first local webcam
        self.is_open: bool = False

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_source(self, camera_type: str, camera_url: Optional[str] = None) -> None:
       
        if camera_type == "local_camera":
            self._source = 0
        elif camera_type == "ip_camera":
            if not camera_url:
                raise ValueError("camera_url is required for ip_camera")
            self._source = camera_url
        else:
            raise ValueError(f"Unknown camera_type: {camera_type!r}")

        logger.info("Camera source configured: type=%s  source=%s", camera_type, self._source)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
       
        if self.is_open:
            logger.debug("Camera already open, skipping open()")
            return

        self._capture = cv2.VideoCapture(self._source)

        if not self._capture.isOpened():
            self._capture = None
            raise RuntimeError(f"Failed to open camera source: {self._source!r}")

        self.is_open = True
        logger.info("Camera opened: source=%s", self._source)

    def close(self) -> None:
        
        if self._capture is not None:
            self._capture.release()
            self._capture = None

        self.is_open = False
        cv2.destroyAllWindows()
        logger.info("Camera closed.")

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def read_frame(self) -> Tuple[bool, Optional[object]]:
      
        if not self.is_open or self._capture is None:
            return False, None

        ret, frame = self._capture.read()
        return ret, frame if ret else None