"""
config/settings.py
------------------

  - Environment variables are read once at startup via AppSettings.
  - DeviceConfig auto-detects GPU/CPU and chooses the right DeepFace backend.
  - Nothing in this file imports from services/ or models/ (no circular deps).

"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPU / accelerator detection
# ---------------------------------------------------------------------------

try:
    import torch

    _TORCH_AVAILABLE = True
    _CUDA_AVAILABLE = torch.cuda.is_available()
    _MPS_AVAILABLE = (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
except ImportError:
    _TORCH_AVAILABLE = False
    _CUDA_AVAILABLE = False
    _MPS_AVAILABLE = False

logger.info(
    "Accelerator probe: torch=%s  CUDA=%s  MPS=%s",
    _TORCH_AVAILABLE,
    _CUDA_AVAILABLE,
    _MPS_AVAILABLE,
)


# ---------------------------------------------------------------------------
# DeviceConfig – chooses the best DeepFace detector backend
# ---------------------------------------------------------------------------

class DeviceConfig:
    """
    Stateless configuration object that records which accelerator is available
    and selects the optimal DeepFace detector backend accordingly.

    Priority ladder:
      CUDA  → RetinaFace  (best accuracy, NVIDIA GPU offload)
      MPS   → MTCNN       (better accuracy than OpenCV, stable on Apple Silicon)
      CPU   → OpenCV      (fastest pure-CPU option)
    """

    def __init__(self) -> None:

        self.torch_available: bool = _TORCH_AVAILABLE
        self.cuda_available: bool = _CUDA_AVAILABLE
        self.mps_available: bool = _MPS_AVAILABLE
        self.detector_backend: str = self._select_backend()

        logger.info(
            "DeviceConfig: CUDA=%s  MPS=%s  → backend=%s",
            self.cuda_available,
            self.mps_available,
            self.detector_backend,
        )

    # ------------------------------------------------------------------

    def _select_backend(self) -> str:

        """
        DeepFace backend selection ladder.

        RetinaFace  – most accurate, but heavy. Only worth it with CUDA because
                    on CPU it runs ~4x slower than OpenCV.

        MTCNN       – good accuracy, moderate weight. On Apple Silicon the MPS
                    Metal backend accelerates the underlying TF/Keras ops, giving
                    better accuracy than OpenCV at acceptable latency.

        OpenCV      – fastest on pure CPU. Accuracy is lower but
                    latency matters more than precision for a live-feed dashboard.
        """

        if self.cuda_available:
            return "retinaface"
        if self.mps_available:
            return "mtcnn"
        return "opencv"


# ---------------------------------------------------------------------------
# AppSettings – all tunables in one place (reads from env with defaults)
# ---------------------------------------------------------------------------

class AppSettings:
    """
    All application-level settings.

    Every value can be overridden with an environment variable so the same
    Docker image works across local / staging / production without code changes.

    Usage:
        from config.settings import settings
        print(settings.detection_interval)
    """

    # --- camera ---
    default_camera_type: str = os.getenv("CAMERA_TYPE", "local_camera")
    default_camera_url: str | None = os.getenv("CAMERA_URL", None)

    # --- detection loop ---
    # Run DeepFace every N frames (higher = faster loop, lower = more detections)
    detection_interval: int = int(os.getenv("DETECTION_INTERVAL", "5"))

    # --- analysis ---
    # Width to downscale frames to before DeepFace inference (saves CPU/GPU time)
    analysis_target_width: int = int(os.getenv("ANALYSIS_TARGET_WIDTH", "640"))

    # DeepFace actions to request (changing this affects latency)
    deepface_actions: list[str] = ["age", "gender", "emotion"]

    # --- deduplication ---
    # Spatial quantisation bucket size (pixels) for signature generation
    dedup_bucket_px: int = int(os.getenv("DEDUP_BUCKET_PX", "20"))

    # Maximum unique detections kept in the feed
    max_unique_detections: int = int(os.getenv("MAX_UNIQUE_DETECTIONS", "30"))

    # Prune the signature set when it grows beyond this
    signature_prune_threshold: int = int(os.getenv("SIGNATURE_PRUNE_THRESHOLD", "40"))

    # --- face crop ---
    face_crop_size: tuple[int, int] = (80, 80)

    # --- server ---
    host: str = os.getenv("HOST", "127.0.0.1")
    port: int = int(os.getenv("PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "info")



settings = AppSettings()