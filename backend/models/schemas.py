"""
models/schemas.py
-----------------
Pydantic schemas that define the API contracts for the face-detection pipeline.

"""

from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Primitive geometry
# ---------------------------------------------------------------------------

class BBox(BaseModel):
    """Bounding box in pixel coordinates (top-left origin)."""

    x1: int = Field(..., ge=0, description="Left edge")
    y1: int = Field(..., ge=0, description="Top edge")
    x2: int = Field(..., ge=0, description="Right edge")
    y2: int = Field(..., ge=0, description="Bottom edge")

    def to_list(self) -> List[int]:
        """Return [x1, y1, x2, y2] – the wire format used in JSON responses."""
        return [self.x1, self.y1, self.x2, self.y2]

    @classmethod
    def from_list(cls, bbox: List[int]) -> "BBox":
        """Construct from [x1, y1, x2, y2]."""
        return cls(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3])

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


# ---------------------------------------------------------------------------
# Detection results
# ---------------------------------------------------------------------------

class Face(BaseModel):
    """A single detected face with its attributes."""

    bbox: BBox
    age: int = Field(..., ge=0, le=120)
    gender: str
    emotion: str


class FaceWithImage(Face):
    """
    Face detection enriched with a base64-encoded 80×80 JPG crop.
    Used in the detection feed (unique detections panel).
    """

    image: str = Field("", description="Base64-encoded 80×80 JPG crop, empty if crop failed")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class DetectionMetrics(BaseModel):
    """
    Real-time performance snapshot.
    Returned as part of /live-frame and /detection-feed responses.
    """

    fps: float = Field(..., ge=0)
    inference_latency_ms: float = Field(..., ge=0)
    face_count: int = Field(..., ge=0)
    unique_detections_count: int = Field(..., ge=0)


# ---------------------------------------------------------------------------
# API response envelopes
# ---------------------------------------------------------------------------

class SnapshotResponse(BaseModel):
    timestamp: str
    faces: List[dict]  # [{bbox, emotion, age, gender}]


class StreamResponse(BaseModel):
    timestamp: str
    faces: List[dict]


class LiveFrameResponse(BaseModel):
    frame: str          # base64 JPG
    faces: List[dict]
    timestamp: Optional[str]
    metrics: dict


class DetectionFeedResponse(BaseModel):
    face_images: List[dict]
    timestamp: Optional[str]
    metrics: dict


class HealthResponse(BaseModel):
    status: str
    is_active: bool
    fps: float
    has_frame: bool