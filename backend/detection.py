"""
detection.py
------------
Acts as a unified entry point for the detection pipeline by re-exporting key services (DetectionManager, CameraService, AnalysisService, MetricsService), data models, and configuration.
This simplifies imports and decouples external code from the internal project structure.

"""

from services.detection_manager import DetectionManager         
from services.camera import CameraService                        
from services.analysis import AnalysisService                  
from services.metrics import MetricsService                      
from models.schemas import BBox, Face, FaceWithImage, DetectionMetrics  
from config.settings import DeviceConfig, settings              

__all__ = [
    "DetectionManager",
    "CameraService",
    "AnalysisService",
    "MetricsService",
    "BBox",
    "Face",
    "FaceWithImage",
    "DetectionMetrics",
    "DeviceConfig",
    "settings",
]
