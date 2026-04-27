"""
detection.py
------------
"""

from services.detection_manager import DetectionManager          # noqa: F401
from services.camera import CameraService                        # noqa: F401
from services.analysis import AnalysisService                    # noqa: F401
from services.metrics import MetricsService                      # noqa: F401
from models.schemas import BBox, Face, FaceWithImage, DetectionMetrics  # noqa: F401
from config.settings import DeviceConfig, settings               # noqa: F401

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






# import cv2
# import asyncio 
# import time
# import logging
# from deepface import DeepFace
# import base64


# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # GPU/CUDA/MPS Detection
# try:
#     import torch
#     TORCH_AVAILABLE = True
#     CUDA_AVAILABLE = torch.cuda.is_available()
#     # Apple Silicon / Mac GPU support 
#     MPS_AVAILABLE = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
# except ImportError:
#     TORCH_AVAILABLE = False
#     CUDA_AVAILABLE = False
#     MPS_AVAILABLE = False

# logger.info(f"GPU Detection: torch={TORCH_AVAILABLE}, CUDA={CUDA_AVAILABLE}, MPS={MPS_AVAILABLE}")


# class DeviceConfig:
#     """
#     GPU/CPU device configuration.
#     Automatically checks CUDA availability and selects optimal backend.
#     """
    
#     def __init__(self):
#         # Store detection results
#         self.cuda_available = CUDA_AVAILABLE
#         self.mps_available = MPS_AVAILABLE
#         self.torch_available = TORCH_AVAILABLE
        
#         # Select backend based on device
#         self.detector_backend = self._select_backend()
        
#         # Log results
#         logger.info(f"Device Config: CUDA={self.cuda_available}, MPS={self.mps_available}, Torch={self.torch_available}")
#         logger.info(f"Selected backend: {self.detector_backend}")
#         logger.info(f"Rationale: {self._rationale()}")
    
#     def _select_backend(self) -> str:
       
#         if self.cuda_available:
#             # GPU available: prioritize accuracy + speed
#             return "retinaface"
        
#         #Apple Silicon GPU -better performance - mtcnn better than retinaface in most cases
#         elif self.mps_available:
#             return "mtcnn"
#         else:
#             # CPU only: prioritize speed
#             return "opencv"
    
#     def _rationale(self) -> str:
#         """
#         Explain why this backend was chosen.
#         """
#         if self.cuda_available:
#             return (
#             "CUDA detected: RetinaFace selected. "
#             "Best accuracy + speed on NVIDIA GPU (~20-40ms/frame). "
#             "3-5x faster than CPU. Accuracy and speed both achievable."
#             )
#         elif self.mps_available:
#             return (
    
#             "MPS detected: MTCNN selected. "
#             "RetinaFace has unstable MPS support in DeepFace. "
#             "MTCNN gives better accuracy than OpenCV with stable MPS support (~40-80ms/frame). "
#             )
    
#         else:
#             return (
#             "CPU-only: OpenCV selected. "
#             "Fastest detector on CPU (~50-100ms/frame). "
#             "Heavier backends add 2-4x latency with marginal accuracy gain."
#             )

# class DetectionManager:
#     def __init__(self, use_local_camera=True, detection_interval=5):
#         # Initialize device config (GPU/CPU detection)
#         self.device_config = DeviceConfig()  
        
#         # Camera
#         self.camera = None
#         self.camera_url = 0
#         self.is_active = False
        
#         # Async
#         self.state_lock = asyncio.Lock()
#         self.stop_requested = False
        
                         
#         self.latest_frame = None             
#         self.detection_interval = detection_interval
#         self.frame_count = 0

#         # Detection results
#         self.detection_results = []
#         self.face_images = []
#         self.detection_events = []
        
#         # Unique detection tracking (for feed)
#         self.unique_detections = []
#         self.detection_signatures = set()

#         # Metrics
#         self.fps = 0.0
#         self.last_inference_latency_ms = 0.0
#         self.last_timestamp = None


#     # -------------------------------------------------------------------------
#     # CAMERA SETUP
#     # -------------------------------------------------------------------------

#     def set_camera(self, camera_type: str, camera_url: str = None):
#         if camera_type == "local_camera":
#             self.camera_url = 0
#         elif camera_type == "ip_camera":
#             self.camera_url = camera_url
#         logger.info(f"Camera configured: {camera_type}, URL: {self.camera_url}")

#     def start_camera(self):
#         self.stop_camera_sync()  # release any existing camera first
    
#         # Reset stop flag so the loop can run again
#         self.stop_requested = False  
#         self.frame_count = 0
        
#         self.camera = cv2.VideoCapture(self.camera_url)
#         if not self.camera.isOpened():
#             self.is_active = False
#             raise RuntimeError("Failed to open camera")
#         self.is_active = True
#         logger.info("Camera started.")

#     async def stop_camera_sync(self):
#         # Signal the loop to stop
#         self.stop_requested = True
        
#         # Wait for the loop to finish its current iteration
#         await asyncio.sleep(0.2)
        
#         # Release camera
#         if self.camera:
#             self.camera.release()
#             self.camera = None
        
#         self.is_active = False
#         self.frame_count = 0  # ← reset so detection_interval works fresh on restart
        
#         cv2.destroyAllWindows()
        
#         async with self.state_lock:
#             self.unique_detections = []
#             self.detection_signatures = set()
        
#         logger.info("Camera stopped.")

#     # -------------------------------------------------------------------------
#     # START / STOP DETECTION
#     # -------------------------------------------------------------------------

#     async def start_detection(self):
#         if self.is_active:
#             logger.info("Detection already active")
#             return
#         try:
#             self.start_camera()         
#             self.is_active = True
#             self.stop_requested = False
#             logger.info("Detection pipeline started")
#         except Exception as e:
#             logger.exception("Error starting detection")
#             self.is_active = False
#             raise

#     async def stop_detection(self):
#         self.stop_requested = True
#         logger.info("Detection pipeline stopped")

#     async def stop_camera(self):
#         await self.stop_detection()
#         self.is_active = False
#         if self.camera:
#             self.camera.release()
#             self.camera = None
#         cv2.destroyAllWindows()
#         async with self.state_lock:
#             self.unique_detections = []
#             self.detection_signatures = set()
#         logger.info("Camera stopped.")


#     # -------------------------------------------------------------------------
#     # MAIN DETECTION LOOP
#     # -------------------------------------------------------------------------

#     async def run_detection_loop(self):
#         if not self.is_active:
#             await self.start_detection()
        
#         loop = asyncio.get_running_loop()
#         prev_time = time.time()

#         while not self.stop_requested:
#             try:
#                 # Read frame
#                 ret, frame = self.camera.read()
#                 if not ret or frame is None:
#                     await asyncio.sleep(0.05)
#                     continue

#                 # FPS calculation
#                 now = time.time()
#                 self.fps = 1.0 / max(now - prev_time, 1e-6)
#                 prev_time = now

#                 self.frame_count += 1

#                 # Only run detection every N frames
#                 if self.frame_count % self.detection_interval != 0:
#                     await asyncio.sleep(0.01)
#                     continue

#                 #  Run CPU-bound DeepFace inference in a thread pool executor
#                 # so the event loop is not blocked during inference.
#                 frame_copy = frame.copy()
#                 start_time = time.time()
#                 detections, face_images = await loop.run_in_executor(
#                     None, self._analyze_frame, frame_copy
#                 )
#                 inference_time_ms = (time.time() - start_time) * 1000

#                 # Draw bounding boxes
#                 frame_with_boxes = frame.copy()
#                 for face in detections:
#                     x1, y1, x2, y2 = face["bbox"]
#                     cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     text = f"{face['gender']} | {face['age']} | {face['emotion']}"
#                     cv2.putText(frame_with_boxes, text, (x1, y1 - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#                 timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
#                 current_time_ms = int(time.time() * 1000)

#                 # Precompute candidate signatures outside the lock, then perform
#                 # atomic check-and-insert under a single lock to eliminate race conditions.
#                 candidates = []
#                 for idx, face_img in enumerate(face_images):
#                     x1, y1, x2, y2 = face_img["bbox"]
#                     sig_x = (x1 // 20) * 20
#                     sig_y = (y1 // 20) * 20
#                     signature = f"{sig_x}_{sig_y}_{face_img['age']}_{face_img['gender']}_{face_img['emotion']}"
#                     candidates.append((signature, idx, face_img))

                
               
#                 # Atomic state update 
#                 async with self.state_lock:
#                     self.latest_frame = frame_with_boxes.copy()
#                     self.detection_results = detections
#                     self.face_images = face_images
#                     self.last_inference_latency_ms = inference_time_ms
#                     self.last_timestamp = timestamp

#                     for signature, idx, face_img in candidates:
                        
#                         if signature not in self.detection_signatures:
#                             self.unique_detections.insert(0, {
#                                 "_uid": f"{current_time_ms}_{idx}",
#                                 "timestamp": timestamp,
#                                 "bbox": face_img["bbox"],
#                                 "image": face_img["image"],
#                                 "age": face_img["age"],
#                                 "gender": face_img["gender"],
#                                 "emotion": face_img["emotion"]
#                             })
#                             self.detection_signatures.add(signature)

#                     # Keep the feed list bounded
#                     self.unique_detections = self.unique_detections[:30]

#                     # Lazy cleanup when signatures grow too large
#                     if len(self.detection_signatures) > 40:
#                         recent = set()
#                         for det in self.unique_detections:
#                             x, y = det["bbox"][0], det["bbox"][1]
#                             sx = (x // 20) * 20
#                             sy = (y // 20) * 20
#                             recent.add(f"{sx}_{sy}_{det['age']}_{det['gender']}_{det['emotion']}")
#                         self.detection_signatures = recent

#                 await asyncio.sleep(0.01)

#             except Exception as e:
#                 logger.exception("Error in detection loop")
#                 await asyncio.sleep(0.1)


#     # -------------------------------------------------------------------------
#     # ANALYSIS  
#     # -------------------------------------------------------------------------

#     def _analyze_frame(self, frame):
#         results_list = []
#         face_images = []

#         try:
#             h, w = frame.shape[:2]
#             target_width = 640
#             if w > target_width:
#                 scale = target_width / float(w)
#                 small_frame = cv2.resize(frame, (target_width, int(h * scale)))
#             else:
#                 small_frame = frame
#                 scale = 1.0

#             rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

#             results = DeepFace.analyze(
#                 rgb_small,
#                 actions=["age", "gender", "emotion"],
#                 enforce_detection=False,
#                 detector_backend=self.device_config.detector_backend,
#             )

#             if not isinstance(results, list):
#                 results = [results]

#             now_ms = int(time.time() * 1000)

#             for idx, r in enumerate(results):
#                 if "region" not in r:
#                     continue

#                 rx, ry, rw, rh = (
#                     r["region"]["x"],
#                     r["region"]["y"],
#                     r["region"]["w"],
#                     r["region"]["h"],
#                 )

#                 x  = max(0, min(int(rx / scale), w - 1))
#                 y  = max(0, min(int(ry / scale), h - 1))
#                 x2 = max(0, min(int((rx + rw) / scale), w))
#                 y2 = max(0, min(int((ry + rh) / scale), h))

#                 crop = frame[y:y2, x:x2]
#                 crop_b64 = None
#                 if crop.size != 0:
#                     _, buffer = cv2.imencode(".jpg", cv2.resize(crop, (80, 80)))
#                     crop_b64 = base64.b64encode(buffer).decode("utf-8")

#                 results_list.append({
#                     "bbox": [x, y, x2, y2],
#                     "age": int(r["age"]),
#                     "gender": r["dominant_gender"],
#                     "emotion": r["dominant_emotion"],
#                 })

#                 face_images.append({
#                     "_uid": f"{now_ms}_{idx}",
#                     "bbox": [x, y, x2, y2],
#                     "image": crop_b64,
#                     "age": int(r["age"]),
#                     "gender": r["dominant_gender"],
#                     "emotion": r["dominant_emotion"],
#                 })

#         except Exception as e:
#             logger.error(f"DeepFace error: {e}")

#         return results_list, face_images


#     # -------------------------------------------------------------------------
#     # FRAME ENCODING
#     # -------------------------------------------------------------------------

#     def _encode_frame(self, frame):
#         _, buffer = cv2.imencode(".jpg", frame)
#         return base64.b64encode(buffer).decode("utf-8")


#     # -------------------------------------------------------------------------
#     # JSON OUTPUTS  
#     # -------------------------------------------------------------------------

#     async def get_health_status(self):
#         async with self.state_lock:
#             return {
#                 "status": "ok" if self.is_active else "error",
#                 "is_active": self.is_active,
#                 "fps": float(self.fps),
#                 "has_frame": self.latest_frame is not None
#             }

#     async def get_snapshot_json(self):
#         async with self.state_lock:
#             if self.latest_frame is None:
#                 return None
#             return {
#                 "timestamp": self.last_timestamp or time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
#                 "faces": [
#                     {
#                         "bbox": face["bbox"],
#                         "emotion": face["emotion"],
#                         "age": face["age"],
#                         "gender": face["gender"]
#                     }
#                     for face in self.detection_results
#                 ]
#             }

#     async def get_stream_json(self):
#         async with self.state_lock:
#             return {
#                 "timestamp": self.last_timestamp or time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
#                 "faces": [
#                     {
#                         "bbox": face["bbox"],
#                         "emotion": face["emotion"],
#                         "age": face["age"],
#                         "gender": face["gender"]
#                     }
#                     for face in self.detection_results
#                 ]
#             }

#     async def get_live_frame_json(self):
#         async with self.state_lock:
#             if self.latest_frame is None:
#                 return None
#             frame_b64 = self._encode_frame(self.latest_frame)
#             formatted_faces = [
#                 {
#                     "region": {"x": face["bbox"][0], "y": face["bbox"][1],
#                                "w": face["bbox"][2] - face["bbox"][0],
#                                "h": face["bbox"][3] - face["bbox"][1]},
#                     "age": face["age"],
#                     "gender": face["gender"],
#                     "emotion": face["emotion"]
#                 }
#                 for face in self.detection_results
#             ]
#             return {
#                 "frame": frame_b64,
#                 "faces": formatted_faces,
#                 "timestamp": self.last_timestamp,
#                 "metrics": {
#                     "fps": float(self.fps),
#                     "last_latency_ms": float(self.last_inference_latency_ms),
#                     "face_count": len(self.detection_results)
#                 }
#             }

#     async def get_detection_feed_json(self):
#         async with self.state_lock:
#             return {
#                 "face_images": self.unique_detections,
#                 "timestamp": self.last_timestamp,
#                 "metrics": {
#                     "fps": float(self.fps),
#                     "last_latency_ms": float(self.last_inference_latency_ms),
#                     "face_count": len(self.detection_results),
#                     "unique_detections_count": len(self.unique_detections)
#                 }
#             }

#     async def get_detection_json(self):
#         """Legacy method - kept for compatibility"""
#         async with self.state_lock:
#             return {
#                 "timestamp": self.last_timestamp,
#                 "faces": self.detection_results,
#                 "face_images": self.face_images,
#                 "events": self.detection_events,
#                 "metrics": {
#                     "fps": float(self.fps),
#                     "last_latency_ms": float(self.last_inference_latency_ms),
#                     "face_count": len(self.detection_results)
#                 }
#             }

#     async def get_frame(self):
#         async with self.state_lock:
#             if self.latest_frame is None:
#                 return None
#             return self._encode_frame(self.latest_frame)
