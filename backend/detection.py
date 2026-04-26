<<<<<<< HEAD

import cv2
import asyncio 
=======
import cv2
import threading
>>>>>>> 058ea6fbbc0b7667d8cb694607e26497fbe88b4d
import time
import logging
from deepface import DeepFace
import base64

logging.basicConfig(level=logging.INFO)
<<<<<<< HEAD
logger = logging.getLogger(__name__)
=======
>>>>>>> 058ea6fbbc0b7667d8cb694607e26497fbe88b4d

class DetectionManager:
    def __init__(self, use_local_camera=True, detection_interval=5):
        self.camera = None
        self.camera_url = 0 if use_local_camera else None
        self.is_active = False
<<<<<<< HEAD
        self.stop_requested = False                    
        self.latest_frame = None
        self.state_lock = asyncio.Lock()               
=======
        self.frame_thread = None
        self.detection_thread = None
        self.stop_thread_flag = threading.Event()
        self.latest_frame = None
        self.frame_lock = threading.Lock()
>>>>>>> 058ea6fbbc0b7667d8cb694607e26497fbe88b4d
        self.detection_interval = detection_interval
        self.frame_count = 0

        # Detection results
<<<<<<< HEAD
        self.detection_results = []
        self.face_images = []
        self.detection_events = []
        
        # Unique detection tracking (for feed)
        self.unique_detections = []
        self.detection_signatures = set()
=======
        self.detection_results = []       # raw face detections
        self.face_images = []             # base64 cropped faces
        self.detection_events = []        # per-face detection events
        
        # Unique detection tracking (for feed)
        self.unique_detections = []      # list of unique detections (max 30)
        self.detection_signatures = set()  # track unique detections by signature
>>>>>>> 058ea6fbbc0b7667d8cb694607e26497fbe88b4d

        # Metrics
        self.fps = 0.0
        self.last_inference_latency_ms = 0.0
        self.last_timestamp = None

<<<<<<< HEAD

    # -------------------------------------------------------------------------
    # CAMERA SETUP
    # -------------------------------------------------------------------------

=======
   
    # CAMERA SETUP
    
>>>>>>> 058ea6fbbc0b7667d8cb694607e26497fbe88b4d
    def set_camera(self, camera_type: str, camera_url: str = None):
        if camera_type == "local_camera":
            self.camera_url = 0
        elif camera_type == "ip_camera":
            self.camera_url = camera_url
<<<<<<< HEAD
        logger.info(f"Camera configured: {camera_type}, URL: {self.camera_url}")

    def start_camera(self):
        self.stop_camera_sync()  # release any existing camera first
=======
        logging.info(f"Camera configured: {camera_type}, URL: {self.camera_url}")

    def start_camera(self):
        self.stop_camera()
>>>>>>> 058ea6fbbc0b7667d8cb694607e26497fbe88b4d
        self.camera = cv2.VideoCapture(self.camera_url)
        if not self.camera.isOpened():
            self.is_active = False
            raise RuntimeError("Failed to open camera")
        self.is_active = True
<<<<<<< HEAD
        logger.info("Camera started.")

    def stop_camera_sync(self):
        """Synchronous camera release (used internally)"""
        if self.camera:
            self.camera.release()
            self.camera = None
        self.is_active = False


    # -------------------------------------------------------------------------
    # START / STOP DETECTION
    # -------------------------------------------------------------------------

    async def start_detection(self):
        if self.is_active:
            logger.info("Detection already active")
            return
        try:
            self.start_camera()         
            self.is_active = True
            self.stop_requested = False
            logger.info("Detection pipeline started")
        except Exception as e:
            logger.exception("Error starting detection")
            self.is_active = False
            raise

    async def stop_detection(self):
        self.stop_requested = True
        logger.info("Detection pipeline stopped")

    async def stop_camera(self):
        await self.stop_detection()
        self.is_active = False
        if self.camera:
            self.camera.release()
            self.camera = None
        cv2.destroyAllWindows()
        async with self.state_lock:
            self.unique_detections = []
            self.detection_signatures = set()
        logger.info("Camera stopped.")


    # -------------------------------------------------------------------------
    # MAIN DETECTION LOOP
    # -------------------------------------------------------------------------

    async def run_detection_loop(self):
        if not self.is_active:
            await self.start_detection()
        
        loop = asyncio.get_running_loop()
        prev_time = time.time()

        while not self.stop_requested:
            try:
                # Read frame
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    await asyncio.sleep(0.05)
                    continue

                # FPS calculation
                now = time.time()
                self.fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now

                self.frame_count += 1

                # Only run detection every N frames
                if self.frame_count % self.detection_interval != 0:
                    await asyncio.sleep(0.01)
                    continue

                # Run CPU-bound DeepFace inference in a thread pool executor
                # to preent the event loop from being blocked during inference.
                frame_copy = frame.copy()
                start_time = time.time()
                detections, face_images = await loop.run_in_executor(
                    None, self._analyze_frame, frame_copy
                )
                inference_time_ms = (time.time() - start_time) * 1000

                # Draw bounding boxes
                frame_with_boxes = frame.copy()
                for face in detections:
                    x1, y1, x2, y2 = face["bbox"]
                    cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    text = f"{face['gender']} | {face['age']} | {face['emotion']}"
                    cv2.putText(frame_with_boxes, text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
                current_time_ms = int(time.time() * 1000)

                # Precompute candidate signatures outside the lock, then perform
                # atomic check-and-insert inside a single lock to eliminate race conditions.
                candidates = []
                for idx, face_img in enumerate(face_images):
                    x1, y1, x2, y2 = face_img["bbox"]
                    sig_x = (x1 // 20) * 20
                    sig_y = (y1 // 20) * 20
                    signature = f"{sig_x}_{sig_y}_{face_img['age']}_{face_img['gender']}_{face_img['emotion']}"
                    candidates.append((signature, idx, face_img))

                # Atomic state update — the signature check, insertion, and all
                # shared-state mutations happen inside a single lock acquisition.
                async with self.state_lock:
                    self.latest_frame = frame_with_boxes.copy()
                    self.detection_results = detections
                    self.face_images = face_images
                    self.last_inference_latency_ms = inference_time_ms
                    self.last_timestamp = timestamp

                    for signature, idx, face_img in candidates:

                        if signature not in self.detection_signatures:
                            self.unique_detections.insert(0, {
                                "_uid": f"{current_time_ms}_{idx}",
                                "timestamp": timestamp,
                                "bbox": face_img["bbox"],
                                "image": face_img["image"],
                                "age": face_img["age"],
                                "gender": face_img["gender"],
                                "emotion": face_img["emotion"]
                            })
                            self.detection_signatures.add(signature)

                    # Keep the feed list bounded
                    self.unique_detections = self.unique_detections[:30]

                    # Lazy cleanup when signatures grow too large
                    if len(self.detection_signatures) > 40:
                        recent = set()
                        for det in self.unique_detections:
                            x, y = det["bbox"][0], det["bbox"][1]
                            sx = (x // 20) * 20
                            sy = (y // 20) * 20
                            recent.add(f"{sx}_{sy}_{det['age']}_{det['gender']}_{det['emotion']}")
                        self.detection_signatures = recent

                await asyncio.sleep(0.01)

            except Exception as e:
                logger.exception("Error in detection loop")
                await asyncio.sleep(0.1)


    # -------------------------------------------------------------------------
    # ANALYSIS  
    # -------------------------------------------------------------------------

    def _analyze_frame(self, frame):
=======
        self.frame_thread = threading.Thread(target=self._frame_capture_loop, daemon=True)
        self.frame_thread.start()
        logging.info("Camera started.")

    def _frame_capture_loop(self):
        prev = time.time()
        while self.is_active and self.camera.isOpened():
            ret, frame = self.camera.read()
            if not ret:
                continue
            with self.frame_lock:
                self.latest_frame = frame.copy()
            now = time.time()
            self.fps = 1.0 / max(now - prev, 1e-6)
            prev = now

   
    # START DETECTION
    
    def start_detection(self):
        if not self.is_active:
            self.start_camera()
        if self.detection_thread and self.detection_thread.is_alive():
            return
        self.stop_thread_flag.clear()
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()
        logging.info("Detection started.")

    def _detection_loop(self):
        while not self.stop_thread_flag.is_set():
            # Get frame quickly (FOR minimal lock time)
            with self.frame_lock:
                frame = self.latest_frame.copy() if self.latest_frame is not None else None

            if frame is None:
                time.sleep(0.05)
                continue

            self.frame_count += 1
            if self.frame_count % self.detection_interval != 0:
                time.sleep(0.01)
                continue

            start_time = time.time()
            detections, face_images = self._analyze_frame(frame)

            # Draw bounding boxes
            for face in detections:
                x1, y1, x2, y2 = face["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{face['gender']} | {face['age']} | {face['emotion']}"
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Create detection events: one per face
            detection_events = []
            timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
            current_time_ms = int(time.time() * 1000)
            
            # Pre-encode raw frame once (used by all detection events)
            raw_frame_b64 = self._encode_frame(frame)

            # Process unique detections OUTSIDE the main frame lock to avoid blocking
            new_unique_detections = []
            signatures_to_add = []
            
            for idx, face_img in enumerate(face_images):
                # Create a signature based on bbox position (rounded to avoid minor position changes)
                x1, y1, x2, y2 = face_img["bbox"]
                # Round bbox to nearest 20 pixels to group similar positions
                sig_x = (x1 // 20) * 20
                sig_y = (y1 // 20) * 20
                signature = f"{sig_x}_{sig_y}_{face_img['age']}_{face_img['gender']}_{face_img['emotion']}"
                
                # Check if this is a unique detection (quick check without lock)
                if signature not in self.detection_signatures:
                    # Add to unique detections
                    unique_detection = {
                        "_uid": f"{current_time_ms}_{idx}",
                        "timestamp": timestamp,
                        "bbox": face_img["bbox"],
                        "image": face_img["image"],
                        "age": face_img["age"],
                        "gender": face_img["gender"],
                        "emotion": face_img["emotion"]
                    }
                    new_unique_detections.append(unique_detection)
                    signatures_to_add.append(signature)
                
                # Create event for detection_events (for compatibility)
                detection_events.append({
                    "_uid": f"{current_time_ms}_{idx}",
                    "timestamp": timestamp,
                    "faces": [detections[idx]],
                    "rawFrame": raw_frame_b64
                })

            # Update all state in a single lock acquisition
            with self.frame_lock:
                self.detection_results = detections
                self.face_images = face_images
                self.latest_frame = frame.copy()
                self.detection_events = detection_events
                self.last_inference_latency_ms = (time.time() - start_time) * 1000
                self.last_timestamp = timestamp
                
                # Update unique detections (only if we have new ones)
                if new_unique_detections:
                    # Add new unique detections to front of list and keep only last 30
                    self.unique_detections = new_unique_detections + self.unique_detections
                    self.unique_detections = self.unique_detections[:30]
                    
                    # Add new signatures
                    for sig in signatures_to_add:
                        self.detection_signatures.add(sig)
                    
                    # Clean up old signatures only when we exceed 30 (lazy cleanup)
                    if len(self.detection_signatures) > 40:  # Clean up when we have 40+ (buffer)
                        recent_signatures = set()
                        for det in self.unique_detections:
                            x, y = det["bbox"][0], det["bbox"][1]
                            sig_x = (x // 20) * 20
                            sig_y = (y // 20) * 20
                            sig = f"{sig_x}_{sig_y}_{det['age']}_{det['gender']}_{det['emotion']}"
                            recent_signatures.add(sig)
                        self.detection_signatures = recent_signatures

            time.sleep(0.01)

   
    # ANALYSIS
   
    def _analyze_frame(self, frame):
        """
        Run DeepFace on a DOWNSCALED version of the frame for speed, then
        map detections back to the original resolution.
        """
>>>>>>> 058ea6fbbc0b7667d8cb694607e26497fbe88b4d
        results_list = []
        face_images = []

        try:
<<<<<<< HEAD
=======
            # Downscale frame to speed up DeepFace (keep aspect ratio)
>>>>>>> 058ea6fbbc0b7667d8cb694607e26497fbe88b4d
            h, w = frame.shape[:2]
            target_width = 640
            if w > target_width:
                scale = target_width / float(w)
<<<<<<< HEAD
                small_frame = cv2.resize(frame, (target_width, int(h * scale)))
=======
                new_w = target_width
                new_h = int(h * scale)
                small_frame = cv2.resize(frame, (new_w, new_h))
>>>>>>> 058ea6fbbc0b7667d8cb694607e26497fbe88b4d
            else:
                small_frame = frame
                scale = 1.0

            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            results = DeepFace.analyze(
                rgb_small,
                actions=["age", "gender", "emotion"],
                enforce_detection=False,
                detector_backend="opencv",
            )

            if not isinstance(results, list):
                results = [results]

            now_ms = int(time.time() * 1000)

            for idx, r in enumerate(results):
                if "region" not in r:
                    continue

                rx, ry, rw, rh = (
                    r["region"]["x"],
                    r["region"]["y"],
                    r["region"]["w"],
                    r["region"]["h"],
                )

<<<<<<< HEAD
                x  = max(0, min(int(rx / scale), w - 1))
                y  = max(0, min(int(ry / scale), h - 1))
                x2 = max(0, min(int((rx + rw) / scale), w))
                y2 = max(0, min(int((ry + rh) / scale), h))

                crop = frame[y:y2, x:x2]
                crop_b64 = None
                if crop.size != 0:
                    _, buffer = cv2.imencode(".jpg", cv2.resize(crop, (80, 80)))
                    crop_b64 = base64.b64encode(buffer).decode("utf-8")

                results_list.append({
                    "bbox": [x, y, x2, y2],
                    "age": int(r["age"]),
                    "gender": r["dominant_gender"],
                    "emotion": r["dominant_emotion"],
                })

                face_images.append({
                    "_uid": f"{now_ms}_{idx}",
                    "bbox": [x, y, x2, y2],
                    "image": crop_b64,
                    "age": int(r["age"]),
                    "gender": r["dominant_gender"],
                    "emotion": r["dominant_emotion"],
                })

        except Exception as e:
            logger.error(f"DeepFace error: {e}")

        return results_list, face_images


    # -------------------------------------------------------------------------
    # FRAME ENCODING
    # -------------------------------------------------------------------------

=======
                # Map bbox back to original frame size
                x = int(rx / scale)
                y = int(ry / scale)
                w_box = int(rw / scale)
                h_box = int(rh / scale)
                x2 = x + w_box
                y2 = y + h_box

                # Clamp to frame bounds
                x = max(0, min(x, w - 1))
                y = max(0, min(y, h - 1))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))

                # Safely crop face and resize to 80x80
                crop = frame[y:y2, x:x2]
                crop_b64 = None
                if crop.size != 0:
                    crop_resized = cv2.resize(crop, (80, 80))
                    _, buffer = cv2.imencode(".jpg", crop_resized)
                    crop_b64 = base64.b64encode(buffer).decode("utf-8")

                results_list.append(
                    {
                        "bbox": [x, y, x2, y2],
                        "age": int(r["age"]),
                        "gender": r["dominant_gender"],
                        "emotion": r["dominant_emotion"],
                    }
                )

                face_images.append(
                    {
                        "_uid": f"{now_ms}_{idx}",
                        "bbox": [x, y, x2, y2],
                        "image": crop_b64,
                        "age": int(r["age"]),
                        "gender": r["dominant_gender"],
                        "emotion": r["dominant_emotion"],
                    }
                )

        except Exception as e:
            logging.error(f"DeepFace error: {e}")

        return results_list, face_images

    
    # FRAME ENCODING
   
>>>>>>> 058ea6fbbc0b7667d8cb694607e26497fbe88b4d
    def _encode_frame(self, frame):
        _, buffer = cv2.imencode(".jpg", frame)
        return base64.b64encode(buffer).decode("utf-8")

<<<<<<< HEAD

    # -------------------------------------------------------------------------
    # JSON OUTPUTS  
    # -------------------------------------------------------------------------

    async def get_health_status(self):
        async with self.state_lock:
            return {
                "status": "ok" if self.is_active else "error",
                "is_active": self.is_active,
                "fps": float(self.fps),
                "has_frame": self.latest_frame is not None
            }

    async def get_snapshot_json(self):
        async with self.state_lock:
            if self.latest_frame is None:
                return None
            return {
                "timestamp": self.last_timestamp or time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
                "faces": [
                    {
                        "bbox": face["bbox"],
                        "emotion": face["emotion"],
                        "age": face["age"],
                        "gender": face["gender"]
                    }
                    for face in self.detection_results
                ]
            }

    async def get_stream_json(self):
        async with self.state_lock:
            return {
                "timestamp": self.last_timestamp or time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
                "faces": [
                    {
                        "bbox": face["bbox"],
                        "emotion": face["emotion"],
                        "age": face["age"],
                        "gender": face["gender"]
                    }
                    for face in self.detection_results
                ]
            }

    async def get_live_frame_json(self):
        async with self.state_lock:
            if self.latest_frame is None:
                return None
            frame_b64 = self._encode_frame(self.latest_frame)
            formatted_faces = [
                {
                    "region": {"x": face["bbox"][0], "y": face["bbox"][1],
                               "w": face["bbox"][2] - face["bbox"][0],
                               "h": face["bbox"][3] - face["bbox"][1]},
                    "age": face["age"],
                    "gender": face["gender"],
                    "emotion": face["emotion"]
                }
                for face in self.detection_results
            ]
            return {
                "frame": frame_b64,
                "faces": formatted_faces,
                "timestamp": self.last_timestamp,
                "metrics": {
                    "fps": float(self.fps),
                    "last_latency_ms": float(self.last_inference_latency_ms),
                    "face_count": len(self.detection_results)
                }
            }

    async def get_detection_feed_json(self):
        async with self.state_lock:
            return {
                "face_images": self.unique_detections,
                "timestamp": self.last_timestamp,
                "metrics": {
                    "fps": float(self.fps),
                    "last_latency_ms": float(self.last_inference_latency_ms),
                    "face_count": len(self.detection_results),
                    "unique_detections_count": len(self.unique_detections)
                }
            }

    async def get_detection_json(self):
        """Legacy method - kept for compatibility"""
        async with self.state_lock:
=======
    def get_frame(self):
        """Get current frame as base64 string"""
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            return self._encode_frame(self.latest_frame)

   
    # JSON FOR FRONTEND / FEED
   
    def get_detection_json(self):
        """Legacy method - kept for compatibility"""
        with self.frame_lock:
>>>>>>> 058ea6fbbc0b7667d8cb694607e26497fbe88b4d
            return {
                "timestamp": self.last_timestamp,
                "faces": self.detection_results,
                "face_images": self.face_images,
                "events": self.detection_events,
                "metrics": {
                    "fps": float(self.fps),
                    "last_latency_ms": float(self.last_inference_latency_ms),
                    "face_count": len(self.detection_results)
                }
            }

<<<<<<< HEAD
    async def get_frame(self):
        async with self.state_lock:
            if self.latest_frame is None:
                return None
            return self._encode_frame(self.latest_frame)




=======
    def get_health_status(self):
        """Get health status for /health endpoint"""
        with self.frame_lock:
            return {
                "status": "ok" if self.is_active else "error",
                "is_active": self.is_active,
                "fps": float(self.fps),
                "has_frame": self.latest_frame is not None
            }

    def get_snapshot_json(self):
        """Get snapshot data for /snapshot endpoint - returns assignment format"""
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            # Return in assignment-required format
            return {
                "timestamp": self.last_timestamp or time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
                "faces": [
                    {
                        "bbox": face["bbox"],
                        "emotion": face["emotion"],
                        "age": face["age"],
                        "gender": face["gender"]
                    }
                    for face in self.detection_results
                ]
            }
    
    def get_stream_json(self):
        """Get stream data in assignment-required format for /stream WebSocket"""
        with self.frame_lock:
            return {
                "timestamp": self.last_timestamp or time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
                "faces": [
                    {
                        "bbox": face["bbox"],
                        "emotion": face["emotion"],
                        "age": face["age"],
                        "gender": face["gender"]
                    }
                    for face in self.detection_results
                ]
            }

    def get_live_frame_json(self):
        """Get live frame data for /frame endpoint and /detection/live websocket"""
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            frame_b64 = self._encode_frame(self.latest_frame)
            
            # Format faces for frontend (with region for bounding boxes)
            formatted_faces = []
            for face in self.detection_results:
                x1, y1, x2, y2 = face["bbox"]
                formatted_faces.append({
                    "region": {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1},
                    "age": face["age"],
                    "gender": face["gender"],
                    "emotion": face["emotion"]
                })
            
            return {
                "frame": frame_b64,
                "faces": formatted_faces,
                "timestamp": self.last_timestamp,
                "metrics": {
                    "fps": float(self.fps),
                    "last_latency_ms": float(self.last_inference_latency_ms),
                    "face_count": len(self.detection_results)
                }
            }

    def get_detection_feed_json(self):
        """Get detection feed data for /detection/feed websocket - returns unique detections only"""
        with self.frame_lock:
            # Return only new unique detections to the feed
           
            return {
                "face_images": self.unique_detections,
                "timestamp": self.last_timestamp,
                "metrics": {
                    "fps": float(self.fps),
                    "last_latency_ms": float(self.last_inference_latency_ms),
                    "face_count": len(self.detection_results),
                    "unique_detections_count": len(self.unique_detections)
                }
            }

   
    # STOP DETECTION / CAMERA
    
    def stop_detection(self):
        self.stop_thread_flag.set()
        if self.detection_thread:
            self.detection_thread.join()
        logging.info("Detection stopped.")

    def stop_camera(self):
        self.is_active = False
        if self.camera:
            self.camera.release()
            self.camera = None
        cv2.destroyAllWindows()
        # Clear detection tracking
        with self.frame_lock:
            self.unique_detections = []
            self.detection_signatures = set()
        logging.info("Camera stopped.")
>>>>>>> 058ea6fbbc0b7667d8cb694607e26497fbe88b4d
