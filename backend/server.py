import asyncio
import traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging

from detection import DetectionManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

<<<<<<< HEAD
detection_manager: DetectionManager = None
detection_task: asyncio.Task = None
=======
detection_manager = DetectionManager()

>>>>>>> 058ea6fbbc0b7667d8cb694607e26497fbe88b4d


# Startup event: auto-start detection

@app.on_event("startup")
async def startup_event():
<<<<<<< HEAD
    global detection_manager, detection_task          
    try:
        detection_manager = DetectionManager()       
        await detection_manager.start_detection()     
        detection_task = asyncio.create_task(         
            detection_manager.run_detection_loop()
        )
=======
    try:
        detection_manager.start_detection()
>>>>>>> 058ea6fbbc0b7667d8cb694607e26497fbe88b4d
        logger.info("Camera and detection auto-started on startup.")
    except Exception as e:
        logger.warning(f"Could not auto-start detection on startup: {e}")



# Root

@app.get("/")
async def root():
    return {"message": "Vision service running"}



# Camera controls

@app.post("/detection/set-camera")
async def set_camera(source_type: str = Body(...), camera_url: str = Body(None)):
<<<<<<< HEAD
=======
    """
    Configure camera. source_type: "webcam" or "ip_camera".
    camera_url: optional URL for IP camera.
    """
>>>>>>> 058ea6fbbc0b7667d8cb694607e26497fbe88b4d
    try:
        detection_manager.set_camera(source_type, camera_url)
        return {"message": "Camera configured", "camera_url": detection_manager.camera_url}
    except Exception as e:
        logger.exception("Error setting camera")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/detection/start")
async def start_detection():
<<<<<<< HEAD
    try:
        await detection_manager.start_detection()   
=======
    """
    Start camera + detection loop. Idempotent.
    """
    try:
        detection_manager.start_detection()
>>>>>>> 058ea6fbbc0b7667d8cb694607e26497fbe88b4d
        return {"status": "success", "message": "Detection started"}
    except Exception as e:
        logger.exception("Error starting detection")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detection/stop")
async def stop_detection():
<<<<<<< HEAD
    try:
        await detection_manager.stop_detection()      
        await detection_manager.stop_camera()         
=======
    """
    Stop detection and camera gracefully.
    """
    try:
        detection_manager.stop_detection()
        detection_manager.stop_camera()
>>>>>>> 058ea6fbbc0b7667d8cb694607e26497fbe88b4d
        return {"status": "success", "message": "Detection stopped"}
    except Exception as e:
        logger.exception("Error stopping detection")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/detection/status")
async def get_status():
    return {"isActive": detection_manager.is_active}



# Health, snapshot, and frame endpoints
<<<<<<< HEAD

@app.get("/health")
async def health():
    try:
        return await detection_manager.get_health_status()    
=======
@app.get("/health")
async def health():
    try:
        return detection_manager.get_health_status()
>>>>>>> 058ea6fbbc0b7667d8cb694607e26497fbe88b4d
    except Exception:
        logger.exception("Health check error")
        return JSONResponse({"status": "error"}, status_code=500)


@app.get("/snapshot")
async def snapshot():
<<<<<<< HEAD
    try:
        data = await detection_manager.get_snapshot_json()    
=======
    """
    Returns the latest inference result in assignment-required format.
    Format: { "timestamp": "...", "faces": [{ "bbox": [x1,y1,x2,y2], "emotion": "...", "age": int, "gender": "..." }] }
    """
    try:
        data = detection_manager.get_snapshot_json()
>>>>>>> 058ea6fbbc0b7667d8cb694607e26497fbe88b4d
        if not data:
            return JSONResponse({"timestamp": "", "faces": []}, status_code=200)
        return data
    except Exception:
        logger.exception("Snapshot error")
        return JSONResponse({"error": "Internal error"}, status_code=500)


@app.get("/frame")
async def frame():
    try:
<<<<<<< HEAD
        data = await detection_manager.get_live_frame_json()  
=======
        data = detection_manager.get_live_frame_json()
>>>>>>> 058ea6fbbc0b7667d8cb694607e26497fbe88b4d
        if not data:
            return JSONResponse({"frame": None, "faces": [], "metrics": {}}, status_code=200)
        return JSONResponse(data, status_code=200)
    except Exception as e:
        logger.exception("Frame error")
        return JSONResponse({"error": "Internal error", "details": str(e)}, status_code=500)



# WebSocket: Live frame

@app.websocket("/detection/live")
async def live_ws(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected to /detection/live websocket")
    try:
        while True:
            try:
<<<<<<< HEAD
                data = await detection_manager.get_live_frame_json()  
                if data:
                    await websocket.send_json(data)
=======
                data = detection_manager.get_live_frame_json()
                if data:
                    # send JSON; if send fails, break to cleanup
                    await websocket.send_json(data)
                # throttle; ~20 FPS send attempt
>>>>>>> 058ea6fbbc0b7667d8cb694607e26497fbe88b4d
                await asyncio.sleep(0.05)
            except WebSocketDisconnect:
                logger.info("Live WebSocket: client disconnected")
                break
            except Exception as e:
<<<<<<< HEAD
                logger.debug("Exception inside live_ws loop (non-fatal): %s", e)
=======
                # If send_json fails (broken pipe) or other error, log and try to continue
                logger.debug("Exception inside live_ws loop (non-fatal): %s", e)
                # small delay before continuing to avoid busy-looping on persistent error
>>>>>>> 058ea6fbbc0b7667d8cb694607e26497fbe88b4d
                await asyncio.sleep(0.2)
    except Exception:
        logger.exception("Unhandled exception in live websocket handler")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info("Live websocket connection closed")



<<<<<<< HEAD
# WebSocket: /stream

@app.websocket("/stream")
async def stream_ws(websocket: WebSocket):
=======
# WebSocket: /stream 

@app.websocket("/stream")
async def stream_ws(websocket: WebSocket):
    """
    Assignment-required WebSocket endpoint for real-time detections.
    Sends JSON in format: { "timestamp": "...", "faces": [{ "bbox": [...], "emotion": "...", "age": int, "gender": "..." }] }
    """
>>>>>>> 058ea6fbbc0b7667d8cb694607e26497fbe88b4d
    await websocket.accept()
    logger.info("Client connected to /stream websocket")
    try:
        while True:
            try:
<<<<<<< HEAD
                data = await detection_manager.get_stream_json()      
                if data:
                    await websocket.send_json(data)
=======
                data = detection_manager.get_stream_json()
                if data:
                    await websocket.send_json(data)
                # Send updates at ~10 FPS
>>>>>>> 058ea6fbbc0b7667d8cb694607e26497fbe88b4d
                await asyncio.sleep(0.1)
            except WebSocketDisconnect:
                logger.info("Stream WebSocket: client disconnected")
                break
            except Exception as e:
                logger.debug("Exception inside stream_ws loop (non-fatal): %s", e)
                await asyncio.sleep(0.2)
    except Exception:
        logger.exception("Unhandled exception in stream websocket handler")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info("Stream websocket connection closed")



# WebSocket: Detection feed

@app.websocket("/detection/feed")
async def feed_ws(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected to /detection/feed websocket")
    try:
        while True:
            try:
<<<<<<< HEAD
                data = await detection_manager.get_detection_feed_json() 
                if data:
                    await websocket.send_json(data)
=======
                data = detection_manager.get_detection_feed_json()
                if data:
                    await websocket.send_json(data)
                # send fewer feed updates (face thumbnails) than frames
>>>>>>> 058ea6fbbc0b7667d8cb694607e26497fbe88b4d
                await asyncio.sleep(0.2)
            except WebSocketDisconnect:
                logger.info("Feed WebSocket: client disconnected")
                break
            except Exception as e:
                logger.debug("Exception inside feed_ws loop (non-fatal): %s", e)
                await asyncio.sleep(0.2)
    except Exception:
        logger.exception("Unhandled exception in feed websocket handler")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info("Feed websocket connection closed")



<<<<<<< HEAD
# Shutdown hooks

@app.on_event("shutdown")
async def shutdown_event():                              
    global detection_task
    try:
        logger.info("Shutting down: stopping detection and camera")
        if detection_manager:
            await detection_manager.stop_detection()      
            await detection_manager.stop_camera()         
        if detection_task and not detection_task.done():
            detection_task.cancel()
            try:
                await detection_task
            except asyncio.CancelledError:
                pass
=======
#  shutdown hooks

@app.on_event("shutdown")
def shutdown_event():
    try:
        logger.info("Shutting down: stopping detection and camera")
        detection_manager.stop_detection()
        detection_manager.stop_camera()
>>>>>>> 058ea6fbbc0b7667d8cb694607e26497fbe88b4d
    except Exception:
        logger.exception("Error during shutdown")



# Run with uvicorn

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000)



<<<<<<< HEAD







=======
>>>>>>> 058ea6fbbc0b7667d8cb694607e26497fbe88b4d
