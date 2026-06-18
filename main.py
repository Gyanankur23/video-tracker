"""
FastAPI Application for YOLO Classroom Object Detection
Provides REST API and live webcam demo
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import cv2
import numpy as np
from ultralytics import YOLO
import os
import io
from pathlib import Path
import asyncio
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="YOLO Classroom Object Detector", version="1.0.0")

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('templates', exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('uploads', exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Global model variable
model = None
model_lock = asyncio.Lock()

# COCO class names (80 classes - accurate pre-trained model)
COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep",
    19: "cow", 20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe",
    24: "backpack", 25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase",
    29: "frisbee", 30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard", 37: "surfboard",
    38: "tennis racket", 39: "bottle", 40: "wine glass", 41: "cup",
    42: "fork", 43: "knife", 44: "spoon", 45: "bowl", 46: "banana",
    47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
    51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake",
    56: "chair", 57: "couch", 58: "potted plant", 59: "bed", 60: "dining table",
    61: "toilet", 62: "tv", 63: "laptop", 64: "mouse", 65: "remote",
    66: "keyboard", 67: "cell phone", 68: "microwave", 69: "oven",
    70: "toaster", 71: "sink", 72: "refrigerator", 73: "book", 74: "clock",
    75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier",
    79: "toothbrush"
}


def load_model():
    """
    Load YOLO pre-trained COCO model for accurate object detection
    """
    global model
    
    # Always use pre-trained COCO model for accurate identification
    logger.info("Loading pre-trained YOLOv8n COCO model for accurate detection")
    
    if os.path.exists('yolov8n.pt'):
        model = YOLO('yolov8n.pt')
    elif os.path.exists('models/best.pt'):
        model = YOLO('models/best.pt')
    else:
        logger.info("Downloading pre-trained YOLOv8n model...")
        model = YOLO('yolov8n.pt')
    
    return model, "coco"


def get_class_name(class_id: int):
    """
    Get class name from COCO dataset
    """
    return COCO_CLASSES.get(class_id, f"class_{class_id}")


def draw_detections(frame, results, confidence_threshold: float = 0.5):
    """
    Draw detection boxes on frame with green color
    """
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Get confidence and class
            confidence = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            
            # Filter by confidence
            if confidence < confidence_threshold:
                continue
            
            # Get class name from COCO
            class_name = get_class_name(class_id)
            
            # Draw green bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Draw label with confidence
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Draw label background
            cv2.rectangle(frame, (int(x1), int(y1) - label_size[1] - 10),
                          (int(x1) + label_size[0], int(y1)), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(frame, label, (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return frame


@app.on_event("startup")
async def startup_event():
    """
    Load model on startup
    """
    logger.info("Loading YOLO model...")
    load_model()
    logger.info("Model loaded successfully")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Serve the live demo page
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    """
    Detect objects in uploaded image using COCO pre-trained model
    """
    global model
    
    if model is None:
        load_model()
    
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Run inference
        results = model(img)
        
        # Draw detections with COCO classes
        annotated_img = draw_detections(img.copy(), results)
        
        # Convert to bytes
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_bytes = buffer.tobytes()
        
        # Extract detection results
        detections = []
        for result in results:
            for box in result.boxes:
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                if confidence >= 0.5:
                    detections.append({
                        "class": get_class_name(class_id),
                        "confidence": confidence,
                        "bbox": box.xyxy[0].cpu().numpy().tolist()
                    })
        
        return StreamingResponse(
            io.BytesIO(img_bytes),
            media_type="image/jpeg",
            headers={
                "X-Detection-Count": str(len(detections)),
                "X-Detections": str(detections)
            }
        )
        
    except Exception as e:
        logger.error(f"Error during detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/video_feed")
async def video_feed():
    """
    Streaming endpoint for webcam video with detection using COCO model
    """
    global model
    
    if model is None:
        load_model()
    
    def generate_frames():
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            logger.error("Could not open webcam")
            return
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Run inference
                results = model(frame)
                
                # Draw detections with COCO classes
                annotated_frame = draw_detections(frame, results)
                
                # Convert to JPEG
                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        finally:
            cap.release()
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": "COCO pre-trained (80 classes)"
    }


@app.get("/model_info")
async def model_info():
    """
    Get model information
    """
    global model
    
    if model is None:
        load_model()
    
    return {
        "model_type": "COCO pre-trained YOLOv8n",
        "model_path": "yolov8n.pt",
        "classes": COCO_CLASSES,
        "num_classes": 80,
        "description": "Accurate object detection using 80 COCO classes including person, chair, bottle, backpack, and many classroom objects"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
