"""
FastAPI Application for YOLO Classroom Object Detection
Provides REST API for client-side webcam frame processing
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from contextlib import asynccontextmanager
import cv2
import numpy as np
from ultralytics import YOLO
import os
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variable
model = None

# Lifespan manager replaces deprecated startup events
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    logger.info("Loading YOLO model on startup...")
    # Always use pre-trained COCO model for accurate identification
    if os.path.exists('yolov8n.pt'):
        model = YOLO('yolov8n.pt')
    else:
        logger.info("Downloading pre-trained YOLOv8n model...")
        model = YOLO('yolov8n.pt')
    logger.info("Model loaded successfully.")
    yield
    # Clean up resources here if needed
    logger.info("Shutting down application...")

# Initialize FastAPI app
app = FastAPI(title="YOLO Classroom Object Detector", version="1.0.0", lifespan=lifespan)

# Create necessary directories
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Mount static files & templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

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

def get_class_name(class_id: int):
    return COCO_CLASSES.get(class_id, f"class_{class_id}")

def draw_detections(frame, results, confidence_threshold: float = 0.5):
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            
            if confidence < confidence_threshold:
                continue
            
            class_name = get_class_name(class_id)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (int(x1), int(y1) - label_size[1] - 10),
                          (int(x1) + label_size[0], int(y1)), (0, 255, 0), -1)
            cv2.putText(frame, label, (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return frame

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Note: Removed "async" so blocking YOLO processing runs safely in FastAPI's threadpool
@app.post("/detect")
def detect_objects(file: UploadFile = File(...)):
    global model
    if model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        # Read uploaded image frame
        contents = file.file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Run inference
        results = model(img)
        annotated_img = draw_detections(img.copy(), results)
        
        # Convert frame back to JPEG
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_bytes = buffer.tobytes()
        
        return StreamingResponse(io.BytesIO(img_bytes), media_type="image/jpeg")
        
    except Exception as e:
        logger.error(f"Error during detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
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
