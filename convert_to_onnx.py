"""
Convert YOLOv8 model to ONNX format for web deployment
"""

from ultralytics import YOLO
import os

# Load the YOLOv8n model
model = YOLO('yolov8n.pt')

# Export to ONNX format
# opset=11 is more compatible with ONNX Runtime Web
# simplify=True simplifies the model for better web performance
model.export(
    format='onnx',
    dynamic=False,
    simplify=True,
    opset=11,
    imgsz=640
)

print("Model exported to ONNX format successfully!")
print("Output file: yolov8n.onnx")
