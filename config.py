import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for YOLO object detection"""
    
    # YOLO Model Configuration
    MODEL_NAME = os.getenv('MODEL_NAME', 'yolov8m.pt')  # yolov8m = medium model
    
    # Video Source Configuration
    # Can be:
    # - Local video file path: 'videos/sample.mp4'
    # - RTSP stream: 'rtsp://username:password@ip:port/stream'
    # - HTTP stream: 'http://example.com/stream.m3u8'
    # - Webcam: 0, 1, 2, etc.
    VIDEO_SOURCE = os.getenv('VIDEO_SOURCE', 'sample.mp4')
    
    # Frame Processing Configuration
    FRAME_SKIP = int(os.getenv('FRAME_SKIP', '1'))  # Process every Nth frame (1 = all frames)
    
    # Detection Configuration
    CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))
    
    # Target Classes (leave empty list to detect all classes)
    # Common COCO classes: person, car, truck, bus, motorcycle, bicycle, etc.
    TARGET_CLASSES_STR = os.getenv('TARGET_CLASSES', 'person,car,truck,bus,motorcycle,bicycle')
    TARGET_CLASSES = [cls.strip() for cls in TARGET_CLASSES_STR.split(',') if cls.strip()] if TARGET_CLASSES_STR else []
    
    # Display Configuration
    DISPLAY_VIDEO = os.getenv('DISPLAY_VIDEO', 'False').lower() == 'true'
    
    # Output Configuration
    SAVE_OUTPUT = os.getenv('SAVE_OUTPUT', 'False').lower() == 'true'
    OUTPUT_PATH = os.getenv('OUTPUT_PATH', 'output/')

    # MongoDB Configuration
    ENABLE_MONGODB = os.getenv('ENABLE_MONGODB', 'False').lower() == 'true'
    MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
    MONGO_DB_NAME = os.getenv('MONGO_DB_NAME', 'yolo_detections')
    MONGO_COLLECTION_NAME = os.getenv('MONGO_COLLECTION_NAME', 'detections')
    
    @classmethod
    def display_config(cls):
        """Display current configuration"""
        print("\nCurrent Configuration:")
        print(f"  Model: {cls.MODEL_NAME}")
        print(f"  Video Source: {cls.VIDEO_SOURCE}")
        print(f"  Frame Skip: {cls.FRAME_SKIP}")
        print(f"  Confidence: {cls.CONFIDENCE_THRESHOLD}")
        print(f"  Target Classes: {cls.TARGET_CLASSES if cls.TARGET_CLASSES else 'All classes'}")
        print(f"  Display Video: {cls.DISPLAY_VIDEO}")
        print(f"  Save Output: {cls.SAVE_OUTPUT}")
        print()


# COCO Dataset Class Names (80 classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]