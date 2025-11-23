import cv2
from ultralytics import YOLO
import time
from collections import defaultdict
from datetime import datetime
from config import Config
from database import MongoDB

class ObjectDetector:
    def __init__(self):
        self.config = Config()
        print(f"Loading YOLO model: {self.config.MODEL_NAME}")
        self.model = YOLO(self.config.MODEL_NAME)
        self.class_counts = defaultdict(int)
        self.frame_count = 0
        self.processed_frames = 0
        
        # Initialize MongoDB connection
        self.db = None
        if self.config.ENABLE_MONGODB:
            try:
                self.db = MongoDB()
                print(f"✓ Connected to MongoDB: {self.config.MONGO_DB_NAME}.{self.config.MONGO_COLLECTION_NAME}")
            except Exception as e:
                print(f"✗ MongoDB connection failed: {e}")
                print("  Continuing without database logging...")
        
    def process_video(self, source):
        """Process video from file or stream URL"""
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Error: Cannot open video source: {source}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n{'='*60}")
        print(f"Video Source: {source}")
        print(f"FPS: {fps:.2f}")
        if total_frames > 0:
            print(f"Total Frames: {total_frames}")
        print(f"Frame Skip: {self.config.FRAME_SKIP}")
        print(f"Confidence Threshold: {self.config.CONFIDENCE_THRESHOLD}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                # Skip frames based on config
                if self.frame_count % self.config.FRAME_SKIP != 0:
                    continue
                
                self.processed_frames += 1
                
                # Run YOLO detection
                results = self.model(frame, conf=self.config.CONFIDENCE_THRESHOLD, verbose=False)
                
                # Reset counts for this frame
                frame_counts = defaultdict(int)
                
                # Process detections
                detections_list = []
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        class_name = self.model.names[cls_id]
                        confidence = float(box.conf[0])
                        
                        # Filter by target classes if specified
                        if self.config.TARGET_CLASSES and class_name not in self.config.TARGET_CLASSES:
                            continue
                        
                        frame_counts[class_name] += 1
                        
                        # Collect detection data for MongoDB
                        x1, y1, x2, y2 = map(float, box.xyxy[0])
                        detection_data = {
                            'class_name': class_name,
                            'confidence': round(confidence, 4),
                            'bounding_box': {
                                'x1': round(x1, 2),
                                'y1': round(y1, 2),
                                'x2': round(x2, 2),
                                'y2': round(y2, 2)
                            }
                        }
                        detections_list.append(detection_data)
                        
                        # Draw bounding box if display is enabled
                        if self.config.DISPLAY_VIDEO:
                            x1_int, y1_int, x2_int, y2_int = map(int, [x1, y1, x2, y2])
                            cv2.rectangle(frame, (x1_int, y1_int), (x2_int, y2_int), (0, 255, 0), 2)
                            label = f"{class_name}: {confidence:.2f}"
                            cv2.putText(frame, label, (x1_int, y1_int-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Update total counts
                for cls_name, count in frame_counts.items():
                    self.class_counts[cls_name] = max(self.class_counts[cls_name], count)
                
                # Save to MongoDB if enabled and detections exist
                if self.db and detections_list:
                    self._save_to_mongodb(detections_list, frame_counts, source)
                
                # Display counts in terminal
                self._display_counts(frame_counts)
                
                # Display video window if enabled
                if self.config.DISPLAY_VIDEO:
                    # Add frame info overlay
                    cv2.putText(frame, f"Frame: {self.frame_count}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.imshow('YOLOv8 Detection', frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nStopping detection (user interrupted)")
                        break
                
        except KeyboardInterrupt:
            print("\n\nDetection interrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Display summary
            self._display_summary(start_time)
    
    def _save_to_mongodb(self, detections_list, frame_counts, source):
        """Save detection data to MongoDB"""
        try:
            document = {
                'timestamp': datetime.utcnow(),
                'video_source': source,
                'frame_number': self.frame_count,
                'processed_frame_number': self.processed_frames,
                'total_objects_detected': len(detections_list),
                'object_counts': dict(frame_counts),
                'detections': detections_list,
                'model_info': {
                    'model_name': self.config.MODEL_NAME,
                    'confidence_threshold': self.config.CONFIDENCE_THRESHOLD
                }
            }
            
            self.db.insert_detection(document)
            
        except Exception as e:
            print(f"\n✗ MongoDB insert error: {e}")
    
    def _display_counts(self, frame_counts):
        """Display object counts in terminal"""
        print(f"\rFrame {self.frame_count} (Processed: {self.processed_frames}) | ", end="")
        
        if frame_counts:
            count_str = " | ".join([f"{cls}: {count}" for cls, count in sorted(frame_counts.items())])
            print(count_str, end="", flush=True)
        else:
            print("No objects detected", end="", flush=True)
    
    def _display_summary(self, start_time):
        """Display detection summary"""
        elapsed = time.time() - start_time
        
        print(f"\n\n{'='*60}")
        print("DETECTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total Frames: {self.frame_count}")
        print(f"Processed Frames: {self.processed_frames}")
        print(f"Processing Time: {elapsed:.2f}s")
        print(f"Average FPS: {self.processed_frames/elapsed:.2f}")
        print(f"\nMaximum Object Counts Detected:")
        print(f"{'-'*60}")
        
        if self.class_counts:
            for cls_name, count in sorted(self.class_counts.items()):
                print(f"  {cls_name:20s}: {count}")
        else:
            print("  No objects detected")
        
        print(f"{'='*60}\n")


def main():
    config = Config()
    detector = ObjectDetector()
    
    print("\n" + "="*60)
    print("YOLOv8 Object Detection System")
    print("="*60)
    
    # Process video source
    detector.process_video(config.VIDEO_SOURCE)


if __name__ == "__main__":
    main()