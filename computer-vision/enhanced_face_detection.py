import cv2
import torch
from facenet_pytorch import MTCNN
from deepface import DeepFace
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import os
import time
import numpy as np
import tempfile
import argparse

# Try to import config locally
try:
    import config
except ImportError:
    # If config not found (e.g. run from different dir), define defaults or rely on fallback
    class Config:
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        INPUT_SIZE = (300, 300)
    config = Config()

class EmotionDetector:
    def __init__(self):
        self.emotion_colors = {
            'happy': (0, 255, 0),      # Green
            'neutral': (255, 255, 0),  # Cyan
            'angry': (0, 0, 255),      # Red
            'stressed': (0, 165, 255)  # Orange
        }
    
    def detect_emotion(self, face_image):
        try:
            # Save face temporarily for mood analysis
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                cv2.imwrite(tmp.name, face_image)
                tmp_path = tmp.name
            
            result = DeepFace.analyze(
                img_path=tmp_path,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv'
            )
            
            if isinstance(result, list):
                result = result[0]
            
            emotions = result['emotion']
            
            # Map to 3 classes
            three_class = {
                'happy': emotions.get('happy', 0),
                'neutral': emotions.get('neutral', 0),
                'stressed': emotions.get('fear', 0) + emotions.get('sad', 0) + emotions.get('disgust', 0) + emotions.get('angry', 0)
            }
            
            dominant = max(three_class, key=three_class.get)
            confidence = three_class[dominant] / 100.0
            
            return dominant, confidence, three_class
            
        except Exception as e:
            print(f"Emotion detection error: {e}")
            return 'neutral', 0.0, {}
        finally:
            # Clean up temp file
            try:
                import os
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except:
                pass

class EfficientNetEmotionDetector:
    def __init__(self, model_path="checkpoints/best_efficientnet_b3.pth", num_classes=3, class_names=None):
        self.device = config.DEVICE
        self.emotion_colors = {
            'happy': (0, 255, 0),      # Green
            'neutral': (255, 255, 0),  # Cyan
            'stressed': (0, 165, 255)  # Orange
        }
        
        self.class_names = class_names if class_names else ['happy', 'neutral', 'stressed']
        self.model = self._load_model(model_path, num_classes)
        
        self.transform = transforms.Compose([
            transforms.Resize(config.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def _load_model(self, model_path, num_classes):
        print(f"Loading EfficientNet-B3 from {model_path}...")
        try:
            from torchvision.models import EfficientNet_B3_Weights
            model = models.efficientnet_b3(weights=None)
        except ImportError:
            model = models.efficientnet_b3(pretrained=False)
            
        # Re-create classifier head
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            # handle if 'model_state_dict' is key or direct state dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                if 'class_names' in checkpoint:
                    self.class_names = checkpoint['class_names'] 
            else:
                model.load_state_dict(checkpoint)
            print("Custom model loaded successfully.")
        else:
            print(f"Warning: Model checkpoint {model_path} not found. Using random weights.")
            
        model.to(self.device)
        model.eval()
        return model

    def detect_emotion(self, face_image):
        if face_image is None or face_image.size == 0:
             return 'neutral', 0.0, {}
             
        # Convert BGR (OpenCV) to RGB (PIL/Torch)
        rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_face)
        
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, 1)
            
        dominant_emotion = self.class_names[pred_idx.item()]
        confidence = conf.item()
        
        # Format similar to DeepFace return for compatibility
        emotion_dict = {name: probs[0][i].item() * 100 for i, name in enumerate(self.class_names)}
        
        return dominant_emotion, confidence, emotion_dict

class EnhancedFaceDetector:
    def __init__(self, model_type='deepface'):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        self.mtcnn = MTCNN(
            keep_all=True,
            device=device,
            min_face_size=40,
            post_process=False  
        )
        
        if model_type == 'efficientnet':
            self.emotion_detector = EfficientNetEmotionDetector()
        else:
            self.emotion_detector = EmotionDetector()
            
        self.fps_history = []
        self.face_history = {} 
        self.emotion_cache = {}  
        self.last_emotion_time = {}
    
    def detect_faces(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        boxes, probs, landmarks = self.mtcnn.detect(rgb_frame, landmarks=True)
        
        return boxes, probs, landmarks
    
    def estimate_distance(self, box):
        if box is None:
            return None
        
        face_width = box[2] - box[0]
        
        if face_width > 500:
            return "Very Close"
        elif face_width > 400:
            return "Close"
        elif face_width > 300:
            return "Medium"
        else:
            return "Far"
    
    def draw_enhanced_boxes(self, frame, boxes, probs, landmarks, emotions=None):
        if boxes is None:
            return frame
        
        for i, (box, prob) in enumerate(zip(boxes, probs)):
            x1, y1, x2, y2 = box.astype(int)
            
            # Color by confidence
            if prob > 0.95:
                color = (0, 255, 0)
            elif prob > 0.85:
                color = (0, 255, 255)
            else:
                color = (0, 165, 255)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Calculate face size
            face_width = x2 - x1
            face_height = y2 - y1
            
            distance = self.estimate_distance(box)
            
            info_y = y1 - 70 if y1 > 100 else y2 + 20
            cv2.putText(frame, f"Face {i+1}", (x1, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"Conf: {prob:.2%}", (x1, info_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(frame, f"Size: {face_width}x{face_height}", (x1, info_y + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(frame, f"Dist: {distance}", (x1, info_y + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw emotion if available
            if emotions and i in emotions:
                emotion, conf = emotions[i]
                emotion_color = self.emotion_detector.emotion_colors.get(emotion, (255, 255, 255))
                cv2.putText(frame, f"Emotion: {emotion.upper()} ({conf:.0%})", 
                           (x1, info_y + 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, emotion_color, 2)
            
            if landmarks is not None and i < len(landmarks):
                try:
                    landmark_points = landmarks[i]
                    if landmark_points is not None and len(landmark_points) > 0:
                        for point in landmark_points:
                            x, y = point.astype(int)
                            # Validate coordinates are within frame bounds
                            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
                except (TypeError, ValueError, IndexError):
                    # Skip if invald data
                    pass
        
        return frame
    
    def add_dashboard(self, frame, boxes, probs):
        h, w = frame.shape[:2]
        
        dashboard_height = 120
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, dashboard_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        cv2.putText(frame, "SAP Face Detection Demo", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        num_faces = len(boxes) if boxes is not None else 0        
        cv2.putText(frame, f"Faces: {num_faces}", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        device = "GPU" if torch.cuda.is_available() else "CPU"
        cv2.putText(frame, f"Device: {device}", (w - 200, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(frame, "Press 'q' to quit | 's' to screenshot", (20, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame

def main():
    parser = argparse.ArgumentParser(description='Enhanced Face Detection with Emotion Recognition')
    parser.add_argument('--video', type=str, default='0', help='Path to video file or camera index (default: 0 for camera)')
    parser.add_argument('--model', type=str, default='deepface', choices=['deepface', 'efficientnet'], help='Emotion model to use')
    args = parser.parse_args()
    
    detector = EnhancedFaceDetector(model_type=args.model)

    if args.video.isdigit():
        cap = cv2.VideoCapture(int(args.video))
    else:
        cap = cv2.VideoCapture(args.video)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    screenshot_count = 0
    emotion_update_interval = 0.5  
    last_emotion_update = time.time()
    
    print("=" * 50)
    print("Enhanced Face Detection with Emotion Recognition")
    print("=" * 50)
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save screenshot")
    print("=" * 50)
    
    fps_start = time.time()
    fps_counter = 0
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        fps_counter += 1
        if time.time() - fps_start > 1.0:
            fps = fps_counter / (time.time() - fps_start)
            fps_counter = 0
            fps_start = time.time()

        boxes, probs, landmarks = detector.detect_faces(frame)
        
        emotions = {}
        current_time = time.time()
        if boxes is not None and (current_time - last_emotion_update) > emotion_update_interval:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.astype(int)
                # Extract face region
                face_img = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                
                if face_img.size > 0:
                    emotion, conf, _ = detector.emotion_detector.detect_emotion(face_img)
                    emotions[i] = (emotion, conf)
            
            last_emotion_update = current_time
        
        frame = detector.draw_enhanced_boxes(frame, boxes, probs, landmarks, emotions)
        
        frame = detector.add_dashboard(frame, boxes, probs)
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 200, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('SAP Enhanced Face Detection with Emotion', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            screenshot_name = f"screenshot_{screenshot_count:03d}.jpg"
            cv2.imwrite(screenshot_name, frame)
            print(f"Screenshot saved: {screenshot_name}")
            screenshot_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print("Face detection stopped.")

if __name__ == "__main__":
    main()