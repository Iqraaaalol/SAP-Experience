import cv2
import torch
from facenet_pytorch import MTCNN
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

class ConvNeXtEmotionDetector:
    def __init__(self, model_path="checkpoints/best_convnext_base.pth", num_classes=config.NUM_CLASSES, class_names=None):
        self.device = config.DEVICE
        base_colors = {
            'angry': (0, 0, 255),       # Red
            'disgust': (0, 128, 0),     # Dark green
            'fear': (128, 0, 128),      # Purple
            'happy': (0, 255, 0),       # Green
            'neutral': (255, 255, 0),   # Cyan
            'sad': (255, 0, 0),         # Blue
            'surprise': (255, 0, 255)   # Magenta
        }
        self.emotion_colors = base_colors
        
        # Default fallback if checkpoint doesn't have names
        self.class_names = class_names if class_names else config.CLASS_NAMES
        self.model = self._load_model(model_path, num_classes)
        
        self.transform = transforms.Compose([
            transforms.Resize(config.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Alignment settings
        self.output_size = config.INPUT_SIZE  # (224, 224)
        self.desired_left_eye = (0.35, 0.35)  # Left eye position in normalized coords
        
    def align_face(self, image, landmarks):
        """
        Align face using eye landmarks via affine transformation.
        
        Args:
            image: BGR image (numpy array)
            landmarks: 5 facial landmarks [left_eye, right_eye, nose, left_mouth, right_mouth]
        
        Returns:
            Aligned face image (numpy array)
        """
        if landmarks is None or len(landmarks) < 2:
            return image
            
        # Extract eye coordinates (MTCNN order: left_eye, right_eye, nose, left_mouth, right_mouth)
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        
        # Calculate angle between eyes
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))
        
        # Calculate the desired right eye position based on left eye
        desired_right_eye_x = 1.0 - self.desired_left_eye[0]
        
        # Calculate distance between eyes
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        
        # Calculate desired distance between eyes in output image
        desired_dist = (desired_right_eye_x - self.desired_left_eye[0]) * self.output_size[0]
        
        # Calculate scale factor
        scale = desired_dist / dist if dist > 0 else 1.0
        
        # Calculate center point between eyes
        eyes_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
        
        # Update translation component of matrix
        tX = self.output_size[0] * 0.5
        tY = self.output_size[1] * self.desired_left_eye[1]
        M[0, 2] += (tX - eyes_center[0])
        M[1, 2] += (tY - eyes_center[1])
        
        # Apply affine transformation
        aligned = cv2.warpAffine(image, M, self.output_size, flags=cv2.INTER_CUBIC)
        
        return aligned
        
    def _load_model(self, model_path, num_classes):
        print(f"Loading ConvNeXt-Base from {model_path}...")
        try:
            from torchvision.models import ConvNeXt_Base_Weights
            model = models.convnext_base(weights=None)
        except ImportError:
            model = models.convnext_base(pretrained=False)
            
        # Re-create classifier head
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
        
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

    def detect_emotion(self, face_image, landmarks=None):
        """
        Detect emotion from face image.
        
        Args:
            face_image: BGR face image (numpy array)
            landmarks: Optional 5 facial landmarks for alignment
        
        Returns:
            (dominant_emotion, confidence, emotion_dict)
        """
        if face_image is None or face_image.size == 0:
             return 'neutral', 0.0, {}
        
        # Apply face alignment if landmarks provided
        if landmarks is not None:
            face_image = self.align_face(face_image, landmarks)
             
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

class FaceDetector:
    def __init__(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        self.mtcnn = MTCNN(
            keep_all=True,
            device=device,
            min_face_size=40,
            post_process=False  
        )
        
        self.emotion_detector = ConvNeXtEmotionDetector()
            
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
            emotion_data = None
            if emotions:
                if isinstance(emotions, dict) and i in emotions:
                    emotion_data = emotions[i]
                elif isinstance(emotions, list) and i < len(emotions):
                    emotion_data = emotions[i]
            
            if emotion_data:
                emotion, conf = emotion_data
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
    args = parser.parse_args()
    
    detector = FaceDetector()

    if args.video.isdigit():
        cap = cv2.VideoCapture(int(args.video))
    else:
        cap = cv2.VideoCapture(args.video)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    screenshot_count = 0
    emotion_update_interval = 0.3  
    last_emotion_update = time.time()
    cached_emotions = {}  # Persist emotions between frames
    
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
        
        # Helper to create stable face IDs based on position
        def get_face_id(box):
            cx = int((box[0] + box[2]) / 2 / 50) * 50
            cy = int((box[1] + box[3]) / 2 / 50) * 50
            return f"{cx}_{cy}"
        
        emotions_list = []
        current_time = time.time()
        current_face_ids = set()
        
        if boxes is not None and len(boxes) > 0:
            # Update emotions if interval passed
            if (current_time - last_emotion_update) > emotion_update_interval:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.astype(int)
                    # Extract face region
                    face_img = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                    
                    # Get landmarks for this face (if available)
                    face_landmarks = None
                    if landmarks is not None and i < len(landmarks) and landmarks[i] is not None:
                        # Adjust landmarks to be relative to face crop
                        face_landmarks = landmarks[i].copy()
                        face_landmarks[:, 0] -= x1  # Adjust x coordinates
                        face_landmarks[:, 1] -= y1  # Adjust y coordinates
                    
                    if face_img.size > 0:
                        emotion, conf, _ = detector.emotion_detector.detect_emotion(face_img, face_landmarks)
                        face_id = get_face_id(box)
                        cached_emotions[face_id] = (emotion, conf)
                        current_face_ids.add(face_id)
                
                last_emotion_update = current_time
            else:
                # Just track current face IDs without detecting
                for box in boxes:
                    current_face_ids.add(get_face_id(box))
            
            # Build emotions list from cache for each detected face
            for box in boxes:
                face_id = get_face_id(box)
                if face_id in cached_emotions:
                    emotions_list.append(cached_emotions[face_id])
                else:
                    emotions_list.append(None)
            
            # Clean up old faces no longer in frame
            cached_emotions = {k: v for k, v in cached_emotions.items() if k in current_face_ids}
        else:
            cached_emotions.clear()
        
        frame = detector.draw_enhanced_boxes(frame, boxes, probs, landmarks, emotions_list)
        
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