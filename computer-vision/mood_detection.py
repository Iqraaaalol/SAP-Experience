import cv2
import torch
from facenet_pytorch import MTCNN
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import os
import time
import numpy as np
import argparse

# Try to import config locally
try:
    import config
except ImportError:
    class Config:
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        INPUT_SIZE = (224, 224)
        NUM_CLASSES = 7
        CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    config = Config()

# Import seat manager from separate module
from seat_manager import SeatManager, SeatState, SeatCalibrator, CALIBRATION_FILE, create_seat_manager_from_calibration


def run_calibration(video_source, width, height, num_seats):
    """Run interactive seat calibration."""
    seat_names = getattr(config, 'SEAT_NAMES', None)
    if seat_names and len(seat_names) < num_seats:
        seat_names = None
    
    calibrator = SeatCalibrator(
        frame_width=width,
        frame_height=height,
        num_seats=num_seats,
        seat_names=seat_names[:num_seats] if seat_names else None
    )
    
    video = int(video_source) if video_source.isdigit() else video_source
    result = calibrator.run(video)
    
    if result:
        create_seat_manager_from_calibration(result, width, height, save_to_file=CALIBRATION_FILE)
        print(f"Calibration saved to {CALIBRATION_FILE}")
        return True
    return False


class ConvNeXtEmotionDetector:
    def __init__(self, model_path="checkpoints/affectnet_best_convnext_base.pth", num_classes=None, class_names=None):
        self.device = config.DEVICE
        num_classes = num_classes or getattr(config, 'NUM_CLASSES', 7)
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
        self.emotion_groups = {
            'angry': 'distressed',
            'disgust': 'distressed',
            'fear': 'distressed',
            'sad': 'distressed',
            'happy': 'happy',
            'neutral': 'neutral',
            'surprise': 'surprise'
        }
        
        # Define colors for grouped emotions
        self.grouped_colors = {
            'distressed': (0, 0, 255),    # Red for distress
            'happy': (0, 255, 0),         # Green
            'neutral': (255, 255, 0),     # Cyan
            'surprise': (255, 0, 255)     # Magenta
        }
        
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
        Detect emotion from face image with grouping.
        
        Args:
            face_image: BGR face image (numpy array)
            landmarks: Optional 5 facial landmarks for alignment
        
        Returns:
            (grouped_emotion, confidence, emotion_dict)
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
            
        # Get original emotion
        original_emotion = self.class_names[pred_idx.item()]
        confidence = conf.item()
        
        # Create original emotion probability dict (not grouped)
        emotion_dict = {}
        for i, emotion in enumerate(self.class_names):
            emotion_dict[emotion] = probs[0][i].item()
        
        return original_emotion, confidence, emotion_dict

class FaceDetector:
    def __init__(self, frame_width: int = 1280, frame_height: int = 720, use_seats: bool = True):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        self.mtcnn = MTCNN(
            keep_all=True,
            device=device,
            min_face_size=40,
            post_process=False  
        )
        
        self.emotion_detector = ConvNeXtEmotionDetector()
        
        # Initialize seat manager (optional)
        self.use_seats = use_seats
        self.seat_manager = SeatManager(frame_width, frame_height) if use_seats else None
            
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
    
    def draw_enhanced_boxes(self, frame, boxes, probs, landmarks, seat_assignments=None, emotions=None):
        """Draw face boxes with seat assignments.
        
        Args:
            frame: Video frame
            boxes: Face bounding boxes
            probs: Detection probabilities
            landmarks: Facial landmarks
            seat_assignments: Dict mapping seat_id -> (face_idx, box)
            emotions: Dict mapping seat_id -> (emotion, confidence)
        """
        if boxes is None:
            return frame
        
        # Create reverse mapping: face_idx -> seat_id
        face_to_seat = {}
        if seat_assignments:
            for seat_id, (face_idx, _) in seat_assignments.items():
                face_to_seat[face_idx] = seat_id
        
        for i, (box, prob) in enumerate(zip(boxes, probs)):
            x1, y1, x2, y2 = box.astype(int)
            
            # Get seat assignment for this face
            seat_id = face_to_seat.get(i)
            
            # Color by seat if assigned, otherwise by confidence
            if seat_id:
                color = self.seat_manager.seat_colors.get(seat_id, (0, 255, 0))
            elif prob > 0.95:
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
            
            # Show seat ID instead of "Face N" if assigned
            if seat_id:
                cv2.putText(frame, f"Seat {seat_id}", (x1, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                cv2.putText(frame, f"Unassigned", (x1, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
            
            cv2.putText(frame, f"Conf: {prob:.2%}", (x1, info_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(frame, f"Size: {face_width}x{face_height}", (x1, info_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(frame, f"Dist: {distance}", (x1, info_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw emotion if available for this seat
            emotion_data = None
            if seat_id and emotions and seat_id in emotions:
                emotion_data = emotions[seat_id]
            
            if emotion_data:
                emotion, conf = emotion_data
                # Use original emotion colors
                emotion_color = self.emotion_detector.emotion_colors.get(emotion, (255, 255, 255))
                cv2.putText(frame, f"Emotion: {emotion.upper()} ({conf:.0%})", 
                        (x1, info_y + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, emotion_color, 2)
            
            # Draw landmarks
            if landmarks is not None and i < len(landmarks):
                try:
                    landmark_points = landmarks[i]
                    if landmark_points is not None and len(landmark_points) > 0:
                        for point in landmark_points:
                            x, y = point.astype(int)
                            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
                except (TypeError, ValueError, IndexError):
                    pass
        
        return frame
    
    def add_dashboard(self, frame, boxes, probs):
        h, w = frame.shape[:2]
        
        dashboard_height = 120
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, dashboard_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        title = "SAP Seat-Based Emotion Detection" if self.use_seats else "SAP Face Detection"
        cv2.putText(frame, title, (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        num_faces = len(boxes) if boxes is not None else 0        
        cv2.putText(frame, f"Faces: {num_faces}", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show seat occupancy summary (only if using seats)
        if self.use_seats and self.seat_manager:
            seat_summary = self.seat_manager.get_seat_summary()
            occupied_count = sum(1 for s in seat_summary.values() if s['occupied'])
            total_seats = len(seat_summary)
            cv2.putText(frame, f"Seats: {occupied_count}/{total_seats}", (20, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show individual seat status
            x_offset = 200
            for seat_id, info in seat_summary.items():
                color = self.seat_manager.seat_colors.get(seat_id, (255, 255, 255))
                status = "●" if info['occupied'] else "○"
                emotion_str = f" ({info['emotion']})" if info['emotion'] else ""
                cv2.putText(frame, f"{seat_id}:{status}{emotion_str}", (x_offset, 85),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                x_offset += 150
        
        device = "GPU" if torch.cuda.is_available() else "CPU"
        cv2.putText(frame, f"Device: {device}", (w - 200, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(frame, "Press 'q' to quit | 's' to screenshot | 'r' to reset | 'c' to calibrate", (20, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame

def main():
    parser = argparse.ArgumentParser(description='Seat-Based Face Detection with Emotion Recognition')
    parser.add_argument('--video', type=str, default='0', help='Path to video file or camera index (default: 0 for camera)')
    parser.add_argument('--width', type=int, default=1280, help='Frame width')
    parser.add_argument('--height', type=int, default=720, help='Frame height')
    parser.add_argument('--calibrate', action='store_true', help='Run seat calibration before starting')
    parser.add_argument('--seats', type=int, default=4, help='Number of seats to calibrate')
    parser.add_argument('--no-seats', action='store_true', help='Disable seat-based tracking (just detect faces)')
    args = parser.parse_args()

    if args.video.isdigit():
        cap = cv2.VideoCapture(int(args.video))
    else:
        cap = cv2.VideoCapture(args.video)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    # Get actual frame dimensions
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Run calibration if requested or if no calibration file exists
    if args.calibrate and not args.no_seats:
        cap.release()  # Release for calibrator to use
        print("\nStarting seat calibration...")
        run_calibration(args.video, actual_width, actual_height, args.seats)
        # Reopen capture after calibration
        if args.video.isdigit():
            cap = cv2.VideoCapture(int(args.video))
        else:
            cap = cv2.VideoCapture(args.video)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    elif args.calibrate and args.no_seats:
        print("Warning: --calibrate ignored when --no-seats is enabled")
    
    # Initialize detector with actual frame size
    use_seats = not args.no_seats
    detector = FaceDetector(frame_width=actual_width, frame_height=actual_height, use_seats=use_seats)
    
    screenshot_count = 0
    emotion_update_interval = 0.3  
    last_emotion_update = time.time()
    seat_emotions = {}  # seat_id -> (emotion, confidence)
    
    print("=" * 60)
    if use_seats:
        print("SAP Seat-Based Face Detection with Emotion Recognition")
    else:
        print("SAP Face Detection with Emotion Recognition")
    print("=" * 60)
    print(f"Frame size: {actual_width}x{actual_height}")
    if use_seats and detector.seat_manager:
        print(f"Seat grid: {detector.seat_manager.rows}x{detector.seat_manager.cols}")
        print(f"Vacancy timeout: {detector.seat_manager.vacancy_timeout}s")
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save screenshot")
    if use_seats:
        print("  'r' - Reset all seats")
        print("  'c' - Calibrate seats")
    print("=" * 60)
    
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

        current_time = time.time()
        
        # Detect faces
        boxes, probs, landmarks = detector.detect_faces(frame)
        
        # Update seat assignments (if seats enabled)
        seat_assignments = None
        if use_seats and detector.seat_manager:
            seat_assignments = detector.seat_manager.update_seats(boxes, frame, current_time)
        
        # Update emotions for assigned seats (or all faces if no seats)
        if (current_time - last_emotion_update) > emotion_update_interval:
            if use_seats and seat_assignments:
                for seat_id, (face_idx, box) in seat_assignments.items():
                    x1, y1, x2, y2 = box.astype(int)
                    face_img = frame[max(0, y1):min(frame.shape[0], y2), 
                                    max(0, x1):min(frame.shape[1], x2)]
                    
                    # Get landmarks for this face (if available)
                    face_landmarks = None
                    if landmarks is not None and face_idx < len(landmarks) and landmarks[face_idx] is not None:
                        face_landmarks = landmarks[face_idx].copy()
                        face_landmarks[:, 0] -= x1
                        face_landmarks[:, 1] -= y1
                    
                    if face_img.size > 0:
                        emotion, conf, emotion_probs = detector.emotion_detector.detect_emotion(face_img, face_landmarks)
                        seat_emotions[seat_id] = (emotion, conf)
                        detector.seat_manager.update_seat_emotion(seat_id, emotion, conf, emotion_probs)
            else:
                # No seats mode - detect emotions for all faces
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = box.astype(int)
                        face_img = frame[max(0, y1):min(frame.shape[0], y2), 
                                        max(0, x1):min(frame.shape[1], x2)]
                        
                        face_landmarks = None
                        if landmarks is not None and i < len(landmarks) and landmarks[i] is not None:
                            face_landmarks = landmarks[i].copy()
                            face_landmarks[:, 0] -= x1
                            face_landmarks[:, 1] -= y1
                        
                        if face_img.size > 0:
                            emotion, conf, emotion_probs = detector.emotion_detector.detect_emotion(face_img, face_landmarks)
                            seat_emotions[i] = (emotion, conf)  # Use face index as key
            
            last_emotion_update = current_time
        
        # Clean up emotions for vacated seats (only in seat mode)
        if use_seats and detector.seat_manager:
            for seat_id in list(seat_emotions.keys()):
                if seat_id in detector.seat_manager.seats and not detector.seat_manager.seats[seat_id].is_occupied:
                    del seat_emotions[seat_id]
            
            # Draw seat zones first (background)
            frame = detector.seat_manager.draw_seat_zones(frame)
        
        # Draw face boxes with seat assignments (or just faces if no seats)
        if not use_seats and boxes is not None:
            # In no-seats mode, create a simple emotions dict indexed by face
            face_emotions = {i: seat_emotions.get(i) for i in range(len(boxes)) if i in seat_emotions}
            frame = detector.draw_enhanced_boxes(frame, boxes, probs, landmarks, None, face_emotions)
        else:
            frame = detector.draw_enhanced_boxes(frame, boxes, probs, landmarks, 
                                                 seat_assignments, seat_emotions)
        
        # Draw dashboard
        frame = detector.add_dashboard(frame, boxes, probs)
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        window_title = 'SAP Seat-Based Emotion Detection' if use_seats else 'SAP Face Detection'
        cv2.imshow(window_title, frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            screenshot_name = f"screenshot_{screenshot_count:03d}.jpg"
            cv2.imwrite(screenshot_name, frame)
            print(f"Screenshot saved: {screenshot_name}")
            screenshot_count += 1
        elif key == ord('r') and use_seats:
            # Reset all seats
            detector.seat_manager.reset_all_seats()
            seat_emotions.clear()
            print("All seats reset")
        elif key == ord('c') and use_seats:
            # Run calibration
            cap.release()
            cv2.destroyAllWindows()
            print("\nStarting seat calibration...")
            if run_calibration(args.video, actual_width, actual_height, len(detector.seat_manager.seats)):
                # Reload seat manager with new calibration
                detector.seat_manager = SeatManager(actual_width, actual_height)
            # Reopen capture
            if args.video.isdigit():
                cap = cv2.VideoCapture(int(args.video))
            else:
                cap = cv2.VideoCapture(args.video)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            seat_emotions.clear()
    
    cap.release()
    cv2.destroyAllWindows()
    print("Face detection stopped.")

if __name__ == "__main__":
    main()