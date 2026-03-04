import cv2
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import os
import time
import json
import numpy as np
import argparse
from collections import deque

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

# Import sleep detector
from sleep_detector import SleepDetector
from attention import CoordinateAttention

# MQTT publisher for sending seat emotions to Arduino
try:
    import paho.mqtt.client as mqtt
    import paho.mqtt
    MQTT_AVAILABLE = True
    print(f"[MQTT] paho-mqtt version: {paho.mqtt.__version__}")
except ImportError:
    MQTT_AVAILABLE = False
    print("Warning: paho-mqtt not installed. MQTT publishing disabled.")
    print("Install with: pip install paho-mqtt")


_MQTT_RC_CODES = {
    0: "Connection accepted",
    1: "Refused — unacceptable protocol version",
    2: "Refused — identifier rejected",
    3: "Refused — server unavailable",
    4: "Refused — bad username or password",
    5: "Refused — not authorised",
}


def _mqtt_socket_probe(host: str, port: int, timeout: float = 3.0) -> str:
    """Try a raw TCP connect and return a human-readable status string."""
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout) as s:
            local = s.getsockname()
            return f"OK — TCP socket open from {local[0]}:{local[1]} → {host}:{port}"
    except socket.timeout:
        return f"TIMEOUT — no response from {host}:{port} within {timeout}s (firewall drop?)"
    except ConnectionRefusedError as e:
        return (
            f"REFUSED — {e}\n"
            f"  → Mosquitto is likely bound to 'localhost' only.\n"
            f"  → Edit mosquitto.conf and add:\n"
            f"        listener 1883 0.0.0.0\n"
            f"        allow_anonymous true\n"
            f"  → Then restart the Mosquitto service."
        )
    except OSError as e:
        return f"OS ERROR — {e}"


class MQTTPublisher:
    """Publishes seat emotion data to an MQTT broker (e.g. Mosquitto)."""

    def __init__(self, broker_host=None, broker_port=None, topic=None,
                 client_id=None, publish_interval=None):
        self.broker_host = broker_host or getattr(config, 'MQTT_BROKER_HOST', '10.110.0.13')
        self.broker_port = broker_port or getattr(config, 'MQTT_BROKER_PORT', 1883)
        self.base_topic = topic or getattr(config, 'MQTT_TOPIC', 'sap/seats/emotion')
        self.client_id = client_id or getattr(config, 'MQTT_CLIENT_ID', 'sap-emotion-detector')
        self.publish_interval = publish_interval or getattr(config, 'MQTT_PUBLISH_INTERVAL', 1.0)
        self.last_publish_time = 0.0
        self.connected = False          # kept for logging; use client.is_connected() for gating
        self._last_reconnect_attempt = 0.0
        self._reconnect_cooldown = 5.0   # seconds between reconnect attempts

        if not MQTT_AVAILABLE:
            print("[MQTT] Publisher disabled (paho-mqtt not installed)")
            return

        print(f"[MQTT] ── Diagnostics ──────────────────────────────────────")
        print(f"[MQTT] Broker target : {self.broker_host}:{self.broker_port}")
        print(f"[MQTT] Client ID     : {self.client_id}")
        print(f"[MQTT] Base topic    : {self.base_topic}")
        print(f"[MQTT] TCP probe     : {_mqtt_socket_probe(self.broker_host, self.broker_port)}")
        print(f"[MQTT] ────────────────────────────────────────────────────")

        self.client = mqtt.Client(client_id=self.client_id,
                                  callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_log = self._on_log  # verbose paho-level logging

        try:
            self.client.connect(self.broker_host, self.broker_port, keepalive=60)
            self.client.loop_start()  # Non-blocking background network loop
            print(f"[MQTT] Async connect initiated → {self.broker_host}:{self.broker_port}")
        except Exception as e:
            print(f"[MQTT] connect() raised: {type(e).__name__}: {e}")
            self.connected = False

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        rc_int = rc.value if hasattr(rc, 'value') else (int(rc) if not isinstance(rc, int) else rc)
        meaning = _MQTT_RC_CODES.get(rc_int, f"unknown rc={rc}")
        if rc_int == 0:
            self.connected = True
            print(f"[MQTT] ✓ Connected to {self.broker_host}:{self.broker_port} — {meaning}")
        else:
            print(f"[MQTT] ✗ Connection failed (rc={rc_int}): {meaning}")

    def _on_disconnect(self, client, userdata, flags, rc, properties=None):
        self.connected = False
        rc_int = rc.value if hasattr(rc, 'value') else (int(rc) if not isinstance(rc, int) else rc)
        meaning = _MQTT_RC_CODES.get(rc_int, f"rc={rc}")
        if rc_int != 0:
            print(f"[MQTT] Unexpected disconnect — {meaning}. Will auto-reconnect.")
        else:
            print(f"[MQTT] Disconnected cleanly.")

    def _on_log(self, client, userdata, level, buf):
        # Only print MQTT_LOG_ERR and MQTT_LOG_WARNING to avoid noise
        if level in (mqtt.MQTT_LOG_ERR, mqtt.MQTT_LOG_WARNING):
            label = "ERR " if level == mqtt.MQTT_LOG_ERR else "WARN"
            print(f"[MQTT][{label}] {buf}")

    def _is_connected(self) -> bool:
        """Use paho's own internal state — never stale."""
        try:
            return self.client.is_connected()
        except Exception:
            return False

    def _try_reconnect(self, current_time: float):
        """Attempt a reconnect if the cooldown has elapsed."""
        if (current_time - self._last_reconnect_attempt) < self._reconnect_cooldown:
            return
        self._last_reconnect_attempt = current_time
        probe = _mqtt_socket_probe(self.broker_host, self.broker_port)
        print(f"[MQTT] Reconnecting… TCP probe: {probe}")
        try:
            self.client.reconnect()
            print(f"[MQTT] reconnect() called — waiting for CONNACK")
        except Exception as e:
            print(f"[MQTT] reconnect() failed: {type(e).__name__}: {e}")

    def publish_emotions(self, seat_emotions: dict, current_time: float):
        """
        Publish current seat emotions if enough time has elapsed.

        Args:
            seat_emotions: Dict[seat_id, (emotion, confidence)]
            current_time: Current timestamp
        """
        if not MQTT_AVAILABLE:
            return
        if not self._is_connected():
            self._try_reconnect(current_time)
            return
        if (current_time - self.last_publish_time) < self.publish_interval:
            return

        for seat_id, (emotion, confidence) in seat_emotions.items():
            payload = json.dumps({
                "seat": seat_id,
                "emotion": emotion,
                "confidence": round(confidence, 3)
            })
            topic = f"{self.base_topic}/{seat_id}"
            self.client.publish(topic, payload, qos=0)

        # Also publish a combined summary on the base topic
        summary = {seat_id: {"emotion": emo, "confidence": round(conf, 3)}
                   for seat_id, (emo, conf) in seat_emotions.items()}
        self.client.publish(self.base_topic, json.dumps(summary), qos=0)

        self.last_publish_time = current_time

    def stop(self):
        """Cleanly disconnect from the broker."""
        if MQTT_AVAILABLE and hasattr(self, 'client'):
            self.client.loop_stop()
            self.client.disconnect()
            print("[MQTT] Disconnected.")


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
    def __init__(self, model_path=None, num_classes=None, class_names=None):
        # Use AffectNet fine-tuned model by default if available, otherwise fall back to FER
        if model_path is None:
            affectnet_path = os.path.join("checkpoints", getattr(config, 'AFFECTNET_MODEL_PATH', 'affectnet_best_convnext_base.pth'))
            fer_path = os.path.join("checkpoints", config.BEST_MODEL_PATH)
            model_path = affectnet_path if os.path.exists(affectnet_path) else fer_path
        
        self.device = config.DEVICE
        num_classes = num_classes or getattr(config, 'NUM_CLASSES', 7)
        sleep_color = getattr(config, 'SLEEP_EMOTION_COLOR', (128, 128, 128))
        base_colors = {
            'angry': (0, 0, 255),       # Red
            'disgust': (0, 128, 0),     # Dark green
            'fear': (128, 0, 128),      # Purple
            'happy': (0, 255, 0),       # Green
            'neutral': (255, 255, 0),   # Cyan
            'sad': (255, 0, 0),         # Blue
            'surprise': (255, 0, 255),  # Magenta
            'sleeping': sleep_color     # Gray
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
            'surprise': 'surprise',
            'sleeping': 'sleeping'
        }
        
        # Define colors for grouped emotions
        self.grouped_colors = {
            'distressed': (0, 0, 255),    # Red for distress
            'happy': (0, 255, 0),         # Green
            'neutral': (255, 255, 0),     # Cyan
            'surprise': (255, 0, 255),    # Magenta
            'sleeping': sleep_color       # Gray
        }
        
    def align_face(self, image, landmarks):
        """
        Align face using eye landmarks via affine transformation.
        
        Args:
            image: BGR image (numpy array)
            landmarks: 5 facial landmarks [left_eye_center, right_eye_center,
                       nose_tip, left_mouth_corner, right_mouth_corner]
                       as returned by SleepDetector.detect_faces_in_frame
                       (viewer's perspective, pixel coordinates relative to the crop).
        
        Returns:
            Aligned face image (numpy array)
        """
        if landmarks is None or len(landmarks) < 2:
            return image
            
        # Extract eye coordinates (viewer's perspective: [0]=image-left, [1]=image-right)
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
            
        # Re-create classifier head. If checkpoint used CoordinateAttention, we'll insert it below
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)

            # If checkpoint indicates CoordinateAttention was used, insert it into classifier
            ck_used_ca = False
            if isinstance(checkpoint, dict):
                ck_used_ca = checkpoint.get('used_coord_attn', False)

            if ck_used_ca:
                try:
                    print("Checkpoint was trained with CoordinateAttention — adding CA to classifier...")
                    layer_norm = model.classifier[0]
                    flatten = model.classifier[1]
                    model.classifier = nn.Sequential(
                        layer_norm,
                        CoordinateAttention(channels=1024, reduction=32),
                        flatten,
                        nn.Linear(1024, num_classes)
                    )
                    print("✓ CoordinateAttention added to classifier")
                except Exception as e:
                    print(f"Warning: failed to add CoordinateAttention to classifier: {e}")

            # handle if 'model_state_dict' is key or direct state dict
            state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
            try:
                model.load_state_dict(state_dict)
                if isinstance(checkpoint, dict) and 'class_names' in checkpoint:
                    self.class_names = checkpoint['class_names'] 
                print("Custom model loaded successfully.")
            except RuntimeError as e:
                print(f"Warning: strict model load failed: {e}")
                try:
                    model.load_state_dict(state_dict, strict=False)
                    print("Loaded model with strict=False (partial match).")
                    if isinstance(checkpoint, dict) and 'class_names' in checkpoint:
                        self.class_names = checkpoint['class_names'] 
                except Exception as e2:
                    print(f"Error: failed to load model state dict: {e2}")
                    raise
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
        
        self.emotion_detector = ConvNeXtEmotionDetector()

        # detect_faces_in_frame uses MediaPipe FaceLandmarker (468 landmarks)
        self.sleep_detector = SleepDetector()
        if self.sleep_detector.available:
            print("Face detection + sleep detection enabled (MediaPipe FaceLandmarker)")
        else:
            print("MediaPipe unavailable — face detection and sleep detection disabled")
        
        # Seat-based tracking may be disabled; store flag and init only when enabled
        self.use_seats = use_seats
        if self.use_seats:
            self.seat_manager = SeatManager(frame_width, frame_height)
        else:
            self.seat_manager = None
            
        self.fps_history = []
        self.face_history = {} 
        self.emotion_cache = {}  
        self.last_emotion_time = {}
    
    def detect_faces(self, frame):
        # MediaPipe FaceLandmarker handles BGR->RGB conversion internally.
        # Returns (boxes, probs, landmarks) in the same shape as MTCNN did:
        #   boxes:     ndarray (N, 4)    [x1, y1, x2, y2]
        #   probs:     ndarray (N,)      confidence (1.0 placeholder)
        #   landmarks: ndarray (N, 5, 2) [left_eye, right_eye, nose,
        #                                 left_mouth, right_mouth] viewer-perspective
        boxes, probs, landmarks = self.sleep_detector.detect_faces_in_frame(frame)
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
    parser.add_argument('--no-mqtt', action='store_true', help='Disable MQTT publishing')
    parser.add_argument('--mqtt-host', type=str, default=None, help='MQTT broker hostname/IP (default: localhost)')
    parser.add_argument('--mqtt-port', type=int, default=None, help='MQTT broker port (default: 1883)')
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
    
    # Initialize MQTT publisher
    mqtt_enabled = (not args.no_mqtt) and getattr(config, 'MQTT_ENABLED', True)
    mqtt_publisher = MQTTPublisher(
        broker_host=args.mqtt_host,
        broker_port=args.mqtt_port,
    ) if mqtt_enabled else None
    
    screenshot_count = 0
    emotion_update_interval = 0.3  
    last_emotion_update = time.time()
    seat_emotions = {}  # seat_id -> (emotion, confidence)
    
    # Temporal smoothing setup
    smoothing_window = 5  # number of recent predictions to average
    emotion_history = {}  # seat_id/face_idx -> deque of prob vectors
    
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
        
        # Update emotions for assigned seats (only when seats are enabled)
        if (current_time - last_emotion_update) > emotion_update_interval:
            if seat_assignments and detector.use_seats and detector.seat_manager:
                for seat_id, (face_idx, box) in seat_assignments.items():
                    x1, y1, x2, y2 = box.astype(int)
                    face_img = frame[max(0, y1):min(frame.shape[0], y2), 
                                    max(0, x1):min(frame.shape[1], x2)]
                    
                    if face_img.size > 0:
                        # Check for sleeping via EAR before running ConvNeXt
                        if detector.sleep_detector.available:
                            ear = detector.sleep_detector.compute_ear(face_img)
                            is_sleeping = detector.sleep_detector.update(seat_id, ear, current_time)
                        else:
                            is_sleeping = False
                        
                        if is_sleeping:
                            # Override emotion with sleeping — skip ConvNeXt inference
                            seat_emotions[seat_id] = ("sleeping", 1.0)
                            detector.seat_manager.update_seat_emotion(
                                seat_id, "sleeping", 1.0, {"sleeping": 1.0})
                        else:
                            # Normal ConvNeXt emotion detection
                            face_landmarks = None
                            if landmarks is not None and face_idx < len(landmarks) and landmarks[face_idx] is not None:
                                face_landmarks = landmarks[face_idx].copy()
                                face_landmarks[:, 0] -= x1
                                face_landmarks[:, 1] -= y1
                            
                            emotion, conf, emotion_probs = detector.emotion_detector.detect_emotion(face_img, face_landmarks)
                            seat_emotions[seat_id] = (emotion, conf)
                            detector.seat_manager.update_seat_emotion(seat_id, emotion, conf, emotion_probs)

            # (If seats are disabled we currently skip per-face emotion updates here.)
            last_emotion_update = current_time
        
        # Publish seat emotions via MQTT
        if mqtt_publisher and seat_emotions:
            mqtt_publisher.publish_emotions(seat_emotions, current_time)
        
        # Clean up emotions and sleep state for vacated seats (only when seat manager exists)
        if detector.use_seats and detector.seat_manager:
            for seat_id in list(seat_emotions.keys()):
                if not detector.seat_manager.seats[seat_id].is_occupied:
                    del seat_emotions[seat_id]
                    detector.sleep_detector.reset(seat_id)
        
        # Draw seat zones first (background) when seat manager is available
        if detector.use_seats and detector.seat_manager:
            frame = detector.seat_manager.draw_seat_zones(frame)
        
        # Draw face boxes with seat assignments
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
            detector.sleep_detector.reset_all()
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
    if mqtt_publisher:
        mqtt_publisher.stop()
    print("Face detection stopped.")

if __name__ == "__main__":
    main()