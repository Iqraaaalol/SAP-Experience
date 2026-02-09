"""
Seat Manager Module with Interactive Calibration

This module provides seat zone management with:
- Interactive click-to-calibrate zones (4 corners per seat)
- JSON persistence for calibration data
- Face embedding-based re-identification
- Vacancy timeout and seat locking

Usage:
    # Run standalone to calibrate:
    python seat_manager.py --calibrate --video 0
    
    # Import in mood_detection.py:
    from seat_manager import SeatManager, SeatState
"""

import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import numpy as np
import json
import os
import argparse
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

# Try to import config locally
try:
    import config
except ImportError:
    class Config:
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        SEAT_GRID_ROWS = 2
        SEAT_GRID_COLS = 2
        SEAT_VACANCY_TIMEOUT = 5.0
        SEAT_EMBEDDING_THRESHOLD = 0.7
        SEAT_NAMES = ["1A", "1B", "2A", "2B"]
    config = Config()

# Default calibration file path
CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), "seat_calibration.json")


@dataclass
class SeatState:
    """Represents the state of a single seat."""
    seat_id: str
    zone: Tuple[int, int, int, int]  # Bounding box (x1, y1, x2, y2)
    polygon: Optional[List[Tuple[int, int]]] = None  # 4 corner points for non-rectangular zones
    is_occupied: bool = False
    last_seen_time: float = 0.0
    locked_embedding: Optional[np.ndarray] = None
    current_emotion: Optional[str] = None
    current_confidence: float = 0.0
    emotion_history: List[str] = field(default_factory=list)
    emotion_probability_history: List[Dict[str, float]] = field(default_factory=list)  # Store probability distributions for smoothing


class SeatManager:
    """
    Manages seat zones and face-to-seat assignments.
    Supports both rectangular grid zones and custom polygon zones via calibration.
    """
    
    def __init__(self, frame_width: int, frame_height: int, 
                 rows: int = None, cols: int = None,
                 vacancy_timeout: float = None,
                 embedding_threshold: float = None,
                 calibration_file: str = None,
                 auto_load_calibration: bool = True):
        """
        Initialize seat manager.
        
        Args:
            frame_width: Width of video frame
            frame_height: Height of video frame  
            rows: Number of seat rows (for auto-grid)
            cols: Number of seat columns (for auto-grid)
            vacancy_timeout: Seconds before empty seat can be reassigned
            embedding_threshold: Cosine similarity threshold for face matching
            calibration_file: Path to calibration JSON file
            auto_load_calibration: Whether to auto-load calibration if file exists
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.rows = rows or getattr(config, 'SEAT_GRID_ROWS', 2)
        self.cols = cols or getattr(config, 'SEAT_GRID_COLS', 2)
        self.vacancy_timeout = vacancy_timeout or getattr(config, 'SEAT_VACANCY_TIMEOUT', 3.0)
        self.embedding_threshold = embedding_threshold or getattr(config, 'SEAT_EMBEDDING_THRESHOLD', 0.7)
        self.seat_names = getattr(config, 'SEAT_NAMES', None)
        self.calibration_file = calibration_file or CALIBRATION_FILE
        
        # Initialize face embedding model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Loading face embedding model (InceptionResnetV1)...")
        self.embedding_model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        # Preprocessing for embedding model
        self.embedding_transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Seat storage
        self.seats: Dict[str, SeatState] = {}
        
        # Seat colors for visualization
        self.seat_colors = {
            "1A": (255, 100, 100),   # Light blue
            "1B": (100, 255, 100),   # Light green  
            "2A": (100, 100, 255),   # Light red
            "2B": (255, 255, 100),   # Light cyan
        }
        
        # Try to load calibration, fall back to auto-grid
        calibration_loaded = False
        if auto_load_calibration and os.path.exists(self.calibration_file):
            calibration_loaded = self.load_calibration()
        
        if not calibration_loaded:
            self._create_grid_zones()
        
        print(f"Seat Manager initialized: {len(self.seats)} seats, {self.vacancy_timeout}s timeout")
    
    def _create_grid_zones(self):
        """Create default rectangular grid zones."""
        dashboard_height = 120
        usable_height = self.frame_height - dashboard_height
        
        cell_width = self.frame_width // self.cols
        cell_height = usable_height // self.rows
        
        seat_idx = 0
        for row in range(self.rows):
            for col in range(self.cols):
                x1 = col * cell_width
                y1 = dashboard_height + row * cell_height
                x2 = x1 + cell_width
                y2 = y1 + cell_height
                
                if self.seat_names and seat_idx < len(self.seat_names):
                    seat_id = self.seat_names[seat_idx]
                else:
                    seat_id = f"{row+1}{chr(65+col)}"
                
                # Create polygon from rectangle corners
                polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                
                self.seats[seat_id] = SeatState(
                    seat_id=seat_id,
                    zone=(x1, y1, x2, y2),
                    polygon=polygon
                )
                seat_idx += 1
        
        print(f"Created {len(self.seats)} auto-grid seat zones: {list(self.seats.keys())}")
    
    def save_calibration(self, filepath: str = None):
        """Save current seat zones to JSON file."""
        filepath = filepath or self.calibration_file
        
        data = {
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "seats": {}
        }
        
        for seat_id, seat in self.seats.items():
            data["seats"][seat_id] = {
                "zone": list(seat.zone),
                "polygon": seat.polygon
            }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Calibration saved to {filepath}")
        return True
    
    def load_calibration(self, filepath: str = None) -> bool:
        """Load seat zones from JSON file."""
        filepath = filepath or self.calibration_file
        
        if not os.path.exists(filepath):
            print(f"Calibration file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Validate frame dimensions match (warn if different)
            if data.get("frame_width") != self.frame_width or data.get("frame_height") != self.frame_height:
                print(f"Warning: Calibration was done at {data.get('frame_width')}x{data.get('frame_height')}, "
                      f"current frame is {self.frame_width}x{self.frame_height}")
                # Scale the zones
                scale_x = self.frame_width / data.get("frame_width", self.frame_width)
                scale_y = self.frame_height / data.get("frame_height", self.frame_height)
            else:
                scale_x = scale_y = 1.0
            
            self.seats.clear()
            
            for seat_id, seat_data in data.get("seats", {}).items():
                zone = seat_data.get("zone", [0, 0, 100, 100])
                polygon = seat_data.get("polygon")
                
                # Scale if needed
                if scale_x != 1.0 or scale_y != 1.0:
                    zone = [
                        int(zone[0] * scale_x),
                        int(zone[1] * scale_y),
                        int(zone[2] * scale_x),
                        int(zone[3] * scale_y)
                    ]
                    if polygon:
                        polygon = [(int(p[0] * scale_x), int(p[1] * scale_y)) for p in polygon]
                
                self.seats[seat_id] = SeatState(
                    seat_id=seat_id,
                    zone=tuple(zone),
                    polygon=polygon
                )
            
            print(f"Loaded calibration from {filepath}: {list(self.seats.keys())}")
            return True
            
        except Exception as e:
            print(f"Error loading calibration: {e}")
            return False
    
    def get_face_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Generate 512-dim face embedding."""
        if face_image is None or face_image.size == 0:
            return None
        
        try:
            rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_face)
            img_tensor = self.embedding_transform(pil_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.embedding_model(img_tensor)
                embedding = F.normalize(embedding, p=2, dim=1)
                
            return embedding.cpu().numpy().flatten()
        except Exception as e:
            print(f"Embedding error: {e}")
            return None
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        if emb1 is None or emb2 is None:
            return 0.0
        return float(np.dot(emb1, emb2))
    
    def get_face_centroid(self, box: np.ndarray) -> Tuple[float, float]:
        """Get centroid of face bounding box."""
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        return (cx, cy)
    
    def get_zone_center(self, seat: SeatState) -> Tuple[float, float]:
        """Get center point of a seat zone."""
        if seat.polygon:
            xs = [p[0] for p in seat.polygon]
            ys = [p[1] for p in seat.polygon]
            return (sum(xs) / len(xs), sum(ys) / len(ys))
        else:
            x1, y1, x2, y2 = seat.zone
            return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def point_in_polygon(self, point: Tuple[float, float], polygon: List[Tuple[int, int]]) -> bool:
        """Check if point is inside polygon using ray casting algorithm."""
        x, y = point
        n = len(polygon)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        
        return inside
    
    def point_in_zone(self, point: Tuple[float, float], seat: SeatState) -> bool:
        """Check if point is inside seat zone (polygon or rectangle)."""
        if seat.polygon:
            return self.point_in_polygon(point, seat.polygon)
        else:
            x, y = point
            x1, y1, x2, y2 = seat.zone
            return x1 <= x <= x2 and y1 <= y <= y2
    
    def distance_to_zone_center(self, point: Tuple[float, float], seat: SeatState) -> float:
        """Calculate distance from point to zone center."""
        center = self.get_zone_center(seat)
        return np.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2)
    
    def find_seat_for_face(self, box: np.ndarray) -> Optional[str]:
        """Find which seat zone a face belongs to based on centroid."""
        centroid = self.get_face_centroid(box)
        
        best_seat = None
        min_distance = float('inf')
        
        for seat_id, seat in self.seats.items():
            if self.point_in_zone(centroid, seat):
                dist = self.distance_to_zone_center(centroid, seat)
                if dist < min_distance:
                    min_distance = dist
                    best_seat = seat_id
        
        return best_seat
    
    def update_seats(self, boxes: np.ndarray, frame: np.ndarray, 
                     current_time: float) -> Dict[str, Tuple[int, np.ndarray]]:
        """Update seat assignments based on detected faces."""
        assignments = {}
        
        if boxes is None or len(boxes) == 0:
            for seat_id, seat in self.seats.items():
                if seat.is_occupied:
                    if current_time - seat.last_seen_time > self.vacancy_timeout:
                        self._vacate_seat(seat_id)
            return assignments
        
        face_to_potential_seats = {}
        
        for i, box in enumerate(boxes):
            potential_seat = self.find_seat_for_face(box)
            if potential_seat:
                centroid = self.get_face_centroid(box)
                dist = self.distance_to_zone_center(centroid, self.seats[potential_seat])
                face_to_potential_seats[i] = (potential_seat, dist, box)
        
        assigned_faces = set()
        assigned_seats = set()
        
        # First pass: Match by embedding
        for i, box in enumerate(boxes):
            if i in assigned_faces:
                continue
                
            x1, y1, x2, y2 = box.astype(int)
            face_img = frame[max(0, y1):min(frame.shape[0], y2), 
                            max(0, x1):min(frame.shape[1], x2)]
            
            if face_img.size == 0:
                continue
                
            face_embedding = self.get_face_embedding(face_img)
            
            for seat_id, seat in self.seats.items():
                if seat_id in assigned_seats:
                    continue
                if not seat.is_occupied or seat.locked_embedding is None:
                    continue
                    
                similarity = self.compute_similarity(face_embedding, seat.locked_embedding)
                
                if similarity >= self.embedding_threshold:
                    assignments[seat_id] = (i, box)
                    seat.last_seen_time = current_time
                    if face_embedding is not None:
                        seat.locked_embedding = 0.9 * seat.locked_embedding + 0.1 * face_embedding
                        seat.locked_embedding /= np.linalg.norm(seat.locked_embedding)
                    assigned_faces.add(i)
                    assigned_seats.add(seat_id)
                    break
        
        # Second pass: Assign by centroid
        for i, (potential_seat, dist, box) in face_to_potential_seats.items():
            if i in assigned_faces or potential_seat in assigned_seats:
                continue
                
            seat = self.seats[potential_seat]
            
            seat_available = (
                not seat.is_occupied or 
                (current_time - seat.last_seen_time > self.vacancy_timeout)
            )
            
            if seat_available:
                x1, y1, x2, y2 = box.astype(int)
                face_img = frame[max(0, y1):min(frame.shape[0], y2),
                                max(0, x1):min(frame.shape[1], x2)]
                
                face_embedding = self.get_face_embedding(face_img) if face_img.size > 0 else None
                
                seat.is_occupied = True
                seat.last_seen_time = current_time
                seat.locked_embedding = face_embedding
                seat.emotion_history = []
                
                assignments[potential_seat] = (i, box)
                assigned_faces.add(i)
                assigned_seats.add(potential_seat)
        
        # Check for timeouts
        for seat_id, seat in self.seats.items():
            if seat_id not in assigned_seats and seat.is_occupied:
                if current_time - seat.last_seen_time > self.vacancy_timeout:
                    self._vacate_seat(seat_id)
        
        return assignments
    
    def _vacate_seat(self, seat_id: str):
        """Mark a seat as vacant."""
        seat = self.seats[seat_id]
        seat.is_occupied = False
        seat.locked_embedding = None
        seat.current_emotion = None
        seat.current_confidence = 0.0
        seat.emotion_history = []
        print(f"Seat {seat_id} vacated")
    
    def update_seat_emotion(self, seat_id: str, emotion: str, confidence: float, emotion_probs: Optional[Dict[str, float]] = None):
        """Update emotion state for a seat with temporal smoothing.
        
        Args:
            seat_id: Seat identifier
            emotion: Detected emotion (may be ignored if using smoothing)
            confidence: Confidence of detection
            emotion_probs: Dictionary of emotion probabilities for smoothing
        """
        if seat_id in self.seats:
            seat = self.seats[seat_id]
            seat.current_confidence = confidence
            
            # Store probability distribution if provided
            if emotion_probs:
                seat.emotion_probability_history.append(emotion_probs)
                if len(seat.emotion_probability_history) > 10:
                    seat.emotion_probability_history = seat.emotion_probability_history[-10:]
                
                # Calculate smoothed emotion from probability history
                smoothed_emotion = self._get_smoothed_emotion(seat)
                seat.current_emotion = smoothed_emotion
                seat.emotion_history.append(smoothed_emotion)
            else:
                # Fallback to direct emotion if no probabilities provided
                seat.current_emotion = emotion
                seat.emotion_history.append(emotion)
            
            # Keep history at max 10 entries
            if len(seat.emotion_history) > 10:
                seat.emotion_history = seat.emotion_history[-10:]
    
    def _get_smoothed_emotion(self, seat: SeatState) -> str:
        """Get emotion by averaging probability distributions over time.
        
        This reduces flickering by considering recent probability distributions
        rather than just the max prediction at each timestep.
        
        Args:
            seat: SeatState with probability history
            
        Returns:
            Smoothed emotion label
        """
        if not seat.emotion_probability_history:
            return seat.current_emotion or 'neutral'
        
        # Average probabilities across recent history
        avg_probs = {}
        for prob_dict in seat.emotion_probability_history:
            for emotion, prob in prob_dict.items():
                avg_probs[emotion] = avg_probs.get(emotion, 0.0) + prob
        
        # Normalize by number of samples
        num_samples = len(seat.emotion_probability_history)
        for emotion in avg_probs:
            avg_probs[emotion] /= num_samples
        
        # Return emotion with highest average probability
        if avg_probs:
            return max(avg_probs, key=avg_probs.get)
        return 'neutral'
    
    def draw_seat_zones(self, frame: np.ndarray, show_labels: bool = True) -> np.ndarray:
        """Draw seat zone boundaries on frame."""
        for seat_id, seat in self.seats.items():
            color = self.seat_colors.get(seat_id, (128, 128, 128))
            thickness = 2 if seat.is_occupied else 1
            
            if seat.polygon:
                pts = np.array(seat.polygon, np.int32).reshape((-1, 1, 2))
                
                # Semi-transparent fill for occupied seats
                if seat.is_occupied:
                    overlay = frame.copy()
                    cv2.fillPoly(overlay, [pts], color)
                    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
                
                # Draw polygon border
                cv2.polylines(frame, [pts], True, color, thickness)
                
                # Label position at top-left of polygon
                label_pos = (min(p[0] for p in seat.polygon) + 5, 
                            min(p[1] for p in seat.polygon) + 25)
            else:
                x1, y1, x2, y2 = seat.zone
                
                if seat.is_occupied:
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                label_pos = (x1 + 5, y1 + 25)
            
            if show_labels:
                label = f"Seat {seat_id}"
                status = "OCCUPIED" if seat.is_occupied else "EMPTY"
                cv2.putText(frame, label, label_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame, status, (label_pos[0], label_pos[1] + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return frame
    
    def get_seat_summary(self) -> Dict[str, dict]:
        """Get summary of all seat states."""
        summary = {}
        for seat_id, seat in self.seats.items():
            summary[seat_id] = {
                'occupied': seat.is_occupied,
                'emotion': seat.current_emotion,
                'confidence': seat.current_confidence,
            }
        return summary
    
    def reset_all_seats(self):
        """Reset all seats to vacant state."""
        for seat_id in self.seats:
            self._vacate_seat(seat_id)


class SeatCalibrator:
    """
    Interactive calibration tool for defining seat zones.
    Click 4 corners per seat in order: top-left, top-right, bottom-right, bottom-left.
    """
    
    def __init__(self, frame_width: int, frame_height: int, 
                 num_seats: int = 4, seat_names: List[str] = None):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.num_seats = num_seats
        self.seat_names = seat_names or getattr(config, 'SEAT_NAMES', None) or \
                          [f"{i//2 + 1}{chr(65 + i%2)}" for i in range(num_seats)]
        
        self.current_seat_idx = 0
        self.current_points: List[Tuple[int, int]] = []
        self.calibrated_seats: Dict[str, List[Tuple[int, int]]] = {}
        
        self.colors = [
            (255, 100, 100),  # Light blue
            (100, 255, 100),  # Light green
            (100, 100, 255),  # Light red
            (255, 255, 100),  # Light cyan
        ]
        
        self.window_name = "Seat Calibration"
        self.frame = None
        self.original_frame = None
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for calibration."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_seat_idx >= self.num_seats:
                return
            
            self.current_points.append((x, y))
            print(f"Point {len(self.current_points)}/4 for {self.seat_names[self.current_seat_idx]}: ({x}, {y})")
            
            if len(self.current_points) == 4:
                seat_id = self.seat_names[self.current_seat_idx]
                self.calibrated_seats[seat_id] = self.current_points.copy()
                print(f"Seat {seat_id} calibrated!")
                
                self.current_points = []
                self.current_seat_idx += 1
                
                if self.current_seat_idx >= self.num_seats:
                    print("\nAll seats calibrated! Press 's' to save, 'r' to restart, 'q' to quit.")
            
            self._update_display()
    
    def _update_display(self):
        """Update the calibration display."""
        if self.original_frame is None:
            return
        
        self.frame = self.original_frame.copy()
        
        # Draw already calibrated seats
        for i, (seat_id, points) in enumerate(self.calibrated_seats.items()):
            color = self.colors[i % len(self.colors)]
            pts = np.array(points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(self.frame, [pts], True, color, 2)
            cv2.putText(self.frame, seat_id, (points[0][0] + 5, points[0][1] + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw corner numbers
            for j, pt in enumerate(points):
                cv2.circle(self.frame, pt, 5, color, -1)
                cv2.putText(self.frame, str(j+1), (pt[0] + 8, pt[1] - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw current points being added
        if self.current_points:
            color = self.colors[self.current_seat_idx % len(self.colors)]
            for i, pt in enumerate(self.current_points):
                cv2.circle(self.frame, pt, 8, color, -1)
                cv2.putText(self.frame, str(i+1), (pt[0] + 10, pt[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw lines between points
            if len(self.current_points) > 1:
                for i in range(len(self.current_points) - 1):
                    cv2.line(self.frame, self.current_points[i], self.current_points[i+1], color, 2)
        
        # Instructions
        if self.current_seat_idx < self.num_seats:
            seat_id = self.seat_names[self.current_seat_idx]
            point_num = len(self.current_points) + 1
            corner_names = ["top-left", "top-right", "bottom-right", "bottom-left"]
            instruction = f"Click {corner_names[len(self.current_points)]} corner for Seat {seat_id} ({point_num}/4)"
        else:
            instruction = "Calibration complete! Press 's' to save, 'r' to restart, 'q' to quit"
        
        # Draw instruction bar
        cv2.rectangle(self.frame, (0, 0), (self.frame_width, 50), (0, 0, 0), -1)
        cv2.putText(self.frame, instruction, (10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow(self.window_name, self.frame)
    
    def run(self, video_source=0) -> Optional[Dict[str, List[Tuple[int, int]]]]:
        """
        Run the interactive calibration.
        
        Args:
            video_source: Camera index or video file path
            
        Returns:
            Dict of seat_id -> polygon points, or None if cancelled
        """
        if isinstance(video_source, str) and not video_source.isdigit():
            cap = cv2.VideoCapture(video_source)
        else:
            cap = cv2.VideoCapture(int(video_source) if isinstance(video_source, str) else video_source)
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("\n" + "=" * 60)
        print("SEAT CALIBRATION MODE")
        print("=" * 60)
        print(f"Calibrating {self.num_seats} seats: {self.seat_names}")
        print("\nInstructions:")
        print("  - Click 4 corners per seat: top-left, top-right, bottom-right, bottom-left")
        print("  - Press 's' to save calibration")
        print("  - Press 'r' to restart calibration")
        print("  - Press 'u' to undo last point")
        print("  - Press 'q' to quit without saving")
        print("=" * 60 + "\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            self.original_frame = frame.copy()
            self._update_display()
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Calibration cancelled")
                cap.release()
                cv2.destroyAllWindows()
                return None
            
            elif key == ord('s'):
                if len(self.calibrated_seats) > 0:
                    cap.release()
                    cv2.destroyAllWindows()
                    return self.calibrated_seats
                else:
                    print("No seats calibrated yet!")
            
            elif key == ord('r'):
                print("Restarting calibration...")
                self.current_seat_idx = 0
                self.current_points = []
                self.calibrated_seats = {}
            
            elif key == ord('u'):
                if self.current_points:
                    removed = self.current_points.pop()
                    print(f"Undid point: {removed}")
                elif self.calibrated_seats:
                    # Undo last seat
                    last_seat = list(self.calibrated_seats.keys())[-1]
                    self.current_points = self.calibrated_seats.pop(last_seat)
                    self.current_seat_idx -= 1
                    print(f"Undid seat {last_seat}, re-editing...")
                self._update_display()
        
        cap.release()
        cv2.destroyAllWindows()
        return None


def create_seat_manager_from_calibration(calibration: Dict[str, List[Tuple[int, int]]], 
                                          frame_width: int, frame_height: int,
                                          save_to_file: str = None) -> SeatManager:
    """
    Create a SeatManager from calibration data.
    
    Args:
        calibration: Dict of seat_id -> polygon points
        frame_width: Frame width
        frame_height: Frame height
        save_to_file: Optional path to save calibration JSON
        
    Returns:
        Configured SeatManager instance
    """
    manager = SeatManager(frame_width, frame_height, auto_load_calibration=False)
    manager.seats.clear()
    
    for seat_id, polygon in calibration.items():
        # Calculate bounding box from polygon
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        zone = (min(xs), min(ys), max(xs), max(ys))
        
        manager.seats[seat_id] = SeatState(
            seat_id=seat_id,
            zone=zone,
            polygon=polygon
        )
    
    if save_to_file:
        manager.calibration_file = save_to_file
        manager.save_calibration()
    
    return manager


def main():
    """Run standalone calibration or test mode."""
    parser = argparse.ArgumentParser(description='Seat Manager Calibration Tool')
    parser.add_argument('--calibrate', action='store_true', help='Run interactive calibration')
    parser.add_argument('--video', type=str, default='0', help='Video source (camera index or file path)')
    parser.add_argument('--width', type=int, default=1920, help='Frame width')
    parser.add_argument('--height', type=int, default=1080, help='Frame height')
    parser.add_argument('--seats', type=int, default=4, help='Number of seats to calibrate')
    parser.add_argument('--output', type=str, default=None, help='Output calibration file path')
    args = parser.parse_args()
    
    if args.calibrate:
        # Get seat names from config
        seat_names = getattr(config, 'SEAT_NAMES', None)
        if seat_names and len(seat_names) < args.seats:
            seat_names = None  # Fall back to auto-generated names
        
        calibrator = SeatCalibrator(
            frame_width=args.width,
            frame_height=args.height,
            num_seats=args.seats,
            seat_names=seat_names[:args.seats] if seat_names else None
        )
        
        video_source = int(args.video) if args.video.isdigit() else args.video
        result = calibrator.run(video_source)
        
        if result:
            output_path = args.output or CALIBRATION_FILE
            manager = create_seat_manager_from_calibration(
                result, args.width, args.height, save_to_file=output_path
            )
            print(f"\nCalibration complete! Saved to: {output_path}")
            print(f"Seats configured: {list(manager.seats.keys())}")
    else:
        # Test mode - just show current calibration
        print("Seat Manager Test Mode")
        print(f"Looking for calibration file: {CALIBRATION_FILE}")
        
        manager = SeatManager(args.width, args.height)
        print(f"\nLoaded {len(manager.seats)} seats:")
        for seat_id, seat in manager.seats.items():
            print(f"  {seat_id}: zone={seat.zone}, polygon={seat.polygon is not None}")


if __name__ == "__main__":
    main()
