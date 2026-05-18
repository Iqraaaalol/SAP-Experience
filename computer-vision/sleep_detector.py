"""
Face detection via MTCNN with MediaPipe FaceMesh fallback for EAR-based sleep detection.

MTCNN provides the initial full-frame detector so distant/small faces are easier to pick up.
When a face crop is provided, FaceMesh extracts 468 landmarks, from which the 6 left-eye
and 6 right-eye contour points are used to compute EAR. Sustained low EAR (eyes closed)
over a configurable duration triggers a 'sleeping' classification.
"""

import numpy as np
import os
import urllib.request

import torch

try:
    from facenet_pytorch import MTCNN
    facenet_pytorch = True
except ImportError:
    MTCNN = None
    facenet_pytorch = None
    print("Warning: facenet-pytorch not installed. MTCNN face detection will be disabled.")
    print("Install with: pip install facenet-pytorch")

try:
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    mp = True
except ImportError:
    mp = None
    mp_python = None
    vision = None
    print("Warning: mediapipe not installed. Sleep detection will be disabled.")
    print("Install with: pip install mediapipe")

# FaceMesh landmark indices for sleep detection
FACE_LANDMARKER_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
FACE_LANDMARKER_MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

# Try to import config for thresholds
try:
    import config
    EAR_THRESHOLD = getattr(config, 'EAR_THRESHOLD', 0.25)
    SLEEP_DURATION = getattr(config, 'SLEEP_DURATION', 3.0)
    MTCNN_MIN_FACE_SIZE = int(getattr(config, 'MTCNN_MIN_FACE_SIZE', 16))
    MTCNN_THRESHOLDS = getattr(config, 'MTCNN_THRESHOLDS', [0.55, 0.65, 0.65])
    MTCNN_FACTOR = float(getattr(config, 'MTCNN_FACTOR', 0.709))
except ImportError:
    EAR_THRESHOLD = 0.25
    SLEEP_DURATION = 3.0
    MTCNN_MIN_FACE_SIZE = 16
    MTCNN_THRESHOLDS = [0.55, 0.65, 0.65]
    MTCNN_FACTOR = 0.709
    
# MediaPipe FaceMesh landmark indices for eye contours
# Left eye (from the subject's perspective)
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
# Right eye (from the subject's perspective)
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]


def _normalize_thresholds(thresholds):
    if isinstance(thresholds, (list, tuple)) and len(thresholds) == 3:
        return [max(0.0, min(1.0, float(value))) for value in thresholds]
    return [0.55, 0.65, 0.65]


def _eye_aspect_ratio(eye_points):
    """
    Compute the Eye Aspect Ratio (EAR) for a single eye.

    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

    where p1..p6 are the 6 eye contour points ordered as:
        p1: outer corner, p2: upper-outer, p3: upper-inner,
        p4: inner corner, p5: lower-inner, p6: lower-outer

    Args:
        eye_points: ndarray of shape (6, 2) with (x, y) coordinates.

    Returns:
        EAR value (float).
    """
    p1, p2, p3, p4, p5, p6 = eye_points

    # Vertical distances
    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)

    # Horizontal distance
    h = np.linalg.norm(p1 - p4)

    if h < 1e-6:
        return 0.0

    return (v1 + v2) / (2.0 * h)


class SleepDetector:
    """
    Detects sleeping state per-seat by tracking Eye Aspect Ratio over time.

    Usage:
        detector = SleepDetector()
        ear = detector.compute_ear(face_crop_bgr)
        is_sleeping = detector.update(seat_id, ear, current_time)
    """

    def __init__(self, ear_threshold=None, sleep_duration=None):
        self.ear_threshold = ear_threshold if ear_threshold is not None else EAR_THRESHOLD
        self.sleep_duration = sleep_duration if sleep_duration is not None else SLEEP_DURATION
        self.mtcnn_min_face_size = max(8, int(MTCNN_MIN_FACE_SIZE))
        self.mtcnn_thresholds = _normalize_thresholds(MTCNN_THRESHOLDS)
        self.mtcnn_factor = max(0.1, min(0.99, float(MTCNN_FACTOR)))

        # Per-seat state: seat_id -> timestamp when eyes first closed
        self._eyes_closed_since = {}
        self._last_full_face_landmarks = None

        # Initial face detector used by the live pipeline.
        self._mtcnn = self._create_mtcnn() if facenet_pytorch is not None else None

        # Initialize MediaPipe FaceMesh for EAR-based sleep detection on face crops.
        self._face_mesh = None
        if mp is not None and vision is not None:
            # Download model if not present
            if not os.path.exists(FACE_LANDMARKER_MODEL_PATH):
                print(f"Downloading face landmarker model to {FACE_LANDMARKER_MODEL_PATH}...")
                try:
                    urllib.request.urlretrieve(FACE_LANDMARKER_MODEL_URL, FACE_LANDMARKER_MODEL_PATH)
                    print("Model downloaded successfully.")
                except Exception as e:
                    print(f"Failed to download face landmarker model: {e}")
                    print("Sleep detection will be disabled.")
                    return
            
            self._face_mesh = self._create_landmarker(num_faces=1)

    def _create_mtcnn(self):
        """Create the MTCNN detector used for initial full-frame face detection."""
        if MTCNN is None:
            return None

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return MTCNN(
            image_size=160,
            margin=0,
            min_face_size=self.mtcnn_min_face_size,
            thresholds=self.mtcnn_thresholds,
            factor=self.mtcnn_factor,
            post_process=False,
            keep_all=True,
            select_largest=False,
            device=device,
        )

    def _create_landmarker(self, num_faces):
        """Create a FaceLandmarker with shared confidence thresholds."""
        base_options = mp_python.BaseOptions(model_asset_path=FACE_LANDMARKER_MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=num_faces,
            min_face_detection_confidence=self.min_face_detection_confidence,
            min_face_presence_confidence=self.min_face_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            result_callback=None,
        )
        return vision.FaceLandmarker.create_from_options(options)

    @property
    def available(self):
        """Whether the initial MTCNN face detector is available."""
        return self._mtcnn is not None

    @property
    def sleep_available(self):
        """Whether MediaPipe FaceMesh is available for crop-based EAR detection."""
        return self._face_mesh is not None

    def compute_ear(self, face_crop_bgr):
        """
        Compute the average Eye Aspect Ratio from a BGR face crop.

        Args:
            face_crop_bgr: BGR image (numpy array) of a cropped face.

        Returns:
            Average EAR (float), or None if FaceMesh fails to detect a face.
        """
        if not self.sleep_available:
            return None

        if face_crop_bgr is None or face_crop_bgr.size == 0:
            return None

        import cv2
        rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        
        # Convert to MediaPipe Image format
        from mediapipe import Image, ImageFormat
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        
        # Detect face landmarks
        results = self._face_mesh.detect(mp_image)

        if not results.face_landmarks:
            return None

        face_landmarks = results.face_landmarks[0]
        h, w = face_crop_bgr.shape[:2]

        def _get_points(indices):
            pts = []
            for idx in indices:
                lm = face_landmarks[idx]
                pts.append(np.array([lm.x * w, lm.y * h]))
            return np.array(pts)

        left_eye = _get_points(LEFT_EYE_INDICES)
        right_eye = _get_points(RIGHT_EYE_INDICES)

        left_ear = _eye_aspect_ratio(left_eye)
        right_ear = _eye_aspect_ratio(right_eye)

        return (left_ear + right_ear) / 2.0

    def compute_ear_from_landmarks(self, face_landmarks_mp):
        """Compute EAR from full-frame MediaPipe landmarks when available.

        MTCNN only provides 5-point facial landmarks, which are not enough for EAR, so this
        method returns None for that format and the caller should fall back to `compute_ear()`.
        """
        if face_landmarks_mp is None:
            return None

        max_required_index = max(max(LEFT_EYE_INDICES), max(RIGHT_EYE_INDICES))

        try:
            if isinstance(face_landmarks_mp, np.ndarray):
                if (
                    face_landmarks_mp.ndim != 2
                    or face_landmarks_mp.shape[0] <= max_required_index
                    or face_landmarks_mp.shape[1] < 2
                ):
                    return None

                def _get_points(indices):
                    return np.array([face_landmarks_mp[i, :2] for i in indices], dtype=np.float32)
            else:
                if len(face_landmarks_mp) <= max_required_index:
                    return None

                def _get_points(indices):
                    return np.array([
                        [face_landmarks_mp[i].x, face_landmarks_mp[i].y]
                        for i in indices
                    ], dtype=np.float32)

            left_eye = _get_points(LEFT_EYE_INDICES)
            right_eye = _get_points(RIGHT_EYE_INDICES)
        except (AttributeError, IndexError, TypeError, ValueError):
            return None

        left_ear = _eye_aspect_ratio(left_eye)
        right_ear = _eye_aspect_ratio(right_eye)

        return (left_ear + right_ear) / 2.0

    def update(self, seat_id, ear, timestamp):
        """
        Update per-seat sleep tracking state.

        Args:
            seat_id: Unique identifier for the seat/face (str).
            ear: Eye Aspect Ratio value, or None if detection failed.
            timestamp: Current time in seconds (float).

        Returns:
            True if the person in this seat is classified as sleeping.
        """
        if ear is None:
            # Could not compute EAR (no mesh detected) — don't change state
            # This avoids false resets when FaceMesh temporarily fails
            if seat_id in self._eyes_closed_since:
                elapsed = timestamp - self._eyes_closed_since[seat_id]
                return elapsed >= self.sleep_duration
            return False

        if ear < self.ear_threshold:
            # Eyes are closed
            if seat_id not in self._eyes_closed_since:
                self._eyes_closed_since[seat_id] = timestamp
            elapsed = timestamp - self._eyes_closed_since[seat_id]
            return elapsed >= self.sleep_duration
        else:
            # Eyes are open — reset tracker
            self._eyes_closed_since.pop(seat_id, None)
            return False

    def reset(self, seat_id):
        """Clear sleep tracking state for a seat (e.g., when vacated)."""
        self._eyes_closed_since.pop(seat_id, None)

    def reset_all(self):
        """Clear all tracking state."""
        self._eyes_closed_since.clear()

    @staticmethod
    def _compute_iou(box_a, box_b):
        """Compute Intersection over Union of two [x1, y1, x2, y2] boxes."""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - intersection

        if union < 1e-6:
            return 0.0
        return intersection / union

    def detect_faces_in_frame(self, frame_bgr, max_faces=None):
        """
        Detect all faces in a full frame using MTCNN.

        Returns output compatible with MTCNN's detect(..., landmarks=True):
            boxes     - ndarray (N, 4)  [x1, y1, x2, y2] in pixel coords
            probs     - ndarray (N,)    detection confidence
            landmarks - ndarray (N, 5, 2) ordered as:
                          [left_eye, right_eye, nose_tip, left_mouth, right_mouth]

        All three are None when no faces are detected or MTCNN is unavailable.
        """
        if not self.available or frame_bgr is None or frame_bgr.size == 0:
            self._last_full_face_landmarks = None
            return None, None, None

        import cv2

        if max_faces is None:
            max_faces = int(getattr(config, 'MTCNN_MAX_FACES', 4))
        max_faces = max(1, int(max_faces))

        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        boxes, probs, points = self._mtcnn.detect(rgb_frame, landmarks=True)

        if boxes is None or probs is None or points is None:
            self._last_full_face_landmarks = None
            return None, None, None

        boxes = np.asarray(boxes, dtype=np.float32)
        probs = np.asarray(probs, dtype=np.float32)
        points = np.asarray(points, dtype=np.float32)

        if boxes.ndim == 1:
            boxes = boxes.reshape(1, -1)
        if probs.ndim == 0:
            probs = probs.reshape(1)
        if points.ndim == 2:
            points = points.reshape(1, *points.shape)

        if boxes.shape[0] == 0:
            self._last_full_face_landmarks = None
            return None, None, None

        order = np.argsort(probs)[::-1]
        if order.size > max_faces:
            order = order[:max_faces]

        boxes = boxes[order]
        probs = probs[order]
        points = points[order]

        self._last_full_face_landmarks = points.copy()

        return boxes, probs, points

    def get_ear_state(self, seat_id):
        """
        Get current sleep tracking info for a seat (for debugging/display).

        Returns:
            Dict with 'eyes_closed_since' (float or None) and 'is_tracking' (bool).
        """
        closed_since = self._eyes_closed_since.get(seat_id)
        return {
            'eyes_closed_since': closed_since,
            'is_tracking': closed_since is not None,
        }

    def get_last_full_face_landmarks(self):
        """Return landmarks from the most recent face detection pass."""
        return self._last_full_face_landmarks
