"""
Sleep detection via Eye Aspect Ratio (EAR) using MediaPipe FaceMesh.

When a face crop is provided, FaceMesh extracts 468 landmarks,
from which the 6 left-eye and 6 right-eye contour points are used
to compute EAR. Sustained low EAR (eyes closed) over a configurable
duration triggers a 'sleeping' classification.
"""

import numpy as np
import os
import urllib.request

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

# Model download URL
FACE_LANDMARKER_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
FACE_LANDMARKER_MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

# Try to import config for thresholds
try:
    import config
    EAR_THRESHOLD = getattr(config, 'EAR_THRESHOLD', 0.25)
    SLEEP_DURATION = getattr(config, 'SLEEP_DURATION', 3.0)
except ImportError:
    EAR_THRESHOLD = 0.25
    SLEEP_DURATION = 3.0

# MediaPipe FaceMesh landmark indices for eye contours
# Left eye (from the subject's perspective)
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
# Right eye (from the subject's perspective)
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]


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

        # Per-seat state: seat_id -> timestamp when eyes first closed
        self._eyes_closed_since = {}

        # Initialize MediaPipe FaceMesh
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
            
            # MediaPipe 0.10+ uses tasks API
            base_options = mp_python.BaseOptions(model_asset_path=FACE_LANDMARKER_MODEL_PATH)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                num_faces=1,
                min_face_detection_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False
            )
            self._face_mesh = vision.FaceLandmarker.create_from_options(options)

    @property
    def available(self):
        """Whether sleep detection is available (mediapipe installed)."""
        return self._face_mesh is not None

    def compute_ear(self, face_crop_bgr):
        """
        Compute the average Eye Aspect Ratio from a BGR face crop.

        Args:
            face_crop_bgr: BGR image (numpy array) of a cropped face.

        Returns:
            Average EAR (float), or None if FaceMesh fails to detect a face.
        """
        if not self.available:
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
