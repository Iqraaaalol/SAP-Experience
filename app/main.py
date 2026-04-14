"""
Aircraft Travel Assistant - Main FastAPI Application

A multi-language travel assistant for aircraft passengers with:
- Destination information from ChromaDB knowledge base
- Multilingual support (EN, ES, FR, DE, HI, PT, TH)
- In-flight service request detection
- Crew dashboard with WebSocket alerts
- Conversation history per seat
"""
import asyncio
import json
import os
import secrets
import socket
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import cv2
import threading
import numpy as np
from pathlib import Path
import time

from fastapi import Depends, FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
import uvicorn

# Add app directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(__file__))

# Import from services package
from services import (
    # Config
    STATIC_DIR, MODEL_NAME, CHROMA_PERSIST_DIR,
    # Models
    QueryRequest, CrewAlertRequest, CrewLoginRequest,
    # Language
    get_language_name, get_service_message, translate_to_english,
    # Services
    crew_manager,
    build_context_from_chroma, init_chroma_manager, get_chroma_manager,
    init_llm,
    query_cache, conversation_history,
    mood_analytics,
    init_db, get_stats, log_query, verify_crew_credentials,
)


# Initialize services
Llama = init_llm()
chroma_manager = init_chroma_manager(CHROMA_PERSIST_DIR)

#============================================================================
# CV STREAM PROCESSOR
# =============================================================================

# Mood alert configuration
_NEGATIVE_EMOTIONS        = {'angry', 'disgust', 'fear', 'sad'}
_MOOD_ALERT_CONFIDENCE    = 0.80   # minimum confidence to fire an alert
_MOOD_ALERT_COOLDOWN      = 30.0   # seconds between alerts for the same seat

# Event loop captured when CV starts so the background thread can schedule coroutines
_app_event_loop = None

# Global variables for video stream
camera = None
current_frame = None
frame_lock = threading.Lock()

# =============================================================================
# LIGHTING OVERRIDE STATE
# =============================================================================

# In-memory override state — mirrors what has been sent to the Arduino
_lighting_override: dict = {"enabled": False, "scene": "NEUTRAL"}

_VALID_SCENES = {"HAPPY", "NEUTRAL", "STRESSED", "ANGRY", "SAD", "ANXIOUS", "SLEEP"}

# Crew auth configuration/state
CREW_SESSION_COOKIE = "crew_session_token"
CREW_SESSION_TTL_SECONDS = 8 * 60 * 60
_crew_sessions: dict = {}


def _utcnow() -> datetime:
    return datetime.utcnow()


def _cleanup_expired_crew_sessions() -> None:
    now = _utcnow()
    expired_tokens = [
        token for token, payload in _crew_sessions.items()
        if payload.get("expires_at") <= now
    ]
    for token in expired_tokens:
        _crew_sessions.pop(token, None)


def _create_crew_session(username: str) -> str:
    _cleanup_expired_crew_sessions()
    token = secrets.token_urlsafe(32)
    _crew_sessions[token] = {
        "username": username,
        "expires_at": _utcnow() + timedelta(seconds=CREW_SESSION_TTL_SECONDS),
    }
    return token


def _get_crew_user_from_cookie(request: Request) -> str | None:
    _cleanup_expired_crew_sessions()
    token = request.cookies.get(CREW_SESSION_COOKIE)
    if not token:
        return None
    session_payload = _crew_sessions.get(token)
    if not session_payload:
        return None
    return session_payload.get("username")


def _get_crew_user_from_websocket(websocket: WebSocket) -> str | None:
    _cleanup_expired_crew_sessions()
    token = websocket.cookies.get(CREW_SESSION_COOKIE)
    if not token:
        return None
    session_payload = _crew_sessions.get(token)
    if not session_payload:
        return None
    return session_payload.get("username")


async def require_crew_auth(request: Request) -> str:
    username = _get_crew_user_from_cookie(request)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Crew authentication required")
    return username


def _do_publish_lighting_override(enabled: bool, scene: str) -> None:
    """Blocking MQTT publish — runs in a background thread."""
    import socket as _socket
    try:
        import paho.mqtt.client as paho_mqtt
    except ImportError:
        print("[MQTT] paho-mqtt not installed — override not published to Arduino")
        return

    broker  = os.environ.get("MQTT_BROKER_HOST", "192.168.0.100")
    port    = int(os.environ.get("MQTT_BROKER_PORT", 1883))
    topic   = "sap/lighting/override"
    payload = json.dumps({"enabled": enabled, "scene": scene})

    try:
        client = paho_mqtt.Client(client_id="sap-crew-override")
        client.connect_timeout = 4          # seconds for TCP + CONNACK
        client.socket_timeout  = 4          # attribute checked by some versions
        client.connect(broker, port, keepalive=5)
        # Loop just long enough to get CONNACK then deliver the message
        client.loop_start()
        result = client.publish(topic, payload, qos=1)
        result.wait_for_publish(timeout=4)  # wait for PUBACK (QoS 1)
        client.loop_stop()
        client.disconnect()
        print(f"[MQTT] Lighting override published → {payload}")
    except Exception as exc:
        print(f"[MQTT] Lighting override publish failed: {exc}")


def _publish_lighting_override(enabled: bool, scene: str) -> None:
    """Fire-and-forget: publish override in a daemon thread so the API never blocks."""
    t = threading.Thread(
        target=_do_publish_lighting_override,
        args=(enabled, scene),
        daemon=True,
    )
    t.start()

class CVStreamProcessor:
    def __init__(self, camera_index=0, no_mqtt: bool | None = None):
        # Import CV modules
        sys.path.insert(0, str(Path(__file__).parent.parent / "computer-vision"))
        import config as cv_config
        from mood_detection import FaceDetector, MQTTPublisher
        from seat_manager import SeatManager

        self.detector = FaceDetector()
        # Reuse the same SleepDetector instance used by FaceDetector.
        self.sleep_detector = self.detector.sleep_detector
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Get actual dimensions
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create seat manager and try to load calibration
        self.seat_manager = SeatManager(
            self.frame_width, 
            self.frame_height,
            auto_load_calibration=False
        )
        
        calibration_path = Path(__file__).parent.parent / "computer-vision" / "seat_calibration.json"
        if calibration_path.exists():
            print(f"📍 Loading seat calibration from {calibration_path}")
            if self.seat_manager.load_calibration(str(calibration_path)):
                print(f"✅ Seat manager loaded with {len(self.seat_manager.seats)} seats")
            else:
                print(f"⚠️  Failed to load calibration, using default grid")
                self.seat_manager._create_grid_zones()
        else:
            print(f"⚠️  No calibration found at {calibration_path}, using default grid")
            self.seat_manager._create_grid_zones()
        
        self.emotion_update_interval = float(getattr(cv_config, 'EMOTION_UPDATE_INTERVAL', 0.6))
        self.seat_update_interval = float(getattr(cv_config, 'SEAT_UPDATE_INTERVAL', 0.2))
        self.sleep_check_interval = float(getattr(cv_config, 'SLEEP_CHECK_INTERVAL', 0.6))
        self.last_seat_update = 0.0
        self.last_emotion_update = time.time()
        self.cached_emotions = {}
        self.cached_seat_assignments = {}
        self._last_sleep_check = {}
        self._mood_cooldowns: dict = {}  # seat_id -> timestamp of last mood alert
        self._last_analytics_snapshot: float = 0.0

        # MQTT publisher — sends seat emotions to Arduino.
        # Priority: explicit route payload override > env flag > CV config.
        env_no_mqtt = os.environ.get("CV_NO_MQTT", "").strip().lower() in {"1", "true", "yes", "on"}
        config_mqtt_enabled = bool(getattr(cv_config, 'MQTT_ENABLED', True))
        if no_mqtt is None:
            self.no_mqtt = env_no_mqtt or (not config_mqtt_enabled)
        else:
            self.no_mqtt = bool(no_mqtt)

        self.mqtt_publisher = None if self.no_mqtt else MQTTPublisher()
        print(f"📡 CV MQTT: {'disabled' if self.no_mqtt else 'enabled'}")

        self.running = False
        self.thread = None
    
    def get_face_id(self, box):
        """Create stable face IDs based on position"""
        cx = int((box[0] + box[2]) / 2 / 50) * 50
        cy = int((box[1] + box[3]) / 2 / 50) * 50
        return f"{cx}_{cy}"
    
    def process_frame(self, frame):
        """Process a single frame with face and emotion detection"""
        boxes, probs, landmarks = self.detector.detect_faces(frame)
        
        emotions_list = []
        seat_assignments = {}
        seat_emotions = {}
        current_time = time.time()
        current_face_ids = set()
        
        if boxes is not None and len(boxes) > 0:
            cached_face_count = len({idx for idx, _ in self.cached_seat_assignments.values()})
            should_update_seats = (
                (current_time - self.last_seat_update) >= self.seat_update_interval
                or not self.cached_seat_assignments
                or cached_face_count != len(boxes)
            )

            if should_update_seats:
                seat_assignments = self.seat_manager.update_seats(boxes, frame, current_time)
                self.cached_seat_assignments = seat_assignments
                self.last_seat_update = current_time
            else:
                seat_assignments = self.cached_seat_assignments
            
            # Update emotions if interval passed
            if (current_time - self.last_emotion_update) > self.emotion_update_interval:
                face_records = []
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.astype(int)
                    face_img = frame[max(0, y1):min(frame.shape[0], y2), 
                                   max(0, x1):min(frame.shape[1], x2)]
                    
                    # Get landmarks for this face
                    face_landmarks = None
                    if landmarks is not None and i < len(landmarks) and landmarks[i] is not None:
                        face_landmarks = landmarks[i].copy()
                        face_landmarks[:, 0] -= x1
                        face_landmarks[:, 1] -= y1
                    
                    if face_img.size > 0:
                        # Resolve assigned seat(s) for this face so sleeping can be tracked per seat.
                        assigned_seats = [
                            seat_id for seat_id, (assigned_idx, _) in seat_assignments.items()
                            if assigned_idx == i
                        ]

                        # Skip expensive mood processing for unassigned faces.
                        if not assigned_seats:
                            continue

                        is_sleeping = False
                        if self.sleep_detector.available and assigned_seats:
                            should_compute_ear = any(
                                (current_time - self._last_sleep_check.get(seat_id, 0.0)) >= self.sleep_check_interval
                                for seat_id in assigned_seats
                            )
                            ear = self.sleep_detector.compute_ear(face_img) if should_compute_ear else None
                            for seat_id in assigned_seats:
                                if should_compute_ear:
                                    self._last_sleep_check[seat_id] = current_time
                                if self.sleep_detector.update(seat_id, ear, current_time):
                                    is_sleeping = True

                        face_records.append({
                            'box': box,
                            'face_img': face_img,
                            'face_landmarks': face_landmarks,
                            'assigned_seats': assigned_seats,
                            'is_sleeping': is_sleeping,
                        })

                # Run all non-sleeping face crops through one batched model inference.
                batch_images = [r['face_img'] for r in face_records if not r['is_sleeping']]
                batch_landmarks = [r['face_landmarks'] for r in face_records if not r['is_sleeping']]
                batch_results = self.detector.emotion_detector.detect_emotions_batch(batch_images, batch_landmarks)

                batch_result_idx = 0
                for record in face_records:
                    if record['is_sleeping']:
                        emotion, conf, emotion_probs = "sleeping", 1.0, {"sleeping": 1.0}
                    else:
                        emotion, conf, emotion_probs = batch_results[batch_result_idx]
                        batch_result_idx += 1

                    face_id = self.get_face_id(record['box'])
                    self.cached_emotions[face_id] = (emotion, conf)
                    current_face_ids.add(face_id)

                    # Update seat manager with emotion
                    for seat_id in record['assigned_seats']:
                        self.seat_manager.update_seat_emotion(seat_id, emotion, conf, emotion_probs)
                        seat_emotions[seat_id] = (emotion, conf)
                
                # Publish seat emotions via MQTT
                if self.mqtt_publisher and seat_emotions:
                    self.mqtt_publisher.publish_emotions(seat_emotions, current_time)

                # Fire crew alerts for negative emotions above confidence threshold
                self._check_mood_alerts(seat_emotions, current_time)

                # Record mood snapshot for analytics (throttled to ~10 s)
                if seat_emotions and (current_time - self._last_analytics_snapshot) >= 10.0:
                    mood_analytics.record_snapshot(seat_emotions)
                    self._last_analytics_snapshot = current_time

                self.last_emotion_update = current_time
            else:
                # Use cached emotions
                # Use cached emotions
                for box in boxes:
                    current_face_ids.add(self.get_face_id(box))
                
                # Build seat_emotions from seat manager
                for seat_id in seat_assignments.keys():
                    seat = self.seat_manager.seats.get(seat_id)
                    if seat and seat.current_emotion:
                        seat_emotions[seat_id] = (seat.current_emotion, seat.current_confidence)

            # Clear sleep tracking for seats that became vacant
            for seat_id, seat in self.seat_manager.seats.items():
                if not seat.is_occupied:
                    self.sleep_detector.reset(seat_id)
                    self._last_sleep_check.pop(seat_id, None)
            
            # Build emotions list from cache for drawing for drawing
            for box in boxes:
                face_id = self.get_face_id(box)
                if face_id in self.cached_emotions:
                    emotions_list.append(self.cached_emotions[face_id])
                else:
                    emotions_list.append(None)
            
            # Clean up old faces and sleep state
            disappeared_ids = set(self.cached_emotions.keys()) - current_face_ids
            self.cached_emotions = {k: v for k, v in self.cached_emotions.items() 
                                   if k in current_face_ids}
        else:
            # No faces detected - update seat manager with empty boxes
            self.seat_manager.update_seats(None, frame, current_time)
            self.cached_seat_assignments = {}
            self.cached_emotions.clear()
            self.sleep_detector.reset_all()
            self._last_sleep_check.clear()
        
        # Draw detection boxes and seat assignments
        frame = self.detector.draw_enhanced_boxes(
            frame, boxes, probs, landmarks, 
            seat_assignments=seat_assignments,
            emotions=seat_emotions
        )
        
        # Draw seat zones overlay
        frame = self.seat_manager.draw_seat_zones(frame)
        
        frame = self.detector.add_dashboard(frame, boxes, probs)
        
        return frame
    
    def capture_loop(self):
        """Continuous capture and processing loop"""
        global current_frame
        import time
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            # Process the frame
            processed_frame = self.process_frame(frame)
            
            # Update global frame with thread safety
            with frame_lock:
                current_frame = processed_frame.copy()
            
            # Small sleep to prevent CPU overload
            time.sleep(0.01)
    
    def _check_mood_alerts(self, seat_emotions: dict, current_time: float):
        """Broadcast a crew alert when a negative emotion is detected above the confidence threshold."""
        global _app_event_loop
        if _app_event_loop is None:
            return
        for seat_id, (emotion, conf) in seat_emotions.items():
            if emotion not in _NEGATIVE_EMOTIONS:
                continue
            if conf < _MOOD_ALERT_CONFIDENCE:
                continue
            # Respect per-seat cooldown to avoid alert spam
            last_alert = self._mood_cooldowns.get(seat_id, 0.0)
            if (current_time - last_alert) < _MOOD_ALERT_COOLDOWN:
                continue
            self._mood_cooldowns[seat_id] = current_time
            alert = {
                'seatNumber':  seat_id,
                'serviceType': 'mood',
                'emotion':     emotion,
                'confidence':  round(conf, 3),
                'message':     f"Detected {emotion} emotion ({round(conf * 100)}% confidence)",
                'priority':    'high',
                'timestamp':   datetime.now().isoformat(),
            }
            asyncio.run_coroutine_threadsafe(crew_manager.broadcast(alert), _app_event_loop)
            print(f"\U0001f62f Mood alert: seat {seat_id} \u2014 {emotion} ({round(conf * 100)}%)")

    def start(self):
        """Start the capture thread"""
        self.running = True
        self.thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.thread.start()
        print("📹 CV Stream Processor started")
    
    def stop(self):
        """Stop the capture thread"""
        self.running = False
        if self.thread:
            self.thread.join()
        self.cap.release()
        if self.mqtt_publisher:
            self.mqtt_publisher.stop()
        print("📹 CV Stream Processor stopped")


def generate_frames():
    """Generator function for MJPEG stream"""
    global current_frame
    import time
    
    while True:
        with frame_lock:
            if current_frame is None:
                # Send blank frame if no frame available
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Waiting for camera...", (180, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                frame = blank
            else:
                frame = current_frame.copy()
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
        
        # Yield frame in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        # Control frame rate
        time.sleep(0.033)  # ~30 fps


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    global camera
    
    await init_db()
    
    # Create static directory if it doesn't exist
    os.makedirs(STATIC_DIR, exist_ok=True)
    
    # CV Stream Processor is NOT started on startup.
    # Enable it from the Crew Dashboard via POST /api/cv/start
    
    # Print network access info
    hostname = socket.gethostname()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "127.0.0.1"
    
    print(f"\n{'='*60}")
    print(f"🚀 Travel Assistant Started!")
    print(f"{'='*60}")
    print(f"Passenger Interface: https://{local_ip}:8000")
    print(f"Crew Dashboard: https://{local_ip}:8000/crew-dashboard")
    print(f"📁 Static Files:    {STATIC_DIR}")
    print(f"📷 Computer Vision: disabled at startup (enable from Crew Dashboard)")
    print(f"{'='*60}\n")
    
    yield
    
    # Cleanup
    if camera:
        camera.stop()
    print("Travel Assistant shutting down")

# FastAPI Application
app = FastAPI(
    title="Aircraft Travel Assistant with Destinations",
    description="Multi-language travel assistant for aircraft passengers",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# =============================================================================
# ROUTES - Static Pages
# =============================================================================

@app.get("/")
async def read_root():
    """Serve the main HTML interface."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Place your index.html in the static directory"}


@app.get("/crew-dashboard")
async def crew_dashboard(request: Request):
    """Serve crew login or dashboard page depending on auth state."""
    if not _get_crew_user_from_cookie(request):
        login_path = os.path.join(STATIC_DIR, "crew-login.html")
        if os.path.exists(login_path):
            return FileResponse(login_path)
        return {"message": "Crew login page not found. Create crew-login.html in static directory"}

    dashboard_path = os.path.join(STATIC_DIR, "crew-dashboard.html")
    if os.path.exists(dashboard_path):
        return FileResponse(dashboard_path)
    return {"message": "Crew dashboard not found. Create crew-dashboard.html in static directory"}


@app.post("/api/crew/login")
async def crew_login(payload: CrewLoginRequest, request: Request, response: Response):
    """Authenticate crew member and create an HTTP-only session cookie."""
    username = payload.username.strip()
    password = payload.password

    if not username or not password:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username and password are required")

    is_valid = await verify_crew_credentials(username, password)
    if not is_valid:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")

    token = _create_crew_session(username)
    response.set_cookie(
        key=CREW_SESSION_COOKIE,
        value=token,
        max_age=CREW_SESSION_TTL_SECONDS,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
    )
    return {"status": "success", "username": username}


@app.post("/api/crew/logout")
async def crew_logout(request: Request, response: Response):
    """Terminate the crew dashboard session."""
    token = request.cookies.get(CREW_SESSION_COOKIE)
    if token:
        _crew_sessions.pop(token, None)
    response.delete_cookie(CREW_SESSION_COOKIE)
    return {"status": "success"}


@app.get("/api/crew/session")
async def crew_session(username: str = Depends(require_crew_auth)):
    """Return current authenticated crew member."""
    return {"authenticated": True, "username": username}


# =============================================================================
# ROUTES - API Endpoints
# =============================================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "chroma_available": chroma_manager is not None
    }


@app.post("/api/query")
async def handle_query(request: QueryRequest):
    """Handle passenger query with destination context."""
    
    # Get language name for responses
    language_name = get_language_name(request.language)
    
    # Use LLM function calling to detect service requests
    service_request = await Llama.detect_service_with_tools(request.query, request.language)
    if service_request:
        # Return detection info to the client for confirmation — do NOT alert crew yet
        return {
            "service_detected": True,
            "service_type": service_request['serviceType'],
            "priority": service_request['priority'],
            "destination": request.destination,
            "timestamp": datetime.now().isoformat()
        }
    
    # Include language in cache key
    cache_key = query_cache.generate_key(request.destination, f"{request.language}:{request.query}")
    
    # Get conversation history for this seat
    conv_context = conversation_history.format_for_prompt(request.seatNumber)
    has_history = bool(conv_context)
    
    # Only use cache if there's no conversation history (fresh queries can be cached)
    if not has_history:
        cached_response = query_cache.get(cache_key)
        if cached_response:
            # Store in conversation history even for cached responses
            conversation_history.add_exchange(request.seatNumber, request.query, cached_response)
            await log_query(request.seatNumber, request.destination, request.query, cached_response)
            ts = datetime.now().isoformat()
            print(f"[handle_query] cache hit - returning timestamp: {ts}")
            return {
                "answer": cached_response,
                "destination": request.destination,
                "from_cache": True,
                "timestamp": ts
            }
    
    # Translate query to English for better ChromaDB search (if not already English)
    query_english = await translate_to_english(request.query, request.language, Llama)
    
    # Pass English query, target language, and conversation history to context builder
    context_prompt = build_context_from_chroma(
        chroma_manager,
        request.destination, 
        query_english=query_english, 
        language=request.language, 
        conversation_context=conv_context,
        top_k=3
    )

    if not context_prompt and chroma_manager:
        try:
            # Use filtered search with destination for better relevance
            kb_docs = chroma_manager.query(query_english, top_k=3, destination=request.destination)
            if kb_docs:
                kb_lines = []
                for d in kb_docs:
                    if isinstance(d, dict):
                        title = d.get('title', 'Source')
                        content = d.get('content') or d.get('text') or str(d)
                        kb_lines.append(f"**{title}**:\n{content[:1024]}")
                    else:
                        kb_lines.append(f"- {d}")
                kb_section = "\n\nKNOWLEDGE BASE:\n" + "\n\n".join(kb_lines)
                context_prompt = f"""You are a helpful travel assistant. You MUST respond entirely in {language_name}.

CONVERSATION CONTINUITY:
- If the passenger sends a short reply like "yes", "no", "sure", "tell me more", look at the conversation history to understand what they're responding to.
{conv_context}

{kb_section}

PASSENGER MESSAGE:
"""
        except Exception as e:
            print(f"KB lookup failed: {e}")

    if not context_prompt:
        raise HTTPException(status_code=400, detail=f"No knowledge base entries found for '{request.destination}' and no fallback available.")

    full_prompt = f"{context_prompt}\n\n{request.query}\n\nAnswer (in {language_name}):"
    
    print(f"🤖 Querying for: {request.query[:50]}...")
    answer = await Llama.generate_response(full_prompt)
    
    # Only cache if no conversation history (generic queries)
    if not has_history:
        query_cache.set(cache_key, answer)
    
    # Always store in conversation history and persist to DB for analytics
    conversation_history.add_exchange(request.seatNumber, request.query, answer)
    await log_query(request.seatNumber, request.destination, request.query, answer)
    
    ts = datetime.now().isoformat()
    print(f"[handle_query] model response - returning timestamp: {ts}")
    return {
        "answer": answer,
        "destination": request.destination,
        "from_cache": False,
        "has_history": has_history,
        "timestamp": ts
    }



@app.post("/api/service-request")
async def service_request(request: QueryRequest):
    """Passenger confirms a service request — alert crew and return localized acknowledgment."""
    service_info = await Llama.detect_service_with_tools(request.query, request.language)
    service_type = service_info['serviceType'] if service_info else 'assistance'
    priority = service_info['priority'] if service_info else 'medium'

    alert = {
        'seatNumber': request.seatNumber,
        'serviceType': service_type,
        'message': request.query,
        'priority': priority,
        'timestamp': datetime.now().isoformat()
    }
    await crew_manager.broadcast(alert)
    print(f"🚨 Service request confirmed: {service_type} from seat {request.seatNumber}")

    response_message = get_service_message(
        request.language,
        service_type,
        request.seatNumber
    )
    await log_query(request.seatNumber, request.destination, request.query, response_message)
    return {
        "answer": response_message,
        "destination": request.destination,
        "from_cache": False,
        "service_alert_sent": True,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/crew-request")
async def crew_request(request: QueryRequest):
    """Passenger requests crew assistance."""
    return {
        "status": "success",
        "message": "Crew request submitted (not logged)",
        "seatNumber": request.seatNumber
    }


@app.get("/api/stats")
async def get_statistics(_crew_user: str = Depends(require_crew_auth)):
    """Get system statistics."""
    db_stats = await get_stats()
    
    return {
        **db_stats,
        "model": MODEL_NAME,
        "cache_size": query_cache.size(),
        "active_conversations": conversation_history.active_count(),
        "chroma_available": chroma_manager is not None
    }


# =============================================================================
# ROUTES - Conversation Management
# =============================================================================

@app.get("/api/conversation/{seat_number}")
async def get_conversation(seat_number: str):
    """Get conversation history for a specific seat."""
    history = conversation_history.get_history(seat_number)
    return {
        "seat_number": seat_number,
        "turns": len(history) // 2,
        "messages": history
    }


@app.delete("/api/conversation/{seat_number}")
async def clear_conversation(seat_number: str):
    """Clear conversation history for a specific seat."""
    cleared = conversation_history.clear_history(seat_number)
    if cleared:
        return {
            "status": "success",
            "message": f"Conversation history cleared for seat {seat_number}"
        }
    return {
        "status": "not_found",
        "message": f"No conversation history found for seat {seat_number}"
    }


@app.delete("/api/conversations")
async def clear_all_conversations():
    """Clear all conversation histories (admin endpoint)."""
    count = conversation_history.clear_all()
    return {
        "status": "success",
        "message": f"Cleared {count} conversation sessions"
    }


# =============================================================================
# ROUTES - WebSocket & Crew
# =============================================================================

@app.websocket("/ws/crew")
async def crew_websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for crew dashboard."""
    if not _get_crew_user_from_websocket(websocket):
        await websocket.close(code=1008, reason="Crew authentication required")
        return

    await crew_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(data)
    except WebSocketDisconnect:
        crew_manager.disconnect(websocket)


@app.get("/api/crew/alerts")
async def get_crew_alerts(_crew_user: str = Depends(require_crew_auth)):
    """Get all stored crew alerts."""
    return crew_manager.get_alerts()


@app.post("/api/crew/acknowledge")
async def acknowledge_crew_alert(request: dict, _crew_user: str = Depends(require_crew_auth)):
    """Acknowledge a crew alert."""
    alert_id = request.get('alertId')
    if alert_id is not None:
        success = crew_manager.acknowledge_alert(alert_id)
        if success:
            return {"status": "success", "message": "Alert acknowledged"}
        return {"status": "not_found", "message": "Alert not found"}
    return {"status": "error", "message": "No alertId provided"}


@app.delete("/api/crew/alerts")
async def clear_crew_alerts(acknowledged_only: bool = True, _crew_user: str = Depends(require_crew_auth)):
    """Clear crew alerts. By default only clears acknowledged alerts."""
    if acknowledged_only:
        count = crew_manager.clear_acknowledged()
        return {"status": "success", "message": f"Cleared {count} acknowledged alerts"}
    else:
        count = crew_manager.clear_all_alerts()
        return {"status": "success", "message": f"Cleared all {count} alerts"}


@app.get("/api/seats")
async def get_seat_status(_crew_user: str = Depends(require_crew_auth)):
    """Get seat calibration data and current seat status."""
    if camera is None or not hasattr(camera, 'seat_manager'):
        return {
            "error": "Camera or seat manager not available",
            "seats": {}
        }
    
    seat_manager = camera.seat_manager
    seat_summary = seat_manager.get_seat_summary()
    
    # Add calibration zones
    calibration_data = {
        "frame_width": seat_manager.frame_width,
        "frame_height": seat_manager.frame_height,
        "seats": {}
    }
    
    for seat_id, seat_state in seat_manager.seats.items():
        calibration_data["seats"][seat_id] = {
            "zone": list(seat_state.zone),
            "polygon": [list(p) for p in seat_state.polygon] if seat_state.polygon else None,
            "occupied": seat_summary[seat_id]["occupied"],
            "emotion": seat_summary[seat_id]["emotion"],
            "confidence": seat_state.current_confidence if seat_state.is_occupied else 0.0
        }
    
    return calibration_data


# =============================================================================
# ROUTES - Analytics
# =============================================================================

@app.get("/api/analytics/mood-timeline")
async def analytics_mood_timeline(_crew_user: str = Depends(require_crew_auth)):
    """Time-bucketed mood group counts for the line chart."""
    return mood_analytics.get_mood_over_time(bucket_seconds=60)


@app.get("/api/analytics/mood-distribution")
async def analytics_mood_distribution(_crew_user: str = Depends(require_crew_auth)):
    """Total per-emotion counts across the session for the doughnut chart."""
    return mood_analytics.get_mood_distribution()


@app.get("/api/analytics/alerts")
async def analytics_alerts(_crew_user: str = Depends(require_crew_auth)):
    """Alert frequency over time and average crew response time."""
    all_alerts = crew_manager.get_alerts()

    # Bucket alerts into 5-minute windows
    buckets: dict[str, int] = {}
    response_times: list[float] = []
    emotion_counts: dict[str, int] = {}

    for a in all_alerts:
        ts = a.get("timestamp")
        if ts:
            try:
                dt = datetime.fromisoformat(ts)
                # Round to 5-minute bucket
                minute = dt.minute - (dt.minute % 5)
                label = dt.strftime("%H:") + f"{minute:02d}"
                buckets[label] = buckets.get(label, 0) + 1
            except (ValueError, TypeError):
                pass

        # Response time
        ack_ts = a.get("acknowledged_at")
        if ack_ts and ts:
            try:
                dt_created = datetime.fromisoformat(ts)
                dt_acked = datetime.fromisoformat(ack_ts)
                delta = (dt_acked - dt_created).total_seconds()
                if delta >= 0:
                    response_times.append(delta)
            except (ValueError, TypeError):
                pass

        # Emotion breakdown
        emo = a.get("emotion")
        if emo:
            emotion_counts[emo] = emotion_counts.get(emo, 0) + 1

    # Sort buckets chronologically
    sorted_labels = sorted(buckets.keys())
    sorted_counts = [buckets[l] for l in sorted_labels]

    avg_response = round(sum(response_times) / len(response_times), 1) if response_times else None

    return {
        "buckets": sorted_labels,
        "counts": sorted_counts,
        "total": len(all_alerts),
        "pending": len([a for a in all_alerts if not a.get("acknowledged")]),
        "avg_response_seconds": avg_response,
        "emotion_breakdown": emotion_counts,
    }


@app.get("/api/analytics/queries")
async def analytics_queries(_crew_user: str = Depends(require_crew_auth)):
    """Query volume over time, broken down by topic category and seat."""
    import aiosqlite
    from services.config import DB_FILE

    # Topic category keyword lists — a query can match multiple categories
    QUERY_CATEGORIES: dict[str, list[str]] = {
        "Food & Beverage": [
            "food", "eat", "eating", "drink", "meal", "water", "juice", "coffee", "tea",
            "snack", "hungry", "thirsty", "menu", "dinner", "lunch", "breakfast", "wine",
            "beer", "alcohol", "vegetarian", "vegan", "beverage", "bottle", "hot chocolate",
            "soda", "dessert", "fruit", "bread", "butter", "sandwich",
        ],
        "Entertainment": [
            "movie", "film", "music", "show", "tv", "television", "game", "channel",
            "headphone", "earphone", "wifi", "wi-fi", "internet", "screen", "watch",
            "listen", "play", "entertainment", "song", "video", "radio", "podcast",
            "streaming", "series", "episode",
        ],
        "Comfort": [
            "pillow", "blanket", "seat", "cold", "hot", "warm", "temperature",
            "uncomfortable", "pain", "sick", "ill", "sleep", "tired", "rest", "recline",
            "air", "noise", "light", "dark", "legroom", "space", "smelly", "smell",
            "smell", "headache", "nausea", "shaking",
        ],
        "Travel Info": [
            "destination", "airport", "landing", "arrival", "arrive", "departure",
            "depart", "time", "hour", "delay", "gate", "connection", "flight", "route",
            "turbulence", "weather", "when", "where", "how long", "duration", "distance",
            "altitude", "speed", "timezone", "local time",
        ],
        "Assistance": [
            "help", "assist", "assistance", "need", "request", "call", "staff", "crew",
            "attendant", "problem", "issue", "wrong", "broken", "emergency", "medical",
            "doctor", "allergy", "wheelchair", "lost", "missing", "complaint",
        ],
    }

    session_start = mood_analytics.get_session_start()

    try:
        db = await aiosqlite.connect(DB_FILE)
        db.row_factory = aiosqlite.Row

        if session_start:
            cursor = await db.execute(
                "SELECT timestamp, seatNumber, queryText FROM passengerQueries WHERE timestamp >= ? ORDER BY timestamp",
                (session_start,),
            )
        else:
            cursor = await db.execute(
                "SELECT timestamp, seatNumber, queryText FROM passengerQueries ORDER BY timestamp"
            )
        rows = await cursor.fetchall()
        await db.close()
    except Exception as e:
        return {"error": str(e), "buckets": [], "by_category": {}, "by_seat": {}, "total": 0}

    # Collect all time-bucket labels
    all_labels: set[str] = set()
    # by_category_buckets[category][label] = count
    by_category_buckets: dict[str, dict[str, int]] = {cat: {} for cat in QUERY_CATEGORIES}
    by_seat: dict[str, int] = {}

    for row in rows:
        ts_str = row["timestamp"]
        seat = row["seatNumber"]
        query_text = (row["queryText"] or "").lower()

        label: str | None = None
        if ts_str:
            try:
                dt = datetime.fromisoformat(ts_str)
                minute = dt.minute - (dt.minute % 5)
                label = dt.strftime("%H:") + f"{minute:02d}"
                all_labels.add(label)
            except (ValueError, TypeError):
                pass

        if seat:
            by_seat[seat] = by_seat.get(seat, 0) + 1

        if label:
            for category, keywords in QUERY_CATEGORIES.items():
                if any(kw in query_text for kw in keywords):
                    by_category_buckets[category][label] = (
                        by_category_buckets[category].get(label, 0) + 1
                    )

    sorted_labels = sorted(all_labels)
    by_category = {
        cat: [by_category_buckets[cat].get(lbl, 0) for lbl in sorted_labels]
        for cat in QUERY_CATEGORIES
    }

    return {
        "buckets": sorted_labels,
        "by_category": by_category,
        "by_seat": by_seat,
        "total": len(rows),
    }


#=============================================================================
# ROUTES - CV Video Stream
# =============================================================================

@app.post("/api/cv/start")
async def start_cv(payload: dict | None = None, _crew_user: str = Depends(require_crew_auth)):
    """Start the CV stream processor (called from Crew Dashboard)."""
    global camera, _app_event_loop
    _app_event_loop = asyncio.get_event_loop()
    if camera is not None and camera.running:
        return {"status": "already_running", "message": "CV stream is already active"}

    no_mqtt_override = None
    if isinstance(payload, dict) and "no_mqtt" in payload:
        no_mqtt_override = bool(payload.get("no_mqtt"))

    try:
        camera = CVStreamProcessor(camera_index=0, no_mqtt=no_mqtt_override)
        camera.start()
        print(f"📹 CV Stream started from Crew Dashboard (no_mqtt={camera.no_mqtt})")
        return {
            "status": "started",
            "message": "CV stream started",
            "no_mqtt": camera.no_mqtt,
            "mqtt_enabled": not camera.no_mqtt,
        }
    except Exception as e:
        camera = None
        print(f"⚠️  CV Stream failed to start: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start CV stream: {e}")


@app.post("/api/cv/stop")
async def stop_cv(_crew_user: str = Depends(require_crew_auth)):
    """Stop the CV stream processor (called from Crew Dashboard)."""
    global camera
    if camera is None:
        return {"status": "not_running", "message": "CV stream is not active"}
    try:
        camera.stop()
        camera = None
        print("📹 CV Stream stopped from Crew Dashboard")
        return {"status": "stopped", "message": "CV stream stopped"}
    except Exception as e:
        print(f"⚠️  Error stopping CV stream: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop CV stream: {e}")


@app.post("/api/cv/calibrate")
async def calibrate_cv(_crew_user: str = Depends(require_crew_auth)):
    """Run interactive seat calibration, then restart CV stream with updated zones."""
    global camera, current_frame, _app_event_loop

    if camera is None or not camera.running:
        raise HTTPException(status_code=400, detail="CV stream must be running before calibration")

    frame_width = int(getattr(camera, "frame_width", 1280))
    frame_height = int(getattr(camera, "frame_height", 720))
    no_mqtt_mode = bool(getattr(camera, "no_mqtt", False))
    num_seats = 4
    if hasattr(camera, "seat_manager") and getattr(camera.seat_manager, "seats", None):
        num_seats = max(1, len(camera.seat_manager.seats))

    try:
        camera.stop()
        camera = None
        with frame_lock:
            current_frame = None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop CV stream for calibration: {e}")

    cv_dir = Path(__file__).parent.parent / "computer-vision"
    if str(cv_dir) not in sys.path:
        sys.path.insert(0, str(cv_dir))

    calibration_error = None
    calibration_saved = False
    try:
        run_calibration = __import__("mood_detection", fromlist=["run_calibration"]).run_calibration
        calibration_saved = await asyncio.to_thread(
            run_calibration,
            "0",
            frame_width,
            frame_height,
            num_seats,
        )
    except Exception as e:
        calibration_error = str(e)

    restart_error = None
    try:
        camera = CVStreamProcessor(camera_index=0, no_mqtt=no_mqtt_mode)
        camera.start()
        _app_event_loop = asyncio.get_running_loop()
    except Exception as e:
        camera = None
        restart_error = str(e)

    if restart_error:
        raise HTTPException(status_code=500, detail=f"Calibration finished but failed to restart CV stream: {restart_error}")

    if calibration_saved:
        return {
            "status": "calibrated",
            "message": "Seat calibration saved and CV stream restarted"
        }

    if calibration_error:
        return {
            "status": "failed",
            "message": f"Calibration failed: {calibration_error}. CV stream was restarted with previous calibration."
        }

    return {
        "status": "cancelled",
        "message": "Calibration cancelled. CV stream restarted with previous calibration."
    }


@app.get("/api/cv/status")
async def cv_status(_crew_user: str = Depends(require_crew_auth)):
    """Get current CV stream status."""
    return {
        "enabled": camera is not None and camera.running,
        "message": "CV stream active" if (camera is not None and camera.running) else "CV stream inactive",
        "no_mqtt": bool(getattr(camera, "no_mqtt", False)) if camera is not None else None,
        "mqtt_enabled": (not bool(getattr(camera, "no_mqtt", False))) if camera is not None else None,
    }


# =============================================================================
# ROUTES - Lighting Override
# =============================================================================

@app.get("/api/lighting/override")
async def get_lighting_override(_crew_user: str = Depends(require_crew_auth)):
    """Return the current lighting override state."""
    return _lighting_override


@app.post("/api/lighting/override")
async def set_lighting_override(payload: dict, _crew_user: str = Depends(require_crew_auth)):
    """Set the lighting override state and publish to MQTT.

    Body: {"enabled": bool, "scene": "HAPPY" | "NEUTRAL" | "STRESSED" | "ANGRY" | "SAD" | "ANXIOUS"}
    """
    global _lighting_override
    enabled = bool(payload.get("enabled", False))
    scene   = str(payload.get("scene", "NEUTRAL")).upper()
    if scene not in _VALID_SCENES:
        raise HTTPException(status_code=400, detail=f"Invalid scene '{scene}'. Valid: {sorted(_VALID_SCENES)}")
    _lighting_override = {"enabled": enabled, "scene": scene}
    _publish_lighting_override(enabled, scene)
    return _lighting_override


@app.get("/video_feed")
async def video_feed(_crew_user: str = Depends(require_crew_auth)):
    """Video streaming route for CV emotion detection. Returns MJPEG stream."""
    if camera is None:
        raise HTTPException(status_code=503, detail="Camera not available")
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0", 
        port=8000,
        reload=True
    )
