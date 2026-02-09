"""
Aircraft Travel Assistant - Main FastAPI Application

A multi-language travel assistant for aircraft passengers with:
- Destination information from ChromaDB knowledge base
- Multilingual support (EN, ES, FR, DE, HI, PT, TH)
- In-flight service request detection
- Crew dashboard with WebSocket alerts
- Conversation history per seat
"""
import os
import socket
from datetime import datetime
from contextlib import asynccontextmanager
import cv2
import threading
import numpy as np
from pathlib import Path
import time

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
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
    QueryRequest, FeedbackRequest,
    # Language
    get_language_name, get_service_message, translate_to_english,
    # Services
    crew_manager,
    build_context_from_chroma, init_chroma_manager, get_chroma_manager,
    init_llm,
    query_cache, conversation_history,
    init_db, get_stats,
)


# Initialize services
Llama = init_llm()
chroma_manager = init_chroma_manager(CHROMA_PERSIST_DIR)

#============================================================================
# CV STREAM PROCESSOR
# =============================================================================

# Global variables for video stream
camera = None
current_frame = None
frame_lock = threading.Lock()

class CVStreamProcessor:
    def __init__(self, camera_index=0):
        # Import CV modules
        sys.path.insert(0, str(Path(__file__).parent.parent / "computer-vision"))
        from mood_detection import FaceDetector
        from seat_manager import SeatManager
        from sleep_detector import SleepDetector
        from seat_manager import SeatManager
        
        self.detector = FaceDetector()
        self.sleep_detector = SleepDetector()
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
                self.seat_manager._generate_grid_zones()
        else:
            print(f"⚠️  No calibration found at {calibration_path}, using default grid")
            self.seat_manager._generate_grid_zones()
        
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
                self.seat_manager._generate_grid_zones()
        else:
            print(f"⚠️  No calibration found at {calibration_path}, using default grid")
            self.seat_manager._generate_grid_zones()
        
        self.emotion_update_interval = 0.3
        self.last_emotion_update = time.time()
        self.cached_emotions = {}
        
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
            # Update seat assignments (pass frame for embedding extraction)
            seat_assignments = self.seat_manager.update_seats(boxes, frame, current_time)
            
            # Update seat assignments (pass frame for embedding extraction)
            seat_assignments = self.seat_manager.update_seats(boxes, frame, current_time)
            
            # Update emotions if interval passed
            if (current_time - self.last_emotion_update) > self.emotion_update_interval:
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
                        emotion, conf, emotion_probs = self.detector.emotion_detector.detect_emotion(
                            face_img, face_landmarks
                        )
                        face_id = self.get_face_id(box)
                        self.cached_emotions[face_id] = (emotion, conf)
                        current_face_ids.add(face_id)
                        
                        # Update seat manager with emotion
                        for seat_id, (assigned_idx, _) in seat_assignments.items():
                            if assigned_idx == i:
                                self.seat_manager.update_seat_emotion(seat_id, emotion, conf, emotion_probs)
                                seat_emotions[seat_id] = (emotion, conf)
                
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
                
                # Build seat_emotions from seat manager
                for seat_id in seat_assignments.keys():
                    seat = self.seat_manager.seats.get(seat_id)
                    if seat and seat.current_emotion:
                        seat_emotions[seat_id] = (seat.current_emotion, seat.current_confidence)
            
            # Build emotions list from cache for drawing for drawing
            for box in boxes:
                face_id = self.get_face_id(box)
                if face_id in self.cached_emotions:
                    emotions_list.append(self.cached_emotions[face_id])
                else:
                    emotions_list.append(None)
            
            # Clean up old faces and sleep state
            disappeared_ids = set(self.cached_emotions.keys()) - current_face_ids
            for old_id in disappeared_ids:
                self.sleep_detector.reset(old_id)
            self.cached_emotions = {k: v for k, v in self.cached_emotions.items() 
                                   if k in current_face_ids}
        else:
            # No faces detected - update seat manager with empty boxes
            self.seat_manager.update_seats(None, frame, current_time)
            # No faces detected - update seat manager with empty boxes
            self.seat_manager.update_seats(None, frame, current_time)
            self.cached_emotions.clear()
            self.sleep_detector.reset_all()
        
        # Draw detection boxes and seat assignments
        frame = self.detector.draw_enhanced_boxes(
            frame, boxes, probs, landmarks, 
            seat_assignments=seat_assignments,
            emotions=seat_emotions
        )
        
        # Draw seat zones overlay
        frame = self.seat_manager.draw_seat_zones(frame)
        
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
    
    # Initialize CV Stream Processor
    try:
        camera = CVStreamProcessor(camera_index=0)
        camera.start()
    except Exception as e:
        print(f"⚠️  CV Stream failed to start: {e}")
        camera = None
    
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
    print(f"📍 Local Access:    http://localhost:8000")
    print(f"🌐 Network Access:  http://{local_ip}:8000")
    print(f"📁 Static Files:    {STATIC_DIR}")
    if camera:
        print(f"📹 Video Stream:    http://localhost:8000/video_feed")
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
async def crew_dashboard():
    """Serve crew dashboard HTML."""
    dashboard_path = os.path.join(STATIC_DIR, "crew-dashboard.html")
    if os.path.exists(dashboard_path):
        return FileResponse(dashboard_path)
    return {"message": "Crew dashboard not found. Create crew-dashboard.html in static directory"}


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
        alert = {
            'seatNumber': request.seatNumber,
            'serviceType': service_request['serviceType'],
            'message': request.query,  # Show the original passenger query
            'priority': service_request['priority'],
            'timestamp': datetime.now().isoformat()
        }
        await crew_manager.broadcast(alert)
        print(f"🚨 Service request detected: {service_request['serviceType']} from seat {request.seatNumber}")
        response_message = get_service_message(
            request.language, 
            service_request['serviceType'], 
            request.seatNumber
        )
        return {
            "answer": response_message,
            "destination": request.destination,
            "from_cache": False,
            "service_alert_sent": True,
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
    
    # Always store in conversation history
    conversation_history.add_exchange(request.seatNumber, request.query, answer)
    
    ts = datetime.now().isoformat()
    print(f"[handle_query] model response - returning timestamp: {ts}")
    return {
        "answer": answer,
        "destination": request.destination,
        "from_cache": False,
        "has_history": has_history,
        "timestamp": ts
    }


@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Receive comfort feedback."""
    return {
        "status": "success",
        "message": "Feedback received (not logged)",
        "seatNumber": request.seatNumber
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
async def get_statistics():
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
    await crew_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(data)
    except WebSocketDisconnect:
        crew_manager.disconnect(websocket)


@app.get("/api/crew/alerts")
async def get_crew_alerts():
    """Get all stored crew alerts."""
    return crew_manager.get_alerts()


@app.post("/api/crew/acknowledge")
async def acknowledge_crew_alert(request: dict):
    """Acknowledge a crew alert."""
    alert_id = request.get('alertId')
    if alert_id is not None:
        success = crew_manager.acknowledge_alert(alert_id)
        if success:
            return {"status": "success", "message": "Alert acknowledged"}
        return {"status": "not_found", "message": "Alert not found"}
    return {"status": "error", "message": "No alertId provided"}


@app.delete("/api/crew/alerts")
async def clear_crew_alerts(acknowledged_only: bool = True):
    """Clear crew alerts. By default only clears acknowledged alerts."""
    if acknowledged_only:
        count = crew_manager.clear_acknowledged()
        return {"status": "success", "message": f"Cleared {count} acknowledged alerts"}
    else:
        count = crew_manager.clear_all_alerts()
        return {"status": "success", "message": f"Cleared all {count} alerts"}


@app.get("/api/seats")
async def get_seat_status():
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

#=============================================================================
# ROUTES - CV Video Stream
# =============================================================================

@app.get("/video_feed")
async def video_feed():
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
