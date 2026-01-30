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

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    await init_db()
    
    # Create static directory if it doesn't exist
    os.makedirs(STATIC_DIR, exist_ok=True)
    
    # Print network access info
    hostname = socket.gethostname()
    try:
        # Get actual local IP by connecting to external address
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
    print(f"{'='*60}\n")
    
    yield
    
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
