from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
import json
import re
from datetime import datetime
import aiosqlite
import os
import uvicorn
import asyncio
from ollama import Client
from dotenv import load_dotenv
from wikivoyage_chromadb_bot import ChromaDBManager
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://192.168.0.179:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2:3b")
DB_FILE = os.getenv("DB_FILE", "./data/aircraft.db")

# Chroma config
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "wikivoyage_travel")

# Static files directory
STATIC_DIR = os.getenv("STATIC_DIR", "./static")


class QueryRequest(BaseModel):
    query: str
    destination: str
    seatNumber: str
    language: str = "en"


class FeedbackRequest(BaseModel):
    seatNumber: str
    temperature: float
    lighting: float
    noise_level: float
    overall_comfort: str


class CrewAlertRequest(BaseModel):
    seatNumber: str
    serviceType: str
    message: str
    priority: str = "medium"


class ConnectionManager:
    """Manage WebSocket connections for crew dashboard"""
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"✅ Crew dashboard connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"❌ Crew dashboard disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast alert to all connected crew dashboards"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Error sending to connection: {e}")
                disconnected.append(connection)
        
        for conn in disconnected:
            if conn in self.active_connections:
                self.active_connections.remove(conn)


crew_manager = ConnectionManager()


def detect_service_request(query: str) -> Optional[Dict[str, str]]:
    """Detect if user is requesting in-flight services using pattern matching"""
    query_lower = query.lower()
    
    service_patterns = {
        'beverage': r'(water|drink|beverage|coffee|tea|juice|soda|wine|beer|alcohol|cocktail|champagne|thirsty)',
        'food': r'(food|meal|snack|hungry|eat|breakfast|lunch|dinner|sandwich|salad)',
        'blanket': r'(blanket|cold|freezing|chilly|cover)',
        'pillow': r'(pillow|cushion|headrest)',
        'assistance': r'(help me|assist me|call.*(attendant|crew)|emergency)',
        'medical': r'(sick|ill|medicine|pain|doctor|medical|nausea|headache|not feeling well)',
        'entertainment': r'(movie|music|headphone|entertainment|wifi|internet)',
    }
    
    for service_type, pattern in service_patterns.items():
        if re.search(pattern, query_lower):
            priority = 'high' if service_type == 'medical' else 'medium'
            if service_type in ['entertainment']:
                priority = 'low'
            
            return {
                'serviceType': service_type,
                'priority': priority,
                'message': query
            }
    
    return None


def build_context_from_chroma(destination: str, top_k: int = 5) -> str:
    """Build a model prompt using Chroma KB as the primary source."""
    if not chroma_manager:
        return ""

    try:
        kb_docs = chroma_manager.query(destination, top_k=top_k)
        if not kb_docs:
            return ""
            
        kb_lines = []
        for doc in kb_docs:
            title = doc.get('title') or 'Source'
            content = doc.get('content') or doc.get('text') or str(doc)
            snippet = content[:2048].strip()
            kb_lines.append(f"**{title}**:\n{snippet}")

        kb_section = "\n\nKNOWLEDGE BASE (Chroma):\n" + "\n\n".join(kb_lines)

        context = f"""
        You are a helpful travel assistant on an aircraft providing accurate destination information to passengers.

        IMPORTANT INSTRUCTIONS:
        - If query is not related to travel, culture, language, destination, politely inform the passenger that you can only assist with travel-related questions.
        - Only provide information that is FACTUALLY ACCURATE
        - If you're unsure about specific details (addresses, prices, opening hours), say "I don't have specific details, but..."
        - Do NOT make up business names, addresses, or locations
        - For activities like skydiving, mention general areas or nearby cities rather than specific made-up venues
        - Always distinguish between general information and specific recommendations
        - If asked about specific services, provide general guidance rather than inventing locations

        FORMATTING INSTRUCTIONS:
        - Use **bold** for emphasis on key points
        - Use bullet points (- or *) for lists
        - Use numbered lists (1. 2. 3.) for sequential information
        - Use headers (## or ###) for section titles
        - Add line breaks between paragraphs for readability
        - Use > for important notes or tips
        - Format your response in Markdown for better readability

        {kb_section}

        INSTRUCTIONS FOR RESPONSES:
        - Be warm and welcoming but keep responses concise 
        - Use the Chroma knowledge base above as your primary reference
        - Avoid fabricating details about businesses, venues, or services
        - If unsure, admit lack of specific details rather than fabricating

        PASSENGER MESSAGE:
        """

        return context
    except Exception as e:
        print(f"Chroma context build error: {e}")
        return ""


class ChromaManager:
    """Wrapper around ChromaDBManager for compatibility with travel_assistant"""
    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR, collection_name: str = "wikivoyage_travel", embedding_model: str = EMBEDDING_MODEL):
        try:
            print(persist_dir)
            self.chroma_db = ChromaDBManager(
                db_path=persist_dir, 
                model_name=embedding_model, 
                collection_name=collection_name
            )
            self.collection = self.chroma_db.collection
            print(f"✅ Chroma initialized (persist_dir={persist_dir}, collection={collection_name})")
        except Exception as e:
            print(f"⚠️  Chroma init failed: {e}")
            self.chroma_db = None
            self.collection = None

    def query(self, query_text: str, top_k: int = 3) -> list:
        """Return top_k documents (strings) relevant to the query_text."""
        if not self.chroma_db:
            return []
        try:
            search_results = self.chroma_db.search(query_text, n_results=top_k)
            return search_results
        except Exception as e:
            print(f"Chroma query error: {e}")
            return []


chroma_manager = None


class LlamaInterface:
    def __init__(self, ollama_url: str, model_name: str):
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.client = Client(host=ollama_url)
        print(f"LlamaInterface initialized")
        print(f"   Model: {self.model_name}")
        print(f"   URL: {self.ollama_url}")
    
    async def generate_response(self, prompt: str, temperature: float = 0.7) -> str:
        """Query model using Ollama library"""
        try:
            print(f"Querying model: {self.model_name}")
            
            loop = asyncio.get_event_loop()
            
            response = await loop.run_in_executor(
                None,
                lambda: self.client.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options = {
                        "temperature": 0.1,        
                        "top_p": 0.6,
                        "top_k": 3
                    },
                    keep_alive=-1,
                    stream=False
                )
            )
            
            answer = response['response'].strip()
            print(f"Got response: {answer[:50]}...")
            return answer
        
        except Exception as e:
            print(f"Error: {e}")
            return f"Sorry, an error occurred: {str(e)}"


print(f"\nInitializing Llama...")
Llama = LlamaInterface(OLLAMA_URL, MODEL_NAME)
print(f"Llama ready!\n")

try:
    chroma_manager = ChromaManager(CHROMA_PERSIST_DIR)
except Exception as e:
    print(f"ChromaManager instantiation error: {e}")
    chroma_manager = None


class QueryCache:
    def __init__(self, ttl_seconds: int = 3600):
        self.cache = {}
        self.ttl = ttl_seconds
    
    def get(self, key: str):
        """Get cached response"""
        if key in self.cache:
            entry = self.cache[key]
            if datetime.now().timestamp() < entry['expires']:
                return entry['value']
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: str):
        """Cache response"""
        self.cache[key] = {
            'value': value,
            'expires': datetime.now().timestamp() + self.ttl
        }
    
    def generate_key(self, destination: str, query: str) -> str:
        """Generate cache key"""
        return f"query:{destination.lower()}:{hash(query)}"


query_cache = QueryCache(ttl_seconds=3600)


async def init_db():
    """Initialize SQLite database"""
    try:
        os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
        
        db = await aiosqlite.connect(DB_FILE)
        
        await db.execute("""
            CREATE TABLE IF NOT EXISTS passengerQueries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                seatNumber TEXT NOT NULL,
                destination TEXT NOT NULL,
                queryText TEXT NOT NULL,
                responseText TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        await db.execute("""
            CREATE TABLE IF NOT EXISTS comfortFeedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                seatNumber TEXT NOT NULL,
                temperature FLOAT,
                lighting FLOAT,
                noise_level FLOAT,
                overall_comfort TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        await db.commit()
        await db.close()
        print("Database initialized")
    except Exception as e:
        print(f"Database error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    await init_db()
    
    # Create static directory if it doesn't exist
    os.makedirs(STATIC_DIR, exist_ok=True)
    
    # Print network access info
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"\n{'='*60}")
    print(f"🚀 Travel Assistant Started!")
    print(f"{'='*60}")
    print(f"📍 Local Access:    http://localhost:8000")
    print(f"🌐 Network Access:  http://{local_ip}:8000")
    print(f"📁 Static Files:    {STATIC_DIR}")
    print(f"{'='*60}\n")
    
    yield
    
    print("Travel Assistant shutting down")


# FASTAPI APP
app = FastAPI(
    title="Aircraft Travel Assistant with Destinations",
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


# Root endpoint serves the main HTML file
@app.get("/")
async def read_root():
    """Serve the main HTML interface"""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Place your index.html in the static directory"}


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "chroma_available": chroma_manager is not None
    }


@app.post("/api/query")
async def handle_query(request: QueryRequest):
    """Handle passenger query with destination context"""
    
    # First, check if this is a service request
    service_request = detect_service_request(request.query)
    if service_request:
        alert = {
            'seatNumber': request.seatNumber,
            'serviceType': service_request['serviceType'],
            'message': service_request['message'],
            'priority': service_request['priority'],
            'timestamp': datetime.now().isoformat()
        }
        
        await crew_manager.broadcast(alert)
        print(f"🚨 Service request detected: {service_request['serviceType']} from seat {request.seatNumber}")
        
        return {
            "answer": f"I've notified the cabin crew about your **{service_request['serviceType']}** request. A flight attendant will assist you at seat **{request.seatNumber}** shortly.\n\n> ✅ Your request has been sent to the crew dashboard.",
            "destination": request.destination,
            "from_cache": False,
            "service_alert_sent": True,
            "timestamp": datetime.now().isoformat()
        }
    
    cache_key = query_cache.generate_key(request.destination, request.query)
    cached_response = query_cache.get(cache_key)
    
    if cached_response:
        ts = datetime.now().isoformat()
        print(f"[handle_query] cache hit - returning timestamp: {ts}")
        return {
            "answer": cached_response,
            "destination": request.destination,
            "from_cache": True,
            "timestamp": ts
        }
    
    context_prompt = build_context_from_chroma(request.destination, top_k=3)

    if not context_prompt and chroma_manager:
        try:
            kb_docs = chroma_manager.query(request.query, top_k=3)
            if kb_docs:
                kb_lines = [f"- {d}" for d in kb_docs]
                kb_section = "\n\nKNOWLEDGE BASE:\n" + "\n\n".join(kb_lines)
                context_prompt = f"You are a helpful travel assistant.\n\n{kb_section}\n\nPASSENGER MESSAGE:\n"
        except Exception as e:
            print(f"KB lookup failed: {e}")

    if not context_prompt:
        raise HTTPException(status_code=400, detail=f"No knowledge base entries found for '{request.destination}' and no fallback available.")

    full_prompt = f"{context_prompt}\n\n{request.query}\n\nAnswer:"
    
    print(f"🤖 Querying for: {request.query[:50]}...")
    answer = await Llama.generate_response(full_prompt)
    
    query_cache.set(cache_key, answer)
    
    ts = datetime.now().isoformat()
    print(f"[handle_query] model response - returning timestamp: {ts}")
    return {
        "answer": answer,
        "destination": request.destination,
        "from_cache": False,
        "timestamp": ts
    }


@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Receive comfort feedback"""
    return {
        "status": "success",
        "message": "Feedback received (not logged)",
        "seatNumber": request.seatNumber
    }


@app.post("/api/crew-request")
async def crew_request(request: QueryRequest):
    """Passenger requests crew assistance"""
    return {
        "status": "success",
        "message": "Crew request submitted (not logged)",
        "seatNumber": request.seatNumber
    }


@app.get("/api/stats")
async def get_statistics():
    """Get system statistics"""
    try:
        db = await aiosqlite.connect(DB_FILE)
        
        cursor = await db.execute("SELECT COUNT(*) FROM passengerQueries")
        total_queries = (await cursor.fetchone())[0]
        
        cursor = await db.execute("SELECT COUNT(*) FROM comfortFeedback")
        total_feedback = (await cursor.fetchone())[0]
        
        await db.close()
        
        return {
            "total_queries": total_queries,
            "total_feedback_submissions": total_feedback,
            "model": MODEL_NAME,
            "cache_size": len(query_cache.cache),
            "chroma_available": chroma_manager is not None
        }
    except Exception as e:
        return {"error": str(e)}


@app.websocket("/ws/crew")
async def crew_websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for crew dashboard"""
    await crew_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(data)
    except WebSocketDisconnect:
        crew_manager.disconnect(websocket)


@app.post("/api/crew/acknowledge")
async def acknowledge_crew_alert(request: dict):
    """Acknowledge a crew alert"""
    return {"status": "success", "message": "Alert acknowledged"}


@app.get("/crew-dashboard")
async def crew_dashboard():
    """Serve crew dashboard HTML"""
    dashboard_path = os.path.join(STATIC_DIR, "crew-dashboard.html")
    if os.path.exists(dashboard_path):
        return FileResponse(dashboard_path)
    return {"message": "Crew dashboard not found. Create crew-dashboard.html in static directory"}


if __name__ == "__main__":
    uvicorn.run(
        "travel_assistant:app",
        host="0.0.0.0", 
        port=8000,
        reload=True
    )