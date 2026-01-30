"""
Configuration settings loaded from environment variables.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Enable offline mode for HuggingFace - use cached models only
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Base directories - use absolute paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # SAP-Experience root
APP_DIR = BASE_DIR / "app"

# Ollama/LLM Configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://192.168.0.179:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2:3b")

# Database Configuration
DB_FILE = os.getenv("DB_FILE", str(BASE_DIR / "data" / "aircraft.db"))

# ChromaDB Configuration
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "knowledge_base" / "chroma_db"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "wikivoyage_travel")

# Static Files Configuration
STATIC_DIR = os.getenv("STATIC_DIR", str(APP_DIR / "static"))

# Cache Configuration
QUERY_CACHE_TTL = int(os.getenv("QUERY_CACHE_TTL", "3600"))  # 1 hour
CONVERSATION_TTL = int(os.getenv("CONVERSATION_TTL", "7200"))  # 2 hours
MAX_CONVERSATION_TURNS = int(os.getenv("MAX_CONVERSATION_TURNS", "10"))
