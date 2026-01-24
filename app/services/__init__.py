"""
Services package for Travel Assistant.
"""
from .config import (
    OLLAMA_URL, MODEL_NAME, DB_FILE, 
    CHROMA_PERSIST_DIR, EMBEDDING_MODEL, CHROMA_COLLECTION_NAME,
    STATIC_DIR
)
from .models import QueryRequest, FeedbackRequest, CrewAlertRequest
from .language import (
    LANGUAGE_NAMES, SERVICE_MESSAGES, 
    get_language_name, get_service_message, translate_to_english
)
from .service_detection import detect_service_request
from .connections import ConnectionManager, crew_manager
from .chroma_service import (
    ChromaManager, build_context_from_chroma, 
    init_chroma_manager, get_chroma_manager
)
from .llm_service import LlamaInterface, init_llm
from .cache import QueryCache, ConversationHistory, query_cache, conversation_history
from .database import init_db, get_stats, log_query, log_feedback

__all__ = [
    # Config
    'OLLAMA_URL', 'MODEL_NAME', 'DB_FILE',
    'CHROMA_PERSIST_DIR', 'EMBEDDING_MODEL', 'CHROMA_COLLECTION_NAME',
    'STATIC_DIR',
    # Models
    'QueryRequest', 'FeedbackRequest', 'CrewAlertRequest',
    # Language
    'LANGUAGE_NAMES', 'SERVICE_MESSAGES',
    'get_language_name', 'get_service_message', 'translate_to_english',
    # Service Detection
    'detect_service_request',
    # Connections
    'ConnectionManager', 'crew_manager',
    # Chroma
    'ChromaManager', 'build_context_from_chroma',
    'init_chroma_manager', 'get_chroma_manager',
    # LLM
    'LlamaInterface', 'init_llm',
    # Cache
    'QueryCache', 'ConversationHistory', 'query_cache', 'conversation_history',
    # Database
    'init_db', 'get_stats', 'log_query', 'log_feedback',
]
