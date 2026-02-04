"""
Services package for Travel Assistant.
"""
from .config import (
    OLLAMA_URL, MODEL_NAME, DB_FILE, 
    CHROMA_PERSIST_DIR, EMBEDDING_MODEL, CHROMA_COLLECTION_NAME,
    STATIC_DIR
)
from .models import QueryRequest, CrewAlertRequest
from .language import (
    LANGUAGE_NAMES, SERVICE_MESSAGES, 
    get_language_name, get_service_message, translate_to_english, translate_to_language
)
from .connections import ConnectionManager, crew_manager
from .chroma_service import (
    ChromaManager, build_context_from_chroma, 
    init_chroma_manager, get_chroma_manager, get_translated_prompt
)
from .llm_service import LlamaInterface, init_llm
from .cache import QueryCache, ConversationHistory, query_cache, conversation_history
from .database import init_db, get_stats, log_query, log_conversation_message

__all__ = [
    # Config
    'OLLAMA_URL', 'MODEL_NAME', 'DB_FILE',
    'CHROMA_PERSIST_DIR', 'EMBEDDING_MODEL', 'CHROMA_COLLECTION_NAME',
    'STATIC_DIR',
    # Models
    'QueryRequest', 'CrewAlertRequest',
    # Language
    'LANGUAGE_NAMES', 'SERVICE_MESSAGES',
    'get_language_name', 'get_service_message', 'translate_to_english', 'translate_to_language',
    # Connections
    'ConnectionManager', 'crew_manager',
    # Chroma
    'ChromaManager', 'build_context_from_chroma',
    'init_chroma_manager', 'get_chroma_manager', 'get_translated_prompt',
    # LLM
    'LlamaInterface', 'init_llm',
    # Cache
    'QueryCache', 'ConversationHistory', 'query_cache', 'conversation_history',
    # Database
    'init_db', 'get_stats', 'log_query', 'log_conversation_message',
]
