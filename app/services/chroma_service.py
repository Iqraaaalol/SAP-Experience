"""
ChromaDB service for knowledge base queries and context building.
"""
import sys
import os

# Add knowledge_base to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'knowledge_base'))

from wikivoyage_chromadb_bot import ChromaDBManager
from .config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL
from .language import LANGUAGE_NAMES


class ChromaManager:
    """Wrapper around ChromaDBManager for compatibility with travel_assistant."""
    
    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR, 
                 collection_name: str = "wikivoyage_travel", 
                 embedding_model: str = EMBEDDING_MODEL):
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


def build_context_from_chroma(chroma_manager: ChromaManager, 
                               destination: str, 
                               query_english: str = "", 
                               language: str = "en", 
                               conversation_context: str = "", 
                               top_k: int = 5) -> str:
    """Build a model prompt using Chroma KB as the primary source."""
    if not chroma_manager:
        return ""

    try:
        # Use English query combined with destination for better semantic matching
        search_text = f"{destination} {query_english}".strip() if query_english else destination
        kb_docs = chroma_manager.query(search_text, top_k=top_k)
        
        # Fallback to just destination if combined search returns nothing
        if not kb_docs and query_english:
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

        # Get language name for instructions
        language_name = LANGUAGE_NAMES.get(language, 'English')

        context = f"""
        You are a helpful travel assistant on an aircraft providing accurate destination information to passengers.

        CRITICAL LANGUAGE INSTRUCTION:
        - You MUST respond entirely in {language_name}.
        - All text, explanations, and formatting must be in {language_name}.
        - Do not mix languages - use only {language_name} throughout your response.

        CONVERSATION CONTINUITY:
        - You are having an ongoing conversation with this passenger.
        - IMPORTANT: Pay close attention to the conversation history below.
        - If the passenger sends a short reply like "yes", "no", "sure", "tell me more", "that sounds good", etc., look at YOUR PREVIOUS RESPONSE to understand what they're responding to.
        - Reference previous messages when relevant to provide continuity.
        - Address them warmly as a returning conversationalist.
        {conversation_context}

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
        - REMEMBER: Your entire response must be in {language_name}!

        PASSENGER MESSAGE:
        """

        return context
    except Exception as e:
        print(f"Chroma context build error: {e}")
        return ""


# Initialize ChromaDB manager
chroma_manager = None

def init_chroma_manager(persist_dir: str = CHROMA_PERSIST_DIR):
    """Initialize the global chroma manager."""
    global chroma_manager
    try:
        chroma_manager = ChromaManager(persist_dir)
    except Exception as e:
        print(f"ChromaManager instantiation error: {e}")
        chroma_manager = None
    return chroma_manager


def get_chroma_manager():
    """Get the global chroma manager instance."""
    return chroma_manager
