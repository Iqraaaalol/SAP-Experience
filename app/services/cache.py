"""
Caching services for queries and conversation history.
"""
from datetime import datetime
from typing import Dict, List
from .config import QUERY_CACHE_TTL, CONVERSATION_TTL, MAX_CONVERSATION_TURNS


class QueryCache:
    """Cache for query responses to reduce LLM calls."""
    
    def __init__(self, ttl_seconds: int = QUERY_CACHE_TTL):
        self.cache = {}
        self.ttl = ttl_seconds
    
    def get(self, key: str):
        """Get cached response."""
        if key in self.cache:
            entry = self.cache[key]
            if datetime.now().timestamp() < entry['expires']:
                return entry['value']
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: str):
        """Cache response."""
        self.cache[key] = {
            'value': value,
            'expires': datetime.now().timestamp() + self.ttl
        }
    
    def generate_key(self, destination: str, query: str) -> str:
        """Generate cache key."""
        return f"query:{destination.lower()}:{hash(query)}"
    
    def clear(self):
        """Clear all cached items."""
        self.cache.clear()
    
    def size(self) -> int:
        """Return number of cached items."""
        return len(self.cache)


class ConversationHistory:
    """Manage conversation history per seat for context-aware responses."""
    
    def __init__(self, max_turns: int = MAX_CONVERSATION_TURNS, ttl_seconds: int = CONVERSATION_TTL):
        self.histories: Dict[str, Dict] = {}  # seat_number -> {messages, last_updated}
        self.max_turns = max_turns  # Maximum conversation turns to keep
        self.ttl = ttl_seconds  # 2 hours default TTL
    
    def _cleanup_expired(self):
        """Remove expired conversation histories."""
        now = datetime.now().timestamp()
        expired = [seat for seat, data in self.histories.items() 
                   if now - data['last_updated'] > self.ttl]
        for seat in expired:
            del self.histories[seat]
            print(f"🧹 Cleared expired conversation history for seat {seat}")
    
    def get_history(self, seat_number: str) -> List[Dict[str, str]]:
        """Get conversation history for a seat."""
        self._cleanup_expired()
        if seat_number in self.histories:
            return self.histories[seat_number]['messages']
        return []
    
    def add_exchange(self, seat_number: str, user_message: str, assistant_response: str):
        """Add a conversation exchange (user query + assistant response)."""
        self._cleanup_expired()
        
        if seat_number not in self.histories:
            self.histories[seat_number] = {
                'messages': [],
                'last_updated': datetime.now().timestamp()
            }
        
        history = self.histories[seat_number]
        history['messages'].append({
            'role': 'user',
            'content': user_message
        })
        history['messages'].append({
            'role': 'assistant', 
            'content': assistant_response
        })
        history['last_updated'] = datetime.now().timestamp()
        
        # Trim to max_turns (each turn = 2 messages: user + assistant)
        max_messages = self.max_turns * 2
        if len(history['messages']) > max_messages:
            history['messages'] = history['messages'][-max_messages:]
        
        print(f"💬 Conversation history for seat {seat_number}: {len(history['messages'])//2} turns")
    
    def clear_history(self, seat_number: str) -> bool:
        """Clear conversation history for a specific seat."""
        if seat_number in self.histories:
            del self.histories[seat_number]
            print(f"🗑️ Cleared conversation history for seat {seat_number}")
            return True
        return False
    
    def clear_all(self) -> int:
        """Clear all conversation histories."""
        count = len(self.histories)
        self.histories.clear()
        print(f"🗑️ Cleared all conversation histories ({count} sessions)")
        return count
    
    def format_for_prompt(self, seat_number: str) -> str:
        """Format conversation history for inclusion in LLM prompt."""
        history = self.get_history(seat_number)
        if not history:
            return ""
        
        formatted_lines = ["\n=== CONVERSATION HISTORY ==="]
        formatted_lines.append("(Review this to understand context for follow-up questions)\n")
        
        turn_num = 1
        for i in range(0, len(history), 2):
            user_msg = history[i] if i < len(history) else None
            assistant_msg = history[i+1] if i+1 < len(history) else None
            
            if user_msg:
                content = user_msg['content'][:400] + "..." if len(user_msg['content']) > 400 else user_msg['content']
                formatted_lines.append(f"Turn {turn_num} - Passenger: {content}")
            if assistant_msg:
                # Show more of the last response since that's what follow-ups refer to
                max_len = 800 if i >= len(history) - 2 else 300
                content = assistant_msg['content'][:max_len] + "..." if len(assistant_msg['content']) > max_len else assistant_msg['content']
                formatted_lines.append(f"Turn {turn_num} - Avia: {content}")
            turn_num += 1
            formatted_lines.append("")  # Empty line between turns
        
        formatted_lines.append("=== END CONVERSATION HISTORY ===")
        formatted_lines.append("")
        formatted_lines.append("FOLLOW-UP HANDLING:")
        formatted_lines.append("- If the new message is short (e.g., 'yes', 'no', 'sure', 'more', 'okay'), it's a response to your LAST message above.")
        formatted_lines.append("- Continue naturally from where the conversation left off.")
        formatted_lines.append("- Reference specific things you mentioned if the passenger agrees or asks for more.\n")
        
        return "\n".join(formatted_lines)
    
    def active_count(self) -> int:
        """Return number of active conversation sessions."""
        self._cleanup_expired()
        return len(self.histories)

query_cache = QueryCache()
conversation_history = ConversationHistory()
