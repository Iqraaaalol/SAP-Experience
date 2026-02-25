"""
Database initialization and operations.
"""
import os
import aiosqlite
from .config import DB_FILE


async def init_db():
    """Initialize SQLite database with required tables."""
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
        
        # Conversation history table: store individual messages with role and timestamp
        await db.execute("""
            CREATE TABLE IF NOT EXISTS conversationHistory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                seatNumber TEXT NOT NULL,
                role TEXT NOT NULL, -- 'user' or 'assistant'
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        await db.commit()
        await db.close()
        print("Database initialized")
    except Exception as e:
        print(f"Database error: {e}")


async def get_stats() -> dict:
    """Get database statistics."""
    try:
        db = await aiosqlite.connect(DB_FILE)
        
        cursor = await db.execute("SELECT COUNT(*) FROM passengerQueries")
        total_queries = (await cursor.fetchone())[0]
        
        await db.close()

        return {
            "total_queries": total_queries
        }
    except Exception as e:
        return {"error": str(e)}


async def log_query(seat_number: str, destination: str, query_text: str, response_text: str):
    """Log a passenger query to the database."""
    try:
        db = await aiosqlite.connect(DB_FILE)
        await db.execute(
            "INSERT INTO passengerQueries (seatNumber, destination, queryText, responseText) VALUES (?, ?, ?, ?)",
            (seat_number, destination, query_text, response_text)
        )
        await db.commit()
        await db.close()
    except Exception as e:
        print(f"Error logging query: {e}")


async def log_conversation_message(seat_number: str, role: str, content: str):
    """Persist a single conversation message (user or assistant) with timestamp."""
    try:
        db = await aiosqlite.connect(DB_FILE)
        await db.execute(
            "INSERT INTO conversationHistory (seatNumber, role, content) VALUES (?, ?, ?)",
            (seat_number, role, content)
        )
        await db.commit()
        await db.close()
    except Exception as e:
        print(f"Error logging conversation message: {e}")


