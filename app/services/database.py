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


async def get_stats() -> dict:
    """Get database statistics."""
    try:
        db = await aiosqlite.connect(DB_FILE)
        
        cursor = await db.execute("SELECT COUNT(*) FROM passengerQueries")
        total_queries = (await cursor.fetchone())[0]
        
        cursor = await db.execute("SELECT COUNT(*) FROM comfortFeedback")
        total_feedback = (await cursor.fetchone())[0]
        
        await db.close()
        
        return {
            "total_queries": total_queries,
            "total_feedback_submissions": total_feedback
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


async def log_feedback(seat_number: str, temperature: float, lighting: float, 
                       noise_level: float, overall_comfort: str):
    """Log comfort feedback to the database."""
    try:
        db = await aiosqlite.connect(DB_FILE)
        await db.execute(
            "INSERT INTO comfortFeedback (seatNumber, temperature, lighting, noise_level, overall_comfort) VALUES (?, ?, ?, ?, ?)",
            (seat_number, temperature, lighting, noise_level, overall_comfort)
        )
        await db.commit()
        await db.close()
    except Exception as e:
        print(f"Error logging feedback: {e}")
