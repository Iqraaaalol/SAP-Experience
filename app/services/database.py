"""
Database initialization and operations.
"""
import os
import hashlib
import secrets
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

        await db.execute("""
            CREATE TABLE IF NOT EXISTS crewCredentials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                passwordHash TEXT NOT NULL,
                passwordSalt TEXT NOT NULL,
                createdAt DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor = await db.execute(
            "SELECT COUNT(*) FROM crewCredentials WHERE username = ?",
            ("admin",)
        )
        admin_exists = (await cursor.fetchone())[0] > 0
        if not admin_exists:
            salt = secrets.token_hex(16)
            password_hash = _hash_password("admin", salt)
            await db.execute(
                "INSERT INTO crewCredentials (username, passwordHash, passwordSalt) VALUES (?, ?, ?)",
                ("admin", password_hash, salt)
            )
            print("Created default crew credential: admin/admin")
        
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


def _hash_password(password: str, salt: str) -> str:
    """Create a deterministic password hash using PBKDF2-HMAC."""
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        120000,
    ).hex()


async def verify_crew_credentials(username: str, password: str) -> bool:
    """Validate crew username/password against stored salted hash."""
    try:
        db = await aiosqlite.connect(DB_FILE)
        cursor = await db.execute(
            "SELECT passwordHash, passwordSalt FROM crewCredentials WHERE username = ?",
            (username,)
        )
        row = await cursor.fetchone()
        await db.close()

        if not row:
            return False

        stored_hash, stored_salt = row
        candidate_hash = _hash_password(password, stored_salt)
        return secrets.compare_digest(stored_hash, candidate_hash)
    except Exception as e:
        print(f"Error verifying crew credentials: {e}")
        return False


