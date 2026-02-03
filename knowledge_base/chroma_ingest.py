from wikivoyage_chromadb_bot import WikiVoyageProcessor, ChromaDBManager, TravelChatBot
from dotenv import load_dotenv
from pathlib import Path
import os

# Load env so CHROMA_PERSIST_DIR can be used
load_dotenv()

# Parse your WikiVoyage XML dump
processor = WikiVoyageProcessor("wikivoyage.xml")
pages = processor.parse_xml()  # Extract destination pages
chunks = processor.chunk_content()  # Create searchable chunks

# Use CHROMA_PERSIST_DIR if set, otherwise default to knowledge_base/chroma_db
default_persist_dir = str(Path(__file__).parent / "chroma_db")
persist_dir = os.getenv("CHROMA_PERSIST_DIR", default_persist_dir)

# Initialize knowledge base
db = ChromaDBManager(persist_dir)
# Use larger batch and verbose output for ingestion
db.add_documents(chunks, batch_size=512, verbose=True)
db.persist()  # Save to disk
print("Knowledge base created with", len(chunks), "chunks.")
