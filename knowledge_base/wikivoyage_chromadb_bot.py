import xml.etree.ElementTree as ET
import re
import os
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import List, Dict, Tuple

# Enable offline mode for HuggingFace - use cached models only
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

class WikiVoyageProcessor:
    """Processes WikiVoyage XML dumps and extracts travel content"""
    
    def __init__(self, xml_file_path: str):
        self.xml_file = xml_file_path
        self.pages = []
        
    def parse_xml(self) -> List[Dict]:
        """Parse WikiVoyage XML and extract page content"""
        tree = ET.parse(self.xml_file)
        root = tree.getroot()
        
        # Define namespace
        ns = {'mw': 'http://www.mediawiki.org/xml/export-0.11/'}
        
        pages_data = []
        
        for page in root.findall('.//mw:page', ns):
            title_elem = page.find('mw:title', ns)
            ns_elem = page.find('mw:ns', ns)
            
            # Skip redirects and non-travel content (namespace 0 = main)
            redirect = page.find('mw:redirect', ns)
            if redirect is not None or (ns_elem is not None and ns_elem.text != '0'):
                continue
            
            if title_elem is None:
                continue
            
            title = title_elem.text
            
            # Extract latest revision content
            revision = page.find('.//mw:revision', ns)
            if revision is None:
                continue
            
            text_elem = revision.find('mw:text', ns)
            if text_elem is None or text_elem.text is None:
                continue
            
            raw_text = text_elem.text
            
            # Skip very short articles (likely stubs)
            if len(raw_text) < 200:
                continue
            
            # Clean wikitext markup
            cleaned_text = self._clean_wikitext(raw_text)
            
            pages_data.append({
                'title': title,
                'content': cleaned_text,
                'raw_length': len(raw_text)
            })
        
        self.pages = pages_data
        return pages_data
    
    def _clean_wikitext(self, text: str) -> str:
        """Remove MediaWiki/Wikitext markup"""
        
        # Remove templates like {{...}}
        text = re.sub(r'\{\{[^}]*\}\}', '', text)
        
        # Remove internal links [[...]] but keep text
        text = re.sub(r'\[\[([^\]|]+)\|?([^\]]*)\]\]', r'\2 \1', text)
        
        # Remove external links [... ...]
        text = re.sub(r'\[https?://[^\s\]]+\s+([^\]]+)\]', r'\1', text)
        text = re.sub(r'https?://[^\s]+', '', text)
        
        # Remove categories
        text = re.sub(r'\[\[Category:[^\]]+\]\]', '', text)
        
        # Remove files/images
        text = re.sub(r'\[\[File:[^\]]+\]\]', '', text)
        text = re.sub(r'\[\[Image:[^\]]+\]\]', '', text)
        
        # Convert headings
        text = re.sub(r'===+\s*([^=]+)\s*===+', r'\1', text)
        text = re.sub(r'==\s*([^=]+)\s*==', r'\n## \1\n', text)
        
        # Remove markup tags
        text = re.sub(r"'''([^']+)'''", r'\1', text)  # Bold
        text = re.sub(r"''([^']+)''", r'\1', text)    # Italic
        
        # Remove HTML comments
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def chunk_content(self, min_chunk_size: int = 500, max_chunk_size: int = 1000,
                      overlap: int = 100) -> List[Dict]:
        """Split content into overlapping chunks for better context"""
        
        chunks = []
        
        for page in self.pages:
            title = page['title']
            content = page['content']
            
            # Split by sentences/paragraphs first
            paragraphs = content.split('\n\n')
            
            current_chunk = ""
            
            for para in paragraphs:
                # If adding this paragraph exceeds max_chunk_size, save current chunk
                if len(current_chunk) + len(para) > max_chunk_size and len(current_chunk) > 0:
                    if len(current_chunk) >= min_chunk_size:
                        chunks.append({
                            'title': title,
                            'content': current_chunk.strip(),
                            'source': f"{title} (WikiVoyage)"
                        })
                    # Start new chunk with overlap
                    current_chunk = current_chunk[-overlap:] + "\n\n" + para
                else:
                    current_chunk += "\n\n" + para if current_chunk else para
            
            # Don't forget the last chunk
            if len(current_chunk) >= min_chunk_size:
                chunks.append({
                    'title': title,
                    'content': current_chunk.strip(),
                    'source': f"{title} (WikiVoyage)"
                })
        
        return chunks


class ChromaDBManager:
    """Manages ChromaDB collection for travel knowledge base"""
    
    def __init__(self, db_path: str = "./travel_db", model_name: str = "all-MiniLM-L6-v2", collection_name: str = "wikivoyage_travel"):
        """
        Initialize ChromaDB manager
        
        Args:
            db_path: Path to persist ChromaDB
            model_name: Sentence transformer model for embeddings
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        
        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        
        # Load embedding model
        self.embedder = SentenceTransformer(model_name)
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(self, chunks: List[Dict], batch_size: int = 256, verbose: bool = False) -> int:
        """Add document chunks to ChromaDB.

        Supports batched ingestion, optional verbose progress, and uses the
        configured `SentenceTransformer` to precompute embeddings for faster
        ingestion.

        Args:
            chunks: list of chunk dicts with keys `title`, `content`, `source`.
            batch_size: number of documents to add per batch (default 256).
            verbose: if True, show progress (uses `tqdm` if available).

        Returns:
            Number of documents added.
        """
        # batch_size and verbose are accepted as parameters (see signature)

        total = len(chunks)
        if total == 0:
            return 0

        # Use tqdm if verbose and available
        pbar = None
        try:
            if verbose:
                from tqdm import tqdm
                pbar = tqdm(total=total, desc="Ingesting to Chroma")
        except Exception:
            pbar = None

        # Helper to add a batch
        added = 0
        for i in range(0, total, batch_size):
            batch = chunks[i:i+batch_size]
            ids = [f"doc_{(i+j):06d}" for j in range(len(batch))]
            documents = [c['content'] for c in batch]
            metadatas = [{'title': c['title'], 'source': c['source']} for c in batch]

            # Precompute embeddings using sentence-transformers for better performance
            try:
                embeddings = self.embedder.encode(documents, convert_to_numpy=True)
            except Exception:
                embeddings = None

            add_kwargs = {
                'ids': ids,
                'documents': documents,
                'metadatas': metadatas,
            }
            if embeddings is not None:
                add_kwargs['embeddings'] = embeddings

            # Add to Chroma collection
            self.collection.add(**add_kwargs)

            added += len(batch)
            if pbar:
                pbar.update(len(batch))
            elif verbose:
                print(f"Added {added}/{total} documents to collection")

        if pbar:
            pbar.close()

        return added
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for relevant documents"""
        # Prefer vector search using the sentence-transformers embedder for better relevance
        try:
            query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        except Exception:
            query_embedding = None

        if query_embedding is not None:
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results
            )
        else:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
        
        if not results['documents'] or not results['documents'][0]:
            return []
        
        search_results = []
        for i, doc in enumerate(results['documents'][0]):
            search_results.append({
                'content': doc,
                'title': results['metadatas'][0][i]['title'],
                'source': results['metadatas'][0][i]['source'],
                'distance': results['distances'][0][i] if results['distances'] else None
            })
        
        return search_results
    
    def persist(self):
        """Persist the collection to disk"""
        # No longer needed in Chroma v1.0.0+ - writes are saved instantly
        pass


class TravelChatBot:
    """Offline travel assistant chatbot using ChromaDB"""
    
    def __init__(self, db_manager: ChromaDBManager):
        self.db = db_manager
        self.conversation_history = []
    
    def generate_response(self, user_query: str, n_context: int = 3) -> Dict:
        """Generate response with retrieved context"""
        
        # Search for relevant content
        context_docs = self.db.search(user_query, n_results=n_context)
        
        if not context_docs:
            return {
                'query': user_query,
                'response': "I couldn't find relevant travel information for your query. Try asking about destinations, attractions, getting around, or local tips.",
                'sources': [],
                'context_used': 0
            }
        
        # Build context string
        context = "\n\n---\n\n".join([
            f"**{doc['title']}**:\n{doc['content'][:500]}..." 
            if len(doc['content']) > 500 
            else f"**{doc['title']}**:\n{doc['content']}"
            for doc in context_docs
        ])
        
        # Generate response prompt
        prompt = f"""You are a helpful travel assistant with knowledge about global destinations.
        
Based on the following travel information, answer the user's question concisely and helpfully:

---CONTEXT---
{context}
---END CONTEXT---

User Question: {user_query}

Answer:"""
        
        # Simple response generation (in production, use an LLM)
        response = self._generate_simple_response(user_query, context_docs)
        
        return {
            'query': user_query,
            'response': response,
            'sources': list(set([doc['title'] for doc in context_docs])),
            'context_used': len(context_docs)
        }
    
    def _generate_simple_response(self, query: str, docs: List[Dict]) -> str:
        """Simple response generation without LLM (for offline use)"""
        
        # Extract key info from relevant documents
        response_parts = []
        
        for doc in docs:
            # Check if document is relevant
            content = doc['content'][:800]  # First 800 chars
            response_parts.append(f"From **{doc['title']}**:\n{content}")
        
        # For full LLM integration, use ollama or similar:
        # from ollama import generate
        # response = generate(model='mistral', prompt=prompt)
        
        combined = "\n\n".join(response_parts)
        return f"Based on WikiVoyage information:\n\n{combined}"
    
    def chat(self, user_input: str) -> str:
        """Interactive chat interface"""
        result = self.generate_response(user_input)
        self.conversation_history.append({
            'user': user_input,
            'bot': result
        })
        return result


# module provides WikiVoyageProcessor, ChromaDBManager and TravelChatBot for import/use