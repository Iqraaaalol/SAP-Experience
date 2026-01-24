"""
LLM service using Ollama for response generation.
"""
import asyncio
from ollama import Client
from .config import OLLAMA_URL, MODEL_NAME


class LlamaInterface:
    """Interface to interact with Ollama LLM."""
    
    def __init__(self, ollama_url: str = OLLAMA_URL, model_name: str = MODEL_NAME):
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.client = Client(host=ollama_url)
        print(f"LlamaInterface initialized")
        print(f"   Model: {self.model_name}")
        print(f"   URL: {self.ollama_url}")
    
    async def generate_response(self, prompt: str, temperature: float = 0.7) -> str:
        """Query model using Ollama library."""
        try:
            print(f"Querying model: {self.model_name}")
            
            loop = asyncio.get_event_loop()
            
            response = await loop.run_in_executor(
                None,
                lambda: self.client.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options={
                        "temperature": 0.1,        
                        "top_p": 0.6,
                        "top_k": 3
                    },
                    keep_alive=-1,
                    stream=False
                )
            )
            
            answer = response['response'].strip()
            print(f"Got response: {answer[:50]}...")
            return answer
        
        except Exception as e:
            print(f"Error: {e}")
            return f"Sorry, an error occurred: {str(e)}"


# Initialize LLM interface
def init_llm(ollama_url: str = OLLAMA_URL, model_name: str = MODEL_NAME) -> LlamaInterface:
    """Initialize and return a LlamaInterface instance."""
    print(f"\nInitializing Llama...")
    llm = LlamaInterface(ollama_url, model_name)
    print(f"Llama ready!\n")
    return llm
