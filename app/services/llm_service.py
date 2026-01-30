"""
LLM service using Ollama for response generation.
"""
import asyncio
from ollama import Client
from .config import OLLAMA_URL, MODEL_NAME


# Service tool definition for function calling
SERVICE_TOOL = {
    'type': 'function',
    'function': {
        'name': 'request_service',
        'description': 'Request in-flight service from the cabin crew. Use this when the passenger wants something from the flight attendants like drinks, food, blankets, pillows, medical help, or any other cabin service.',
        'parameters': {
            'type': 'object',
            'properties': {
                'service_type': {
                    'type': 'string',
                    'enum': ['beverage', 'food', 'blanket', 'pillow', 'medical', 'assistance', 'entertainment'],
                    'description': 'The type of service requested'
                },
                'details': {
                    'type': 'string',
                    'description': 'Specific details about the request (e.g., "coffee", "water", "headphones", "feeling dizzy")'
                },
                'priority': {
                    'type': 'string',
                    'enum': ['low', 'medium', 'high'],
                    'description': 'Priority level. Use "high" for medical emergencies, "low" for entertainment, "medium" for everything else.'
                }
            },
            'required': ['service_type', 'details', 'priority']
        }
    }
}


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

    async def detect_service_with_tools(self, query: str, language: str = "en") -> dict | None:
        """
        Use LLM function calling to detect if query is a service request.
        
        Returns:
            dict with service details if service detected, None otherwise
        """
        try:
            loop = asyncio.get_event_loop()
            
            system_prompt = """You are Avia, an assistant on a flydubai aircraft in the air. You have NO knowledge about in-flight services or crew operations.
Your job is to determine if the passenger needs an in-flight service from the cabin crew.

ONLY use the request_service function if the passenger is asking for something the cabin crew can physically provide, such as:
- Beverages (water, coffee, tea, juice, soda, wine, beer)
- Food (meals, snacks, sandwiches)
- Comfort items (blanket, pillow)
- Medical assistance (feeling sick, dizzy, need medication)
- General crew assistance (help with seat, overhead bin)
- Entertainment (headphones, WiFi issues, screen not working)

DO NOT use the function for:
- Questions about destinations, travel tips, culture, or things to do
- General conversation or greetings
- Questions about flight duration, arrival time, or flight info
- Any informational queries

If unsure, DO NOT call the function - just respond normally."""

            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': query}
            ]
            
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat(
                    model=self.model_name,
                    messages=messages,
                    tools=[SERVICE_TOOL],
                    options={"temperature": 0.1},
                    stream=False
                )
            )
            
            # Check if model called the service tool
            if response.message.tool_calls:
                tool_call = response.message.tool_calls[0]
                if tool_call.function.name == 'request_service':
                    args = tool_call.function.arguments
                    print(f"🔧 Tool call detected: {args}")
                    
                    # Handle args whether it's a dict or needs parsing
                    if isinstance(args, str):
                        import json
                        try:
                            args = json.loads(args)
                        except:
                            args = {}
                    
                    # Handle different naming conventions the LLM might use
                    service_type = args.get('service_type') or args.get('type', 'assistance')
                    details = args.get('details') or args.get('description', '')
                    priority = args.get('priority', 'medium')
                    
                    # Format a clean, readable message
                    service_labels = {
                        'beverage': '🍹 Beverage',
                        'food': '🍽️ Food',
                        'blanket': '🛏️ Blanket',
                        'pillow': '🛏️ Pillow',
                        'medical': '🏥 Medical',
                        'assistance': '🆘 Assistance',
                        'entertainment': '🎬 Entertainment'
                    }
                    
                    # Build readable message from details
                    if details:
                        # Capitalize first letter for nicer display
                        message = details.capitalize() if isinstance(details, str) else str(details)
                    else:
                        message = f"{service_labels.get(service_type, service_type)} requested"
                    
                    return {
                        'serviceType': service_type,
                        'details': details,
                        'priority': priority,
                        'message': message
                    }
            
            return None
            
        except Exception as e:
            print(f"Tool detection error: {e}")
            return None


# Initialize LLM interface
def init_llm(ollama_url: str = OLLAMA_URL, model_name: str = MODEL_NAME) -> LlamaInterface:
    """Initialize and return a LlamaInterface instance."""
    print(f"\nInitializing Llama...")
    llm = LlamaInterface(ollama_url, model_name)
    print(f"Llama ready!\n")
    return llm
