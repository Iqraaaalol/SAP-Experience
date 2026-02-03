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
            
        }
    }
}

# Default fallback tool to prevent the model from inventing tools.
DEFAULT_TOOL = {
    'type': 'function',
    'function': {
        'name': 'default_tool',
        'description': 'Fallback tool. The model should select this when none of the service tools apply; indicates no crew action is needed.',
        'parameters': {
            'type': 'object',
            'properties': {}
        }
    }
}


class LlamaInterface:
    """Interface to interact with Ollama LLM."""
    
    def __init__(self, ollama_url: str = OLLAMA_URL, model_name: str = MODEL_NAME, max_concurrency: int = 4):
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.client = Client(host=ollama_url)
        self._sema = asyncio.Semaphore(max_concurrency)
        print(f"LlamaInterface initialized")
        print(f"   Model: {self.model_name}")
        print(f"   URL: {self.ollama_url}")
    
    async def generate_response(self, prompt: str, temperature: float = 0.7) -> str:
        """Query model using Ollama library with retries and concurrency control."""
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                print(f"Querying model: {self.model_name} (attempt {attempt})")
                async with self._sema:
                    response = await asyncio.to_thread(
                        lambda: Client(host=self.ollama_url).generate(
                            model=self.model_name,
                            prompt=prompt,
                            options={
                                "temperature": float(temperature),
                                "top_p": 0.6,
                                "top_k": 3
                            },
                            stream=False
                        )
                    )

                # Attempt to extract answer safely
                answer = response.get('response') if isinstance(response, dict) else getattr(response, 'response', None)
                if not answer:
                    answer = str(response)
                answer = answer.strip()
                print(f"Got response: {answer[:120]}")
                return answer

            except Exception as e:
                print(f"generate_response attempt {attempt} failed: {e}")
                if attempt < max_retries:
                    backoff = 0.5 * (2 ** (attempt - 1))
                    await asyncio.sleep(backoff)
                    continue
                return f"Sorry, an error occurred: {str(e)}"

    async def detect_service_with_tools(self, query: str, language: str = "en") -> dict | None:
        """
        Use LLM function calling to detect if query is a service request.
        
        Returns:
            dict with service details if service detected, None otherwise
        """
        try:
            system_prompt = """You are Avia, a helpful assistant on a flydubai aircraft in the air. 

You CAN answer:
- Questions about yourself (your name is Avia)
- General conversation, greetings, and pleasantries
- Travel information, destination tips, and cultural questions
- Flight information, duration, and general aviation topics
- Entertainment and casual chat

You have NO knowledge about THIS SPECIFIC FLIGHT'S services, crew, or real-time operations.

CRITICAL: ONLY use the request_service function when a passenger explicitly requests a physical item or crew assistance:
- Beverages (water, coffee, tea, juice, soda, wine, beer)
- Food (meals, snacks, sandwiches)
- Comfort items (blanket, pillow)
- Medical assistance (feeling sick, need medication)
- Crew help (seat adjustment, overhead bin, call button)
- Entertainment hardware (headphones, screen issues)

DO NOT use request_service for:
- Questions or conversation (even about services)
- "What's your name?" "How are you?" "Thank you" "Goodbye", or questions relating to YOU
- "Tell me about Dubai" or any informational queries
- Anything that doesn't require immediate crew action

DEFAULT BEHAVIOR: Answer directly. Only call the function when the passenger clearly wants something RIGHT NOW from the crew.

Examples:
User: "What's your name?" → Answer: "I'm Avia, your in-flight assistant!"
User: "Can I get some water?" → Call: request_service("water")
User: "What drinks are available?" → Answer: request_service("Information")
User: "I'd like a coffee" → Call: request_service("coffee")
"""

            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': query}
            ]
            
            # Use a small retry loop to avoid transient Ollama runner restarts
            max_retries = 3
            response = None
            for attempt in range(1, max_retries + 1):
                try:
                    print(f"Detect service attempt {attempt}")
                    async with self._sema:
                        response = await asyncio.to_thread(
                            lambda: Client(host=self.ollama_url).chat(
                                model=self.model_name,
                                messages=messages,
                                tools=[SERVICE_TOOL, DEFAULT_TOOL],
                                options={"temperature": 0.1},
                                stream=False
                            )
                        )
                    break
                except Exception as e:
                    print(f"detect_service_with_tools attempt {attempt} failed: {e}")
                    if attempt < max_retries:
                        backoff = 0.5 * (2 ** (attempt - 1))
                        await asyncio.sleep(backoff)
                        continue
                    print(f"Tool detection error after retries: {e}")
                    return None
            
            # Check if model called the service tool
            if response.message.tool_calls:
                tool_call = response.message.tool_calls[0]
                # If the model selected the request_service tool, handle it.
                # If it selected the default_tool or any other tool, treat as no-service.
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
            
                else:
                    # Model chose default or another tool (fallback) — do not treat as service.
                    print(f"[LLM Service] Non-service tool selected: {tool_call.function.name}; treating as no-service")
                    return None

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
