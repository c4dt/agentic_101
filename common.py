from dotenv import load_dotenv
from agno.models.lmstudio import LMStudio
from agno.models.openai import OpenAIChat
from agno.models.openai.like import OpenAILike
from agno.models.anthropic import Claude
import os
import json
import httpx

load_dotenv()
if os.environ.get("ANTHROPIC_API_KEY", "0") != "0":
  model=Claude(id="claude-3-7-sonnet-latest")
elif os.environ.get("OPENAI_API_KEY", "0") != "0":
  model=OpenAIChat(id="gpt-4.1")
elif os.environ.get("OPENAI_LIKE", "0") != "0":
  model=OpenAILike(api_key=os.getenv("OPENAI_LIKE"),
		id="c4dt",
		base_url="http://localhost:3001/api/v1/openai")
else:
  model=LMStudio()

def cache_to_file(func):
    """Decorator to cache function results to a local file."""
    
    def wrapper(*args):
        cache_file = f"{func.__name__}_cache.json"
        # Load cache from file if it exists
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                cache = json.load(f)
        else:
            cache = {}
        
        # Check if result is already cached
        key = str(args)
        if key in cache:
            return cache[key]
        
        # Call the function and cache the result
        result = func(*args)
        cache[key] = result
        with open(cache_file, "w") as f:
            json.dump(cache, f)
        return result
    
    return wrapper

def get_response_cached(url: str) -> httpx.Response:
    if not (url.startswith("https://") or url.startswith("http://")):
        url = f"https://{url}"
    
    response = httpx.get(url)
    response.raise_for_status()
    return response
    
@cache_to_file
def get_url_cached(url: str) -> str:
    return get_response_cached(url).text

@cache_to_file
def get_json_cached(url: str) -> str:
    return get_response_cached(url).json()

