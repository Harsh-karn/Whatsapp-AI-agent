import os
import json
from src.llm import LLMClient

# Ensure key is loaded
from dotenv import load_dotenv
load_dotenv()

def test_gemini():
    client = LLMClient()
    print(f"Provider: {client.provider}")
    print(f"Model: {client.model}")
    print(f"Key preview: {client.api_key[:5]}...{client.api_key[-5:] if client.api_key else ''}")
    
    res = client.complete("You are a helpful assistant.", "Say 'Gemini works!'")
    print(f"\nResult: {res}")

if __name__ == "__main__":
    test_gemini()
