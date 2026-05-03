import os
import json
from urllib import request, error

def test_gemini_list_models():
    key = os.getenv("LLM_API_KEY", "AIzaSyD09EYFvdnxPxMbtHZ16IIQfnAtzFb_EFc").strip()
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    
    try:
        req = request.Request(url) # GET request
        resp = request.urlopen(req)
        data = json.loads(resp.read().decode())
        
        print("Available Models:")
        for m in data.get("models", []):
            if "generateContent" in m.get("supportedGenerationMethods", []):
                print(f" - {m.get('name')}")
                
    except error.HTTPError as e:
        print(f"HTTP ERROR: {e.code}")
        print(f"Response Body: {e.read().decode()}")
    except Exception as e:
        print(f"OTHER ERROR: {e}")

if __name__ == "__main__":
    test_gemini_list_models()
