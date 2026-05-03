import os
import json
from urllib import request, error

def test_gemini_2_0():
    key = os.getenv("LLM_API_KEY", "AIzaSyD09EYFvdnxPxMbtHZ16IIQfnAtzFb_EFc").strip()
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={key}"
    body = json.dumps({
        "contents": [{"parts": [{"text": "Hello"}]}]
    }).encode("utf-8")
    
    req = request.Request(url, data=body, headers={"Content-Type": "application/json"})
    
    try:
        resp = request.urlopen(req)
        print("SUCCESS!")
        print(resp.read().decode())
    except error.HTTPError as e:
        print(f"HTTP ERROR: {e.code}")
        print(f"Response Body: {e.read().decode()}")
    except Exception as e:
        print(f"OTHER ERROR: {e}")

if __name__ == "__main__":
    test_gemini_2_0()
