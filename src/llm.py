"""Multi-provider LLM client for Vera message composition."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Optional
from urllib import request as urlrequest
from urllib import error as urlerror


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class LLMClient:
    """LLM client supporting Gemini, Mistral, OpenAI, Groq, DeepSeek."""

    OPENAI_ENDPOINTS = {
        "mistral": "https://api.mistral.ai/v1/chat/completions",
        "openai": "https://api.openai.com/v1/chat/completions",
        "groq": "https://api.groq.com/openai/v1/chat/completions",
        "deepseek": "https://api.deepseek.com/v1/chat/completions",
    }

    DEFAULTS = {
        "gemini": "gemini-2.5-flash",
        "mistral": "mistral-small-latest",
        "openai": "gpt-4o-mini",
        "groq": "llama-3.1-70b-versatile",
        "deepseek": "deepseek-chat",
    }

    def __init__(self) -> None:
        self.provider = os.getenv("LLM_PROVIDER", "gemini").strip()
        self.api_key = os.getenv("LLM_API_KEY", "").strip()
        self.model = os.getenv("LLM_MODEL", self.DEFAULTS.get(self.provider, "gemini-2.5-flash")).strip()
        self.timeout = _to_int(os.getenv("LLM_TIMEOUT_SECONDS", "30"), 30)
        self.enabled = bool(self.api_key)

    def complete(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Call the LLM and return raw text response. Returns None on failure."""
        if not self.enabled:
            return None
        if self.provider == "gemini":
            return self._complete_gemini(system_prompt, user_prompt)
        return self._complete_openai(system_prompt, user_prompt)

    def _complete_gemini(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Call Google Gemini API with broad model fallback."""
        model_candidates = [
            self.model,
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
            "gemini-flash-latest",
        ]
        api_versions = ["v1beta", "v1"]

        # Combine system + user into one prompt for maximum compatibility
        full_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"

        body = json.dumps({
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": 800,
            },
        }).encode("utf-8")

        tried = set()
        last_error = None

        for api_ver in api_versions:
            for model_name in model_candidates:
                cache_key = f"{api_ver}/{model_name}"
                if cache_key in tried:
                    continue
                tried.add(cache_key)

                try:
                    url = (
                        f"https://generativelanguage.googleapis.com/{api_ver}/models/"
                        f"{model_name}:generateContent?key={self.api_key}"
                    )
                    req = urlrequest.Request(
                        url, data=body, headers={"Content-Type": "application/json"}
                    )
                    resp = urlrequest.urlopen(req, timeout=self.timeout)
                    data = json.loads(resp.read().decode("utf-8"))

                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            text = parts[0].get("text", "")
                            if text:
                                print(f"[LLM] Gemini OK: {api_ver}/{model_name}")
                                return text

                except urlerror.HTTPError as e:
                    last_error = f"{api_ver}/{model_name}: HTTP {e.code}"
                    if e.code == 429:
                        print(f"[LLM] Gemini rate limit hit (429). Waiting 15 seconds...")
                        time.sleep(15)
                        tried.remove(cache_key) # Allow retry
                    continue
                except Exception as e:
                    last_error = f"{api_ver}/{model_name}: {e}"
                    continue

        print(f"[LLM] Gemini all models failed. Last: {last_error}")
        return None

    def _complete_openai(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Call OpenAI-compatible API (Mistral, OpenAI, Groq, DeepSeek)."""
        endpoint = self.OPENAI_ENDPOINTS.get(self.provider, self.OPENAI_ENDPOINTS["mistral"])
        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "max_tokens": 800,
        }).encode("utf-8")

        req = urlrequest.Request(
            endpoint, data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        try:
            resp = urlrequest.urlopen(req, timeout=self.timeout)
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[LLM] {self.provider} error: {e}")
            return None

    def complete_json(self, system_prompt: str, user_prompt: str) -> Optional[dict]:
        """Call the LLM and parse the response as JSON."""
        raw = self.complete(system_prompt, user_prompt)
        if not raw:
            return None
        # Try direct parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        # Extract JSON from markdown fences or raw text
        try:
            # Remove markdown code fences if present
            cleaned = re.sub(r"```json\s*", "", raw)
            cleaned = re.sub(r"```\s*", "", cleaned)
            match = re.search(r"\{[\s\S]*\}", cleaned)
            if match:
                return json.loads(match.group())
        except (json.JSONDecodeError, Exception) as e:
            print(f"[LLM] JSON parse error: {e}")
        return None
