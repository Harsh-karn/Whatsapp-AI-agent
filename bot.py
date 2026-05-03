"""Challenge bot server implementing Vera message engine endpoints."""

from __future__ import annotations

import os
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv optional
from typing import Any, Literal, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

from src.engine import StateStore, VeraComposer, make_ack_id, utc_now_iso
from src.llm import LLMClient

TEAM_NAME = os.getenv("TEAM_NAME", "Team VeraEngine")
TEAM_MEMBERS = os.getenv("TEAM_MEMBERS", "Harsh Karn").split(",")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash-with-deterministic-fallback")
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "candidate@example.com")
VERSION = os.getenv("BOT_VERSION", "2.0.0")
APP_STARTED = time.time()

app = FastAPI(title="Vera Message Engine")
store = StateStore()
llm = LLMClient()
composer = VeraComposer(store=store, llm=llm)


class ContextPushBody(BaseModel):
    scope: Literal["category", "merchant", "customer", "trigger"]
    context_id: str = Field(min_length=1)
    version: int = Field(ge=1)
    payload: dict[str, Any]
    delivered_at: str


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = Field(default_factory=list)


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: Literal["merchant", "customer"]
    message: str = Field(min_length=1)
    received_at: str
    turn_number: int = Field(ge=1)


@app.get("/v1/healthz")
async def healthz() -> dict[str, Any]:
    """Liveness endpoint."""
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - APP_STARTED),
        "contexts_loaded": store.context_counts(),
    }


@app.get("/v1/metadata")
async def metadata() -> dict[str, Any]:
    """Metadata endpoint."""
    return {
        "team_name": TEAM_NAME,
        "team_members": [m.strip() for m in TEAM_MEMBERS if m.strip()],
        "model": MODEL_NAME,
        "approach": "LLM-powered composer (Mistral/OpenAI) with category-voice prompts, "
                    "compulsion levers, Hindi-English code-mix, and deterministic fallback. "
                    "Replay-hardened for auto-reply, intent transition, and hostile handling.",
        "contact_email": CONTACT_EMAIL,
        "version": VERSION,
        "submitted_at": utc_now_iso(),
    }


@app.post("/v1/context")
async def push_context(body: ContextPushBody) -> dict[str, Any]:
    """Receive context update with idempotent version semantics."""
    accepted, current_version = store.upsert_context(
        scope=body.scope, context_id=body.context_id,
        version=body.version, payload=body.payload,
    )
    if not accepted:
        return JSONResponse(
            status_code=409,
            content={"accepted": False, "reason": "stale_version", "current_version": current_version},
        )
    return {"accepted": True, "ack_id": make_ack_id(body.context_id, body.version), "stored_at": utc_now_iso()}


@app.post("/v1/tick")
async def tick(body: TickBody) -> dict[str, Any]:
    """Periodic wake-up endpoint for proactive actions."""
    actions: list[dict[str, Any]] = []
    for trigger_id in body.available_triggers[:20]:
        action = composer.compose_for_tick(trigger_id=trigger_id, now_iso=body.now)
        if action:
            actions.append(action)
    return {"actions": actions}


@app.post("/v1/reply")
async def reply(body: ReplyBody) -> dict[str, Any]:
    """Process simulated merchant/customer reply."""
    decision = composer.handle_reply(
        conversation_id=body.conversation_id, merchant_id=body.merchant_id,
        customer_id=body.customer_id, message=body.message,
    )
    if decision.get("action") == "send" and not decision.get("body"):
        return JSONResponse(status_code=400, content={"accepted": False, "reason": "invalid_send_body"})
    return decision


@app.post("/v1/teardown")
async def teardown() -> dict[str, Any]:
    """Wipe state at end of test (privacy compliance)."""
    store.contexts.clear()
    store.conversations.clear()
    store.sent_suppressions.clear()
    store.opted_out_merchants.clear()
    return {"status": "wiped"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
