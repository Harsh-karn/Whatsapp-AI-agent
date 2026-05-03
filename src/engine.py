"""Core deterministic Vera message engine with optional LLM fallback."""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from urllib import request as urlrequest


def utc_now_iso() -> str:
    """Return UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass
class ContextEntry:
    """Versioned context entry."""

    version: int
    payload: dict[str, Any]
    stored_at: str = field(default_factory=utc_now_iso)


@dataclass
class ConversationState:
    """Conversation memory for `/v1/reply` decisions."""

    conversation_id: str
    merchant_id: Optional[str]
    customer_id: Optional[str]
    trigger_id: Optional[str]
    turn_count: int = 0
    auto_reply_count: int = 0
    last_user_message: str = ""
    sent_bodies: list[str] = field(default_factory=list)
    ended: bool = False


class StateStore:
    """In-memory challenge state store."""

    def __init__(self) -> None:
        self.contexts: dict[tuple[str, str], ContextEntry] = {}
        self.conversations: dict[str, ConversationState] = {}
        self.sent_suppressions: set[str] = set()
        self.opted_out_merchants: set[str] = set()
        self.started_at = time.time()

    def context_counts(self) -> dict[str, int]:
        counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
        for (scope, _), _entry in self.contexts.items():
            counts[scope] = counts.get(scope, 0) + 1
        return counts

    def upsert_context(
        self, scope: str, context_id: str, version: int, payload: dict[str, Any]
    ) -> tuple[bool, Optional[int]]:
        key = (scope, context_id)
        current = self.contexts.get(key)
        if current and current.version >= version:
            return False, current.version
        self.contexts[key] = ContextEntry(version=version, payload=payload)
        return True, None

    def get_context(self, scope: str, context_id: str) -> Optional[dict[str, Any]]:
        entry = self.contexts.get((scope, context_id))
        return entry.payload if entry else None

    def get_or_create_conversation(
        self,
        conversation_id: str,
        merchant_id: Optional[str],
        customer_id: Optional[str],
        trigger_id: Optional[str],
    ) -> ConversationState:
        conv = self.conversations.get(conversation_id)
        if conv:
            return conv
        conv = ConversationState(
            conversation_id=conversation_id,
            merchant_id=merchant_id,
            customer_id=customer_id,
            trigger_id=trigger_id,
        )
        self.conversations[conversation_id] = conv
        return conv


class LLMFallback:
    """Small optional OpenAI fallback wrapper with strict deterministic settings."""

    def __init__(self) -> None:
        self.enabled = os.getenv("LLM_FALLBACK_ENABLED", "false").lower() == "true"
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.timeout_seconds = _to_int(os.getenv("LLM_TIMEOUT_SECONDS", "6"), 6)

    def improve(
        self,
        draft_body: str,
        cta: str,
        rationale: str,
        category: dict[str, Any],
        merchant: dict[str, Any],
        trigger: dict[str, Any],
        customer: Optional[dict[str, Any]],
    ) -> Optional[dict[str, str]]:
        if not self.enabled or not self.api_key:
            return None

        prompt = {
            "instruction": (
                "Rewrite the draft into sharper WhatsApp copy while preserving only facts "
                "present in inputs. Keep one CTA. Keep under 420 chars."
            ),
            "draft": {"body": draft_body, "cta": cta, "rationale": rationale},
            "context": {
                "category_slug": category.get("slug"),
                "merchant_name": merchant.get("identity", {}).get("name"),
                "trigger_kind": trigger.get("kind"),
                "trigger_payload": trigger.get("payload", {}),
                "customer": customer or None,
            },
            "output_schema": {"body": "str", "cta": "str", "rationale": "str"},
        }
        body = json.dumps(
            {
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {
                        "role": "system",
                        "content": "Return strict JSON only with keys body, cta, rationale.",
                    },
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                ],
            }
        ).encode("utf-8")
        req = urlrequest.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            resp = urlrequest.urlopen(req, timeout=self.timeout_seconds)
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            payload = json.loads(content)
            if not payload.get("body"):
                return None
            return {
                "body": str(payload.get("body", draft_body)),
                "cta": str(payload.get("cta", cta)),
                "rationale": str(payload.get("rationale", rationale)),
            }
        except Exception:
            return None


class VeraComposer:
    """Deterministic composition engine for merchant and customer messaging."""

    AUTO_REPLY_PATTERNS = [
        re.compile(r"thank you for contacting", re.I),
        re.compile(r"our team will respond", re.I),
        re.compile(r"automated assistant", re.I),
        re.compile(r"auto(?:mated)? reply", re.I),
    ]
    HOSTILE_PATTERNS = [
        re.compile(r"\bstop\b", re.I),
        re.compile(r"spam", re.I),
        re.compile(r"useless", re.I),
        re.compile(r"don't message", re.I),
        re.compile(r"not interested", re.I),
    ]
    COMMIT_PATTERNS = [
        re.compile(r"\blet'?s do it\b", re.I),
        re.compile(r"\bgo ahead\b", re.I),
        re.compile(r"\bok(?:ay)?\b.*\bwhat'?s next\b", re.I),
        re.compile(r"\bconfirm\b", re.I),
        re.compile(r"\byes\b", re.I),
    ]

    def __init__(self, store: StateStore, llm_fallback: LLMFallback) -> None:
        self.store = store
        self.llm_fallback = llm_fallback

    def compose_for_tick(
        self,
        trigger_id: str,
        now_iso: str,
    ) -> Optional[dict[str, Any]]:
        trigger = self.store.get_context("trigger", trigger_id)
        if not trigger:
            return None
        merchant_id = trigger.get("merchant_id")
        if not merchant_id:
            return None
        if merchant_id in self.store.opted_out_merchants:
            return None

        merchant = self.store.get_context("merchant", merchant_id)
        if not merchant:
            return None
        category_slug = merchant.get("category_slug")
        category = self.store.get_context("category", category_slug or "")
        if not category:
            return None
        customer_id = trigger.get("customer_id")
        customer = (
            self.store.get_context("customer", customer_id) if customer_id else None
        )

        suppression_key = str(trigger.get("suppression_key", ""))
        if suppression_key and suppression_key in self.store.sent_suppressions:
            return None

        draft = self._compose_message(category, merchant, trigger, customer)
        improved = self.llm_fallback.improve(
            draft_body=draft["body"],
            cta=draft["cta"],
            rationale=draft["rationale"],
            category=category,
            merchant=merchant,
            trigger=trigger,
            customer=customer,
        )
        if improved:
            draft["body"] = improved["body"]
            draft["cta"] = improved["cta"]
            draft["rationale"] = improved["rationale"]

        conversation_id = (
            f"conv_{merchant_id}_{trigger_id}_{_to_int(time.time() * 1000)}"
        )
        self.store.get_or_create_conversation(
            conversation_id=conversation_id,
            merchant_id=merchant_id,
            customer_id=customer_id,
            trigger_id=trigger_id,
        )
        if suppression_key:
            self.store.sent_suppressions.add(suppression_key)

        send_as = "merchant_on_behalf" if customer else "vera"
        template_name = (
            "merchant_customer_trigger_v1" if customer else "vera_merchant_trigger_v1"
        )
        template_params = self._build_template_params(draft["body"], merchant, customer)

        return {
            "conversation_id": conversation_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": send_as,
            "trigger_id": trigger_id,
            "template_name": template_name,
            "template_params": template_params,
            "body": draft["body"],
            "cta": draft["cta"],
            "suppression_key": suppression_key,
            "rationale": draft["rationale"],
        }

    def handle_reply(
        self,
        conversation_id: str,
        merchant_id: Optional[str],
        customer_id: Optional[str],
        message: str,
    ) -> dict[str, Any]:
        conv = self.store.get_or_create_conversation(
            conversation_id=conversation_id,
            merchant_id=merchant_id,
            customer_id=customer_id,
            trigger_id=None,
        )
        msg = message.strip()
        msg_lower = msg.lower()
        conv.turn_count += 1

        if any(p.search(msg) for p in self.HOSTILE_PATTERNS):
            conv.ended = True
            if merchant_id:
                self.store.opted_out_merchants.add(merchant_id)
            return {
                "action": "end",
                "rationale": "Merchant signaled opt-out/hostility; ending gracefully.",
            }

        if any(p.search(msg) for p in self.AUTO_REPLY_PATTERNS) or msg == conv.last_user_message:
            conv.auto_reply_count += 1
            conv.last_user_message = msg
            if conv.auto_reply_count == 1:
                return {
                    "action": "send",
                    "body": "Looks like an auto-reply. When the owner sees this, reply YES and I will continue from there.",
                    "cta": "binary_yes_no",
                    "rationale": "Detected canned auto-reply; sending one owner-directed prompt.",
                }
            if conv.auto_reply_count == 2:
                return {
                    "action": "wait",
                    "wait_seconds": 14400,
                    "rationale": "Repeated auto-reply pattern; backing off 4 hours.",
                }
            conv.ended = True
            return {
                "action": "end",
                "rationale": "Auto-reply repeated 3+ times with no human engagement.",
            }

        if any(p.search(msg_lower) for p in self.COMMIT_PATTERNS):
            next_body = (
                "Great. I am moving to execution now: I will draft the exact next message "
                "and prep the action pack. Reply CONFIRM to execute, or tell me what to tweak."
            )
            return {
                "action": "send",
                "body": next_body,
                "cta": "binary_confirm_cancel",
                "rationale": "Detected explicit commitment and switched from qualification to action mode.",
            }

        if "gst" in msg_lower:
            return {
                "action": "send",
                "body": "I cannot help with GST filing directly. Staying on your current campaign thread, should I continue with the drafted next step?",
                "cta": "open_ended",
                "rationale": "Out-of-scope ask redirected to the active business context.",
            }

        follow_up = (
            "Understood. I can proceed with a concise next step tailored to your latest trigger. "
            "Reply YES to continue or STOP to end."
        )
        return {
            "action": "send",
            "body": follow_up,
            "cta": "binary_yes_no",
            "rationale": "Default acknowledgment with low-friction continuation choice.",
        }

    def _compose_message(
        self,
        category: dict[str, Any],
        merchant: dict[str, Any],
        trigger: dict[str, Any],
        customer: Optional[dict[str, Any]],
    ) -> dict[str, str]:
        merchant_name = merchant.get("identity", {}).get("name", "there")
        owner_name = merchant.get("identity", {}).get("owner_first_name")
        addressee = owner_name or merchant_name
        trigger_kind = str(trigger.get("kind", "generic"))
        payload = trigger.get("payload", {}) or {}
        signals = merchant.get("signals", [])
        active_offers = [
            offer.get("title")
            for offer in merchant.get("offers", [])
            if str(offer.get("status", "")).lower() == "active"
        ]
        ctr = _to_float(merchant.get("performance", {}).get("ctr"), 0.0)
        peer_ctr = _to_float(category.get("peer_stats", {}).get("avg_ctr"), 0.0)
        city = merchant.get("identity", {}).get("city", "")
        hi_mode = "hi" in merchant.get("identity", {}).get("languages", [])

        prefix = "Hi" if not hi_mode else "Hi,"
        if trigger.get("scope") == "customer" and customer:
            customer_name = customer.get("identity", {}).get("name", "there")
            return self._compose_customer_message(
                customer_name=customer_name,
                trigger_kind=trigger_kind,
                payload=payload,
                merchant_name=merchant_name,
                hi_mode=hi_mode,
                active_offers=active_offers,
            )

        if trigger_kind in {"research_digest", "category_research_digest_release"}:
            top_item = payload.get("top_item", {})
            headline = top_item.get("title") or payload.get("top_item_id") or "new category insight"
            source = top_item.get("source") or "category digest"
            body = (
                f"{prefix} {addressee}, fresh {category.get('slug', 'category')} digest landed: "
                f"{headline}. Source: {source}. "
                f"Want me to turn this into a 4-line customer-ready WhatsApp draft for {merchant_name}?"
            )
            rationale = "Research trigger with source-cited specificity and low-friction next step."
            cta = "open_ended"
        elif trigger_kind in {"perf_dip", "seasonal_perf_dip"}:
            metric = payload.get("metric", "performance")
            delta_pct = _to_float(payload.get("delta_pct"), 0.0) * 100
            body = (
                f"{prefix} {addressee}, quick alert: your {metric} moved {delta_pct:.0f}% this cycle in {city}. "
                f"Current CTR is {ctr:.3f} vs peer {peer_ctr:.3f}. "
                "I can draft one corrective post + one offer message now. Reply YES to proceed."
            )
            rationale = "Perf dip trigger anchored on merchant and peer metrics; proposes immediate corrective action."
            cta = "binary_yes_no"
        elif trigger_kind in {"regulation_change", "supply_alert", "compliance"}:
            deadline = payload.get("deadline_iso") or trigger.get("expires_at", "")
            body = (
                f"{prefix} {addressee}, compliance update for {merchant_name}: "
                f"{json.dumps(payload, ensure_ascii=False)[:120]}... "
                f"Please action before {deadline}. Want me to draft the merchant-safe customer note now?"
            )
            rationale = "Compliance trigger with urgency and execution help."
            cta = "binary_yes_no"
        elif trigger_kind in {"active_planning_intent"}:
            topic = payload.get("intent_topic", "next campaign")
            body = (
                f"{prefix} {addressee}, great call on {topic}. "
                "I have a ready-to-run draft with pricing + CTA flow. "
                "Reply CONFIRM and I will share the final version."
            )
            rationale = "Explicit planning intent should transition directly to action execution."
            cta = "binary_confirm_cancel"
        elif trigger_kind in {"curious_ask_due"}:
            body = (
                f"{prefix} {addressee}, one quick check: what service got most asks this week at {merchant_name}? "
                "Share one line and I will convert it into a Google Post + WhatsApp reply pack."
            )
            rationale = "Curiosity-led engagement prompt with effort externalization."
            cta = "open_ended"
        else:
            offer = active_offers[0] if active_offers else "your top service"
            signal_note = signals[0] if signals else trigger_kind
            body = (
                f"{prefix} {addressee}, trigger update for {merchant_name}: {signal_note}. "
                f"Recommended spotlight: {offer}. "
                "Want me to draft the exact message and schedule suggestion? Reply YES or STOP."
            )
            rationale = "Generic fallback still anchored in merchant signal and active offer."
            cta = "binary_yes_no"

        return {"body": self._sanitize_body(body, category), "cta": cta, "rationale": rationale}

    def _compose_customer_message(
        self,
        customer_name: str,
        trigger_kind: str,
        payload: dict[str, Any],
        merchant_name: str,
        hi_mode: bool,
        active_offers: list[str],
    ) -> dict[str, str]:
        offer = active_offers[0] if active_offers else "special service offer"
        if trigger_kind in {"recall_due", "appointment_tomorrow", "chronic_refill_due"}:
            slots = payload.get("available_slots", [])
            slot_text = ", ".join([slot.get("label", "") for slot in slots[:2]]) if slots else "your preferred evening slot"
            body = (
                f"Hi {customer_name}, {merchant_name} here. Your due service reminder is active. "
                f"Available slots: {slot_text}. {offer}. "
                f"{'Reply 1/2 ya preferred time bhej dein.' if hi_mode else 'Reply 1/2 or share your preferred time.'}"
            )
            cta = "multi_choice_slot"
            rationale = "Customer reminder anchored on due event, concrete slots, and active offer."
        elif trigger_kind in {"customer_lapsed_soft", "customer_lapsed_hard", "trial_followup"}:
            body = (
                f"Hi {customer_name}, {merchant_name} checking in. "
                f"We have a low-commitment restart option ready: {offer}. "
                f"{'Reply YES, no pressure.' if hi_mode else 'Reply YES for a no-pressure restart.'}"
            )
            cta = "binary_yes_no"
            rationale = "Winback nudge with low-friction commitment and non-judgment tone."
        else:
            body = (
                f"Hi {customer_name}, message from {merchant_name}. "
                f"We prepared an update relevant to you. {offer}. Reply YES to continue."
            )
            cta = "binary_yes_no"
            rationale = "Customer-scope fallback with consent-safe continuation prompt."
        return {"body": body, "cta": cta, "rationale": rationale}

    def _sanitize_body(self, body: str, category: dict[str, Any]) -> str:
        taboos = (
            category.get("voice", {}).get("taboos")
            or category.get("voice", {}).get("vocab_taboo")
            or []
        )
        result = body
        for taboo in taboos:
            if isinstance(taboo, str) and taboo:
                result = re.sub(rf"\b{re.escape(taboo)}\b", "", result, flags=re.I)
        result = re.sub(r"\s{2,}", " ", result).strip()
        return result[:450]

    def _build_template_params(
        self,
        body: str,
        merchant: dict[str, Any],
        customer: Optional[dict[str, Any]],
    ) -> list[str]:
        merchant_name = merchant.get("identity", {}).get("name", "Merchant")
        first = customer.get("identity", {}).get("name", "") if customer else merchant_name
        second = merchant_name
        third = body[:160]
        return [first, second, third]


def make_ack_id(context_id: str, version: int) -> str:
    """Build deterministic ack id."""
    token = uuid.uuid5(uuid.NAMESPACE_DNS, f"{context_id}:{version}")
    return f"ack_{token.hex[:12]}"

