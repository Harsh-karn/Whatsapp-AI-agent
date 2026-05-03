"""Core Vera message engine with LLM-powered composition."""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from src.llm import LLMClient
from src.prompts import build_compose_prompt, build_reply_prompt, _get_lang_instruction


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _to_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def make_ack_id(context_id: str, version: int) -> str:
    token = uuid.uuid5(uuid.NAMESPACE_DNS, f"{context_id}:{version}")
    return f"ack_{token.hex[:12]}"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class ContextEntry:
    version: int
    payload: dict[str, Any]
    stored_at: str = field(default_factory=utc_now_iso)


@dataclass
class ConversationState:
    conversation_id: str
    merchant_id: Optional[str]
    customer_id: Optional[str]
    trigger_id: Optional[str]
    turn_count: int = 0
    auto_reply_count: int = 0
    last_user_message: str = ""
    sent_bodies: list[str] = field(default_factory=list)
    turns: list[dict] = field(default_factory=list)
    ended: bool = False


# ---------------------------------------------------------------------------
# State Store
# ---------------------------------------------------------------------------

class StateStore:
    def __init__(self) -> None:
        self.contexts: dict[tuple[str, str], ContextEntry] = {}
        self.conversations: dict[str, ConversationState] = {}
        self.sent_suppressions: set[str] = set()
        self.opted_out_merchants: set[str] = set()
        self.merchant_auto_replies: dict[str, int] = {}
        self.started_at = time.time()

    def context_counts(self) -> dict[str, int]:
        counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
        for (scope, _) in self.contexts:
            counts[scope] = counts.get(scope, 0) + 1
        return counts

    def upsert_context(self, scope: str, context_id: str, version: int, payload: dict) -> tuple[bool, Optional[int]]:
        key = (scope, context_id)
        current = self.contexts.get(key)
        if current and current.version >= version:
            return False, current.version
        self.contexts[key] = ContextEntry(version=version, payload=payload)
        return True, None

    def get_context(self, scope: str, context_id: str) -> Optional[dict[str, Any]]:
        entry = self.contexts.get((scope, context_id))
        return entry.payload if entry else None

    def get_or_create_conversation(self, conversation_id: str, merchant_id: Optional[str],
                                    customer_id: Optional[str], trigger_id: Optional[str]) -> ConversationState:
        conv = self.conversations.get(conversation_id)
        if conv:
            return conv
        conv = ConversationState(
            conversation_id=conversation_id, merchant_id=merchant_id,
            customer_id=customer_id, trigger_id=trigger_id,
        )
        self.conversations[conversation_id] = conv
        return conv


# ---------------------------------------------------------------------------
# Deterministic Fallback Composer
# ---------------------------------------------------------------------------

class DeterministicFallback:
    """Template-based fallback when LLM is unavailable."""

    def compose(self, category: dict, merchant: dict, trigger: dict, customer: Optional[dict]) -> dict:
        identity = merchant.get("identity", {})
        name = identity.get("owner_first_name") or identity.get("name", "there")
        kind = trigger.get("kind", "update")
        payload = trigger.get("payload", {}) or {}
        hi = "hi" in identity.get("languages", [])
        offers = [o.get("title") for o in merchant.get("offers", []) if str(o.get("status", "")).lower() == "active"]
        perf = merchant.get("performance", {})
        peer = category.get("peer_stats", {})

        if customer and trigger.get("scope") == "customer":
            return self._customer_msg(customer, merchant, trigger, hi, offers)

        ctr = _to_float(perf.get("ctr"), 0)
        peer_ctr = _to_float(peer.get("avg_ctr"), 0)
        city = identity.get("city", "")

        if kind in {"research_digest", "category_research_digest_release"}:
            top = payload.get("top_item", {})
            headline = top.get("title") or "new category insight"
            source = top.get("source") or "category digest"
            body = (f"{'Hi' if not hi else 'Hi,'} {name}, fresh {category.get('slug', '')} digest: "
                    f"{headline}. Source: {source}. "
                    f"{'Ek draft banau patient-ed WhatsApp ka?' if hi else 'Want me to draft a patient-ed WhatsApp you can share?'}")
            return {"body": body[:420], "cta": "open_ended", "rationale": "Research digest with source citation and effort externalization."}

        elif kind in {"perf_dip", "seasonal_perf_dip"}:
            delta = _to_float(payload.get("delta_pct"), 0) * 100
            metric = payload.get("metric", "performance")
            body = (f"{'Hi,' if hi else 'Hi'} {name}, alert: {metric} {'gira' if hi else 'dropped'} {abs(delta):.0f}% "
                    f"this week in {city}. CTR {ctr:.3f} vs peer {peer_ctr:.3f}. "
                    f"{'Main ek corrective post + offer draft kar dun? Reply YES.' if hi else 'I can draft a corrective post + offer now. Reply YES.'}")
            return {"body": body[:420], "cta": "binary_yes_no", "rationale": "Perf dip with merchant vs peer metrics and corrective action."}

        elif kind in {"festival_upcoming", "ipl_match_today"}:
            event = payload.get("festival") or payload.get("match") or kind
            body = (f"{'Hi,' if hi else 'Hi'} {name}, {event} "
                    f"{'aa raha hai' if hi else 'is coming up'}. "
                    f"{'Aapke liye ek special campaign draft karun? Reply YES.' if hi else 'Want me to draft a campaign? Reply YES.'}")
            return {"body": body[:420], "cta": "binary_yes_no", "rationale": "Event-driven campaign with effort externalization."}

        elif kind in {"competitor_opened"}:
            comp = payload.get("competitor_name", "A new competitor")
            dist = payload.get("distance_km", "?")
            body = (f"{'Hi,' if hi else 'Hi'} {name}, {comp} opened {dist}km away. "
                    f"{'Aapka current offer: ' if hi else 'Your active offer: '}{offers[0] if offers else 'none'}. "
                    f"{'Visibility boost plan banau? Reply YES.' if hi else 'Want me to plan a visibility boost? Reply YES.'}")
            return {"body": body[:420], "cta": "binary_yes_no", "rationale": "Competitor alert with loss aversion."}

        elif kind in {"milestone_reached"}:
            val = payload.get("value_now", "?")
            milestone = payload.get("milestone_value", "?")
            body = (f"{'Hi,' if hi else 'Hi'} {name}, you're at {val} reviews — {milestone} milestone "
                    f"{'ke karib! Ek celebration post draft karun?' if hi else 'is close! Want me to draft a celebration post?'}")
            return {"body": body[:420], "cta": "open_ended", "rationale": "Milestone proximity with curiosity and social proof."}

        elif kind in {"review_theme_emerged"}:
            theme = payload.get("theme", "feedback")
            count = payload.get("occurrences_30d", "?")
            quote = payload.get("common_quote", "")
            body = (f"{'Hi,' if hi else 'Hi'} {name}, {count} reviews in 30d mention \"{theme}\""
                    f"{f' — \"{quote}\"' if quote else ''}. "
                    f"{'Ek response template banau? Reply YES.' if hi else 'Want me to draft a response template? Reply YES.'}")
            return {"body": body[:420], "cta": "binary_yes_no", "rationale": "Review theme alert with specificity."}

        elif kind in {"renewal_due"}:
            days = payload.get("days_remaining", "?")
            body = (f"{'Hi,' if hi else 'Hi'} {name}, subscription {days} din mein expire hogi. "
                    f"Views: {perf.get('views', '?')}, Calls: {perf.get('calls', '?')}. "
                    f"{'Renew karein? Reply YES.' if hi else 'Renew now? Reply YES.'}")
            return {"body": body[:420], "cta": "binary_yes_no", "rationale": "Renewal urgency with performance recap."}

        elif kind in {"dormant_with_vera"}:
            days_dormant = payload.get("days_since_last_merchant_message", "?")
            body = (f"{'Hi,' if hi else 'Hi'} {name}, {days_dormant} din ho gaye! "
                    f"{'Aapka profile check kiya — kuch updates hain. 2 min lagenge. Dekhein?' if hi else 'Checked your profile — have some updates. Takes 2 min. Want to see?'}")
            return {"body": body[:420], "cta": "open_ended", "rationale": "Re-engagement with curiosity and low time commitment."}

        elif kind in {"regulation_change", "supply_alert", "compliance"}:
            deadline = payload.get("deadline_iso") or trigger.get("expires_at", "")
            body = (f"{'Hi,' if hi else 'Hi'} {name}, compliance update: "
                    f"{json.dumps(payload, ensure_ascii=False)[:150]}. "
                    f"{'Action before' if not hi else 'Deadline:'} {deadline[:10]}. "
                    f"{'Customer note draft karun? Reply YES.' if hi else 'Want me to draft customer note? Reply YES.'}")
            return {"body": body[:420], "cta": "binary_yes_no", "rationale": "Compliance urgency with deadline and effort externalization."}

        elif kind == "active_planning_intent":
            topic = payload.get("intent_topic", "next campaign")
            body = (f"{'Hi,' if hi else 'Hi'} {name}, {topic} ke liye draft ready hai. "
                    f"{'Reply CONFIRM to execute.' if not hi else 'Reply CONFIRM, main bhej deta/deti hoon.'}")
            return {"body": body[:420], "cta": "binary_confirm_cancel", "rationale": "Planning intent switched to action execution."}

        elif kind == "curious_ask_due":
            body = (f"{'Hi,' if hi else 'Hi'} {name}, quick check: "
                    f"{'is hafte sabse zyada kaunsi service puchi gayi?' if hi else 'what service got most asks this week?'} "
                    f"{'Batao, main uska Google Post + WhatsApp pack bana dunga.' if hi else 'Share one line and I will convert it into a post + reply pack.'}")
            return {"body": body[:420], "cta": "open_ended", "rationale": "Curiosity lever with effort externalization."}

        elif kind in {"gbp_unverified"}:
            uplift = _to_float(payload.get("estimated_uplift_pct"), 0) * 100
            body = (f"{'Hi,' if hi else 'Hi'} {name}, aapka Google profile unverified hai. "
                    f"Verify karne se ~{uplift:.0f}% zyada visibility mil sakti hai. "
                    f"{'Main guide kar dun? Reply YES.' if hi else '5-min process. Want me to guide you? Reply YES.'}")
            return {"body": body[:420], "cta": "binary_yes_no", "rationale": "GBP verification with quantified uplift."}

        elif kind == "perf_spike":
            metric = payload.get("metric", "performance")
            delta = _to_float(payload.get("delta_pct"), 0) * 100
            driver = payload.get("likely_driver", "")
            body = (f"{'Hi,' if hi else 'Hi'} {name}, your {metric} {'badha' if hi else 'jumped'} +{delta:.0f}% this week"
                    f"{f' — likely from {driver}' if driver else ''}. "
                    f"{'Momentum build karun? Reply YES.' if hi else 'Want me to build on this momentum? Reply YES.'}")
            return {"body": body[:420], "cta": "binary_yes_no", "rationale": "Perf spike celebration with momentum continuation."}

        # Generic fallback
        offer = offers[0] if offers else "your top service"
        signal = signals[0] if (signals := merchant.get("signals", [])) else kind
        body = (f"{'Hi,' if hi else 'Hi'} {name}, update: {signal}. "
                f"Recommended spotlight: {offer}. "
                f"{'Draft bhejun? Reply YES ya STOP.' if hi else 'Want me to draft the message? Reply YES or STOP.'}")
        return {"body": body[:420], "cta": "binary_yes_no", "rationale": "Fallback anchored on merchant signal and active offer."}

    def _customer_msg(self, customer: dict, merchant: dict, trigger: dict, hi: bool, offers: list) -> dict:
        cname = customer.get("identity", {}).get("name", "there")
        mname = merchant.get("identity", {}).get("name", "")
        kind = trigger.get("kind", "")
        payload = trigger.get("payload", {}) or {}
        offer = offers[0] if offers else "special service"
        lang_pref = customer.get("identity", {}).get("language_pref", "en")
        use_hi = "hi" in lang_pref

        if kind in {"recall_due", "appointment_tomorrow", "chronic_refill_due"}:
            slots = payload.get("available_slots", [])
            slot_text = ", ".join([s.get("label", "") for s in slots[:2]]) if slots else "your preferred slot"
            body = (f"Hi {cname}, {mname} here. "
                    f"{'Aapki due service reminder active hai.' if use_hi else 'Your due service reminder is active.'} "
                    f"Slots: {slot_text}. {offer}. "
                    f"{'Reply 1/2 ya preferred time bhejein.' if use_hi else 'Reply 1/2 or share preferred time.'}")
            return {"body": body[:420], "cta": "multi_choice_slot", "rationale": "Customer recall with concrete slots and offer."}

        elif kind in {"customer_lapsed_soft", "customer_lapsed_hard", "trial_followup"}:
            body = (f"Hi {cname}, {mname} checking in. "
                    f"{'Ek low-commitment restart option ready hai:' if use_hi else 'We have a restart option:'} {offer}. "
                    f"{'Reply YES, no pressure.' if use_hi else 'Reply YES for a no-pressure restart.'}")
            return {"body": body[:420], "cta": "binary_yes_no", "rationale": "Winback with low-friction commitment."}

        body = (f"Hi {cname}, {mname} se update. {offer}. "
                f"{'Reply YES to continue.' if not use_hi else 'Reply YES.'}")
        return {"body": body[:420], "cta": "binary_yes_no", "rationale": "Customer fallback with offer."}


# ---------------------------------------------------------------------------
# Main Composer
# ---------------------------------------------------------------------------

class VeraComposer:
    """LLM-powered composition engine with deterministic fallback."""

    AUTO_REPLY_PATTERNS = [
        re.compile(r"thank you for contacting", re.I),
        re.compile(r"our team will respond", re.I),
        re.compile(r"automated assistant", re.I),
        re.compile(r"auto(?:mated)? reply", re.I),
    ]
    HOSTILE_PATTERNS = [
        re.compile(r"\bstop\b", re.I), re.compile(r"spam", re.I),
        re.compile(r"useless", re.I), re.compile(r"don't message", re.I),
        re.compile(r"not interested", re.I),
    ]
    COMMIT_PATTERNS = [
        re.compile(r"\blet'?s do it\b", re.I), re.compile(r"\bgo ahead\b", re.I),
        re.compile(r"\bok(?:ay)?\b.*\bwhat'?s next\b", re.I),
        re.compile(r"\bconfirm\b", re.I), re.compile(r"\byes\b", re.I),
    ]

    def __init__(self, store: StateStore, llm: LLMClient) -> None:
        self.store = store
        self.llm = llm
        self.fallback = DeterministicFallback()

    def compose_for_tick(self, trigger_id: str, now_iso: str) -> Optional[dict[str, Any]]:
        trigger = self.store.get_context("trigger", trigger_id)
        if not trigger:
            return None
        merchant_id = trigger.get("merchant_id")
        if not merchant_id or merchant_id in self.store.opted_out_merchants:
            return None

        merchant = self.store.get_context("merchant", merchant_id)
        if not merchant:
            return None
        category = self.store.get_context("category", merchant.get("category_slug", ""))
        if not category:
            return None
        customer_id = trigger.get("customer_id")
        customer = self.store.get_context("customer", customer_id) if customer_id else None

        suppression_key = str(trigger.get("suppression_key", ""))
        if suppression_key and suppression_key in self.store.sent_suppressions:
            return None

        # Try LLM composition first, fall back to deterministic
        draft = self._llm_compose(category, merchant, trigger, customer)
        if not draft:
            draft = self.fallback.compose(category, merchant, trigger, customer)

        # Sanitize against taboos
        draft["body"] = self._sanitize(draft["body"], category)

        conversation_id = f"conv_{merchant_id}_{trigger_id}_{_to_int(time.time() * 1000)}"
        self.store.get_or_create_conversation(conversation_id, merchant_id, customer_id, trigger_id)
        if suppression_key:
            self.store.sent_suppressions.add(suppression_key)

        send_as = "merchant_on_behalf" if customer else "vera"
        mname = merchant.get("identity", {}).get("name", "Merchant")

        return {
            "conversation_id": conversation_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": send_as,
            "trigger_id": trigger_id,
            "template_name": "vera_llm_v1" if not customer else "merchant_customer_v1",
            "template_params": [mname, draft["body"][:160]],
            "body": draft["body"],
            "cta": draft.get("cta", "open_ended"),
            "suppression_key": suppression_key,
            "rationale": draft.get("rationale", ""),
        }

    def handle_reply(self, conversation_id: str, merchant_id: Optional[str],
                     customer_id: Optional[str], message: str) -> dict[str, Any]:
        conv = self.store.get_or_create_conversation(conversation_id, merchant_id, customer_id, None)
        msg = message.strip()
        conv.turn_count += 1
        conv.turns.append({"from": "merchant", "body": msg})

        # --- Deterministic pattern checks (critical reliability) ---
        if any(p.search(msg) for p in self.HOSTILE_PATTERNS):
            conv.ended = True
            if merchant_id:
                self.store.opted_out_merchants.add(merchant_id)
            return {"action": "end", "rationale": "Merchant signaled opt-out; ending gracefully."}

        if any(p.search(msg) for p in self.AUTO_REPLY_PATTERNS) or (conv.last_user_message and msg == conv.last_user_message):
            conv.last_user_message = msg
            if merchant_id:
                count = self.store.merchant_auto_replies.get(merchant_id, 0) + 1
                self.store.merchant_auto_replies[merchant_id] = count
            else:
                conv.auto_reply_count += 1
                count = conv.auto_reply_count
                
            if count == 1:
                body = "Looks like an auto-reply. Jab owner dekhein, reply YES and I'll continue."
                conv.turns.append({"from": "vera", "body": body})
                return {"action": "send", "body": body, "cta": "binary_yes_no",
                        "rationale": "Auto-reply detected; one owner-directed prompt."}
            if count == 2:
                return {"action": "wait", "wait_seconds": 14400,
                        "rationale": "Repeated auto-reply; backing off 4 hours."}
            conv.ended = True
            return {"action": "end", "rationale": "Auto-reply 3+ times; no human engagement."}

        if any(p.search(msg.lower()) for p in self.COMMIT_PATTERNS):
            hi = self._is_hindi(merchant_id)
            body = ("Done. Main action mode mein hoon — exact draft aur execution plan ready kar raha hoon. "
                    "Reply CONFIRM to execute, ya batao kya tweak karna hai." if hi else
                    "Done. Switching to action mode — drafting the exact message and prep the action pack. "
                    "Reply CONFIRM to execute, or tell me what to tweak.")
            conv.turns.append({"from": "vera", "body": body})
            return {"action": "send", "body": body, "cta": "binary_confirm_cancel",
                    "rationale": "Commitment detected; switched to action mode immediately."}

        if "gst" in msg.lower():
            body = ("GST filing mein directly help nahi kar sakta. Aapke campaign thread pe chalte hain — "
                    "next step draft karun?" if self._is_hindi(merchant_id) else
                    "Can't help with GST directly. Staying on your campaign — shall I draft the next step?")
            conv.turns.append({"from": "vera", "body": body})
            return {"action": "send", "body": body, "cta": "open_ended",
                    "rationale": "Off-topic redirected to active business context."}

        conv.last_user_message = msg

        # --- LLM-powered reply for engaged conversations ---
        merchant = self.store.get_context("merchant", merchant_id) if merchant_id else None
        if merchant and self.llm.enabled:
            result = self._llm_reply(merchant, conv.turns, msg)
            if result:
                conv.turns.append({"from": "vera", "body": result.get("body", "")})
                return result

        # Deterministic fallback reply
        hi = self._is_hindi(merchant_id)
        body = ("Samajh gaya. Aapke trigger ke hisaab se next step ready hai. "
                "Reply YES to continue ya STOP to end." if hi else
                "Got it. Next step tailored to your latest update is ready. "
                "Reply YES to continue or STOP to end.")
        conv.turns.append({"from": "vera", "body": body})
        return {"action": "send", "body": body, "cta": "binary_yes_no",
                "rationale": "Default acknowledgment with continuation choice."}

    # --- Private methods ---

    def _llm_compose(self, category: dict, merchant: dict, trigger: dict,
                     customer: Optional[dict]) -> Optional[dict]:
        if not self.llm.enabled:
            return None
        system, user = build_compose_prompt(category, merchant, trigger, customer)
        result = self.llm.complete_json(system, user)
        if result and result.get("body"):
            return {
                "body": str(result["body"])[:450],
                "cta": str(result.get("cta", "open_ended")),
                "rationale": str(result.get("rationale", "")),
            }
        return None

    def _llm_reply(self, merchant: dict, turns: list, message: str) -> Optional[dict]:
        system, user = build_reply_prompt(merchant, turns, message)
        result = self.llm.complete_json(system, user)
        if result and result.get("body"):
            return {
                "action": result.get("action", "send"),
                "body": str(result["body"])[:400],
                "cta": str(result.get("cta", "open_ended")),
                "rationale": str(result.get("rationale", "")),
            }
        return None

    def _is_hindi(self, merchant_id: Optional[str]) -> bool:
        if not merchant_id:
            return False
        merchant = self.store.get_context("merchant", merchant_id)
        if not merchant:
            return False
        return "hi" in merchant.get("identity", {}).get("languages", [])

    def _sanitize(self, body: str, category: dict) -> str:
        taboos = category.get("voice", {}).get("taboos", []) or category.get("voice", {}).get("vocab_taboo", [])
        result = body
        for t in taboos:
            if isinstance(t, str) and t:
                result = re.sub(rf"\b{re.escape(t)}\b", "", result, flags=re.I)
        return re.sub(r"\s{2,}", " ", result).strip()[:450]


# Keep backward compatibility alias
LLMFallback = LLMClient
