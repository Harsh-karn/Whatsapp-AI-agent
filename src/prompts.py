"""Prompt templates for Vera message composition."""

from __future__ import annotations

from typing import Any, Optional


# ---------------------------------------------------------------------------
# Category voice definitions
# ---------------------------------------------------------------------------

CATEGORY_VOICES = {
    "dentists": {
        "tone": "peer/clinical — speak as a fellow dental professional, not a salesperson",
        "style": "Technical terms welcome (fluoride varnish, caries, recall, scaling). "
                 "Use 'Dr.' prefix for owner. Cite journal sources when referencing research.",
        "avoid": "Never say 'cure', 'guaranteed', 'best dentist'. No promotional hype or exclamation marks.",
    },
    "salons": {
        "tone": "warm, friendly, practical — speak as a supportive beauty business ally",
        "style": "Beauty/style vocabulary. Mention specific services and prices (Haircut @ ₹99). "
                 "Be personal and encouraging. Reference seasonal trends (bridal, festive).",
        "avoid": "Don't be overly formal or clinical. No condescending tone.",
    },
    "restaurants": {
        "tone": "operator-to-operator — practical, results-focused, direct",
        "style": "F&B vocabulary (covers, footfall, orders, delivery). Reference real ops data. "
                 "Be direct about business impact and operational wins.",
        "avoid": "Don't lecture on food quality. Don't use fine-dining language for street-food joints.",
    },
    "gyms": {
        "tone": "coaching, motivational — speak as a growth partner who gets fitness business",
        "style": "Fitness vocabulary (members, retention, trials, churn). Be energetic but professional. "
                 "Reference seasonal patterns (Jan resolution, summer body).",
        "avoid": "Don't make personal health claims. Focus on business metrics not individual fitness.",
    },
    "pharmacies": {
        "tone": "trustworthy, precise — speak with authority, accuracy, and care",
        "style": "Pharmaceutical vocabulary (molecules, batches, compliance, refills). "
                 "Be exact with drug names and dates. Reference regulatory bodies when relevant.",
        "avoid": "Never make medical claims or diagnoses. Focus on operational compliance and patient care.",
    },
}


# ---------------------------------------------------------------------------
# System prompt for message composition
# ---------------------------------------------------------------------------

COMPOSE_SYSTEM = """You are Vera, magicpin's AI merchant assistant on WhatsApp. You compose ONE sharp, personalized message.

## YOUR VOICE FOR THIS CATEGORY ({cat_slug}):
- Tone: {voice_tone}
- Style: {voice_style}
- Avoid: {voice_avoid}
- Category taboo words (NEVER use): {taboos}

## COMPULSION LEVERS — use 1-2 per message:
1. Specificity — cite exact numbers, dates, sources from context (e.g. "2,100-patient trial", "CTR 2.1%")
2. Loss aversion — frame as missed opportunity ("6,777 missed searches", "before window closes")
3. Social proof — reference peer benchmarks ("peer avg CTR is 3.0%, yours is 2.1%")
4. Effort externalization — "I've drafted X — just say go" / "2-min setup"
5. Curiosity — open a loop they want to close ("want to see the data?")
6. Reciprocity — "I noticed X about your account, thought you'd want to know"
7. Asking the merchant — "what's your most-asked service this week?"

## HARD RULES:
- Use ONLY facts from the provided context. NEVER fabricate data, citations, or competitor names.
- Use service+price format ("Dental Cleaning @ ₹299") not generic discounts ("10% off").
- ONE primary CTA in the LAST sentence. Don't give multiple options.
- NO long preambles ("I hope you're doing well...").
- NO re-introducing yourself ("Hi, I'm Vera...").
- Keep the message under 400 characters.
- {lang_instruction}

## OUTPUT — return ONLY this JSON, no markdown fences:
{{"body": "the WhatsApp message", "cta": "binary_yes_no|open_ended|none|binary_confirm_cancel|multi_choice_slot", "rationale": "1-2 sentences: what lever you used and why this message should work"}}"""


# ---------------------------------------------------------------------------
# System prompt for reply handling
# ---------------------------------------------------------------------------

REPLY_SYSTEM = """You are Vera, magicpin's AI merchant assistant on WhatsApp. The merchant/customer just replied. Compose your next response.

## RULES:
- If they said YES/agreed, switch to ACTION mode immediately (draft, send, execute). Do NOT ask qualifying questions.
- If they asked a question, answer it using context data only.
- If they shared info, acknowledge and advance the conversation.
- Don't repeat yourself — say something new.
- Keep under 300 characters.
- {lang_instruction}

## OUTPUT — return ONLY this JSON, no markdown fences:
{{"action": "send", "body": "your reply", "cta": "binary_yes_no|open_ended|none", "rationale": "1 sentence"}}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _get_lang_instruction(merchant: dict) -> str:
    """Return language instruction based on merchant's language preference."""
    languages = merchant.get("identity", {}).get("languages", ["en"])
    if "hi" in languages:
        return (
            "Use natural Hindi-English code-mix (Hinglish). Example: "
            "'Aapka CTR 2.1% hai, peer avg 3% hai. Ek post draft karun?'"
        )
    return "Use English."


def build_compose_prompt(
    category: dict, merchant: dict, trigger: dict, customer: Optional[dict]
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for message composition."""

    cat_slug = category.get("slug", "unknown")
    voice = CATEGORY_VOICES.get(cat_slug, CATEGORY_VOICES["restaurants"])
    taboos = category.get("voice", {}).get("taboos", []) or category.get("voice", {}).get("vocab_taboo", [])
    lang_inst = _get_lang_instruction(merchant)

    system = COMPOSE_SYSTEM.format(
        cat_slug=cat_slug,
        voice_tone=voice["tone"],
        voice_style=voice["style"],
        voice_avoid=voice["avoid"],
        taboos=", ".join(taboos) if taboos else "none specified",
        lang_instruction=lang_inst,
    )

    # --- Build rich user prompt with all context data ---
    identity = merchant.get("identity", {})
    perf = merchant.get("performance", {})
    peer = category.get("peer_stats", {})
    signals = merchant.get("signals", [])
    offers = merchant.get("offers", [])
    active_offers = [o.get("title") for o in offers if str(o.get("status", "")).lower() == "active"]
    expired_offers = [o.get("title") for o in offers if str(o.get("status", "")).lower() == "expired"]
    conv_hist = merchant.get("conversation_history", [])
    cust_agg = merchant.get("customer_aggregate", {})
    reviews = merchant.get("review_themes", [])
    digest = category.get("digest", [])
    seasonal = category.get("seasonal_beats", [])
    trends = category.get("trend_signals", [])
    payload = trigger.get("payload", {}) or {}

    # Format digest items
    digest_lines = []
    for d in digest[:4]:
        line = f"  - {d.get('title', '?')} (Source: {d.get('source', '?')}"
        if d.get("trial_n"):
            line += f", N={d['trial_n']}"
        line += ")"
        digest_lines.append(line)

    # Format conversation history
    hist_lines = []
    for h in conv_hist[-3:]:
        hist_lines.append(f"  [{h.get('from', '?')}]: {h.get('body', '')[:120]}")

    # Format review themes
    review_lines = []
    for r in reviews[:3]:
        s = f"  - {r.get('theme', '?')} ({r.get('sentiment', '?')}, {r.get('occurrences_30d', '?')}x/30d)"
        if r.get("common_quote"):
            s += f' — "{r["common_quote"]}"'
        review_lines.append(s)

    user = f"""## MERCHANT:
- Name: {identity.get('name', '?')}
- Owner: {identity.get('owner_first_name', '?')}
- Location: {identity.get('locality', '?')}, {identity.get('city', '?')}
- Languages: {identity.get('languages', ['en'])}
- Verified GBP: {identity.get('verified', False)}
- Subscription: {merchant.get('subscription', {}).get('status', '?')} ({merchant.get('subscription', {}).get('plan', '?')}), {merchant.get('subscription', {}).get('days_remaining', '?')} days left

## PERFORMANCE (30d):
- Views: {perf.get('views', '?')} | Calls: {perf.get('calls', '?')} | Directions: {perf.get('directions', '?')}
- CTR: {perf.get('ctr', '?')} (peer avg: {peer.get('avg_ctr', '?')})
- 7d delta: views {perf.get('delta_7d', {}).get('views_pct', '?')}, calls {perf.get('delta_7d', {}).get('calls_pct', '?')}
- Peer benchmarks: avg rating {peer.get('avg_rating', '?')}, avg reviews {peer.get('avg_reviews', '?')}

## SIGNALS: {signals if signals else 'none'}
## ACTIVE OFFERS: {active_offers if active_offers else 'none'}
## EXPIRED OFFERS: {expired_offers if expired_offers else 'none'}

## CUSTOMER AGGREGATE: {cust_agg}

## REVIEW THEMES:
{chr(10).join(review_lines) if review_lines else '  none'}

## RECENT CONVERSATION:
{chr(10).join(hist_lines) if hist_lines else '  no prior conversation'}

## CATEGORY DIGEST (recent research/news):
{chr(10).join(digest_lines) if digest_lines else '  none available'}

## SEASONAL BEATS: {[f"{s.get('month_range','')}: {s.get('note','')}" for s in seasonal[:3]] if seasonal else 'none'}
## TREND SIGNALS: {[f"{t.get('query','')}: +{_fmt_float(t.get('delta_yoy',0))*100:.0f}% YoY" for t in trends[:3]] if trends else 'none'}

## TRIGGER (the reason for this message):
- Kind: {trigger.get('kind', '?')}
- Source: {trigger.get('source', '?')}
- Scope: {trigger.get('scope', '?')}
- Urgency: {trigger.get('urgency', '?')}/5
- Payload: {_compact_json(payload)}
- Suppression key: {trigger.get('suppression_key', '')}"""

    if customer:
        cust_id = customer.get("identity", {})
        cust_rel = customer.get("relationship", {})
        user += f"""

## CUSTOMER (this message is ON BEHALF of the merchant TO the customer):
- Name: {cust_id.get('name', '?')}
- Language: {cust_id.get('language_pref', 'en')}
- State: {customer.get('state', '?')}
- Visits: {cust_rel.get('visits_total', '?')}, Last visit: {cust_rel.get('last_visit', '?')}
- Services: {cust_rel.get('services_received', [])}
- Preferences: {customer.get('preferences', {})}
- NOTE: Set send_as="merchant_on_behalf". Message appears from the merchant's WhatsApp, not Vera."""

    user += "\n\nCompose the message now. Return ONLY the JSON."
    return system, user


def build_reply_prompt(
    merchant: dict, conversation_turns: list, latest_message: str
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for reply handling."""

    lang_inst = _get_lang_instruction(merchant)
    system = REPLY_SYSTEM.format(lang_instruction=lang_inst)

    identity = merchant.get("identity", {})
    turns_text = "\n".join(
        [f"  [{t.get('from', '?')}]: {t.get('body', '')[:150]}" for t in conversation_turns[-5:]]
    )

    user = f"""## MERCHANT: {identity.get('name', '?')} ({identity.get('locality', '?')}, {identity.get('city', '?')})
## ACTIVE OFFERS: {[o.get('title') for o in merchant.get('offers', []) if str(o.get('status','')).lower() == 'active']}

## CONVERSATION SO FAR:
{turns_text if turns_text.strip() else '  (first interaction)'}

## MERCHANT'S LATEST MESSAGE:
"{latest_message}"

Compose your reply. Return ONLY the JSON."""

    return system, user


def _compact_json(obj: Any) -> str:
    """Compact JSON string, truncated to 300 chars."""
    import json
    s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return s[:300] + "..." if len(s) > 300 else s
