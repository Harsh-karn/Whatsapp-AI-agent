#  Message Engine

## Approach

LLM-powered composition engine that uses the 4-context framework (Category, Merchant, Trigger, Customer) to produce sharp, personalized WhatsApp messages for merchants and their customers.

### Architecture

1. **LLM Composer (primary)**: Sends rich structured prompts to an LLM (Mistral/OpenAI/Groq/DeepSeek) with all 4 contexts, category-specific voice rules, and compulsion lever instructions. Temperature=0 for deterministic output.

2. **Deterministic Fallback**: Template-based composer covering all 15+ trigger kinds. Activates when LLM is unavailable or times out. Ensures the bot never returns empty.

3. **Replay-Hardened Reply Handler**: Pattern-matching for critical scenarios (auto-reply detection, intent transition, hostile handling, off-topic redirect) with LLM fallback for engaged conversations.

### Key Design Decisions

- **Category voice differentiation**: Dentists get clinical/peer tone, salons get warm/friendly, restaurants get operator-to-operator, gyms get coaching, pharmacies get precise/trustworthy.
- **Hindi-English code-mix**: Automatically uses Hinglish when merchant's language list includes "hi".
- **Compulsion levers**: Each message uses 1-2 of: specificity, loss aversion, social proof, effort externalization, curiosity, reciprocity, asking.
- **Taboo sanitization**: Post-composition filter removes category-banned words.
- **Suppression dedup**: Prevents duplicate messages via suppression keys.

### What Would Have Helped Most

- Real production Vera conversation logs for fine-tuning prompt templates
- Actual merchant reply distribution data (% auto-reply, % engaged, % hostile)
- A/B test results on which compulsion levers work best per category

## Quick Start

```bash
pip install -r requirements.txt
```

Set your LLM key in `.env` or environment:
```bash
export LLM_PROVIDER=mistral
export LLM_API_KEY=your_key_here
export LLM_MODEL=mistral-small-latest
```

Run bot:
```bash
uvicorn bot:app --host 0.0.0.0 --port 8080
```

Generate submission:
```bash
python generate_submission.py
```

Run judge simulator:
```bash
python judge_simulator.py
```

## File Structure

```
bot.py                    — FastAPI server (5 endpoints + teardown)
src/engine.py             — Core composer with LLM + fallback
src/llm.py                — Multi-provider LLM client
src/prompts.py            — Prompt templates with category voices
generate_submission.py    — Generates submission.jsonl
submission.jsonl          — 30 test pair compositions
dataset/                  — Base dataset (seeds + expanded)
judge_simulator.py        — Local testing harness
```
