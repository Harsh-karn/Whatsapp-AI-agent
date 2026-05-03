#!/usr/bin/env python3
"""Generate submission.jsonl by composing messages for test pairs."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.engine import StateStore, VeraComposer
from src.llm import LLMClient


def load_dataset(dataset_dir: Path):
    """Load dataset from seed files or expanded directory."""
    categories = {}
    merchants = {}
    customers = {}
    triggers = {}

    # Try expanded first, fall back to seeds
    expanded = dataset_dir / "expanded"
    if expanded.exists():
        for f in (expanded / "categories").glob("*.json"):
            data = json.load(open(f))
            categories[data.get("slug", f.stem)] = data
        for f in (expanded / "merchants").glob("*.json"):
            data = json.load(open(f))
            merchants[data["merchant_id"]] = data
        for f in (expanded / "customers").glob("*.json"):
            data = json.load(open(f))
            customers[data["customer_id"]] = data
        for f in (expanded / "triggers").glob("*.json"):
            data = json.load(open(f))
            triggers[data["id"]] = data
        return categories, merchants, customers, triggers

    # Seed files
    cat_dir = dataset_dir / "categories"
    if cat_dir.exists():
        for f in cat_dir.glob("*.json"):
            data = json.load(open(f))
            categories[data.get("slug", f.stem)] = data

    for name, key in [("merchants_seed.json", "merchant_id"), ("customers_seed.json", "customer_id")]:
        path = dataset_dir / name
        if path.exists():
            data = json.load(open(path))
            items = data.get("merchants", data.get("customers", []))
            store = merchants if "merchant" in name else customers
            for item in items:
                if key in item:
                    store[item[key]] = item

    path = dataset_dir / "triggers_seed.json"
    if path.exists():
        data = json.load(open(path))
        for item in data.get("triggers", []):
            if "id" in item:
                triggers[item["id"]] = item

    return categories, merchants, customers, triggers


def get_test_pairs(dataset_dir: Path, triggers: dict) -> list[dict]:
    """Load test_pairs.json or generate 30 pairs from triggers."""
    tp_path = dataset_dir / "expanded" / "test_pairs.json"
    if tp_path.exists():
        return json.load(open(tp_path)).get("pairs", [])[:30]

    # Generate from seed triggers
    pairs = []
    for i, (tid, t) in enumerate(triggers.items()):
        if i >= 30:
            break
        pairs.append({
            "test_id": f"T{i+1:02d}",
            "trigger_id": tid,
            "merchant_id": t.get("merchant_id"),
            "customer_id": t.get("customer_id"),
        })
    return pairs


def main():
    dataset_dir = Path(__file__).parent / "dataset"
    output_path = Path(__file__).parent / "submission.jsonl"

    print("Loading dataset...")
    categories, merchants, customers, triggers = load_dataset(dataset_dir)
    print(f"  {len(categories)} categories, {len(merchants)} merchants, "
          f"{len(customers)} customers, {len(triggers)} triggers")

    # Set up composer
    store = StateStore()
    llm = LLMClient()
    composer = VeraComposer(store=store, llm=llm)

    # Push all contexts to store
    for slug, cat in categories.items():
        store.upsert_context("category", slug, 1, cat)
    for mid, m in merchants.items():
        store.upsert_context("merchant", mid, 1, m)
    for cid, c in customers.items():
        store.upsert_context("customer", cid, 1, c)
    for tid, t in triggers.items():
        store.upsert_context("trigger", tid, 1, t)

    pairs = get_test_pairs(dataset_dir, triggers)
    print(f"  {len(pairs)} test pairs to compose")

    results = []
    for pair in pairs:
        tid = pair["trigger_id"]
        mid = pair["merchant_id"]
        test_id = pair["test_id"]

        print(f"  Composing {test_id}: {tid}...", end=" ")
        start = time.time()

        action = composer.compose_for_tick(trigger_id=tid, now_iso="2026-04-26T12:00:00Z")

        if action:
            result = {
                "test_id": test_id,
                "body": action["body"],
                "cta": action.get("cta", "open_ended"),
                "send_as": action.get("send_as", "vera"),
                "suppression_key": action.get("suppression_key", ""),
                "rationale": action.get("rationale", ""),
            }
            elapsed = time.time() - start
            print(f"OK ({elapsed:.1f}s) — {len(action['body'])} chars")
        else:
            result = {
                "test_id": test_id,
                "body": f"[No composition possible for trigger {tid}]",
                "cta": "none",
                "send_as": "vera",
                "suppression_key": "",
                "rationale": "Trigger or merchant context missing",
            }
            print("SKIP (missing context)")

        results.append(result)

    # Write JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nDone! Wrote {len(results)} lines to {output_path}")


if __name__ == "__main__":
    main()
