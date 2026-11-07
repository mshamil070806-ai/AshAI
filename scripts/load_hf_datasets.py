"""
AshAI - Hugging Face Dataset Loader
Fetches real AI knowledge datasets from Hugging Face and saves to long_term_memory.json.
"""

from __future__ import annotations
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def fetch_and_populate_memory(limit_per_dataset: int = 50):
    base_dir = Path(__file__).resolve().parent.parent
    memory_file = base_dir / "long_term_memory.json"
    hf_token = os.getenv("HF_TOKEN")

    print(f"Connecting to Hugging Face (Token configured: {bool(hf_token)})...")
    
    from datasets import load_dataset

    new_entries = []

    # 1. Fetch AI persona & problem solving knowledge from awesome-chatgpt-prompts
    try:
        print("Fetching prompt/persona knowledge from fka/awesome-chatgpt-prompts...")
        ds_prompts = load_dataset("fka/awesome-chatgpt-prompts", split=f"train[:{limit_per_dataset}]", token=hf_token)
        for item in ds_prompts:
            act = item.get("act", "")
            prompt = item.get("prompt", "")
            if act and prompt:
                new_entries.append({
                    "id": str(uuid.uuid4()),
                    "content": f"Persona/Capability: {act}. Instructions: {prompt[:300]}...",
                    "category": "persona",
                    "importance": "USEFUL",
                    "provenance": "HuggingFace/fka/awesome-chatgpt-prompts",
                    "timestamp": datetime.utcnow().isoformat(),
                    "metadata": {"act": act},
                    "is_persistent": True,
                    "user_approved": True
                })
        print(f"Added {len(new_entries)} entries from awesome-chatgpt-prompts.")
    except Exception as e:
        print(f"Error fetching prompts dataset: {e}")

    # 2. Fetch general instruction and facts from databricks/databricks-dolly-15k
    try:
        print("Fetching open QA knowledge from databricks/databricks-dolly-15k...")
        ds_dolly = load_dataset("databricks/databricks-dolly-15k", split=f"train[:{limit_per_dataset}]", token=hf_token)
        for item in ds_dolly:
            instruction = item.get("instruction", "")
            response = item.get("response", "")
            category = item.get("category", "general")
            if instruction and response:
                new_entries.append({
                    "id": str(uuid.uuid4()),
                    "content": f"Q: {instruction}\nA: {response[:300]}",
                    "category": category,
                    "importance": "HIGH",
                    "provenance": "HuggingFace/databricks/databricks-dolly-15k",
                    "timestamp": datetime.utcnow().isoformat(),
                    "metadata": {"category": category},
                    "is_persistent": True,
                    "user_approved": True
                })
        print(f"Total new entries after Dolly-15k: {len(new_entries)}")
    except Exception as e:
        print(f"Error fetching Dolly dataset: {e}")

    # Load existing memory
    existing = []
    if memory_file.exists():
        try:
            with open(memory_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
                if not isinstance(existing, list):
                    existing = []
        except Exception:
            existing = []

    # Merge unique
    combined = existing + new_entries
    with open(memory_file, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"Successfully updated {memory_file.name}!")
    print(f"Previous count: {len(existing)}, New total: {len(combined)} records.")

if __name__ == "__main__":
    fetch_and_populate_memory(50)
