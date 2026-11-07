"""
AshAI - Hugging Face Full Dataset Loader
Fetches complete AI knowledge datasets from Hugging Face and saves to long_term_memory.json without restriction.
"""

from __future__ import annotations
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def fetch_and_populate_memory(limit_per_dataset: int | None = None):
    base_dir = Path(__file__).resolve().parent.parent
    memory_file = base_dir / "long_term_memory.json"
    hf_token = os.getenv("HF_TOKEN")

    print(f"Connecting to Hugging Face (Token: {'Active' if hf_token else 'None'})...")
    
    from datasets import load_dataset

    new_entries = []

    # 1. Fetch ALL prompt/persona knowledge from fka/awesome-chatgpt-prompts
    try:
        print("Fetching ALL prompts from fka/awesome-chatgpt-prompts...")
        split_arg = "train" if limit_per_dataset is None else f"train[:{limit_per_dataset}]"
        ds_prompts = load_dataset("fka/awesome-chatgpt-prompts", split=split_arg, token=hf_token)
        count_prompts = 0
        for item in ds_prompts:
            act = item.get("act", "")
            prompt = item.get("prompt", "")
            if act and prompt:
                new_entries.append({
                    "id": str(uuid.uuid4()),
                    "content": f"Persona/Capability: {act}. Instructions: {prompt}",
                    "category": "persona",
                    "importance": "USEFUL",
                    "provenance": "HuggingFace/fka/awesome-chatgpt-prompts",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "metadata": {"act": act},
                    "is_persistent": True,
                    "user_approved": True
                })
                count_prompts += 1
        print(f"Successfully loaded {count_prompts} complete prompt records!")
    except Exception as e:
        print(f"Error fetching prompts dataset: {e}")

    # 2. Fetch ALL general instruction and QA facts from databricks/databricks-dolly-15k
    try:
        print("Fetching ALL records from databricks/databricks-dolly-15k...")
        split_arg = "train" if limit_per_dataset is None else f"train[:{limit_per_dataset}]"
        ds_dolly = load_dataset("databricks/databricks-dolly-15k", split=split_arg, token=hf_token)
        count_dolly = 0
        for item in ds_dolly:
            instruction = item.get("instruction", "")
            context = item.get("context", "")
            response = item.get("response", "")
            category = item.get("category", "general")
            if instruction and response:
                content_text = f"Q: {instruction}\n"
                if context:
                    content_text += f"Context: {context[:500]}\n"
                content_text += f"A: {response}"

                new_entries.append({
                    "id": str(uuid.uuid4()),
                    "content": content_text,
                    "category": category,
                    "importance": "HIGH",
                    "provenance": "HuggingFace/databricks/databricks-dolly-15k",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "metadata": {"category": category},
                    "is_persistent": True,
                    "user_approved": True
                })
                count_dolly += 1
        print(f"Successfully loaded {count_dolly} complete Q&A records!")
    except Exception as e:
        print(f"Error fetching Dolly dataset: {e}")

    print(f"\nTotal new records gathered from Hugging Face: {len(new_entries)}")

    # Load existing memory to prevent duplicate loss
    existing = []
    if memory_file.exists():
        try:
            with open(memory_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
                if not isinstance(existing, list):
                    existing = []
        except Exception:
            existing = []

    # Deduplicate based on content prefix
    seen_content = set()
    combined = []
    for item in existing:
        key = item.get("content", "")[:100]
        if key not in seen_content:
            seen_content.add(key)
            combined.append(item)

    for item in new_entries:
        key = item.get("content", "")[:100]
        if key not in seen_content:
            seen_content.add(key)
            combined.append(item)

    print(f"Writing {len(combined)} total records to {memory_file}...")
    with open(memory_file, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"All datasets successfully saved! Total memory size: {len(combined)} records.")

if __name__ == "__main__":
    fetch_and_populate_memory(None)
