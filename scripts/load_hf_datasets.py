"""
AshAI - Hugging Face Multi-Domain Knowledge Loader
Fetches complete datasets across Persona, Instruction, Science, and Coding.
"""

from __future__ import annotations
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def populate_all_knowledge():
    base_dir = Path(__file__).resolve().parent.parent
    memory_file = base_dir / "long_term_memory.json"
    hf_token = os.getenv("HF_TOKEN")

    print(f"Connecting to Hugging Face Hub (Token: {'Active' if hf_token else 'None'})...")
    from datasets import load_dataset

    new_entries = []

    # 1. Science & Reason knowledge (allenai/sciq - 11,679 records)
    try:
        print("\n[1/2] Loading Science dataset (allenai/sciq)...")
        sciq = load_dataset("allenai/sciq", split="train", token=hf_token)
        count_sciq = 0
        for item in sciq:
            question = item.get("question", "")
            correct_ans = item.get("correct_answer", "")
            support = item.get("support", "")
            if question and correct_ans:
                content = f"Science Q: {question}\nAnswer: {correct_ans}"
                if support:
                    content += f"\nExplanation: {support}"
                new_entries.append({
                    "id": str(uuid.uuid4()),
                    "content": content,
                    "category": "science",
                    "importance": "HIGH",
                    "provenance": "HuggingFace/allenai/sciq",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "metadata": {"topic": "science"},
                    "is_persistent": True,
                    "user_approved": True
                })
                count_sciq += 1
        print(f"Loaded {count_sciq} Science Q&A records.")
    except Exception as e:
        print(f"Error loading sciq: {e}")

    # 2. Coding & Programming knowledge (sahil2801/CodeAlpaca-20k - 20,022 records)
    try:
        print("\n[2/2] Loading Coding dataset (sahil2801/CodeAlpaca-20k)...")
        code_ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train", token=hf_token)
        count_code = 0
        for item in code_ds:
            instruction = item.get("instruction", "")
            inp = item.get("input", "")
            output = item.get("output", "")
            if instruction and output:
                content = f"Code Task: {instruction}\n"
                if inp:
                    content += f"Input: {inp}\n"
                content += f"Solution:\n{output}"
                new_entries.append({
                    "id": str(uuid.uuid4()),
                    "content": content,
                    "category": "coding",
                    "importance": "HIGH",
                    "provenance": "HuggingFace/sahil2801/CodeAlpaca-20k",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "metadata": {"topic": "coding"},
                    "is_persistent": True,
                    "user_approved": True
                })
                count_code += 1
        print(f"Loaded {count_code} Coding & Algorithm records.")
    except Exception as e:
        print(f"Error loading CodeAlpaca: {e}")

    print(f"\nTotal new knowledge items prepared: {len(new_entries)}")

    # Load existing memory to merge
    existing = []
    if memory_file.exists():
        try:
            with open(memory_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
                if not isinstance(existing, list):
                    existing = []
        except Exception:
            existing = []

    seen = set()
    combined = []
    for item in existing:
        key = item.get("content", "")[:120]
        if key not in seen:
            seen.add(key)
            combined.append(item)

    for item in new_entries:
        key = item.get("content", "")[:120]
        if key not in seen:
            seen.add(key)
            combined.append(item)

    print(f"Writing {len(combined)} total unified records to {memory_file.name}...")
    with open(memory_file, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"Complete! Total memory records in AshAI: {len(combined)}")

if __name__ == "__main__":
    populate_all_knowledge()
