"""
AshAI - Public APIs Harvester
Downloads the entire 1,400+ API index from github.com/public-apis/public-apis,
saves it to data/public_apis.json, and registers them into long_term_memory.json.
"""

from __future__ import annotations
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
import httpx

BASE_DIR = Path(__file__).resolve().parent.parent

def fetch_and_index_public_apis():
    url = "https://raw.githubusercontent.com/public-apis/public-apis/master/README.md"
    print("Downloading full directory from github.com/public-apis/public-apis...")

    resp = httpx.get(url, timeout=15.0)
    if resp.status_code != 200:
        print(f"Failed to download README.md: status {resp.status_code}")
        return

    text = resp.text
    print(f"Downloaded {len(text)} characters of API documentation.")

    apis = []
    current_category = "General"

    lines = text.split("\n")
    for line in lines:
        line_strip = line.strip()
        # Track category
        if line_strip.startswith("### "):
            current_category = line_strip.replace("### ", "").strip()
            continue

        # Table row: | API | Description | Auth | HTTPS | CORS |
        if line_strip.startswith("|") and not line_strip.startswith("|---") and not line_strip.startswith("| API"):
            parts = [p.strip() for p in line_strip.split("|") if p.strip()]
            if len(parts) >= 3:
                # Markdown link format: [API Name](url)
                api_col = parts[0]
                desc_col = parts[1] if len(parts) > 1 else ""
                auth_col = parts[2] if len(parts) > 2 else "No"
                https_col = parts[3] if len(parts) > 3 else "Yes"
                cors_col = parts[4] if len(parts) > 4 else "Unknown"

                match = re.search(r"\[(.*?)\]\((.*?)\)", api_col)
                if match:
                    api_name = match.group(1).strip()
                    api_url = match.group(2).strip()
                else:
                    api_name = api_col
                    api_url = ""

                apis.append({
                    "name": api_name,
                    "url": api_url,
                    "description": desc_col,
                    "auth": auth_col,
                    "https": https_col,
                    "cors": cors_col,
                    "category": current_category
                })

    print(f"Successfully parsed {len(apis)} public APIs across all categories!")

    # Save to data/public_apis.json
    data_dir = BASE_DIR / "data"
    data_dir.mkdir(exist_ok=True)
    out_file = data_dir / "public_apis.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(apis, f, indent=2, ensure_ascii=False)
    print(f"Saved complete catalog to {out_file} ({out_file.stat().st_size} bytes).")

    # Add knowledge records into long_term_memory.json
    mem_file = BASE_DIR / "long_term_memory.json"
    existing = []
    if mem_file.exists():
        try:
            with open(mem_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
                if not isinstance(existing, list):
                    existing = []
        except Exception:
            existing = []

    new_memory_records = []
    for item in apis:
        content = (
            f"Public API: {item['name']} (Category: {item['category']})\n"
            f"Description: {item['description']}\n"
            f"Auth: {item['auth']} | HTTPS: {item['https']} | CORS: {item['cors']}\n"
            f"Link: {item['url']}"
        )
        new_memory_records.append({
            "id": str(uuid.uuid4()),
            "content": content,
            "category": "public_api",
            "importance": "USEFUL",
            "provenance": "GitHub/public-apis/public-apis",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": {
                "api_name": item["name"],
                "category": item["category"],
                "url": item["url"]
            },
            "is_persistent": True,
            "user_approved": True
        })

    # Merge unique
    seen = {m.get("content", "")[:100] for m in existing}
    combined = list(existing)
    for r in new_memory_records:
        key = r.get("content", "")[:100]
        if key not in seen:
            seen.add(key)
            combined.append(r)

    print(f"Writing {len(combined)} total records into {mem_file.name}...")
    with open(mem_file, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"Done! AshAI long-term memory now contains {len(combined)} total records.")

if __name__ == "__main__":
    fetch_and_index_public_apis()
