"""
AshAI Core Tools & Public APIs
Provides real-time capabilities: Weather, Wikipedia, Knowledge Base RAG, and Calculations.
"""

from __future__ import annotations
import json
import re
import urllib.parse
from pathlib import Path
import httpx

BASE_DIR = Path(__file__).resolve().parent.parent

# In-memory index cache for fast search across 49,500+ records
_MEMORY_CACHE = None

def get_memory_data() -> list[dict]:
    global _MEMORY_CACHE
    if _MEMORY_CACHE is None:
        mem_file = BASE_DIR / "long_term_memory.json"
        if mem_file.exists():
            try:
                with open(mem_file, "r", encoding="utf-8") as f:
                    _MEMORY_CACHE = json.load(f)
            except Exception:
                _MEMORY_CACHE = []
        else:
            _MEMORY_CACHE = []
    return _MEMORY_CACHE

async def fetch_weather(location: str) -> str:
    """Fetch live weather for any location worldwide using public wttr API."""
    try:
        clean_loc = urllib.parse.quote(location.strip())
        url = f"https://wttr.in/{clean_loc}?format=j1"
        async with httpx.AsyncClient(timeout=6.0) as client:
            resp = await client.get(url, headers={"User-Agent": "AshAI/1.0"})
            if resp.status_code == 200:
                data = resp.json()
                current = data.get("current_condition", [{}])[0]
                temp_c = current.get("temp_C", "N/A")
                desc = current.get("weatherDesc", [{}])[0].get("value", "Clear")
                humidity = current.get("humidity", "N/A")
                wind = current.get("windspeedKmph", "N/A")
                return f"🌦️ Weather for {location.title()}:\n• Condition: {desc}\n• Temperature: {temp_c}°C\n• Humidity: {humidity}%\n• Wind Speed: {wind} km/h"
    except Exception as e:
        pass
    return f"Unable to fetch live weather for '{location}' at the moment."

async def fetch_wikipedia(query: str) -> str | None:
    """Search live Wikipedia summaries with redirect following and opensearch fallback."""
    try:
        clean_q = urllib.parse.quote(query.strip())
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{clean_q}"
        headers = {"User-Agent": "AshAI/1.0 (https://github.com/mshamil070806-ai/AshAI; contact@ashai.org)"}
        async with httpx.AsyncClient(timeout=6.0, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                extract = data.get("extract", "")
                title = data.get("title", query)
                if extract:
                    return f"📚 Wikipedia ({title}):\n{extract}"

            # Fallback to Wikipedia search API
            search_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={clean_q}&limit=1&namespace=0&format=json"
            s_resp = await client.get(search_url, headers=headers)
            if s_resp.status_code == 200:
                s_data = s_resp.json()
                if len(s_data) > 2 and s_data[1] and s_data[2]:
                    title = s_data[1][0]
                    snippet = s_data[2][0]
                    if snippet:
                        return f"📚 Wikipedia ({title}):\n{snippet}"
    except Exception:
        pass
    return None

def search_long_term_memory(query: str, top_k: int = 3) -> list[dict]:
    """Fast keyword & topic search through AshAI's 49,500+ stored Hugging Face records."""
    records = get_memory_data()
    if not records:
        return []

    tokens = [t.lower() for t in re.findall(r"\w+", query) if len(t) > 2]
    if not tokens:
        return []

    scored = []
    for item in records:
        content = item.get("content", "").lower()
        category = item.get("category", "").lower()
        score = 0
        for tok in tokens:
            if tok in content:
                score += 2
            if tok in category:
                score += 1

        if score > 0:
            scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:top_k]]

async def execute_smart_tools(user_message: str) -> dict:
    """
    Intelligently inspects user query and invokes the right public API or Knowledge tool.
    Returns tool result and tool name.
    """
    msg_lower = user_message.lower().strip()

    # 1. Weather intent
    weather_match = re.search(r"(?:weather|climate|rain|temperature|choodu|mazha)(?:\s+(?:in|of|at|for))?\s+([a-zA-Z\s]+)", msg_lower)
    if weather_match or any(k in msg_lower for k in ["weather in", "temperature in", "weather of"]):
        loc = weather_match.group(1).strip() if weather_match and weather_match.group(1) else "Kochi"
        # remove punctuation
        loc = re.sub(r"[?!.]", "", loc).strip()
        if not loc or loc in ["today", "now"]:
            loc = "Kochi"
        weather_info = await fetch_weather(loc)
        return {"tool": "Live Weather API", "content": weather_info}

    # 2. Wikipedia intent
    wiki_match = re.search(r"(?:who is|what is|tell me about|explain|wiki)\s+([a-zA-Z0-9\s]+)", msg_lower)
    if wiki_match:
        subject = wiki_match.group(1).strip()
        subject = re.sub(r"[?!.]", "", subject).strip()
        if len(subject) > 2:
            wiki_res = await fetch_wikipedia(subject)
            if wiki_res:
                return {"tool": "Wikipedia Public API", "content": wiki_res}

    # 3. Knowledge Memory Search (49,500+ Hugging Face records)
    matched_records = search_long_term_memory(user_message, top_k=2)
    if matched_records:
        snippets = []
        for r in matched_records:
            cat = r.get("category", "General").title()
            prov = r.get("provenance", "Hugging Face").split("/")[-1]
            content = r.get("content", "")
            snippets.append(f"[{cat} - {prov}]\n{content}")
        
        combined_text = "\n\n---\n\n".join(snippets)
        return {"tool": "AshAI Knowledge Base (49k Records)", "content": combined_text}

    return {"tool": None, "content": None}
