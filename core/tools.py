"""
AshAI Core Tools & Public APIs
Provides live capabilities: Weather, Wikipedia, Crypto prices, Currency rates,
Advice/Quotes, Public API Directory Search, and 51,900+ Knowledge Base RAG.
"""

from __future__ import annotations
import json
import re
import urllib.parse
from pathlib import Path
import httpx

BASE_DIR = Path(__file__).resolve().parent.parent

_MEMORY_CACHE = None
_PUBLIC_APIS_CACHE = None

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

def get_public_apis_catalog() -> list[dict]:
    global _PUBLIC_APIS_CACHE
    if _PUBLIC_APIS_CACHE is None:
        cat_file = BASE_DIR / "data" / "public_apis.json"
        if cat_file.exists():
            try:
                with open(cat_file, "r", encoding="utf-8") as f:
                    _PUBLIC_APIS_CACHE = json.load(f)
            except Exception:
                _PUBLIC_APIS_CACHE = []
        else:
            _PUBLIC_APIS_CACHE = []
    return _PUBLIC_APIS_CACHE

async def fetch_weather(location: str) -> str:
    """Live weather for any city worldwide."""
    try:
        clean_loc = urllib.parse.quote(location.strip())
        url = f"https://wttr.in/{clean_loc}?format=j1"
        async with httpx.AsyncClient(timeout=6.0) as client:
            resp = await client.get(url, headers={"User-Agent": "AshAI/2.0"})
            if resp.status_code == 200:
                data = resp.json()
                current = data.get("current_condition", [{}])[0]
                temp_c = current.get("temp_C", "N/A")
                desc = current.get("weatherDesc", [{}])[0].get("value", "Clear")
                humidity = current.get("humidity", "N/A")
                wind = current.get("windspeedKmph", "N/A")
                return f"🌦️ Weather for {location.title()}:\n• Condition: {desc}\n• Temperature: {temp_c}°C\n• Humidity: {humidity}%\n• Wind Speed: {wind} km/h"
    except Exception:
        pass
    return f"Unable to fetch live weather for '{location}' at the moment."

async def fetch_crypto_prices(symbols: list[str] | None = None) -> str:
    """Live cryptocurrency prices from Binance Public API."""
    try:
        default_pairs = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"]
        pairs_param = json.dumps(symbols or default_pairs, separators=(",", ":"))
        url = f"https://api.binance.com/api/v3/ticker/price?symbols={urllib.parse.quote(pairs_param)}"
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                tickers = resp.json()
                lines = ["🪙 Live Crypto Prices (Binance Public API):"]
                for t in tickers:
                    sym = t.get("symbol", "").replace("USDT", "")
                    price = float(t.get("price", 0))
                    lines.append(f"• {sym}: ${price:,.2f} USD")
                return "\n".join(lines)
    except Exception as e:
        pass
    return "Crypto price service temporarily unavailable."

async def fetch_exchange_rates(target: str = "INR") -> str:
    """Live foreign currency exchange rates from Open Exchange API."""
    try:
        url = "https://open.er-api.com/v6/latest/USD"
        async with httpx.AsyncClient(timeout=6.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                rates = data.get("rates", {})
                target_upper = target.upper().strip()
                inr_rate = rates.get("INR", 0)
                eur_rate = rates.get("EUR", 0)
                aed_rate = rates.get("AED", 0)
                sar_rate = rates.get("SAR", 0)
                gbp_rate = rates.get("GBP", 0)

                lines = ["💵 Live Currency Exchange Rates (Base: 1 USD):"]
                lines.append(f"• 1 USD = ₹{inr_rate:,.2f} INR (Indian Rupee)")
                lines.append(f"• 1 USD = €{eur_rate:,.2f} EUR (Euro)")
                lines.append(f"• 1 USD = {aed_rate:,.2f} AED (UAE Dirham)")
                lines.append(f"• 1 USD = {sar_rate:,.2f} SAR (Saudi Riyal)")
                lines.append(f"• 1 USD = £{gbp_rate:,.2f} GBP (British Pound)")
                return "\n".join(lines)
    except Exception:
        pass
    return "Currency exchange service unavailable."

async def fetch_random_advice() -> str:
    """Inspirational advice from Advice Slip API."""
    try:
        url = "https://api.adviceslip.com/advice"
        async with httpx.AsyncClient(timeout=4.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                advice = resp.json().get("slip", {}).get("advice", "")
                if advice:
                    return f"💡 Daily Thought: \"{advice}\""
    except Exception:
        pass
    return "💡 Thought: Keep learning, every line of code takes you closer to mastery."

async def fetch_wikipedia(query: str) -> str | None:
    """Live Wikipedia summary lookup."""
    try:
        clean_q = urllib.parse.quote(query.strip())
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{clean_q}"
        headers = {"User-Agent": "AshAI/2.0 (contact: m.shamil.07.08.06@gmail.com)"}
        async with httpx.AsyncClient(timeout=6.0, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                extract = data.get("extract", "")
                title = data.get("title", query)
                if extract:
                    return f"📚 Wikipedia ({title}):\n{extract}"

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

def search_public_apis_catalog(query: str, max_results: int = 5) -> list[dict]:
    """Search through 2,400+ indexed public APIs from github.com/public-apis/public-apis."""
    catalog = get_public_apis_catalog()
    if not catalog:
        return []

    tokens = [t.lower() for t in re.findall(r"\w+", query) if len(t) > 2]
    if not tokens:
        return catalog[:max_results]

    scored = []
    for item in catalog:
        name = item.get("name", "").lower()
        desc = item.get("description", "").lower()
        cat = item.get("category", "").lower()
        score = 0
        for tok in tokens:
            if tok in name:
                score += 3
            if tok in cat:
                score += 2
            if tok in desc:
                score += 1
        if score > 0:
            scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:max_results]]

def search_long_term_memory(query: str, top_k: int = 3) -> list[dict]:
    """Fast keyword & topic search across 51,900+ unified knowledge records."""
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
    """Intelligently matches user prompt (English, Malayalam, or Manglish) with the best live tool or knowledge base."""
    msg = user_message.lower().strip()

    # 1. Weather Intent (English, Manglish, Malayalam)
    weather_keywords = ["weather", "temperature", "climate", "rain", "mazha", "choodu", "കാലാവസ്ഥ", "മഴ", "ചൂട്", "തണുപ്പ്"]
    if any(k in msg for k in weather_keywords):
        loc_match = re.search(r"(?:in|of|at|for)\s+([a-zA-Z\s]+)", msg)
        loc = loc_match.group(1).strip() if loc_match else ""
        # Malayalam locations
        if not loc:
            if "കൊച്ചി" in msg or "kochi" in msg:
                loc = "Kochi"
            elif "തിരുവനന്തപുരം" in msg or "trivandrum" in msg:
                loc = "Trivandrum"
            elif "കോഴിക്കോട്" in msg or "calicut" in msg:
                loc = "Kozhikode"
            elif "ഡൽഹി" in msg or "delhi" in msg:
                loc = "Delhi"
            elif "ദുബായ്" in msg or "dubai" in msg:
                loc = "Dubai"
            else:
                loc = "Kochi"

        loc = re.sub(r"[?!.]", "", loc).strip() or "Kochi"
        weather_res = await fetch_weather(loc)
        return {"tool": "Live Weather API", "content": weather_res}

    # 2. Crypto Intent
    crypto_keywords = ["crypto", "bitcoin", "btc", "ethereum", "eth", "solana", "sol", "doge", "coin price", "ക്രിപ്റ്റോ", "ബിറ്റ്കോയിൻ"]
    if any(k in msg for k in crypto_keywords):
        crypto_res = await fetch_crypto_prices()
        return {"tool": "Binance Crypto Public API", "content": crypto_res}

    # 3. Currency / Forex Exchange Intent
    forex_keywords = ["dollar", "rupee", "currency", "exchange rate", "inr", "rate of usd", "money conversion", "ഡോളർ", "രൂപ", "വിനിമയ നിരക്ക്"]
    if any(k in msg for k in forex_keywords):
        forex_res = await fetch_exchange_rates()
        return {"tool": "Open Exchange Rates Public API", "content": forex_res}

    # 4. Advice / Quote Intent
    advice_keywords = ["quote", "advice", "inspiration", "motivation", "thought", "ചിന്ത", "ഉപദേശം", "സുഭാഷിതം"]
    if any(k in msg for k in advice_keywords):
        adv_res = await fetch_random_advice()
        return {"tool": "Advice Slip Public API", "content": adv_res}

    # 5. Public API Directory Intent
    api_keywords = ["find api", "list api", "public api", "free api", "apis for", "api list", "പബ്ലിക് എപിഐ"]
    if any(k in msg for k in api_keywords):
        apis = search_public_apis_catalog(msg, max_results=4)
        if apis:
            lines = ["🌐 Free APIs from public-apis directory:"]
            for a in apis:
                lines.append(f"\n• {a['name']} ({a['category']})\n  Description: {a['description']}\n  Auth: {a['auth']} | Link: {a['url']}")
            return {"tool": "Public APIs Directory (2,400+ APIs)", "content": "\n".join(lines)}

    # 6. Wikipedia Knowledge Intent
    wiki_triggers = ["who is", "what is", "tell me about", "explain", "history of", "wiki", "ആരാണ്", "എന്താണ്"]
    if any(k in msg for k in wiki_triggers):
        sub = re.sub(r"(?:who is|what is|tell me about|explain|history of|wiki|ആരാണ്|എന്താണ്)\s+", "", msg).strip()
        sub = re.sub(r"[?!.]", "", sub).strip()
        if len(sub) > 2:
            wiki_res = await fetch_wikipedia(sub)
            if wiki_res:
                return {"tool": "Wikipedia Public API", "content": wiki_res}

    # 7. Unified 51,900+ Memory Search (Hugging Face + Public APIs)
    matched = search_long_term_memory(user_message, top_k=2)
    if matched:
        snippets = []
        for r in matched:
            cat = r.get("category", "General").title()
            prov = r.get("provenance", "AshAI Knowledge").split("/")[-1]
            content = r.get("content", "")
            snippets.append(f"[{cat} - {prov}]\n{content}")
        return {"tool": "AshAI Knowledge Base (51.9k Records)", "content": "\n\n---\n\n".join(snippets)}

    return {"tool": None, "content": None}
