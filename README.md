# ⚡ AshAI – Bilingual AI Engine & MCP Server

AshAI is a high-performance **Bilingual AI System (English & Malayalam)** equipped with a full **Model Context Protocol (MCP)** server, live public APIs, and a 51,900+ record knowledge base.

---

## 🌟 Key Capabilities

1. **Bilingual Conversational AI (മലയാളം & English):**
   - Natural language comprehension in English, Malayalam script, and Manglish phonetics.
   - Intelligent automatic language detection and culturally resonant formatting.

2. **Model Context Protocol (MCP) Server:**
   - Compliant with Anthropic/Open MCP specification over standard I/O and HTTP (`/mcp`).
   - Seamlessly connectable to Claude Desktop, Cursor, Antigravity, or any agentic system.

3. **Live Real-Time Tools:**
   - 🌦️ **Weather API:** Real-time global weather, temperature, humidity, and wind speed.
   - 🪙 **Crypto Ticker:** Live cryptocurrency prices (BTC, ETH, SOL, DOGE) from Binance.
   - 💵 **Forex Rates:** Real-time global currency exchange rates (USD to INR, EUR, AED, SAR, GBP).
   - 📚 **Wikipedia Knowledge:** Live topic and biographical summaries.
   - 🌐 **Public APIs Directory:** Instant search across 2,400+ free APIs.

4. **Extensive Knowledge Base:**
   - 51,900+ curated records from Hugging Face (`databricks-dolly-15k`, `sciq`, `CodeAlpaca-20k`, `awesome-chatgpt-prompts`, and `public-apis`).

---

## 🚀 Quick Start

### 1. Run the Web Application
```bash
cd C:\Users\ahsas\Documents\AshAI
pip install -r requirements.txt
python main.py
```
Open **`http://localhost:7860`** in your browser.

---

### 2. Connect as an MCP Server

#### For Claude Desktop or Cursor:
Add the following to your `claude_desktop_config.json` or Cursor MCP settings:
```json
{
  "mcpServers": {
    "ashai": {
      "command": "python",
      "args": [
        "C:\\Users\\ahsas\\Documents\\AshAI\\mcp_server.py"
      ],
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

---

## 🛠️ MCP Tools Exposed

| Tool Name | Description |
|-----------|-------------|
| `ashai_bilingual_chat` | Bilingual reasoning in English, Malayalam, or Manglish. |
| `get_weather` | Real-time weather worldwide by city name. |
| `get_crypto_prices` | Live Binance crypto prices (BTC, ETH, SOL, DOGE). |
| `get_currency_rates` | Live foreign exchange rates (USD to INR, EUR, AED, SAR). |
| `search_wikipedia` | Instant summaries from Wikipedia. |
| `search_knowledge_base` | Query 51,900+ Hugging Face science, coding, and QA records. |
| `search_public_apis` | Search across 2,400+ free APIs in the public-apis catalog. |

---

## 📜 License
MIT License
