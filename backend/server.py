from __future__ import annotations

import json
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from core.tools import execute_smart_tools, fetch_weather, fetch_wikipedia, search_long_term_memory
from core.bilingual_engine import detect_language, format_bilingual_response
from mcp_server import process_json_rpc, TOOLS, SERVER_INFO

try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None

class ChatRequest(BaseModel):
    message: str
    model: str | None = None
    temperature: float = 0.7
    max_tokens: int = 512

class ChatResponse(BaseModel):
    reply: str
    model: str
    tool_used: str | None = None
    language: str = "en"

def create_app(config) -> FastAPI:
    app = FastAPI(
        title="AshAI Bilingual Engine & MCP Server",
        description="Bilingual LLM/LMM Assistant (Malayalam & English) with Model Context Protocol (MCP) and 51.9k Knowledge Base",
        version="2.5.0"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    base_dir = Path(__file__).resolve().parent.parent

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AshAI – Bilingual AI & MCP Server</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #070d18; color: #f8fafc; display: flex; flex-direction: column; height: 100vh; }
    header { padding: 1rem 1.5rem; background: #0f172a; border-bottom: 1px solid #1e293b; display: flex; align-items: center; justify-content: space-between; }
    header h1 { font-size: 1.25rem; font-weight: 700; color: #38bdf8; display: flex; align-items: center; gap: 0.5rem; }
    header .status-bar { display: flex; gap: 0.5rem; align-items: center; }
    .badge { font-size: 0.75rem; padding: 0.25rem 0.6rem; border-radius: 9999px; background: #0369a1; color: #e0f2fe; }
    .mcp-badge { font-size: 0.75rem; padding: 0.25rem 0.6rem; border-radius: 9999px; background: #4f46e5; color: #e0e7ff; font-weight: 600; }
    .status { font-size: 0.75rem; padding: 0.25rem 0.6rem; border-radius: 9999px; background: #065f46; color: #34d399; }
    #chat-box { flex: 1; overflow-y: auto; padding: 1.5rem; display: flex; flex-direction: column; gap: 1rem; }
    .msg-wrap { display: flex; flex-direction: column; max-width: 82%; }
    .msg-wrap.user { align-self: flex-end; align-items: flex-end; }
    .msg-wrap.bot { align-self: flex-start; align-items: flex-start; }
    .tool-tag { font-size: 0.75rem; color: #94a3b8; margin-bottom: 0.25rem; display: flex; align-items: center; gap: 0.25rem; font-weight: 600; }
    .msg { padding: 0.85rem 1.25rem; border-radius: 1rem; line-height: 1.6; font-size: 0.95rem; white-space: pre-wrap; word-break: break-word; }
    .user .msg { background: #2563eb; color: #fff; border-bottom-right-radius: 0.2rem; }
    .bot .msg { background: #1e293b; color: #e2e8f0; border-bottom-left-radius: 0.2rem; border: 1px solid #334155; }
    .quick-actions { padding: 0.5rem 1.5rem; display: flex; gap: 0.5rem; overflow-x: auto; background: #0f172a; border-top: 1px solid #1e293b; }
    .chip { padding: 0.35rem 0.75rem; background: #1e293b; border: 1px solid #334155; border-radius: 9999px; font-size: 0.8rem; color: #94a3b8; cursor: pointer; white-space: nowrap; transition: 0.2s; }
    .chip:hover { background: #334155; color: #f8fafc; border-color: #38bdf8; }
    .footer { padding: 1rem 1.5rem; background: #0f172a; border-top: 1px solid #1e293b; display: flex; gap: 0.75rem; }
    input[type="text"] { flex: 1; padding: 0.85rem 1rem; border-radius: 0.6rem; border: 1px solid #334155; background: #070d18; color: #f8fafc; font-size: 1rem; outline: none; }
    input[type="text"]:focus { border-color: #38bdf8; }
    button { padding: 0.85rem 1.5rem; border-radius: 0.6rem; border: none; background: #0284c7; color: white; font-weight: 600; cursor: pointer; transition: 0.2s; }
    button:hover { background: #0369a1; }
  </style>
</head>
<body>
  <header>
    <h1>⚡ AshAI (Bilingual LMM & MCP)</h1>
    <div class="status-bar">
      <span class="mcp-badge">🔌 MCP Protocol Active</span>
      <span class="badge">🌐 മലയാളം + English</span>
      <span class="badge">🧠 51,900+ Records</span>
      <span class="status">● Online</span>
    </div>
  </header>
  <div id="chat-box">
    <div class="msg-wrap bot">
      <div class="tool-tag">⚡ AshAI Bilingual Core</div>
      <div class="msg">നമസ്കാരം! I am AshAI — a bilingual AI assistant & Model Context Protocol (MCP) server.

നിങ്ങൾക്ക് മലയാളത്തിലോ (English, Malayalam, Manglish) ചോദ്യങ്ങൾ ചോദിക്കാം. ലൈവ് കാലാവസ്ഥ, ബിറ്റ്കോയിൻ വില, ഡോളർ നിരക്ക്, വിക്കിപീഡിയ, കോഡിംഗ് എന്നിവയെല്ലാം ലഭ്യമാണ്!</div>
    </div>
  </div>
  <div class="quick-actions">
    <div class="chip" onclick="quickSend('കൊച്ചിയിലെ കാലാവസ്ഥ എങ്ങനെയുണ്ട്?')">🌦️ കൊച്ചിയിലെ കാലാവസ്ഥ</div>
    <div class="chip" onclick="quickSend('Live crypto prices')">🪙 Bitcoin & Crypto Price</div>
    <div class="chip" onclick="quickSend('Dollar to INR exchange rate')">💵 ഡോളർ നിരക്ക് (INR)</div>
    <div class="chip" onclick="quickSend('Who is Alan Turing?')">👤 Who is Alan Turing?</div>
    <div class="chip" onclick="quickSend('Python function to reverse string')">💻 Python Code</div>
    <div class="chip" onclick="quickSend('Find free APIs for sports')">🌐 Free Public APIs</div>
    <div class="chip" onclick="quickSend('Inspirational advice')">💡 Daily Thought</div>
  </div>
  <div class="footer">
    <input type="text" id="userInput" placeholder="മലയാളത്തിലോ ഇംഗ്ലീഷിലോ എന്തും ചോദിക്കാം..." onkeydown="if(event.key==='Enter') sendMsg()" />
    <button onclick="sendMsg()">Send</button>
  </div>
  <script>
    function quickSend(txt) {
      document.getElementById('userInput').value = txt;
      sendMsg();
    }

    async function sendMsg() {
      const input = document.getElementById('userInput');
      const text = input.value.trim();
      if (!text) return;
      
      const box = document.getElementById('chat-box');
      box.innerHTML += `
        <div class="msg-wrap user">
          <div class="msg">${escapeHtml(text)}</div>
        </div>`;
      input.value = '';
      box.scrollTop = box.scrollHeight;
      
      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text })
        });
        const data = await res.json();
        const toolLabel = data.tool_used ? `🔧 ${data.tool_used}` : '⚡ AshAI Bilingual Core';
        box.innerHTML += `
          <div class="msg-wrap bot">
            <div class="tool-tag">${toolLabel} (${data.language.toUpperCase()})</div>
            <div class="msg">${escapeHtml(data.reply)}</div>
          </div>`;
      } catch (err) {
        box.innerHTML += `
          <div class="msg-wrap bot">
            <div class="msg" style="color:#f87171;">Error: Unable to connect to server.</div>
          </div>`;
      }
      box.scrollTop = box.scrollHeight;
    }

    function escapeHtml(str) {
      return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }
  </script>
</body>
</html>"""

    @app.get("/api/health")
    async def health():
        return {
            "status": "healthy",
            "app": config.app_name,
            "version": "2.5.0",
            "bilingual_support": ["English", "Malayalam (മലയാളം)", "Manglish"],
            "mcp_enabled": True,
            "mcp_tools": [t["name"] for t in TOOLS],
            "total_knowledge_records": 51917
        }

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest):
        model_name = req.model or config.default_model
        token = config.hf_token or os.getenv("HF_TOKEN")

        # 1. Detect language (Malayalam, Manglish, or English)
        lang = detect_language(req.message)

        # 2. Execute smart tool router
        tool_result = await execute_smart_tools(req.message)
        tool_name = tool_result.get("tool")
        tool_content = tool_result.get("content")

        if tool_content:
            formatted_reply = format_bilingual_response(lang, tool_content)
            return ChatResponse(reply=formatted_reply, model="ashai-bilingual-engine", tool_used=tool_name, language=lang)

        # 3. Fallback bilingual response
        if lang in ("ml", "manglish"):
            fallback_text = (
                "AshAI: നിങ്ങളുടെ ചോദ്യം ലഭിച്ചു. കാലാവസ്ഥ (Weather), ബിറ്റ്കോയിൻ വില (Crypto), ഡോളർ നിരക്ക്, "
                "വിക്കിപീഡിയ വിവരങ്ങൾ, കോഡിംഗ് പ്രശ്നങ്ങൾ എന്നിവയെല്ലാം ചോദിക്കാവുന്നതാണ്!"
            )
        else:
            fallback_text = (
                f"AshAI Core: I received your request: '{req.message}'. Ask me about live weather, crypto prices, "
                f"currency exchange rates, Wikipedia, coding, or our 51,900+ knowledge records!"
            )

        return ChatResponse(
            reply=fallback_text,
            model="ashai-bilingual-core",
            tool_used="AshAI Orchestrator",
            language=lang
        )

    # Standard Model Context Protocol (MCP) HTTP Endpoint
    @app.post("/mcp")
    async def mcp_endpoint(request: Request):
        try:
            body = await request.json()
            response = await process_json_rpc(json.dumps(body))
            return JSONResponse(content=response or {})
        except Exception as e:
            return JSONResponse(content={"jsonrpc": "2.0", "error": {"code": -32603, "message": str(e)}}, status_code=500)

    @app.get("/mcp/tools")
    async def list_mcp_tools():
        return {"tools": TOOLS, "serverInfo": SERVER_INFO}

    return app
