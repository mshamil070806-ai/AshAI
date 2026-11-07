from __future__ import annotations

import json
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from core.tools import execute_smart_tools, fetch_weather, fetch_wikipedia, search_long_term_memory

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

def create_app(config) -> FastAPI:
    app = FastAPI(
        title="AshAI API & Live Tools",
        description="Modular AI system powered by Hugging Face, Live Public APIs, and 49k Knowledge Base",
        version="2.0.0"
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
  <title>AshAI – Intelligent Multi-Tool AI</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0b0f19; color: #f8fafc; display: flex; flex-direction: column; height: 100vh; }
    header { padding: 1rem 1.5rem; background: #131b2e; border-bottom: 1px solid #1e293b; display: flex; align-items: center; justify-content: space-between; }
    header h1 { font-size: 1.25rem; font-weight: 700; color: #38bdf8; display: flex; align-items: center; gap: 0.5rem; }
    header .status-bar { display: flex; gap: 0.5rem; align-items: center; }
    .badge { font-size: 0.75rem; padding: 0.25rem 0.6rem; border-radius: 9999px; background: #0369a1; color: #e0f2fe; }
    .status { font-size: 0.75rem; padding: 0.25rem 0.6rem; border-radius: 9999px; background: #065f46; color: #34d399; }
    #chat-box { flex: 1; overflow-y: auto; padding: 1.5rem; display: flex; flex-direction: column; gap: 1rem; }
    .msg-wrap { display: flex; flex-direction: column; max-width: 80%; }
    .msg-wrap.user { align-self: flex-end; align-items: flex-end; }
    .msg-wrap.bot { align-self: flex-start; align-items: flex-start; }
    .tool-tag { font-size: 0.75rem; color: #94a3b8; margin-bottom: 0.25rem; display: flex; align-items: center; gap: 0.25rem; font-weight: 600; }
    .msg { padding: 0.85rem 1.25rem; border-radius: 1rem; line-height: 1.6; font-size: 0.95rem; white-space: pre-wrap; word-break: break-word; }
    .user .msg { background: #2563eb; color: #fff; border-bottom-right-radius: 0.2rem; }
    .bot .msg { background: #1e293b; color: #e2e8f0; border-bottom-left-radius: 0.2rem; border: 1px solid #334155; }
    .quick-actions { padding: 0.5rem 1.5rem; display: flex; gap: 0.5rem; overflow-x: auto; background: #0f172a; border-top: 1px solid #1e293b; }
    .chip { padding: 0.35rem 0.75rem; background: #1e293b; border: 1px solid #334155; border-radius: 9999px; font-size: 0.8rem; color: #94a3b8; cursor: pointer; white-space: nowrap; transition: 0.2s; }
    .chip:hover { background: #334155; color: #f8fafc; border-color: #38bdf8; }
    .footer { padding: 1rem 1.5rem; background: #131b2e; border-top: 1px solid #1e293b; display: flex; gap: 0.75rem; }
    input[type="text"] { flex: 1; padding: 0.85rem 1rem; border-radius: 0.6rem; border: 1px solid #334155; background: #0b0f19; color: #f8fafc; font-size: 1rem; outline: none; }
    input[type="text"]:focus { border-color: #38bdf8; }
    button { padding: 0.85rem 1.5rem; border-radius: 0.6rem; border: none; background: #0284c7; color: white; font-weight: 600; cursor: pointer; transition: 0.2s; }
    button:hover { background: #0369a1; }
  </style>
</head>
<body>
  <header>
    <h1>⚡ AshAI Engine</h1>
    <div class="status-bar">
      <span class="badge">🧠 51,900+ Records</span>
      <span class="badge">🌐 2,400+ Public APIs</span>
      <span class="status">● Online</span>
    </div>
  </header>
  <div id="chat-box">
    <div class="msg-wrap bot">
      <div class="tool-tag">⚡ AshAI Core Online</div>
      <div class="msg">Hello! I am AshAI. I am equipped with live Weather, Binance Crypto, Global Currency Exchange Rates, Wikipedia, 2,400+ Free Public APIs, and 51,900+ AI records from Hugging Face. Ask me anything!</div>
    </div>
  </div>
  <div class="quick-actions">
    <div class="chip" onclick="quickSend('Live crypto prices')">🪙 Bitcoin & Crypto Price</div>
    <div class="chip" onclick="quickSend('Current dollar to rupee exchange rate')">💵 Dollar to Rupee Rate</div>
    <div class="chip" onclick="quickSend('Weather in Kochi')">🌦️ Weather in Kochi</div>
    <div class="chip" onclick="quickSend('Find free APIs for weather')">🌐 Find Weather APIs</div>
    <div class="chip" onclick="quickSend('Give me some advice')">💡 Daily Thought</div>
    <div class="chip" onclick="quickSend('What is Artificial Intelligence?')">📚 What is AI?</div>
    <div class="chip" onclick="quickSend('Write a python function to reverse a string')">💻 Python Code</div>
  </div>
  <div class="footer">
    <input type="text" id="userInput" placeholder="Ask a question, query science, weather, or coding..." onkeydown="if(event.key==='Enter') sendMsg()" />
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
        const toolLabel = data.tool_used ? `🔧 ${data.tool_used}` : '⚡ AshAI Model';
        box.innerHTML += `
          <div class="msg-wrap bot">
            <div class="tool-tag">${toolLabel}</div>
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
            "hf_configured": bool(config.hf_token),
            "live_apis": ["Weather API (wttr)", "Wikipedia REST API", "AshAI 49k Knowledge Base"]
        }

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest):
        model_name = req.model or config.default_model
        token = config.hf_token or os.getenv("HF_TOKEN")

        # 1. First run smart tool router (Live Weather, Wikipedia, or 49k Knowledge Memory)
        tool_result = await execute_smart_tools(req.message)
        tool_name = tool_result.get("tool")
        tool_content = tool_result.get("content")

        # If a live tool or knowledge base returned definitive information
        if tool_content:
            # If Hugging Face LLM is available, synthesize naturally
            if token and InferenceClient is not None:
                try:
                    client = InferenceClient(api_key=token)
                    prompt = (
                        f"Context information retrieved from {tool_name}:\n{tool_content}\n\n"
                        f"User Question: {req.message}\n"
                        f"Please provide a helpful, natural, and accurate answer using the context provided above:"
                    )
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=req.temperature,
                        max_tokens=req.max_tokens,
                    )
                    reply = response.choices[0].message.content
                    return ChatResponse(reply=reply, model=model_name, tool_used=tool_name)
                except Exception:
                    pass
            
            # Return high-fidelity formatted tool response
            return ChatResponse(reply=tool_content, model="ashai-hybrid-engine", tool_used=tool_name)

        # 2. General conversation fallback
        return ChatResponse(
            reply=f"AshAI Core: I received your request: '{req.message}'. You can ask me about live weather anywhere in the world, Wikipedia summaries, coding problems, science questions, or explore my 49,500+ knowledge records!",
            model="ashai-core",
            tool_used="AshAI Orchestrator"
        )

    @app.get("/api/tools/weather")
    async def api_weather(city: str = "Kochi"):
        return {"result": await fetch_weather(city)}

    @app.get("/api/tools/wiki")
    async def api_wiki(q: str):
        res = await fetch_wikipedia(q)
        return {"result": res or "Not found"}

    @app.get("/api/memory/search")
    async def api_memory_search(q: str):
        return {"matches": search_long_term_memory(q, top_k=5)}

    return app
