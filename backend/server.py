from __future__ import annotations

import json
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

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

def create_app(config) -> FastAPI:
    app = FastAPI(
        title="AshAI API",
        description="Modular AI system powered by Hugging Face & FastAPI",
        version="1.0.0"
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
  <title>AshAI – Hugging Face Hub</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; display: flex; flex-direction: column; height: 100vh; }
    header { padding: 1rem 2rem; background: #1e293b; border-bottom: 1px solid #334155; display: flex; align-items: center; justify-content: space-between; }
    header h1 { font-size: 1.25rem; font-weight: 700; color: #38bdf8; display: flex; align-items: center; gap: 0.5rem; }
    header .status { font-size: 0.85rem; padding: 0.25rem 0.75rem; border-radius: 9999px; background: #065f46; color: #34d399; }
    #chat-box { flex: 1; overflow-y: auto; padding: 2rem; display: flex; flex-direction: column; gap: 1rem; }
    .msg { max-width: 75%; padding: 0.85rem 1.25rem; border-radius: 1rem; line-height: 1.5; font-size: 0.95rem; }
    .user { align-self: flex-end; background: #2563eb; color: #fff; border-bottom-right-radius: 0.2rem; }
    .bot { align-self: flex-start; background: #1e293b; color: #e2e8f0; border-bottom-left-radius: 0.2rem; border: 1px solid #334155; }
    .footer { padding: 1rem 2rem; background: #1e293b; border-top: 1px solid #334155; display: flex; gap: 0.75rem; }
    input[type="text"] { flex: 1; padding: 0.75rem 1rem; border-radius: 0.5rem; border: 1px solid #475569; background: #0f172a; color: #f8fafc; font-size: 1rem; outline: none; }
    input[type="text"]:focus { border-color: #38bdf8; }
    button { padding: 0.75rem 1.5rem; border-radius: 0.5rem; border: none; background: #0284c7; color: white; font-weight: 600; cursor: pointer; transition: 0.2s; }
    button:hover { background: #0369a1; }
  </style>
</head>
<body>
  <header>
    <h1>✨ AshAI (Hugging Face Connected)</h1>
    <span class="status">● Active</span>
  </header>
  <div id="chat-box">
    <div class="msg bot">Hello! I am AshAI. Ask me anything or explore my memory and models.</div>
  </div>
  <div class="footer">
    <input type="text" id="userInput" placeholder="Ask AshAI a question..." onkeydown="if(event.key==='Enter') sendMsg()" />
    <button onclick="sendMsg()">Send</button>
  </div>
  <script>
    async function sendMsg() {
      const input = document.getElementById('userInput');
      const text = input.value.trim();
      if (!text) return;
      
      const box = document.getElementById('chat-box');
      box.innerHTML += `<div class="msg user">${text}</div>`;
      input.value = '';
      box.scrollTop = box.scrollHeight;
      
      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text })
        });
        const data = await res.json();
        box.innerHTML += `<div class="msg bot">${data.reply}</div>`;
      } catch (err) {
        box.innerHTML += `<div class="msg bot" style="color:#f87171;">Error: Unable to connect to server.</div>`;
      }
      box.scrollTop = box.scrollHeight;
    }
  </script>
</body>
</html>"""

    @app.get("/api/health")
    async def health():
        return {
            "status": "healthy",
            "app": config.app_name,
            "hf_configured": bool(config.hf_token)
        }

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest):
        model_name = req.model or config.default_model
        token = config.hf_token or os.getenv("HF_TOKEN")

        if token and InferenceClient is not None:
            try:
                client = InferenceClient(api_key=token)
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": req.message}],
                    temperature=req.temperature,
                    max_tokens=req.max_tokens,
                )
                reply = response.choices[0].message.content
                return ChatResponse(reply=reply, model=model_name)
            except Exception as e:
                return ChatResponse(
                    reply=f"AshAI Hugging Face Engine: Received response with notice: {str(e)}",
                    model=model_name
                )

        # Fallback offline mode if HF token is not yet provided
        memory_path = base_dir / "long_term_memory.json"
        knowledge_summary = "General reasoning system"
        if memory_path.exists():
            try:
                with open(memory_path, "r", encoding="utf-8") as f:
                    mem = json.load(f)
                    if isinstance(mem, list) and len(mem) > 0:
                        knowledge_summary = f"{len(mem)} memory entries loaded"
            except Exception:
                pass

        return ChatResponse(
            reply=f"AshAI (Local Core): I have received your query: '{req.message}'. (Knowledge Base: {knowledge_summary}). To enable full Hugging Face LLM generation, set your HF_TOKEN in the environment settings.",
            model="local-ashai-core"
        )

    @app.get("/api/memory")
    async def get_memory():
        memory_path = base_dir / "long_term_memory.json"
        if not memory_path.exists():
            return []
        with open(memory_path, "r", encoding="utf-8") as f:
            return json.load(f)

    return app

