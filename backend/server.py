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
        description="Bilingual LLM/LMM Assistant with Model Context Protocol & 51.9k Knowledge Base",
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
  <title>AshAI – Next-Gen Bilingual AI Studio</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-base: #060913;
      --bg-surface: #0d1322;
      --bg-card: rgba(18, 26, 44, 0.75);
      --bg-glass: rgba(15, 23, 42, 0.65);
      --border-subtle: rgba(56, 189, 248, 0.12);
      --border-hover: rgba(56, 189, 248, 0.35);
      --primary: #38bdf8;
      --primary-glow: rgba(56, 189, 248, 0.25);
      --accent: #6366f1;
      --accent-gradient: linear-gradient(135deg, #38bdf8 0%, #6366f1 50%, #a855f7 100%);
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --radius-sm: 8px;
      --radius-md: 14px;
      --radius-lg: 22px;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
      background: var(--bg-base);
      color: var(--text-main);
      display: flex;
      height: 100vh;
      overflow: hidden;
      background-image: 
        radial-gradient(circle at 15% 15%, rgba(56, 189, 248, 0.08) 0%, transparent 40%),
        radial-gradient(circle at 85% 85%, rgba(99, 102, 241, 0.08) 0%, transparent 40%);
    }

    /* Sidebar */
    aside {
      width: 290px;
      background: var(--bg-surface);
      border-right: 1px solid var(--border-subtle);
      display: flex;
      flex-direction: column;
      padding: 1.5rem 1.25rem;
      gap: 1.5rem;
      backdrop-filter: blur(20px);
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      z-index: 20;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      text-decoration: none;
    }
    .brand-logo {
      width: 42px;
      height: 42px;
      border-radius: var(--radius-md);
      background: var(--accent-gradient);
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 0 20px var(--primary-glow);
      font-size: 1.3rem;
      font-weight: 800;
      color: #fff;
    }
    .brand-title {
      font-size: 1.2rem;
      font-weight: 800;
      letter-spacing: -0.02em;
      background: linear-gradient(180deg, #ffffff 0%, #cbd5e1 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .brand-badge {
      font-size: 0.65rem;
      text-transform: uppercase;
      font-weight: 700;
      color: #38bdf8;
      background: rgba(56, 189, 248, 0.1);
      border: 1px solid rgba(56, 189, 248, 0.25);
      padding: 2px 6px;
      border-radius: 99px;
      letter-spacing: 0.05em;
    }

    .new-chat-btn {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
      padding: 0.75rem 1rem;
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      color: var(--text-main);
      font-weight: 600;
      font-size: 0.9rem;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .new-chat-btn:hover {
      background: rgba(56, 189, 248, 0.1);
      border-color: var(--primary);
      transform: translateY(-1px);
    }

    .stats-card {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 1rem;
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }
    .stats-title {
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .stat-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 0.85rem;
      color: #e2e8f0;
    }
    .stat-val {
      font-family: 'JetBrains Mono', monospace;
      font-weight: 600;
      color: #38bdf8;
    }

    .sidebar-footer {
      margin-top: auto;
      display: flex;
      flex-direction: column;
      gap: 0.6rem;
      font-size: 0.8rem;
      color: var(--text-muted);
    }
    .live-pulse {
      display: inline-block;
      width: 8px;
      height: 8px;
      background: #10b981;
      border-radius: 50%;
      box-shadow: 0 0 10px #10b981;
      animation: pulse 2s infinite;
    }
    @keyframes pulse {
      0% { transform: scale(0.95); opacity: 0.7; }
      50% { transform: scale(1.2); opacity: 1; }
      100% { transform: scale(0.95); opacity: 0.7; }
    }

    /* Main Chat Arena */
    main {
      flex: 1;
      display: flex;
      flex-direction: column;
      height: 100vh;
      position: relative;
    }

    header {
      padding: 1.1rem 2rem;
      background: rgba(13, 19, 34, 0.75);
      backdrop-filter: blur(20px);
      border-bottom: 1px solid var(--border-subtle);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .header-left {
      display: flex;
      align-items: center;
      gap: 1rem;
    }
    .header-pills {
      display: flex;
      gap: 0.5rem;
    }
    .header-pill {
      font-size: 0.75rem;
      padding: 0.25rem 0.65rem;
      border-radius: 99px;
      font-weight: 600;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--border-subtle);
      color: #cbd5e1;
      display: flex;
      align-items: center;
      gap: 0.35rem;
    }
    .header-pill.mcp {
      background: rgba(99, 102, 241, 0.15);
      border-color: rgba(99, 102, 241, 0.35);
      color: #a5b4fc;
    }

    #chat-scroll {
      flex: 1;
      overflow-y: auto;
      padding: 2rem;
      display: flex;
      flex-direction: column;
      gap: 1.5rem;
      scroll-behavior: smooth;
    }

    .msg-group {
      display: flex;
      gap: 1rem;
      max-width: 850px;
      margin: 0 auto;
      width: 100%;
    }
    .msg-group.user {
      justify-content: flex-end;
    }
    .msg-group.bot {
      justify-content: flex-start;
    }

    .avatar {
      width: 36px;
      height: 36px;
      border-radius: var(--radius-sm);
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 700;
      font-size: 0.9rem;
      flex-shrink: 0;
    }
    .bot .avatar {
      background: var(--accent-gradient);
      box-shadow: 0 0 12px var(--primary-glow);
      color: #fff;
    }
    .user .avatar {
      background: #2563eb;
      color: #fff;
    }

    .bubble-wrapper {
      display: flex;
      flex-direction: column;
      max-width: 78%;
    }
    .bubble-meta {
      font-size: 0.75rem;
      color: var(--text-muted);
      margin-bottom: 0.35rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-weight: 600;
    }
    .tool-badge {
      display: inline-flex;
      align-items: center;
      gap: 0.3rem;
      background: rgba(56, 189, 248, 0.12);
      border: 1px solid rgba(56, 189, 248, 0.25);
      color: #7dd3fc;
      padding: 2px 7px;
      border-radius: 99px;
      font-size: 0.7rem;
    }

    .bubble {
      padding: 1rem 1.35rem;
      border-radius: var(--radius-md);
      font-size: 0.95rem;
      line-height: 1.65;
      white-space: pre-wrap;
      word-break: break-word;
      position: relative;
    }
    .user .bubble {
      background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
      color: #ffffff;
      border-bottom-right-radius: 4px;
      box-shadow: 0 4px 15px rgba(37, 99, 235, 0.2);
    }
    .bot .bubble {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      color: #e2e8f0;
      border-bottom-left-radius: 4px;
      backdrop-filter: blur(12px);
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }

    /* Copy Button */
    .copy-btn {
      position: absolute;
      top: 0.5rem;
      right: 0.5rem;
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid var(--border-subtle);
      border-radius: 6px;
      color: var(--text-muted);
      padding: 3px 8px;
      font-size: 0.7rem;
      cursor: pointer;
      opacity: 0;
      transition: opacity 0.2s ease;
    }
    .bubble:hover .copy-btn {
      opacity: 1;
    }
    .copy-btn:hover {
      color: #fff;
      background: rgba(255, 255, 255, 0.18);
    }

    /* Quick Action Chips */
    .chips-bar {
      max-width: 850px;
      margin: 0 auto;
      width: 100%;
      padding: 0.5rem 2rem;
      display: flex;
      gap: 0.6rem;
      overflow-x: auto;
      scrollbar-width: none;
    }
    .chips-bar::-webkit-scrollbar { display: none; }
    .chip {
      padding: 0.45rem 0.9rem;
      background: rgba(18, 26, 44, 0.8);
      border: 1px solid var(--border-subtle);
      border-radius: 99px;
      font-size: 0.8rem;
      font-weight: 500;
      color: #94a3b8;
      cursor: pointer;
      white-space: nowrap;
      transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
      backdrop-filter: blur(10px);
    }
    .chip:hover {
      background: rgba(56, 189, 248, 0.12);
      border-color: var(--primary);
      color: #38bdf8;
      transform: translateY(-1px);
    }

    /* Input Footer */
    .input-container {
      max-width: 850px;
      margin: 0 auto;
      width: 100%;
      padding: 0 2rem 1.5rem 2rem;
    }
    .input-box {
      background: rgba(18, 26, 44, 0.85);
      border: 1px solid rgba(56, 189, 248, 0.2);
      border-radius: var(--radius-lg);
      padding: 0.6rem 0.75rem 0.6rem 1.25rem;
      display: flex;
      align-items: center;
      gap: 0.75rem;
      backdrop-filter: blur(25px);
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4), 0 0 15px rgba(56, 189, 248, 0.05);
      transition: all 0.25s ease;
    }
    .input-box:focus-within {
      border-color: var(--primary);
      box-shadow: 0 10px 35px rgba(0, 0, 0, 0.5), 0 0 25px var(--primary-glow);
    }

    textarea {
      flex: 1;
      background: transparent;
      border: none;
      outline: none;
      color: var(--text-main);
      font-family: inherit;
      font-size: 0.95rem;
      resize: none;
      height: 24px;
      max-height: 140px;
      line-height: 24px;
    }
    textarea::placeholder {
      color: #64748b;
    }

    .send-btn {
      width: 42px;
      height: 42px;
      border-radius: var(--radius-md);
      background: var(--accent-gradient);
      border: none;
      color: #ffffff;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      transition: all 0.2s ease;
      flex-shrink: 0;
      box-shadow: 0 4px 12px var(--primary-glow);
    }
    .send-btn:hover {
      transform: scale(1.05);
      box-shadow: 0 6px 18px rgba(56, 189, 248, 0.4);
    }
    .send-btn svg {
      width: 18px;
      height: 18px;
      fill: none;
      stroke: currentColor;
      stroke-width: 2.2;
      stroke-linecap: round;
      stroke-linejoin: round;
    }

    /* Typing Loader */
    .typing-indicator {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 0.75rem 1rem;
    }
    .typing-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: #38bdf8;
      animation: bounce 1.4s infinite ease-in-out;
    }
    .typing-dot:nth-child(1) { animation-delay: -0.32s; }
    .typing-dot:nth-child(2) { animation-delay: -0.16s; }
    @keyframes bounce {
      0%, 80%, 100% { transform: scale(0); opacity: 0.3; }
      40% { transform: scale(1); opacity: 1; }
    }
  </style>
</head>
<body>

  <!-- Sidebar -->
  <aside>
    <a href="/" class="brand">
      <div class="brand-logo">⚡</div>
      <div>
        <div style="display:flex; align-items:center; gap:0.4rem;">
          <span class="brand-title">AshAI</span>
          <span class="brand-badge">PRO</span>
        </div>
        <div style="font-size:0.75rem; color:#64748b;">Bilingual AI & MCP Engine</div>
      </div>
    </a>

    <button class="new-chat-btn" onclick="clearChat()">
      <span>✨</span>
      <span>New Conversation</span>
    </button>

    <div class="stats-card">
      <div class="stats-title">
        <span>Knowledge Hub</span>
        <span class="live-pulse"></span>
      </div>
      <div class="stat-row">
        <span>Curated Records</span>
        <span class="stat-val">51,917</span>
      </div>
      <div class="stat-row">
        <span>Free Public APIs</span>
        <span class="stat-val">2,405</span>
      </div>
      <div class="stat-row">
        <span>Languages</span>
        <span class="stat-val">ML • EN</span>
      </div>
      <div class="stat-row">
        <span>MCP Protocols</span>
        <span class="stat-val">v2.5.0</span>
      </div>
    </div>

    <div class="stats-card">
      <div class="stats-title">
        <span>Active Live Tools</span>
      </div>
      <div class="stat-row" style="font-size:0.8rem;">
        <span>🌦️ Global Weather</span>
        <span style="color:#10b981;">Online</span>
      </div>
      <div class="stat-row" style="font-size:0.8rem;">
        <span>🪙 Binance Crypto</span>
        <span style="color:#10b981;">Online</span>
      </div>
      <div class="stat-row" style="font-size:0.8rem;">
        <span>💵 Forex Rates</span>
        <span style="color:#10b981;">Online</span>
      </div>
      <div class="stat-row" style="font-size:0.8rem;">
        <span>📚 Wikipedia REST</span>
        <span style="color:#10b981;">Online</span>
      </div>
    </div>

    <div class="sidebar-footer">
      <div>Created for <strong>M Shamil</strong></div>
      <div style="font-size:0.7rem; color:#475569;">Repository: mshamil070806-ai/AshAI</div>
    </div>
  </aside>

  <!-- Main Chat Arena -->
  <main>
    <header>
      <div class="header-left">
        <div style="font-weight:700; font-size:1.05rem;">Bilingual Intelligence Studio</div>
      </div>
      <div class="header-pills">
        <div class="header-pill mcp">🔌 MCP Server Active</div>
        <div class="header-pill">🌐 Malayalam & English</div>
        <div class="header-pill">⚡ Port 7860</div>
      </div>
    </header>

    <div id="chat-scroll">
      <div class="msg-group bot">
        <div class="avatar">⚡</div>
        <div class="bubble-wrapper">
          <div class="bubble-meta">
            <span class="tool-badge">✨ AshAI Bilingual Intelligence</span>
            <span>Just now</span>
          </div>
          <div class="bubble">
നമസ്കാരം! I am <strong>AshAI</strong> — your bilingual AI assistant and Model Context Protocol (MCP) server.

നിങ്ങൾക്ക് മലയാളത്തിലോ (മലയാളം ലിപി / മംഗ്ലീഷ്), ഇംഗ്ലീഷിലോ എന്തും ചോദിക്കാം:
• <strong>ലൈവ് കാലാവസ്ഥ:</strong> "കൊച്ചിയിലെ കാലാവസ്ഥ എങ്ങനെയുണ്ട്?"
• <strong>ക്രിപ്റ്റോ & കറൻസി:</strong> "Live Bitcoin price" അല്ലെങ്കിൽ "Dollar to INR rate"
• <strong>ശാസ്ത്രം & കോഡിംഗ്:</strong> 51,900+ Hugging Face നോളജ് റെക്കോർഡുകൾ സജ്ജമാണ്!
          </div>
        </div>
      </div>
    </div>

    <!-- Quick Action Chips -->
    <div class="chips-bar">
      <div class="chip" onclick="quickSend('കൊച്ചിയിലെ കാലാവസ്ഥ എങ്ങനെയുണ്ട്?')">🌦️ കൊച്ചിയിലെ കാലാവസ്ഥ</div>
      <div class="chip" onclick="quickSend('Live crypto prices')">🪙 Bitcoin & Crypto Price</div>
      <div class="chip" onclick="quickSend('Current dollar to rupee rate')">💵 ഡോളർ വിനിമയ നിരക്ക് (INR)</div>
      <div class="chip" onclick="quickSend('Who is Alan Turing?')">👤 Who is Alan Turing?</div>
      <div class="chip" onclick="quickSend('Python function to reverse string')">💻 Python QuickSort / Code</div>
      <div class="chip" onclick="quickSend('Find free APIs for sports')">🌐 Free Public APIs</div>
      <div class="chip" onclick="quickSend('Inspirational quote')">💡 Daily Thought</div>
    </div>

    <!-- Input Footer -->
    <div class="input-container">
      <div class="input-box">
        <textarea id="userInput" placeholder="മലയാളത്തിലോ ഇംഗ്ലീഷിലോ എന്തും ചോദിക്കാം (Type a question...)" rows="1" onkeydown="handleKeyDown(event)"></textarea>
        <button class="send-btn" onclick="sendMsg()" title="Send">
          <svg viewBox="0 0 24 24"><path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"></path></svg>
        </button>
      </div>
    </div>
  </main>

  <script>
    function quickSend(txt) {
      document.getElementById('userInput').value = txt;
      sendMsg();
    }

    function handleKeyDown(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMsg();
      }
    }

    function clearChat() {
      const scroll = document.getElementById('chat-scroll');
      scroll.innerHTML = `
        <div class="msg-group bot">
          <div class="avatar">⚡</div>
          <div class="bubble-wrapper">
            <div class="bubble-meta">
              <span class="tool-badge">✨ New Conversation</span>
              <span>Just now</span>
            </div>
            <div class="bubble">New chat started. How can I help you today? (എന്താണ് അറിയേണ്ടത്?)</div>
          </div>
        </div>`;
    }

    async function sendMsg() {
      const input = document.getElementById('userInput');
      const text = input.value.trim();
      if (!text) return;

      const scroll = document.getElementById('chat-scroll');
      
      // Add User Message
      scroll.innerHTML += `
        <div class="msg-group user">
          <div class="bubble-wrapper">
            <div class="bubble-meta" style="justify-content: flex-end;">
              <span>You</span>
            </div>
            <div class="bubble">${escapeHtml(text)}</div>
          </div>
          <div class="avatar">👤</div>
        </div>`;

      input.value = '';
      scroll.scrollTop = scroll.scrollHeight;

      // Add Loading Indicator
      const loadingId = 'loading-' + Date.now();
      scroll.innerHTML += `
        <div class="msg-group bot" id="${loadingId}">
          <div class="avatar">⚡</div>
          <div class="bubble-wrapper">
            <div class="bubble">
              <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
              </div>
            </div>
          </div>
        </div>`;
      scroll.scrollTop = scroll.scrollHeight;

      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text })
        });
        const data = await res.json();
        
        const loader = document.getElementById(loadingId);
        if (loader) loader.remove();

        const toolLabel = data.tool_used ? `🔧 ${data.tool_used}` : '⚡ AshAI Core';
        const langTag = (data.language || 'en').toUpperCase();
        const copyContent = escapeHtml(data.reply).replace(/"/g, '&quot;');

        scroll.innerHTML += `
          <div class="msg-group bot">
            <div class="avatar">⚡</div>
            <div class="bubble-wrapper">
              <div class="bubble-meta">
                <span class="tool-badge">${toolLabel}</span>
                <span>${langTag}</span>
              </div>
              <div class="bubble">
                <button class="copy-btn" onclick="copyText(this)">Copy</button>
                ${escapeHtml(data.reply)}
              </div>
            </div>
          </div>`;
      } catch (err) {
        const loader = document.getElementById(loadingId);
        if (loader) loader.remove();

        scroll.innerHTML += `
          <div class="msg-group bot">
            <div class="avatar">⚡</div>
            <div class="bubble-wrapper">
              <div class="bubble" style="color:#f87171;">
                Error: Unable to connect to server. Please ensure AshAI server is running.
              </div>
            </div>
          </div>`;
      }
      scroll.scrollTop = scroll.scrollHeight;
    }

    function copyText(btn) {
      const text = btn.parentElement.innerText.replace('Copy', '').trim();
      navigator.clipboard.writeText(text);
      btn.innerText = 'Copied!';
      setTimeout(() => btn.innerText = 'Copy', 1500);
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

        lang = detect_language(req.message)
        tool_result = await execute_smart_tools(req.message)
        tool_name = tool_result.get("tool")
        tool_content = tool_result.get("content")

        if tool_content:
            formatted_reply = format_bilingual_response(lang, tool_content)
            return ChatResponse(reply=formatted_reply, model="ashai-bilingual-engine", tool_used=tool_name, language=lang)

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
