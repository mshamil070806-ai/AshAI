"""
AshAI Model Context Protocol (MCP) Server
Implements standard Anthropic/Open MCP protocol over stdio for Claude, Cursor, Antigravity, and AI Agents.
"""

from __future__ import annotations
import asyncio
import json
import sys
from core.tools import (
    fetch_weather,
    fetch_crypto_prices,
    fetch_exchange_rates,
    fetch_random_advice,
    fetch_wikipedia,
    search_public_apis_catalog,
    search_long_term_memory,
    execute_smart_tools
)
from core.bilingual_engine import detect_language, format_bilingual_response

SERVER_INFO = {
    "name": "ashai-bilingual-mcp",
    "version": "2.0.0"
}

TOOLS = [
    {
        "name": "ashai_bilingual_chat",
        "description": "Bilingual (English, Malayalam, Manglish) conversational AI with access to 51,900+ knowledge records and real-time tools.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The user query in English, Malayalam, or Manglish."
                }
            },
            "required": ["message"]
        }
    },
    {
        "name": "get_weather",
        "description": "Fetch real-time weather information for any city worldwide.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city or location name (e.g., 'Kochi', 'London', 'Dubai')."
                }
            },
            "required": ["location"]
        }
    },
    {
        "name": "get_crypto_prices",
        "description": "Get real-time live cryptocurrency prices (BTC, ETH, SOL, DOGE) from Binance.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "get_currency_rates",
        "description": "Get live foreign exchange rates (USD to INR, EUR, AED, SAR, GBP).",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "search_wikipedia",
        "description": "Search Wikipedia summaries on any topic, person, science concept, or historical event.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search term or topic name."
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "search_knowledge_base",
        "description": "Search AshAI's 51,900+ Hugging Face records across Science, Coding, and General QA.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords or topic to search."
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to retrieve (default 3)."
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "search_public_apis",
        "description": "Search 2,400+ free public APIs from github.com/public-apis catalog.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "API category or topic (e.g., 'sports', 'crypto', 'weather')."
                }
            },
            "required": ["query"]
        }
    }
]

async def handle_tool_call(name: str, arguments: dict) -> str:
    if name == "ashai_bilingual_chat":
        msg = arguments.get("message", "")
        lang = detect_language(msg)
        result = await execute_smart_tools(msg)
        content = result.get("content") or "AshAI processed your query."
        return format_bilingual_response(lang, content)

    elif name == "get_weather":
        loc = arguments.get("location", "Kochi")
        return await fetch_weather(loc)

    elif name == "get_crypto_prices":
        return await fetch_crypto_prices()

    elif name == "get_currency_rates":
        return await fetch_exchange_rates()

    elif name == "search_wikipedia":
        q = arguments.get("query", "")
        res = await fetch_wikipedia(q)
        return res or f"No Wikipedia entry found for '{q}'."

    elif name == "search_knowledge_base":
        q = arguments.get("query", "")
        top_k = arguments.get("top_k", 3)
        records = search_long_term_memory(q, top_k=top_k)
        if not records:
            return f"No records found in knowledge base for '{q}'."
        snippets = [f"[{r.get('category')}]\n{r.get('content')}" for r in records]
        return "\n\n---\n\n".join(snippets)

    elif name == "search_public_apis":
        q = arguments.get("query", "")
        apis = search_public_apis_catalog(q, max_results=5)
        if not apis:
            return f"No public APIs found matching '{q}'."
        lines = [f"• {a['name']} ({a['category']}): {a['description']} (Link: {a['url']})" for a in apis]
        return "\n".join(lines)

    return f"Unknown tool: {name}"

async def process_json_rpc(line: str) -> dict | None:
    try:
        req = json.loads(line)
    except Exception:
        return None

    msg_id = req.get("id")
    method = req.get("method")
    params = req.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": SERVER_INFO
            }
        }

    elif method == "notifications/initialized":
        return None

    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "tools": TOOLS
            }
        }

    elif method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        try:
            result_text = await handle_tool_call(tool_name, arguments)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": result_text
                        }
                    ],
                    "isError": False
                }
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Error executing tool {tool_name}: {str(e)}"
                        }
                    ],
                    "isError": True
                }
            }

    elif method == "ping":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

    # Unknown method
    if msg_id is not None:
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {
                "code": -32601,
                "message": f"Method {method} not found"
            }
        }
    return None

def main():
    """Robust cross-platform stdio transport for MCP server."""
    for line in sys.stdin:
        line_clean = line.strip()
        if not line_clean:
            continue
        try:
            response = asyncio.run(process_json_rpc(line_clean))
            if response is not None:
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
        except Exception as e:
            sys.stderr.write(f"MCP error: {e}\n")
            sys.stderr.flush()

if __name__ == "__main__":
    main()
