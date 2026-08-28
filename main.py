"""
AshAI – Application Entry Point

Run with:
    python main.py
    # or
    uvicorn main:app --reload
"""

from __future__ import annotations

import uvicorn

from core.config import get_config
from backend.server import create_app

config = get_config()
app = create_app(config)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
    )
