"""
AshAI Bilingual Engine (English & Malayalam)
Provides bilingual comprehension (English, Malayalam script, Manglish phonetics),
and multimodal (LMM) capabilities.
"""

from __future__ import annotations
import re
from typing import Tuple

# Common Manglish phonetics and vocabulary mappings
MANGLISH_PATTERNS = [
    (r"\b(entha|enthanu|enthokke|entharu)\b", "what"),
    (r"\b(engane|enganeya)\b", "how"),
    (r"\b(eppozha|eppol)\b", "when"),
    (r"\b(evide|evidaya)\b", "where"),
    (r"\b(aara|aaranu)\b", "who"),
    (r"\b(mazha|choodu|kaathu)\b", "weather"),
    (r"\b(vilaya|vila|ethraya|ethra)\b", "price"),
    (r"\b(nanni|namaskaram|sukhamano|vishesham)\b", "greeting"),
    (r"\b(sahayikkyo|sahayam)\b", "help"),
]

def detect_language(text: str) -> str:
    """
    Detects if text is Malayalam script ('ml'), Manglish phonetics ('manglish'), or English ('en').
    """
    # Check for Malayalam Unicode range (0x0D00 - 0x0D7F)
    if any("\u0d00" <= char <= "\u0d7f" for char in text):
        return "ml"

    text_lower = text.lower()
    manglish_score = 0
    for pattern, _ in MANGLISH_PATTERNS:
        if re.search(pattern, text_lower):
            manglish_score += 1

    if manglish_score > 0:
        return "manglish"

    return "en"

def format_bilingual_response(lang: str, english_text: str, ml_summary: str | None = None) -> str:
    """
    Returns an appropriate response based on the detected language preference.
    """
    if lang == "ml":
        if ml_summary:
            return f"{ml_summary}\n\n(English Reference:\n{english_text})"
        return f"ഇതാ നിങ്ങളുടെ ചോദ്യത്തിനുള്ള വിവരങ്ങൾ:\n\n{english_text}"
    
    if lang == "manglish":
        if ml_summary:
            return f"{ml_summary}\n\n(Details:\n{english_text})"
        return f"Itha ningal chodhicha details:\n\n{english_text}"

    return english_text

