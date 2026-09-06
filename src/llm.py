from __future__ import annotations

import os
from typing import Any


def get_groq_chat_model() -> Any | None:
    if os.getenv("USE_LLM", "1").lower() in {"0", "false", "no"}:
        return None
    if not os.getenv("GROQ_API_KEY"):
        return None
    try:
        from langchain_groq import ChatGroq
    except ImportError:
        return None

    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        temperature=0,
    )
