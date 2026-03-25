"""
Guardrail node — classifies each user message before the main agent sees it.
Uses gpt-5.4-mini for speed and cost efficiency.
Returns: "in_scope" | "out_of_scope" | "unsafe"
"""

import os
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

_SYSTEM_PROMPT = """You are a content classifier for a women's clothing retail assistant.

Classify the user's message into exactly one of these categories:
- in_scope: anything related to clothing, fashion, products, styling, outfits, or shopping
- out_of_scope: off-topic requests unrelated to clothing (e.g. "write my CV", "what's the weather")
- unsafe: harmful, abusive, or inappropriate content

Respond with only one word: in_scope, out_of_scope, or unsafe."""

_llm: AzureChatOpenAI | None = None


def _get_llm() -> AzureChatOpenAI:
    global _llm
    if _llm is None:
        _llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_CHAT_MINI_DEPLOYMENT", "gpt-5.4-mini"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            temperature=0,
            max_tokens=10,
        )
    return _llm


def classify(user_message: str) -> str:
    """Returns 'in_scope', 'out_of_scope', or 'unsafe'."""
    from openai import BadRequestError
    try:
        response = _get_llm().invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ])
    except BadRequestError as e:
        if e.code == "content_filter":
            # Azure content filter already determined the message is unsafe
            return "unsafe"
        raise
    result = response.content.strip().lower()
    if result not in ("in_scope", "out_of_scope", "unsafe"):
        return "in_scope"
    return result
