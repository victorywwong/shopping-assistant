"""
LangGraph orchestrator — the main conversational agent.

Graph structure:
    [START] → guardrail_node
                  ↓ in_scope          ↓ out_of_scope / unsafe
              agent_node           refusal_node → [END]
              ↙         ↘
        tool_node     [END]
              ↓
          agent_node (loop until no tool calls)
"""

import os
import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

from agents.guardrails import classify
from agents.product_retriever import search_products, get_article_details

load_dotenv()

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ConversationState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    guardrail_result: str  # "in_scope" | "out_of_scope" | "unsafe"


# ---------------------------------------------------------------------------
# LLM + tools
# ---------------------------------------------------------------------------

# analyze_image is intentionally excluded: images are passed as multimodal
# HumanMessages so the LLM sees them directly — no tool call needed.
TOOLS = [search_products, get_article_details]

_llm: AzureChatOpenAI | None = None


def _get_llm() -> AzureChatOpenAI:
    global _llm
    if _llm is None:
        _llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-5.4"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            temperature=0.3,
        )
    return _llm


SYSTEM_PROMPT = """You are a helpful fashion assistant for a women's clothing retailer.
Your goal is to find the perfect products for each customer.

Guidelines:
- When the request is ambiguous (no type, colour, or occasion mentioned), ask 1–2 focused clarifying questions before searching
- Never ask for information the customer has already provided in this conversation
- Only recommend products retrieved via search_products — never invent article IDs
- Always include the article_id when recommending a product
- If search returns no results, say so honestly and suggest relaxing one constraint
- If results are a low-confidence match, present them as "closest matches" and ask for confirmation
- Keep responses concise, warm, and helpful"""


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def guardrail_node(state: ConversationState) -> dict:
    last_user_message = next(
        (m.content for m in reversed(state["messages"]) if m.type == "human"),
        "",
    )
    result = classify(last_user_message)
    return {"guardrail_result": result}


def agent_node(state: ConversationState) -> dict:
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = _get_llm().bind_tools(TOOLS).invoke(messages)
    return {"messages": [response]}


def refusal_node(state: ConversationState) -> dict:
    result = state["guardrail_result"]
    if result == "unsafe":
        text = "I'm not able to help with that."
    else:
        text = "I can only help with clothing and fashion questions. Is there something I can help you find today?"
    return {"messages": [AIMessage(content=text)]}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_guardrail(state: ConversationState) -> str:
    return "agent" if state["guardrail_result"] == "in_scope" else "refusal"


def route_after_agent(state: ConversationState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(ConversationState)

    graph.add_node("guardrail", guardrail_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("refusal", refusal_node)

    graph.set_entry_point("guardrail")

    graph.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {"agent": "agent", "refusal": "refusal"},
    )
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("refusal", END)

    return graph.compile(checkpointer=MemorySaver())


# Singleton graph — compiled once at import time
graph = build_graph()


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def chat(message: str, thread_id: str, image_base64: str | None = None) -> str:
    """
    Send a message and get a response. thread_id isolates each user session.
    Optionally attach a base64-encoded image.
    """
    if image_base64:
        human_msg = HumanMessage(
            content=[
                {"type": "text", "text": message or "What is this item?"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
            ]
        )
    else:
        human_msg = HumanMessage(content=message)

    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {"messages": [human_msg], "guardrail_result": ""},
        config=config,
    )

    return result["messages"][-1].content
