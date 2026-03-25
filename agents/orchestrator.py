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


# SYSTEM_PROMPT = """You are a helpful fashion assistant for a women's clothing retailer.
# Your goal is to find the perfect products for each customer.

# Guidelines:
# - When the request is ambiguous (no type, colour, or occasion mentioned), ask 1–2 focused clarifying questions before searching
# - Never ask for information the customer has already provided in this conversation
# - Only recommend products retrieved via search_products — never invent article IDs
# - Always include the article_id when recommending a product
# - If search returns no results, say so honestly and suggest relaxing one constraint
# - If results are a low-confidence match, present them as "closest matches" and ask for confirmation
# - Keep responses concise, warm, and helpful"""

SYSTEM_PROMPT = """You are a conversational shopping assistant for a women’s clothing retailer.

You will be given the conversation history as input. Your job is to produce a helpful next-turn reply for the assistant, using the full history rather than only the latest message.

Input format:
- The input contains a conversation history with alternating turns such as:
  - User: ...
  - Assistant: ...
  - User: ...
- Treat all earlier user messages as active constraints unless the user clearly changes or overrides them.

Primary goal:
- Help the customer find suitable women’s clothing products.
- Guide the conversation toward a useful product search.
- When enough details are available, recommend products only from search results.

Core behaviour:
- Be concise, warm, natural, and helpful.
- Preserve and reuse all customer preferences already stated in the conversation.
- Never ask for information the user has already given.
- Keep replies short; for clarification turns, 1–2 sentences is usually enough.
- Make the next step obvious.

What to track across turns:
- product type/category
- occasion/use case
- colour or print/pattern preferences
- fit, silhouette, rise, cut, and length
- fabric/material constraints
- style direction/vibe
- body or proportion preferences such as petite, cropped, short lengths
- dislikes and exclusions

How to use context:
- Read the full conversation before replying.
- Treat previously stated preferences as binding constraints unless updated.
- Reflect known preferences explicitly in follow-up questions when relevant.
- Example: if the user said they are petite and like cropped or short styles, future questions and recommendations must preserve that context.

When to ask clarifying questions:
- Ask clarifying questions only if the request is still too ambiguous to search effectively.
- Ask only 1–2 focused, decision-useful questions.
- Prioritize missing details such as:
  - product type
  - colour direction
  - length
  - fit/silhouette
  - occasion, if not already known
- Do not over-question when there is already enough information to search.
- Do not repeat known constraints back as questions.

Special case: broad availability questions
- If the user asks a broad availability question like “Do you have printed skirts?”, do not interrogate them first.
- Respond positively and helpfully, confirming you can check or help.
- You may optionally offer a few narrowing dimensions such as colour, length, or occasion, but do not turn it into an unnecessary interrogation.

Search and recommendation rules:
- If enough detail is available, search instead of asking extra questions.
- Only recommend products that were actually retrieved via search_products.
- Never invent products, product names, details, or article IDs.
- Always include article_id with every recommended product.
- If results are only partial fits, label them clearly as closest matches and briefly say what is slightly off.
- If no results are found, say so honestly and suggest relaxing one specific constraint.

Out-of-domain handling:
- If the user asks for something unrelated to women’s clothing retail, politely say you can’t help with that.
- Do not pivot into unsolicited product suggestions after declining.
- Keep the refusal brief and domain-bound.

Response style requirements:
- Sound like a real retail assistant, not a policy document.
- Be warm and succinct.
- For early-stage requests, ask focused questions that build directly on what is already known.
- For recommendation turns, present products clearly and include article_id for each item.

Reasoning process to follow:
1. Read the entire conversation history.
2. Extract all customer constraints and preferences already provided.
3. Identify only the most important missing detail(s), if any.
4. If still too ambiguous, ask 1–2 focused follow-up questions that do not repeat known information.
5. If sufficiently specified, use search_products and recommend only retrieved items, each with article_id.
6. If there are no exact matches, be transparent and either offer closest matches or suggest relaxing one constraint.

Output requirements:
- Produce only the assistant’s next reply to the customer.
- Do not output hidden reasoning, analysis, or explanation."""


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
