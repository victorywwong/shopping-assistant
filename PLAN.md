# Implementation Plan: Conversational Product Recommendation Agent

## Overview

A multi-turn, multimodal conversational agent that recommends women's clothing products from `articles.csv`. Built in Python using the Azure OpenAI API, with semantic vector search for product retrieval, a Streamlit chat UI, and optional image understanding. Agent orchestration is handled by **LangGraph** with **LangChain** for tool binding and Azure OpenAI integration.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit Chat UI                   │
│              (app.py / chat_interface.py)            │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│               Orchestrator Agent                     │
│  - Manages conversation state & history             │
│  - Decides when to clarify vs. recommend            │
│  - Routes to sub-agents as tools                    │
└──────┬──────────────┬──────────────────┬────────────┘
       │              │                  │
       ▼              ▼                  ▼
┌──────────┐  ┌──────────────┐  ┌────────────────┐
│ Product  │  │    Image     │  │   Guardrail    │
│ Retriever│  │  Analyzer    │  │   Checker      │
│ (search) │  │ (multimodal) │  │ (scope filter) │
└──────────┘  └──────────────┘  └────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────┐
│         Product Catalog + Vector Index               │
│         (articles.csv + embeddings cache)            │
└──────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
DigitalGenius/
├── app.py                     # Streamlit chat interface (entry point)
├── docker-compose.yml         # Spins up Postgres + pgvector
├── requirements.txt
├── articles.csv
├── images/                    # Unzipped product images
├── data/
│   ├── catalog.py             # Loads and wraps articles.csv
│   ├── db.py                  # Postgres connection, schema creation, upsert helpers
│   └── embeddings.py          # Generates embeddings & stores them in pgvector
├── agents/
│   ├── orchestrator.py        # LangGraph StateGraph — orchestrates all agent nodes
│   ├── product_retriever.py   # Semantic search tool
│   └── guardrails.py          # Out-of-scope / unsafe request filter
├── eval/
│   ├── test_cases.py          # Predefined conversation scenarios
│   └── evaluator.py           # Automated evaluation runner
└── THOUGHTS.md                # Design rationale & decisions
```

---

## Implementation Steps

### Step 1 — Data Preparation (`data/`)

**`catalog.py`**
- Load `articles.csv` with pandas
- Expose: `get_article(id)`, `search_by_filter(color, type, group)`, `all_articles()`
- Columns: `article_id`, `prod_name`, `product_type_name`, `product_group_name`, `graphical_appearance_name`, `colour_group_name`, `garment_group_name`, `detail_desc`

**`db.py`**
- Manages the Postgres connection (via `psycopg2` + `pgvector`)
- On startup: runs `CREATE EXTENSION IF NOT EXISTS vector` and creates the `articles` table:
  ```sql
  CREATE TABLE IF NOT EXISTS articles (
      article_id   TEXT PRIMARY KEY,
      prod_name    TEXT,
      product_type_name TEXT,
      product_group_name TEXT,
      colour_group_name TEXT,
      garment_group_name TEXT,
      graphical_appearance_name TEXT,
      detail_desc  TEXT,
      embedding    vector(1024)  -- 1024-dim: text-embedding-3-large with dimensions=1024
  );
  CREATE INDEX IF NOT EXISTS articles_embedding_idx
      ON articles USING hnsw (embedding vector_cosine_ops);
  ```
- Exposes: `get_connection()`, `upsert_article(row, embedding)`, `is_populated() -> bool`

**`embeddings.py`**
- Generates text embeddings using Azure OpenAI `text-embedding-3-large` with `dimensions=1024`
- Embed a combined field: `"{prod_name} {product_type_name} {colour_group_name} {graphical_appearance_name} {detail_desc}"`
- On first run: iterates all articles, upserts embeddings into Postgres via `db.py`; skips if `db.is_populated()` returns true
- Expose: `similarity_search(query_text, top_k=5, filters={}) -> list[dict]`
  - Builds a single SQL query combining vector cosine similarity with optional `WHERE` filters (colour, product type, garment group)
  - Example: `SELECT ... FROM articles WHERE colour_group_name = %s ORDER BY embedding <=> %s LIMIT %s`

**`docker-compose.yml`**
```yaml
services:
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: digitalgenius
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5433:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
volumes:
  pgdata:
```

### Step 2 — Product Retriever Agent (`agents/product_retriever.py`)

- Tool the orchestrator can call: `search_products(query, filters={})`
- Runs semantic similarity search via embeddings
- Optionally narrows results with structured filters (color, garment type)
- Returns top-K matching articles with `article_id`, `prod_name`, `colour_group_name`, `detail_desc`
- Handles the "no results" failure mode: returns empty list + fallback message

### Step 3 — Image Understanding (multimodal, inline)

- Images are passed directly as multimodal `HumanMessage` content to `gpt-5.4`
- No separate image analyzer agent — the LLM sees the image inline and describes it naturally before calling `search_products`
- `app.py` converts uploaded bytes to base64 and injects them into the message

### Step 4 — Guardrails (`agents/guardrails.py`)

- Lightweight check run before the orchestrator processes each message
- Uses a small gpt-5.4-mini prompt to classify: `in_scope | out_of_scope | unsafe`
- `in_scope`: clothing/fashion product queries
- `out_of_scope`: off-topic (e.g., "write my CV") → polite refusal
- `unsafe`: harmful content → hard refusal
- Keeps the main agent focused without bloating its system prompt

### Step 5 — Orchestrator Agent (`agents/orchestrator.py`)

This is the core design piece. It is a stateful, multi-turn LangGraph agent backed by `gpt-5.4`.

**LangGraph StateGraph:**
```
[START] → guardrail_node → agent_node ⇄ tool_node → [END]
                ↓ (out-of-scope / unsafe)
            refusal_node → [END]
```

**Conversation State (`ConversationState`):**
- `messages: list` — full LangChain message history (managed by LangGraph, annotated with `operator.add`)
- `collected_preferences: dict` — structured store of user preferences (colour, type, occasion) updated each turn
- `pending_clarifications: list[str]` — questions already asked, to avoid repetition

**Nodes:**
| Node | Model | Role |
|---|---|---|
| `guardrail_node` | `gpt-5.4-mini` | Classifies `in_scope / out_of_scope / unsafe` before anything else |
| `agent_node` | `gpt-5.4` | Main reasoning node — decides to clarify or call tools |
| `tool_node` | — | LangGraph `ToolNode` — executes bound tools and returns results |
| `refusal_node` | — | Returns a static polite refusal, no LLM call needed |

**Tools bound to `agent_node`:**
| Tool | Description |
|---|---|
| `search_products(query, filters)` | Semantic + filtered product search |
| `get_article_details(article_id)` | Fetch full article info by ID |

**Decision Logic (expressed via system prompt + conditional edges):**
1. Ambiguous query → ask 1–2 targeted clarifying questions
2. Sufficient context → call `search_products`, present top results with IDs
3. Image attached → LLM sees it inline in the multimodal message, describes it, then calls `search_products`
4. No results → inform user, suggest broadening criteria
5. Low-confidence match → present results with caveat, ask for confirmation

**Why LangGraph:**
- Checkpointing to Postgres gives resumable sessions with zero extra code
- `ToolNode` handles the tool-call → result → next-step loop reliably
- Each conversation runs as an isolated thread — no manual session management
- Easy to add nodes (e.g., preference extractor, re-ranker) without restructuring

### Step 6 — Chat Interface (`app.py`)

- **Streamlit** web app with `st.chat_message` components
- Supports text input and file uploader for images (JPG/PNG)
- Session state holds `ConversationSession` object
- Renders product cards with `article_id`, name, colour, description
- Clear conversation button resets state

### Step 7 — Automated Evaluation (`eval/`)

**`test_cases.py`** — defines scenarios:
- Specific product query (e.g., "Do you have Howie shorts in blue?")
- Vague query requiring clarification (e.g., "I want some new clothes")
- Multi-turn outfit building (trousers + top + jacket)
- Image-based query
- Out-of-scope request
- No-results scenario

**`evaluator.py`** — runs each scenario programmatically:
- Checks: did the agent ask a clarifying question when expected?
- Checks: are recommended article IDs real (exist in catalog)?
- Checks: was the out-of-scope request refused?
- Outputs a simple pass/fail report per scenario

---

## Key Design Decisions

### When to Clarify vs. Recommend
- Clarify if: query is under-specified (no type, colour, or occasion given) **and** no prior context in conversation
- Recommend if: enough constraints exist to narrow to ≤ 10 plausible products
- Ask at most 2 clarifying questions per turn to avoid feeling interrogative

### Context Maintenance
- Full message history is passed to the model on each turn (within context limits)
- A `collected_preferences` dict is updated by the agent after each turn, giving a compact structured summary that survives long conversations

### Failure Modes
| Failure | Recovery |
|---|---|
| No search results | Inform user, suggest relaxing one constraint |
| Low-confidence match | Present with "these are the closest matches" language |
| Image not clothing | Ask user to attach a clearer image |
| Out-of-scope | Polite refusal, redirect to clothing help |
| Unsafe request | Hard refusal |

### Multimodal (Optional but included)
- Streamlit file uploader → image bytes → base64 inline in `HumanMessage` → LLM describes → `search_products`
- If no matching images directory entry, still works via text-only attributes

---

## Requirements (`requirements.txt`)

```
openai>=1.0.0                  # Azure OpenAI (embeddings + chat + vision)
langchain>=0.2.0               # Tool binding, message types
langchain-openai>=0.1.0        # AzureChatOpenAI integration
langgraph>=0.2.0               # StateGraph orchestration + Postgres checkpointing
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.26.0
psycopg2-binary>=2.9.0         # Postgres driver
pgvector>=0.2.0                # pgvector Python adapter
sqlalchemy>=2.0.0
alembic>=1.13.0
Pillow>=10.0.0                 # image handling
python-dotenv>=1.0.0
```

---

## Environment Variables (`.env`)

```
OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://dg-openai-api-sweden.openai.azure.com
AZURE_OPENAI_API_VERSION=2024-12-01-preview
EMBEDDING_DEPLOYMENT=azure-text-embedding-3-large
AZURE_CHAT_DEPLOYMENT=gpt-5.4
AZURE_CHAT_MINI_DEPLOYMENT=gpt-5.4-mini
POSTGRES_DSN=postgresql://postgres:postgres@localhost:5433/digitalgenius
```

---

## Phased Delivery Order

| Phase | Deliverable | Status | Why First |
|---|---|---|---|
| 1 | `catalog.py` + `db.py` + `embeddings.py` + `docker-compose.yml` | ✅ Done | Foundation everything else depends on |
| 2 | `product_retriever.py` | ✅ Done | Core retrieval capability |
| 3 | `guardrails.py` | ✅ Done | Keeps agent safe before exposing to conversation |
| 4 | `orchestrator.py` | ✅ Done | Main LangGraph agent, all tools wired |
| 5 | `app.py` (Streamlit UI) | ✅ Done | Makes it interactive |
| 6 | `eval/` | ✅ Done | Validates end-to-end quality + GEPA prompt optimisation |
