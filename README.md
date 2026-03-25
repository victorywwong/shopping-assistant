# Fashion Assistant

A multi-turn, multimodal conversational agent that recommends women's clothing products via a Streamlit chat interface. Built with LangGraph, Azure OpenAI, and pgvector.

![Demo](demo.png)

## Architecture

```
Streamlit UI (app.py)
       │
       ▼
LangGraph Orchestrator (gpt-5.4)
  ├── Guardrail node       — classifies in_scope / out_of_scope / unsafe (gpt-5.4-mini)
  ├── Agent node           — reasons and calls tools
  ├── Tool node
  │     ├── search_products      — pgvector semantic search + filters
  │     └── get_article_details  — catalog lookup by article_id
  └── Refusal node         — static response for blocked requests
       │
       ▼
Postgres + pgvector
  └── articles table       — product metadata + image_desc + vector(1024) embeddings
```

Images are passed as multimodal messages directly to gpt-5.4 — no separate vision tool needed.

---

## Prerequisites

- Docker
- Python 3.11+
- An activated Python virtual environment

---

## Setup

### 1. Start the database

```bash
docker compose up -d
```

### 2. Create a `.env` file

```
OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://...
AZURE_OPENAI_API_VERSION=2024-12-01-preview
EMBEDDING_DEPLOYMENT=azure-text-embedding-3-large
AZURE_CHAT_DEPLOYMENT=gpt-5.4
AZURE_CHAT_MINI_DEPLOYMENT=gpt-5.4-mini
POSTGRES_DSN=postgresql://postgres:postgres@localhost:5433/digitalgenius
WANDB_API_KEY=...                        # optional — enables GEPA run tracking
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run database migrations

```bash
alembic upgrade head
```

### 5. Unzip product images

```bash
unzip images.zip -d images/
```

### 6. Seed the product catalog

```bash
python seed.py
```

For each article: describes the product image in parallel via `gpt-5.4` vision, then generates embeddings via `text-embedding-3-large` (dimensions=1024) and upserts everything into pgvector. Skips automatically if already seeded.

![pgvector](pgvector.png)

### 7. Run the app

```bash
streamlit run app.py
```

Available at `http://localhost:8501`.

---

## Evaluation & Prompt Optimisation

### Run evaluation

Scores the live agent against conversation scenarios and guardrail test cases using an LLM-as-judge.

```bash
python -m eval.run_eval
python -m eval.run_eval --guardrail-only
python -m eval.run_eval --orchestrator-only
```

Reports are saved to `eval/results/`.

### Run prompt optimisation

Uses GEPA (DSPy) to optimise the system prompts for the guardrail and/or orchestrator. Optimised prompts are saved to `eval/optimised/`.

```bash
python -m eval.run_optimise
python -m eval.run_optimise --guardrail-only
python -m eval.run_optimise --orchestrator-only
```

---

## Project Structure

```
├── app.py                      # Streamlit chat UI
├── seed.py                     # One-time DB seeding script
├── docker-compose.yml          # Postgres + pgvector
├── alembic/                    # DB migrations
├── agents/
│   ├── orchestrator.py         # LangGraph StateGraph
│   ├── product_retriever.py    # search_products + get_article_details tools
│   └── guardrails.py           # Message classifier (in_scope / out_of_scope / unsafe)
├── data/
│   ├── models.py               # SQLAlchemy ORM model
│   ├── db.py                   # Engine, session, similarity search
│   ├── catalog.py              # articles.csv loader
│   ├── embeddings.py           # embed_texts + similarity_search
│   └── ingest.py               # Full seeding pipeline (vision → embed → upsert)
└── eval/
    ├── test_cases.py           # Scenario + guardrail test case definitions
    ├── evaluator.py            # LLM-as-judge scoring, report persistence
    ├── dspy_modules.py         # DSPy signatures and modules
    ├── optimizer.py            # GEPA optimisation for guardrail + orchestrator
    ├── run_eval.py             # CLI: evaluation only
    └── run_optimise.py         # CLI: prompt optimisation only
```
