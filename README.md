# Fashion Assistant

A multi-turn, multimodal conversational agent that recommends women's clothing products via a Streamlit chat interface. Built with LangGraph, Azure OpenAI, and pgvector.

## Prerequisites

- Docker
- Python 3.11+
- An activated Python virtual environment

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
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run database migrations

```bash
alembic upgrade head
```

### 5. Seed the product catalog

```bash
python seed.py
```

For each article: describes the product image via `gpt-5.4` vision (in parallel), then computes embeddings via `text-embedding-3-large` and stores everything in pgvector. Skips automatically if already seeded.

### 6. Run the app

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

## Project Structure

```
├── app.py                  # Streamlit chat UI
├── seed.py                 # One-time DB seeding script
├── docker-compose.yml      # Postgres + pgvector
├── agents/
│   ├── orchestrator.py     # LangGraph StateGraph
│   ├── product_retriever.py
│   └── guardrails.py
├── data/
│   ├── catalog.py
│   ├── db.py
│   ├── embeddings.py
│   └── models.py
└── eval/
    ├── test_cases.py
    └── evaluator.py
```
