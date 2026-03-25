"""
Ingestion pipeline — orchestrates the full seeding flow:
  1. Load articles from catalog
  2. Describe product images in parallel via gpt-5.4 vision
  3. Generate embeddings via text-embedding-3-large
  4. Upsert everything into Postgres
"""

import base64
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator

from openai import AzureOpenAI
from dotenv import load_dotenv

from data.catalog import all_articles, article_to_embed_text
from data import db

load_dotenv()

logger = logging.getLogger(__name__)

CHAT_DEPLOYMENT = os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-5.4")
BATCH_SIZE = 100
IMAGE_WORKERS = 10

IMAGES_DIR = Path(__file__).parent.parent / "images"

IMAGE_PROMPT = (
    "Describe this clothing item concisely for a product search index. "
    "Include: garment type, colour, pattern or print, fabric texture if visible, "
    "fit or silhouette, and any notable design details. "
    "Two to four sentences maximum."
)

_client: AzureOpenAI | None = None


def _get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        _client = AzureOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        )
    return _client


def _describe_image(article_id: str) -> tuple[str, str | None]:
    """Returns (article_id, image_desc or None)."""
    image_path = IMAGES_DIR / f"{article_id}.jpg"
    if not image_path.exists():
        return article_id, None
    try:
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        response = _get_client().chat.completions.create(
            model=CHAT_DEPLOYMENT,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": IMAGE_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ],
            }],
            max_completion_tokens=200,
            temperature=0,
        )
        return article_id, response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Vision API failed for {article_id}: {e}")
        return article_id, None


def _describe_batch(rows: list[dict]) -> dict[str, str | None]:
    """Describe all images in a batch in parallel. Returns {article_id: image_desc}."""
    results: dict[str, str | None] = {}
    with ThreadPoolExecutor(max_workers=IMAGE_WORKERS) as pool:
        futures = {pool.submit(_describe_image, r["article_id"]): r["article_id"] for r in rows}
        for future in as_completed(futures):
            article_id, desc = future.result()
            results[article_id] = desc
    return results


def _batched(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def ingest(force: bool = False):
    """
    Seed all articles into pgvector.
    For each batch: describe images → generate embeddings → upsert.
    Skips if already populated unless force=True.
    """
    if not force and db.is_populated():
        logger.info("Already seeded — skipping. Use force=True to re-seed.")
        return

    from data.embeddings import embed_texts

    df = all_articles()
    rows = df.to_dict(orient="records")
    total = len(rows)
    logger.info(f"Seeding {total} articles (image description + embedding)...")

    for i, batch in enumerate(_batched(rows, BATCH_SIZE)):
        done = min((i + 1) * BATCH_SIZE, total)

        logger.info(f"  [{done}/{total}] Describing images...")
        image_descs = _describe_batch(batch)
        for row in batch:
            row["image_desc"] = image_descs.get(row["article_id"])

        logger.info(f"  [{done}/{total}] Embedding...")
        texts = [article_to_embed_text(row) for row in batch]
        embeddings = embed_texts(texts)

        db.upsert_articles_batch(batch, embeddings)
        logger.info(f"  [{done}/{total}] Upserted.")

    logger.info("Seeding complete.")
