"""
SQLAlchemy engine, session factory, and query helpers.
Schema is managed via Alembic migrations — do not call Base.metadata.create_all() here.
"""

import os
from contextlib import contextmanager
from collections.abc import Generator

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv

from data.models import Article

load_dotenv()

POSTGRES_DSN = os.getenv(
    "POSTGRES_DSN",
    "postgresql://postgres:postgres@localhost:5433/digitalgenius",
)

_engine = create_engine(POSTGRES_DSN, pool_pre_ping=True)
_SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def is_populated() -> bool:
    with get_session() as session:
        count = session.scalar(
            select(func.count()).select_from(Article).where(Article.embedding.isnot(None))
        )
    return (count or 0) > 0


def upsert_articles_batch(rows: list[dict], embeddings: list[list[float]]):
    """Bulk upsert articles with their embeddings."""
    with get_session() as session:
        for row, embedding in zip(rows, embeddings):
            article = session.get(Article, str(row["article_id"]))
            if article is None:
                article = Article(article_id=str(row["article_id"]))
                session.add(article)
            article.prod_name = row.get("prod_name")
            article.product_type_name = row.get("product_type_name")
            article.product_group_name = row.get("product_group_name")
            article.graphical_appearance_name = row.get("graphical_appearance_name")
            article.colour_group_name = row.get("colour_group_name")
            article.garment_group_name = row.get("garment_group_name")
            article.detail_desc = row.get("detail_desc")
            article.image_desc = row.get("image_desc")
            article.embedding = embedding


def similarity_search(
    query_embedding: list[float],
    top_k: int = 5,
    filters: dict | None = None,
) -> list[dict]:
    """
    Returns top_k articles ordered by cosine similarity.
    Optional filters: colour_group_name, product_type_name, garment_group_name.
    """
    distance = Article.embedding.cosine_distance(query_embedding).label("distance")

    stmt = (
        select(Article, distance)
        .where(Article.embedding.isnot(None))
        .order_by(distance)
        .limit(top_k)
    )

    if filters:
        if filters.get("colour_group_name"):
            stmt = stmt.where(
                func.lower(Article.colour_group_name).like(f"%{filters['colour_group_name'].lower()}%")
            )
        if filters.get("product_type_name"):
            stmt = stmt.where(
                func.lower(Article.product_type_name).like(f"%{filters['product_type_name'].lower()}%")
            )
        if filters.get("garment_group_name"):
            stmt = stmt.where(
                func.lower(Article.garment_group_name).like(f"%{filters['garment_group_name'].lower()}%")
            )

    with get_session() as session:
        results = session.execute(stmt).all()

    return [
        {**article.to_dict(), "similarity": round(1 - float(distance), 4)}
        for article, distance in results
    ]
