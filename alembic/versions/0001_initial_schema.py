"""Initial schema: articles table with pgvector embedding

Revision ID: 0001
Revises:
Create Date: 2026-03-24
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "articles",
        sa.Column("article_id", sa.Text, primary_key=True),
        sa.Column("prod_name", sa.Text, nullable=True),
        sa.Column("product_type_name", sa.Text, nullable=True),
        sa.Column("product_group_name", sa.Text, nullable=True),
        sa.Column("graphical_appearance_name", sa.Text, nullable=True),
        sa.Column("colour_group_name", sa.Text, nullable=True),
        sa.Column("garment_group_name", sa.Text, nullable=True),
        sa.Column("detail_desc", sa.Text, nullable=True),
        # 1024-dim: text-embedding-3-large with dimensions=1024 parameter.
        # Well within pgvector's 2000-dim HNSW limit, no halfvec needed.
        sa.Column("embedding", Vector(1024), nullable=True),
    )

    op.create_index(
        "articles_embedding_hnsw_idx",
        "articles",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index("articles_embedding_hnsw_idx", table_name="articles")
    op.drop_table("articles")
