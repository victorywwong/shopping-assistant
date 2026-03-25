"""
SQLAlchemy ORM model for the articles table.
"""

from pgvector.sqlalchemy import Vector
from sqlalchemy import Index, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Article(Base):
    __tablename__ = "articles"

    article_id: Mapped[str] = mapped_column(Text, primary_key=True)
    prod_name: Mapped[str | None] = mapped_column(Text)
    product_type_name: Mapped[str | None] = mapped_column(Text)
    product_group_name: Mapped[str | None] = mapped_column(Text)
    graphical_appearance_name: Mapped[str | None] = mapped_column(Text)
    colour_group_name: Mapped[str | None] = mapped_column(Text)
    garment_group_name: Mapped[str | None] = mapped_column(Text)
    detail_desc: Mapped[str | None] = mapped_column(Text)
    image_desc: Mapped[str | None] = mapped_column(Text, nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1024), nullable=True)

    __table_args__ = (
        Index(
            "articles_embedding_hnsw_idx",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    def to_dict(self) -> dict:
        return {
            "article_id": self.article_id,
            "prod_name": self.prod_name,
            "product_type_name": self.product_type_name,
            "product_group_name": self.product_group_name,
            "graphical_appearance_name": self.graphical_appearance_name,
            "colour_group_name": self.colour_group_name,
            "garment_group_name": self.garment_group_name,
            "detail_desc": self.detail_desc,
            "image_desc": self.image_desc,
        }
