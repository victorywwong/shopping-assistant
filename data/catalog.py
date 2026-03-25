"""
Loads articles.csv and exposes lightweight accessors.
The catalog is read-only; all mutation goes through db.py.
"""

import os
import pandas as pd

_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "articles.csv")

_df: pd.DataFrame | None = None


def _load() -> pd.DataFrame:
    global _df
    if _df is None:
        _df = pd.read_csv(_CSV_PATH, dtype={"article_id": str})
        _df["article_id"] = _df["article_id"].str.strip()
    return _df


def all_articles() -> pd.DataFrame:
    return _load()


def get_article(article_id: str) -> dict | None:
    df = _load()
    row = df[df["article_id"] == str(article_id)]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def article_to_embed_text(row: dict | pd.Series) -> str:
    """Builds the text string used for embedding a product.
    Includes image_desc if available (populated during seeding).
    """
    parts = [
        row.get("prod_name", ""),
        row.get("product_type_name", ""),
        row.get("colour_group_name", ""),
        row.get("graphical_appearance_name", ""),
        row.get("detail_desc", ""),
        row.get("image_desc", ""),
    ]
    return " ".join(str(p) for p in parts if p and str(p) != "nan")
