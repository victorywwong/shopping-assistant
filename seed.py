"""
Seed the database with articles and their embeddings.
Run once after `alembic upgrade head`.

Usage:
    python seed.py
"""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)

from data.ingest import ingest

if __name__ == "__main__":
    ingest()
