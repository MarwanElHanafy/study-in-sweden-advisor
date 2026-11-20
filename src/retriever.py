"""Semantic search over the programme dataset.

Each offering in data/programs.json is turned into one text "passage" and embedded with a
local multilingual model (default intfloat/multilingual-e5-small). Multilingual matters
because 28% of programmes only have a Swedish description.

Search = hard metadata filters (city, tuition, credits, university, deadline) followed by
cosine similarity ranking. With ~1,000 documents exact search in numpy takes milliseconds,
so no vector database is needed.
"""

from __future__ import annotations

import json
import os
import unicodedata
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

EMBED_MODEL = os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-small")
PROGRAMS_PATH = Path("data/programs.json")
INDEX_DIR = Path("data/index")


def fold(text: str) -> str:
    """Lowercase and strip accents, so 'Göteborg' matches 'goteborg'."""
    nfkd = unicodedata.normalize("NFKD", text or "")
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def passage_text(r: dict) -> str:
    """The text that represents one programme in the vector index."""
    parts = [
        f"{r['title']}.",
        f"{r['university']}, {r['city']}.",
        f"Keywords: {', '.join(r.get('keywords') or r.get('keywords_sv') or [])}." if (r.get("keywords") or r.get("keywords_sv")) else "",
        r.get("search_description") or "",
        f"Eligibility: {r['eligibility']}" if r.get("eligibility") else "",
    ]
    return " ".join(p for p in parts if p)


class Embedder:
    """Thin wrapper around sentence-transformers with the e5 query/passage prefixes."""

    def __init__(self, model_name: str = EMBED_MODEL):
        from sentence_transformers import SentenceTransformer  # imported lazily (slow import)

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.e5 = "e5" in model_name.lower()

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        if self.e5:
            texts = [f"{prefix}: {t}" for t in texts]
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32,
                                 show_progress_bar=len(texts) > 50, convert_to_numpy=True).astype(np.float32)

    def passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, "passage")

    def query(self, text: str) -> np.ndarray:
        return self._encode([text], "query")[0]


@dataclass
class Hit:
    score: float
    record: dict


class ProgramIndex:
    def __init__(self, programs_path: Path = PROGRAMS_PATH, index_dir: Path = INDEX_DIR, embedder: Embedder | None = None):
        self.programs: list[dict] = json.loads(Path(programs_path).read_text(encoding="utf-8"))
        self.embeddings = np.load(Path(index_dir) / "embeddings.npy")
        meta = json.loads((Path(index_dir) / "index_meta.json").read_text(encoding="utf-8"))
        if meta["n"] != len(self.programs):
            raise RuntimeError("Index is out of date: run  python src/build_index.py")
        self.embedder = embedder or Embedder(meta["model"])
        self.by_id = {r["id"]: r for r in self.programs}
        self.cities = sorted({r["city"] for r in self.programs})

    def search(self, query: str, city: str | None = None, max_tuition_sek: float | None = None,
               credits: float | None = None, university: str | None = None,
               include_passed_deadlines: bool = False, top_k: int = 5) -> list[Hit]:
        today = date.today().isoformat()
        mask = np.ones(len(self.programs), dtype=bool)
        for i, r in enumerate(self.programs):
            if city and fold(r["city"]) != fold(city):
                mask[i] = False
            elif max_tuition_sek is not None and (r.get("tuition_total_sek") or 0) > max_tuition_sek:
                mask[i] = False
            elif credits is not None and r.get("credits") != float(credits):
                mask[i] = False
            elif university and fold(university) not in fold(r["university"]):
                mask[i] = False
            elif not include_passed_deadlines and r.get("application_deadline") and r["application_deadline"] < today:
                mask[i] = False
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            return []
        scores = self.embeddings[idx] @ self.embedder.query(query)
        hits, seen = [], set()
        for j in np.argsort(-scores):
            r = self.programs[idx[j]]
            key = (r["title"], r["university"], r["city"])  # same programme, several intakes/campuses
            if key in seen:
                continue
            seen.add(key)
            hits.append(Hit(float(scores[j]), r))
            if len(hits) >= top_k:
                break
        return hits
