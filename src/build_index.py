"""Embed every programme in data/programs.json and save the vectors to data/index/.

Usage:
  python src/build_index.py
The first run downloads the embedding model (about 470 MB) from Hugging Face.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from retriever import EMBED_MODEL, INDEX_DIR, PROGRAMS_PATH, Embedder, passage_text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--programs", default=str(PROGRAMS_PATH))
    ap.add_argument("--out", default=str(INDEX_DIR))
    ap.add_argument("--model", default=EMBED_MODEL)
    args = ap.parse_args()

    raw = Path(args.programs).read_bytes()
    programs = json.loads(raw)
    texts = [passage_text(r) for r in programs]

    t0 = time.time()
    emb = Embedder(args.model).passages(texts)
    elapsed = time.time() - t0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", emb)
    meta = {
        "model": args.model,
        "n": len(programs),
        "dim": int(emb.shape[1]),
        "programs_sha1": hashlib.sha1(raw).hexdigest(),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedding_seconds": round(elapsed, 1),
        "avg_passage_chars": round(sum(map(len, texts)) / len(texts)),
    }
    (out / "index_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
