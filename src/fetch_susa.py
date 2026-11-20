"""Download all higher-education (HS) data from Skolverket's Susa-navet API.

Source: Skolverket, Susa-navet open API (EMIL 3), licence CC0 1.0.
Docs:   https://www.skolverket.se/om-skolverket/oppna-data/api-for-utbildningstillfallen-susa-navet

Three resources are downloaded and saved page by page under data/raw/:
  educationInfos      what a program is (title, description, level, credits, eligibility)
  educationEvents     a concrete offering (university, city, language, start, tuition, deadline)
  educationProviders  the universities themselves

Usage:
  python src/fetch_susa.py            # full download (a few minutes)
  python src/fetch_susa.py --max-pages 1   # quick test, first page of each resource only
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_URL = os.environ.get("SUSA_BASE_URL", "https://api.skolverket.se/susa-navet/emil3")
RESOURCES = ["educationInfos", "educationEvents", "educationProviders"]
PAGE_SIZE = 2000  # API maximum
HEADERS = {
    "Accept": "application/json",
    "User-Agent": "study-in-sweden-advisor (student project, Uppsala University)",
}
RETRY_STATUS = {429, 500, 502, 503, 504}


def get_with_retry(session: requests.Session, url: str, params: dict, attempts: int = 5) -> dict:
    for i in range(attempts):
        try:
            r = session.get(url, params=params, timeout=180)
        except requests.RequestException as e:
            err = str(e)
        else:
            if r.status_code == 200:
                return r.json()
            if r.status_code not in RETRY_STATUS:
                raise RuntimeError(f"HTTP {r.status_code} for {r.url}: {r.text[:300]}")
            err = f"HTTP {r.status_code}"
        wait = 2 ** (i + 1)
        print(f"  retry {i + 1}/{attempts} after {err}, waiting {wait}s")
        time.sleep(wait)
    raise RuntimeError(f"Giving up on {url} {params}")


def fetch_resource(session, name: str, out_dir: Path, school_type: str, max_pages: int | None, pause: float) -> int:
    target = out_dir / name
    target.mkdir(parents=True, exist_ok=True)
    for old in target.glob("page_*.json"):
        old.unlink()

    page, total_pages, n_items = 0, None, 0
    while total_pages is None or page < total_pages:
        params = {"schoolType": school_type, "page": page, "size": PAGE_SIZE}
        data = get_with_retry(session, f"{BASE_URL}/{name}", params)
        (target / f"page_{page:03d}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        meta = data.get("page") or {}
        total_pages = int(meta.get("totalPages") or 1)
        if max_pages is not None:
            total_pages = min(total_pages, max_pages)
        n_items += len(data.get(name) or [])
        print(f"{name}: page {page + 1}/{total_pages}, {n_items} of {meta.get('totalElements', '?')} items")
        page += 1
        time.sleep(pause)
    return n_items


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/raw", help="output folder (default: data/raw)")
    ap.add_argument("--school-type", default="HS", help="HS = Högskola (universities)")
    ap.add_argument("--max-pages", type=int, default=None, help="limit pages per resource, for testing")
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between requests (be polite)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update(HEADERS)

    counts = {}
    for name in RESOURCES:
        counts[name] = fetch_resource(session, name, out_dir, args.school_type, args.max_pages, args.pause)

    manifest = {
        "source": "Skolverket Susa-navet open API (EMIL 3)",
        "base_url": BASE_URL,
        "licence": "CC0 1.0",
        "school_type": args.school_type,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "counts": counts,
        "max_pages": args.max_pages,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nDone:", json.dumps(counts))


if __name__ == "__main__":
    main()
