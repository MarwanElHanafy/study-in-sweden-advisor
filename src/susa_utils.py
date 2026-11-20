"""Small helpers for reading the raw Susa-navet (EMIL 3) JSON pages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

EN_CODES = ("en", "eng")
SV_CODES = ("sv", "swe")


def load_resource(raw_dir: str | Path, name: str) -> list[dict]:
    """Return all items of one resource, e.g. 'educationEvents', across all saved pages."""
    items: list[dict] = []
    for page in sorted(Path(raw_dir, name).glob("page_*.json")):
        data = json.loads(page.read_text(encoding="utf-8"))
        items.extend(data.get(name) or [])
    return items


def content(item: dict) -> dict:
    """List responses wrap each record as {id, status, content}."""
    return item.get("content") or {}


def lang_text(langstring: dict | None, prefer: Iterable[str] = EN_CODES) -> str:
    """Pick a text from a LangString, preferring the given language codes."""
    nodes = (langstring or {}).get("strings") or []
    prefer = tuple(prefer)
    for code in prefer:
        for n in nodes:
            if (n.get("lang") or "").lower() == code and n.get("value"):
                return n["value"]
    for n in nodes:
        if n.get("value"):
            return n["value"]
    return ""


def lang_codes(langstring: dict | None) -> list[str]:
    return [(n.get("lang") or "?") for n in (langstring or {}).get("strings") or []]


def url_for(langurl: dict | None, prefer: Iterable[str] = EN_CODES) -> str:
    nodes = (langurl or {}).get("urls") or []
    for code in prefer:
        for n in nodes:
            if (n.get("lang") or "").lower() == code and n.get("value"):
                return n["value"]
    return next((n["value"] for n in nodes if n.get("value")), "")


def extension(record: dict, type_name: str) -> dict:
    """Return the first extension of a given type, e.g. 'UHEventExtension'."""
    for ext in record.get("extensions") or []:
        if ext.get("type") == type_name:
            return ext
    return {}
