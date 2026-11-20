"""Build a clean dataset of English-taught master's programmes from the raw Susa-navet data.

Filters (decided from data/inspect_report.txt):
  - info.configuration == "program"              (not single courses)
  - info.educationLevels contains "avancerad"     (master's level; "grundavancerad" kept too)
  - event.languageOfInstructions contains "eng"
  - UHEventExtension.applicationDetails.visibleToInternationalApplicants == True
  - event.isCancelled is not True
  - event.execution.start >= --from-date          (default: today, i.e. upcoming intakes only)

City: the "town" field is almost always empty, so the city comes from the Swedish
municipality code (locations[].areaCode) with the campus name as fallback.

Output:
  data/programs.json   one record per offering
  printed report       coverage stats, run with  > data/build_report.txt
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path

from susa_utils import EN_CODES, SV_CODES, content, extension, lang_text, load_resource, url_for

# Municipality code (kommunkod) -> city, for municipalities with universities or campuses.
# Codes not listed here fall back to the campus name and are reported for review.
MUNICIPALITY_CITY = {
    "0180": "Stockholm", "0184": "Stockholm", "0126": "Stockholm", "0183": "Stockholm",
    "0181": "Södertälje", "0380": "Uppsala", "0980": "Visby", "0484": "Eskilstuna",
    "0580": "Linköping", "0581": "Norrköping", "0680": "Jönköping", "0780": "Växjö",
    "0880": "Kalmar", "1080": "Karlskrona", "1082": "Karlshamn", "1280": "Malmö",
    "1281": "Lund", "1283": "Helsingborg", "1262": "Alnarp", "1290": "Kristianstad",
    "1380": "Halmstad", "1480": "Gothenburg", "1488": "Trollhättan", "1490": "Borås",
    "1495": "Skara", "1496": "Skövde", "1780": "Karlstad", "1784": "Arvika", "1880": "Örebro",
    "1980": "Västerås", "2080": "Falun", "2081": "Borlänge", "2180": "Gävle",
    "2280": "Härnösand", "2281": "Sundsvall", "2380": "Östersund", "2480": "Umeå",
    "2482": "Skellefteå", "2580": "Luleå", "2581": "Piteå", "2584": "Kiruna",
}
ONLINE_MARKERS = ("distans", "distance", "location independent", "ortsoberoende",
                  "no physical attendance", "online")
ABROAD = "Abroad (joint programme)"
SEMESTER = {"HT": "Autumn", "VT": "Spring"}
LEVELS_KEPT = {"avancerad", "grundavancerad"}


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def city_for(event: dict) -> tuple[str, str, str]:
    """Return (city, municipality_code, campus_name) for an event."""
    ev = content(event)
    uh = extension(ev, "UHEventExtension")
    locs = ev.get("locations") or []
    loc = locs[0] if locs else {}
    code = (loc.get("areaCode") or "").strip()
    campus = clean(lang_text(loc.get("studyLocation"), EN_CODES))
    if uh.get("itdistance") or any(m in campus.lower() for m in ONLINE_MARKERS):
        return "Online (distance)", code, campus
    if loc.get("town"):
        return loc["town"], code, campus
    if code in MUNICIPALITY_CITY:
        return MUNICIPALITY_CITY[code], code, campus
    # No municipality code: joint programmes with partner universities abroad.
    # Use the Swedish city if one is named ("Tartu and Uppsala"), otherwise mark as abroad.
    for city in MUNICIPALITY_CITY.values():
        if city.lower() in campus.lower():
            return city, code, campus
    return (ABROAD if campus else "Unknown"), code, campus


def keywords(ev: dict, codes: tuple[str, ...]) -> list[str]:
    out = []
    for ls in ev.get("keywords") or []:
        for n in ls.get("strings") or []:
            if (n.get("lang") or "").lower() in codes and n.get("value"):
                out.append(clean(n["value"]))
    return sorted(set(out), key=str.lower)


def only_lang(langstring: dict | None, codes: tuple[str, ...]) -> str:
    for n in (langstring or {}).get("strings") or []:
        if (n.get("lang") or "").lower() in codes and n.get("value"):
            return clean(n["value"])
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/programs.json")
    ap.add_argument("--from-date", default=date.today().isoformat(), help="keep offerings starting on or after this date")
    args = ap.parse_args()

    infos = {content(i).get("identifier") or i.get("id"): content(i) for i in load_resource(args.raw, "educationInfos")}
    providers = {content(p).get("identifier") or p.get("id"): content(p) for p in load_resource(args.raw, "educationProviders")}
    events = load_resource(args.raw, "educationEvents")
    manifest_path = Path(args.raw, "manifest.json")
    fetched_at = json.loads(manifest_path.read_text())["fetched_at"] if manifest_path.exists() else ""

    funnel = Counter()
    records, unmapped = [], Counter()
    for event in events:
        ev = content(event)
        info = infos.get(ev.get("education")) or {}
        uh = extension(ev, "UHEventExtension")
        funnel["all events"] += 1
        if (info.get("configuration") or {}).get("code") != "program":
            continue
        funnel["programs"] += 1
        levels = {l.get("code") for l in info.get("educationLevels") or []}
        if not levels & LEVELS_KEPT:
            continue
        funnel["+ master's level"] += 1
        if "eng" not in [l.lower() for l in ev.get("languageOfInstructions") or []]:
            continue
        funnel["+ taught in English"] += 1
        if not (uh.get("applicationDetails") or {}).get("visibleToInternationalApplicants"):
            continue
        funnel["+ open to international applicants"] += 1
        if ev.get("isCancelled") is True:
            continue
        funnel["+ not cancelled"] += 1
        start = (ev.get("execution") or {}).get("start") or ""
        if not start or start < args.from_date:
            continue
        funnel[f"+ starts on/after {args.from_date}"] += 1

        city, muni, campus = city_for(event)
        if city not in MUNICIPALITY_CITY.values() and not city.startswith("Online"):
            unmapped[(muni, campus)] += 1
        desc_en = only_lang(info.get("description"), EN_CODES)
        desc_sv = only_lang(info.get("description"), SV_CODES)
        period = (uh.get("startPeriod") or {}).get("period") or {}
        fee = uh.get("tuitionFee") or {}
        credits = (info.get("credits") or {}).get("credits")
        provider_id = (ev.get("providers") or [""])[0]
        deadline = (ev.get("application") or {}).get("last") or ""

        records.append({
            "id": ev.get("identifier"),
            "application_code": (ev.get("application") or {}).get("code"),
            "title": clean(lang_text(info.get("title"), EN_CODES)),
            "title_sv": only_lang(info.get("title"), SV_CODES),
            "university": clean(lang_text(providers.get(provider_id, {}).get("name"), EN_CODES)) or provider_id,
            "provider_id": provider_id,
            "city": city,
            "campus": campus,
            "municipality_code": muni,
            "level": "master" if "avancerad" in levels else "bachelor/master",
            "credits": float(credits) if credits is not None else None,
            "duration_years": round(float(credits) / 60, 1) if credits else None,
            "degree": clean(lang_text((info.get("degrees") or [{}])[0], EN_CODES)) if info.get("degrees") else "",
            "start_date": start,
            "start_term": f"{SEMESTER.get(period.get('semester'), period.get('semester') or '')} {period.get('year') or ''}".strip(),
            "application_deadline": deadline,
            "deadline_passed": bool(deadline) and deadline < date.today().isoformat(),
            "tuition_fee_applies": fee.get("value"),
            "tuition_total_sek": fee.get("total"),
            "tuition_first_installment_sek": fee.get("first"),
            "pace_percent": (ev.get("paceOfStudy") or {}).get("percentage"),
            "time_of_study": (ev.get("timeOfStudy") or {}).get("code"),
            "description": desc_en,
            "description_sv": desc_sv,
            # Text used for search: English if available, else Swedish (handled by a multilingual embedding model)
            "search_description": desc_en or desc_sv,
            "search_description_lang": "en" if desc_en else ("sv" if desc_sv else ""),
            "joint_programme_abroad": not muni and bool(campus) and not any(m in campus.lower() for m in ONLINE_MARKERS),
            "eligibility": only_lang(info.get("eligibility"), EN_CODES),
            "eligibility_sv": only_lang(info.get("eligibility"), SV_CODES),
            "keywords": keywords(ev, EN_CODES),
            "keywords_sv": keywords(ev, SV_CODES),
            "subject_codes": [s.get("code") for s in info.get("subjects") or []],
            "url": url_for(ev.get("url"), EN_CODES) or url_for(info.get("url"), EN_CODES),
            "last_edited": ev.get("lastEdited"),
            "source": "Skolverket Susa-navet (CC0)",
            "fetched_at": fetched_at,
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------- report ----------
    print("# Filter funnel")
    for step, n in funnel.items():
        print(f"  {n:7d}  {step}")
    n = len(records) or 1
    print(f"\n# {len(records)} offerings written to {args.out}")
    print(f"  distinct programmes (title+university): {len({(r['title'], r['university']) for r in records})}")
    for field in ["description", "description_sv", "search_description", "eligibility", "keywords", "url", "application_deadline", "tuition_total_sek"]:
        have = sum(1 for r in records if r[field])
        print(f"  has {field:22s} {have:6d}  ({100 * have / n:.0f}%)")
    for title, key in [("city", "city"), ("university", "university"), ("start_term", "start_term"),
                       ("application_deadline", "application_deadline"), ("pace_percent", "pace_percent"),
                       ("credits", "credits")]:
        c = Counter(r[key] for r in records)
        print(f"\n## {title} (distinct {len(c)})")
        for v, k in c.most_common(45):
            print(f"  {k:6d}  {v}")
    print("\n## municipality codes NOT in the city map (code, campus) -> check these")
    for (code, campus), k in unmapped.most_common(40):
        print(f"  {k:6d}  {code!r:8s} {campus}")
    print("\n## mapped municipality codes actually used (code -> city, example campus)")
    seen = {}
    for r in records:
        seen.setdefault(r["municipality_code"], (r["city"], r["campus"]))
    for code, (city, campus) in sorted(seen.items()):
        print(f"  {code!r:8s} {city:20s} {campus}")


if __name__ == "__main__":
    main()
