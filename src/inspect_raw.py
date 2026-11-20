"""Summarise the downloaded Susa-navet data so we can choose the right filters.

Prints value distributions for the fields we care about (language, city, level,
program vs course, tuition, deadlines) and saves one example of each record type
to data/samples/ for a closer look.

Usage:
  python src/inspect_raw.py > data/inspect_report.txt
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from susa_utils import content, extension, lang_codes, lang_text, load_resource


def show(title: str, counter: Counter, top: int = 25) -> None:
    total = sum(counter.values())
    print(f"\n## {title}  (total {total}, distinct {len(counter)})")
    for value, n in counter.most_common(top):
        print(f"  {n:7d}  {value}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--samples", default="data/samples")
    args = ap.parse_args()

    infos = load_resource(args.raw, "educationInfos")
    events = load_resource(args.raw, "educationEvents")
    providers = load_resource(args.raw, "educationProviders")
    print(f"# Counts: infos={len(infos)} events={len(events)} providers={len(providers)}")

    # ---------- educationInfos ----------
    info_by_id = {}
    for it in infos:
        c = content(it)
        info_by_id[c.get("identifier") or it.get("id")] = it
    show("info.status", Counter(it.get("status") for it in infos))
    show("info.configuration (kurs / program)", Counter((content(it).get("configuration") or {}).get("code") for it in infos))
    show("info.educationLevels (type:code)", Counter(
        ",".join(f"{l.get('type')}:{l.get('code')}" for l in content(it).get("educationLevels") or []) or None
        for it in infos))
    show("info.qualificationLevel", Counter((content(it).get("qualificationLevel") or {}).get("code") for it in infos))
    show("info.credits (system, credits)", Counter(
        ((content(it).get("credits") or {}).get("system", {}) or {}).get("code") if isinstance((content(it).get("credits") or {}).get("system"), dict)
        else (content(it).get("credits") or {}).get("system") for it in infos), top=10)
    show("info.credits value (programs only)", Counter(
        (content(it).get("credits") or {}).get("credits") for it in infos
        if (content(it).get("configuration") or {}).get("code") == "program"), top=15)
    show("info.title languages", Counter(",".join(lang_codes(content(it).get("title"))) for it in infos), top=10)
    show("info.description languages", Counter(",".join(lang_codes(content(it).get("description"))) for it in infos), top=10)
    show("info.subject code types", Counter(s.get("type") for it in infos for s in content(it).get("subjects") or []), top=10)
    show("info.resultIsDegree", Counter(content(it).get("resultIsDegree") for it in infos))

    # ---------- educationEvents ----------
    show("event.status", Counter(ev.get("status") for ev in events))
    show("event.languageOfInstructions", Counter(",".join(content(ev).get("languageOfInstructions") or []) or None for ev in events))
    show("event.locations[].town", Counter((loc.get("town") or None) for ev in events for loc in content(ev).get("locations") or []), top=40)
    show("event has no location", Counter(not content(ev).get("locations") for ev in events))
    show("event.isCancelled", Counter(content(ev).get("isCancelled") for ev in events))
    show("event.paceOfStudy %", Counter((content(ev).get("paceOfStudy") or {}).get("percentage") for ev in events), top=10)
    show("event.timeOfStudy", Counter((content(ev).get("timeOfStudy") or {}).get("code") for ev in events), top=10)
    show("event extension types", Counter(x.get("type") for ev in events for x in content(ev).get("extensions") or []))

    uh = [extension(content(ev), "UHEventExtension") for ev in events]
    show("UH visibleToInternationalApplicants", Counter((u.get("applicationDetails") or {}).get("visibleToInternationalApplicants") for u in uh))
    show("UH tuitionFee.value", Counter((u.get("tuitionFee") or {}).get("value") for u in uh))
    show("UH tuitionFee.total (sample)", Counter((u.get("tuitionFee") or {}).get("total") for u in uh if (u.get("tuitionFee") or {}).get("total")), top=10)
    show("UH startPeriod (year, semester)", Counter(
        (((u.get("startPeriod") or {}).get("period") or {}).get("year"), ((u.get("startPeriod") or {}).get("period") or {}).get("semester"))
        for u in uh), top=15)
    show("UH closeDate (month)", Counter(((u.get("applicationDetails") or {}).get("closeDate") or "")[:7] or None for u in uh), top=15)
    show("UH itdistance", Counter(u.get("itdistance") for u in uh))
    show("event.fees currency", Counter(f.get("currency") for ev in events for f in content(ev).get("fees") or []), top=5)

    # ---------- joins ----------
    linked = sum(1 for ev in events if content(ev).get("education") in info_by_id)
    print(f"\n# events whose 'education' matches an info identifier: {linked}/{len(events)}")
    prov_ids = {content(p).get("identifier") or p.get("id") for p in providers}
    linked_p = sum(1 for ev in events if set(content(ev).get("providers") or []) & prov_ids)
    print(f"# events whose provider matches a provider identifier: {linked_p}/{len(events)}")
    show("providers (name)", Counter(lang_text(content(p).get("name")) for p in providers), top=60)

    # ---------- English + program candidates ----------
    def is_program(ev):
        info = info_by_id.get(content(ev).get("education"))
        return info is not None and (content(info).get("configuration") or {}).get("code") == "program"

    english = [ev for ev in events if any(l.lower() in ("eng", "en") for l in content(ev).get("languageOfInstructions") or [])]
    eng_prog = [ev for ev in english if is_program(ev)]
    print(f"\n# English-taught events: {len(english)}; of which programs: {len(eng_prog)}")
    show("English programs: education levels", Counter(
        ",".join(f"{l.get('type')}:{l.get('code')}" for l in content(info_by_id[content(ev).get("education")]).get("educationLevels") or [])
        for ev in eng_prog))
    show("English programs: town", Counter(
        (content(ev).get("locations") or [{}])[0].get("town") for ev in eng_prog), top=30)

    # ---------- samples ----------
    out = Path(args.samples)
    out.mkdir(parents=True, exist_ok=True)
    sample_ev = (eng_prog or english or events or [None])[0]
    if sample_ev:
        (out / "sample_event.json").write_text(json.dumps(sample_ev, ensure_ascii=False, indent=2), encoding="utf-8")
        info = info_by_id.get(content(sample_ev).get("education"))
        if info:
            (out / "sample_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    if providers:
        (out / "sample_provider.json").write_text(json.dumps(providers[0], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSamples written to {out}/")


if __name__ == "__main__":
    main()
