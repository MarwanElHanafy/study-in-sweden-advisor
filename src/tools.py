"""The tools the LLM agent can call, plus their JSON-schema declarations for Gemini.

Every factual claim the agent makes about programmes, fees, deadlines or cities must come
from one of these tools. Results are kept compact to save tokens on the free API tier.
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from retriever import ProgramIndex, fold

CITIES_PATH = Path("knowledge/cities.json")
SNIPPET_CHARS = 280
MAX_TEXT_CHARS = 1500


def _short(text: str, n: int) -> str:
    text = text or ""
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " ..."


class Toolbox:
    def __init__(self, index: ProgramIndex, cities_path: Path = CITIES_PATH):
        self.index = index
        kb = json.loads(Path(cities_path).read_text(encoding="utf-8"))
        self.national = kb["national"]
        self.profiles = {c["city"]: c for c in kb["cities"]}
        self.alias = {}
        for c in kb["cities"]:
            for name in [c["city"], *c.get("aliases", [])]:
                self.alias[fold(name)] = c["city"]
        for city in index.cities:
            self.alias.setdefault(fold(city), city)
        self.stats = defaultdict(Counter)  # city -> university -> number of programmes
        for r in index.programs:
            self.stats[r["city"]][r["university"]] += 1

    # ---------- helpers ----------
    def resolve_city(self, name: str | None) -> str | None:
        if not name:
            return None
        key = fold(name)
        if key in self.alias:
            return self.alias[key]
        for alias, city in self.alias.items():  # partial match, e.g. "online"
            if key in alias or alias in key:
                return city
        return name

    @staticmethod
    def _summary(r: dict, score: float | None = None) -> dict:
        out = {
            "program_id": r["id"],
            "title": r["title"],
            "university": r["university"],
            "city": r["city"],
            "credits": r["credits"],
            "start": r["start_term"],
            "application_deadline": r["application_deadline"] or "not stated",
            "tuition_total_sek": r["tuition_total_sek"],
            "url": r["url"],
            "summary": _short(r.get("search_description") or "", SNIPPET_CHARS),
        }
        if r.get("search_description_lang") == "sv":
            out["summary_language"] = "Swedish (no English description in source)"
        if score is not None:
            out["relevance"] = round(score, 3)
        return out

    # ---------- tools ----------
    def search_programs(self, query: str, city: str | None = None, max_tuition_sek: float | None = None,
                        credits: float | None = None, university: str | None = None, top_k: int = 5) -> dict:
        city_resolved = self.resolve_city(city)
        hits = self.index.search(query, city=city_resolved, max_tuition_sek=max_tuition_sek, credits=credits,
                                 university=university, top_k=max(1, min(int(top_k or 5), 10)))
        result = {"query": query, "filters": {k: v for k, v in {"city": city_resolved, "max_tuition_sek": max_tuition_sek,
                  "credits": credits, "university": university}.items() if v is not None},
                  "results": [self._summary(h.record, h.score) for h in hits]}
        if not hits:
            result["note"] = "No programmes match these filters. Try removing a filter (e.g. city or budget)."
        return result

    def get_program_details(self, program_id: str) -> dict:
        r = self.index.by_id.get(program_id)
        if not r:
            return {"error": f"Unknown program_id '{program_id}'. Use an id returned by search_programs."}
        out = self._summary(r)
        out.pop("summary", None)
        out.update({
            "title_swedish": r.get("title_sv"),
            "campus": r.get("campus"),
            "degree": r.get("degree"),
            "duration_years": r.get("duration_years"),
            "pace_percent": r.get("pace_percent"),
            "start_date": r.get("start_date"),
            "tuition_first_installment_sek": r.get("tuition_first_installment_sek"),
            "application_code_universityadmissions": r.get("application_code"),
            "description": _short(r.get("description") or "", MAX_TEXT_CHARS),
            "description_swedish": "" if r.get("description") else _short(r.get("description_sv") or "", MAX_TEXT_CHARS),
            "eligibility": _short(r.get("eligibility") or r.get("eligibility_sv") or "", MAX_TEXT_CHARS),
            "keywords": r.get("keywords") or r.get("keywords_sv"),
            "joint_programme_abroad": r.get("joint_programme_abroad"),
            "data_source": f"{r['source']}, fetched {r.get('fetched_at', '')[:10]}",
        })
        return out

    def get_city_info(self, city: str) -> dict:
        name = self.resolve_city(city)
        unis = self.stats.get(name)
        out = {"city": name}
        if name in self.profiles:
            out.update({k: v for k, v in self.profiles[name].items() if k not in ("city", "aliases")})
        else:
            out["note"] = "No curated profile for this city; only programme statistics are available."
        if unis:
            out["programmes_in_dataset"] = sum(unis.values())
            out["universities"] = [{"name": u, "programmes": n} for u, n in unis.most_common()]
        elif name not in self.profiles:
            out = {"error": f"No data for '{city}'. Call list_cities to see available cities."}
        return out

    def compare_cities(self, cities: list[str]) -> dict:
        return {"cities": [self.get_city_info(c) for c in (cities or [])[:4]]}

    def list_cities(self) -> dict:
        rows = sorted(((c, sum(u.values())) for c, u in self.stats.items()), key=lambda x: -x[1])
        return {"cities": [{"city": c, "programmes": n, "has_profile": c in self.profiles} for c, n in rows]}

    def get_general_info(self) -> dict:
        return {
            "monthly_maintenance_requirement_sek": self.national["maintenance_requirement_sek_month"],
            "maintenance_requirement_note": self.national["maintenance_requirement_note"],
            "maintenance_requirement_source": self.national["maintenance_requirement_source"],
            "average_student_budget_sek_month": self.national["average_budget_sek_month"],
            "average_budget_source": self.national["average_budget_source"],
            "tuition_fees": "Tuition fees apply to citizens of countries outside the EU/EEA and Switzerland. EU/EEA/Swiss citizens pay no tuition. Fees in the dataset are the total for the whole programme.",
            "how_to_apply": "Most master's programmes are applied to through universityadmissions.se. The main round for autumn intakes closes in mid-January; check each programme's deadline.",
        }

    # ---------- dispatch ----------
    def call(self, name: str, args: dict) -> tuple[dict, float]:
        fn = {
            "search_programs": self.search_programs,
            "get_program_details": self.get_program_details,
            "get_city_info": self.get_city_info,
            "compare_cities": self.compare_cities,
            "list_cities": self.list_cities,
            "get_general_info": self.get_general_info,
        }.get(name)
        t0 = time.perf_counter()
        if fn is None:
            return {"error": f"Unknown tool {name}"}, 0.0
        try:
            result = fn(**(args or {}))
        except TypeError as e:
            result = {"error": f"Bad arguments for {name}: {e}"}
        return result, (time.perf_counter() - t0) * 1000


TOOL_DECLARATIONS = [
    {
        "name": "search_programs",
        "description": "Semantic search over English-taught master's programmes in Sweden that are open to international applicants and start in 2027. Use it for any question about which programmes exist. Returns up to top_k programmes with university, city, credits, tuition, deadline and link.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What the student wants to study, in natural language, e.g. 'machine learning and data science' or 'sustainable urban planning'. English or Swedish."},
                "city": {"type": "string", "description": "Optional: only programmes in this city, e.g. 'Uppsala', 'Gothenburg', 'Online (distance)'."},
                "max_tuition_sek": {"type": "number", "description": "Optional: maximum total tuition fee for the whole programme in SEK."},
                "credits": {"type": "number", "description": "Optional: 120 for a two-year master, 60 for a one-year master."},
                "university": {"type": "string", "description": "Optional: only this university (part of the name is enough)."},
                "top_k": {"type": "integer", "description": "Number of programmes to return, 1 to 10. Default 5."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_program_details",
        "description": "Full details of one programme: description, eligibility requirements, start date, duration, tuition instalments, application code and link.",
        "parameters": {
            "type": "object",
            "properties": {"program_id": {"type": "string", "description": "The program_id from search_programs."}},
            "required": ["program_id"],
        },
    },
    {
        "name": "get_city_info",
        "description": "Profile of a Swedish study city: population, location, train time to Stockholm, student rent and budget estimates with sources, character, pros and cons, and which universities offer programmes there.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name, e.g. 'Lund' or 'Göteborg'."}},
            "required": ["city"],
        },
    },
    {
        "name": "compare_cities",
        "description": "Side-by-side profiles of 2 to 4 cities, to help a student decide where to live and study.",
        "parameters": {
            "type": "object",
            "properties": {"cities": {"type": "array", "items": {"type": "string"}, "description": "2 to 4 city names."}},
            "required": ["cities"],
        },
    },
    {
        "name": "list_cities",
        "description": "All cities in the dataset with their number of programmes, and whether a detailed city profile exists.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_general_info",
        "description": "National facts for international students: residence permit maintenance requirement, average monthly student budget, who pays tuition fees, and how to apply.",
        "parameters": {"type": "object", "properties": {}},
    },
]
