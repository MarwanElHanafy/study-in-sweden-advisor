"""Evaluate Kompass: agent with tools (RAG) vs the same model without tools (baseline).

Metrics
  retrieval_hit@k   retrieval questions: did the search return at least one programme that matches
                    the expected criteria (title keywords, city, budget)? Checked against data/programs.json.
  url_grounded      share of links in the answer that exist in the dataset or the city knowledge base.
  sek_grounded      share of SEK amounts (>= 1,000) in the answer that appear in the dataset or knowledge base.
  task_success      city questions: right tool used; fact question: correct figure in the answer.
  latency, LLM calls, tokens per answer.
Adversarial answers are written to the report for manual judgement.

Usage
  python src/evaluate.py --retrieval-only      # no API calls: search quality only
  python src/evaluate.py                       # full run (about 60 Gemini calls, paced for the free tier)
  python src/evaluate.py --only R1 R2          # a subset
Outputs go to eval/results/<timestamp>/ : answers.jsonl, summary.json, answers.md
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, median

from retriever import ProgramIndex, fold

QUESTIONS = Path("eval/questions.json")
OUT_ROOT = Path("eval/results")


# ---------------- ground truth helpers ----------------
def matches(r: dict, e: dict) -> bool:
    title = f" {fold(r['title'])} "
    if e.get("title_any") and not any(fold(k) in title for k in e["title_any"]):
        return False
    if e.get("city") and fold(r["city"]) != fold(e["city"]):
        return False
    if e.get("city_not") and fold(r["city"]) == fold(e["city_not"]):
        return False
    if e.get("max_tuition") and (r.get("tuition_total_sek") or 0) > e["max_tuition"]:
        return False
    return True


def known_urls_and_amounts(programs: list[dict]) -> tuple[set[str], set[int]]:
    urls = {norm_url(r["url"]) for r in programs if r.get("url")}
    amounts = set()
    for r in programs:
        for k in ("tuition_total_sek", "tuition_first_installment_sek"):
            if r.get(k):
                amounts.add(int(r[k]))
    kb = json.loads(Path("knowledge/cities.json").read_text(encoding="utf-8"))
    text = json.dumps(kb, ensure_ascii=False)
    urls |= {norm_url(u) for u in re.findall(r"https?://[^\s\"'),]+", text)}
    amounts |= {int(a.replace(",", "")) for a in re.findall(r"\b\d{1,3}(?:,\d{3})+\b|\b\d{4,6}\b", text)}
    return urls, amounts


def norm_url(u: str) -> str:
    return u.lower().rstrip("/.").replace("http://", "https://").replace("://www.", "://")


def extract_urls(text: str) -> list[str]:
    return [norm_url(u) for u in re.findall(r"https?://[^\s\)\]>\"']+", text or "")]


def extract_sek(text: str) -> list[int]:
    """Amounts written near 'SEK'/'kr', e.g. '290,000 SEK', 'SEK 72 500'."""
    found = []
    num = r"(\d{1,3}(?:[ ,. ]\d{3})+|\d{4,7})(?:\.\d{1,2})?"
    pat = rf"(?:SEK|kr)\s*{num}|{num}\s*(?:SEK|kr\b|kronor)"
    for a, b in re.findall(pat, text or "", flags=re.I):
        digits = re.sub(r"\D", "", a or b)
        if digits and int(digits) >= 1000:
            found.append(int(digits))
    return found


def share(found: list, known: set) -> tuple[int, int]:
    return sum(1 for x in found if x in known), len(found)


# ---------------- runs ----------------
def retrieval_only(index: ProgramIndex, questions: list[dict], k: int) -> dict:
    rows = []
    for q in questions:
        if q["type"] != "retrieval":
            continue
        e = q["expect"]
        gt = [r for r in index.programs if matches(r, e)]
        city = e.get("city")
        hits = index.search(q["question"], city=city, max_tuition_sek=e.get("max_tuition"), top_k=k)
        rel = [h for h in hits if matches(h.record, e)]
        first = next((i + 1 for i, h in enumerate(hits) if matches(h.record, e)), None)
        rows.append({"id": q["id"], "ground_truth_count": len(gt), "hit": bool(rel), "precision": len(rel) / max(1, len(hits)),
                     "reciprocal_rank": 1 / first if first else 0.0, "top": [h.record["title"] for h in hits[:3]]})
    valid = [r for r in rows if r["ground_truth_count"] > 0]
    return {"k": k, "questions": rows,
            f"hit@{k}": round(mean(r["hit"] for r in valid), 3) if valid else None,
            f"precision@{k}": round(mean(r["precision"] for r in valid), 3) if valid else None,
            "mrr": round(mean(r["reciprocal_rank"] for r in valid), 3) if valid else None}


def run_system(name: str, advisor, questions, programs, known_urls, known_amounts, pause: float) -> list[dict]:
    rows = []
    for i, q in enumerate(questions, 1):
        print(f"[{name}] {i}/{len(questions)} {q['id']}: {q['question'][:60]}")
        try:
            r = advisor.new_session().send(q["question"])
        except Exception as ex:
            rows.append({"system": name, "id": q["id"], "type": q["type"], "error": str(ex)})
            print("   error:", ex)
            time.sleep(pause)
            continue
        e = q["expect"]
        row = {"system": name, "id": q["id"], "type": q["type"], "question": q["question"], "answer": r.answer,
               "latency_s": r.latency_s, "llm_calls": r.llm_calls, "input_tokens": r.input_tokens,
               "output_tokens": r.output_tokens, "tools": [t["name"] for t in r.tool_calls]}
        u_ok, u_n = share(extract_urls(r.answer), known_urls)
        s_ok, s_n = share(extract_sek(r.answer), known_amounts)
        row.update(urls_grounded=u_ok, urls_total=u_n, sek_grounded=s_ok, sek_total=s_n)
        if q["type"] == "retrieval":
            by_id = {p["id"]: p for p in programs}
            retrieved = [by_id[i] for i in r.retrieved_ids if i in by_id]
            row["ground_truth_count"] = sum(1 for p in programs if matches(p, e))
            answerable = row["ground_truth_count"] > 0
            row["retrieval_hit"] = any(matches(p, e) for p in retrieved) if name == "rag" and answerable else None
        elif "tools_any" in e:
            row["task_success"] = any(t in row["tools"] for t in e["tools_any"])
        elif "answer_any" in e:
            row["task_success"] = any(s in r.answer for s in e["answer_any"])
        rows.append(row)
        time.sleep(pause)
    return rows


def summarise(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    def ratio(a, b):
        tot = sum(r[b] for r in ok)
        return round(sum(r[a] for r in ok) / tot, 3) if tot else None
    hits = [r["retrieval_hit"] for r in ok if r.get("retrieval_hit") is not None]
    succ = [r["task_success"] for r in ok if r.get("task_success") is not None]
    return {
        "answered": len(ok), "errors": len(rows) - len(ok),
        "retrieval_hit_rate": round(mean(hits), 3) if hits else None,
        "task_success_rate": round(mean(succ), 3) if succ else None,
        "url_grounded_share": ratio("urls_grounded", "urls_total"), "urls_total": sum(r["urls_total"] for r in ok),
        "sek_grounded_share": ratio("sek_grounded", "sek_total"), "sek_total": sum(r["sek_total"] for r in ok),
        "latency_median_s": round(median(r["latency_s"] for r in ok), 2) if ok else None,
        "latency_mean_s": round(mean(r["latency_s"] for r in ok), 2) if ok else None,
        "llm_calls_mean": round(mean(r["llm_calls"] for r in ok), 2) if ok else None,
        "tokens_mean": round(mean(r["input_tokens"] + r["output_tokens"] for r in ok)) if ok else None,
    }


def write_markdown(path: Path, rows: list[dict], summary: dict) -> None:
    lines = ["# Kompass evaluation", "", "```json", json.dumps(summary, indent=2), "```", ""]
    for r in rows:
        lines += [f"## {r['id']} [{r['system']}] {r.get('question', '')}", ""]
        if "error" in r:
            lines += [f"ERROR: {r['error']}", ""]
            continue
        lines += [f"tools: {r['tools']} | {r['latency_s']} s | urls grounded {r['urls_grounded']}/{r['urls_total']} | "
                  f"SEK grounded {r['sek_grounded']}/{r['sek_total']}", "", r["answer"], ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--only", nargs="*", help="question ids, e.g. R1 C1")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--pause", type=float, default=8.0, help="seconds between questions (free-tier rate limits)")
    ap.add_argument("--skip-baseline", action="store_true")
    args = ap.parse_args()

    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    if args.only:
        questions = [q for q in questions if q["id"] in set(args.only)]
    out = OUT_ROOT / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    index = ProgramIndex()
    retr = retrieval_only(index, questions, args.k)
    print("\n# Retrieval only (no LLM)")
    for r in retr["questions"]:
        print(f"  {r['id']:4s} gt={r['ground_truth_count']:3d} hit={r['hit']!s:5s} P={r['precision']:.2f} top={r['top'][:2]}")
    print({k: v for k, v in retr.items() if k != "questions"})
    summary = {"retrieval_only": {k: v for k, v in retr.items()}}

    if not args.retrieval_only:
        from agent import StudyAdvisor, load_advisor
        from tools import Toolbox
        programs = index.programs
        known_urls, known_amounts = known_urls_and_amounts(programs)
        rag = StudyAdvisor(Toolbox(index), use_tools=True)
        rows = run_system("rag", rag, questions, programs, known_urls, known_amounts, args.pause)
        if not args.skip_baseline:
            rows += run_system("baseline", load_advisor(use_tools=False), questions, programs, known_urls, known_amounts, args.pause)
        with (out / "answers.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary["rag"] = summarise([r for r in rows if r["system"] == "rag"])
        if not args.skip_baseline:
            summary["baseline"] = summarise([r for r in rows if r["system"] == "baseline"])
        write_markdown(out / "answers.md", rows, {k: v for k, v in summary.items() if k != "retrieval_only"})
        print("\n# Agent vs baseline")
        print(json.dumps({k: v for k, v in summary.items() if k != "retrieval_only"}, indent=2))

    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
