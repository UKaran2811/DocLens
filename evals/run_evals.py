"""Eval harness: python -m evals.run_evals --dataset evals/dataset.jsonl

Metrics
  hit@k          a gold (source, page) chunk was retrieved
  MRR            1 / rank of the first gold chunk
  cited_gold     the answer actually cites a gold chunk
  keywords       all `must_mention` strings appear in the answer
  faithful       LLM judge: every claim is supported by the retrieved excerpts
  abstain_rate   on unanswerable questions, the system says "not in the manuals"
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

from app.rag import answer, make_llm, to_text

from app.config import JUDGE_MODEL, TOP_K

JUDGE = """You are grading a RAG answer.
Question: {question}

Excerpts the system retrieved:
{context}

Answer given:
{answer}

Return ONLY JSON: {{"faithful": true|false, "abstained": true|false, "reason": "<one sentence>"}}
- faithful: every factual claim in the answer is supported by the excerpts.
- abstained: the answer says the manuals do not contain the information."""


def judge(question: str, out: dict) -> dict:
    ctx = "\n\n".join(f"[{s['ref']}] {s['text']}" for s in out["sources"])
    msg = make_llm(JUDGE_MODEL, 300).invoke(
        JUDGE.format(question=question, context=ctx, answer=out["answer"])
    )
    raw = re.sub(r"^```(?:json)?|```$", "", to_text(msg.content).strip(), flags=re.M).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"faithful": False, "abstained": False, "reason": f"unparseable judge output: {raw[:80]}"}


def gold_refs(out: dict, row: dict) -> list[int]:
    pages = set(row.get("expected_pages") or [])
    return [
        s["ref"]
        for s in out["sources"]
        if s["source"] == row["expected_source"] and (not pages or s["page"] in pages)
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="evals/dataset.jsonl")
    ap.add_argument("--k", type=int, default=TOP_K)
    ap.add_argument("--min-hit", type=float, default=0.0, help="exit 1 if hit@k falls below this")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.dataset).read_text().splitlines() if l.strip()]
    results = []
    for row in rows:
        out = answer(row["question"], k=args.k)
        j = judge(row["question"], out)
        rec = {"question": row["question"], "answer": out["answer"], "judge_reason": j.get("reason")}
        if row.get("answerable", True):
            gold = gold_refs(out, row)
            rec |= {
                "hit": bool(gold),
                "rr": 1 / min(gold) if gold else 0.0,
                "cited_gold": any(r in gold for r in out["cited_refs"]),
                "keywords": all(k.lower() in out["answer"].lower() for k in row.get("must_mention", [])),
                "faithful": bool(j.get("faithful")),
            }
        else:
            rec["abstained"] = bool(j.get("abstained"))
        results.append(rec)
        print(("OK  " if rec.get("hit", rec.get("abstained")) else "MISS"), row["question"][:80])

    ans = [r for r in results if "hit" in r]
    unans = [r for r in results if "abstained" in r]
    mean = lambda key, rs: round(statistics.mean(r[key] for r in rs), 3) if rs else None
    summary = {
        "n_answerable": len(ans),
        "hit@k": mean("hit", ans),
        "mrr": mean("rr", ans),
        "cited_gold": mean("cited_gold", ans),
        "keywords": mean("keywords", ans),
        "faithful": mean("faithful", ans),
        "n_unanswerable": len(unans),
        "abstain_rate": mean("abstained", unans),
    }
    print(json.dumps(summary, indent=2))
    Path("evals/results.json").write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    return 1 if (summary["hit@k"] or 0) < args.min_hit else 0


if __name__ == "__main__":
    sys.exit(main())
