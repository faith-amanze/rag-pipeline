"""eval.py - retrieval + generation evaluation.

eval_set.jsonl, one per line:
 {"question": "...", "answer": "ground truth", "sources": ["doc1.md"], "answerable": true}
 (for out-of-KB questions: "answerable": false, "sources": [])

Usage: python eval.py --eval eval_set.jsonl -k 5 [--no-generation] [--ablate]
"""
import json, argparse, re
from statistics import mean
from retrieve_generate import Retriever, answer, llm

JUDGE = """You are grading a RAG answer. Reply with JSON only: {{"correct": 0|1, "faithful": 0|1}}
correct = answer agrees with the ground truth on the key facts.
faithful = every claim in the answer is supported by the context.
Question: {q}
Ground truth: {gt}
Answer: {a}
Context:
{ctx}"""
ABSTAIN = "i don't know based on the provided documents"


def retrieval_metrics(rows, retriever, k, hybrid):
    hit, rec, mrr = [], [], []
    for r in rows:
        if not r.get("answerable", True):
            continue
        srcs = [h["source"] for h in retriever.retrieve(r["question"], k, hybrid)]
        gold = set(r["sources"])
        hit.append(int(any(s in gold for s in srcs)))
        rec.append(len(gold & set(srcs)) / len(gold))
        rr = next((1 / (i + 1) for i, s in enumerate(srcs) if s in gold), 0)
        mrr.append(rr)
    return {f"hit@{k}": mean(hit), f"recall@{k}": mean(rec), "MRR": mean(mrr)}


def generation_metrics(rows, retriever, k, hybrid):
    per, out = [], {"correct": [], "faithful": [], "cite_valid": [], "cite_hits_gold": [], "abstain_ok": []}
    for r in rows:
        res = answer(r["question"], retriever, k, hybrid)
        abstained = ABSTAIN in res["answer"].lower()
        row = {"q": r["question"], "answer": res["answer"], "cited": res["cited_sources"], "abstained": abstained}
        if r.get("answerable", True):
            ctx = "\n\n".join(h["text"] for h in res["retrieved"])
            try:
                j = json.loads(re.search(r"\{.*\}", llm([{"role": "user", "content": JUDGE.format(
                    q=r["question"], gt=r["answer"], a=res["answer"], ctx=ctx)}]), re.S).group())
            except Exception:
                j = {"correct": 0, "faithful": 0}
            out["correct"].append(j["correct"]); out["faithful"].append(j["faithful"])
            out["cite_valid"].append(int(all(i <= len(res["retrieved"]) for i in res["cited_ids"]) and bool(res["cited_ids"]) or abstained))
            out["cite_hits_gold"].append(int(bool(set(res["cited_sources"]) & set(r["sources"]))))
            row.update(j)
        else:
            out["abstain_ok"].append(int(abstained))
            row["abstain_ok"] = int(abstained)
        per.append(row)
    return {k_: mean(v) for k_, v in out.items() if v}, per


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="eval_set.jsonl")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--no-generation", action="store_true")
    ap.add_argument("--ablate", action="store_true", help="compare dense-only / hybrid / hybrid+rerank retrieval")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.eval, encoding="utf-8") if l.strip()]
    report = {}
    if a.ablate:
        for name, hybrid, rerank in [("dense", False, False), ("hybrid", True, False), ("hybrid+rerank", True, True)]:
            report[name] = retrieval_metrics(rows, Retriever(rerank=rerank), a.k, hybrid)
            print(name, report[name])
    ret = Retriever(rerank=True)
    report["retrieval"] = retrieval_metrics(rows, ret, a.k, True); print("retrieval", report["retrieval"])
    if not a.no_generation:
        report["generation"], per = generation_metrics(rows, ret, a.k, True); print("generation", report["generation"])
        json.dump(per, open("eval_per_question.json", "w"), indent=1)
    json.dump(report, open("eval_results.json", "w"), indent=1)
