import json
from retrieve_generate import Retriever

rows = [json.loads(l) for l in open("eval_set.jsonl", encoding="utf-8") if l.strip()]
for name, hybrid, rerank in [("dense", False, False), ("hybrid", True, False), ("hybrid+rerank", True, True)]:
    r = Retriever(rerank=rerank)
    print("\n==", name)
    for row in rows:
        if not row.get("answerable", True):
            continue
        srcs = [h["source"] for h in r.retrieve(row["question"], 5, hybrid)]
        rank = next((i + 1 for i, s in enumerate(srcs) if s in set(row["sources"])), None)
        if rank != 1:
            print(" rank", rank, "|", row["question"][:80], "| top3:", srcs[:3])