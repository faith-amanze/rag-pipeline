"""retrieve_generate.py - hybrid retriever (BM25 + dense, RRF) -> optional cross-encoder rerank
-> grounded prompt with [S#] citations -> LLM.

LLM config (any OpenAI-compatible endpoint, incl. Ollama / vLLM / llama.cpp server):
  export OPENAI_API_KEY=...            (any string for local servers)
  export OPENAI_BASE_URL=http://localhost:11434/v1   # optional
  export LLM_MODEL=gpt-4o-mini                       # or llama3.1 etc.

Usage: python retrieve_generate.py "your question"   |   python retrieve_generate.py   (interactive)
"""
import os, re, sys, json, argparse

import faiss, tiktoken
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI
from index import EMBED_MODEL, QUERY_PREFIX, INDEX, load_chunks

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
ENC = tiktoken.get_encoding("cl100k_base")
tok = lambda s: re.findall(r"\w+", s.lower())

SYSTEM = """You are a careful assistant that answers ONLY from the provided context.
Rules:
1. Use only facts in the context blocks. Never use outside knowledge.
2. Cite every factual claim with the block id(s), e.g. [S1] or [S2][S3].
3. If the context does not contain the answer, reply exactly: "I don't know based on the provided documents." (no citations).
4. If the context is partial or conflicting, say so.
5. End with one line: "Confidence: High|Medium|Low - <one-sentence reason>"."""

USER = """Context:
{context}

Question: {question}

Answer (with citations):"""


class Retriever:
    def __init__(self, rerank=True, alpha_pool=4):
        self.chunks = load_chunks()
        self.index = faiss.read_index(INDEX)
        self.emb = SentenceTransformer(EMBED_MODEL)
        self.bm25 = BM25Okapi([tok(c["text"]) for c in self.chunks])
        self.reranker = CrossEncoder(RERANK_MODEL) if rerank else None
        self.pool = alpha_pool  # candidate pool = k * pool

    def retrieve(self, query, k=5, hybrid=True):
        n = k * self.pool
        q = self.emb.encode([QUERY_PREFIX + query], normalize_embeddings=True).astype("float32")
        _, dense_ids = self.index.search(q, n)
        dense_ids = [int(i) for i in dense_ids[0] if i >= 0]
        fused = {}
        for r, i in enumerate(dense_ids):
            fused[i] = fused.get(i, 0) + 1 / (60 + r)           # reciprocal rank fusion
        if hybrid:
            scores = self.bm25.get_scores(tok(query))
            for r, i in enumerate(scores.argsort()[::-1][:n]):
                fused[int(i)] = fused.get(int(i), 0) + 1 / (60 + r)
        cand = sorted(fused, key=fused.get, reverse=True)[:n]
        if self.reranker:
            ce = self.reranker.predict([(query, self.chunks[i]["text"]) for i in cand])
            cand = [i for _, i in sorted(zip(ce, cand), reverse=True)]
            score = dict(zip(cand, sorted(ce, reverse=True)))
        else:
            score = fused
        return [{**self.chunks[i], "score": float(score[i])} for i in cand[:k]]


def build_context(hits, max_tokens=3000):
    """Chunks are already ranked; add best-first until token budget is hit (truncate the last one)."""
    blocks, used, kept = [], 0, []
    for n, h in enumerate(hits, 1):
        head = f"[S{n}] (source: {h['source']}" + (f", p.{h['page']}" if h.get("page") else "") + f", title: {h['title']})\n"
        body = h["text"]
        cost = len(ENC.encode(head + body))
        if used + cost > max_tokens:
            room = max_tokens - used - len(ENC.encode(head))
            if room < 80:
                break
            body = ENC.decode(ENC.encode(body)[:room]) + " ..."
            cost = max_tokens - used
        blocks.append(head + body); kept.append(h); used += cost
    return "\n\n".join(blocks), kept


def llm(messages, model=None, temperature=0.0):
    client = OpenAI()  # reads OPENAI_API_KEY / OPENAI_BASE_URL
    r = client.chat.completions.create(model=model or os.getenv("LLM_MODEL", "gpt-4o-mini"),
                                       messages=messages, temperature=temperature)
    return r.choices[0].message.content.strip()


def answer(question, retriever, k=5, hybrid=True, max_ctx_tokens=3000):
    hits = retriever.retrieve(question, k=k, hybrid=hybrid)
    context, kept = build_context(hits, max_ctx_tokens)
    text = llm([{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER.format(context=context, question=question)}])
    text = re.sub(r"[【\[]\s*S(\d+)\s*[】\]]", r"[S\1]", text).replace("**", "")
    cited = sorted({int(n) for n in re.findall(r"[\[【]S(\d+)[\]】]", text)})
    return {"question": question, "answer": text, "retrieved": kept,
            "cited_ids": cited, "cited_sources": [kept[i - 1]["source"] for i in cited if i <= len(kept)]}


def show(res):
    print("\n" + res["answer"] + "\n\nSources:")
    for n, h in enumerate(res["retrieved"], 1):
        mark = "*" if n in res["cited_ids"] else " "
        print(f" {mark}[S{n}] {h['source']}" + (f" p.{h['page']}" if h.get("page") else "") + f" - {h['title']} (score {h['score']:.2f})")
    print(" (* = cited in answer)\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--no-hybrid", action="store_true")
    ap.add_argument("--no-rerank", action="store_true")
    a = ap.parse_args()
    r = Retriever(rerank=not a.no_rerank)
    if a.query:
        show(answer(a.query, r, a.k, not a.no_hybrid))
    else:
        while (q := input("Q> ").strip()):
            show(answer(q, r, a.k, not a.no_hybrid))
