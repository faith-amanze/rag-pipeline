"""index.py - embed chunks and build a FAISS index (cosine via normalized inner product).

Usage: python index.py                       # build
       python index.py --test "your query"   # sanity-check similarity search
"""
import argparse, json
from pathlib import Path

import faiss, numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge query instruction
CHUNKS, INDEX = "data/chunks.jsonl", "data/index.faiss"


def load_chunks(path=CHUNKS):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def build(chunks_path=CHUNKS, index_path=INDEX):
    chunks = load_chunks(chunks_path)
    model = SentenceTransformer(EMBED_MODEL)
    emb = model.encode([c["text"] for c in chunks], batch_size=64, normalize_embeddings=True,
                       show_progress_bar=True).astype("float32")
    index = faiss.IndexFlatIP(emb.shape[1])   # exact search; swap for IVF/HNSW at >1M chunks
    index.add(emb)
    faiss.write_index(index, index_path)
    print(f"indexed {index.ntotal} chunks (dim={emb.shape[1]}) -> {index_path}")


def test(query, k=5):
    chunks = load_chunks()
    index = faiss.read_index(INDEX)
    q = SentenceTransformer(EMBED_MODEL).encode([QUERY_PREFIX + query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q, k)
    for s, i in zip(scores[0], ids[0]):
        c = chunks[i]
        print(f"{s:.3f}  {c['chunk_id']}  | {c['text'][:110]!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", help="run a similarity query instead of building")
    a = ap.parse_args()
    test(a.test) if a.test else build()
