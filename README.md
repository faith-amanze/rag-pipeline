# RAG Pipeline: ingest -> index -> hybrid retrieve -> grounded, cited answers

Stack: Python, sentence-transformers (BGE) embeddings, FAISS, BM25 + reciprocal-rank-fusion hybrid search,
optional cross-encoder rerank, any OpenAI-compatible LLM (OpenAI, or local Llama via Ollama/vLLM).

```
kb/ (pdf, md, txt, html) --ingest.py--> data/chunks.jsonl --index.py--> data/index.faiss
question --retrieve_generate.py--> hybrid top-k -> rerank -> prompt with [S#] ids -> LLM -> answer + citations + confidence
eval_set.jsonl --eval.py--> hit@k, recall@k, MRR, correctness, faithfulness, citation + abstention checks
```

## Quickstart
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...                 # or a dummy key + OPENAI_BASE_URL=http://localhost:11434/v1
export LLM_MODEL=gpt-4o-mini                 # or llama3.1 for Ollama

cp your_docs/* kb/
python ingest.py --kb kb                     # clean + chunk (500 tok, 80 overlap) + metadata
python index.py                              # embed + FAISS
python index.py --test "sample query"        # sanity check similarity search
python retrieve_generate.py "your question"  # answer with citations
python retrieve_generate.py                  # interactive

cp eval_set.example.jsonl eval_set.jsonl     # write 30-50 real questions
python eval.py --eval eval_set.jsonl --ablate
```

## Design notes
- **Chunking:** paragraph-aware, 500 tokens, 80 overlap; metadata = source, title, page, date, optional URL (`--url-map`).
- **Retrieval:** dense (BGE with query instruction) + BM25 fused with RRF, candidate pool 4k, cross-encoder rerank to top-k (k=3-8).
- **Prompt:** context-only rule, mandatory `[S#]` citations, exact refusal string, confidence line. Token budget enforced in `build_context` (best chunks first, last one truncated).
- **Eval:** retrieval (hit@k / recall@k / MRR, with dense vs hybrid vs rerank ablation) and generation (LLM-judged correctness + faithfulness, citation validity, correct abstention on out-of-KB questions).
- Swap-outs: embeddings -> OpenAI/Cohere (edit `index.py`), vector DB -> Pinecone/Weaviate/Milvus (replace FAISS calls in `index.py` and `Retriever`).

## Demo
Run the interactive mode and record a GIF (e.g. with `asciinema` + `agg`, or ScreenToGif) showing 3 queries: a factoid, a multi-chunk question and an out-of-KB question that is refused. Save as `demo.gif`.
