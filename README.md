# RAG Pipeline: ingest, index, hybrid retrieve, cited answers

A retrieval-augmented generation pipeline over a small knowledge base on technology, misinformation and privacy (Reuters Institute *Digital News Report 2026* and Pew Research Center privacy studies). It ingests documents, builds a vector index, retrieves relevant chunks, and answers with `\[S#]` citations and a confidence line, refusing when the documents don't contain the answer.

**Stack:** Python, sentence-transformers (`BAAI/bge-small-en-v1.5`), FAISS, BM25 + reciprocal-rank fusion, cross-encoder reranking, and any OpenAI-compatible LLM (tested with `openai/gpt-oss-120b` on Groq).

```
kb/ (pdf, html, md, txt) --ingest.py--> data/chunks.jsonl --index.py--> data/index.faiss
question --retrieve\_generate.py--> hybrid top-k -> rerank -> grounded prompt -> LLM -> answer + citations
eval\_set.jsonl --eval.py--> retrieval and generation metrics
```

## Demo

\*\*Factoid question\*\* (answer and cited sources)



!\[factoid](demo/factoid.png)



\*\*Multi-source question\*\* (answer combining the executive summary and the full report)



!\[multi-source](demo/multi\_source.png)



\*\*Out-of-scope question\*\* (the system refuses; note the negative reranker scores)



!\[refusal](demo/refusal.png)

## 

## Quickstart (Windows PowerShell)

Requires Python 3.12 (some dependencies do not yet support newer versions).

```powershell
python -m venv venv            # or: py -3.12 -m venv venv
venv\\Scripts\\Activate.ps1
pip install -r requirements.txt

# LLM: any OpenAI-compatible endpoint. Example with Groq (use your own key):
$env:OPENAI\_API\_KEY="<your key>"
$env:OPENAI\_BASE\_URL="https://api.groq.com/openai/v1"
$env:LLM\_MODEL="openai/gpt-oss-120b"     # check current names:
# python -c "from openai import OpenAI; \[print(m.id) for m in OpenAI().models.list().data]"

python fetch\_kb.py             # downloads the sample documents into kb/
python ingest.py --kb kb       # clean + chunk (500 tokens, 80 overlap)
python index.py                # embed + FAISS index
python retrieve\_generate.py "What percentage of Americans always or often read privacy policies?"
```

Never commit your API key. It is read from environment variables only.

## Evaluation

```powershell
python eval.py --eval eval\_set.jsonl --ablate    # dense vs hybrid vs hybrid+rerank, plus generation metrics
```

16 questions (12 answerable, 4 out-of-scope). Full analysis in [EVAL\_REPORT.md](EVAL_REPORT.md).

|Retrieval (k=5)|hit@5|MRR|
|-|-|-|
|Dense only|1.00|1.000|
|Hybrid|1.00|0.958|
|Hybrid + rerank|1.00|0.958|

|Generation|Score|
|-|-|
|Correct|0.917|
|Faithful|0.917|
|Valid citations|1.00|
|Correct refusals|1.00|

Key finding: aggregate scores hid two real failures, a "countries" vs "markets" vocabulary mismatch that caused a wrong refusal, and a wider context (k=10) that led the model to report a 40-country subset as the total. See the report.

## Files

|File|Purpose|
|-|-|
|`fetch\_kb.py`|Downloads the sample knowledge base into `kb/`|
|`ingest.py`|Parsing, cleaning, chunking, metadata|
|`index.py`|Embeddings and FAISS index (`--test "query"` for a sanity check)|
|`retrieve\_generate.py`|Hybrid retriever, reranker, prompt, LLM call, citations|
|`eval.py`, `eval\_set.jsonl`|Retrieval and generation evaluation|
|`debug\_ranks.py`|Lists questions where the right document was not ranked first|
|`EVAL\_REPORT.md`|Evaluation report and reflection|

## Limitations

Small eval set (one miss is about 8 points), document-level retrieval metrics, and the generator also acts as the judge.

