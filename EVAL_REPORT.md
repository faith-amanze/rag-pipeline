# Evaluation Report: RAG over technology, misinformation and privacy sources

## 1. Dataset
**Knowledge base (6 documents, 573 chunks).** Reuters Institute *Digital News Report 2026* (178-page PDF plus its executive summary page) and four Pew Research Center pieces on data privacy (2019 x2, 2020, 2023). Chunking: 500 tokens, 80 overlap, paragraph-aware; metadata = source, title, page, date.

**Eval set (16 questions).** 12 answerable factual questions (each with a ground-truth answer and the file(s) that contain it) and 4 out-of-scope questions (e.g. "What is the capital of Australia?") where the correct behaviour is to refuse. Questions and answers were drafted with AI assistance from source summaries and then checked against the documents. One label was corrected after retrieval analysis (the 4,272-respondent question is answered by two Pew pages, so both are gold sources). *[Confirm: I verified all ground-truth answers against the source pages.]*

## 2. Setup
Embeddings `BAAI/bge-small-en-v1.5` (FAISS flat inner product, cosine); BM25 + reciprocal-rank fusion for hybrid search; cross-encoder rerank `ms-marco-MiniLM-L-6-v2`; k=5; generator and judge `openai/gpt-oss-120b` via Groq, temperature 0. The prompt requires context-only answers, `[S#]` citations, an exact refusal string and a confidence line.

## 3. Results
**Retrieval (12 answerable questions; document-level match).**

| Method | hit@1 | recall@1 | MRR (k=1) | hit@5 | recall@5 | MRR (k=5) |
|---|---|---|---|---|---|---|
| Dense only | 1.000 | 0.875 | 1.000 | 1.00 | 1.00 | 1.000 |
| Hybrid (BM25 + dense) | 0.917 | 0.792 | 0.917 | 1.00 | 1.00 | 0.958 |
| Hybrid + rerank | 0.917 | 0.792 | 0.917 | 1.00 | 1.00 | 0.958 |

recall@1 has a ceiling of 0.875 because three questions have two acceptable source files; dense search reached it. Differences are one question out of 12 and are not statistically meaningful.

**Generation (hybrid + rerank, k=5).**

| Metric | Score |
|---|---|
| Answer correctness (LLM judge) | 0.917 (11/12) |
| Faithfulness (LLM judge) | 0.917 (11/12) |
| Valid citations (point to retrieved chunks) | 1.000 |
| Cites a gold source | 0.917 (11/12) |
| Correct refusal on out-of-scope questions | 1.000 (4/4) |

The 11/12 scores all come from one question (failure 1 below).

## 4. Qualitative failure cases
1. **Over-abstention from a vocabulary mismatch.** "How many *countries* does the 2026 report cover?" returned "I don't know". The index contains the answer, but the report says "48 *markets*", and the right document was retrieved (so hit@5 stayed 1.0) without the passages stating the number. Rewording to "markets" gave the correct answer (48, page 4). The model also printed "Confidence: High" beside a refusal, which is inconsistent.
2. **Misreading from wider context (found outside the eval set).** Re-asking the same question with k=10 returned "40 countries" with a citation to page 63. That page says "the 40 countries we covered in 2020 and again in 2026", a subset used for trend analysis. The model reported a subset as the total and the answer was wrong, even though the citation was real.
3. **Reranker demoted a correct chunk.** On the advertisers-and-children's-data question, dense search ranked the 2023 Pew page first, while hybrid + rerank put a 2019 page first. A 2013 phone-survey question showed a similar BM25 effect in hybrid. Both were rank-2 misses.
4. **Eval-set labelling flaw (fixed).** The 4,272-respondent question was marked as answered by one file, but a second Pew page repeats the figure, so correct retrievals were counted as misses.

## 5. Hallucination analysis
Across the 16 questions the system never produced an unsupported number at k=5. All 4 out-of-scope questions were refused, and the one answerable miss was a refusal rather than an invented figure. In the out-of-scope cases the reranker scores were strongly negative (about -7 to -10) versus positive (about 2.5 to 8.7) for in-scope questions, so retrieval gave a clear "nothing relevant here" signal that the prompt turned into a refusal.

Retrieval is not a complete safeguard. Failure 2 shows that more context can introduce a wrong answer: the model took a figure from the wrong scope and cited it. Mitigations that fit what we saw: a retrieval-score threshold that refuses before calling the LLM, query expansion with synonyms ("markets"), keeping k modest, and requiring the model to quote the exact sentence behind any number. A no-retrieval baseline was not run, so I cannot quantify how much retrieval reduced hallucination relative to the bare model.

## 6. Limitations
- 12 answerable and 4 out-of-scope questions: one miss moves a score by about 8 points, so figures are indicative, not precise.
- Retrieval metrics are document-level. Failure 1 passed hit@5 but missed the passage; chunk-level gold labels would catch this.
- The same model generated and judged the answers, which tends to make grading lenient.
- Out-of-scope questions were easy (unrelated trivia); topic-adjacent ones would be a harder test.

## 7. Reflection
The pipeline worked end to end: ingestion, a FAISS index, hybrid retrieval with reranking, and grounded answers whose citations were valid every time, with all out-of-scope questions refused. The main challenge was that the failures were subtle. A document-level retrieval score of 100% hid a missed passage, and a correct-looking cited answer ("40 countries") was wrong because the model confused a subset with the total. Hybrid search and reranking gave no measurable gain over plain dense search on a small, clean corpus of six documents, so the extra complexity was not justified here and would likely matter only on a larger or noisier knowledge base. The most useful finding came from reading failures, not aggregate scores. The next improvements are chunk-level evaluation labels, a reranker-score threshold that refuses before the LLM call, query expansion for vocabulary mismatches, a larger eval set with topic-adjacent out-of-scope questions, and a different judge model from the generator.
