"""ingest.py - parse, clean and chunk a knowledge base into data/chunks.jsonl

Usage: python ingest.py --kb kb --out data/chunks.jsonl --chunk-tokens 500 --overlap 80
Supports .md .txt .html .pdf
"""
import argparse, json, re, unicodedata, datetime as dt
from collections import Counter
from pathlib import Path

import tiktoken
from bs4 import BeautifulSoup
from pypdf import PdfReader

ENC = tiktoken.get_encoding("cl100k_base")


def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "").replace("\x00", "")
    text = re.sub(r"-\n(\w)", r"\1", text)                  # de-hyphenate line breaks
    text = re.sub(r"[^\S\n]+", " ", text)                    # collapse spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_boilerplate(pages: list[str]) -> list[str]:
    """Drop lines (headers/footers) repeated on >40% of pages, and bare page numbers."""
    if len(pages) < 4:
        return pages
    counts = Counter(l.strip() for p in pages for l in set(p.splitlines()) if l.strip())
    common = {l for l, c in counts.items() if c > 0.4 * len(pages) and len(l) < 120}
    out = []
    for p in pages:
        lines = [l for l in p.splitlines() if l.strip() not in common and not re.fullmatch(r"\s*(page )?\d+\s*", l, re.I)]
        out.append("\n".join(lines))
    return out


def load(path: Path):
    """Return list of (page_number|None, text)."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        pages = [(p.extract_text() or "") for p in PdfReader(str(path)).pages]
        pages = strip_boilerplate(pages)
        return [(i + 1, t) for i, t in enumerate(pages)]
    raw = path.read_text(encoding="utf-8", errors="replace")
    if ext in (".html", ".htm"):
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        raw = soup.get_text("\n")
    return [(None, raw)]


def guess_title(text: str, fallback: str) -> str:
    m = re.search(r"^#\s+(.+)$", text, re.M)
    if m:
        return m.group(1).strip()
    first = next((l.strip() for l in text.splitlines() if l.strip()), "")
    return first[:100] if 0 < len(first) <= 100 else fallback


def chunk_text(text: str, size: int, overlap: int):
    """Paragraph-aware token windows with overlap."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur, cur_len = [], [], 0
    for p in paras:
        n = len(ENC.encode(p))
        if n > size:  # oversized paragraph: hard split by tokens
            if cur:
                chunks.append("\n\n".join(cur)); cur, cur_len = [], 0
            toks = ENC.encode(p)
            step = size - overlap
            for i in range(0, len(toks), step):
                chunks.append(ENC.decode(toks[i:i + size]))
                if i + size >= len(toks):
                    break
            continue
        if cur_len + n > size and cur:
            chunks.append("\n\n".join(cur))
            # carry over tail of the previous chunk as overlap
            tail = ENC.decode(ENC.encode(chunks[-1])[-overlap:]) if overlap else ""
            cur, cur_len = ([tail] if tail else []), len(ENC.encode(tail)) if tail else 0
        cur.append(p); cur_len += n
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default="kb")
    ap.add_argument("--out", default="data/chunks.jsonl")
    ap.add_argument("--chunk-tokens", type=int, default=500)
    ap.add_argument("--overlap", type=int, default=80)
    ap.add_argument("--url-map", default=None, help="optional JSON {filename: url}")
    a = ap.parse_args()

    urls = json.loads(Path(a.url_map).read_text()) if a.url_map else {}
    files = sorted(p for p in Path(a.kb).rglob("*") if p.suffix.lower() in {".md", ".txt", ".html", ".htm", ".pdf"})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for path in files:
            date = dt.date.fromtimestamp(path.stat().st_mtime).isoformat()
            pages = load(path)
            title = guess_title(clean(pages[0][1]), path.stem)
            for page, raw in pages:
                for j, c in enumerate(chunk_text(clean(raw), a.chunk_tokens, a.overlap)):
                    rec = {
                        "chunk_id": f"{path.stem}:{page or 0}:{j}",
                        "text": c,
                        "source": path.name, "title": title, "page": page,
                        "url": urls.get(path.name), "date": date,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n"); n += 1
    print(f"{len(files)} files -> {n} chunks -> {a.out}")


if __name__ == "__main__":
    main()
