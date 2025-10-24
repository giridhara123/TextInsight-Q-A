# src/utils/textinsight.py
from __future__ import annotations

import os, uuid
import re
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from PIL import Image

# --- robust imports across LangChain versions (kept/adapted) ---
try:
    from langchain_community.vectorstores import Chroma
except Exception:  # older LC
    from langchain.vectorstores import Chroma

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:  # newer splitters package
    from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
except Exception:
    # older LC package locations
    from langchain.document_loaders import WebBaseLoader, PyPDFLoader

try:
    from langchain.schema import Document
except Exception:
    # newer location
    from langchain_core.documents import Document

from utils.load_config import APPCFG

# ---- image extensions + OCR availability ----
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"}

# Try to import OCR stack
try:
    import pytesseract
    _OCR_OK = True
except Exception:
    _OCR_OK = False

# Try to locate the tesseract binary (macOS Homebrew first, then common paths, then env)
if _OCR_OK:
    from pathlib import Path
    _CANDIDATES = [
        os.getenv("TESSERACT_CMD"),
        "/opt/homebrew/bin/tesseract",   # Apple Silicon (brew)
        "/usr/local/bin/tesseract",      # Intel macOS (brew)
        "/usr/bin/tesseract",            # Linux
    ]
    _CANDIDATES = [p for p in _CANDIDATES if p and Path(p).exists()]
    if _CANDIDATES:
        pytesseract.pytesseract.tesseract_cmd = _CANDIDATES[0]

# ---- Chunker like your TextInsight settings ----
_TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=120, length_function=len,
    separators=["\n\n", "\n", " ", ""],
)

MONTHS_RE = r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"

def _normalize_url(u: str) -> str:
    u = (u or "").strip()
    u = re.sub(r"[?&](utm_[^=]+|gclid|fbclid)=[^&]+", "", u, flags=re.I)
    return u[:-1] if u.endswith("/") else u


@dataclass
class Cite:
    id: str
    collection: str
    source: str
    preview: str
    distance: float


# ---------- helpers: image facts extraction (NEW) ----------
def _extract_image_facts(text: str) -> Dict[str, str]:
    """Pull common 'poster/invoice/warranty' facts from OCR text."""
    t = (text or "").strip()

    facts: Dict[str, str] = {}

    # Event name: look for patterns like "Something Summit 2025", "Conference/Expo"
    m = re.search(r"([A-Z][A-Za-z@&\s\-]+?(?:Summit|Conference|Expo)\s*20\d{2})", t, re.I)
    if m:
        facts["event_name"] = m.group(1).strip()

    # Event dates: e.g., "April 14-16, 2025" or "Jun 12–14, 2025"
    m = re.search(rf"{MONTHS_RE}\s+\d{{1,2}}\s*(?:–|-|to)\s*\d{{1,2}},\s*20\d{{2}}", t, re.I)
    if m:
        facts["event_dates"] = m.group(0)

    # Venue / hall
    m = re.search(r"(Convention Center|Center|Hall\s*[A-Z0-9]+)[^\n]*", t, re.I)
    if m:
        facts["venue"] = m.group(0).strip()

    # Keynote time/title/speaker
    m_time = re.search(r"(Opening\s+Keynote|Keynote)[^\n]*?(\d{1,2}:\d{2}\s*(?:AM|PM)?(?:\s*[–-]\s*\d{1,2}:\d{2}\s*(?:AM|PM)?)?)", t, re.I)
    if m_time:
        facts["keynote_time"] = (m_time.group(2) or "").strip()

    m_title = re.search(r"(?:Keynote|Opening Keynote)[:\-\s]+“([^”]+)”|\"([^\"]+)\"", t)
    if m_title:
        facts["keynote_title"] = (m_title.group(1) or m_title.group(2) or "").strip()

    m_speaker = re.search(r"(?:Keynote\s*by|Speaker|by)\s*([A-Z][A-Za-z\.\-]+(?:\s+[A-Z][A-Za-z\.\-]+)+)", t)
    if m_speaker:
        facts["keynote_speaker"] = m_speaker.group(1).strip()

    # Early-bird
    m = re.search(r"(early[-\s]?bird)[^\n]*?(\d{1,2})%\s*off[^\n]*?(?:ends|until)\s*([A-Za-z]+\s*\d{1,2},\s*20\d{2})", t, re.I)
    if m:
        facts["early_bird"] = f"{m.group(2)}% off; ends {m.group(3)}"

    # Ticket prices
    for tier in ("Standard", "Student", "Team", "Workshops?"):
        m = re.search(rf"\b({tier})\b[^\n$]*\$?\s?(\d{{2,5}})", t, re.I)
        if m:
            facts[f"price_{m.group(1).lower().rstrip('s')}"] = m.group(2)

    # Expo hours
    m = re.search(r"(Expo|Exhibition)[^\n]*?(\d{1,2}:\d{2})\s*[–-]\s*(\d{1,2}:\d{2})\s*(daily|each day)?", t, re.I)
    if m:
        facts["expo_hours"] = f"{m.group(2)}–{m.group(3)}" + (" daily" if m.group(4) else "")

    # Invoice number (for invoice-like tests)
    m = re.search(r"(Invoice\s*(No\.?|#)\s*[:\-]?\s*([A-Za-z0-9\-]+))", t, re.I)
    if m:
        facts["invoice_number"] = m.group(3)

    # Warranty phrases
    m = re.search(r"(warranty|guarantee)\s*(?:period|:)?\s*([0-9]+)\s*(year|month)s?", t, re.I)
    if m:
        qty = int(m.group(2))
        unit = m.group(3).lower()
        facts["warranty"] = f"{qty} {unit}{'' if qty==1 else 's'}"
        if unit.startswith("year"):
            facts["warranty_months"] = str(qty * 12)
        else:
            facts["warranty_months"] = str(qty)

    return facts


def _direct_answer_from_text(question: str, text: str, meta: Dict[str, str]) -> Optional[str]:
    """Heuristic extractor for common Q types before LLM."""
    q = (question or "").lower()
    t = (text or "")
    facts = _extract_image_facts(t)

    def has(*keys):
        return all(k in facts and facts[k] for k in keys)

    # Event name
    if "event name" in q or ("what" in q and "event" in q and "name" in q):
        if has("event_name"):
            return facts["event_name"]

    # Event dates
    if "event date" in q or "dates" in q:
        if has("event_dates"):
            return facts["event_dates"]

    # Venue
    if "venue" in q or "where" in q:
        if has("venue"):
            return facts["venue"]

    # Keynote time
    if ("keynote" in q and "time" in q) or ("opening keynote" in q and "time" in q):
        if has("keynote_time"):
            return facts["keynote_time"]

    # Keynote speaker
    if ("keynote" in q and ("who" in q or "speaker" in q)):
        if has("keynote_speaker"):
            return facts["keynote_speaker"]

    # Keynote title
    if ("keynote" in q and "title" in q):
        if has("keynote_title"):
            return f"“{facts['keynote_title']}”"

    # Early-bird discount/deadline
    if "early-bird" in q or "early bird" in q:
        if has("early_bird"):
            return facts["early_bird"]

    # Ticket price queries
    if "standard" in q and "price" in q and "price_standard" in facts:
        return f"${facts['price_standard']}"
    if "student" in q and "price" in q and "price_student" in facts:
        return f"${facts['price_student']}"
    if ("team" in q or "4-pack" in q) and "price" in q and "price_team" in facts:
        return f"${facts['price_team']}"
    if "workshop" in q and "price" in q and "price_workshop" in facts:
        return f"${facts['price_workshop']}"

    # Expo hours
    if "expo" in q and ("hour" in q or "time" in q or "schedule" in q):
        if has("expo_hours"):
            return facts["expo_hours"]

    # Invoice number
    if "invoice" in q and ("no" in q or "number" in q):
        if has("invoice_number"):
            return facts["invoice_number"]

    # Warranty
    if "warranty" in q:
        # months requested?
        if "month" in q and has("warranty_months"):
            return f"{facts['warranty_months']} months"
        if has("warranty"):
            return facts["warranty"]

    return None


# ---------------- Ingest URLs ----------------
def ingest_urls(urls: List[str]) -> Dict[str, int]:
    col = APPCFG.chroma_client.get_or_create_collection(
        APPCFG.TI_COLLECTIONS.get("urls", "ti_web_urls")
    )

    # normalize + dedupe input URLs
    cleaned: List[str] = []
    seen = set()
    for u in urls or []:
        nu = _normalize_url(u)
        if nu and nu not in seen:
            cleaned.append(nu)
            seen.add(nu)

    if not cleaned:
        return {"added": 0, "files": []}

    # Load all URLs
    try:
        docs = WebBaseLoader(cleaned).load()
    except ModuleNotFoundError as e:
        # Commonly missing: bs4/lxml/html5lib
        raise RuntimeError(
            "BeautifulSoup-based HTML parsing is missing. Install: "
            "`pip install beautifulsoup4 lxml html5lib`"
        ) from e

    if not docs:
        # No content; still reflect which URLs we tried
        return {"added": 0, "files": [{"name": u, "chunks": 0} for u in cleaned]}

    # Split into chunks
    chunks = _TEXT_SPLITTER.split_documents(docs)
    if not chunks:
        return {"added": 0, "files": [{"name": u, "chunks": 0} for u in cleaned]}

    # Build vectors + per-URL chunk counts
    texts, ids, metas = [], [], []
    per_url_counts = {u: 0 for u in cleaned}

    for i, c in enumerate(chunks):
        text = c.page_content or ""
        meta = dict(c.metadata or {})
        src = _normalize_url(meta.get("source") or meta.get("url") or "")
        if not src:
            # fallback to first provided url if loader didn't set source
            src = cleaned[0]
        per_url_counts[src] = per_url_counts.get(src, 0) + 1

        texts.append(text)
        metas.append({**meta, "source": src})
        ids.append(f"{src}#chunk={i}")

    try:
        vecs = APPCFG.embeddings.embed_documents(texts)
    except ModuleNotFoundError as e:
        # Helpful hint when embeddings come from langchain-openai
        raise RuntimeError(
            "Embeddings backend is missing. If you use LangChain OpenAI embeddings, install: "
            "`pip install langchain-openai` and set your API key."
        ) from e

    col.upsert(ids=ids, documents=texts, embeddings=vecs, metadatas=metas)

    # Build per-file list (include any normalized sources not in original)
    per_file = [{"name": u, "chunks": per_url_counts.get(u, 0)} for u in cleaned]
    for src, cnt in per_url_counts.items():
        if src not in seen:
            per_file.append({"name": src, "chunks": cnt})

    return {"added": len(ids), "files": per_file}


# ---------------- Ingest PDFs ----------------
def ingest_pdfs(file_paths: List[str]) -> Dict[str, int]:
    col = APPCFG.chroma_client.get_or_create_collection(
        APPCFG.TI_COLLECTIONS.get("pdfs", "ti_pdf_docs")
    )
    texts, ids, metas = [], [], []
    total_chunks = 0
    per_file = []

    for path in file_paths or []:
        if not path or not os.path.exists(path):
            continue
        try:
            docs = PyPDFLoader(path).load()
        except ModuleNotFoundError as e:
            # PyPDFLoader depends on pypdf
            raise RuntimeError(
                "pypdf is required for PDF ingestion. Install with: `pip install pypdf`"
            ) from e
        except Exception as e:
            # Any other loader error
            raise RuntimeError(f"PDF loader failed for {path}: {e}") from e

        if not docs:
            per_file.append({"name": os.path.basename(path), "chunks": 0})
            continue

        chunks = _TEXT_SPLITTER.split_documents(docs)
        n_chunks = 0
        for i, c in enumerate(chunks):
            src = c.metadata.get("source", path)
            page = c.metadata.get("page", None)
            meta = {**c.metadata, "source": src}
            texts.append(c.page_content)
            ids.append(f"{src}#page={page}_chunk={i}")
            metas.append(meta)
            total_chunks += 1
            n_chunks += 1
        per_file.append({"name": os.path.basename(path), "chunks": n_chunks})

    if not texts:
        return {"added": 0, "files": per_file}

    try:
        vecs = APPCFG.embeddings.embed_documents(texts)
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Embeddings backend is missing. If you use LangChain OpenAI embeddings, install: "
            "`pip install langchain-openai` and set your API key."
        ) from e

    col.upsert(ids=ids, documents=texts, embeddings=vecs, metadatas=metas)
    return {"added": total_chunks, "files": per_file}


# ---------------- Ingest Images (OCR) ----------------
def ingest_images(file_paths: List[str]) -> Dict[str, int]:
    """
    OCR images → store as a SINGLE chunk when possible → upsert into Chroma.
    Also returns an OCR preview and metadata with extracted facts to aid retrieval.
    """
    col_name = APPCFG.TI_COLLECTIONS.get("images", "ti_images")
    col = APPCFG.chroma_client.get_or_create_collection(col_name)

    texts, ids, metas = [], [], []
    per_file = []
    total_chunks = 0

    for p in file_paths or []:
        if not p or not os.path.exists(p):
            continue
        base = os.path.basename(p)

        # Try OCR with light preprocessing
        ocr_text = ""
        if _OCR_OK:
            try:
                img = Image.open(p)
                # upscale small images to help OCR
                w, h = img.size
                if max(w, h) < 1200:
                    scale = 2
                    img = img.resize((w * scale, h * scale))
                # grayscale + simple threshold helps bold/print
                img = img.convert("L")
                img = img.point(lambda x: 0 if x < 180 else 255, mode="1")
                # page segmentation: assume a block of text
                ocr_text = pytesseract.image_to_string(img, config="--psm 6") or ""
            except Exception:
                ocr_text = ""

        if not ocr_text.strip():
            # Fallback: at least index something searchable
            ocr_text = base

        # Prefer 1 big chunk for posters/invoices so phrases stay together
        text_chunk = ocr_text.strip()
        facts = _extract_image_facts(text_chunk)

        texts.append(text_chunk)
        ids.append(f"{p}#img_chunk=0")   # deterministic ID → safe upsert
        meta = {"source": p, "type": "image", "file": base}
        # include helpful facts in metadata (only lightweight keys)
        for k, v in facts.items():
            # Keep metadata short-ish
            if isinstance(v, str) and len(v) > 160:
                v = v[:160]
            meta[f"fact:{k}"] = v
        metas.append(meta)

        per_file.append({
            "name": base,
            "chunks": 1,
            "ocr_chars": len(text_chunk),
            "ocr_preview": text_chunk.replace("\n", " ")[:140]
        })
        total_chunks += 1

    if not texts:
        return {"added": 0, "files": per_file}

    try:
        vecs = APPCFG.embeddings.embed_documents(texts)
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Embeddings backend is missing. If you use LangChain OpenAI embeddings, install: "
            "`pip install langchain-openai` and set your API key."
        ) from e

    if hasattr(col, "upsert"):
        col.upsert(ids=ids, documents=texts, embeddings=vecs, metadatas=metas)
    else:
        col.add(ids=ids, documents=texts, embeddings=vecs, metadatas=metas)

    return {"added": total_chunks, "files": per_file}


# ---------------- Ingest CSV/XLSX to Chroma ----------------
def ingest_csv_xlsx_to_chroma(file_paths):
    """
    Create semantic 'snippets' for CSV/XLSX into the 'csv_xlsx_docs' collection.
    Returns {'rows': total_rows, 'files': [{name, rows, chunks}]}
    """
    # Use the same direct Chroma client path used elsewhere, to avoid extra deps
    coll_name = APPCFG.TI_COLLECTIONS.get("csv_xlsx_docs", "ti_tabular_docs")
    col = APPCFG.chroma_client.get_or_create_collection(coll_name)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    docs_texts: List[str] = []
    docs_meta: List[Dict] = []
    docs_ids: List[str] = []

    total_rows = 0
    per_file = []

    for p in file_paths or []:
        path = p
        if not path or not os.path.exists(path):
            candidate = os.path.join(APPCFG.uploads_dir.as_posix(), os.path.basename(path or ""))
            if os.path.exists(candidate):
                path = candidate
            else:
                per_file.append({"name": os.path.basename(path or "?"), "rows": 0, "chunks": 0})
                continue

        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".csv":
                df = pd.read_csv(path)
            elif ext in (".xlsx", ".xls"):
                df = pd.read_excel(path)
            else:
                per_file.append({"name": os.path.basename(path), "rows": 0, "chunks": 0})
                continue
        except Exception:
            per_file.append({"name": os.path.basename(path), "rows": 0, "chunks": 0})
            continue

        rows = int(len(df))
        if rows == 0:
            per_file.append({"name": os.path.basename(path), "rows": 0, "chunks": 0})
            continue

        total_rows += rows
        text_blob = df.to_csv(index=False)
        chunks = splitter.split_text(text_blob)

        for i, chunk in enumerate(chunks):
            docs_texts.append(chunk)
            docs_meta.append({
                "source": os.path.basename(path),
                "type": "table",
                "rows": rows,
                "chunk": i
            })
            # deterministic ID per file/chunk (safe for upsert)
            docs_ids.append(f"{os.path.basename(path)}#tab_chunk={i}")

        per_file.append({"name": os.path.basename(path), "rows": rows, "chunks": len(chunks)})

    if docs_texts:
        try:
            embs = APPCFG.embeddings.embed_documents(docs_texts)
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "Embeddings backend is missing. If you use LangChain OpenAI embeddings, install: "
                "`pip install langchain-openai` and set your API key."
            ) from e

        if hasattr(col, "upsert"):
            col.upsert(ids=docs_ids, documents=docs_texts, embeddings=embs, metadatas=docs_meta)
        else:
            col.add(ids=docs_ids, documents=docs_texts, embeddings=embs, metadatas=docs_meta)

    return {"rows": total_rows, "files": per_file}


# ---------------- Retrieval + Answer ----------------
def _safe_query_collection(col, qvec, k: int):
    """Query a collection safely with right-sized k and a simple fallback."""
    try:
        count = 0
        try:
            count = int(col.count())
        except Exception:
            count = 0
        n = max(1, min(k, count if count else k))
        out = col.query(query_embeddings=[qvec], n_results=n)
        return out
    except Exception:
        # Fallback: return first N docs if query fails
        try:
            rec = col.get(include=["documents", "metadatas", "ids"], limit=k)
            return {
                "documents": [rec.get("documents", [])],
                "metadatas": [rec.get("metadatas", [])],
                "ids": [rec.get("ids", [])],
                "distances": [[0.0 for _ in rec.get("ids", [])]],
            }
        except Exception:
            return {"documents": [[]], "metadatas": [[]], "ids": [[]], "distances": [[]]}


def retrieve_top_k(query: str, top_k: int = 4) -> List[Cite]:
    try:
        qvec = APPCFG.embeddings.embed_query(query)
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Embeddings backend is missing. If you use LangChain OpenAI embeddings, install: "
            "`pip install langchain-openai` and set your API key."
        ) from e

    results: List[Cite] = []

    for cname in (
        APPCFG.TI_COLLECTIONS.get("images", "ti_images"),         # prioritize images first
        APPCFG.TI_COLLECTIONS.get("pdfs", "ti_pdf_docs"),
        APPCFG.TI_COLLECTIONS.get("urls", "ti_web_urls"),
        APPCFG.TI_COLLECTIONS.get("csv_xlsx_docs", "ti_tabular_docs"),
    ):
        try:
            col = APPCFG.chroma_client.get_collection(name=cname)
        except Exception:
            continue

        out = _safe_query_collection(col, qvec, top_k)
        docs = (out.get("documents") or [[]])[0]
        metas = (out.get("metadatas") or [[]])[0]
        ids = (out.get("ids") or [[]])[0]
        dists = (out.get("distances") or [[]])[0]
        for doc, meta, _id, dist in zip(docs, metas, ids, dists):
            src = (meta or {}).get("source", "")
            preview = (doc or "")[:220].strip().replace("\n", " ")
            results.append(Cite(id=_id, collection=cname, source=src, preview=preview, distance=float(dist or 0.0)))

    # sort by distance but keep image items slightly favored on ties
    results.sort(key=lambda c: (c.distance, 0 if "ti_images" in c.collection else 1))
    return results[:top_k]


def _try_direct_from_images(question: str, cites: List[Cite]) -> Optional[Tuple[str, List[Cite]]]:
    """Try regex/heuristic extraction from the actual OCR text before LLM."""
    for c in cites:
        if "ti_images" not in c.collection:
            continue
        try:
            col = APPCFG.chroma_client.get_collection(name=c.collection)
            got = col.get(ids=[c.id], include=["documents", "metadatas"])
            docs = got.get("documents") or []
            if docs and isinstance(docs[0], list):
                docs = docs[0]
            metas = got.get("metadatas") or []
            if metas and isinstance(metas[0], list):
                metas = metas[0]
            text = (docs[0] if docs else "") or c.preview
            meta = (metas[0] if metas else {}) or {}
            ans = _direct_answer_from_text(question, text, meta)
            if ans:
                # Return a minimal answer with the strongest image cite first
                return ans, [c]
        except Exception:
            continue
    return None

def _extract_used_cite_indices(answer: str, max_n: int) -> List[int]:
    """Return sorted unique indices (1-based) like [1], [3] that appear in answer and are <= max_n."""
    used = set()
    for m in re.finditer(r"\[(\d+)\]", answer or ""):
        try:
            n = int(m.group(1))
            if 1 <= n <= max_n:
                used.add(n)
        except Exception:
            pass
    return sorted(used)

def _dedupe_by_source(cites: List["Cite"]) -> List["Cite"]:
    """Keep first cite per distinct source path/URL."""
    seen = set()
    uniq = []
    for c in cites or []:
        key = c.source or c.id
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq

def answer_with_rag(question: str, top_k: int = 4) -> Tuple[str, List[Cite]]:
    # Pull a slightly bigger candidate set so we can dedupe & still have variety
    raw_cites = retrieve_top_k(question, top_k=top_k * 3)
    if not raw_cites:
        return "I couldn't find anything relevant in your indexed sources.", []

    # Build unique-by-source list for the prompt (clean numbering, no duplicates)
    cites = _dedupe_by_source(raw_cites)
    if not cites:
        return "I couldn't find anything relevant in your indexed sources.", []

    # Build the numbered context the model will cite against
    numbered = []
    for i, c in enumerate(cites, 1):
        numbered.append(f"[{i}] Source: {c.source}\nSnippet: {c.preview}\n")
    context = "\n\n".join(numbered)

    system = getattr(APPCFG, "rag_llm_system_role", "You are a concise assistant. Use ONLY the provided context.")
    user = (
        f"Question: {question}\n\n"
        f"Use ONLY this context:\n{context}\n\n"
        "Answer briefly. Cite minimally by including [n] only for the specific item numbers you actually used. "
        "If nothing in the context answers the question, reply exactly: I don't know."
    )

    # Back-compat: support either APPCFG.langchain_llm or APPCFG.rag_llm
    llm = getattr(APPCFG, "langchain_llm", None) or getattr(APPCFG, "rag_llm", None)
    if llm is None:
        # Avoid crashing if config changed; return context-only answer
        return f"(No LLM configured)\n\nTop cites:\n{context}", cites[:top_k]

    resp = llm.invoke([{"role": "system", "content": system},
                       {"role": "user", "content": user}])
    text = getattr(resp, "content", str(resp))

    # Keep only the sources the model actually referenced in its answer
    used_indices = _extract_used_cite_indices(text, len(cites))
    if used_indices:
        used_cites = [cites[i - 1] for i in used_indices]
    else:
        # Fallback: just show the single most relevant source
        used_cites = [cites[0]]

    # Final safety: dedupe again by source in case model cited multiple numbers from same source
    used_cites = _dedupe_by_source(used_cites)

    # Trim to at most top_k for display neatness
    return text, used_cites[:top_k]

# ---------------- Snippet helper for a specific source ----------------
def top_snippets_for_source(question: str, source: str, n: int = 3) -> List[str]:
    """
    Return up to n best-matching text chunks for a given `source` (exact metadata match)
    using the question embedding as the query, searching across all TI collections.
    """
    try:
        qvec = APPCFG.embeddings.embed_query(question)
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Embeddings backend is missing. If you use LangChain OpenAI embeddings, install: "
            "`pip install langchain-openai` and set your API key."
        ) from e

    candidates: List[Tuple[float, str]] = []
    collections = (
        APPCFG.TI_COLLECTIONS.get("pdfs", "ti_pdf_docs"),
        APPCFG.TI_COLLECTIONS.get("urls", "ti_web_urls"),
        APPCFG.TI_COLLECTIONS.get("images", "ti_images"),
        APPCFG.TI_COLLECTIONS.get("csv_xlsx_docs", "ti_tabular_docs"),
    )

    for cname in collections:
        try:
            col = APPCFG.chroma_client.get_collection(name=cname)
        except Exception:
            continue

        try:
            out = col.query(
                query_embeddings=[qvec],
                n_results=max(n, 5),                    # fetch a few to sort/filter
                where={"source": source},               # ONLY this resource
                include=["documents", "distances"],     # we want the text chunks
            )
        except Exception:
            continue

        docs = (out.get("documents") or [[]])[0]
        dists = (out.get("distances") or [[]])[0]
        for doc, dist in zip(docs, dists):
            if isinstance(doc, str) and doc.strip():
                candidates.append((float(dist), doc))

    # best first
    candidates.sort(key=lambda t: t[0])
    return [c[1] for c in candidates[:n]]


# ---------------- Legacy helper (kept, but made safe) ----------------
CHROMA = getattr(APPCFG, "chroma_client", None)

def _embed_texts(texts: List[str]) -> List[List[float]]:
    return APPCFG.embeddings.embed_documents(texts)

def _add_to_collection(
    collection_name: str,
    documents: List[str],
    metadatas: List[Dict],
    ids: List[str],
) -> int:
    """Upsert docs to a Chroma collection, skipping duplicates safely."""
    if not documents:
        return 0
    if CHROMA is None:
        return 0  # no client available; noop

    col = CHROMA.get_or_create_collection(name=collection_name)
    embs = _embed_texts(documents)

    # Determine which ids already exist to avoid "ID exists" errors
    try:
        existing = set(col.get(ids=ids).get("ids", []))
    except Exception:
        existing = set()

    new_docs, new_meta, new_ids, new_embs = [], [], [], []
    for i, id_ in enumerate(ids):
        if id_ in existing:
            continue
        new_docs.append(documents[i])
        new_meta.append(metadatas[i])
        new_ids.append(id_)
        new_embs.append(embs[i])

    if not new_ids:
        return 0  # nothing new

    if hasattr(col, "upsert"):
        col.upsert(
            ids=new_ids,
            documents=new_docs,
            metadatas=new_meta,
            embeddings=new_embs,
        )
    else:
        col.add(
            ids=new_ids,
            documents=new_docs,
            metadatas=new_meta,
            embeddings=new_embs,
        )
    return len(new_ids)