# src/utils/prepare_images_vectordb.py
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Iterable, List

from PIL import Image
import pytesseract

# Prefer the new package; fall back if project is older
try:
    from langchain_openai import OpenAIEmbeddings
except Exception:  # pragma: no cover
    from langchain.embeddings import OpenAIEmbeddings  # type: ignore

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from utils.load_config import APPCFG  # your existing config

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _ocr_text(path: str) -> str:
    img = Image.open(path)
    raw = pytesseract.image_to_string(img) or ""
    # collapse whitespace; helps retrieval + regex
    return " ".join(raw.split())


def _chunk(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    return splitter.split_text(text)


def ingest_image_to_chroma(path: str, collection: str = "ti_images") -> int:
    """
    OCR a single image file, chunk the text, and upsert into Chroma.
    Returns number of chunks added (0 if no text).
    """
    text = _ocr_text(path)
    if not text:
        return 0

    chunks = _chunk(text)
    base_meta = {
        "source": str(Path(path).resolve()),
        "type": "image",
        "ocr_engine": "tesseract",
    }
    docid = _sha1(base_meta["source"] + "|" + text)

    docs = [
        Document(page_content=chunk, metadata=base_meta | {"chunk": i, "doc_id": docid})
        for i, chunk in enumerate(chunks)
    ]
    ids = [f"{docid}:{i}" for i in range(len(docs))]

    vs = Chroma(
        collection_name=collection,
        persist_directory=APPCFG.chroma_persist_dir,
        embedding_function=OpenAIEmbeddings(),
    )
    vs.add_documents(docs, ids=ids)
    vs.persist()
    return len(docs)


def ingest_images(paths: Iterable[str], collection: str = "ti_images") -> int:
    """
    Convenience: ingest multiple images. Returns total chunks added.
    """
    total = 0
    for p in paths:
        if Path(p).suffix.lower() in IMAGE_EXTS:
            total += ingest_image_to_chroma(p, collection=collection)
    return total