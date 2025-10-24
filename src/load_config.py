
# utils/load_config.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import chromadb

# LangChain (OpenAI backend)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

try:
    from openai import OpenAI as _OpenAIClient
except Exception:
    _OpenAIClient = None


class LoadConfig:
    """Centralized, robust configuration for paths, DB URIs, models, and clients.
    This UNIFIES the SQLite path across the entire app (Fix #1).
    """

    def __init__(self) -> None:
        load_dotenv(override=True)

        # ---------- Project roots ----------
        self.project_root = Path(os.getenv("TEXTINSIGHT_HOME", Path.cwd())).resolve()

        # Data directories
        self.data_dir = (self.project_root / "data").resolve()
        self.sqlite_dir = (self.data_dir / "sqlite").resolve()
        self.uploads_dir = (self.data_dir / "uploads").resolve()

        # Vector store directory (Chroma)
        self.persist_directory = (self.project_root / "Vector_DB - Documents").resolve()

        # Ensure directories exist (Fix #9 related; harmless here)
        for d in [self.data_dir, self.sqlite_dir, self.uploads_dir, self.persist_directory]:
            d.mkdir(parents=True, exist_ok=True)

        # ---------- SQLite unified path & URIs ----------
        self.sqlite_db_name = os.getenv("TI_SQLITE_DB_NAME", "textinsight.db")
        self.sqlite_db_path = (self.sqlite_dir / self.sqlite_db_name).resolve()

        # SQLAlchemy RW URI (used for ingestion)
        # sqlite:////ABS/PATH.db
        self.sqlite_rw_uri = f"sqlite:///{self.sqlite_db_path.as_posix()}"

        # SQLite read-only URI (sqlite3 or pandas read_sql with uri=True)
        # file:/ABS/PATH.db?mode=ro
        self.sqlite_ro_uri = f"file:{self.sqlite_db_path.as_posix()}?mode=ro"

        # ---------- OpenAI (or compatible) ----------
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        # Lightweight fail-fast check (Fix #1 ergonomics): only warn, don't crash at import
        self.use_openai = bool(self.openai_api_key)

        # Models
        self.embedding_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
        self.chat_model = os.getenv("LLM_MODEL", "gpt-4o-mini")

        # Initialize clients lazily to avoid startup crashes in UIs that just render
        self._openai_client = None
        self._embedding = None
        self._rag_llm = None

        # ---------- Chroma client (pinned 0.4.24) ----------
        # Create a persistent client at a known location to avoid tenant issues.
        self.chroma_client = chromadb.PersistentClient(path=str(self.persist_directory))

        # ---------- Prompts ----------
        self.rag_llm_system_role = (
            "You are a concise assistant. Answer ONLY using the provided context. "
            "If the context is insufficient, say you do not know."
        )

    # --- Lazy factories -----------------------------------------------------
    @property
    def openai_client(self):
        if self._openai_client is None and _OpenAIClient is not None and self.use_openai:
            self._openai_client = _OpenAIClient(api_key=self.openai_api_key)
        return self._openai_client

    @property
    def embedding(self):
        if self._embedding is None:
            self._embedding = OpenAIEmbeddings(model=self.embedding_model)
        return self._embedding

    @property
    def rag_llm(self):
        if self._rag_llm is None:
            self._rag_llm = ChatOpenAI(model=self.chat_model, temperature=0.2, streaming=True)
        return self._rag_llm


# Singleton-like instance
APPCFG = LoadConfig()
