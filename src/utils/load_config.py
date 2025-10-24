# src/utils/load_config.py
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Optional: keep imports here so properties can instantiate lazily
# (we import inside properties to avoid import-time failures if deps aren't installed)

class LoadConfig:
    """
    Central configuration + lazy clients.

    Exposes:
      APPCFG.sqlite_db_path (Path)
      APPCFG.uploaded_files_sqldb_path (Path, alias)
      APPCFG.uploads_dir / sqlite_dir / persist_directory (Path)
      APPCFG.PERSIST_DIR (str alias)  + property for back-compat
      APPCFG.TI_COLLECTIONS (dict of collection names)
      APPCFG.embeddings  (langchain_openai.OpenAIEmbeddings)
      APPCFG.rag_llm / APPCFG.langchain_llm  (langchain_openai.ChatOpenAI)
      APPCFG.chroma_client  (chromadb PersistentClient)
      APPCFG.rag_llm_system_role (str)
    """

    def __init__(self) -> None:
        load_dotenv(override=True)

        # ---- paths ----
        # Use TEXTINSIGHT_HOME if set; otherwise anchor to repo's /src (parent of this file)
        default_root = Path(__file__).resolve().parents[1]  # .../src
        self.project_root = Path(os.getenv("TEXTINSIGHT_HOME", default_root)).resolve()

        self.data_dir = (self.project_root / "data").resolve()
        self.sqlite_dir = (self.data_dir / "sqlite").resolve()
        self.uploads_dir = (self.data_dir / "uploads").resolve()
        # Keep your existing Chroma persist directory name
        self.persist_directory = (self.project_root / "Vector_DB - Documents").resolve()

        for d in (self.data_dir, self.sqlite_dir, self.uploads_dir, self.persist_directory):
            d.mkdir(parents=True, exist_ok=True)

        # Some code uses a string alias; provide both
        self.PERSIST_DIR = self.persist_directory.as_posix()

        # ---- SQLite DB ----
        self._sqlite_db_name = os.getenv("TI_SQLITE_DB_NAME", "textinsight.db")
        self._sqlite_db_path = (self.sqlite_dir / self._sqlite_db_name).resolve()
        self.sqlite_rw_uri = f"sqlite:///{self._sqlite_db_path.as_posix()}"
        self.sqlite_ro_uri = f"file:{self._sqlite_db_path.as_posix()}?mode=ro"

        # ---- models & API keys ----
        self._openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.embedding_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
        self.chat_model = os.getenv("LLM_MODEL", "gpt-4o-mini")

        # ---- lazy singletons ----
        self._embeddings = None
        self._rag_llm = None
        self._chroma = None

        # ---- collection names used across your app ----
        # (keep these stable; existing indexes will continue to work)
        self.TI_COLLECTIONS = {
            "urls": "ti_urls",
            "pdfs": "ti_pdfs",
            "images": "ti_images",
            "csv_xlsx_docs": "ti_tabular_docs",
        }

        # System prompt for RAG
        self._rag_system = (
            "You are a concise assistant. Use ONLY the provided context. "
            "If the context is insufficient, say you do not know and ask for details."
        )

    # ---------- path properties ----------
    @property
    def sqlite_db_path(self) -> Path:
        return self._sqlite_db_path

    # Alias used by utils/chatbot.py
    @property
    def uploaded_files_sqldb_path(self) -> Path:
        return self.sqlite_db_path

    # Some older code accesses PERSIST_DIR as a property
    @property
    def PERSIST_DIR_PROP(self) -> str:
        return self.PERSIST_DIR

    # ---------- API key ----------
    @property
    def openai_api_key(self) -> str:
        return self._openai_api_key

    # ---------- clients (lazy) ----------
    @property
    def embeddings(self):
        if self._embeddings is None:
            from langchain_openai import OpenAIEmbeddings
            self._embeddings = OpenAIEmbeddings(model=self.embedding_model)
        return self._embeddings

    @property
    def rag_llm(self):
        if self._rag_llm is None:
            from langchain_openai import ChatOpenAI
            self._rag_llm = ChatOpenAI(model=self.chat_model, temperature=0.2, streaming=True)
        return self._rag_llm

    # Some modules expect this name
    @property
    def langchain_llm(self):
        return self.rag_llm

    @property
    def chroma_client(self):
        if self._chroma is None:
            import chromadb
            # Chroma 0.4.x persistent client
            self._chroma = chromadb.PersistentClient(path=self.PERSIST_DIR)
        return self._chroma

    # ---------- prompts ----------
    @property
    def rag_llm_system_role(self) -> str:
        return self._rag_system


# Export the singleton expected by: from utils.load_config import APPCFG
APPCFG = LoadConfig()