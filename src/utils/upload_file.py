# utils/upload_file.py
from __future__ import annotations
import os, re, hashlib, io, datetime as dt
from typing import List, Tuple

import pandas as pd
from sqlalchemy import create_engine, text, inspect
from pathlib import Path

# CHANGED: import image ingestion + extensions from textinsight
from utils.textinsight import ingest_images, IMAGE_EXTS

from utils.load_config import APPCFG


# -------- Helpers ------------------------------------------------------------

def _table_name_from_file(path: str) -> str:
    base = os.path.basename(path)
    name, _ext = os.path.splitext(base)
    # normalize to sqlite-friendly table name
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_").lower()
    return name or "table_1"

def _file_hash(path_or_bytes) -> str:
    # accepts a file path OR raw bytes (Streamlit UploadedFile.getvalue())
    h = hashlib.sha256()
    if isinstance(path_or_bytes, (bytes, bytearray)):
        b = io.BytesIO(path_or_bytes)
        while True:
            chunk = b.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    else:
        with open(path_or_bytes, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()

def _short_db_path(path: str, keep_dirs: int = 2) -> str:
    # returns …/<tail dirs>
    parts = os.path.abspath(path).split(os.sep)
    return f"…{os.sep}{os.path.join(*parts[-keep_dirs-1:])}" if len(parts) > keep_dirs+1 else path


# -------- Ingestion with manifest -------------------------------------------

class ProcessFiles:
    """
    CSV/XLSX ingestion into the unified SQLite database with a manifest that tracks:
      - file_path (basename)
      - table_name
      - file_hash (sha256)
      - row_count
      - updated_at
    This gives precise per-file messages and allows "already processed" reporting.
    """

    def __init__(self, files_dir: List[str], chatbot: List):
        self.files_dir = files_dir
        self.chatbot = chatbot
        self.engine = create_engine(APPCFG.sqlite_rw_uri, future=True)

        # Create manifest table if not exists
        with self.engine.begin() as con:
            con.execute(text("""
                CREATE TABLE IF NOT EXISTS meta_ingest (
                  file_path TEXT PRIMARY KEY,
                  table_name TEXT NOT NULL,
                  file_hash TEXT NOT NULL,
                  row_count INTEGER NOT NULL,
                  updated_at TEXT NOT NULL
                )
            """))

    def _append_msg(self, text: str):
        # push into your chat log / UI; adapt if your app uses a different structure
        self.chatbot.append(("assistant", text))

    def _read_dataframe(self, file_path: str) -> pd.DataFrame:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".csv":
            return pd.read_csv(file_path)
        elif ext in {".xlsx", ".xls"}:
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _get_manifest(self, basename: str):
        with self.engine.begin() as con:
            row = con.execute(
                text("SELECT file_path, table_name, file_hash, row_count FROM meta_ingest WHERE file_path = :p"),
                {"p": basename}
            ).mappings().first()
            return dict(row) if row else None

    def _upsert_manifest(self, basename: str, table: str, file_hash: str, row_count: int):
        now = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
        with self.engine.begin() as con:
            con.execute(
                text("""
                    INSERT INTO meta_ingest (file_path, table_name, file_hash, row_count, updated_at)
                    VALUES (:p, :t, :h, :r, :u)
                    ON CONFLICT(file_path) DO UPDATE SET
                      table_name=excluded.table_name,
                      file_hash=excluded.file_hash,
                      row_count=excluded.row_count,
                      updated_at=excluded.updated_at
                """),
                {"p": basename, "t": table, "h": file_hash, "r": row_count, "u": now}
            )

    def _ingest_one(self, file_path: str) -> str:
        basename = os.path.basename(file_path)
        table = _table_name_from_file(file_path)
        digest = _file_hash(file_path)

        # Check manifest (already processed?)
        previous = self._get_manifest(basename)

        if previous and previous.get("file_hash") == digest:
            # Already processed and unchanged
            msg = (
                f"Already processed: **{basename}** — rows: {previous['row_count']} — "
                f"DB: {_short_db_path(APPCFG.sqlite_db_path.as_posix())} — table: `{previous['table_name']}`"
            )
            self._append_msg(msg)
            return msg

        # (Re)ingest
        df = self._read_dataframe(file_path)
        rows = int(len(df))

        # Replace the table (idempotent)
        df.to_sql(table, self.engine, if_exists="replace", index=False)

        # Update manifest
        self._upsert_manifest(basename, table, digest, rows)

        # Message: changed vs new
        status = "Re-ingested (changed)" if previous else "Processed"
        msg = (
            f"{status}: **{basename}** — rows: {rows} — "
            f"DB: {_short_db_path(APPCFG.sqlite_db_path.as_posix())} — table: `{table}`"
        )
        self._append_msg(msg)
        return msg

    def _post_summary(self) -> str:
        insp = inspect(self.engine)
        tables = insp.get_table_names()
        return (
            f"SQLite: {_short_db_path(APPCFG.sqlite_db_path.as_posix())} — "
            f"Tables: {', '.join(tables) if tables else '(none)'}"
        )

    def run(self) -> Tuple[str, List]:
        if not self.files_dir:
            summary = "No files provided."
            self._append_msg(summary)
            self.engine.dispose()
            return summary, self.chatbot

        # NEW: split inputs by type to avoid CSV handler trying to open images
        csv_xlsx_paths = [p for p in self.files_dir if Path(p).suffix.lower() in {".csv", ".xlsx", ".xls"}]
        image_paths = [p for p in self.files_dir if Path(p).suffix.lower() in IMAGE_EXTS]

        results = []

        # Ingest CSV/XLSX → SQLite (existing behavior)
        for path in csv_xlsx_paths:
            try:
                results.append(self._ingest_one(path))
            except Exception as e:
                self._append_msg(f"Failed to ingest {os.path.basename(path)}: {e}")

        # NEW: OCR + index images to Chroma (TextInsight)
        if image_paths:
            ocr_res = ingest_images(image_paths)
            self._append_msg(
                f"[ti_images] indexed {ocr_res.get('added', 0)} text chunk(s) from {len(image_paths)} image file(s)."
            )
            for f in ocr_res.get("files", []):
                self._append_msg(f"  - {f.get('name','?')}: {f.get('chunks',0)} chunk(s)")

        summary = self._post_summary()
        self._append_msg(summary)
        self.engine.dispose()
        return summary, self.chatbot


class UploadFile:
    """Controller for upload pipelines; extend here if you add more modes."""
    @staticmethod
    def run_pipeline(files_dir: List[str], chatbot: List, chatbot_functionality: str) -> Tuple[str, List]:
        if chatbot_functionality == "Process files":
            pipeline = ProcessFiles(files_dir=files_dir, chatbot=chatbot)
            return pipeline.run()
        # fallback no-op
        return "", chatbot