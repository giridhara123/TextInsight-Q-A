
# utils/prepare_sqlitedb_from_csv_xlsx.py
from __future__ import annotations
import os
import pandas as pd
from sqlalchemy import create_engine, inspect
from utils.load_config import APPCFG


class PrepareSQLFromTabularData:
    """Turn a directory of CSV/XLSX files into tables in the unified SQLite DB."""

    def __init__(self, files_dir: str) -> None:
        self.files_dir = files_dir
        self.engine = create_engine(APPCFG.sqlite_rw_uri, future=True)

    def _iter_files(self):
        for root, _dirs, files in os.walk(self.files_dir):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in {'.csv', '.xlsx', '.xls'}:
                    yield os.path.join(root, f)

    def _ingest(self, path: str):
        name = os.path.splitext(os.path.basename(path))[0]
        table = name.lower().replace(" ", "_")
        if path.lower().endswith('.csv'):
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)
        df.to_sql(table, self.engine, if_exists='replace', index=False)
        return table

    def _validate(self):
        insp = inspect(self.engine)
        tables = insp.get_table_names()
        print("==============================")
        print("SQLite DB:", APPCFG.sqlite_db_path)
        print("Tables:", tables)
        print("==============================")

    def run_pipeline(self):
        for p in self._iter_files():
            self._ingest(p)
        self._validate()
        self.engine.dispose()


if __name__ == "__main__":
    # Example runner; set your dataset directory here if needed.
    d = APPCFG.uploads_dir.as_posix()
    PrepareSQLFromTabularData(d).run_pipeline()
