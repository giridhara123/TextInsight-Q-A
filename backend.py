from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import sqlite3
from datetime import datetime

app = FastAPI()

# Allow Streamlit (localhost:8501) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- SQLite setup ----
conn = sqlite3.connect("search_history.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS searches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        sources TEXT,
        timestamp TEXT NOT NULL
    )
"""
)
conn.commit()

class Search(BaseModel):
    question: str
    answer: str
    sources: str = ""
    timestamp: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ---- Routes ----
@app.post("/searches/")
async def create_search(search: Search):
    cursor.execute(
        "INSERT INTO searches (question, answer, sources, timestamp) VALUES (?, ?, ?, ?)",
        (search.question, search.answer, search.sources, search.timestamp)
    )
    conn.commit()
    return {"message": "Search added"}

@app.get("/searches/", response_model=List[Search])
async def get_searches():
    cursor.execute("SELECT question, answer, sources, timestamp FROM searches ORDER BY id DESC")
    rows = cursor.fetchall()
    return [Search(question=r[0], answer=r[1], sources=r[2], timestamp=r[3]) for r in rows]

@app.on_event("shutdown")
def shutdown_event():
    conn.close()