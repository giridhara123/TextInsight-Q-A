# src/utils/chatbot.py
from __future__ import annotations

import os
from typing import List, Tuple

from sqlalchemy import create_engine, text
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

from utils.load_config import APPCFG
from utils.textinsight import answer_with_rag


def _looks_like_sql(q: str) -> bool:
    ql = (q or "").strip().lower()
    return ql.startswith(("select", "with", "pragma", "explain", "update", "insert", "delete"))


def _format_rows(headers, rows, max_rows: int = 25) -> str:
    """Pretty print first rows as a simple table-like text."""
    if not rows:
        return "(no rows)"
    hdr = " | ".join(headers)
    sep = "-+-".join("-" * len(h) for h in headers)
    body = []
    for r in rows[:max_rows]:
        # r can be RowMapping/Row -> cast to list of values
        body.append(" | ".join(str(v) for v in list(r)))
    return "\n".join([hdr, sep] + body)


class ChatBot:
    """
    Two chat types:
      - "Q&A with Uploaded CSV/XLSX SQL-DB" : Ask SQL directly or natural language via LLM agent
      - "RAG with source uploads"           : Grounded QA over PDFs/URLs/Images/CSV-text (Chroma)
    """

    @staticmethod
    def respond(chatbot: List, message: str, chat_type: str, app_functionality: str) -> Tuple:
        # Only handle when user is in Chat mode
        if app_functionality != "Chat":
            return "", chatbot

        # ---------- SQL path (uploaded CSV/XLSX -> SQLite) ----------
        if chat_type == "Q&A with Uploaded CSV/XLSX SQL-DB":
            db_path = getattr(APPCFG, "uploaded_files_sqldb_path", None) or APPCFG.sqlite_db_path
            if not os.path.exists(db_path):
                chatbot.append(
                    (message, f"SQLite DB not found yet at {db_path}. Upload CSV/XLSX and click Process.")
                )
                return "", chatbot

            engine = create_engine(f"sqlite:///{db_path}")

            # If the user typed raw SQL, run it directly
            if _looks_like_sql(message):
                try:
                    with engine.connect() as conn:
                        result = conn.execute(text(message))
                        headers = list(result.keys())
                        rows = result.fetchmany(50)  # limit to keep output tidy
                        resp = _format_rows(headers, rows)
                except Exception as e:
                    resp = f"SQL error: {e}"
                finally:
                    # release db handle so further ingests/uploads won't lock
                    engine.dispose()

                chatbot.append((message, resp))
                return "", chatbot

            # Otherwise, use the LLM SQL agent (LangChain)
            try:
                db = SQLDatabase(engine=engine)
                agent = create_sql_agent(
                    APPCFG.langchain_llm,
                    db=db,
                    agent_type="openai-tools",
                    verbose=True
                )
                result = agent.invoke({"input": message})
                resp = result.get("output", str(result))
            except Exception as e:
                resp = f"Agent error: {e}"
            finally:
                engine.dispose()

            chatbot.append((message, resp))
            return "", chatbot

        # ---------- RAG path (TextInsight) ----------
        elif chat_type == "RAG with source uploads":
            try:
                answer, cites = answer_with_rag(message, top_k=4)
                if cites:
                    tail = "\n\nSources:\n" + "\n".join(
                        f"[{i}] {c.source} ({c.collection})" for i, c in enumerate(cites, 1)
                    )
                else:
                    tail = ""
                resp = answer + tail
            except Exception as e:
                resp = f"RAG error: {e}"

            chatbot.append((message, resp))
            return "", chatbot

        # ---------- Unknown chat type ----------
        chatbot.append((message, "Unsupported chat type."))
        return "", chatbot