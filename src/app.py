from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import gradio as gr
import sqlite3
import traceback, sys
import re
from datetime import datetime
import warnings

# ---------- BEGIN: Robust monkey patches for old gradio_client schema bug ----------
try:
    import gradio_client.utils as _gc_utils

    _orig_get_type = getattr(_gc_utils, "get_type", None)
    _orig_json_to_py = getattr(_gc_utils, "_json_schema_to_python_type", None)
    _orig_json_schema_to_python_type = getattr(_gc_utils, "json_schema_to_python_type", None)

    def _safe_any_from_schema(schema):
        # Normalize any "weird" schema to a permissive type string.
        return "Any"

    def _patched_get_type(schema):
        # Some schemas are literally True/False because of JSON Schema's additionalProperties flag
        if isinstance(schema, bool) or schema is None:
            return _safe_any_from_schema(schema)
        return _orig_get_type(schema) if _orig_get_type else _safe_any_from_schema(schema)

    def _patched__json_schema_to_python_type(schema, defs):
        # Intercept booleans and Nones anywhere in the recursive descent.
        if isinstance(schema, bool) or schema is None:
            return _safe_any_from_schema(schema)
        try:
            return _orig_json_to_py(schema, defs) if _orig_json_to_py else _safe_any_from_schema(schema)
        except Exception:
            # Last-ditch fallback so UI still renders
            return _safe_any_from_schema(schema)

    def _patched_json_schema_to_python_type(schema):
        if isinstance(schema, bool) or schema is None:
            return _safe_any_from_schema(schema)
        try:
            # Call through to (possibly) patched private helper
            return (_orig_json_schema_to_python_type(schema)
                    if _orig_json_schema_to_python_type else _safe_any_from_schema(schema))
        except Exception:
            return _safe_any_from_schema(schema)

    # Apply patches
    if _orig_get_type:
        _gc_utils.get_type = _patched_get_type
    if _orig_json_to_py:
        _gc_utils._json_schema_to_python_type = _patched__json_schema_to_python_type
    if _orig_json_schema_to_python_type:
        _gc_utils.json_schema_to_python_type = _patched_json_schema_to_python_type

except Exception:
    # If these patches can't be applied, continue; we also patch Gradio entry points below.
    pass

# Defensive: also wrap Gradio's API info paths so a parse error never crashes startup.
try:
    import gradio.blocks as _gr_blocks
    _orig_get_api_info = getattr(_gr_blocks.Blocks, "get_api_info", None)

    if _orig_get_api_info:
        def _patched_get_api_info(self):
            try:
                return _orig_get_api_info(self)
            except Exception:
                # Returning empty metadata is fine for regular UI use.
                return {}
        _gr_blocks.Blocks.get_api_info = _patched_get_api_info
except Exception:
    pass

try:
    import gradio.routes as _gr_routes
    _orig_api_info = getattr(_gr_routes, "api_info", None)

    if _orig_api_info:
        def _patched_api_info(include_examples: bool = True):
            try:
                return _orig_api_info(include_examples)
            except Exception:
                return {}
        _gr_routes.api_info = _patched_api_info
except Exception:
    pass
# ---------- END: monkey patches ----------

from utils.load_config import APPCFG  # ensures config loads
from utils.upload_file import UploadFile
from utils.chatbot import ChatBot
from utils.textinsight import top_snippets_for_source

warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

# TextInsight ingestors
from utils.textinsight import (
    ingest_pdfs, ingest_urls, ingest_images, ingest_csv_xlsx_to_chroma
)

# ------------------------
# Path + save helpers
# ------------------------
def _ensure_dirs():
    """Create required directories (idempotent)."""
    for d in (APPCFG.uploads_dir, APPCFG.sqlite_dir, APPCFG.persist_directory):
        Path(d).mkdir(parents=True, exist_ok=True)

def _save_to_uploads(files: Optional[List]) -> List[str]:
    """
    Save uploaded files (Gradio file objects or plain paths) into data/uploads/.
    Returns absolute file paths.
    """
    saved: List[str] = []
    if not files:
        return saved

    _ensure_dirs()

    for f in files:
        # Case 1: already a path on disk
        if isinstance(f, (str, os.PathLike)) and os.path.exists(str(f)):
            src = str(f)
            dst = Path(APPCFG.uploads_dir) / os.path.basename(src)
            if str(dst) != src:
                os.makedirs(dst.parent, exist_ok=True)
                shutil.copy2(src, dst)
            saved.append(str(dst))
            continue

        # Case 2: object with .name pointing to a temp path created by Gradio
        if hasattr(f, "name") and isinstance(f.name, str) and os.path.exists(f.name):
            src = f.name
            dst = Path(APPCFG.uploads_dir) / os.path.basename(src)
            if str(dst) != src:
                os.makedirs(dst.parent, exist_ok=True)
                shutil.copy2(src, dst)
            saved.append(str(dst))
            continue

        # Case 3: bytes-like
        base = os.path.basename(getattr(f, "name", "upload.bin")) or f"upload_{len(saved)+1}.bin"
        dst = Path(APPCFG.uploads_dir) / base
        os.makedirs(dst.parent, exist_ok=True)
        if hasattr(f, "read"):
            with open(dst, "wb") as w:
                w.write(f.read())
        elif hasattr(f, "getvalue"):
            with open(dst, "wb") as w:
                w.write(f.getvalue())
        else:
            with open(dst, "w", encoding="utf-8") as w:
                w.write(str(f))
        saved.append(str(dst))

    return saved

def _count_added(ret) -> int:
    """Handle dict/int returns from various ingest_* functions."""
    if isinstance(ret, dict):
        for k in ("added", "rows", "count", "n"):
            if k in ret:
                try:
                    return int(ret[k])
                except Exception:
                    pass
        try:
            return sum(int(v) for v in ret.values() if isinstance(v, (int, float)))
        except Exception:
            return 0
    if isinstance(ret, (int, float)):
        return int(ret)
    return 0

# ------------------------
# DB readiness check
# ------------------------
def _db_has_tables() -> bool:
    """Return True if SQLite has at least one user table."""
    try:
        uri = f"file:{APPCFG.sqlite_db_path.as_posix()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as conn:
            cur = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' LIMIT 1"
            )
            return cur.fetchone() is not None
    except Exception:
        return False

# ------------------------
# Snippets parsing + cache builders
# ------------------------
def _parse_sources_from_answer(answer_text: str) -> List[Tuple[int, str]]:
    """
    From the bot's message text, parse the 'Sources:' block and return [(index, source_str), ...].
    Assumes lines like: [1] /path/or/url (ti_images)
    """
    if not answer_text:
        return []
    m = re.search(r"(?im)^Sources:\s*$", answer_text)
    if not m:
        return []
    start = m.end()
    tail = answer_text[start:]
    lines = tail.strip().splitlines()
    out: List[Tuple[int, str]] = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if ln.lower().startswith("message") or ln.lower().startswith("type your question"):
            break
        mm = re.match(r"^\[(\d+)\]\s+(.+)$", ln)
        if not mm:
            mm = re.match(r"^[\-\*]?\s*\[(\d+)\]\s+(.+)$", ln)
        if mm:
            idx = int(mm.group(1))
            src = mm.group(2).strip()
            src = re.sub(r"\s*\(ti_[^)]+\)\s*$", "", src).strip()
            out.append((idx, src))
        else:
            break
    return out

def _build_snippet_cache_for_last_answer(
    user_question: str, bot_text: str, max_buttons: int = 4
) -> Tuple[Dict[int, Dict[int, str]], Dict[int, int]]:
    """
    Returns:
      - cache: {message_index_placeholder(0): {source_index: snippet_md}}
      - slot_map: {slot_no(1..max_buttons): source_index}
    """
    pairs = _parse_sources_from_answer(bot_text)
    pairs = pairs[:max_buttons]
    cache_entry: Dict[int, str] = {}
    slot_map: Dict[int, int] = {}
    for slot, (idx, src) in enumerate(pairs, 1):
        try:
            chunks = top_snippets_for_source(user_question, src, n=3)
        except Exception as e:
            chunks = [f"(Failed to fetch snippets for this source: {e})"]
        content = [f"**Source [{idx}]**  \n`{src}`", ""]
        for j, ch in enumerate(chunks, 1):
            content.append(f"**Chunk {j}**")
            content.append(ch.strip())
            content.append("---")
        snippet_md = "\n".join(content).rstrip("-")
        cache_entry[idx] = snippet_md
        slot_map[slot] = idx
    return ({0: cache_entry}, slot_map)

# --- Chat handlers (tuples style chatbot) ---
def handle_send(message, chat, chat_type, app_functionality,
                snip_cache_state, slot_map_state, last_idx_state):
    """
    Calls ChatBot.respond(...) which returns ("", updated_chat).
    We return updates for chat, msg, snippet buttons, and snippet viewer.
    """
    if not message or not message.strip():
        return (
            chat, "",
            gr.update(visible=False), gr.update(visible=False),
            gr.update(visible=False), gr.update(visible=False),
            gr.update(value="", visible=False),
            snip_cache_state, slot_map_state, last_idx_state
        )

    try:
        # Guard 1: API key presence
        if not getattr(APPCFG, "openai_api_key", ""):
            note = "⚠️ OPENAI_API_KEY is not set. Add it to your .env and restart."
            updated_chat = chat + [(message, note)]
            return (
                updated_chat, "",
                gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=False), gr.update(visible=False),
                gr.update(value="", visible=False),
                snip_cache_state, slot_map_state, last_idx_state
            )

        # Guard 2: SQL mode requires a ready DB
        if chat_type == "Q&A with Uploaded CSV/XLSX SQL-DB":
            if not APPCFG.sqlite_db_path.exists() or not _db_has_tables():
                note = (
                    f"⚠️ No SQLite data found at:\n{APPCFG.sqlite_db_path}\n\n"
                    "Please ingest CSV/XLSX via **Process sources** first."
                )
                updated_chat = chat + [(message, note)]
                return (
                    updated_chat, "",
                    gr.update(visible=False), gr.update(visible=False),
                    gr.update(visible=False), gr.update(visible=False),
                    gr.update(value="", visible=False),
                    snip_cache_state, slot_map_state, last_idx_state
                )

        # Normal path
        _, updated_chat = ChatBot.respond(
            chatbot=chat,
            message=message,
            chat_type=chat_type,
            app_functionality=app_functionality
        )

        # --- Build snippet cache for the LAST bot message only ---
        last_idx = len(updated_chat) - 1
        last_bot_text = ""
        if last_idx >= 0:
            try:
                last_bot_text = updated_chat[last_idx][1] or ""
            except Exception:
                last_bot_text = ""

        cache_stub, slot_map = _build_snippet_cache_for_last_answer(message, last_bot_text, max_buttons=4)
        new_cache = dict(snip_cache_state or {})
        if cache_stub.get(0):
            new_cache[last_idx] = cache_stub[0]

        # Prepare button updates
        btn_updates = [gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)]
        for slot in range(1, 5):
            if slot in slot_map:
                idx = slot_map[slot]
                btn_updates[slot-1] = gr.update(value=f"Snippet [{idx}]", visible=True)

        # Clear snippet view (closed by default)
        snippet_update = gr.update(value="", visible=False)

        return (
            updated_chat, "",
            btn_updates[0], btn_updates[1], btn_updates[2], btn_updates[3],
            snippet_update,
            new_cache, slot_map, last_idx
        )

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        updated_chat = chat + [(message, f"⚠️ Error while answering: {e}")]
        return (
            updated_chat, "",
            gr.update(visible=False), gr.update(visible=False),
            gr.update(visible=False), gr.update(visible=False),
            gr.update(value="", visible=False),
            snip_cache_state, slot_map_state, last_idx_state
        )

def handle_clear():
    """(legacy) Keep for reference; not used now."""
    return [], ""

# ------------------------
# Ingestion helpers
# ------------------------
def _split_common_files(files):
    """Direct the common uploader files to the right ingestor buckets."""
    pdfs, imgs, tabular = [], [], []
    for f in files or []:
        path = f.name if hasattr(f, "name") else str(f)
        ext = os.path.splitext(path)[1].lower()
        if ext in (".pdf",):
            pdfs.append(f)
        elif ext in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
            imgs.append(f)
        elif ext in (".csv", ".xlsx"):
            tabular.append(f)
    return pdfs, imgs, tabular

def process_sources_ui(common_files, pdf_files, image_files, urls_text, csv_xlsx_files):
    """
    Process everything the user provided. Never return early; aggregate successes & errors.
    Also clear the widgets after each run by returning updates for them.
    """
    # Totals we show at the top
    added_urls = added_pdfs = added_imgs = added_tab = 0
    # Full results for per-file lines
    res_urls = res_pdfs = res_imgs = res_sem = None

    errors = []
    sqlite_summary = ""  # capture SQLite path + tables

    # Route common files
    try:
        if common_files:
            cpdfs, cimgs, ctab = _split_common_files(common_files)
            pdf_files = (pdf_files or []) + cpdfs
            image_files = (image_files or []) + cimgs
            csv_xlsx_files = (csv_xlsx_files or []) + ctab
    except Exception as e:
        errors.append(f"Common upload routing failed: {e}")

    # URLs
    try:
        urls = []
        if urls_text:
            raw = [u.strip() for u in urls_text.replace(",", "\n").splitlines()]
            urls = [u for u in raw if u]
        if urls:
            res_urls = ingest_urls(urls)
            added_urls = _count_added(res_urls)  # total URL chunks
    except Exception as e:
        errors.append(f"URL ingest error: {e}")

    # PDFs  (save to uploads first)
    try:
        pdf_paths = _save_to_uploads(pdf_files)
        if pdf_paths:
            res_pdfs = ingest_pdfs(pdf_paths)
            added_pdfs = _count_added(res_pdfs)  # total PDF chunks
    except Exception as e:
        errors.append(f"PDF ingest error: {e}")

    # Images  (save to uploads first)
    try:
        img_paths = _save_to_uploads(image_files)
        if img_paths:
            res_imgs = ingest_images(img_paths)
            added_imgs = _count_added(res_imgs)  # total image 'chunks' (labels)
    except Exception as e:
        errors.append(f"Image ingest error: {e}")

    # CSV/XLSX -> Chroma + SQLite
    try:
        csv_paths = _save_to_uploads(csv_xlsx_files)
        if csv_paths:
            res_sem = ingest_csv_xlsx_to_chroma(csv_paths)
            added_tab = _count_added(res_sem)  # total rows across files

            # Also into SQLite for SQL Q&A (and get DB summary string)
            try:
                sqlite_summary, _ = UploadFile.run_pipeline(
                    files_dir=csv_paths,
                    chatbot=[],
                    chatbot_functionality="Process files"
                )
            except Exception as sub_e:
                errors.append(f"SQLite ingest error (CSV/XLSX): {sub_e}")
    except Exception as e:
        errors.append(f"Chroma ingest error (CSV/XLSX): {e}")

    # -------- Build status message (files + chunks/rows) --------
    url_file_count = len(res_urls.get("files", [])) if isinstance(res_urls, dict) else 0
    pdf_file_count = len(res_pdfs.get("files", [])) if isinstance(res_pdfs, dict) else 0
    img_file_count = len(res_imgs.get("files", [])) if isinstance(res_imgs, dict) else 0
    csv_file_count = len(res_sem.get("files", [])) if isinstance(res_sem, dict) else 0

    csv_total_chunks = 0
    if isinstance(res_sem, dict) and res_sem.get("files"):
        try:
            csv_total_chunks = sum(int(f.get("chunks", 0)) for f in res_sem["files"])
        except Exception:
            csv_total_chunks = 0

    header = (
        f"Processed — "
        f"URLs: {url_file_count} file(s), {added_urls} chunks; "
        f"PDFs: {pdf_file_count} file(s), {added_pdfs} chunks; "
        f"Images: {img_file_count} file(s), {added_imgs} chunks; "
        f"CSV/XLSX: {csv_file_count} file(s), {added_tab} rows"
        + (f", {csv_total_chunks} chunks" if csv_total_chunks else "")
        + "."
    )
    lines = [header]

    if isinstance(res_urls, dict) and res_urls.get("files"):
        for f in res_urls["files"]:
            lines.append(f"- **URL**: {f.get('name','?')} — {f.get('chunks',0)} chunks")

    if isinstance(res_pdfs, dict) and res_pdfs.get("files"):
        for f in res_pdfs["files"]:
            lines.append(f"- **PDF**: {f.get('name','?')} — {f.get('chunks',0)} chunks")

    if isinstance(res_imgs, dict) and res_imgs.get("files"):
        for f in res_imgs["files"]:
            lines.append(f"- **Image**: {f.get('name','?')} — {f.get('chunks',0)} chunks")

    if isinstance(res_sem, dict) and res_sem.get("files"):
        for f in res_sem["files"]:
            lines.append(
                f"- **TABLE**: {f.get('name','?')} — {f.get('rows',0)} rows, {f.get('chunks',0)} chunks"
            )

    status = "\n".join(lines)

    if sqlite_summary:
        status += f"\n\n{sqlite_summary}"

    if errors:
        status += f"\n\n**Warnings/Errors**:\n- " + "\n- ".join(errors)

    return (
        status,
        None,  # common_files
        None,  # pdf_files
        None,  # image_files
        "",    # urls_text
        None,  # csv_xlsx_files
    )

def _build_status_markdown() -> str:
    import sqlite3
    try:
        uri = f"file:{APPCFG.sqlite_db_path.as_posix()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
            tables = [r[0] for r in rows]
    except Exception as e:
        tables = [f"(error: {e})"]

    client = APPCFG.chroma_client
    coll_lines = []
    totals = {"chunks": 0, "sources": 0}

    def _stats_for_collection(cname: str, is_tabular: bool = False):
        try:
            col = client.get_collection(name=cname)
        except Exception:
            return 0, 0, 0

        try:
            chunks = int(col.count())
        except Exception:
            chunks = 0

        sources = 0
        total_rows = 0
        try:
            rec = col.get(include=["metadatas"], limit=100000)
            metas = rec.get("metadatas") or []
            src_set = set()
            rows_by_src = {}
            for m in metas:
                if not m:
                    continue
                src = m.get("source")
                if src:
                    src_set.add(src)
                if is_tabular and src is not None:
                    r = m.get("rows")
                    if r is not None and src not in rows_by_src:
                        try:
                            rows_by_src[src] = int(r)
                        except Exception:
                            pass
            sources = len(src_set)
            if is_tabular:
                total_rows = sum(rows_by_src.values())
        except Exception:
            pass

        return chunks, sources, total_rows

    for label, cname in APPCFG.TI_COLLECTIONS.items():
        is_tabular = (label == "csv_xlsx_docs")
        chunks, sources, rows_total = _stats_for_collection(cname, is_tabular=is_tabular)
        totals["chunks"] += chunks
        totals["sources"] += sources

        line = f"- **{cname}** ({label}): {chunks} chunks, {sources} source(s)"
        if is_tabular and rows_total:
            line += f" — ~{rows_total} rows"
        coll_lines.append(line)

    key_set = "✅ set" if getattr(APPCFG, "openai_api_key", "") else "❌ missing"

    md = [
        f"**SQLite DB**: `{APPCFG.sqlite_db_path}`",
        f"Tables ({len(tables)}): " + (", ".join(tables) if tables else "—"),
        "",
        f"**Chroma dir**: `{APPCFG.persist_directory}`",
        f"Collections (totals: {totals['sources']} sources, {totals['chunks']} chunks):",
        *coll_lines,
        "",
        f"**Uploads dir**: `{APPCFG.uploads_dir}`",
        f"**OPENAI_API_KEY**: {key_set}",
    ]
    return "\n".join(md)

# ------------------------
# Snippet click handlers
# ------------------------
def _snippet_click_impl(slot_no: int, snip_cache, last_idx, slot_map):
    try:
        idx = (slot_map or {}).get(slot_no)
        if idx is None:
            return gr.update(value="(No snippet bound to this button yet.)", visible=True)
        content = (snip_cache or {}).get(last_idx, {}).get(idx, "")
        if not content:
            return gr.update(value="(No snippet found for this source.)", visible=True)
        return gr.update(value=content, visible=True)
    except Exception as e:
        return gr.update(value=f"(Error showing snippet: {e})", visible=True)

def snippet_click_1(snip_cache, last_idx, slot_map):
    return _snippet_click_impl(1, snip_cache, last_idx, slot_map)

def snippet_click_2(snip_cache, last_idx, slot_map):
    return _snippet_click_impl(2, snip_cache, last_idx, slot_map)

def snippet_click_3(snip_cache, last_idx, slot_map):
    return _snippet_click_impl(3, snip_cache, last_idx, slot_map)

def snippet_click_4(snip_cache, last_idx, slot_map):
    return _snippet_click_impl(4, snip_cache, last_idx, slot_map)

# ------------------------
# Session history helpers
# ------------------------
def _format_session_choice(s: Dict) -> str:
    """Human label for dropdown: '<id> · <title> (<n> msgs)'."""
    n = len(s.get("chat", []))
    return f"{s.get('id','?')} · {s.get('title','Session')} ({n} msgs)"

def _choices_from_sessions(sessions: List[Dict]) -> List[str]:
    return [_format_session_choice(s) for s in sessions]

def _id_from_choice(choice: Optional[str]) -> Optional[str]:
    if not choice or not isinstance(choice, str):
        return None
    return choice.split(" · ", 1)[0].strip()

def handle_clear_and_archive(chat, sessions_state, next_session_id, active_session_id):
    """
    Archive current chat into sessions (if non-empty) and clear the chat.
    """
    sessions = list(sessions_state or [])
    next_id = int(next_session_id) if next_session_id is not None else 1

    # Archive only if there is content
    if isinstance(chat, list) and chat:
        # Title = first user message (trimmed) + timestamp
        first_user_msg = ""
        for pair in chat:
            if isinstance(pair, (list, tuple)) and pair and isinstance(pair[0], str) and pair[0].strip():
                first_user_msg = pair[0].strip()
                break
        first_part = (first_user_msg[:44] + "…") if len(first_user_msg) > 45 else (first_user_msg or "Untitled")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        session_title = f"{ts} · {first_part}"

        if active_session_id and active_session_id in [s.get("id") for s in sessions]:
            # Update existing session
            for i, s in enumerate(sessions):
                if s.get("id") == active_session_id:
                    sessions[i]["chat"] = chat
                    sessions[i]["title"] = session_title
                    break
        else:
            # Create new session
            session = {
                "id": str(next_id),
                "title": session_title,
                "chat": chat,
            }
            sessions = [session] + sessions
            next_id += 1

    # Build dropdown update
    choices = _choices_from_sessions(sessions)
    hist_update = gr.update(choices=choices, value=None)

    # Return: chat cleared + snippet panel hidden + states reset + sessions
    return (
        [], "",  # chat, msg
        hist_update,  # dropdown
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
        gr.update(value="", visible=False),  # snippet view
        {}, {}, -1,  # snippet states
        sessions, next_id, active_session_id  # sessions_state, next_session_id, active_session_id
    )

def load_session_from_history(choice, sessions_state):
    """
    Load a session (by dropdown label) into the chat window.
    Resets snippet panel (you can click 'Send' again to rebuild snippets).
    """
    sid = _id_from_choice(choice)
    sessions = list(sessions_state or [])
    selected = None
    for s in sessions:
        if s.get("id") == sid:
            selected = s
            break

    if selected is None:
        return gr.update(), "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value="", visible=False), {}, {}, -1, None

    chat_loaded = selected.get("chat", [])
    return (
        chat_loaded, "",
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
        gr.update(value="", visible=False),
        {}, {}, len(chat_loaded) - 1,
        sid  # Set active_session_id to the loaded session ID
    )

# ------------------------
# UI
# ------------------------
def build_ui():
    with gr.Blocks(title="Chatbot · SQL + TextInsight", analytics_enabled=False) as demo:
        gr.Markdown("### 🤖 Chatbot · SQL + RAG (TextInsight)")

        # States for snippet system — use gr.State (not gr.JSON)
        snip_cache_state = gr.State(value={})   # {last_msg_idx: {cite_index: snippet_md}}
        slot_map_state = gr.State(value={})     # {slot_no (1..4): cite_index}
        last_idx_state = gr.Number(value=-1, visible=False)  # last bot message index

        # Session history states
        sessions_state = gr.State(value=[])     # [ {id, title, chat}, ... ]
        next_session_id = gr.State(value=1)
        active_session_id = gr.State(value=None)  # Track the loaded session ID

        with gr.Row():
            # ---------- LEFT COLUMN ----------
            with gr.Column(scale=1):
                chat_type = gr.Dropdown(
                    label="Chat type",
                    choices=[
                        "Q&A with Uploaded CSV/XLSX SQL-DB",
                        "RAG with source uploads",
                    ],
                    value="Q&A with Uploaded CSV/XLSX SQL-DB",
                )
                app_functionality = gr.Radio(
                    label="App functionality",
                    choices=["Chat", "Process files"],
                    value="Chat",
                )

                # Ingestion panel toggled when "Process files" selected
                with gr.Group(visible=False) as ingest_group:
                    gr.Markdown("#### Process sources")
                    # Common uploader above the bars
                    common_up = gr.Files(
                        label="Common upload (PDF/Images/CSV/XLSX)",
                        file_count="multiple"
                    )

                    with gr.Accordion("PDFs upload", open=False):
                        pdf_files = gr.Files(file_count="multiple", file_types=[".pdf"])

                    with gr.Accordion("URLs upload", open=False):
                        urls_text = gr.Textbox(lines=7, placeholder="One URL per line")

                    with gr.Accordion("Images upload", open=False):
                        image_files = gr.Files(
                            file_count="multiple",
                            file_types=[".png", ".jpg", ".jpeg", ".gif", ".webp"]
                        )

                    with gr.Accordion("CSV & XLSX upload", open=False):
                        csv_xlsx_files = gr.Files(
                            file_count="multiple",
                            file_types=[".csv", ".xlsx"]
                        )

                    process_btn = gr.Button("Process sources")
                    process_status = gr.Markdown("")

                    def _toggle_ingest(fn_choice):
                        return gr.update(visible=(fn_choice == "Process files"))

                    app_functionality.change(
                        _toggle_ingest, inputs=[app_functionality], outputs=[ingest_group]
                    )
                    process_btn.click(
                        process_sources_ui,
                        inputs=[common_up, pdf_files, image_files, urls_text, csv_xlsx_files],
                        outputs=[process_status, common_up, pdf_files, image_files, urls_text, csv_xlsx_files],
                    )

                # --- Status / Health --- (Always visible)
                with gr.Accordion("Status / Health", open=False):
                    status_md = gr.Markdown(value="Click **Refresh status** to load.", elem_id="status-md")
                    refresh_btn = gr.Button("Refresh status", variant="secondary")
                refresh_btn.click(
                    lambda: _build_status_markdown(),
                    inputs=[],
                    outputs=[status_md],
                )

                # --- Session history --- (Always visible)
                with gr.Accordion("Session history", open=True):
                    history_dd = gr.Dropdown(
                        label="Saved sessions (click to load)",
                        choices=[],
                        value=None
                    )

            # ---------- RIGHT COLUMN: Chat ----------
            with gr.Column(scale=2):
                chat = gr.Chatbot(height=520)
                with gr.Row():
                    msg = gr.Textbox(
                        label="Message",
                        placeholder="Type your question…",
                        scale=6
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                    clear_btn = gr.Button("Clear", scale=1)

                # ---- Snippets panel ----
                with gr.Accordion("🧩 Snippets for last answer", open=False):
                    gr.Markdown("Click a button to show the exact chunks used from that source.")
                    with gr.Row():
                        snip_btn1 = gr.Button("Snippet [1]", visible=False)
                        snip_btn2 = gr.Button("Snippet [2]", visible=False)
                        snip_btn3 = gr.Button("Snippet [3]", visible=False)
                        snip_btn4 = gr.Button("Snippet [4]", visible=False)
                    snippet_view = gr.Markdown(visible=False)

        # ----- Wiring: Send / Enter -----
        send_btn.click(
            fn=handle_send,
            inputs=[msg, chat, chat_type, app_functionality,
                    snip_cache_state, slot_map_state, last_idx_state],
            outputs=[chat, msg,
                     snip_btn1, snip_btn2, snip_btn3, snip_btn4, snippet_view,
                     snip_cache_state, slot_map_state, last_idx_state]
        )
        msg.submit(
            fn=handle_send,
            inputs=[msg, chat, chat_type, app_functionality,
                    snip_cache_state, slot_map_state, last_idx_state],
            outputs=[chat, msg,
                     snip_btn1, snip_btn2, snip_btn3, snip_btn4, snippet_view,
                     snip_cache_state, slot_map_state, last_idx_state]
        )

        # ----- Wiring: Clear -> archive session + reset UI -----
        clear_btn.click(
            fn=handle_clear_and_archive,
            inputs=[chat, sessions_state, next_session_id, active_session_id],
            outputs=[
                chat, msg,                      # 2
                history_dd,                     # 1
                snip_btn1, snip_btn2, snip_btn3, snip_btn4, snippet_view,  # 5
                snip_cache_state, slot_map_state, last_idx_state,           # 3
                sessions_state, next_session_id, active_session_id          # 3
            ]
        )

        # ----- Wiring: click history -> load session into chat -----
        history_dd.change(
            fn=load_session_from_history,
            inputs=[history_dd, sessions_state],
            outputs=[chat, msg,
                     snip_btn1, snip_btn2, snip_btn3, snip_btn4, snippet_view,
                     snip_cache_state, slot_map_state, last_idx_state, active_session_id]
        )

        # Click handlers for snippet buttons
        snip_btn1.click(snippet_click_1, inputs=[snip_cache_state, last_idx_state, slot_map_state], outputs=[snippet_view])
        snip_btn2.click(snippet_click_2, inputs=[snip_cache_state, last_idx_state, slot_map_state], outputs=[snippet_view])
        snip_btn3.click(snippet_click_3, inputs=[snip_cache_state, last_idx_state, slot_map_state], outputs=[snippet_view])
        snip_btn4.click(snippet_click_4, inputs=[snip_cache_state, last_idx_state, slot_map_state], outputs=[snippet_view])

    return demo

if __name__ == "__main__":
    demo = build_ui()
    # show_api=False avoids rendering the API docs UI; our patches also guarantee stability if still called.
    demo.launch(server_name="0.0.0.0", server_port=7861, share=True, show_api=False)