"""
streamlit_app.py — Blog Expert UI

Tabs
────
  New Blog     — settings form + generate button
  Active Jobs  — live progress for all running jobs (survives page refresh)
  History      — completed / failed runs from SQLite

Fault tolerance
───────────────
  Each generation job writes progress (active agent, progress %, streamed text)
  to the SQLite jobs table every ~0.8 s via JobWriter.
  On page refresh the Active Jobs tab picks up any running jobs from the DB
  and resumes polling — the generation thread itself never stops.

  Cancel button sets cancel_requested=1 in the jobs table.
  The graph thread checks this flag and raises CancelledError.
"""

from __future__ import annotations

import queue
import re
import sys
import time
import threading
from pathlib import Path

import os
import pandas as pd
import streamlit as st
import socket
socket.setdefaulttimeout(30.0)

import sys
import traceback
import threading

def dump_threads():
    try:
        dump_path = Path(__file__).resolve().parent / "thread_dump.txt"
        with open(dump_path, "w", encoding="utf-8") as f:
            for thread_id, frame in sys._current_frames().items():
                thread = threading._active.get(thread_id)
                thread_name = thread.name if thread else "Unknown"
                f.write(f"=== Thread: {thread_name} (ID: {thread_id}) ===\n")
                traceback.print_stack(frame, file=f)
                f.write("\n\n")
    except Exception:
        pass

dump_threads()

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[0]))

from db.database import BlogDatabase
from graph.graph import graph
from graph.state import BlogState
from utilis.job_writer import JobWriter
from utilis.date_formatter import format_local_datetime

_db = BlogDatabase()
RUNNING_THREADS = {}

# ── Constants ─────────────────────────────────────────────────────────────────
IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<url>[^)]+)\)")

RECOMMENDED = {"easy": 3, "medium": 5, "advanced": 7}

AGENT_STATUS = {
    "prepare":              "Scraping & preparing source content",
    "baseline_evaluator":   "Scoring original blog (baseline)",
    "learner":              "Analysing blog intent and gaps",
    "planner":              "Building SEO & content plan",
    "supervisor":           "Supervisor — building 5 focused research briefs",
    "specialists_parallel": "⚡ Running 6 specialist reviews in parallel (Language, Facts, Structure, Images, SEO, GEO)",
    "language":             "Language review",
    "facts":                "Facts & E-E-A-T review",
    "structure":            "Structure review",
    "image":                "Finding relevant images",
    "seo":                  "SEO keyword analysis",
    "geo":                  "GEO / AI-engine optimisation",
    "aggregator":           "Writing long-form draft (streaming)",
    "evaluator":            "Evaluating quality scores",
    "optimizer":            "Optimising article (streaming)",
    "targeted_researcher":  "Targeted research for low-score sections",
    "evaluator_post":       "Re-evaluating after optimisation",
}

AGENT_PCT = {
    "prepare": 2, "baseline_evaluator": 5, "learner": 8, "planner": 12,
    "supervisor": 17, "specialists_parallel": 35, "language": 22, "facts": 27, "structure": 32,
    "image": 37, "seo": 41, "geo": 45, "aggregator": 50,
    "evaluator": 62, "optimizer": 68, "targeted_researcher": 78,
    "evaluator_post": 88,
}

PROVIDER_INFO = {
    "Groq  (Llama-3.3 · Llama-3.1 — free tier)": "groq",
    "Gemini  (Flash 2.0 — free tier)":  "gemini",
    "OpenAI  (GPT-4o)":                 "openai",
    "Custom OpenAI-Compatible LLM":     "custom",
}

DIMENSIONS = ["Language", "Facts", "Structure", "SEO", "GEO", "Freshness"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_clean_title_fallback(url: str) -> str:
    if not url:
        return ""
    parts = [p for p in url.split("/") if p]
    if len(parts) >= 2:
        slug = parts[-1]
        title = slug.replace("-", " ").replace("_", " ").strip()
        if title:
            return title.title()
    return url.split("/")[2] if "//" in url else url

def target_length(pages: int) -> str:
    return f"approximately {pages} web pages, or {pages*550}-{pages*700} words (minimum {pages*550} words)"

def fmt_dur(s: float) -> str:
    s = max(0.0, s)
    if s >= 60:
        return f"{int(s//60)}m {int(s%60):02d}s"
    return f"{s:.1f}s"

def _color(v: int) -> str:
    return "#4ade80" if v >= 90 else ("#fbbf24" if v >= 75 else "#f87171")

def _cell(v: int) -> str:
    return f"<span style='color:{_color(v)};font-weight:700'>{v}</span>"

def _delta(b: int, n: int) -> str:
    d = n - b
    if d > 0: return f"<span style='color:#4ade80;font-weight:700'>▲+{d}</span>"
    if d < 0: return f"<span style='color:#f87171;font-weight:700'>▼{d}</span>"
    return "<span style='color:#6b7280'>—</span>"

def parse_table(block: str):
    lines = [l.strip() for l in block.splitlines() if l.strip()]
    if len(lines) < 2: return None
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    if lines[1].replace("|","").replace("-","").replace(":","").strip(): return None
    rows = [[c.strip() for c in l.strip("|").split("|")] for l in lines[2:] if len(l.strip("|").split("|")) == len(header)]
    return pd.DataFrame(rows, columns=header) if rows else None

def flush_md(buf: list) -> None:
    t = "\n".join(buf).strip()
    if t: st.markdown(t)
    buf.clear()

def _show_tbl_or_md(tbuf: list) -> None:
    tbl = parse_table("\n".join(tbuf))
    if tbl is not None:
        st.table(tbl)
    else:
        st.markdown("\n".join(tbuf))

def _resolve_ui_image(url_str: str) -> Optional[str]:
    if not url_str:
        return None
    from utilis.exporter import _resolve_image_path
    resolved = _resolve_image_path(url_str)
    if resolved and resolved.exists() and resolved.is_file():
        return str(resolved)
    clean_url = url_str.strip().strip("'").strip('"')
    if clean_url.startswith(("http://", "https://")):
        return clean_url
    return None

def render_blog(md: str) -> None:
    lines, buf, tbuf, in_t = md.splitlines(), [], [], False
    for line in lines:
        img = IMAGE_RE.search(line.strip())
        if img:
            flush_md(buf)
            if tbuf:
                _show_tbl_or_md(tbuf)
                tbuf.clear(); in_t = False
            raw_url = img.group("url")
            alt_caption = img.group("alt") or ""
            img_src = _resolve_ui_image(raw_url)
            if img_src:
                try:
                    st.image(img_src, caption=alt_caption or None, use_container_width=True)
                except Exception:
                    if alt_caption:
                        st.caption(f"🖼️ *{alt_caption}*")
            elif alt_caption:
                st.caption(f"🖼️ *{alt_caption}*")
            continue
        if "|" in line and line.strip().startswith("|"):
            flush_md(buf); tbuf.append(line); in_t = True; continue
        if in_t:
            _show_tbl_or_md(tbuf)
            tbuf.clear(); in_t = False
        buf.append(line)
    if tbuf:
        _show_tbl_or_md(tbuf)
    flush_md(buf)

def render_export_buttons(article_text: str, title: str = "Blog Article", key_suffix: str = "") -> None:
    """Renders 1-click download buttons for DOCX, PDF, and Markdown without scores."""
    if not article_text:
        return
    from utilis.exporter import export_to_docx, export_to_pdf, clean_blog_markdown
    
    clean_text = clean_blog_markdown(article_text)
    safe_title = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_") or "blog_article"
    
    st.markdown("<div style='margin: 0.5rem 0 1rem 0;'>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1])
    
    with c1:
        try:
            docx_data = export_to_docx(clean_text, default_title=title)
            st.download_button(
                label="📥 Download Word (.docx)",
                data=docx_data,
                file_name=f"{safe_title}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key=f"dl_docx_{key_suffix}",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"DOCX export error: {e}")
            
    with c2:
        try:
            pdf_data = export_to_pdf(clean_text, default_title=title)
            st.download_button(
                label="📄 Download PDF (.pdf)",
                data=pdf_data,
                file_name=f"{safe_title}.pdf",
                mime="application/pdf",
                key=f"dl_pdf_{key_suffix}",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"PDF export error: {e}")
            
    with c3:
        st.download_button(
            label="📝 Download Markdown (.md)",
            data=clean_text.encode("utf-8"),
            file_name=f"{safe_title}.md",
            mime="text/markdown",
            key=f"dl_md_{key_suffix}",
            use_container_width=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

def score_table(bl: list, en: list, bo: int, no: int) -> None:
    st.markdown("""<style>
.sct{width:100%;border-collapse:collapse;margin-bottom:.8rem}
.sct th{background:#1a1e35;color:#a0a8c0;font-size:.72rem;text-transform:uppercase;
  letter-spacing:.06em;padding:7px 12px;text-align:center;border-bottom:1px solid #2e3350}
.sct td{padding:8px 12px;text-align:center;border-bottom:1px solid #1e2130;font-size:.97rem}
.sct tr:last-child td{border-bottom:none}
.sct td:first-child{text-align:left;color:#e8eaf6;font-weight:600}
.ovr td{background:#1a1e35!important}
</style>""", unsafe_allow_html=True)
    if bo == 0:
        rows = "".join(
            f"<tr><td>{d}</td><td>{_cell(n)}</td></tr>"
            for d, n in zip(DIMENSIONS, en)
        ) + f"<tr class='ovr'><td>Overall</td><td>{_cell(no)}</td></tr>"
        st.markdown(
            f"<table class='sct'><thead><tr>"
            f"<th style='text-align:left'>Dimension</th><th>Score</th>"
            f"</tr></thead><tbody>{rows}</tbody></table>", unsafe_allow_html=True)
    else:
        rows = "".join(
            f"<tr><td>{d}</td><td>{_cell(b)}</td><td>{_cell(n)}</td><td>{_delta(b,n)}</td></tr>"
            for d, b, n in zip(DIMENSIONS, bl, en)
        ) + f"<tr class='ovr'><td>Overall</td><td>{_cell(bo)}</td><td>{_cell(no)}</td><td>{_delta(bo,no)}</td></tr>"
        st.markdown(
            f"<table class='sct'><thead><tr>"
            f"<th style='text-align:left'>Dimension</th><th>Original</th><th>Enhanced</th><th>Delta</th>"
            f"</tr></thead><tbody>{rows}</tbody></table>", unsafe_allow_html=True)

def render_run(run_id: int) -> None:
    run = _db.get_run(run_id); art = _db.get_article(run_id)
    if not run or not art: st.warning("Run not found."); return
    has_resume = (run.status == "failed" and getattr(run, "serialized_state", ""))
    if has_resume:
        col_hdr, col_cont, col_redo, col_del = st.columns([2, 1, 1, 1])
    else:
        col_hdr, col_redo, col_del = st.columns([3, 1, 1])

    with col_hdr:
        if run.status == "failed":
            st.error(f"Failed Run #{run.id} — {run.url}")
        else:
            st.success(f"Run #{run.id} — {run.url}")

    if has_resume:
        with col_cont:
            if st.button("▶️ Continue Run", key=f"cont_top_{run.id}", type="primary", use_container_width=True):
                import json
                import queue
                
                # Deserialise state
                state_data = json.loads(run.serialized_state)
                curr_provider = PROVIDER_INFO.get(st.session_state.get("sidebar_provider_label"), "groq")
                state_data["llm_provider"] = curr_provider
                
                # Override custom LLM settings with current UI settings if applicable
                state_data["custom_model_name"] = st.session_state.get("custom_model_name", "")
                state_data["custom_api_key"] = st.session_state.get("custom_api_key", "")
                state_data["custom_base_url"] = st.session_state.get("custom_base_url", "")
                
                # Jump straight to failed node
                resume_node_name = state_data.get("active_agent") or "prepare"
                state_data["resume_node"] = resume_node_name
                
                # Instantiate state
                state = BlogState(**state_data)
                
                # Setup background job
                sq = queue.Queue()
                target_url = state.url
                job_id = _db.create_job(target_url if state.mode == "enhance" else state.topic_idea)
                jw = JobWriter(_db, job_id)
                jw.start()
                
                state.job_id = job_id
                state.stream_queue = sq
                
                settings = dict(
                    provider=curr_provider,
                    research_level=state.research_level,
                    language_quality=state.language_quality,
                    max_pages=state.max_pages,
                    image_count=state.image_count
                )
                
                t = threading.Thread(
                    target=_run_generation, daemon=True,
                    args=(state, jw, _db, settings, sq),
                )
                t.start()
                
                st.session_state["current_job_id"] = job_id
                st.session_state["next_nav_tab"] = "Active Jobs"
                st.rerun()

    with col_redo:
        if st.button("🔄 Redo Run", key=f"redo_top_{run.id}", type="primary", use_container_width=True):
            # Launch a redo run!
            import queue
            is_generate = run.url.startswith("topic:")
            mode_val = "generate" if is_generate else "enhance"
            topic_idea_val = run.topic_idea or (run.url[6:] if is_generate else "")
            
            target_url = run.url
            sq = queue.Queue()
            job_id = _db.create_job(target_url if mode_val == "enhance" else topic_idea_val)
            jw = JobWriter(_db, job_id)
            jw.start()
            
            state = BlogState(
                url=target_url,
                max_pages=run.max_pages,
                language_quality=run.language_quality,
                research_level=run.research_level,
                research_results=5,
                image_count=run.image_count,
                max_optimizer_passes=run.optimizer_iterations or 3,
                target_length=f"approximately {run.max_pages} web pages, or {run.max_pages*500} words",
                llm_provider=run.llm_provider,
                stream_queue=sq,
                job_id=job_id,
                mode=mode_val,
                topic_idea=topic_idea_val,
                other_info=run.other_info,
                parent_run_id=run.id,
            )
            settings = dict(
                provider=run.llm_provider,
                research_level=run.research_level,
                language_quality=run.language_quality,
                max_pages=run.max_pages,
                image_count=run.image_count
            )
            
            t = threading.Thread(
                target=_run_generation, daemon=True,
                args=(state, jw, _db, settings, sq),
            )
            t.start()
            
            st.session_state["current_job_id"] = job_id
            st.session_state["next_nav_tab"] = "Active Jobs"
            st.rerun()
    with col_del:
        if st.button("🗑️ Delete Run", key=f"del_top_{run.id}", type="secondary", use_container_width=True):
            _db.delete_run(run_id)
            if "view_run_id" in st.session_state:
                del st.session_state["view_run_id"]
            st.rerun()
            
def extract_failed_step_name(err_msg: str) -> str:
    """Extracts a clear, human-readable step name from error message or traceback."""
    if not err_msg:
        return "Generation Pipeline"

    # 1. Regex match for "Failed on step '...'"
    step_match = re.search(r"Failed on step ['\"]?([^'\":\n]+)['\"]?", err_msg, re.IGNORECASE)
    if step_match and step_match.group(1).strip() and step_match.group(1).strip().lower() != "unknown step":
        step_raw = step_match.group(1).strip()
        cleaned = re.sub(r"^[^\w\s]+", "", step_raw).strip()
        return cleaned or step_raw

    # 2. Heuristic inference from traceback and error details
    err_lower = err_msg.lower()
    if "evaluator.py" in err_lower or "evaluatoragent" in err_lower:
        return "Full Evaluation & Scoring"
    if "baseline_evaluator" in err_lower:
        return "Baseline Quality Scoring"
    if "aggregator" in err_lower:
        return "Drafting Comprehensive Blog"
    if "optimizer" in err_lower:
        return "Polishing & Quality Optimization"
    if "scraper.py" in err_lower or "scrape_blog" in err_lower or "scraping" in err_lower:
        return "Scraping & Ingesting Content"
    if "image_agent" in err_lower or "nemotron_image_selector" in err_lower or "firecrawl" in err_lower:
        return "Image Search & Placement"
    if "specialists" in err_lower or "language_agent" in err_lower or "seo_agent" in err_lower or "geo_agent" in err_lower or "facts_agent" in err_lower or "structure_agent" in err_lower:
        return "Specialist Parallel Review"
    if "planner" in err_lower:
        return "Content Planning"
    if "supervisor" in err_lower:
        return "Supervisor Directives"
    if "learner" in err_lower:
        return "Extracting Learnings"
    if "prompt_generator" in err_lower:
        return "Generating Topic Prompt"
    if "connection error" in err_lower or "connecttimeout" in err_lower or "localhost:11434" in err_lower or "ollama" in err_lower:
        return "Local Engine Connection"
    if "groq" in err_lower or "gemini" in err_lower or "openai" in err_lower or "404 not_found" in err_lower:
        return "Cloud LLM Provider API"

    return "Generation Pipeline"


def render_run_view(run_id: int, _db: BlogDatabase, settings: dict, sq: queue.Queue):
    run = _db.get_run(run_id)
    if not run:
        st.error("Run not found.")
        return
    art = _db.get_article(run_id)
    
    col_t, col_re, col_del = st.columns([6, 2, 2])
    with col_t:
        st.subheader(run.title or run.url)
    with col_re:
        if st.button("🔁 Re-run", key=f"rerun_top_{run.id}", type="primary", use_container_width=True):
            job_id = _db.create_job(
                url=run.url,
                settings=settings,
                parent_run_id=run.id,
                title=f"Re-run: {run.title or run.url}"
            )
            jw = JobWriter(job_id=job_id, url=run.url, db=_db)
            
            raw_blog = art.original_blog if (art and art.original_blog) else ""
            cleaned_blog = art.original_blog if (art and art.original_blog) else ""
            topic_idea = getattr(run, "topic_idea", "") or ""
            other_info = getattr(run, "other_info", "") or ""
            mode = "generate" if (topic_idea or run.url.startswith("topic:")) else "enhance"
            
            state = BlogState(
                url=run.url,
                raw_blog=raw_blog,
                cleaned_blog=cleaned_blog,
                topic_idea=topic_idea,
                other_info=other_info,
                mode=mode,
                job_id=job_id,
                parent_run_id=run.id,
                title=run.title,
                **settings
            )
            
            t = threading.Thread(
                target=_run_generation, daemon=True,
                args=(state, jw, _db, settings, sq),
            )
            t.start()
            
            st.session_state["current_job_id"] = job_id
            st.session_state["next_nav_tab"] = "Active Jobs"
            st.rerun()
    with col_del:
        if st.button("🗑️ Delete Run", key=f"del_top_{run.id}", type="secondary", use_container_width=True):
            _db.delete_run(run_id)
            if "view_run_id" in st.session_state:
                del st.session_state["view_run_id"]
            st.rerun()
            
    if run.status == "failed":
        st.caption(f"{format_local_datetime(run.created_at)} · {run.llm_provider or 'unknown'}")
        
        err_msg = run.error_message or ""
        failed_step = extract_failed_step_name(err_msg)
        
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("Failed On Step", failed_step)
        with col_s2:
            st.metric("Evaluation Iterations", run.evaluation_iterations)
        with col_s3:
            st.metric("Optimization Iterations", run.optimizer_iterations)
            
        st.subheader("Error Details")
        st.code(err_msg, language="text")
    else:
        st.caption(f"{format_local_datetime(run.created_at)} · {fmt_dur(run.duration_seconds)} · {run.llm_provider} · {run.optimizer_iterations} pass(es)")
        score_table(
            [run.baseline_language, run.baseline_facts, run.baseline_structure, run.baseline_seo, run.baseline_geo, run.baseline_freshness],
            [run.enhanced_language, run.enhanced_facts, run.enhanced_structure, run.enhanced_seo, run.enhanced_geo, run.enhanced_freshness],
            run.baseline_overall, run.enhanced_overall,
        )
        t1, t2 = st.tabs(["Original Blog", "Enhanced Blog"])
        with t1:
            if run.url.startswith("topic:"):
                st.info("No original blog (Generate Mode).")
            elif art.original_blog:
                render_blog(art.original_blog)
            else:
                st.info("Not saved.")
        with t2:
            if art.enhanced_blog:
                render_export_buttons(art.enhanced_blog, title=run.title or "Enhanced Blog", key_suffix=f"run_{run.id}")
                render_blog(art.enhanced_blog)
            else:
                st.warning("Not generated.")

    # Render past generations / attempts list
    related_runs = [r for r in _db.list_runs(100) if r.url == run.url and r.id != run.id]
    if related_runs:
        st.write("---")
        st.subheader("Other Redone Attempts")
        for rr in related_runs:
            status_icon = "✅" if rr.status == "completed" else "❌"
            label = f"{status_icon} Attempt #{rr.id} ({format_local_datetime(rr.created_at)})"
            if rr.status == "completed":
                label += f" — Overall Score: {rr.enhanced_overall}"
            else:
                label += " — Failed"
            
            c_attempt, c_btn = st.columns([4, 1])
            with c_attempt:
                st.write(label)
            with c_btn:
                if st.button("View Attempt", key=f"view_rr_{rr.id}"):
                    st.session_state["view_run_id"] = rr.id
                    st.rerun()


# ── Background generation ─────────────────────────────────────────────────────

def _run_generation(state: BlogState, jw: JobWriter, db: BlogDatabase,
                    settings: dict, sq: queue.Queue) -> None:
    import traceback
    start = time.time()
    try:
        from utilis.token_counter import reset_token_counter
        reset_token_counter()
        result_holder: dict = {"result": None, "error": None}

        current_agent = ["prepare"]

        def _update_agent_pct(agent_name: str):
            current_agent[0] = agent_name
            pct = AGENT_PCT.get(agent_name, 50)
            jw.set_agent(agent_name, pct)

        state.stream_queue = sq
        state.active_agent_callback = _update_agent_pct

        def _graph_thread():
            try:
                result_holder["result"] = BlogState(**graph.invoke(state))
            except BaseException as exc:
                result_holder["error"] = exc

        gt = threading.Thread(target=_graph_thread, daemon=True)
        gt.start()
        RUNNING_THREADS[state.job_id] = gt

        last_agent   = ""
        streamed     = ""
        stream_agent = ""

        while gt.is_alive():
            if jw.is_cancel_requested():
                result_holder["error"] = "Cancelled by user."
                break

            if state.title and not jw._title:
                jw.set_title(state.title)

            current = current_agent[0]
            if current != last_agent:
                last_agent = current
                if current in ("aggregator", "optimizer"):
                    stream_agent = current
                    streamed = ""

            if stream_agent == current:
                try:
                    while True:
                        chunk = sq.get_nowait()
                        if chunk is None:
                            stream_agent = ""
                            break
                        if isinstance(chunk, list):
                            chunk_str = "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in chunk)
                        else:
                            chunk_str = str(chunk)
                        if chunk_str:
                            streamed += chunk_str
                            jw.append_stream(chunk_str)
                except queue.Empty:
                    pass

            time.sleep(0.3)

        gt.join(timeout=5)

        try:
            while True:
                chunk = sq.get_nowait()
                if chunk is None: break
                if isinstance(chunk, list):
                    chunk_str = "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in chunk)
                else:
                    chunk_str = str(chunk)
                if chunk_str:
                    streamed += chunk_str
                    jw.append_stream(chunk_str)
        except queue.Empty:
            pass

        from utilis.token_counter import get_tokens
        p_tok, c_tok, t_tok = get_tokens()

        err = result_holder.get("error")
        if err:
            agent_failed_on = current_agent[0]
            agent_label = AGENT_STATUS.get(agent_failed_on, agent_failed_on)
            full_err_msg = f"Failed on step '{agent_label}': {err}"
            
            from graph.graph import LAST_ACTIVE_STATE
            latest_state = LAST_ACTIVE_STATE.get(state.job_id) or state
            opt_iter = getattr(latest_state, "optimizer_iteration", 0)
            eval_iter = getattr(latest_state, "iteration", 0)
            try:
                state_json = latest_state.model_dump_json()
            except Exception:
                try:
                    state_json = latest_state.json()
                except Exception:
                    state_json = ""
            failed_run_id = db.save_failed_run(
                url=latest_state.url, error_message=full_err_msg, title=latest_state.title,
                prompt_tokens=p_tok, completion_tokens=c_tok, total_tokens=t_tok,
                optimizer_iterations=opt_iter, evaluation_iterations=eval_iter,
                llm_provider=settings.get("provider", ""),
                research_level=settings.get("research_level", ""),
                language_quality=settings.get("language_quality", ""),
                max_pages=settings.get("max_pages", 5),
                image_count=settings.get("image_count", 3),
                parent_run_id=latest_state.parent_run_id,
                topic_idea=getattr(latest_state, "topic_idea", ""),
                other_info=getattr(latest_state, "other_info", ""),
                serialized_state=state_json
            )
            jw.stop(run_id=failed_run_id, error=full_err_msg)
            return

        result: BlogState = result_holder["result"]
        elapsed = time.time() - start

        if result and result.image_output:
            enhanced_text = result.optimized_blog or result.aggregated_blog or ""
            for img in result.image_output:
                if img.get("is_custom"):
                    continue
                local_p = img.get("local_path") or img.get("url")
                remote_u = img.get("remote_url") or img.get("source_url")
                if local_p and remote_u and enhanced_text and not img.get("is_custom"):
                    enhanced_text = enhanced_text.replace(local_p, remote_u)
                if local_p and os.path.exists(local_p) and not img.get("is_custom"):
                    try:
                        os.remove(local_p)
                    except Exception:
                        pass
            if result.optimized_blog:
                result.optimized_blog = enhanced_text
            elif result.aggregated_blog:
                result.aggregated_blog = enhanced_text

        try:
            st.cache_data.clear()
            st.cache_resource.clear()
        except Exception:
            pass

        try:
            result_json = result.model_dump_json()
        except Exception:
            try:
                result_json = result.json()
            except Exception:
                result_json = ""
        provider_display = settings.get("provider", "groq")
        if getattr(state, "execution_mode", "online") == "offline":
            provider_display = f"Local ({getattr(state, 'local_engine', 'ollama')}: {getattr(state, 'local_model_name', 'local')})"
        elif getattr(state, "execution_mode", "online") == "hybrid":
            provider_display = f"Hybrid (Local {getattr(state, 'local_model_name', 'local')} + {settings.get('provider', '')})"

        run_id = db.save_run(
            url=state.url, title=result.title, duration_seconds=elapsed,
            llm_provider=provider_display, research_level=settings["research_level"],
            language_quality=settings["language_quality"], max_pages=settings["max_pages"],
            image_count=settings["image_count"],
            optimizer_iterations=result.optimizer_iteration,
            evaluation_iterations=result.iteration,
            baseline_overall=result.baseline_overall_score,
            baseline_language=result.baseline_language_score,
            baseline_facts=result.baseline_facts_score,
            baseline_structure=result.baseline_structure_score,
            baseline_seo=result.baseline_seo_score,
            baseline_geo=result.baseline_geo_score,
            baseline_freshness=result.baseline_freshness_score,
            enhanced_overall=result.overall_score,
            enhanced_language=result.language_score,
            enhanced_facts=result.facts_score,
            enhanced_structure=result.structure_score,
            enhanced_seo=result.seo_score,
            enhanced_geo=result.geo_score,
            enhanced_freshness=result.freshness_score,
            original_blog=result.cleaned_blog,
            enhanced_blog=result.optimized_blog or result.aggregated_blog,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            total_tokens=t_tok,
            parent_run_id=state.parent_run_id,
            topic_idea=state.topic_idea,
            other_info=state.other_info,
            serialized_state=result_json,
        )
        jw.stop(run_id=run_id)
    except BaseException as fatal_exc:
        agent_failed_on = current_agent[0] if ('current_agent' in locals() and current_agent) else "prepare"
        agent_label = AGENT_STATUS.get(agent_failed_on, agent_failed_on)
        err_msg = f"Failed on step '{agent_label}': Fatal thread error: {fatal_exc}\n{traceback.format_exc()}"
        try:
            from utilis.token_counter import get_tokens
            p_tok, c_tok, t_tok = get_tokens()
        except Exception:
            p_tok, c_tok, t_tok = 0, 0, 0
            
        from graph.graph import LAST_ACTIVE_STATE
        latest_state = LAST_ACTIVE_STATE.get(state.job_id) or state
        opt_iter = getattr(latest_state, "optimizer_iteration", 0)
        eval_iter = getattr(latest_state, "iteration", 0)
        try:
            state_json = latest_state.model_dump_json()
        except Exception:
            try:
                state_json = latest_state.json()
            except Exception:
                state_json = ""
        failed_run_id = db.save_failed_run(
            url=latest_state.url, error_message=err_msg, title=getattr(latest_state, "title", ""),
            prompt_tokens=p_tok, completion_tokens=c_tok, total_tokens=t_tok,
            optimizer_iterations=opt_iter, evaluation_iterations=eval_iter,
            llm_provider=settings.get("provider", ""),
            research_level=settings.get("research_level", ""),
            language_quality=settings.get("language_quality", ""),
            max_pages=settings.get("max_pages", 5),
            image_count=settings.get("image_count", 3),
            parent_run_id=latest_state.parent_run_id,
            topic_idea=getattr(latest_state, "topic_idea", ""),
            other_info=getattr(latest_state, "other_info", ""),
            serialized_state=state_json
        )
        jw.stop(run_id=failed_run_id, error=err_msg)
    finally:
        RUNNING_THREADS.pop(state.job_id, None)
        from graph.graph import LAST_ACTIVE_STATE
        LAST_ACTIVE_STATE.pop(state.job_id, None)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Blog Expert", page_icon="✍️", layout="wide")
st.markdown("""<style>
.block-container{max-width:1120px;padding-top:2rem}
article,.stMarkdown{font-family:Inter,"Segoe UI",system-ui,sans-serif;font-size:1.02rem;line-height:1.75}
h1{color:#f8fafc;font-weight:800;letter-spacing:-0.02em;line-height:1.25;margin-bottom:1.2rem}
h2{color:#818cf8;font-weight:700;letter-spacing:-0.01em;border-bottom:1px solid #2e3350;padding-bottom:0.4rem;margin-top:2rem;margin-bottom:0.8rem}
h3{color:#38bdf8;font-weight:600;margin-top:1.3rem;margin-bottom:0.5rem}
.stMarkdown strong{color:#38bdf8;font-weight:600}
blockquote{border-left:4px solid #6366f1!important;background:linear-gradient(135deg,rgba(30,41,59,0.7) 0%,rgba(15,23,42,0.85) 100%)!important;padding:14px 20px!important;border-radius:8px!important;margin:1.4rem 0!important;color:#f1f5f9!important;box-shadow:0 4px 14px rgba(0,0,0,0.2)!important}
blockquote p{margin:0!important;font-size:1.01rem!important}
.stMarkdown table{width:100%;border-collapse:separate;border-spacing:0;border-radius:8px;overflow:hidden;margin:1.5rem 0;border:1px solid #334155}
.stMarkdown th{background:#1e293b;color:#38bdf8;font-weight:600;padding:10px 14px;border-bottom:2px solid #38bdf8}
.stMarkdown td{padding:10px 14px;border-top:1px solid #1e293b;background:rgba(15,23,42,0.45);color:#e2e8f0}
.agent-card{background:#1e2130;border:1px solid #2e3350;border-radius:10px;padding:12px 16px;margin-bottom:8px}
.agent-card p{margin:0;font-size:.95rem}
.alabel{color:#a0a8c0;font-size:.75rem;text-transform:uppercase;letter-spacing:.07em;margin-bottom:3px!important}
.avalue{color:#e8eaf6;font-size:1rem;font-weight:600}
.timerv{color:#7c83fd;font-size:1.7rem;font-weight:700;font-variant-numeric:tabular-nums}
</style>""", unsafe_allow_html=True)

st.title("Blog Expert")

# ── Sidebar settings ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    
    # ── Execution Mode ────────────────────────────────────────────────────────
    st.subheader("🤖 Execution Mode")
    exec_mode_label = st.radio(
        "Mode",
        [
            "🌐 Online Cloud (Groq / Gemini / OpenAI)",
            "💻 Offline Local GPU (Ollama / LM Studio)",
            "🔀 Hybrid (Local GPU Analysis + Cloud Prose)",
        ],
        index=0,
        key="sidebar_exec_mode",
        help="Offline runs 100% locally on your GPU/CPU with zero API costs. Hybrid uses local GPU for fast private analysis and Cloud for prose."
    )
    exec_mode_map = {
        "🌐 Online Cloud (Groq / Gemini / OpenAI)": "online",
        "💻 Offline Local GPU (Ollama / LM Studio)": "offline",
        "🔀 Hybrid (Local GPU Analysis + Cloud Prose)": "hybrid",
    }
    selected_exec_mode = exec_mode_map[exec_mode_label]

    local_engine = "ollama"
    local_model_name = "qwen2.5:7b"
    local_base_url = "http://localhost:11434/v1"
    local_api_key = "ollama"

    # Local LLM Controls (if Offline or Hybrid)
    local_is_connected = True
    local_status_msg = "Connected"
    if selected_exec_mode in ("offline", "hybrid"):
        from config.local_llm import discover_ollama_models, discover_lmstudio_models, OLLAMA_DEFAULT_BASE_URL, LMSTUDIO_DEFAULT_BASE_URL
        
        st.markdown("**Local LLM Engine**")
        local_engine = st.selectbox("Engine", ["Ollama", "LM Studio", "Custom Local URL"], key="local_engine_select")
        
        if local_engine == "Ollama":
            local_base_url = st.text_input("Ollama URL", value=OLLAMA_DEFAULT_BASE_URL, key="ollama_base_url")
            local_is_connected, models_found, local_status_msg = discover_ollama_models(local_base_url)
            if local_is_connected:
                st.caption(f"🟢 **Status**: {local_status_msg}")
                local_model_name = st.selectbox("Ollama Model", models_found, key="ollama_model_select")
            else:
                st.caption(f"🔴 **Status**: {local_status_msg}")
                local_model_name = st.text_input("Model Name", value="qwen2.5:3b", key="ollama_model_manual")
            st.caption("💡 **GPU Tip (4GB RTX 3050)**: Pull `qwen2.5:3b` (`ollama pull qwen2.5:3b`) or `llama3.2:3b` to fit 100% inside GPU VRAM at 80+ tok/s.")
                
        elif local_engine == "LM Studio":
            local_base_url = st.text_input("LM Studio URL", value=LMSTUDIO_DEFAULT_BASE_URL, key="lmstudio_base_url")
            local_is_connected, models_found, local_status_msg = discover_lmstudio_models(local_base_url)
            if local_is_connected:
                st.caption(f"🟢 **Status**: {local_status_msg}")
                local_model_name = st.selectbox("LM Studio Model", models_found, key="lmstudio_model_select")
            else:
                st.caption(f"🔴 **Status**: {local_status_msg}")
                local_model_name = st.text_input("Model Name", value="local-model", key="lmstudio_model_manual")
        else:
            local_base_url = st.text_input("Base URL", value="http://localhost:8000/v1", key="custom_local_url")
            local_model_name = st.text_input("Model Name", value="qwen2.5:3b", key="custom_local_model")
            local_api_key = st.text_input("API Key (optional)", value="local", key="custom_local_key")

    # Cloud LLM Provider Controls (if Online or Hybrid)
    selected_provider = "groq"
    if selected_exec_mode in ("online", "hybrid"):
        st.markdown("**Cloud LLM Provider**" if selected_exec_mode == "hybrid" else "**LLM Provider**")
        provider_label = st.selectbox("Cloud Provider", list(PROVIDER_INFO.keys()), key="sidebar_provider_label")
        selected_provider = PROVIDER_INFO[provider_label]
        if selected_provider == "openai":
            from config import settings
            from config.llm_registry import _is_valid_openai_key
            if not _is_valid_openai_key(settings.OPENAI_API_KEY):
                st.error("⚠️ OPENAI_API_KEY in .env is missing or invalid! (Must start with 'sk-').")
            else:
                st.info("GPT-4o — costs apply.", icon="ℹ️")
        elif selected_provider == "gemini":
            from config import settings
            from config.llm_registry import _is_valid_gemini_key
            if not _is_valid_gemini_key(settings.GEMINI_API_KEY):
                st.error("⚠️ GEMINI_API_KEY in .env is missing or invalid! (Must start with 'AIzaSy').")
            else:
                st.info("Gemini 2.5 Flash — free tier.", icon="ℹ️")
        elif selected_provider == "custom":
            st.text_input("Custom Model Name", value="gpt-4o-mini", key="custom_model_name")
            st.text_input("Custom Base URL", value="https://api.openai.com/v1", key="custom_base_url")
            st.text_input("Custom API Key", type="password", value="", key="custom_api_key")
    else:
        selected_provider = "local"

    st.divider()
    if "research_level" not in st.session_state: st.session_state.research_level = "medium"
    if "max_pages"      not in st.session_state: st.session_state.max_pages = 5

    language_quality = st.selectbox("Language quality", ["easy", "medium", "advanced"], index=1)

    use_rec = st.checkbox("Use recommended research settings", value=True)
    if use_rec:
        rl = st.selectbox("Research level", list(RECOMMENDED), key="research_level")
        mp = RECOMMENDED[rl]; st.session_state.max_pages = mp
        st.slider("Max pages", 1, 10, mp, disabled=True, key=f"locked_{rl}")
        st.caption(f"Recommended: {mp} pages for {rl} research.")
    else:
        rl = st.selectbox("Research level", list(RECOMMENDED), key="research_level")
        mp = st.slider("Max pages", 1, 10, value=st.session_state.max_pages, key="max_pages")

    image_count = st.slider("Number of images", 0, 20, 3)

    st.divider()
    st.subheader("⚡ Generation Speed Profile")
    speed_mode_label = st.radio(
        "Speed Profile",
        ["⚡ Turbo (<45s)", "⚖️ Balanced (60–90s)", "🔬 Deep Analysis (2–3 mins)"],
        index=0,
        help="Turbo runs parallel specialist analysis with single-pass synthesis for <45s turnaround."
    )
    speed_mode_map = {
        "⚡ Turbo (<45s)": "turbo",
        "⚖️ Balanced (60–90s)": "balanced",
        "🔬 Deep Analysis (2–3 mins)": "deep",
    }
    selected_speed_mode = speed_mode_map.get(speed_mode_label, "turbo")

    st.divider()
    st.subheader("Research depth")
    research_results = st.slider("Search results to fetch", 3, 15, 5,
        help="3–4: fast. 5–7: default. 10–15: maximum depth.")

    st.divider()
    st.subheader("Optimisation")
    max_passes = st.slider("Max optimisation passes", 1, 8, 3,
        help="1–2: fast/cheap. 3: default. 5–8: max quality.")
    st.caption(f"Est. ~{max_passes*10_000:,} tokens on prose provider per run.")

    st.divider()
    st.subheader("Run History")
    if st.button("Refresh history", use_container_width=True): st.rerun()
    runs = _db.list_runs(limit=50)
    if not runs:
        st.caption("No runs yet.")
    else:
        for run in runs:
            domain   = run.url.split("/")[2] if "//" in run.url else run.url[:28]
            icon     = "✅" if run.status == "completed" else "❌"
            delta    = run.enhanced_overall - run.baseline_overall
            ds       = f"▲+{delta}" if delta > 0 else (f"▼{delta}" if delta < 0 else "—")
            sc1, sc2 = st.columns([4, 1])
            with sc1:
                display_name = run.title or get_clean_title_fallback(run.url)
                if len(display_name) > 35:
                    display_name = display_name[:32] + "..."
                local_time_str = format_local_datetime(run.created_at)
                if st.button(f"{icon} {display_name}\n{local_time_str} · {ds} · {fmt_dur(run.duration_seconds)}",
                             key=f"run_{run.id}", use_container_width=True):
                    st.session_state["view_run_id"] = run.id
                    st.rerun()
            with sc2:
                if st.button("🗑️", key=f"sdel_{run.id}", help="Delete run"):
                    _db.delete_run(run.id)
                    if st.session_state.get("view_run_id") == run.id:
                        del st.session_state["view_run_id"]
                    st.rerun()
# Check if current running job has transitioned to completed/failed
current_job_id = st.session_state.get("current_job_id")
if current_job_id:
    job_row = _db.get_job(current_job_id)
    if job_row and job_row["status"] in ("completed", "failed"):
        run_id = job_row["run_id"]
        status = job_row["status"]
        st.session_state["current_job_id"] = None
        if status == "completed" and run_id:
            st.session_state["last_completed_run_id"] = run_id
            st.session_state["next_nav_tab"] = "New Blog"
        elif status == "failed" and run_id:
            st.session_state["view_run_id"] = run_id
            st.session_state["next_nav_tab"] = "History"
        st.rerun()


# ── Main navigation bar ───────────────────────────────────────────────────────
if "next_nav_tab" in st.session_state:
    st.session_state["nav_tab"] = st.session_state.pop("next_nav_tab")

if "nav_tab" not in st.session_state:
    st.session_state["nav_tab"] = "New Blog"

nav_selection = st.segmented_control(
    "Navigation",
    ["New Blog", "Active Jobs", "History"],
    selection_mode="single",
    key="nav_tab",
    label_visibility="collapsed"
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB: New Blog
# ══════════════════════════════════════════════════════════════════════════════
if nav_selection == "New Blog":
    st.subheader("Create or Enhance a Blog Post")
    
    pathway = st.radio("Choose Pathway", ["Enhance Existing Blog", "Generate New Blog from Topic"], horizontal=True, key="generation_pathway")
    
    if pathway == "Enhance Existing Blog":
        main_url_val = st.text_input("Enter Blog URL to Enhance", value="https://aiindia.ai/corporate-ai-trainings/", key="main_url_val")
        topic_idea = ""
        other_info = ""
        mode = "enhance"
    else:
        topic_idea = st.text_input("Enter Topic or Idea", value="", placeholder="e.g., The Future of AI in Healthcare", key="topic_idea")
        other_info = st.text_area("Important Guidelines / Settings", value="", placeholder="e.g., Focus on radiology, use a formal tone.", key="other_info")
        main_url_val = ""
        mode = "generate"

    # ── Custom Images Section ──────────────────────────────────────────────────
    with st.expander("🖼️ Custom Images & Contextual Placement (Optional)", expanded=False):
        st.markdown(
            "Upload your own images and specify details/context for each. "
            "The AI agent will analyze the details and place each image in the most relevant section of the blog."
        )
        uploaded_files = st.file_uploader(
            "Upload image files",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key="custom_images_uploader"
        )
        custom_images_list = []
        if uploaded_files:
            custom_img_dir = Path.cwd() / "data" / "custom_images"
            custom_img_dir.mkdir(parents=True, exist_ok=True)
            st.markdown(f"**Configuring {len(uploaded_files)} Custom Image(s):**")
            for idx, uf in enumerate(uploaded_files):
                safe_name = re.sub(r"[^\w\.-]", "_", uf.name)
                saved_path = custom_img_dir / f"custom_{idx}_{safe_name}"
                with open(saved_path, "wb") as f:
                    f.write(uf.getbuffer())
                
                with st.container():
                    col_prev, col_fields = st.columns([1, 3])
                    with col_prev:
                        st.image(str(saved_path), caption=uf.name, use_container_width=True)
                    with col_fields:
                        c_caption = st.text_input(
                            f"Caption #{idx+1}",
                            value=Path(uf.name).stem.replace("_", " ").replace("-", " ").title(),
                            key=f"c_cap_{idx}_{uf.name}",
                            help="Displayed underneath the image in the blog"
                        )
                        c_desc = st.text_area(
                            f"Context / Details #{idx+1}",
                            value="",
                            placeholder="Explain what this image illustrates and what topic or section it relates to...",
                            key=f"c_desc_{idx}_{uf.name}",
                            help="The AI uses this to match the image to the most appropriate section."
                        )
                        c_hint = st.text_input(
                            f"Placement Hint (optional) #{idx+1}",
                            value="",
                            placeholder="e.g., In the introduction, or right after discussing architecture",
                            key=f"c_hint_{idx}_{uf.name}"
                        )
                    
                    custom_images_list.append({
                        "image_path": str(saved_path),
                        "caption": c_caption,
                        "description": c_desc,
                        "placement_hint": c_hint,
                        "source": "Custom Upload"
                    })
                    st.markdown("<hr style='margin: 0.5rem 0; border-color: #2e3350;'/>", unsafe_allow_html=True)

    main_gen_btn = st.button("🚀 Start Generation Process", type="primary", use_container_width=True, key="main_gen_btn")

    if main_gen_btn:
        target_url = main_url_val.strip() if mode == "enhance" else ("topic:" + topic_idea.strip())
        if mode == "enhance" and not main_url_val.strip():
            st.error("Enter a blog URL first.")
        elif mode == "generate" and not topic_idea.strip():
            st.error("Enter a topic or idea first.")
        elif selected_exec_mode in ("offline", "hybrid") and not local_is_connected:
            clean_host = local_base_url.replace('/v1', '')
            st.error(
                f"⚠️ Cannot connect to local engine at `{local_base_url}` ({local_status_msg}).\n\n"
                f"Please start {local_engine} on your computer (`{clean_host}`), or switch **Execution Mode** in the sidebar to **🌐 Online Cloud (Groq / Gemini / OpenAI)**."
            )
        else:
            sq  = queue.Queue()
            job_id = _db.create_job(target_url if mode == "enhance" else topic_idea.strip())
            jw  = JobWriter(_db, job_id)
            jw.start()

            state = BlogState(
                url=target_url, max_pages=mp,
                language_quality=language_quality,
                research_level=rl, research_results=research_results,
                image_count=image_count, max_optimizer_passes=max_passes,
                custom_images=custom_images_list,
                speed_mode=selected_speed_mode,
                execution_mode=selected_exec_mode,
                local_engine=local_engine,
                local_model_name=local_model_name,
                local_base_url=local_base_url,
                local_api_key=local_api_key,
                target_length=target_length(mp),
                llm_provider=selected_provider, stream_queue=sq,
                job_id=job_id,
                mode=mode,
                topic_idea=topic_idea.strip(),
                other_info=other_info.strip(),
                custom_model_name=st.session_state.get("custom_model_name", ""),
                custom_api_key=st.session_state.get("custom_api_key", ""),
                custom_base_url=st.session_state.get("custom_base_url", ""),
            )
            settings = dict(provider=selected_provider, research_level=rl,
                            language_quality=language_quality, max_pages=mp,
                            image_count=image_count)

            t = threading.Thread(
                target=_run_generation, daemon=True,
                args=(state, jw, _db, settings, sq),
            )
            t.start()

            st.session_state["current_job_id"] = job_id
            st.session_state["next_nav_tab"] = "Active Jobs"
            st.rerun()
    else:
        if mode == "enhance":
            st.info("Enter your blog URL above and click **🚀 Start Generation Process**.")
        else:
            st.info("Enter your topic idea and click **🚀 Start Generation Process**.")

    # Display the last completed generation if present!
    last_run_id = st.session_state.get("last_completed_run_id")
    if last_run_id:
        st.write("---")
        st.subheader("📝 Last Completed Generation")
        
        # Add a clear button to remove this display from the home screen
        if st.button("🗑️ Clear Last Generation from Screen", key="clear_last_gen", type="secondary"):
            del st.session_state["last_completed_run_id"]
            st.rerun()
            
        run = _db.get_run(last_run_id)
        art = _db.get_article(last_run_id)
        if run and art:
            st.caption(f"Generated at {format_local_datetime(run.created_at)} · {run.llm_provider} · {run.optimizer_iterations} pass(es)")
            score_table(
                [run.baseline_language, run.baseline_facts, run.baseline_structure, run.baseline_seo, run.baseline_geo, run.baseline_freshness],
                [run.enhanced_language, run.enhanced_facts, run.enhanced_structure, run.enhanced_seo, run.enhanced_geo, run.enhanced_freshness],
                run.baseline_overall, run.enhanced_overall,
            )
            
            t1, t2 = st.tabs(["Original Blog", "Enhanced Blog"])
            with t1:
                if run.url.startswith("topic:"):
                    st.info("No original blog (Generate Mode).")
                elif art.original_blog:
                    render_blog(art.original_blog)
                else:
                    st.info("Not saved.")
            with t2:
                if art.enhanced_blog:
                    render_export_buttons(art.enhanced_blog, title=run.title or "Enhanced Blog", key_suffix=f"last_{run.id}")
                    render_blog(art.enhanced_blog)
                else:
                    st.warning("Not generated.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB: Active Jobs
# ══════════════════════════════════════════════════════════════════════════════
elif nav_selection == "Active Jobs":
    active = _db.list_active_jobs()

    if not active:
        st.info("No active jobs. Start a generation from the New Blog tab.")
    else:
        for job_row in active:
            job_id   = job_row["id"]
            job_url  = job_row["url"]
            agent    = job_row["active_agent"] or "prepare"
            pct      = job_row["progress_pct"]
            stream   = job_row["streamed_text"] or ""
            started  = job_row["started_at"]

            # Calculate elapsed duration dynamically
            from datetime import datetime, timezone
            try:
                started_str = started
                if started_str.endswith("Z"):
                    started_str = started_str[:-1] + "+00:00"
                started_dt = datetime.fromisoformat(started_str)
                if started_dt.tzinfo is None:
                    started_dt = started_dt.replace(tzinfo=timezone.utc)
                elapsed_seconds = (datetime.now(timezone.utc) - started_dt).total_seconds()
                elapsed_str = fmt_dur(elapsed_seconds)
            except Exception:
                elapsed_str = "0.0s"

            with st.container():
                col_info, col_timer, col_cancel = st.columns([4, 2, 1.2])
                with col_info:
                    display_name = job_row["title"] or job_url
                    st.markdown(f"**Job #{job_id}** — `{display_name}`")
                    st.caption(f"Started: {format_local_datetime(started)}")
                with col_timer:
                    st.markdown(
                        f'<div style="text-align: right;"><p class="alabel" style="margin:0;">Elapsed Time</p>'
                        f'<p class="timerv" style="margin:0;">{elapsed_str}</p></div>',
                        unsafe_allow_html=True
                    )
                with col_cancel:
                    if st.button("Cancel", key=f"cancel_{job_id}", type="secondary"):
                        _db.cancel_job(job_id)
                        t_obj = RUNNING_THREADS.get(job_id)
                        if t_obj and t_obj.ident:
                            import ctypes
                            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                                ctypes.c_long(t_obj.ident),
                                ctypes.py_object(KeyboardInterrupt)
                            )
                        st.warning(f"Cancelled job #{job_id} immediately.")

                label = AGENT_STATUS.get(agent, agent)
                st.markdown(
                    f'<div class="agent-card"><p class="alabel">Running now</p>'
                    f'<p class="avalue">{label}</p></div>',
                    unsafe_allow_html=True,
                )
                st.progress(min(pct / 100, 0.97))

                if stream:
                    with st.expander("Live article preview", expanded=True):
                        preview = stream[-4000:] if len(stream) > 4000 else stream
                        st.markdown(preview + " ▌")

            st.divider()

        # Auto-refresh every 2 s while jobs are active
        time.sleep(0.1)
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB: History (view past runs)
# ══════════════════════════════════════════════════════════════════════════════
elif nav_selection == "History":
    view_id = st.session_state.get("view_run_id")
    if view_id:
        if st.button("Back to run list"):
            del st.session_state["view_run_id"]
            st.rerun()
        render_run(view_id)
    else:
        completed = [r for r in _db.list_runs(50) if r.status == "completed"]
        failed    = [r for r in _db.list_runs(50) if r.status == "failed"]

        if not completed and not failed:
            st.info("No completed runs yet.")
        else:
            for run in completed:
                display_name = run.title or get_clean_title_fallback(run.url)
                local_time_str = format_local_datetime(run.created_at)
                delta  = run.enhanced_overall - run.baseline_overall
                ds     = f"▲+{delta}" if delta > 0 else (f"▼{delta}" if delta < 0 else "—")
                c1, c2, c3 = st.columns([4, 1, 1])
                with c1:
                    st.markdown(f"**{display_name}**  ·  {local_time_str}  ·  Overall {ds}  ·  {fmt_dur(run.duration_seconds)}")
                with c2:
                    if st.button("View", key=f"view_{run.id}"):
                        st.session_state["view_run_id"] = run.id
                        st.rerun()
                with c3:
                    if st.button("Delete", key=f"hdel_{run.id}", type="secondary"):
                        _db.delete_run(run.id)
                        if st.session_state.get("view_run_id") == run.id:
                            del st.session_state["view_run_id"]
                        st.rerun()

            if failed:
                st.divider()
                st.subheader("Failed runs")
                for run in failed:
                    fc1, fc2, fc3 = st.columns([4, 1, 1])
                    with fc1:
                        display_name = run.title or get_clean_title_fallback(run.url)
                        local_time_str = format_local_datetime(run.created_at)
                        fstep = extract_failed_step_name(run.error_message or "")
                        st.markdown(f"❌ `{display_name}` — {local_time_str}")
                        if run.error_message:
                            st.caption(f"**Step**: `{fstep}` · {run.error_message[:100]}...")
                    with fc2:
                        if st.button("View", key=f"fview_{run.id}"):
                            st.session_state["view_run_id"] = run.id
                            st.rerun()
                    with fc3:
                        if st.button("Delete", key=f"fdel_{run.id}", type="secondary"):
                            _db.delete_run(run.id)
                            if st.session_state.get("view_run_id") == run.id:
                                del st.session_state["view_run_id"]
                            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB: Monitoring (token usage and plan status)
# ══════════════════════════════════════════════════════════════════════════════
# elif nav_selection == "Monitoring":
#     st.subheader("📊 Generation Observability & Token Metrics")
# 
#     runs = _db.list_runs(limit=1000)
#     if not runs:
#         st.info("No generation runs recorded yet. Start a new generation to see monitoring data.")
#     else:
#         from datetime import datetime
#         today_date = datetime.now().date()
# 
#         daily_tokens = {"groq": 0, "gemini": 0, "openai": 0}
#         total_tokens_used = {"groq": 0, "gemini": 0, "openai": 0}
# 
#         for r in runs:
#             provider = (r.llm_provider or "groq").lower()
#             if "gemini" in provider:
#                 prov_key = "gemini"
#             elif "openai" in provider:
#                 prov_key = "openai"
#             else:
#                 prov_key = "groq"
# 
#             toks = getattr(r, "total_tokens", 0) or 0
#             total_tokens_used[prov_key] += toks
# 
#             try:
#                 created_str = r.created_at
#                 if created_str.endswith("Z"):
#                     created_str = created_str[:-1] + "+00:00"
#                 dt = datetime.fromisoformat(created_str)
#                 local_date = dt.astimezone(None).date()
#                 if local_date == today_date:
#                     daily_tokens[prov_key] += toks
#             except Exception:
#                 pass
# 
#         # Split page layout
#         col_setup, col_gauge = st.columns([1, 1])
# 
#         with col_setup:
#             st.markdown("### ⚙️ Plan Configuration")
#             sel_prov = st.selectbox("Inspect LLM Provider Limit", ["Groq", "Gemini", "OpenAI"], index=0)
#             prov_key = sel_prov.lower()
#             plan_tier = st.selectbox("Select Your Plan Tier", ["Free Tier (Default)", "Developer Tier", "Enterprise / Custom"], index=0)
# 
#             if plan_tier == "Free Tier (Default)":
#                 limits = {"groq": 100_000, "gemini": 500_000, "openai": 200_000}
#             elif plan_tier == "Developer Tier":
#                 limits = {"groq": 500_000, "gemini": 2_000_000, "openai": 1_000_000}
#             else:
#                 custom_limit = st.number_input("Set Custom Daily Token Limit", min_value=10_000, max_value=100_000_000, value=5_000_000, step=50_000)
#                 limits = {"groq": custom_limit, "gemini": custom_limit, "openai": custom_limit}
# 
#             limit = limits[prov_key]
#             used_today = daily_tokens[prov_key]
#             remaining = max(0, limit - used_today)
#             pct_used = min(1.0, used_today / limit) if limit > 0 else 0.0
# 
#             st.write("---")
#             col_m1, col_m2 = st.columns(2)
#             with col_m1:
#                 st.metric("Used Today", f"{used_today:,}")
#                 st.metric("Total Tokens Ever", f"{total_tokens_used[prov_key]:,}")
#             with col_m2:
#                 st.metric("Remaining Today", f"{remaining:,}")
#                 st.metric("Daily Limit", f"{limit:,}")
# 
#         with col_gauge:
#             st.markdown("### 📈 Used vs Remaining")
#             st.progress(pct_used, text=f"Daily Token Quota: {pct_used*100:.1f}% used")
# 
#             donut_df = pd.DataFrame({
#                 "Metric": ["Used Today", "Remaining Today"],
#                 "Tokens": [used_today, remaining]
#             })
#             st.bar_chart(donut_df, x="Metric", y="Tokens", color="Metric", use_container_width=True)
# 
#         st.divider()
# 
#         completed_runs = [r for r in runs if r.status == "completed"]
#         if not completed_runs:
#             st.info("No successful runs available to generate performance charts.")
#         else:
#             st.markdown("### 📊 Performance Analysis & Scatter Mapping")
#             st.markdown("Every successful blog enhancement run mapped by **Time Taken** vs **Total Tokens Used**.")
# 
#             chart_rows = []
#             for r in completed_runs:
#                 chart_rows.append({
#                     "Run ID": r.id,
#                     "Article Title": r.title or (r.url.split("/")[2] if "//" in r.url else r.url[:30]),
#                     "Time Taken (s)": r.duration_seconds,
#                     "Token Usage": r.total_tokens,
#                     "Prompt Tokens": r.prompt_tokens,
#                     "Completion Tokens": r.completion_tokens,
#                     "Provider": r.llm_provider or "Groq"
#                 })
#             df_chart = pd.DataFrame(chart_rows)
# 
#             # Scatter chart: Time Taken vs Token Usage
#             st.scatter_chart(
#                 df_chart,
#                 x="Time Taken (s)",
#                 y="Token Usage",
#                 color="Provider",
#                 size="Token Usage"
#             )
# 
#             # Line chart: Token usage over history (Time Series)
#             st.write("#### 🕒 Token Usage Over Run History (Time Series)")
#             st.line_chart(df_chart, x="Run ID", y=["Token Usage", "Prompt Tokens", "Completion Tokens"])



