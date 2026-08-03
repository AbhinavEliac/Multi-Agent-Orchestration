"""
database.py — SQLite persistence layer for Blog Enhancer.

Single file, zero external dependencies beyond stdlib.
DB file: blog/data/blog_enhancer.db  (created automatically on first use).

Schema
──────
runs      — one row per enhancement run (metadata + scores)
articles  — one row per run (full text content, joined on run_id)

Thread safety: SQLite WAL mode + per-call connections.
Each public method opens, uses, and closes its own connection so it is
safe to call from Streamlit's background thread without a shared lock.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from db.models import Article, Run

# DB lives in blog/data/ so it survives git clean of source files
_DB_DIR  = Path(__file__).resolve().parents[1] / "data"
_DB_PATH = _DB_DIR / "blog_enhancer.db"

_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS runs (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    url                   TEXT    NOT NULL,
    title                 TEXT    NOT NULL DEFAULT '',
    created_at            TEXT    NOT NULL,
    duration_seconds      REAL    NOT NULL DEFAULT 0,
    llm_provider          TEXT    NOT NULL DEFAULT '',
    research_level        TEXT    NOT NULL DEFAULT '',
    language_quality      TEXT    NOT NULL DEFAULT '',
    max_pages             INTEGER NOT NULL DEFAULT 5,
    image_count           INTEGER NOT NULL DEFAULT 3,
    optimizer_iterations  INTEGER NOT NULL DEFAULT 0,
    evaluation_iterations INTEGER NOT NULL DEFAULT 0,

    baseline_overall      INTEGER NOT NULL DEFAULT 0,
    baseline_language     INTEGER NOT NULL DEFAULT 0,
    baseline_facts        INTEGER NOT NULL DEFAULT 0,
    baseline_structure    INTEGER NOT NULL DEFAULT 0,
    baseline_seo          INTEGER NOT NULL DEFAULT 0,
    baseline_geo          INTEGER NOT NULL DEFAULT 0,
    baseline_freshness    INTEGER NOT NULL DEFAULT 0,

    enhanced_overall      INTEGER NOT NULL DEFAULT 0,
    enhanced_language     INTEGER NOT NULL DEFAULT 0,
    enhanced_facts        INTEGER NOT NULL DEFAULT 0,
    enhanced_structure    INTEGER NOT NULL DEFAULT 0,
    enhanced_seo          INTEGER NOT NULL DEFAULT 0,
    enhanced_geo          INTEGER NOT NULL DEFAULT 0,
    enhanced_freshness    INTEGER NOT NULL DEFAULT 0,

    status                TEXT    NOT NULL DEFAULT 'completed',
    error_message         TEXT    NOT NULL DEFAULT '',
    prompt_tokens         INTEGER NOT NULL DEFAULT 0,
    completion_tokens     INTEGER NOT NULL DEFAULT 0,
    total_tokens          INTEGER NOT NULL DEFAULT 0,
    parent_run_id         INTEGER REFERENCES runs(id) ON DELETE SET NULL,
    topic_idea            TEXT    NOT NULL DEFAULT '',
    other_info            TEXT    NOT NULL DEFAULT '',
    serialized_state      TEXT    NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS articles (
    run_id         INTEGER PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
    original_blog  TEXT NOT NULL DEFAULT '',
    enhanced_blog  TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS jobs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER,
    url           TEXT    NOT NULL,
    title         TEXT    NOT NULL DEFAULT '',
    status        TEXT    NOT NULL DEFAULT 'running',
    active_agent  TEXT    NOT NULL DEFAULT '',
    progress_pct  INTEGER NOT NULL DEFAULT 0,
    streamed_text TEXT    NOT NULL DEFAULT '',
    error_message TEXT    NOT NULL DEFAULT '',
    started_at    TEXT    NOT NULL,
    updated_at    TEXT    NOT NULL,
    cancel_requested INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
"""


def _connect() -> sqlite3.Connection:
    _DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH, timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_schema() -> None:
    with _connect() as conn:
        conn.executescript(_DDL)
        try:
            conn.execute("SELECT title FROM runs LIMIT 1")
        except sqlite3.OperationalError:
            conn.execute("ALTER TABLE runs ADD COLUMN title TEXT NOT NULL DEFAULT ''")
            conn.commit()
        try:
            conn.execute("SELECT title FROM jobs LIMIT 1")
        except sqlite3.OperationalError:
            conn.execute("ALTER TABLE jobs ADD COLUMN title TEXT NOT NULL DEFAULT ''")
            conn.commit()
        try:
            conn.execute("SELECT prompt_tokens FROM runs LIMIT 1")
        except sqlite3.OperationalError:
            conn.execute("ALTER TABLE runs ADD COLUMN prompt_tokens INTEGER NOT NULL DEFAULT 0")
            conn.execute("ALTER TABLE runs ADD COLUMN completion_tokens INTEGER NOT NULL DEFAULT 0")
            conn.execute("ALTER TABLE runs ADD COLUMN total_tokens INTEGER NOT NULL DEFAULT 0")
            conn.commit()
        try:
            conn.execute("SELECT parent_run_id FROM runs LIMIT 1")
        except sqlite3.OperationalError:
            conn.execute("ALTER TABLE runs ADD COLUMN parent_run_id INTEGER REFERENCES runs(id) ON DELETE SET NULL")
            conn.commit()
        try:
            conn.execute("SELECT topic_idea FROM runs LIMIT 1")
        except sqlite3.OperationalError:
            conn.execute("ALTER TABLE runs ADD COLUMN topic_idea TEXT NOT NULL DEFAULT ''")
            conn.execute("ALTER TABLE runs ADD COLUMN other_info TEXT NOT NULL DEFAULT ''")
            conn.commit()
        try:
            conn.execute("SELECT serialized_state FROM runs LIMIT 1")
        except sqlite3.OperationalError:
            conn.execute("ALTER TABLE runs ADD COLUMN serialized_state TEXT NOT NULL DEFAULT ''")
            conn.commit()


# Ensure schema exists when module is imported
_ensure_schema()


class BlogDatabase:
    """Public API for all DB operations."""

    # ── Write ────────────────────────────────────────────────────────────────

    def save_run(
        self,
        *,
        url: str,
        title: str = "",
        duration_seconds: float,
        llm_provider: str,
        research_level: str,
        language_quality: str,
        max_pages: int,
        image_count: int,
        optimizer_iterations: int,
        evaluation_iterations: int,
        baseline_overall: int,
        baseline_language: int,
        baseline_facts: int,
        baseline_structure: int,
        baseline_seo: int,
        baseline_geo: int,
        baseline_freshness: int,
        enhanced_overall: int,
        enhanced_language: int,
        enhanced_facts: int,
        enhanced_structure: int,
        enhanced_seo: int,
        enhanced_geo: int,
        enhanced_freshness: int,
        original_blog: str,
        enhanced_blog: str,
        status: str = "completed",
        error_message: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        parent_run_id: int | None = None,
        topic_idea: str = "",
        other_info: str = "",
        serialized_state: str = "",
    ) -> int:
        """Insert a completed run + its article content. Returns the new run_id."""
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with _connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO runs (
                    url, title, created_at, duration_seconds,
                    llm_provider, research_level, language_quality,
                    max_pages, image_count,
                    optimizer_iterations, evaluation_iterations,
                    baseline_overall, baseline_language, baseline_facts,
                    baseline_structure, baseline_seo, baseline_geo, baseline_freshness,
                    enhanced_overall, enhanced_language, enhanced_facts,
                    enhanced_structure, enhanced_seo, enhanced_geo, enhanced_freshness,
                    status, error_message,
                    prompt_tokens, completion_tokens, total_tokens, parent_run_id,
                    topic_idea, other_info, serialized_state
                ) VALUES (
                    :url, :title, :created_at, :duration_seconds,
                    :llm_provider, :research_level, :language_quality,
                    :max_pages, :image_count,
                    :optimizer_iterations, :evaluation_iterations,
                    :baseline_overall, :baseline_language, :baseline_facts,
                    :baseline_structure, :baseline_seo, :baseline_geo, :baseline_freshness,
                    :enhanced_overall, :enhanced_language, :enhanced_facts,
                    :enhanced_structure, :enhanced_seo, :enhanced_geo, :enhanced_freshness,
                    :status, :error_message,
                    :prompt_tokens, :completion_tokens, :total_tokens, :parent_run_id,
                    :topic_idea, :other_info, :serialized_state
                )
                """,
                {
                    "url": url,
                    "title": title,
                    "created_at": created_at,
                    "duration_seconds": round(duration_seconds, 1),
                    "llm_provider": llm_provider,
                    "research_level": research_level,
                    "language_quality": language_quality,
                    "max_pages": max_pages,
                    "image_count": image_count,
                    "optimizer_iterations": optimizer_iterations,
                    "evaluation_iterations": evaluation_iterations,
                    "baseline_overall": baseline_overall,
                    "baseline_language": baseline_language,
                    "baseline_facts": baseline_facts,
                    "baseline_structure": baseline_structure,
                    "baseline_seo": baseline_seo,
                    "baseline_geo": baseline_geo,
                    "baseline_freshness": baseline_freshness,
                    "enhanced_overall": enhanced_overall,
                    "enhanced_language": enhanced_language,
                    "enhanced_facts": enhanced_facts,
                    "enhanced_structure": enhanced_structure,
                    "enhanced_seo": enhanced_seo,
                    "enhanced_geo": enhanced_geo,
                    "enhanced_freshness": enhanced_freshness,
                    "status": status,
                    "error_message": error_message,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "parent_run_id": parent_run_id,
                    "topic_idea": topic_idea,
                    "other_info": other_info,
                    "serialized_state": serialized_state,
                },
            )
            run_id = cur.lastrowid
            conn.execute(
                "INSERT INTO articles (run_id, original_blog, enhanced_blog) VALUES (?, ?, ?)",
                (run_id, original_blog, enhanced_blog),
            )
            conn.commit()
        return run_id

    def save_failed_run(
        self,
        *,
        url: str,
        error_message: str,
        title: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        optimizer_iterations: int = 0,
        evaluation_iterations: int = 0,
        llm_provider: str = "",
        research_level: str = "",
        language_quality: str = "",
        max_pages: int = 5,
        image_count: int = 3,
        parent_run_id: int | None = None,
        topic_idea: str = "",
        other_info: str = "",
        serialized_state: str = "",
    ) -> int:
        """Record a run that crashed before completion."""
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with _connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO runs (
                    url, title, created_at, status, error_message,
                    prompt_tokens, completion_tokens, total_tokens,
                    optimizer_iterations, evaluation_iterations,
                    llm_provider, research_level, language_quality,
                    max_pages, image_count, parent_run_id,
                    topic_idea, other_info, serialized_state
                )
                VALUES (?, ?, ?, 'failed', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    url, title, created_at, str(error_message)[:2000],
                    prompt_tokens, completion_tokens, total_tokens,
                    optimizer_iterations, evaluation_iterations,
                    llm_provider, research_level, language_quality,
                    max_pages, image_count, parent_run_id,
                    topic_idea, other_info, serialized_state
                ),
            )
            run_id = cur.lastrowid
            conn.execute(
                "INSERT INTO articles (run_id, original_blog, enhanced_blog) VALUES (?, '', '')",
                (run_id,),
            )
            conn.commit()
        return run_id

    # ── Read ─────────────────────────────────────────────────────────────────

    def list_runs(self, limit: int = 100) -> list[Run]:
        """Return recent runs, newest first."""
        with _connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_run(r) for r in rows]

    def get_run(self, run_id: int) -> Optional[Run]:
        with _connect() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
        return self._row_to_run(row) if row else None

    def get_article(self, run_id: int) -> Optional[Article]:
        with _connect() as conn:
            row = conn.execute(
                "SELECT * FROM articles WHERE run_id = ?", (run_id,)
            ).fetchone()
        if not row:
            return None
        return Article(
            run_id=row["run_id"],
            original_blog=row["original_blog"],
            enhanced_blog=row["enhanced_blog"],
        )

    def delete_run(self, run_id: int) -> None:
        """Delete a run and its article (CASCADE handles articles)."""
        with _connect() as conn:
            conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
            conn.commit()

    # ── Jobs (live progress tracking) ────────────────────────────────────────

    def create_job(self, url: str, title: str = "") -> int:
        """Create a new in-progress job. Returns job_id."""
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with _connect() as conn:
            cur = conn.execute(
                "INSERT INTO jobs (url, title, status, started_at, updated_at) VALUES (?, ?, 'running', ?, ?)",
                (url, title, now, now),
            )
            conn.commit()
            return cur.lastrowid

    def update_job(self, job_id: int, *, active_agent: str = "",
                   progress_pct: int = 0, streamed_text: str = "",
                   title: str = None) -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with _connect() as conn:
            if title is not None:
                conn.execute(
                    """UPDATE jobs SET active_agent=?, progress_pct=?,
                       streamed_text=?, updated_at=?, title=? WHERE id=?""",
                    (active_agent, progress_pct,
                     streamed_text[-12_000:],   # keep last ~12 000 chars of stream
                     now, title, job_id),
                )
            else:
                conn.execute(
                    """UPDATE jobs SET active_agent=?, progress_pct=?,
                       streamed_text=?, updated_at=? WHERE id=?""",
                    (active_agent, progress_pct,
                     streamed_text[-12_000:],   # keep last ~12 000 chars of stream
                     now, job_id),
                )
            conn.commit()

    def finish_job(self, job_id: int, run_id: Optional[int] = None,
                   error: str = "") -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        status = "failed" if error else "completed"
        with _connect() as conn:
            conn.execute(
                """UPDATE jobs SET status=?, run_id=?, error_message=?,
                   progress_pct=100, updated_at=? WHERE id=?""",
                (status, run_id, error[:2000], now, job_id),
            )
            conn.commit()

    def cancel_job(self, job_id: int) -> None:
        with _connect() as conn:
            conn.execute(
                "UPDATE jobs SET cancel_requested=1 WHERE id=?", (job_id,)
            )
            conn.commit()

    def is_cancel_requested(self, job_id: int) -> bool:
        with _connect() as conn:
            row = conn.execute(
                "SELECT cancel_requested FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
        return bool(row and row["cancel_requested"])

    def get_job(self, job_id: int):
        with _connect() as conn:
            return conn.execute(
                "SELECT * FROM jobs WHERE id=?", (job_id,)
            ).fetchone()

    def list_active_jobs(self):
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        with _connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE status='running' ORDER BY started_at DESC"
            ).fetchall()
            
            active = []
            for r in rows:
                try:
                    updated = datetime.fromisoformat(r["updated_at"])
                    if updated.tzinfo is None:
                        updated = updated.replace(tzinfo=timezone.utc)
                    else:
                        updated = updated.astimezone(timezone.utc)
                    
                    if (now - updated).total_seconds() > 300:
                        # Stale job — mark as failed
                        conn.execute(
                            "UPDATE jobs SET status='failed', error_message='Job became stale (thread died or process restarted)' WHERE id=?",
                            (r["id"],)
                        )
                        conn.commit()
                        self.save_failed_run(url=r["url"], error_message="Job became stale (thread died or process restarted)")
                    else:
                        active.append(r)
                except Exception:
                    active.append(r)
            return active

    # ── Internal ─────────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_run(row: sqlite3.Row) -> Run:
        return Run(
            id=row["id"],
            url=row["url"],
            title=row["title"] if "title" in row.keys() else "",
            created_at=row["created_at"],
            duration_seconds=row["duration_seconds"],
            llm_provider=row["llm_provider"],
            research_level=row["research_level"],
            language_quality=row["language_quality"],
            max_pages=row["max_pages"],
            image_count=row["image_count"],
            optimizer_iterations=row["optimizer_iterations"],
            evaluation_iterations=row["evaluation_iterations"],
            baseline_overall=row["baseline_overall"],
            baseline_language=row["baseline_language"],
            baseline_facts=row["baseline_facts"],
            baseline_structure=row["baseline_structure"],
            baseline_seo=row["baseline_seo"],
            baseline_geo=row["baseline_geo"],
            baseline_freshness=row["baseline_freshness"],
            enhanced_overall=row["enhanced_overall"],
            enhanced_language=row["enhanced_language"],
            enhanced_facts=row["enhanced_facts"],
            enhanced_structure=row["enhanced_structure"],
            enhanced_seo=row["enhanced_seo"],
            enhanced_geo=row["enhanced_geo"],
            enhanced_freshness=row["enhanced_freshness"],
            status=row["status"],
            error_message=row["error_message"],
            prompt_tokens=row["prompt_tokens"] if "prompt_tokens" in row.keys() else 0,
            completion_tokens=row["completion_tokens"] if "completion_tokens" in row.keys() else 0,
            total_tokens=row["total_tokens"] if "total_tokens" in row.keys() else 0,
            parent_run_id=row["parent_run_id"] if "parent_run_id" in row.keys() else None,
            topic_idea=row["topic_idea"] if "topic_idea" in row.keys() else "",
            other_info=row["other_info"] if "other_info" in row.keys() else "",
            serialized_state=row["serialized_state"] if "serialized_state" in row.keys() else "",
        )
