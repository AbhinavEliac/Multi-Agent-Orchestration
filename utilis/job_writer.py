"""
job_writer.py — writes live generation progress to SQLite.

The background thread calls JobWriter.tick() every ~0.5 s and
JobWriter.append_stream() on every token. The UI reads from
the jobs table via BlogDatabase so it survives page refreshes.
"""
from __future__ import annotations

import threading
import time
from typing import Optional


class JobWriter:
    """
    Thread-safe progress writer for a single generation job.

    Usage:
        jw = JobWriter(db, job_id)
        jw.start()                       # starts background flush thread
        jw.set_agent("supervisor", 17)   # update active agent + progress %
        jw.append_stream("token")        # accumulate streamed text
        jw.stop(run_id=42)               # mark complete, stop flush thread
        jw.stop(error="something went wrong")  # mark failed
    """

    _FLUSH_INTERVAL = 0.8   # seconds between DB writes

    def __init__(self, db, job_id: int):
        self._db       = db
        self._job_id   = job_id
        self._agent    = ""
        self._pct      = 0
        self._stream   = ""
        self._title    = ""
        self._lock     = threading.Lock()
        self._running  = False
        self._thread: Optional[threading.Thread] = None

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def job_id(self) -> int:
        return self._job_id

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._thread.start()

    def set_agent(self, agent: str, pct: int) -> None:
        with self._lock:
            self._agent = agent
            self._pct   = pct

    def set_title(self, title: str) -> None:
        with self._lock:
            self._title = title

    def append_stream(self, token: str) -> None:
        with self._lock:
            self._stream += token

    def is_cancel_requested(self) -> bool:
        return self._db.is_cancel_requested(self._job_id)

    def stop(self, run_id: Optional[int] = None, error: str = "") -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        # Final flush
        with self._lock:
            agent, pct, stream, title = self._agent, self._pct, self._stream, self._title
        self._db.update_job(self._job_id,
                            active_agent=agent,
                            progress_pct=pct,
                            streamed_text=stream,
                            title=title)
        self._db.finish_job(self._job_id, run_id=run_id, error=error)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _flush_loop(self) -> None:
        while self._running:
            with self._lock:
                agent, pct, stream, title = self._agent, self._pct, self._stream, self._title
            try:
                self._db.update_job(self._job_id,
                                    active_agent=agent,
                                    progress_pct=pct,
                                    streamed_text=stream,
                                    title=title)
            except Exception:
                pass
            time.sleep(self._FLUSH_INTERVAL)
