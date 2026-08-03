from __future__ import annotations

import queue
from datetime import date
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field


class BlogState(BaseModel):

    url: str

    title: str = ""

    raw_blog: str = ""

    cleaned_blog: str = ""

    chunks: list[str] = Field(default_factory=list)

    retrieved_context: str = ""

    target_length: str = "approximately 5 web pages, or 2000-3000 words"

    max_pages: int = 5

    language_quality: str = "medium"

    research_level: str = "medium"

    # Number of Tavily search results to fetch (overrides research_level default).
    # Set from UI slider. 0 = use research_level default.
    research_results: int = 0

    image_count: int = 3

    current_date: str = Field(default_factory=lambda: date.today().isoformat())

    current_year: int = Field(default_factory=lambda: date.today().year)

    learner_output: dict = Field(default_factory=dict)

    planner_output: dict = Field(default_factory=dict)

    research_output: str = ""

    # ── Baseline scores (original blog, evaluated before any enhancement) ──
    baseline_language_score:   int = 0
    baseline_facts_score:      int = 0
    baseline_structure_score:  int = 0
    baseline_seo_score:        int = 0
    baseline_geo_score:        int = 0
    baseline_freshness_score:  int = 0
    baseline_overall_score:    int = 0

    # Per-agent focused research briefs produced by the Supervisor.
    # Each specialist reads its own brief instead of the shared generic output.
    language_research: str = ""
    facts_research: str = ""
    structure_research: str = ""
    seo_research: str = ""
    geo_research: str = ""

    language_output: str = ""

    facts_output: str = ""

    structure_output: str = ""

    seo_output: str = ""

    geo_output: str = ""

    image_output: list[dict] = Field(default_factory=list)

    aggregated_blog: str = ""

    editorial_brief: str = ""

    optimized_blog: str = ""

    targeted_research_output: str = ""

    language_score: int = 0

    facts_score: int = 0

    structure_score: int = 0

    seo_score: int = 0

    geo_score: int = 0

    freshness_score: int = 0

    overall_score: int = 0

    language_feedback: str = ""

    facts_feedback: str = ""

    structure_feedback: str = ""

    seo_feedback: str = ""

    geo_feedback: str = ""

    freshness_feedback: str = ""

    iteration: int = 0

    optimizer_iteration: int = 0

    finished: bool = False

    # User-configurable optimizer pass cap (set from UI slider).
    # Default 3 — good quality without burning quota.
    # Range: 1 (fast/cheap) to 8 (maximum quality).
    max_optimizer_passes: int = 3

    # LLM provider: "groq" or "openai" — set from the UI before graph.invoke()
    llm_provider: str = "groq"

    # Pathway & custom settings fields
    mode: str = "enhance"
    topic_idea: str = ""
    other_info: str = ""
    custom_model_name: str = ""
    custom_api_key: str = ""
    custom_base_url: str = ""
    job_id: int = 0
    parent_run_id: Optional[int] = None
    resume_node: Optional[str] = None

    # Active agent name written by each node so the UI can display live status
    active_agent: str = ""

    # ── Streaming ────────────────────────────────────────────────────────────
    # Queue populated by aggregator/optimizer during streaming LLM output.
    # The UI polls this every 0.15 s and appends chunks to the live preview.
    # None when streaming is not in use (CLI / non-Streamlit runs).
    stream_queue: Optional[Any] = Field(default=None, exclude=True)

    # Callback function to notify the monitoring thread/UI of active agent transitions
    active_agent_callback: Optional[Callable[[str], None]] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def stream_chunk(self, token: str) -> None:
        """Push one token/chunk into the stream queue (no-op if queue is None)."""
        if getattr(self, "job_id", 0):
            from db.database import BlogDatabase
            if BlogDatabase().is_cancel_requested(self.job_id):
                raise RuntimeError("Cancelled by user.")
        if self.stream_queue is not None:
            try:
                self.stream_queue.put_nowait(token)
            except Exception:
                pass

    def stream_done(self) -> None:
        """Signal to the UI that streaming is finished for this agent pass."""
        if self.stream_queue is not None:
            try:
                self.stream_queue.put_nowait(None)  # None = sentinel
            except Exception:
                pass

    def plan_summary(self) -> str:
        """Compact text summary of planner_output for agent prompts."""
        p = self.planner_output
        if not p:
            return ""

        parts: list[str] = []

        def _list(label: str, key: str) -> None:
            val = p.get(key)
            if val and isinstance(val, list):
                parts.append(f"{label}: {'; '.join(str(v) for v in val[:8])}")
            elif val and isinstance(val, str):
                parts.append(f"{label}: {val}")

        _list("Missing topics",   "missing_topics")
        _list("Missing keywords", "missing_keywords")
        _list("Missing entities", "missing_entities")
        _list("Missing FAQs",     "missing_faqs")
        _list("Missing stats",    "missing_statistics")
        _list("Missing examples", "missing_examples")
        _list("Weak sections",    "weak_sections")
        _list("SEO plan",         "seo_plan")
        _list("GEO plan",         "geo_plan")

        outline = p.get("target_outline")
        if outline and isinstance(outline, list):
            parts.append("Target outline: " + " > ".join(str(h) for h in outline[:10]))

        return "\n".join(parts)

    def research_snippet(self, max_chars: int = 1200) -> str:
        text = self.research_output
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rsplit("\n", 1)[0] + "\n…[truncated]"

    def blog_snippet(self, max_chars: int = 2500) -> str:
        if getattr(self, "mode", "enhance") == "generate":
            return "(No existing blog; we are generating a brand new blog from scratch. Plan and outline content from the provided topic and guidelines.)"
        text = self.cleaned_blog
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rsplit("\n", 1)[0] + "\n…[truncated for brevity]"

    def _truncate(self, text: str, max_chars: int) -> str:
        if not text or len(text) <= max_chars:
            return text or ""
        return text[:max_chars].rsplit("\n", 1)[0] + "\n…[truncated]"

    def specialist_outputs_brief(self, max_chars_each: int = 700) -> dict:
        """
        Truncated specialist outputs for the aggregator.
        Structure and SEO get a larger window so the aggregator receives
        the full keyword list, H2 headings, FAQ questions, and skeleton.
        """
        return {
            "language": self._truncate(self.language_output, max_chars_each),
            "facts":    self._truncate(self.facts_output,    max_chars_each),
            "structure":self._truncate(self.structure_output, 1400),
            "seo":      self._truncate(self.seo_output,      1400),
            "geo":      self._truncate(self.geo_output,      max_chars_each),
        }
