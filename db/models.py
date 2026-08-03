"""
models.py — plain dataclasses for database rows.
No ORM, no dependencies beyond stdlib.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Run:
    """One completed (or failed) enhancement run."""
    id:                      int
    url:                     str
    created_at:              str          # ISO-8601 UTC
    duration_seconds:        float
    llm_provider:            str
    research_level:          str
    language_quality:        str
    max_pages:               int
    image_count:             int
    optimizer_iterations:    int
    evaluation_iterations:   int

    # Baseline scores
    baseline_overall:        int
    baseline_language:       int
    baseline_facts:          int
    baseline_structure:      int
    baseline_seo:            int
    baseline_geo:            int
    baseline_freshness:      int

    # Enhanced scores
    enhanced_overall:        int
    enhanced_language:       int
    enhanced_facts:          int
    enhanced_structure:      int
    enhanced_seo:            int
    enhanced_geo:            int
    enhanced_freshness:      int

    status:                  str = "completed"   # completed | failed
    error_message:           str = ""
    title:                   str = ""
    prompt_tokens:           int = 0
    completion_tokens:       int = 0
    total_tokens:            int = 0
    parent_run_id:           Optional[int] = None
    topic_idea:              str = ""
    other_info:              str = ""
    serialized_state:        str = ""


@dataclass
class Article:
    """Full article content linked to a run."""
    run_id:           int
    original_blog:    str   # cleaned_blog — original scraped markdown
    enhanced_blog:    str   # optimized_blog or aggregated_blog
