"""
router.py — routing functions for the LangGraph workflow.

Two routers:

1. evaluation_router  (after evaluator, first pass)
   Routes back to individual specialist agents when a dimension scores <70,
   so the specialist can redo its brief before aggregation.
   Once all dimensions are ≥70, routes to optimizer.
   Cap: MAX_SPECIALIST_ITERATIONS (3) — prevents infinite specialist loops.

2. optimizer_router  (after optimizer)
   The optimizer has written/rewritten the article.
   Evaluator re-scores it.
   - If any dimension <70 → targeted_researcher (max MAX_TARGETED_ITERATIONS times)
   - If any dimension <90 and passes remaining → optimizer (polish pass)
   - All ≥90 or cap reached → end
   Cap: MAX_OPTIMIZER_ITERATIONS (8) — more passes = meaningfully better output.
"""

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from graph.state import BlogState

# Pre-optimizer specialist re-brief loop.
# 3 is enough — if a specialist is still failing after 3 re-runs with the
# same inputs, more loops won't fix it; the optimizer will handle it.
MAX_SPECIALIST_ITERATIONS = 3

# Optimizer → evaluator_post polish loop.
# 8 passes gives the optimizer real room to reach 90+ across all dimensions
# without burning through quota on every run.
MAX_OPTIMIZER_ITERATIONS = 8

# Maximum number of targeted_researcher → optimizer cycles.
# Each brings fresh Tavily data, so 3 is genuinely useful.
MAX_TARGETED_ITERATIONS = 3


def evaluation_router(state: BlogState) -> str:
    """
    Called after the FIRST evaluator run (post-aggregator).
    Sends failing dimensions back to their specialist for a revised brief.
    Once all dimensions ≥70, proceed to optimizer.
    """
    if getattr(state, "speed_mode", "turbo") == "turbo":
        return "optimizer"

    if state.iteration >= MAX_SPECIALIST_ITERATIONS:
        return "optimizer"

    if state.freshness_score < 70:
        return "researcher"

    if state.language_score < 70:
        return "language"

    if state.facts_score < 70:
        return "facts"

    if state.structure_score < 70:
        return "structure"

    if state.seo_score < 70:
        return "seo"

    if state.geo_score < 70:
        return "geo"

    return "optimizer"


def optimizer_router(state: BlogState) -> str:
    """
    Called after the optimizer has written/rewritten the article.

    Priority:
    1. In turbo mode → end immediately (fastest <45s)
    2. Any score <70 AND targeted budget remaining → targeted_researcher
    3. All ≥90 → end
    4. Any <90 AND optimizer budget remaining → optimizer (polish pass)
    5. Budget exhausted → end
    """
    if getattr(state, "speed_mode", "turbo") == "turbo":
        return "end"

    needs_research = (
        state.language_score  < 70
        or state.facts_score     < 70
        or state.structure_score < 70
        or state.seo_score       < 70
        or state.geo_score       < 70
        or state.freshness_score < 70
    )

    targeted_runs = getattr(state, "_targeted_runs", 0)

    if needs_research and targeted_runs < MAX_TARGETED_ITERATIONS:
        object.__setattr__(state, "_targeted_runs", targeted_runs + 1)
        return "targeted_researcher"

    all_pass = (
        state.language_score  >= 90
        and state.facts_score     >= 90
        and state.structure_score >= 90
        and state.seo_score       >= 90
        and state.geo_score       >= 90
        and state.freshness_score >= 90
    )

    if all_pass:
        return "end"

    max_passes = 1 if getattr(state, "speed_mode", "turbo") == "balanced" else state.max_optimizer_passes
    if state.optimizer_iteration >= max_passes:
        return "end"

    return "optimizer"

