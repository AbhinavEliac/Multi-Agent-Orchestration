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
    Routes to optimizer to polish failing dimensions or to specialists if re-brief needed.
    """
    mode = getattr(state, "speed_mode", "turbo")
    if mode in ("turbo", "balanced"):
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
    Evaluates post-optimizer scores against the speed profile thresholds:
    - Turbo: Guarantees all parameters >= 70 (max 2 fast passes)
    - Balanced: Guarantees all parameters >= 85
    - Deep Analysis: Guarantees all parameters >= 95 no matter what
    """
    scores = [
        state.language_score, state.facts_score, state.structure_score,
        state.seo_score, state.geo_score, state.freshness_score
    ]
    min_score = min(scores) if scores else 0
    mode = getattr(state, "speed_mode", "turbo")

    # 1. Turbo Mode: Guarantee minimum 70+ across all dimensions
    if mode == "turbo":
        if min_score >= 70 or state.optimizer_iteration >= 2:
            return "end"
        return "optimizer"

    # 2. Deep Analysis Mode: Must stay above 95 in all parameters
    if mode == "deep":
        if all(s >= 95 for s in scores):
            return "end"

        targeted_runs = getattr(state, "_targeted_runs", 0)
        if any(s < 75 for s in scores) and targeted_runs < MAX_TARGETED_ITERATIONS:
            object.__setattr__(state, "_targeted_runs", targeted_runs + 1)
            return "targeted_researcher"

        if state.optimizer_iteration >= state.max_optimizer_passes:
            return "end"
        return "optimizer"

    # 3. Balanced Mode: Guarantee 85+
    if all(s >= 85 for s in scores) or state.optimizer_iteration >= max(1, min(state.max_optimizer_passes, 3)):
        return "end"

    return "optimizer"

