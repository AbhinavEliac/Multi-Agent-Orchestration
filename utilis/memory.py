from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


MEMORY_DIR = Path(__file__).resolve().parents[1] / ".memory"
RESEARCH_CACHE_PATH = MEMORY_DIR / "research_cache.json"

BLOG_PROMPT_LIMIT = 6000
RESEARCH_PROMPT_LIMIT = 3500
MEMORY_PROMPT_LIMIT = 1800
AGENT_OUTPUT_PROMPT_LIMIT = 2200
PLAN_PROMPT_LIMIT = 1600


def _now():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def compact_text(value: Any, limit: int = 1800):
    if value is None:
        return ""

    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=True, indent=2)
        except TypeError:
            text = str(value)

    text = " ".join(text.split())

    if len(text) <= limit:
        return text

    return f"{text[:limit].rstrip()}..."


def memory_brief(state: Any, agent: str | None = None):
    agent_issues = unresolved_issues(state, agent) if agent else unresolved_issues(state)

    sections = [
        ("Context Memory", state.context_memory),
        ("Research Memory", state.research_memory),
        ("Working Memory", state.working_memory),
        ("Evaluation Memory", state.evaluation_memory[-3:]),
        ("Retry History", state.retry_history),
        ("Open Issues", agent_issues),
    ]

    lines = []

    for label, value in sections:
        brief = compact_text(value, 1200)

        if brief:
            lines.append(f"{label}: {brief}")

    return "\n".join(lines) or "No memory recorded yet."


def prompt_inputs(state: Any, agent: str | None = None):
    return {
        "blog": compact_text(state.cleaned_blog, BLOG_PROMPT_LIMIT),
        "research": compact_text(state.research_output, RESEARCH_PROMPT_LIMIT),
        "plan": compact_text(state.planner_output, PLAN_PROMPT_LIMIT),
        "memory": compact_text(memory_brief(state, agent), MEMORY_PROMPT_LIMIT),
    }


def build_context_memory(learner_output: dict[str, Any]):
    fields = [
        "title",
        "summary",
        "audience",
        "intent",
        "tone",
        "writing_style",
        "seo_intent",
        "primary_keywords",
        "secondary_keywords",
        "structure",
    ]

    return {
        key: learner_output.get(key)
        for key in fields
        if learner_output.get(key) not in (None, "", [], {})
    }


def record_working_memory(state: Any, agent: str, observations: Any):
    entries = state.working_memory.setdefault(agent, [])
    entries.append(
        {
            "iteration": state.iteration,
            "created_at": _now(),
            "observations": compact_text(observations, 1200),
        }
    )
    state.working_memory[agent] = entries[-5:]


def add_evaluation_memory(state: Any, result: dict[str, Any]):
    scores = {
        "language": state.language_score,
        "facts": state.facts_score,
        "structure": state.structure_score,
        "seo": state.seo_score,
        "geo": state.geo_score,
        "freshness": state.freshness_score,
        "overall": state.overall_score,
    }
    feedback = {
        "language": state.language_feedback,
        "facts": state.facts_feedback,
        "structure": state.structure_feedback,
        "seo": state.seo_feedback,
        "geo": state.geo_feedback,
        "freshness": state.freshness_feedback,
    }

    state.evaluation_memory.append(
        {
            "iteration": state.iteration,
            "created_at": _now(),
            "scores": scores,
            "feedback": feedback,
            "raw_result": result,
        }
    )
    state.evaluation_memory = state.evaluation_memory[-8:]


def build_issue_memory(state: Any, result: dict[str, Any]):
    issues = result.get("issues") or result.get("issue_memory") or []

    if isinstance(issues, dict):
        issues = list(issues.values())

    normalized = []

    if isinstance(issues, list):
        for index, issue in enumerate(issues, start=1):
            if not isinstance(issue, dict):
                continue

            agent = str(issue.get("agent") or issue.get("category") or "").lower()
            agent = agent.replace("_agent", "").replace(" agent", "").strip()

            if agent not in {"language", "facts", "structure", "seo", "geo", "freshness", "researcher"}:
                agent = "researcher" if agent == "research" else agent

            normalized.append(
                {
                    "id": issue.get("id") or issue.get("issue_id") or f"ISSUE-{index:03d}",
                    "agent": agent or "optimizer",
                    "severity": issue.get("severity") or "medium",
                    "location": issue.get("location") or "",
                    "problem": issue.get("problem") or issue.get("feedback") or issue.get("description") or "",
                    "status": issue.get("status") or "open",
                    "iteration": state.iteration,
                }
            )

    feedback_by_agent = {
        "language": (state.language_score, state.language_feedback),
        "facts": (state.facts_score, state.facts_feedback),
        "structure": (state.structure_score, state.structure_feedback),
        "seo": (state.seo_score, state.seo_feedback),
        "geo": (state.geo_score, state.geo_feedback),
        "researcher": (state.freshness_score, state.freshness_feedback),
    }

    existing_agents = {issue["agent"] for issue in normalized if issue.get("status") == "open"}

    for agent, (score, feedback) in feedback_by_agent.items():
        if score and score < 70 and feedback and agent not in existing_agents:
            normalized.append(
                {
                    "id": f"{agent.upper()}-{state.iteration + 1:03d}",
                    "agent": agent,
                    "severity": "high" if score < 50 else "medium",
                    "location": "",
                    "problem": feedback,
                    "status": "open",
                    "iteration": state.iteration,
                }
            )

    return normalized


def unresolved_issues(state: Any, agent: str | None = None):
    issues = [
        issue
        for issue in state.issue_memory
        if str(issue.get("status", "open")).lower() == "open"
    ]

    if agent:
        issues = [
            issue
            for issue in issues
            if str(issue.get("agent", "")).lower() == agent.lower()
        ]

    return issues


def increment_retry(state: Any, agent: str):
    state.retry_history[agent] = int(state.retry_history.get(agent, 0)) + 1


def research_cache_key(query: str):
    return hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()


def _load_research_cache():
    try:
        return json.loads(RESEARCH_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def get_cached_research(query: str):
    cache = _load_research_cache()
    return cache.get(research_cache_key(query))


def set_cached_research(query: str, value: dict[str, Any]):
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    cache = _load_research_cache()
    cache[research_cache_key(query)] = {
        **value,
        "query": query,
        "cached_at": _now(),
    }
    RESEARCH_CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
