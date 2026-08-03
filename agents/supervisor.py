"""
supervisor.py — Research Supervisor Agent

Replaces the single ResearchAgent. One Tavily search call,
one LLM call → five focused research briefs, one per specialist:

  language_brief   → state.language_research
  facts_brief      → state.facts_research
  structure_brief  → state.structure_research
  seo_brief        → state.seo_research
  geo_brief        → state.geo_research

Each specialist agent then reads its own dedicated brief instead
of having to extract relevant pieces from a shared generic document.

Also writes state.research_output with the raw search results summary
so targeted_researcher and any legacy code still works.
"""

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from langchain_core.prompts import ChatPromptTemplate

from config.llm_registry import make_llm
from config.tavily_search import search
from graph.state import BlogState
from utilis.json_parser import load_json
from utilis.prompt_loader import load_prompt
from utilis.research_cache import get_researcher_cache
from utilis.retry import invoke_with_retry

prompt = ChatPromptTemplate.from_messages([
    ("system", load_prompt("supervisor.txt")),
    ("human", (
        "Search Results\n\n{search_results}"
        "\n\nContent Plan\n\n{plan_summary}"
        "\n\nPrimary Keyword: {primary_keyword}"
        "\n\nH1 Title: {h1_title}"
        "\n\nH2 Headings: {h2_headings}"
        "\n\nFAQ Questions: {faq_questions}"
        "\n\nSecondary Keywords: {secondary_keywords}"
        "\n\nCurrent Date: {current_date}"
        "\n\nCurrent Year: {current_year}"
        "\n\nTarget Length: {target_length}"
    )),
])

_MAX_RESULT_CHARS = 400
_MAX_TOTAL_CHARS  = 4000   # slightly larger than old researcher — supervisor uses the full budget


def _trim_results(results) -> str:
    items = (
        results.get("results", []) if isinstance(results, dict)
        else results if isinstance(results, list)
        else []
    )
    lines = []
    for item in items:
        content = item.get("content", "")
        if len(content) > _MAX_RESULT_CHARS:
            content = content[:_MAX_RESULT_CHARS].rsplit(" ", 1)[0] + "…"
        lines.append(f"- {item.get('title', '')}\n  {item.get('url', '')}\n  {content}")
    combined = "\n\n".join(lines)
    if len(combined) > _MAX_TOTAL_CHARS:
        combined = combined[:_MAX_TOTAL_CHARS].rsplit("\n", 1)[0] + "\n…[truncated]"
    return combined


def _research_settings(level: str, results_override: int = 0) -> dict:
    defaults = {
        "easy":     {"max_results": 3,  "search_depth": "basic"},
        "basic":    {"max_results": 3,  "search_depth": "basic"},
        "medium":   {"max_results": 5,  "search_depth": "advanced"},
        "advanced": {"max_results": 8,  "search_depth": "advanced"},
    }.get(str(level).lower(), {"max_results": 5, "search_depth": "advanced"})
    # UI slider overrides the count but keeps the depth appropriate for the level
    if results_override > 0:
        defaults["max_results"] = results_override
    return defaults


def _seo_fields(state: BlogState) -> dict:
    """Pull the keyword plan from planner_output into flat strings."""
    p = state.planner_output
    return {
        "primary_keyword":   p.get("primary_keyword", ""),
        "h1_title":          p.get("h1_title", ""),
        "h2_headings":       " | ".join(p.get("h2_headings", [])[:6]),
        "faq_questions":     " | ".join(p.get("faq_questions", [])[:6]),
        "secondary_keywords":", ".join(p.get("secondary_keywords", [])[:10]),
    }


from utilis.tracing import traceable


class SupervisorAgent:
    @traceable
    def invoke(self, state: BlogState) -> BlogState:
        query = (
            state.planner_output.get("research_query")
            or f"{state.url} {state.current_year} latest statistics examples"
        )
        if isinstance(query, list):
            query = " ".join(query)
        query = f"{query} {state.current_year}"

        # Cache check — skip search + LLM if a similar query was seen recently
        cache = get_researcher_cache()
        cached_json = cache.get(query)
        if cached_json:
            briefs = load_json(cached_json) if isinstance(cached_json, str) else cached_json
            self._apply_briefs(state, briefs, cached_json if isinstance(cached_json, str) else "")
            return state

        raw_results = search(query, **_research_settings(state.research_level, state.research_results))
        trimmed     = _trim_results(raw_results)

        # Keep a raw summary in research_output for targeted_researcher
        state.research_output = trimmed

        llm = make_llm(max_tokens=2000, force_groq=True)  # supervisor = Groq (research synthesis)
        seo = _seo_fields(state)

        response = invoke_with_retry(prompt, llm, {
            "search_results":    trimmed,
            "plan_summary":      state.plan_summary(),
            "primary_keyword":   seo["primary_keyword"],
            "h1_title":          seo["h1_title"],
            "h2_headings":       seo["h2_headings"],
            "faq_questions":     seo["faq_questions"],
            "secondary_keywords":seo["secondary_keywords"],
            "current_date":      state.current_date,
            "current_year":      state.current_year,
            "target_length":     state.target_length,
        })

        raw_output = response.content
        briefs = load_json(raw_output)
        cache.set(query, raw_output)

        self._apply_briefs(state, briefs, raw_output)
        return state

    def _apply_briefs(self, state: BlogState, briefs: dict, raw: str) -> None:
        """Write each brief into its dedicated state field with a safe fallback."""
        state.language_research  = briefs.get("language_brief",  raw)
        state.facts_research     = briefs.get("facts_brief",     raw)
        state.structure_research = briefs.get("structure_brief", raw)
        state.seo_research       = briefs.get("seo_brief",       raw)
        state.geo_research       = briefs.get("geo_brief",       raw)
        # Also keep the raw content in research_output for backward compatibility
        if not state.research_output:
            state.research_output = raw
