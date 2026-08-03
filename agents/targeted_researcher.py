"""
targeted_researcher.py

Runs targeted web research for dimensions that scored below 70 after the
optimizer pass. Builds a per-dimension query, searches via Tavily, then
synthesises the results into a structured markdown brief that is stored in
state.targeted_research_output.

The optimizer will pick this up on its next pass and use it to fix the
specific failing sections.
"""

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from langchain_core.prompts import ChatPromptTemplate

from config.llm_registry import make_llm
from config.tavily_search import search
from graph.state import BlogState
from utilis.prompt_loader import load_prompt
from utilis.research_cache import get_targeted_cache
from utilis.retry import invoke_with_retry

prompt = ChatPromptTemplate.from_messages([
    ("system", load_prompt("targeted_researcher.txt")),
    ("human", (
        "Search Results\n\n{search_results}"
    )),
])

# Maps each dimension to a search query suffix that targets the exact gap
_DIMENSION_QUERY_SUFFIX = {
    "language":  "writing style examples tone engaging prose blog",
    "facts":     "latest statistics data named examples case studies verified sources",
    "structure": "blog structure outline FAQ headings word count best practices",
    "seo":       "primary keywords LSI keywords search intent FAQ questions SEO",
    "geo":       "named entities definitions direct answers authoritative statements",
    "freshness": f"latest current statistics trends updated data",
}

SCORE_THRESHOLD = 70


def _get_failing_dimensions(state: BlogState) -> list[tuple[str, int, str]]:
    """
    Returns a list of (dimension_name, score, feedback) tuples
    for every dimension that scored below SCORE_THRESHOLD.
    """
    candidates = [
        ("language",  state.language_score,  state.language_feedback),
        ("facts",     state.facts_score,     state.facts_feedback),
        ("structure", state.structure_score, state.structure_feedback),
        ("seo",       state.seo_score,       state.seo_feedback),
        ("geo",       state.geo_score,       state.geo_feedback),
        ("freshness", state.freshness_score, state.freshness_feedback),
    ]
    return [(name, score, fb) for name, score, fb in candidates if score < SCORE_THRESHOLD]


def _build_failing_brief(failing: list[tuple[str, int, str]]) -> str:
    """Formats the failing dimension info for the prompt."""
    lines = []
    for name, score, feedback in failing:
        lines.append(f"### {name.upper()} (score: {score}/100)")
        lines.append(f"Evaluator feedback: {feedback or 'No specific feedback provided.'}")
        lines.append("")
    return "\n".join(lines)


def _build_search_query(failing: list[tuple[str, int, str]], url: str, year: int) -> str:
    """
    Builds a combined Tavily search query that covers all failing dimensions.
    Pulls the topic from the URL slug and appends dimension-specific suffixes.
    """
    # Extract a readable topic from the URL
    slug = url.rstrip("/").split("/")[-1].replace("-", " ").replace("_", " ")
    topic = slug if slug else url

    suffixes = [_DIMENSION_QUERY_SUFFIX.get(name, "") for name, _, _ in failing]
    combined_suffix = " ".join(dict.fromkeys(s for s in suffixes if s))  # deduplicate

    return f"{topic} {combined_suffix} {year} latest"


from utilis.tracing import traceable


class TargetedResearchAgent:
    _MAX_RESULT_CHARS = 400
    _MAX_TOTAL_CHARS  = 3500

    def _research_settings(self, level: str) -> dict:
        settings = {
            "easy":     {"max_results": 3, "search_depth": "basic"},
            "basic":    {"max_results": 3, "search_depth": "basic"},
            "medium":   {"max_results": 5, "search_depth": "advanced"},
            "advanced": {"max_results": 8, "search_depth": "advanced"},
        }
        return settings.get(str(level).lower(), settings["medium"])

    def _trim_results(self, results) -> str:
        if isinstance(results, dict):
            items = results.get("results", [])
        elif isinstance(results, list):
            items = results
        else:
            return str(results)[:self._MAX_TOTAL_CHARS]

        lines = []
        for item in items:
            title   = item.get("title", "")
            url     = item.get("url", "")
            content = item.get("content", "")
            if len(content) > self._MAX_RESULT_CHARS:
                content = content[:self._MAX_RESULT_CHARS].rsplit(" ", 1)[0] + "…"
            lines.append(f"- {title}\n  {url}\n  {content}")

        combined = "\n\n".join(lines)
        if len(combined) > self._MAX_TOTAL_CHARS:
            combined = combined[:self._MAX_TOTAL_CHARS].rsplit("\n", 1)[0] + "\n…[truncated]"
        return combined

    @traceable
    def invoke(self, state: BlogState) -> BlogState:
        failing = _get_failing_dimensions(state)

        if not failing:
            # Nothing to do — scores all ≥ threshold, router shouldn't have
            # sent us here, but be safe.
            return state

        failing_names = [name for name, _, _ in failing]
        failing_brief = _build_failing_brief(failing)
        query = _build_search_query(failing, state.url, state.current_year)

        # ── Cache check ──────────────────────────────────────────────────────
        cache = get_targeted_cache()
        cached = cache.get(query)
        if cached:
            state.targeted_research_output = (
                f"## Targeted Research for: {', '.join(failing_names)}\n\n"
                + cached
            )
            return state

        # Run the search
        search_results = search(query, **self._research_settings(state.research_level))

        # Synthesise targeted research brief via LLM
        llm = make_llm(size="medium", force_groq=True)  # targeted research = Groq always
        response = invoke_with_retry(prompt, llm, {
            "failing_dimensions_brief": failing_brief,
            "url":          state.url,
            "current_date": state.current_date,
            "current_year": state.current_year,
            "search_results": self._trim_results(search_results),
        })

        result_text = response.content
        cache.set(query, result_text)

        # Store in dedicated field so it doesn't overwrite the original research
        state.targeted_research_output = (
            f"## Targeted Research for: {', '.join(failing_names)}\n\n"
            + result_text
        )

        return state
