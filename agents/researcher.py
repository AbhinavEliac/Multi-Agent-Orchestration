from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from langchain_core.prompts import ChatPromptTemplate

from config.llm_registry import make_llm
from config.tavily_search import search
from graph.state import BlogState
from utilis.prompt_loader import load_prompt
from utilis.research_cache import get_researcher_cache
from utilis.retry import invoke_with_retry

prompt = ChatPromptTemplate.from_messages([
    ("system", load_prompt("researcher.txt")),
    ("human", (
        "Search Results\n\n{research}"
        "\n\nSEO Signals from Planner\n\n{seo_signals}"
        "\n\nCurrent Date\n\n{current_date}"
        "\n\nCurrent Year\n\n{current_year}"
        "\n\nTarget Length\n\n{target_length}"
        "\n\nResearch Level\n\n{research_level}"
    )),
])


from utilis.tracing import traceable


class ResearchAgent:
    _MAX_RESULT_CHARS = 400
    _MAX_TOTAL_CHARS  = 3500

    def _research_settings(self, level):
        return {
            "easy":     {"max_results": 3, "search_depth": "basic"},
            "basic":    {"max_results": 3, "search_depth": "basic"},
            "medium":   {"max_results": 5, "search_depth": "advanced"},
            "advanced": {"max_results": 8, "search_depth": "advanced"},
        }.get(str(level).lower(), {"max_results": 5, "search_depth": "advanced"})

    def _trim_results(self, results) -> str:
        items = results.get("results", []) if isinstance(results, dict) else (results if isinstance(results, list) else [])
        lines = []
        for item in items:
            content = item.get("content", "")
            if len(content) > self._MAX_RESULT_CHARS:
                content = content[:self._MAX_RESULT_CHARS].rsplit(" ", 1)[0] + "…"
            lines.append(f"- {item.get('title', '')}\n  {item.get('url', '')}\n  {content}")
        combined = "\n\n".join(lines)
        if len(combined) > self._MAX_TOTAL_CHARS:
            combined = combined[:self._MAX_TOTAL_CHARS].rsplit("\n", 1)[0] + "\n…[truncated]"
        return combined

    def _seo_signals(self, state: BlogState) -> str:
        """Extract keyword/heading signals from the planner so the researcher
        reinforces them in the brief rather than inventing its own."""
        p = state.planner_output
        if not p:
            return ""
        lines = []
        kw = p.get("primary_keyword", "")
        if kw:
            lines.append(f"Primary keyword: {kw}")
        h1 = p.get("h1_title", "")
        if h1:
            lines.append(f"H1: {h1}")
        h2s = p.get("h2_headings", [])
        if h2s:
            lines.append(f"H2 headings: {' | '.join(h2s[:6])}")
        faqs = p.get("faq_questions", [])
        if faqs:
            lines.append(f"FAQ questions: {' | '.join(faqs[:6])}")
        sec = p.get("secondary_keywords", [])
        if sec:
            lines.append(f"Secondary keywords: {', '.join(sec[:10])}")
        return "\n".join(lines)

    @traceable
    def invoke(self, state: BlogState):
        query = (
            state.planner_output.get("research_query")
            or f"{state.url} {state.current_year} latest statistics examples SEO GEO keywords"
        )
        if isinstance(query, list):
            query = " ".join(query)
        query = f"{query} {state.current_year}"

        cache = get_researcher_cache()
        cached = cache.get(query)
        if cached:
            state.research_output = cached
            return state

        results = search(query, **self._research_settings(state.research_level))
        llm = make_llm(max_tokens=2000)

        response = invoke_with_retry(prompt, llm, {
            "research":      self._trim_results(results),
            "seo_signals":   self._seo_signals(state),
            "current_date":  state.current_date,
            "current_year":  state.current_year,
            "target_length": state.target_length,
            "research_level": state.research_level,
        })
        state.research_output = response.content
        cache.set(query, state.research_output)
        return state
