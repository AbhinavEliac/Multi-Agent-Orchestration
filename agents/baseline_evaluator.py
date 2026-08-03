"""
baseline_evaluator.py

Scores the original scraped blog (state.cleaned_blog) using the same
evaluator rubric before any enhancement is applied.

Runs once, immediately after prepare_blog, and writes to the
baseline_* fields in BlogState so the UI can show a before/after
comparison at the end of the run.

Uses a smaller token budget than the main evaluator (800 tokens) because
the original blog is typically shorter and less structured — less to say.
"""

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from langchain_core.prompts import ChatPromptTemplate

from config.llm_registry import make_llm
from utilis.json_parser import load_json
from utilis.prompt_loader import load_prompt
from utilis.retry import invoke_with_retry

prompt = ChatPromptTemplate.from_messages([
    ("system", load_prompt("evaluator.txt")),
    ("human", (
        "Target Length\n\n{target_length}"
        "\n\nCurrent Date\n\n{current_date}"
        "\n\nCurrent Year\n\n{current_year}"
        "\n\nLanguage Quality\n\n{language_quality}"
        "\n\nArticle\n\n{blog}"
    )),
])

# Reduced from 8 000 — original blogs are short, 5 000 chars is enough.
_BASELINE_MAX_CHARS = 5_000


from utilis.tracing import traceable


class BaselineEvaluatorAgent:
    def _score(self, result, key):
        value = result.get(key) or result.get(f"{key}_score") or result.get(key.title())
        if isinstance(value, dict):
            value = value.get("score") or value.get("Score") or value.get("value")
        return int(value or 0)

    @traceable
    def invoke(self, state):
        llm = make_llm(max_tokens=800, force_groq=True)  # baseline = Groq always

        response = invoke_with_retry(prompt, llm, {
            "blog":             state._truncate(state.cleaned_blog, _BASELINE_MAX_CHARS),
            "current_date":     state.current_date,
            "current_year":     state.current_year,
            "language_quality": state.language_quality,
            "target_length":    state.target_length,
        })

        result = load_json(response.content)

        def s(key):
            return self._score(result, key)

        state.baseline_language_score   = s("language")
        state.baseline_facts_score      = s("facts")
        state.baseline_structure_score  = s("structure")
        state.baseline_seo_score        = s("seo")
        state.baseline_geo_score        = s("geo")
        state.baseline_freshness_score  = s("freshness")

        scores = [
            state.baseline_language_score,
            state.baseline_facts_score,
            state.baseline_structure_score,
            state.baseline_seo_score,
            state.baseline_geo_score,
            state.baseline_freshness_score,
        ]
        valid = [sc for sc in scores if sc > 0]
        state.baseline_overall_score = round(sum(valid) / len(valid)) if valid else 0

        return state
