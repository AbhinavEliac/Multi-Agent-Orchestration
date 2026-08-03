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

# 9 000 chars covers a full 2500-word article (~15 000 chars raw but the
# evaluator needs enough to score all sections, not every word verbatim).
# Reduced from 14 000 to keep total Groq request under 6 000 tokens.
_EVAL_MAX_CHARS = 9_000


from utilis.tracing import traceable


class EvaluatorAgent:
    def _score(self, result, key):
        value = result.get(key) or result.get(f"{key}_score") or result.get(key.title())
        if isinstance(value, dict):
            value = value.get("score") or value.get("Score") or value.get("value")
        return int(value or 0)

    def _feedback(self, result, key):
        value = (
            result.get(f"{key}_feedback")
            or result.get(f"{key}_notes")
            or result.get(f"{key}_review")
            or ""
        )
        if isinstance(value, dict):
            value = value.get("feedback") or value.get("Feedback") or ""
        return str(value)

    def _overall_score(self, state):
        scores = [state.language_score, state.facts_score, state.structure_score,
                  state.seo_score, state.geo_score, state.freshness_score]
        valid = [s for s in scores if s > 0]
        return round(sum(valid) / len(valid)) if valid else 0

    @traceable
    def invoke(self, state):
        llm = make_llm(max_tokens=1200, force_groq=True)  # evaluation = Groq always
        blog_to_eval = state._truncate(
            state.optimized_blog or state.aggregated_blog,
            max_chars=_EVAL_MAX_CHARS,
        )

        response = invoke_with_retry(prompt, llm, {
            "blog":             blog_to_eval,
            "current_date":     state.current_date,
            "current_year":     state.current_year,
            "language_quality": state.language_quality,
            "target_length":    state.target_length,
        })

        result = load_json(response.content)

        state.language_score   = self._score(result, "language")
        state.facts_score      = self._score(result, "facts")
        state.structure_score  = self._score(result, "structure")
        state.seo_score        = self._score(result, "seo")
        state.geo_score        = self._score(result, "geo")
        state.freshness_score  = self._score(result, "freshness")
        state.overall_score    = self._overall_score(state)

        state.language_feedback   = self._feedback(result, "language")
        state.facts_feedback      = self._feedback(result, "facts")
        state.structure_feedback  = self._feedback(result, "structure")
        state.seo_feedback        = self._feedback(result, "seo")
        state.geo_feedback        = self._feedback(result, "geo")
        state.freshness_feedback  = self._feedback(result, "freshness")

        state.iteration += 1
        return state
