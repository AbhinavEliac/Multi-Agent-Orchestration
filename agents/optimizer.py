from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from langchain_core.prompts import ChatPromptTemplate

from config.llm_registry import make_llm, stream_invoke
from utilis.prompt_loader import load_prompt

prompt = ChatPromptTemplate.from_messages([
    ("system", f"{load_prompt('optimizer.txt')}\n\n{load_prompt('images.txt')}"),
    ("human", (
        "Article Draft\n\n{current_draft}"
        "\n\nEvaluator Fix Instructions"
        "\nLanguage ({language_score}/100): {language_feedback}"
        "\nFacts ({facts_score}/100): {facts_feedback}"
        "\nStructure ({structure_score}/100): {structure_feedback}"
        "\nSEO ({seo_score}/100): {seo_feedback}"
        "\nGEO ({geo_score}/100): {geo_feedback}"
        "\nFreshness ({freshness_score}/100): {freshness_feedback}"
        "\n\nTargeted Research\n\n{targeted_research}"
        "\n\nImage Recommendations\n\n{images}"
        "\n\nCurrent Date\n\n{current_date}"
        "\n\nCurrent Year\n\n{current_year}"
        "\n\nLanguage Quality\n\n{language_quality}"
        "\n\nTarget Length\n\n{target_length}"
    )),
])


from utilis.tracing import traceable


class OptimizerAgent:
    _DRAFT_MAX_CHARS    = 25_000  # full draft context for complete rewrite
    _FEEDBACK_MAX_CHARS = 1_200   # rich feedback from evaluator
    _OUTPUT_TOKENS      = 4_500   # full output headroom for complete 2800-word article

    @traceable
    def invoke(self, state):
        llm = make_llm(max_tokens=self._OUTPUT_TOKENS)
        current_draft = state._truncate(
            state.optimized_blog or state.aggregated_blog,
            max_chars=self._DRAFT_MAX_CHARS,
        )
        state.optimizer_iteration += 1

        def _fb(text: str) -> str:
            return state._truncate(text, self._FEEDBACK_MAX_CHARS) if text else "Passing."

        inputs = {
            "current_draft":      current_draft,
            "language_score":     state.language_score,
            "language_feedback":  _fb(state.language_feedback),
            "facts_score":        state.facts_score,
            "facts_feedback":     _fb(state.facts_feedback),
            "structure_score":    state.structure_score,
            "structure_feedback": _fb(state.structure_feedback),
            "seo_score":          state.seo_score,
            "seo_feedback":       _fb(state.seo_feedback),
            "geo_score":          state.geo_score,
            "geo_feedback":       _fb(state.geo_feedback),
            "freshness_score":    state.freshness_score,
            "freshness_feedback": _fb(state.freshness_feedback),
            "targeted_research":  state._truncate(
                state.targeted_research_output, 1200
            ) if state.targeted_research_output else "None.",
            "images":             state.image_output,
            "current_date":       state.current_date,
            "current_year":       state.current_year,
            "language_quality":   state.language_quality,
            "target_length":      state.target_length,
        }

        full_text = stream_invoke(prompt, llm, inputs, state.stream_chunk)
        state.optimized_blog = full_text
        state.stream_done()
        return state
