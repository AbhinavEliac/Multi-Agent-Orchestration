from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from langchain_core.prompts import ChatPromptTemplate

from config.llm_registry import make_llm, stream_invoke
from utilis.prompt_loader import load_prompt

prompt = ChatPromptTemplate.from_messages([
    ("system", f"{load_prompt('aggregator.txt')}\n\n{load_prompt('images.txt')}"),
    ("human", (
        "Original Blog\n\n{blog}"
        "\n\nLanguage Notes\n\n{language}"
        "\n\nFacts Notes\n\n{facts}"
        "\n\nStructure Notes\n\n{structure}"
        "\n\nSEO Notes\n\n{seo}"
        "\n\nGEO Notes\n\n{geo}"
        "\n\nImages\n\n{images}"
        "\n\nDate: {current_date} | Year: {current_year}"
        "\n\nQuality: {language_quality} | Length: {target_length}"
    )),
])

# Input budget per field (chars). Total input ~10 000 chars = ~2 857 tokens.
# With aggregator.txt (~5 315 chars = ~1 519 tokens) system prompt,
# total input = ~4 376 tokens. Output capped at 3 500 = 7 876 total — under 8K.
_BLOG_CHARS      = 3_000   # original blog snippet
_BRIEF_LANG      = 800
_BRIEF_FACTS     = 1_000
_BRIEF_STRUCT    = 1_200   # structure needs room (skeleton + FAQ headings)
_BRIEF_SEO       = 1_200   # SEO needs keyword list + H2s
_BRIEF_GEO       = 600
_OUTPUT_TOKENS   = 4_500   # provides full headroom for 2800-3200 words + FAQs + table


from utilis.tracing import traceable


class AggregatorAgent:
    @traceable
    def invoke(self, state):
        llm = make_llm(max_tokens=_OUTPUT_TOKENS)

        inputs = {
            "blog":             state.blog_snippet(max_chars=_BLOG_CHARS),
            "language":         state._truncate(state.language_output, _BRIEF_LANG),
            "facts":            state._truncate(state.facts_output,    _BRIEF_FACTS),
            "structure":        state._truncate(state.structure_output, _BRIEF_STRUCT),
            "seo":              state._truncate(state.seo_output,      _BRIEF_SEO),
            "geo":              state._truncate(state.geo_output,      _BRIEF_GEO),
            "images":           state.image_output,
            "current_date":     state.current_date,
            "current_year":     state.current_year,
            "language_quality": state.language_quality,
            "target_length":    state.target_length,
        }

        full_text = stream_invoke(prompt, llm, inputs, state.stream_chunk)
        from utilis.image_formatter import ensure_exact_images_in_markdown
        state.aggregated_blog = ensure_exact_images_in_markdown(full_text, state.image_output)
        state.editorial_brief = ""
        state.stream_done()
        return state
