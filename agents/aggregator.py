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
_BLOG_CHARS      = 1_800   # original blog snippet
_BRIEF_LANG      = 500
_BRIEF_FACTS     = 500
_BRIEF_STRUCT    = 900     # structure needs more room (skeleton + FAQ headings)
_BRIEF_SEO       = 900     # SEO needs keyword list + H2s
_BRIEF_GEO       = 400
_OUTPUT_TOKENS   = 3_500   # reduced from 4096 to keep total < 8K on Groq free tier


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
        state.aggregated_blog = full_text
        state.editorial_brief = ""
        state.stream_done()
        return state
