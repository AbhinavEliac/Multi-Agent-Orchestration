from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from langchain_core.prompts import ChatPromptTemplate

from config.llm_registry import make_llm
from utilis.prompt_loader import load_prompt
from utilis.retry import invoke_with_retry

prompt = ChatPromptTemplate.from_messages([
    ("system", load_prompt("geo.txt")),
    ("human", (
        "SEO Output\n\n{seo}"
        "\n\nGEO Research Brief\n\n{geo_research}"
        "\n\nCurrent Date\n\n{current_date}"
        "\n\nCurrent Year\n\n{current_year}"
        "\n\nTarget Length\n\n{target_length}"
    )),
])


from utilis.tracing import traceable


class GeoAgent:
    @traceable
    def invoke(self, state):
        llm = make_llm(max_tokens=700, force_groq=True)
        response = invoke_with_retry(prompt, llm, {
            "seo":          state._truncate(state.seo_output, 900),
            "geo_research": state._truncate(state.geo_research, 1000),
            "current_date": state.current_date,
            "current_year": state.current_year,
            "target_length":state.target_length,
        })
        state.geo_output = response.content
        return state
