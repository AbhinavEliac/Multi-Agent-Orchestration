from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from langchain_core.prompts import ChatPromptTemplate

from config.llm_registry import make_llm
from utilis.prompt_loader import load_prompt
from utilis.retry import invoke_with_retry

prompt = ChatPromptTemplate.from_messages([
    ("system", load_prompt("facts.txt")),
    ("human", (
        "Blog\n\n{blog}"
        "\n\nFacts Research Brief\n\n{facts_research}"
        "\n\nCurrent Date\n\n{current_date}"
        "\n\nCurrent Year\n\n{current_year}"
        "\n\nTarget Length\n\n{target_length}"
    )),
])


from utilis.tracing import traceable


class FactsAgent:
    @traceable
    def invoke(self, state):
        llm = make_llm(max_tokens=700, force_groq=True)
        result = invoke_with_retry(prompt, llm, {
            "blog":           state.blog_snippet(max_chars=2500),
            "facts_research": state._truncate(state.facts_research, 1000),
            "current_date":   state.current_date,
            "current_year":   state.current_year,
            "target_length":  state.target_length,
        })
        state.facts_output = result.content
        return state
