from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from langchain_core.prompts import ChatPromptTemplate

from config.llm_registry import make_llm
from utilis.prompt_loader import load_prompt
from utilis.retry import invoke_with_retry

prompt = ChatPromptTemplate.from_messages([
    ("system", load_prompt("language.txt")),
    ("human", (
        "Blog\n\n{blog}"
        "\n\nLanguage Research Brief\n\n{language_research}"
        "\n\nLanguage Quality\n\n{language_quality}"
        "\n\nTarget Length\n\n{target_length}"
    )),
])


from utilis.tracing import traceable


class LanguageAgent:
    @traceable
    def invoke(self, state):
        llm = make_llm(max_tokens=700, force_groq=True)
        response = invoke_with_retry(prompt, llm, {
            "blog":              state.blog_snippet(max_chars=2500),
            "language_research": state._truncate(state.language_research, 800),
            "language_quality":  state.language_quality,
            "target_length":     state.target_length,
        })
        state.language_output = response.content
        return state
