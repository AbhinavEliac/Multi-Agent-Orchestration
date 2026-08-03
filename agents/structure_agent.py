from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from langchain_core.prompts import ChatPromptTemplate

from config.llm_registry import make_llm
from utilis.prompt_loader import load_prompt
from utilis.retry import invoke_with_retry

prompt = ChatPromptTemplate.from_messages([
    ("system", load_prompt("structure.txt")),
    ("human", (
        "Blog\n\n{blog}"
        "\n\nStructure Research Brief\n\n{structure_research}"
        "\n\nTarget Length\n\n{target_length}"
    )),
])


from utilis.tracing import traceable


class StructureAgent:
    @traceable
    def invoke(self, state):
        llm = make_llm(max_tokens=700, force_groq=True)
        # The supervisor's structure_brief already contains the full article
        # skeleton (H1, H2s, FAQ questions, paragraph topics) aligned to the
        # SEO plan — pass it in full so the structure agent refines rather
        # than rebuilds from scratch.
        response = invoke_with_retry(prompt, llm, {
            "blog":              state.blog_snippet(max_chars=2500),
            "structure_research":state._truncate(state.structure_research, 1200),
            "target_length":     state.target_length,
        })
        state.structure_output = response.content
        return state
