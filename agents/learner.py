from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from langchain_core.prompts import ChatPromptTemplate

from config.llm_registry import make_llm
from graph.state import BlogState
from utilis.json_parser import load_json
from utilis.prompt_loader import load_prompt
from utilis.retry import invoke_with_retry

prompt = ChatPromptTemplate.from_messages([
    ("system", load_prompt("learner.txt")),
    ("human", "Target Length\n\n{target_length}\n\nBlog\n\n{blog}"),
])


from utilis.tracing import traceable


class LearnerAgent:
    @traceable
    def invoke(self, state: BlogState):
        llm = make_llm(max_tokens=600, force_groq=True)
        response = invoke_with_retry(prompt, llm, {
            "blog": state.blog_snippet(max_chars=2500),
            "target_length": state.target_length,
        })
        state.learner_output = load_json(response.content)
        return state
