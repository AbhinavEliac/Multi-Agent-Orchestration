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
    ("system", load_prompt("planner.txt")),
    ("human", "Article\n\n{blog}\n\nLearner\n\n{learner}\n\nCurrent Date\n\n{current_date}\n\nCurrent Year\n\n{current_year}\n\nTarget Length\n\n{target_length}"),
])


from utilis.tracing import traceable


class PlannerAgent:
    @traceable
    def invoke(self, state):
        llm = make_llm(max_tokens=600, force_groq=True)
        if getattr(state, "mode", "enhance") == "generate":
            gen_prompt = ChatPromptTemplate.from_messages([
                ("system", load_prompt("planner.txt")),
                ("human", "Topic\n\n{topic}\n\nGuidelines & Important Info\n\n{other_info}\n\nCurrent Date\n\n{current_date}\n\nCurrent Year\n\n{current_year}\n\nTarget Length\n\n{target_length}"),
            ])
            response = invoke_with_retry(gen_prompt, llm, {
                "topic": state.title or state.topic_idea,
                "other_info": state.other_info,
                "current_date": state.current_date,
                "current_year": state.current_year,
                "target_length": state.target_length,
            })
        else:
            response = invoke_with_retry(prompt, llm, {
                "blog": state.blog_snippet(max_chars=2500),
                "learner": state.learner_output,
                "current_date": state.current_date,
                "current_year": state.current_year,
                "target_length": state.target_length,
            })

        output = load_json(response.content)

        if "research_query" not in output or not output["research_query"]:
            output["research_query"] = (
                output.get("researchQuery")
                or output.get("research_queries")
                or output.get("query")
                or (state.planner_output.get("research_query") if isinstance(state.planner_output, dict) else "")
                or f"{state.title or state.url} {state.current_year} latest statistics examples SEO GEO keywords"
            )
        if isinstance(output["research_query"], list):
            output["research_query"] = " ".join(output["research_query"])

        state.planner_output = output
        return state
