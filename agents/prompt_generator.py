from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from langchain_core.prompts import ChatPromptTemplate
from config.llm_registry import make_llm
from utilis.json_parser import load_json
from utilis.prompt_loader import load_prompt
from utilis.retry import invoke_with_retry
from utilis.tracing import traceable

prompt = ChatPromptTemplate.from_messages([
    ("system", load_prompt("prompt_generator.txt")),
    ("human", "Topic Idea: {topic_idea}\n\nGuidelines & Other Info: {other_info}"),
])


class PromptGeneratorAgent:
    @traceable
    def invoke(self, state):
        llm = make_llm(max_tokens=500, force_groq=True)
        response = invoke_with_retry(prompt, llm, {
            "topic_idea": state.topic_idea,
            "other_info": state.other_info,
        })
        
        output = load_json(response.content)
        state.title = output.get("optimized_topic") or state.topic_idea
        state.research_output = ""
        
        state.planner_output = {
            "research_query": output.get("research_prompt") or state.topic_idea,
            "primary_keyword": state.topic_idea,
            "h1_title": state.title,
            "h2_headings": [],
            "faq_questions": [],
            "secondary_keywords": [],
            "target_outline": []
        }
        return state
