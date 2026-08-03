"""
Compiled LangGraph workflow for the blog optimizer.

Flow:
  prepare → learner → planner → supervisor
  → language → facts → structure → image → seo → geo
  → aggregator → evaluator ─(evaluation_router)─┐
                                                  ├→ supervisor/language/facts/structure/seo/geo (re-brief)
                                                  └→ optimizer
                                                       │
                                                  evaluator_post ─(optimizer_router)─┐
                                                                                       ├→ targeted_researcher → optimizer
                                                                                       ├→ optimizer  (scores 70-89, loop, max 5)
                                                                                       └→ END

Architecture change — Supervisor replaces the single Researcher:
  Old: one researcher → one generic brief → all 5 specialists
  New: supervisor → one Tavily search → one LLM call → 5 focused briefs
       Each specialist reads its own dedicated brief:
         language_research, facts_research, structure_research,
         seo_research, geo_research
"""

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from langgraph.graph import END, StateGraph

from agents.aggregator import AggregatorAgent
from agents.baseline_evaluator import BaselineEvaluatorAgent
from agents.evaluator import EvaluatorAgent
from agents.facts_agent import FactsAgent
from agents.geo_agent import GeoAgent
from agents.image_agent import ImageAgent
from agents.language_agent import LanguageAgent
from agents.learner import LearnerAgent
from agents.optimizer import OptimizerAgent
from agents.planner import PlannerAgent
from agents.supervisor import SupervisorAgent
from agents.seo_agent import SeoAgent
from agents.structure_agent import StructureAgent
from agents.targeted_researcher import TargetedResearchAgent
from config.llm_registry import set_provider
from graph.router import evaluation_router, optimizer_router
from graph.state import BlogState
from rag.chunker import create_chunks
from rag.html_cleaner import clean_html
from rag.ingestion import ingest_blog
from rag.markdown_converter import html_to_markdown
from rag.scraper import scrape_blog


from utilis.tracing import traceable


@traceable
def prepare_blog(state: BlogState):
    if getattr(state, "job_id", 0):
        from db.database import BlogDatabase
        if BlogDatabase().is_cancel_requested(state.job_id):
            raise RuntimeError("Cancelled by user.")

    set_provider(state.llm_provider)
    if state.llm_provider == "custom":
        from config.llm_registry import set_custom_llm_settings
        set_custom_llm_settings(state.custom_model_name, state.custom_api_key, state.custom_base_url)

    state.active_agent = "prepare"
    if hasattr(state, "active_agent_callback") and state.active_agent_callback:
        try:
            state.active_agent_callback("prepare")
        except Exception:
            pass

    if getattr(state, "resume_node", None) and state.resume_node != "prepare":
        return state

    if getattr(state, "mode", "enhance") == "generate":
        from agents.prompt_generator import PromptGeneratorAgent
        pga = PromptGeneratorAgent()
        state = pga.invoke(state)
        state.chunks = []
        state.url = state.url or f"topic:{state.topic_idea}"
        return state

    scraped = scrape_blog(state.url)
    state.title = scraped.get("title") or ""
    state.raw_blog = scraped["html"]
    state.cleaned_blog = html_to_markdown(clean_html(state.raw_blog))
    state.chunks = create_chunks(state.cleaned_blog)
    ingest_blog(state.chunks, state.url)
    return state


LAST_ACTIVE_STATE = {}

def _wrap(agent_name: str, agent_class):
    _instance = {}

    def _inner(state: BlogState):
        state.active_agent = agent_name
        if hasattr(state, "active_agent_callback") and state.active_agent_callback:
            try:
                state.active_agent_callback(agent_name)
            except Exception:
                pass

        if getattr(state, "job_id", 0):
            from db.database import BlogDatabase
            if BlogDatabase().is_cancel_requested(state.job_id):
                raise RuntimeError("Cancelled by user.")
            LAST_ACTIVE_STATE[state.job_id] = state.copy()

        if "obj" not in _instance:
            _instance["obj"] = agent_class()
        return _instance["obj"].invoke(state)

    _inner.__name__ = agent_name
    return _inner


def prepare_router(state: BlogState):
    if getattr(state, "resume_node", None) and state.resume_node != "prepare":
        return state.resume_node
    if getattr(state, "mode", "enhance") == "generate":
        return "planner"
    return "baseline_evaluator"


workflow = StateGraph(BlogState)

workflow.add_node("prepare",              prepare_blog)
workflow.add_node("baseline_evaluator",   _wrap("baseline_evaluator",  BaselineEvaluatorAgent))
workflow.add_node("learner",              _wrap("learner",              LearnerAgent))
workflow.add_node("planner",              _wrap("planner",              PlannerAgent))
workflow.add_node("supervisor",           _wrap("supervisor",           SupervisorAgent))
workflow.add_node("language",             _wrap("language",             LanguageAgent))
workflow.add_node("facts",                _wrap("facts",                FactsAgent))
workflow.add_node("structure",            _wrap("structure",            StructureAgent))
workflow.add_node("image",                _wrap("image",                ImageAgent))
workflow.add_node("seo",                  _wrap("seo",                  SeoAgent))
workflow.add_node("geo",                  _wrap("geo",                  GeoAgent))
workflow.add_node("aggregator",           _wrap("aggregator",           AggregatorAgent))
workflow.add_node("evaluator",            _wrap("evaluator",            EvaluatorAgent))
workflow.add_node("optimizer",            _wrap("optimizer",            OptimizerAgent))
workflow.add_node("targeted_researcher",  _wrap("targeted_researcher",  TargetedResearchAgent))
workflow.add_node("evaluator_post",       _wrap("evaluator_post",       EvaluatorAgent))

workflow.set_entry_point("prepare")

workflow.add_conditional_edges(
    "prepare",
    prepare_router,
    {
        "planner": "planner",
        "baseline_evaluator": "baseline_evaluator",
        "learner": "learner",
        "supervisor": "supervisor",
        "language": "language",
        "facts": "facts",
        "structure": "structure",
        "image": "image",
        "seo": "seo",
        "geo": "geo",
        "aggregator": "aggregator",
        "evaluator": "evaluator",
        "optimizer": "optimizer",
        "targeted_researcher": "targeted_researcher",
        "evaluator_post": "evaluator_post",
    }
)
workflow.add_edge("baseline_evaluator", "learner")
workflow.add_edge("learner",    "planner")
workflow.add_edge("planner",    "supervisor")   # supervisor replaces researcher
workflow.add_edge("supervisor", "language")
workflow.add_edge("language",   "facts")
workflow.add_edge("facts",      "structure")
workflow.add_edge("structure",  "image")
workflow.add_edge("image",      "seo")
workflow.add_edge("seo",        "geo")
workflow.add_edge("geo",        "aggregator")
workflow.add_edge("aggregator", "evaluator")

# evaluation_router re-runs supervisor (not researcher) when freshness < 70
# so all 5 briefs get refreshed from new search results.
workflow.add_conditional_edges(
    "evaluator",
    evaluation_router,
    {
        "researcher": "supervisor",   # freshness re-run → full supervisor refresh
        "language":   "language",
        "facts":      "facts",
        "structure":  "structure",
        "seo":        "seo",
        "geo":        "geo",
        "optimizer":  "optimizer",
    },
)

workflow.add_edge("optimizer",            "evaluator_post")
workflow.add_edge("targeted_researcher",  "optimizer")

workflow.add_conditional_edges(
    "evaluator_post",
    optimizer_router,
    {
        "targeted_researcher": "targeted_researcher",
        "optimizer":           "optimizer",
        "end":                 END,
    },
)

graph = workflow.compile()
