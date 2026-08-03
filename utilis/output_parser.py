from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from langchain_core.output_parsers import JsonOutputParser

from utilis.schemas import *

learner_parser = JsonOutputParser(
    pydantic_object=LearnerSchema
)

planner_parser = JsonOutputParser(
    pydantic_object=PlannerSchema
)

research_parser = JsonOutputParser(
    pydantic_object=ResearchSchema
)

evaluation_parser = JsonOutputParser(
    pydantic_object=EvaluationSchema
)