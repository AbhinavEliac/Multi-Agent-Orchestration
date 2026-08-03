from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from graph.graph import graph

from graph.state import BlogState

state = BlogState(

    url="https://aiindia.ai/corporate-ai-trainings/"

)

result = BlogState(**graph.invoke(state))

print(result.optimized_blog)
