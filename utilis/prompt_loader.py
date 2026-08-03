from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"


def load_prompt(filename: str):

    with open(PROMPT_DIR / filename, "r", encoding="utf8") as f:

        return f.read()
