"""
Robust JSON extraction utility.

Models sometimes wrap JSON in markdown fences, add preamble text, or return
partial JSON. This module strips all of that and falls back gracefully rather
than raising an OutputParserException.
"""

import json
import re


def _strip_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers."""
    # Match ```json\n...\n``` or ```\n...\n```
    fenced = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


def _extract_json_object(text: str) -> str:
    """
    Find the first {...} block in text, handling nested braces.
    Returns the raw JSON string, or the original text if no block found.
    """
    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    in_string = False
    escape_next = False

    for i, ch in enumerate(text[start:], start=start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return text[start:]


def load_json(text) -> dict:
    """
    Parse JSON from an LLM response string or structured content list.
    """
    if isinstance(text, list):
        extracted = []
        for item in text:
            if isinstance(item, dict) and "text" in item:
                extracted.append(item["text"])
            elif isinstance(item, str):
                extracted.append(item)
        text = "\n".join(extracted)
    elif not isinstance(text, str):
        text = str(text)

    Handles:
    - Plain JSON
    - JSON wrapped in ```json ... ``` fences
    - JSON with leading/trailing prose
    - Partial or malformed JSON (returns {} instead of raising)
    """
    if not text or not text.strip():
        return {}

    # Try the raw text first (cheapest path)
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Strip markdown fences and retry
    stripped = _strip_fences(text)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Extract the first {...} block and retry
    extracted = _extract_json_object(stripped)
    try:
        return json.loads(extracted)
    except json.JSONDecodeError:
        pass

    # Last resort: try the raw extracted block from original text
    extracted_raw = _extract_json_object(text)
    try:
        return json.loads(extracted_raw)
    except json.JSONDecodeError:
        return {}
