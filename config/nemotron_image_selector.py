import json
import hashlib
from pathlib import Path

from openai import OpenAI
import requests

from config import settings


IMAGE_DIR = Path(__file__).resolve().parents[1] / "generated_images"


def _client():
    if not settings.NEMOTRON_API_KEY or not settings.NEMOTRON_MODEL:
        return None

    return OpenAI(
        api_key=settings.NEMOTRON_API_KEY,
        base_url=settings.NEMOTRON_BASE_URL,
        timeout=60.0,
    )


def _candidate_text(candidates):
    lines = []

    for index, candidate in enumerate(candidates, start=1):
        lines.append(
            "\n".join(
                [
                    f"{index}. URL: {candidate.get('url', '')}",
                    f"Alt: {candidate.get('alt', '')}",
                    f"Caption: {candidate.get('caption', '')}",
                    f"Source: {candidate.get('source_url', '')}",
                ]
            )
        )

    return "\n\n".join(lines)


def _parse_selection(content, candidates, limit):
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        # Bad JSON from the model — return nothing rather than a random candidate
        return []

    selected = payload.get("selected_images", payload if isinstance(payload, list) else [])
    ranked = []

    for item in selected:
        if not isinstance(item, dict):
            continue

        index = item.get("index")
        score = item.get("score", 0)

        # Hard threshold: below 60 means the model itself considers it irrelevant
        try:
            if int(score) < 60:
                continue
        except (TypeError, ValueError):
            pass

        try:
            candidate = candidates[int(index) - 1].copy()
        except (TypeError, ValueError, IndexError):
            continue

        candidate["relevance_score"] = score
        candidate["relevance_reason"] = item.get("reason", "")
        candidate["alt"] = item.get("alt") or candidate.get("alt")
        candidate["caption"] = item.get("caption") or candidate.get("caption")
        ranked.append(candidate)

    # Do NOT fall back to candidates[:limit] — return empty if nothing passes
    return ranked[:limit]


def _vision_content(prompt, candidates):
    content = [{"type": "text", "text": prompt}]

    for index, candidate in enumerate(candidates[:6], start=1):
        content.append({"type": "text", "text": f"Candidate {index}"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": candidate.get("url", "")},
            }
        )

    return content


def select_relevant_images(context, section, candidates, limit=1):
    if not candidates:
        return []

    from config.llm_registry import make_llm
    from utilis.retry import invoke_with_retry
    from langchain_core.prompts import ChatPromptTemplate

    prompt = f"""
You are a strict editorial image reviewer for a professional business blog.

Blog topic: {section}

Section context (the text this image must illustrate):
{context}

Image candidates:
{_candidate_text(candidates)}

RULES:
- Only select an image if it DIRECTLY illustrates the section context above.
- The image must be a photograph, diagram, infographic, or chart relevant to the blog topic.
- REJECT any image that shows: people (unless they are clearly in a business/workplace context), entertainment, sports, martial arts, movies, TV shows, celebrities, cartoon characters, food, nature scenes unrelated to the topic, or anything that a reader would find confusing next to this blog section.
- A score below 60 means REJECT — do not include it.
- If NO candidate is relevant enough, return an empty selected_images array.

Return ONLY JSON:
{{
  "selected_images": [
    {{
      "index": 1,
      "score": 0-100,
      "reason": "why this image fits the section context",
      "alt": "descriptive alt text for the image",
      "caption": "short caption for display under the image"
    }}
  ]
}}

If nothing is suitable, return: {{"selected_images": []}}
"""

    try:
        llm = make_llm(max_tokens=800, force_groq=True)
        chat_prompt = ChatPromptTemplate.from_messages([("human", prompt)])
        response = invoke_with_retry(chat_prompt, llm, {})
        content = response.content or ""
    except Exception:
        return candidates[:limit]

    return _parse_selection(content, candidates, limit)


def download_image(image, section):
    url = image.get("url")

    if not url:
        return image

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    safe_section = "".join(
        character.lower() if character.isalnum() else "-"
        for character in section
    ).strip("-")[:50] or "blog-image"

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
    except Exception:
        return image

    content_type = response.headers.get("content-type", "").split(";")[0]
    extension = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
    }.get(content_type, ".jpg")

    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    file_name = f"{safe_section}-{url_hash}{extension}"
    file_path = IMAGE_DIR / file_name
    file_path.write_bytes(response.content)

    image["remote_url"] = url
    image["local_path"] = file_path.as_posix()
    image["url"] = file_path.as_posix()

    return image
