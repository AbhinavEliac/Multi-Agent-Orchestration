import logging

from tavily import TavilyClient
from config import settings

logger = logging.getLogger(__name__)

MAX_QUERY_LENGTH = 390


def _get_client():
    key = settings.TAVILY_API_KEY
    if not key:
        return None
    return TavilyClient(api_key=key)


def _limit_query(query, max_length=MAX_QUERY_LENGTH):
    query = " ".join(str(query).split())

    if len(query) <= max_length:
        return query

    return query[:max_length].rsplit(" ", 1)[0].strip()


def search(query, max_results=5, search_depth="advanced"):
    client = _get_client()
    if not client:
        logger.warning("Tavily API key not found.")
        return {"results": []}

    try:
        response = client.search(
            query=_limit_query(query),
            max_results=max_results,
            search_depth=search_depth
        )
        return response
    except Exception as exc:
        logger.warning("Tavily search error: %s", exc)
        return {"results": []}


def search_image_candidates(query, max_results=8, search_depth="advanced"):
    client = _get_client()
    if not client:
        logger.warning("Tavily API key not found.")
        return []

    try:
        response = client.search(
            query=_limit_query(query),
            max_results=max_results,
            search_depth=search_depth,
            include_images=True,
            include_image_descriptions=True
        )
    except Exception as exc:
        logger.warning("Tavily image search error: %s", exc)
        response = {}

    images = response.get("images", []) if isinstance(response, dict) else []
    candidates = []
    seen = set()

    for image in images:
        if isinstance(image, str):
            image_url = image
            description = ""
        else:
            image_url = image.get("url") or image.get("image_url")
            description = image.get("description") or image.get("alt") or ""

        if not image_url or image_url in seen:
            continue

        seen.add(image_url)
        candidates.append(
            {
                "url": image_url,
                "alt": description,
                "caption": description,
                "source_url": image_url,
            }
        )

    results = response.get("results", []) if isinstance(response, dict) else []

    for result in results:
        image_url = result.get("image_url") or result.get("thumbnail")

        if not image_url or image_url in seen:
            continue

        seen.add(image_url)
        candidates.append(
            {
                "url": image_url,
                "alt": result.get("title", ""),
                "caption": result.get("content", "") or result.get("title", ""),
                "source_url": result.get("url") or image_url,
            }
        )

    return candidates
