from bs4 import BeautifulSoup

from config import settings


def _get_value(item, key, default=None):
    if isinstance(item, dict):
        return item.get(key, default)

    return getattr(item, key, default)


def _normalize_url(url, source_url):
    if not url:
        return ""

    if url.startswith("//"):
        return f"https:{url}"

    if url.startswith("http://") or url.startswith("https://"):
        return url

    if source_url and url.startswith("/"):
        from urllib.parse import urljoin

        return urljoin(source_url, url)

    return url


def _is_usable_image_url(url):
    if not url:
        return False

    if not isinstance(url, str):
        return False

    lowered = url.lower()

    if lowered.startswith("data:") or lowered.endswith(".svg"):
        return False

    return lowered.startswith("http://") or lowered.startswith("https://")


def _extract_image_from_html(html, source_url):
    if not html:
        return ""

    soup = BeautifulSoup(html, "lxml")

    for selector in [
        ("meta", {"property": "og:image"}),
        ("meta", {"name": "twitter:image"}),
    ]:
        tag = soup.find(*selector)
        image_url = tag.get("content") if tag else ""

        if _is_usable_image_url(_normalize_url(image_url, source_url)):
            return _normalize_url(image_url, source_url)

    for img in soup.find_all("img"):
        image_url = img.get("src") or img.get("data-src")
        normalized_url = _normalize_url(image_url, source_url)

        if _is_usable_image_url(normalized_url):
            return normalized_url

    return ""


def search_images(query: str, limit: int = 3, context: str = "", placement: str = ""):
    try:
        from firecrawl import Firecrawl
    except ImportError:
        return []

    firecrawl = Firecrawl(api_key=settings.FIRECRAWL_API_KEY)
    search_result = firecrawl.search(query, limit=max(limit * 3, 6))

    web_results = _get_value(search_result, "web", []) or []
    images = []
    seen = set()

    for item in web_results:
        if len(images) >= limit:
            break

        page_url = _get_value(item, "url") or _get_value(item, "source_url")
        title = _get_value(item, "title", "Suggested image")
        description = _get_value(item, "description", "")
        direct_image = (
            _get_value(item, "image_url")
            or _get_value(item, "image")
            or _get_value(item, "thumbnail")
        )

        if not page_url:
            continue

        image_url = _normalize_url(direct_image, page_url)

        if not _is_usable_image_url(image_url):
            continue

        if not image_url or image_url in seen:
            continue

        seen.add(image_url)
        images.append(
            {
                "url": image_url,
                "alt": title,
                "caption": description or title,
                "source_url": page_url,
                "context": context,
                "placement": placement,
            }
        )

    return images
