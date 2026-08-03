from pathlib import Path
import re
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from config.firecrawl_manager import search_images
from config.nemotron_image_selector import download_image, select_relevant_images
from config.tavily_search import search_image_candidates


# ---------------------------------------------------------------------------
# Terms that indicate images are NOT useful for a professional blog:
# cartoons, comic art, logos without context, generic stock icons, etc.
# If an image URL or alt text contains any of these, reject it.
# ---------------------------------------------------------------------------
_REJECT_PATTERNS = re.compile(
    # Entertainment / promotional content
    r"netflix|hulu|disney|amazon prime|hbo|peacock|paramount|"
    r"movie|film\b|cinema|poster\b|trailer\b|series\b|episode|"
    r"actor|actress|celebrity|red.?carpet|billboard\b|"
    # Comic / cartoon art
    r"clipart|cartoon|comic\b|superhero|avengers|marvel|dc.?comics|"
    r"anime|manga|"
    # Generic web junk
    r"vector|icon\b|logo\b|favicon|banner\b|badge|sticker|"
    r"thumbnail|avatar|emoji|smiley|wallpaper|meme|gif\b|"
    # Stock photo filler words in the URL path
    r"gettyimages|shutterstock|istockphoto|dreamstime|depositphotos",
    re.IGNORECASE,
)

# Only accept common photo/diagram image extensions
_ACCEPT_EXTENSIONS = re.compile(r"\.(jpg|jpeg|png|webp)(\?.*)?$", re.IGNORECASE)


from utilis.tracing import traceable


class ImageAgent:

    def _search_depth(self, research_level):
        return "basic" if str(research_level).lower() in {"easy", "basic"} else "advanced"

    def _clean_heading(self, line):
        return re.sub(r"^#+\s*", "", line).strip(" :#")

    def _topic(self, state) -> str:
        """Extract the core blog topic from the planner or URL for query grounding."""
        intent = state.learner_output.get("intent") or ""
        topic = state.planner_output.get("research_query") or intent or state.url
        # Keep it short — first 80 chars
        return str(topic)[:80].strip()

    def _context_windows(self, markdown: str, count: int) -> list[dict]:
        """
        Split the article into `count` evenly distributed section windows.
        Each window carries its heading and the surrounding body text as context
        for a highly specific image search.
        """
        if count <= 0:
            return []

        blocks = []
        current_heading = ""
        current_lines: list[str] = []

        for line in markdown.splitlines():
            if re.match(r"^#{1,3}\s+", line.strip()):
                if current_heading and current_lines:
                    blocks.append({
                        "heading": current_heading,
                        "context": " ".join(current_lines).strip(),
                    })
                current_heading = self._clean_heading(line)
                current_lines = []
                continue

            stripped = line.strip()
            if current_heading and stripped and not stripped.startswith("!"):
                current_lines.append(stripped)

        if current_heading and current_lines:
            blocks.append({
                "heading": current_heading,
                "context": " ".join(current_lines).strip(),
            })

        # Drop very short blocks and FAQ sections (usually no good image)
        blocks = [
            b for b in blocks
            if len(b["context"]) > 80
            and not b["heading"].lower().startswith("faq")
            and not b["heading"].lower().startswith("conclusion")
        ]

        if not blocks:
            return []

        # Distribute `count` windows evenly across blocks
        if len(blocks) <= count:
            # Pad by repeating the richest blocks
            extra = count - len(blocks)
            richest = sorted(blocks, key=lambda b: len(b["context"]), reverse=True)
            blocks += richest[:extra]

        step = len(blocks) / count
        windows = []
        for i in range(count):
            idx = min(int(i * step), len(blocks) - 1)
            b = blocks[idx]
            words = b["context"].split()
            preceding_100 = " ".join(words[-100:]) if len(words) > 100 else b["context"]
            placement = f"Place after the paragraph in '{b['heading']}' that ends with: '{preceding_100[-80:]}'."
            windows.append({
                "heading":   b["heading"],
                "context":   preceding_100,
                "placement": placement,
            })

        return windows

    def _fallback_contexts(self, state, count: int) -> list[dict]:
        topic = self._topic(state)
        words = str(state.research_output[:1000]).split()
        preceding_100 = " ".join(words[:100])
        return [{
            "heading":   "Introduction",
            "context":   f"{topic} — {preceding_100}",
            "placement": "Place after the introductory section.",
        }] * max(count, 1)

    def _image_query(self, state, section: dict) -> str:
        """
        Build a precise, topic-anchored image search query.
        Format: "<blog topic> <section heading> <key context words> photo diagram"
        This greatly reduces the chance of returning irrelevant stock/cartoon images.
        """
        topic = self._topic(state)
        heading = section["heading"]
        # Use up to 15 words of section context as additional signal
        context_words = " ".join(section["context"].split()[:15])
        year = state.current_year

        return f"{topic} {heading} {context_words} {year} photo infographic"[:300]

    def _filter_candidates(self, candidates: list[dict], section: dict) -> list[dict]:
        """
        Remove candidates that are clearly irrelevant or low quality.
        Checks both the URL (full path) and alt/caption text.
        """
        filtered = []
        for c in candidates:
            url = (c.get("url") or "").lower()
            alt = (c.get("alt") or c.get("caption") or "").lower()
            combined = url + " " + alt

            # Reject by combined URL + alt pattern
            if _REJECT_PATTERNS.search(combined):
                continue

            # Require a recognised photo extension
            if not _ACCEPT_EXTENSIONS.search(url):
                continue

            filtered.append(c)

        return filtered

    @traceable
    def invoke(self, state):
        if getattr(state, "image_output", None):
            return state
        count = max(0, min(int(state.image_count or 0), 20))
        if count == 0:
            state.image_output = []
            return state

        source_markdown = state.cleaned_blog
        selected_sections = (
            self._context_windows(source_markdown, count)
            or self._fallback_contexts(state, count)
        )

        images: list[dict] = []
        seen_urls: set[str] = set()
        fallback_sections: list[dict] = []

        for section in selected_sections:
            query = self._image_query(state, section)
            candidates = search_image_candidates(
                query,
                max_results=12,
                search_depth=self._search_depth(state.research_level),
            )
            candidates = self._filter_candidates(candidates, section)

            found = select_relevant_images(
                section["context"][:800],
                section["heading"],
                candidates,
                limit=1,
            )

            # Firecrawl fallback — only use if filter passes
            if not found:
                firecrawl_candidates = search_images(
                    query,
                    limit=3,
                    context=section["context"][:500],
                    placement=section["placement"],
                )
                found = self._filter_candidates(firecrawl_candidates, section)
                if found:
                    found = select_relevant_images(
                        section["context"][:800],
                        section["heading"],
                        found,
                        limit=1,
                    )

            if not found:
                fallback_sections.append(section)
                continue

            found = [download_image(img, section["heading"]) for img in found]

            for img in found:
                url = img.get("url")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                img.setdefault("context",   section["context"])
                img.setdefault("placement", section["placement"])
                img["section"] = section["heading"]
                img["alt"] = img.get("alt") or f"{section['heading']} — {self._topic(state)}"
                img["caption"] = img.get("caption") or img["alt"]
                img["source_url"] = img.get("source_url") or img.get("remote_url") or url
                images.append(img)

        # Second pass for sections that returned nothing
        for section in fallback_sections:
            if len(images) >= count:
                break

            query = self._image_query(state, section)
            candidates = self._filter_candidates(
                search_image_candidates(query, max_results=10,
                                        search_depth=self._search_depth(state.research_level)),
                section,
            )
            found = select_relevant_images(
                section["context"][:800], section["heading"], candidates, limit=1
            )
            if not found:
                firecrawl_candidates = self._filter_candidates(
                    search_images(query, limit=3,
                                  context=section["context"][:500],
                                  placement=section["placement"]),
                    section,
                )
                found = select_relevant_images(
                    section["context"][:800], section["heading"],
                    firecrawl_candidates, limit=1,
                ) if firecrawl_candidates else []
            found = self._filter_candidates(found or [], section)

            for img in found:
                dl = download_image(img, section["heading"])
                url = dl.get("url")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    dl.setdefault("context",   section["context"])
                    dl.setdefault("placement", section["placement"])
                    dl["section"] = section["heading"]
                    dl["source_url"] = dl.get("source_url") or dl.get("remote_url") or url
                    images.append(dl)

        state.image_output = images[:count]
        return state
