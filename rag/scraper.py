"""
scraper.py

Scrapes a webpage using Crawl4AI.

Returns:

raw_html
cleaned_markdown
title
metadata
"""

import asyncio

import requests
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler


class BlogScraper:

    def _metadata_from_html(self, html: str):
        soup = BeautifulSoup(html, "lxml")
        title = ""

        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        metadata = {"title": title}

        for tag in soup.find_all("meta"):
            name = tag.get("name") or tag.get("property")
            content = tag.get("content")

            if name and content:
                metadata[name] = content

        return metadata

    def _fast_http_scrape(self, url: str):
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            html = response.text
            if len(html) > 1200:
                metadata = self._metadata_from_html(html)
                return {
                    "url": url,
                    "title": metadata.get("title", ""),
                    "html": html,
                    "markdown": "",
                    "metadata": metadata,
                }
        except Exception:
            pass
        return None

    def _fallback_scrape(self, url: str, crawl_error: str = ""):
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        try:
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            html = response.text
            metadata = self._metadata_from_html(html)
            return {
                "url": url,
                "title": metadata.get("title", ""),
                "html": html,
                "markdown": "",
                "metadata": metadata,
            }
        except Exception as exc:
            detail = f"Crawl4AI error: {crawl_error}. " if crawl_error else ""
            raise Exception(f"Unable to scrape {url}. {detail}HTTP fallback error: {exc}") from exc



    async def scrape(self, url: str):
        # 1. Try ultra-fast HTTP request first (< 0.5s)
        fast_result = self._fast_http_scrape(url)
        if fast_result is not None:
            return fast_result

        # 2. Fallback to Crawl4AI headless browser for JS-rendered apps
        try:
            async with AsyncWebCrawler() as crawler:
                try:
                    result = await asyncio.wait_for(
                        crawler.arun(
                            url=url,
                            bypass_cache=True,
                        ),
                        timeout=40.0,
                    )
                except Exception as exc:
                    return self._fallback_scrape(url, f"Crawl4AI timeout/error: {exc}")

                if not result.success:
                    crawl_error = (
                        getattr(result, "error_message", "")
                        or getattr(result, "status_message", "")
                        or "unknown Crawl4AI failure"
                    )
                    return self._fallback_scrape(url, crawl_error)

                return {
                    "url": url,
                    "title": result.metadata.get("title", ""),
                    "html": result.html,
                    "markdown": result.markdown,
                    "metadata": result.metadata,
                }
        except Exception as exc:
            return self._fallback_scrape(url, f"Playwright / Crawl4AI startup error: {exc}")


def scrape_blog(url: str):
    scraper = BlogScraper()
    return asyncio.run(scraper.scrape(url))
