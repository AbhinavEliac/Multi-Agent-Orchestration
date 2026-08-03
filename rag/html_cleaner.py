"""
html_cleaner.py

Removes unnecessary HTML.
"""

from bs4 import BeautifulSoup


REMOVE_TAGS = [

    "script",

    "style",

    "svg",

    "iframe",

    "noscript",

    "footer",

    "header",

    "nav",

    "aside"

]


def clean_html(html: str):

    soup = BeautifulSoup(
        html,
        "lxml"
    )

    for tag in REMOVE_TAGS:

        for item in soup.find_all(tag):

            item.decompose()

    return str(soup)