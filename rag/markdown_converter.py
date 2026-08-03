"""
markdown_converter.py
"""

from markdownify import markdownify


def html_to_markdown(html: str):

    markdown = markdownify(

        html,

        heading_style="ATX"

    )

    return markdown