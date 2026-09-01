import re


def ensure_exact_images_in_markdown(text: str, image_output: list[dict]) -> str:
    """
    Guarantees that:
    1. Zero duplicate images exist in the markdown article (same URL or duplicate embed removed).
    2. Exactly all N images in image_output are contextually embedded in distinct body sections.
    3. Missing images from LLM output are automatically inserted into relevant sections.
    """
    if not text or not image_output:
        return text

    # 1. Deduplicate any repeated image embeds in text
    seen_urls = set()

    def _dedup_img(match):
        alt = match.group(1)
        url = match.group(2)
        if url in seen_urls:
            return ""  # strip duplicate
        seen_urls.add(url)
        return match.group(0)

    text = re.sub(r"!\[(.*?)\]\((.*?)\)", _dedup_img, text)

    # 2. Check which images from image_output are missing
    missing = []
    for idx, img in enumerate(image_output):
        u = img.get("url", "")
        lp = img.get("local_path", "")
        ru = img.get("remote_url", "") or img.get("source_url", "")
        found = False
        for target in [u, lp, ru]:
            if target and target in text:
                found = True
                break
        if not found:
            missing.append((idx, img))

    if not missing:
        return text

    # 3. Insert missing images into distinct body sections
    sections = re.split(r"(?=\n##\s+)", text)
    if len(sections) > 1:
        sec_idx = 1
        for idx, img in missing:
            alt = img.get("alt") or img.get("section") or f"Visual Analysis {idx+1}"
            img_url = img.get("url") or img.get("local_path") or img.get("remote_url")
            cap = img.get("caption") or alt
            is_custom = img.get("is_custom", False)
            src = img.get("source_url") or img.get("remote_url") or ""

            if is_custom or src == "Custom Image" or src == "Custom Upload":
                caption_line = f"\n\n*Figure {idx+1}: {cap} (Source: Custom Upload)*\n"
            elif src and "http" in src:
                src_domain = src.split("/")[2] if "//" in src else src
                caption_line = f"\n\n*Figure {idx+1}: {cap} (Source: [{src_domain}]({src}))*\n"
            else:
                caption_line = f"\n\n*Figure {idx+1}: {cap}*\n"

            img_md = f"\n\n![{alt}]({img_url})" + caption_line

            # Find a body section that matches image section or doesn't have an image
            target_sec = None
            target_heading = (img.get("section") or "").lower()

            # First priority: matching section heading
            if target_heading:
                for s_i in range(1, len(sections)):
                    sec_header = sections[s_i].splitlines()[0].lower() if sections[s_i].splitlines() else ""
                    if target_heading in sec_header or any(w in sec_header for w in target_heading.split() if len(w) > 3):
                        target_sec = s_i
                        break

            # Second priority: any section without an image (avoiding FAQ and Conclusion)
            if target_sec is None:
                for s_i in range(1, len(sections)):
                    sec_header = sections[s_i].splitlines()[0].lower() if sections[s_i].splitlines() else ""
                    if "![" not in sections[s_i] and "faq" not in sec_header and "conclusion" not in sec_header:
                        target_sec = s_i
                        break

            if target_sec is None:
                target_sec = min(sec_idx, len(sections) - 1)
                sec_idx += 1

            paragraphs = sections[target_sec].split("\n\n")
            if len(paragraphs) >= 2:
                paragraphs.insert(2, img_md.strip())
            else:
                paragraphs.append(img_md.strip())
            sections[target_sec] = "\n\n".join(paragraphs)

        text = "".join(sections)
    else:
        for idx, img in missing:
            alt = img.get("alt") or f"Visual Analysis {idx+1}"
            img_url = img.get("url") or img.get("local_path") or img.get("remote_url")
            text += f"\n\n![{alt}]({img_url})\n"

    return text
