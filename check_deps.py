import sys
sys.path.insert(0, ".")

modules = [
    ("google.api_core",          "google-api-core"),
    ("google.generativeai",       "google-generativeai"),
    ("google.genai",              "google-genai"),
    ("langchain_google_genai",    "langchain-google-genai"),
    ("langchain_huggingface",     "langchain-huggingface"),
    ("langchain_pinecone",        "langchain-pinecone"),
    ("langchain_openai",          "langchain-openai"),
    ("langchain_groq",            "langchain-groq"),
    ("langchain_community",       "langchain-community"),
    ("pinecone",                  "pinecone"),
    ("tavily",                    "tavily-python"),
    ("firecrawl",                 "firecrawl-py"),
    ("crawl4ai",                  "crawl4ai"),
    ("playwright",                "playwright"),
    ("sentence_transformers",     "sentence-transformers"),
    ("openai",                    "openai"),
    ("groq",                      "groq"),
    ("markdownify",               "markdownify"),
    ("bs4",                       "beautifulsoup4"),
    ("lxml",                      "lxml"),
    ("requests",                  "requests"),
    ("dotenv",                    "python-dotenv"),
    ("pandas",                    "pandas"),
    ("PIL",                       "Pillow"),
    ("httpx",                     "httpx"),
    ("langsmith",                 "langsmith"),
    ("docx",                      "python-docx"),
    ("reportlab",                 "reportlab"),
]

missing = []
ok = []
for mod, pkg in modules:
    try:
        __import__(mod)
        ok.append(pkg)
    except ImportError:
        missing.append((mod, pkg))

print(f"OK ({len(ok)}/{len(modules)}):", ", ".join(ok))
print()
if missing:
    print(f"MISSING ({len(missing)}):")
    for mod, pkg in missing:
        print(f"  pip install {pkg}   # import {mod}")
else:
    print("All packages present.")
