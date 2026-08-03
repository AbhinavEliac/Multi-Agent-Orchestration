# Blog Enhancer — Complete Architecture & Technical Reference

> Last updated: July 2026. Reflects all changes through the fault-tolerance and token-optimisation refactor.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Full Pipeline Flowchart](#2-full-pipeline-flowchart)
3. [Project File Structure](#3-project-file-structure)
4. [LangGraph DAG — Nodes and Edges](#4-langgraph-dag--nodes-and-edges)
5. [BlogState — Complete Field Reference](#5-blogstate--complete-field-reference)
6. [Agent Reference — Every Agent in Detail](#6-agent-reference--every-agent-in-detail)
7. [Routing Logic — Exact Conditions and Caps](#7-routing-logic--exact-conditions-and-caps)
8. [LLM Layer — Providers, Rotation, Mixed Routing](#8-llm-layer--providers-rotation-mixed-routing)
9. [Supervisor — Research Architecture](#9-supervisor--research-architecture)
10. [Image Pipeline](#10-image-pipeline)
11. [RAG Layer](#11-rag-layer)
12. [Persistence Layer — SQLite Schema](#12-persistence-layer--sqlite-schema)
13. [Fault Tolerance — Jobs System](#13-fault-tolerance--jobs-system)
14. [Research Cache](#14-research-cache)
15. [Streaming Architecture](#15-streaming-architecture)
16. [Token Budget Engineering](#16-token-budget-engineering)
17. [Evaluation Rubric — Deduction Tables](#17-evaluation-rubric--deduction-tables)
18. [Prompt Files Reference](#18-prompt-files-reference)
19. [UI — Streamlit App](#19-ui--streamlit-app)
20. [Configuration and .env Reference](#20-configuration-and-env-reference)

---

## 1. System Overview

Blog Enhancer is a production multi-agent AI pipeline. It takes a live blog URL, scrapes the source article, evaluates it as a baseline, runs a structured research and review pipeline, writes a complete enhanced long-form article, evaluates it across six quality dimensions, and iteratively optimises it until all scores reach 90/100 or iteration caps are hit.

**Key design principles:**

- **Mixed-provider routing** — analysis agents (evaluation, research, review) always run on Groq (free, fast, structured output). Prose agents (aggregator, optimizer) use the user-selected provider (Groq / Gemini / OpenAI) to preserve premium quota.
- **Supervisor pattern** — one Tavily search + one LLM call produces five focused research briefs, one per specialist agent. Each specialist reads only what it needs.
- **Fault-tolerant jobs** — every generation job writes progress to SQLite every 0.8 s. Page refreshes do not kill the job. Cancel is cooperative via a DB flag.
- **Token budget engineering** — every agent has exact input size caps that keep total tokens under Groq free-tier TPM limits (6,000 TPM per model).
- **Baseline scoring** — the original blog is scored before any enhancement so the UI can show a before/after delta.

**Entry points:**
- `streamlit_app.py` — three-tab browser UI (New Blog / Active Jobs / History)
- `app.py` — minimal CLI runner

---

## 2. Full Pipeline Flowchart

```
User submits URL via Streamlit UI
            │
            ▼
  BlogDatabase.create_job(url)        ← SQLite jobs table, status='running'
  JobWriter.start()                   ← background flush thread, writes progress every 0.8s
            │
            ▼
  graph.invoke(BlogState)             ← LangGraph DAG, runs in daemon thread
            │
┌───────────────────────────────────────────────────────────────────────────┐
│  PHASE 1 — SETUP                                                          │
│                                                                           │
│  prepare_blog                                                             │
│  ├─ set_provider(state.llm_provider)   ← sets prose provider globally    │
│  ├─ scrape_blog(url)                   ← Crawl4AI async + requests       │
│  ├─ html_to_markdown(clean_html(html)) ← BeautifulSoup + markdownify     │
│  ├─ create_chunks(cleaned_blog)        ← RecursiveCharacterTextSplitter  │
│  └─ ingest_blog(chunks, url)           ← embed → Pinecone upsert         │
│                                                                           │
│  baseline_evaluator                                                       │
│  ├─ Reads: cleaned_blog (max 5,000 chars)                                 │
│  ├─ LLM: Groq, max_tokens=800                                             │
│  ├─ Prompt: evaluator.txt                                                 │
│  └─ Writes: baseline_*_score (×7)     ← never overwritten again          │
│                                                                           │
│  learner                                                                  │
│  ├─ Reads: blog_snippet(2500 chars)                                       │
│  ├─ LLM: Groq, max_tokens=600                                             │
│  ├─ Prompt: learner.txt                                                   │
│  └─ Writes: learner_output (JSON dict)                                    │
│                                                                           │
│  planner                                                                  │
│  ├─ Reads: blog_snippet(2500) + learner_output + date/year/length         │
│  ├─ LLM: Groq, max_tokens=600                                             │
│  ├─ Prompt: planner.txt                                                   │
│  └─ Writes: planner_output (flat JSON with primary_keyword, h1_title,    │
│             h2_headings, faq_questions, secondary_keywords,               │
│             research_query, seo_plan, geo_plan, target_outline)           │
└───────────────────────────────────────────────────────────────────────────┘
            │
┌───────────────────────────────────────────────────────────────────────────┐
│  PHASE 2 — SUPERVISOR RESEARCH                                            │
│                                                                           │
│  supervisor                                                               │
│  ├─ Checks ResearchCache (TF-IDF cosine ≥ 0.82, 6hr TTL)                 │
│  ├─ Cache miss: Tavily search(research_query, max_results=3-15)           │
│  ├─ Trims results to 4,000 chars (400 chars per result)                   │
│  ├─ LLM: Groq, max_tokens=2000                                            │
│  ├─ Prompt: supervisor.txt → JSON with 5 keys                             │
│  └─ Writes:                                                               │
│       state.language_research   ← tone, audience, hook angle             │
│       state.facts_research      ← stats, examples, expert findings       │
│       state.structure_research  ← full article skeleton                   │
│       state.seo_research        ← keyword plan, H2s, FAQs, snippets      │
│       state.geo_research        ← entities, definitions, direct answers  │
│       state.research_output     ← raw trimmed search results              │
└───────────────────────────────────────────────────────────────────────────┘
            │
┌───────────────────────────────────────────────────────────────────────────┐
│  PHASE 3 — SPECIALIST REVIEWS (sequential)                                │
│                                                                           │
│  language_agent                                                           │
│  ├─ Reads: blog(2500) + language_research(800) + quality + length         │
│  ├─ LLM: Groq force, max_tokens=700                                       │
│  └─ Writes: language_output (prose audit notes)                           │
│                                                                           │
│  facts_agent                                                              │
│  ├─ Reads: blog(2500) + facts_research(1000) + date/year/length           │
│  ├─ LLM: Groq force, max_tokens=700                                       │
│  └─ Writes: facts_output (E-E-A-T review notes)                           │
│                                                                           │
│  structure_agent                                                          │
│  ├─ Reads: blog(2500) + structure_research(1200) + length                 │
│  ├─ LLM: Groq force, max_tokens=700                                       │
│  └─ Writes: structure_output (skeleton + thin section flags)              │
│                                                                           │
│  image_agent                                                              │
│  ├─ Reads: cleaned_blog (always — never structure_output)                 │
│  ├─ Splits into N context windows by heading (skips FAQ/Conclusion)       │
│  ├─ Tavily image search per section                                       │
│  ├─ _filter_candidates() — rejects cartoons, logos, stock                 │
│  ├─ Nemotron vision scoring (score ≥ 60 required)                         │
│  ├─ Firecrawl fallback if Nemotron rejects all                            │
│  ├─ download_image() → /generated_images/{slug}-{hash}.ext               │
│  └─ Writes: image_output (list of image dicts)                            │
│                                                                           │
│  seo_agent                                                                │
│  ├─ Reads: blog(2500) + seo_research(1200) + date/year/length             │
│  ├─ LLM: Groq force, max_tokens=700                                       │
│  └─ Writes: seo_output (keyword audit, H2 rewrites, FAQ audit)            │
│                                                                           │
│  geo_agent                                                                │
│  ├─ Reads: seo_output(900) + geo_research(1000) + date/year/length        │
│  ├─ LLM: Groq force, max_tokens=700                                       │
│  └─ Writes: geo_output (entity audit, definition blocks, direct answers)  │
└───────────────────────────────────────────────────────────────────────────┘
            │
┌───────────────────────────────────────────────────────────────────────────┐
│  PHASE 4 — FIRST DRAFT (streaming)                                        │
│                                                                           │
│  aggregator                                                               │
│  ├─ System: aggregator.txt + images.txt (~6,150 chars)                    │
│  ├─ Inputs (hard-capped to stay under 8K TPM):                            │
│  │   blog:      1,800 chars                                               │
│  │   language:  500 chars                                                 │
│  │   facts:     500 chars                                                 │
│  │   structure: 900 chars                                                 │
│  │   seo:       900 chars                                                 │
│  │   geo:       400 chars                                                 │
│  │   misc:      ~300 chars                                                │
│  │   Total input: ~10,300 chars ≈ 2,943 tokens                            │
│  ├─ LLM: prose provider (Gemini/OpenAI/Groq), max_tokens=3,500            │
│  ├─ Uses stream_invoke() — tokens pushed to state.stream_queue            │
│  └─ Writes: aggregated_blog (first complete article draft)                │
└───────────────────────────────────────────────────────────────────────────┘
            │
┌───────────────────────────────────────────────────────────────────────────┐
│  PHASE 5 — FIRST EVALUATION + SPECIALIST LOOP                             │
│                                                                           │
│  evaluator                                                                │
│  ├─ Reads: optimized_blog or aggregated_blog (max 9,000 chars)            │
│  ├─ LLM: Groq force, max_tokens=1,200                                     │
│  ├─ Prompt: evaluator.txt (start-at-100, deduct per flaw)                 │
│  └─ Writes: *_score (×6) + *_feedback (×6) + overall_score               │
│             increments state.iteration                                    │
│                                                                           │
│  evaluation_router                                                        │
│  ├─ Cap: MAX_SPECIALIST_ITERATIONS = 3                                    │
│  ├─ freshness < 70  → supervisor (full research refresh)                  │
│  ├─ language  < 70  → language                                            │
│  ├─ facts     < 70  → facts                                               │
│  ├─ structure < 70  → structure                                           │
│  ├─ seo       < 70  → seo                                                 │
│  ├─ geo       < 70  → geo                                                 │
│  │  (re-run specialist → re-aggregate → re-evaluate)                     │
│  └─ all ≥ 70 or cap hit → optimizer                                       │
└───────────────────────────────────────────────────────────────────────────┘
            │
┌───────────────────────────────────────────────────────────────────────────┐
│  PHASE 6 — OPTIMISATION LOOP (up to max_optimizer_passes, default 3)      │
│                                                                           │
│  optimizer                                                                │
│  ├─ System: optimizer.txt + images.txt (~5,319 chars)                     │
│  ├─ Inputs:                                                               │
│  │   current_draft:  7,000 chars (≈2,000 tokens)                          │
│  │   6× feedback:    300 chars each = 1,800 chars                         │
│  │   targeted_research: 1,200 chars                                       │
│  │   misc: ~400 chars                                                     │
│  │   Total input: ~11,719 chars ≈ 3,348 tokens                            │
│  ├─ LLM: prose provider, max_tokens=3,500                                 │
│  ├─ Uses stream_invoke() — tokens pushed to state.stream_queue            │
│  └─ Writes: optimized_blog, increments optimizer_iteration               │
│                                                                           │
│  evaluator_post  (same EvaluatorAgent, scores optimized_blog)             │
│                                                                           │
│  optimizer_router                                                         │
│  ├─ any score < 70 AND _targeted_runs < 3 → targeted_researcher          │
│  ├─ all scores ≥ 90 → END                                                 │
│  ├─ any score < 90 AND optimizer_iteration < max_optimizer_passes         │
│  │    → optimizer (polish pass)                                           │
│  └─ cap reached → END                                                     │
│                                                                           │
│  targeted_researcher  (only when score < 70)                              │
│  ├─ Builds dimension-specific Tavily query from URL slug + suffixes       │
│  ├─ Checks targeted ResearchCache                                         │
│  └─ Writes: targeted_research_output                                      │
└───────────────────────────────────────────────────────────────────────────┘
            │
            ▼
  state.optimized_blog (or aggregated_blog as fallback)
  JobWriter.stop(run_id=...)      ← writes final job state to SQLite
  BlogDatabase.save_run(...)      ← persists all scores + article text
            │
            ▼
  Streamlit Active Jobs tab picks up completed state
  History tab shows before/after score comparison
```

---

## 3. Project File Structure

```
blog/
├── streamlit_app.py          # Three-tab Streamlit UI (New Blog / Active Jobs / History)
├── app.py                    # Minimal CLI entry point
│
├── graph/
│   ├── graph.py              # LangGraph DAG — 16 nodes, all edges, _wrap() lazy loader
│   ├── router.py             # evaluation_router + optimizer_router with exact caps
│   └── state.py              # BlogState Pydantic model — single state object for all nodes
│
├── agents/
│   ├── baseline_evaluator.py # Scores original blog before any enhancement
│   ├── learner.py            # Analyses blog intent, audience, SEO/GEO gaps
│   ├── planner.py            # Produces flat JSON SEO plan + research_query
│   ├── supervisor.py         # One search → 5 focused briefs (supervisor pattern)
│   ├── language_agent.py     # Prose/tone review using language_research brief
│   ├── facts_agent.py        # E-E-A-T review using facts_research brief
│   ├── structure_agent.py    # Article skeleton using structure_research brief
│   ├── image_agent.py        # Image search, Nemotron scoring, download
│   ├── seo_agent.py          # Keyword audit using seo_research brief
│   ├── geo_agent.py          # Entity/definition audit using geo_research brief
│   ├── aggregator.py         # First draft writer — streaming, capped inputs
│   ├── evaluator.py          # Scores 6 dimensions → JSON, always Groq
│   ├── optimizer.py          # Polish pass — streaming, capped draft + feedback
│   └── targeted_researcher.py # Dimension-specific research when score < 70
│
├── prompts/                  # All system prompts as .txt files
│   ├── learner.txt           # JSON schema for blog analysis
│   ├── planner.txt           # Flat JSON SEO + content plan
│   ├── supervisor.txt        # 5-brief format (no literal {} to avoid LangChain escape issues)
│   ├── language.txt          # 7-section prose audit with emoji check
│   ├── facts.txt             # E-E-A-T audit with exact-fix format
│   ├── structure.txt         # Skeleton + thin section flags
│   ├── seo.txt               # 8-step keyword/heading/FAQ audit
│   ├── geo.txt               # Entity/definition/direct-answer audit
│   ├── images.txt            # Image placement rules (appended to aggregator + optimizer)
│   ├── aggregator.txt        # First draft system prompt with prose benchmark
│   ├── optimizer.txt         # Polish pass prompt with structure repair priority
│   ├── evaluator.txt         # Start-at-100 deduction tables, 6 dimensions
│   └── targeted_researcher.txt # Dimension-specific research guide
│
├── config/
│   ├── config.py             # Settings class — loads .env, exposes all keys
│   ├── __init__.py           # Exports settings singleton
│   ├── llm.py                # RotatingChatGroq — 2-model pool, TPD/TPM/413 handling
│   ├── llm_registry.py       # make_llm() factory — mixed provider routing, force_groq param
│   ├── openai_llm.py         # OpenAIChatLLM — GPT-4o with exponential backoff
│   ├── gemini_llm.py         # GeminiChatLLM — flash → flash-lite → 1.5-flash fallback chain
│   ├── embeddings.py         # Unified embedding router: openai / gemini / huggingface
│   ├── pinecone_manager.py   # Creates/gets Pinecone index, auto-recreates on dim mismatch
│   ├── tavily_search.py      # Tavily web search + image candidate search
│   ├── firecrawl_manager.py  # Firecrawl fallback image search
│   └── nemotron_image_selector.py # Nemotron vision scoring + image download
│
├── rag/
│   ├── scraper.py            # Crawl4AI async scrape + requests fallback
│   ├── html_cleaner.py       # BeautifulSoup — removes nav/footer/script/style
│   ├── markdown_converter.py # markdownify — HTML → ATX markdown
│   ├── chunker.py            # RecursiveCharacterTextSplitter
│   ├── ingestion.py          # Embed + upsert to Pinecone
│   └── retriever.py          # Pinecone similarity retriever (available, not active in pipeline)
│
├── db/
│   ├── database.py           # SQLite WAL, 3 tables, all CRUD + job methods
│   ├── models.py             # Run + Article dataclasses
│   └── __init__.py           # Exports BlogDatabase
│
├── utilis/
│   ├── retry.py              # invoke_with_retry — single LLM call-site, handles rotation
│   ├── job_writer.py         # Thread-safe progress writer to SQLite, 0.8s flush loop
│   ├── research_cache.py     # TF-IDF cosine similarity cache, 6hr TTL, 2 singletons
│   ├── json_parser.py        # Robust JSON extraction from LLM output
│   └── prompt_loader.py      # Loads .txt files from /prompts/
│
├── data/
│   └── blog_enhancer.db      # SQLite database (auto-created on first run)
│
├── generated_images/         # Downloaded article images
├── requirements.txt          # Pinned dependencies
└── .env                      # API keys and model configuration
```

---

## 4. LangGraph DAG — Nodes and Edges

All nodes are registered with `_wrap(agent_name, agent_class)` which:
1. Lazily instantiates the agent on first call (so `make_llm()` runs after `set_provider()`)
2. Stamps `state.active_agent = agent_name` so the UI shows live progress

```
prepare → baseline_evaluator → learner → planner → supervisor
→ language → facts → structure → image → seo → geo
→ aggregator → evaluator
              ├─(evaluation_router)──┐
              │  freshness < 70 → supervisor
              │  language  < 70 → language
              │  facts     < 70 → facts
              │  structure < 70 → structure
              │  seo       < 70 → seo
              │  geo       < 70 → geo
              │  (all ≥ 70 or iter ≥ 3) → optimizer
              └──────────────────────┘
optimizer → evaluator_post
              ├─(optimizer_router)───┐
              │  any < 70, runs < 3 → targeted_researcher → optimizer
              │  all ≥ 90           → END
              │  any < 90, iter < max_optimizer_passes → optimizer
              └─ cap reached        → END
```

**Special wiring:** `evaluation_router` maps the key `"researcher"` to the `"supervisor"` node — so when freshness < 70, the full supervisor research pipeline refreshes, not just a text search.

---

## 5. BlogState — Complete Field Reference

`BlogState` is a single Pydantic `BaseModel`. Every node receives it, mutates its own fields, and returns it. `stream_queue` is excluded from Pydantic serialisation (`exclude=True`) and uses `arbitrary_types_allowed = True`.

| Field | Type | Default | Written by | Purpose |
|---|---|---|---|---|
| `url` | str | — | User | Source blog URL |
| `raw_blog` | str | "" | prepare | Raw scraped HTML |
| `cleaned_blog` | str | "" | prepare | ATX markdown after HTML cleaning |
| `chunks` | list[str] | [] | prepare | Text chunks for Pinecone ingestion |
| `retrieved_context` | str | "" | (unused) | Reserved for RAG retrieval |
| `target_length` | str | "~5 pages…" | User | Length instruction for all agents |
| `max_pages` | int | 5 | User | Controls target_length computation |
| `language_quality` | str | "medium" | User | easy / medium / advanced |
| `research_level` | str | "medium" | User | Controls Tavily max_results |
| `research_results` | int | 0 | User | Explicit Tavily count override (0 = use level) |
| `image_count` | int | 3 | User | Target number of images |
| `current_date` | str | today ISO | auto | Passed to all agents |
| `current_year` | int | today year | auto | Passed to all agents |
| `learner_output` | dict | {} | learner | intent, audience, tone, seo_gaps, geo_gaps |
| `planner_output` | dict | {} | planner | primary_keyword, h1_title, h2_headings, faq_questions, secondary_keywords, research_query, seo_plan, geo_plan, target_outline |
| `research_output` | str | "" | supervisor | Raw trimmed Tavily results |
| `baseline_language_score` | int | 0 | baseline_evaluator | Original blog score |
| `baseline_facts_score` | int | 0 | baseline_evaluator | |
| `baseline_structure_score` | int | 0 | baseline_evaluator | |
| `baseline_seo_score` | int | 0 | baseline_evaluator | |
| `baseline_geo_score` | int | 0 | baseline_evaluator | |
| `baseline_freshness_score` | int | 0 | baseline_evaluator | |
| `baseline_overall_score` | int | 0 | baseline_evaluator | Mean of 6 baseline scores |
| `language_research` | str | "" | supervisor | Focused tone/audience brief |
| `facts_research` | str | "" | supervisor | Focused stats/examples brief |
| `structure_research` | str | "" | supervisor | Full article skeleton brief |
| `seo_research` | str | "" | supervisor | Keyword/heading/FAQ brief |
| `geo_research` | str | "" | supervisor | Entity/definition/answer brief |
| `language_output` | str | "" | language_agent | Prose review notes |
| `facts_output` | str | "" | facts_agent | E-E-A-T review notes |
| `structure_output` | str | "" | structure_agent | Skeleton + thin section flags |
| `seo_output` | str | "" | seo_agent | Keyword audit notes |
| `geo_output` | str | "" | geo_agent | Entity/definition audit notes |
| `image_output` | list[dict] | [] | image_agent | Selected images with url/alt/caption/placement |
| `aggregated_blog` | str | "" | aggregator | First complete article draft |
| `editorial_brief` | str | "" | aggregator | Cleared on each pass (legacy) |
| `optimized_blog` | str | "" | optimizer | Current best article version |
| `targeted_research_output` | str | "" | targeted_researcher | Dimension-specific research brief |
| `language_score` | int | 0 | evaluator | 0–100 |
| `facts_score` | int | 0 | evaluator | 0–100 |
| `structure_score` | int | 0 | evaluator | 0–100 |
| `seo_score` | int | 0 | evaluator | 0–100 |
| `geo_score` | int | 0 | evaluator | 0–100 |
| `freshness_score` | int | 0 | evaluator | 0–100 |
| `overall_score` | int | 0 | evaluator | Mean of 6 scores |
| `language_feedback` | str | "" | evaluator | Exact fix instructions |
| `facts_feedback` | str | "" | evaluator | |
| `structure_feedback` | str | "" | evaluator | |
| `seo_feedback` | str | "" | evaluator | |
| `geo_feedback` | str | "" | evaluator | |
| `freshness_feedback` | str | "" | evaluator | |
| `iteration` | int | 0 | evaluator | Total evaluation cycles |
| `optimizer_iteration` | int | 0 | optimizer | Optimizer-specific pass count |
| `finished` | bool | False | (unused) | Reserved |
| `max_optimizer_passes` | int | 3 | User | Cap for optimizer_router |
| `llm_provider` | str | "groq" | User | "groq" / "gemini" / "openai" |
| `active_agent` | str | "" | each node | Current agent for UI display |
| `stream_queue` | Optional[Any] | None | User | queue.Queue for streaming tokens |

**Key state methods:**

| Method | Behaviour |
|---|---|
| `blog_snippet(max_chars=2500)` | Truncated `cleaned_blog` at paragraph boundary |
| `research_snippet(max_chars=1200)` | Truncated `research_output` |
| `plan_summary()` | Flat text of planner_output key fields (missing topics, keywords, SEO plan, outline) |
| `specialist_outputs_brief(max_chars_each=700)` | Returns dict of truncated specialist outputs; structure and SEO get 1,400 chars each |
| `_truncate(text, max_chars)` | Hard-truncate at newline boundary |
| `stream_chunk(token)` | Non-blocking `queue.put_nowait(token)` |
| `stream_done()` | Puts `None` sentinel into queue |

---

## 6. Agent Reference — Every Agent in Detail

### baseline_evaluator
- **Prompt:** `evaluator.txt`
- **Inputs:** `cleaned_blog` (max 5,000 chars), date, year, quality, length
- **LLM:** Groq `force_groq=True`, `max_tokens=800`
- **Output:** `baseline_language/facts/structure/seo/geo/freshness/overall_score`
- **Runs:** Once, immediately after prepare, before anything else
- **Purpose:** Establish before-score for comparison table

### learner
- **Prompt:** `learner.txt`
- **Inputs:** `blog_snippet(2500)`, `target_length`
- **LLM:** Groq `force_groq=True`, `max_tokens=600`
- **Output:** `learner_output` dict — intent, audience, tone, search_intent, geo_intent, topic_clusters, seo_strengths, seo_gaps, geo_gaps, structure_gaps, recommended_depth
- **Purpose:** Contextualise the source blog for the planner

### planner
- **Prompt:** `planner.txt`
- **Inputs:** `blog_snippet(2500)`, `learner_output`, date, year, length
- **LLM:** Groq `force_groq=True`, `max_tokens=600`
- **Output:** `planner_output` dict — primary_keyword, h1_title, h2_headings (list), faq_questions (list of H3 strings), secondary_keywords (list 8+), research_query, seo_plan (pipe-delimited string), geo_plan, target_outline, missing_topics/keywords/entities/faqs/statistics/examples, weak_sections
- **Fallback:** If `research_query` is missing, constructs one from `url + year + "latest statistics examples SEO GEO keywords"`

### supervisor
- **Prompt:** `supervisor.txt`
- **Inputs:** Tavily results (max 4,000 chars), plan_summary(), primary_keyword, h1_title, h2_headings, faq_questions, secondary_keywords, date, year, length
- **LLM:** Groq `force_groq=True`, `max_tokens=2000`
- **Cache:** ResearchCache (TF-IDF cosine ≥ 0.82, 6hr TTL) — skips Tavily + LLM on hit
- **Output:** 5 dedicated brief fields (language_research, facts_research, structure_research, seo_research, geo_research) + raw research_output
- **Fallback:** If JSON parse fails, all 5 brief fields receive the raw LLM output string
- **Trim logic:** Each Tavily result capped at 400 chars; total search block capped at 4,000 chars

### language_agent
- **Prompt:** `language.txt`
- **Inputs:** `blog_snippet(2500)`, `language_research(800)`, language_quality, target_length
- **LLM:** Groq `force_groq=True`, `max_tokens=700`
- **Output:** `language_output` — 7-section prose audit: emoji audit, hook assessment, banned phrases, rhythm, weak words, section openings, transitions

### facts_agent
- **Prompt:** `facts.txt`
- **Inputs:** `blog_snippet(2500)`, `facts_research(1000)`, date, year, length
- **LLM:** Groq `force_groq=True`, `max_tokens=700`
- **Output:** `facts_output` — unsupported claims, stats to add, named examples, expert authority, FAQ accuracy gaps

### structure_agent
- **Prompt:** `structure.txt`
- **Inputs:** `blog_snippet(2500)`, `structure_research(1200)`, target_length
- **LLM:** Groq `force_groq=True`, `max_tokens=700`
- **Output:** `structure_output` — mandatory checklist, pre-built skeleton with H1/H2/H3/FAQ/CTA, thin section flags (CRITICAL label for 0–1 paragraph sections), FAQ audit, depth gaps

### image_agent
- **Inputs:** `cleaned_blog` (always — NOT structure_output)
- **LLM:** None (no LLM calls)
- **Process:**
  1. `_context_windows(N)` — splits by H1/H2/H3, skips FAQ and Conclusion
  2. Per section: `_image_query()` builds topic-anchored Tavily image search query
  3. Tavily `search_image_candidates()` — returns image URLs from search results
  4. `_filter_candidates()` — rejects cartoons, logos, stock sites, non-photo extensions
  5. `select_relevant_images()` (Nemotron via OpenRouter) — vision scoring, score ≥ 60 required, returns empty if nothing passes
  6. Firecrawl `search_images()` fallback if Nemotron rejects everything
  7. `download_image()` — saves to `/generated_images/{slug}-{url_hash}.ext`, replaces remote URL with local path
- **Output:** `image_output` list of dicts: url (local path), alt, caption, source_url, placement, relevance_score

### seo_agent
- **Prompt:** `seo.txt`
- **Inputs:** `blog_snippet(2500)`, `seo_research(1200)`, date, year, length
- **LLM:** Groq `force_groq=True`, `max_tokens=700`
- **Output:** `seo_output` — 8-step audit: primary keyword identification + placement audit, secondary keyword gaps with exact insertion sentences, H2/H3 heading rewrites, FAQ block audit with exact replacement questions, featured snippet opportunities

### geo_agent
- **Prompt:** `geo.txt`
- **Inputs:** `seo_output(900)`, `geo_research(1000)`, date, year, length
- **LLM:** Groq `force_groq=True`, `max_tokens=700`
- **Output:** `geo_output` — entity audit with 1-sentence descriptions, 3 definition blocks (exact text), 4 direct answer passages (exact text), citation-worthy statements, extractability rating

### aggregator
- **Prompt:** `aggregator.txt` + `images.txt` (system), structured human prompt
- **Inputs (hard-capped for 8K TPM compliance):**
  - blog: 1,800 chars
  - language notes: 500 chars
  - facts notes: 500 chars
  - structure notes: 900 chars
  - seo notes: 900 chars
  - geo notes: 400 chars
  - misc: ~300 chars
- **LLM:** Prose provider (no force_groq), `max_tokens=3,500`
- **Streaming:** Uses `stream_invoke()` — tokens pushed to `state.stream_queue`
- **Output:** `aggregated_blog` — first complete article (2800–3200 words target)

### evaluator
- **Prompt:** `evaluator.txt`
- **Inputs:** `optimized_blog or aggregated_blog` (max 9,000 chars), date, year, quality, length
- **LLM:** Groq `force_groq=True`, `max_tokens=1,200`
- **Output:** 6 scores + 6 feedback strings + overall_score, increments `state.iteration`
- **Score method:** Start at 100, deduct per detected flaw (see Section 17)
- **JSON extraction:** `load_json()` handles markdown fences and prose prefix

### optimizer
- **Prompt:** `optimizer.txt` + `images.txt` (system), structured human prompt
- **Inputs (hard-capped):**
  - current_draft: 7,000 chars (truncated at newline)
  - 6× feedback strings: 300 chars each
  - targeted_research: 1,200 chars
  - misc: ~400 chars
- **LLM:** Prose provider (no force_groq), `max_tokens=3,500`
- **Streaming:** Uses `stream_invoke()`
- **Output:** `optimized_blog`, increments `optimizer_iteration`

### targeted_researcher
- **Prompt:** `targeted_researcher.txt`
- **Triggers:** Only when any score < 70 in `optimizer_router` AND `_targeted_runs < 3`
- **Query building:** URL slug + dimension-specific suffixes (e.g., SEO → "primary keywords LSI keywords search intent FAQ questions SEO")
- **Cache:** `get_targeted_cache()` — separate cache from supervisor's cache
- **LLM:** Groq `force_groq=True`, `size="medium"` (1,200 tokens)
- **Output:** `targeted_research_output` — markdown brief with section per failing dimension

---

## 7. Routing Logic — Exact Conditions and Caps

### evaluation_router (post-aggregator first evaluation)

```python
MAX_SPECIALIST_ITERATIONS = 3

if state.iteration >= 3:              return "optimizer"  # force forward
if state.freshness_score < 70:        return "researcher"  # maps to "supervisor" node
if state.language_score  < 70:        return "language"
if state.facts_score     < 70:        return "facts"
if state.structure_score < 70:        return "structure"
if state.seo_score       < 70:        return "seo"
if state.geo_score       < 70:        return "geo"
return "optimizer"                    # all ≥ 70
```

A failed specialist re-runs and feeds back into aggregator → evaluator. Priority order matters: freshness is checked first because stale data contaminates all other dimensions.

### optimizer_router (post optimizer evaluator_post)

```python
MAX_OPTIMIZER_ITERATIONS = 8   # absolute hard cap
MAX_TARGETED_ITERATIONS  = 3   # max times targeted_researcher fires per run

needs_research = any(score < 70 for score in all_6_scores)
targeted_runs  = getattr(state, "_targeted_runs", 0)  # side-channel counter

if needs_research and targeted_runs < 3:
    _targeted_runs += 1
    return "targeted_researcher"

if all(score >= 90 for score in all_6_scores):
    return "end"

if state.optimizer_iteration >= state.max_optimizer_passes:  # user-set, default 3
    return "end"

return "optimizer"
```

`max_optimizer_passes` is a `BlogState` field set from the UI slider (1–8). It is compared against `optimizer_iteration`, not against `MAX_OPTIMIZER_ITERATIONS`. The hard cap `MAX_OPTIMIZER_ITERATIONS = 8` is the ceiling that `max_optimizer_passes` cannot exceed.

`_targeted_runs` is tracked via `object.__setattr__` (side-channel) because BlogState is a Pydantic model and this counter shouldn't be part of the serialisable state.

---

## 8. LLM Layer — Providers, Rotation, Mixed Routing

### Mixed-Provider Design

The critical design decision: **analysis agents always use Groq, prose agents use the selected provider.**

| Agent group | Provider | Why |
|---|---|---|
| baseline_evaluator, evaluator, evaluator_post | Groq (forced) | JSON output, short; preserves Gemini/OpenAI quota |
| learner, planner | Groq (forced) | JSON output, short |
| supervisor | Groq (forced) | Research synthesis; 2,000 token output fits Groq |
| language, facts, structure, seo, geo | Groq (forced) | Review notes; 700 token output |
| targeted_researcher | Groq (forced) | Research brief; 1,200 token output |
| **aggregator, optimizer** | **Prose provider** | Long-form prose; needs 3,500 output tokens |

`make_llm(force_groq=True)` always returns `RotatingChatGroq` regardless of `_provider`. `make_llm()` without `force_groq` returns the globally set prose provider.

### RotatingChatGroq (`config/llm.py`)

**Rotation pool:**
```python
_MODEL_POOL = [
    "llama-3.3-70b-versatile",   # primary — ~6,000 TPM free tier
    "qwen/qwen3.6-27b",          # fallback on TPD exhaustion
]
```
`openai/gpt-oss-120b` is excluded: its 8,000 TPM limit is too small for aggregator/optimizer when output needs 3,500 tokens.

**Error handling:**
- `RateLimitError` with "tokens per day" / "tpd" → `_rotate()` — marks model exhausted, switches to next, raises `_RotationOccurred` sentinel
- `RateLimitError` with "tokens per minute" / "tpm" → sleep `_parse_wait(msg)` or exponential backoff (up to 120s), retry same model
- `APIStatusError` 413 → parse `Requested N / Limit M` from message; if `N > M * 0.95` sleep 60s then retry; else short 15s sleep and retry
- `APIStatusError` 5xx → sleep 10s, re-raise

**`invoke_with_retry` (`utilis/retry.py`):** Catches `_RotationOccurred`, rebuilds `chain = prompt | llm._client` with the newly active model, retries up to `_MAX_ROTATIONS = 3` times.

### GeminiChatLLM (`config/gemini_llm.py`)

**Fallback chain:** `gemini-2.0-flash` → `gemini-2.0-flash-lite` → `gemini-1.5-flash` → `gemini-1.5-flash-8b`

Rate-limit (429 / `resource_exhausted` / `quota`) → try next fallback first; if all exhausted, sleep with exponential backoff (up to 90s).
Model-not-found → immediately try next fallback.
5xx → sleep 10s.

### OpenAIChatLLM (`config/openai_llm.py`)

**Fallback chain:** `gpt-4o` → `gpt-4-turbo` → `gpt-4` → `gpt-3.5-turbo`

429 RateLimitError → exponential backoff 10s → 20s → 40s → ... up to 120s.
400/413 context error → try next smaller model.

### stream_invoke (`config/llm_registry.py`)

Used by aggregator and optimizer only. Calls `chain.stream(inputs)` and pushes tokens to `chunk_callback`. **Does NOT catch `RateLimitError` or `APIStatusError`** — those surface to `_invoke_chain()` so rotation/backoff logic fires correctly. Only catches generic errors matching "stream/not support/chunk" patterns to fall back to `invoke_with_retry`.

---

## 9. Supervisor — Research Architecture

### Old vs New

| | Old ResearchAgent | New SupervisorAgent |
|---|---|---|
| Output | One generic markdown brief | Five focused briefs (one per specialist) |
| What each specialist receives | Same generic document — extracts relevant parts | Only what it needs for its dimension |
| Tavily calls | 1 | 1 (identical) |
| LLM calls | 1 | 1 (identical) |
| Cost | Same | Same |
| Latency | Same | Same |

### Supervisor Output Fields

Each brief is a string value inside the returned JSON (no literal `{}` in `supervisor.txt` to avoid LangChain template escaping issues):

| JSON key | State field | Content |
|---|---|---|
| `language_brief` | `language_research` | Audience profile, tone, credibility signals, hook angle with real stat |
| `facts_brief` | `facts_research` | 5+ sourced stats, 3+ named examples with outcomes, expert findings |
| `structure_brief` | `structure_research` | Full skeleton — H1 → H2s → paragraph topics → H3s → FAQ → CTA |
| `seo_brief` | `seo_research` | Primary keyword, exact H2s, secondary keywords, FAQ H3s, snippet target |
| `geo_brief` | `geo_research` | Entities with descriptions, 3 definition blocks, 4 direct answer passages |

If JSON parse fails, all 5 fields receive the raw LLM output string as a safe fallback.

---

## 10. Image Pipeline

### Full Flow

```
cleaned_blog (always — NOT structure_output, which contains review notes)
        │
_context_windows(image_count)
Split article by H1/H2/H3 headings into N windows.
Each window: heading text + first 800 chars of body as context.
Skip FAQ and Conclusion sections (no useful images).
        │
For each window:
  _image_query(topic, heading, context_words, year)
  → Tavily search_image_candidates(query, max_results=8)
        │
  _filter_candidates(candidates)
  Reject URLs matching: netflix/movie/cartoon/clipart/icon/logo/
    stock photo sites, non-photo extensions (.svg, .gif excluded)
  Require extension: .jpg/.jpeg/.png/.webp
        │
  select_relevant_images(context, section, candidates, limit=1)
  Nemotron vision model via OpenRouter:
    - Scores each candidate 0–100
    - Rejects score < 60
    - Returns empty list if nothing passes (does NOT fall back to random candidates)
        │
  If empty → Firecrawl search_images(query, limit=3, context, placement)
    → HTML og:image / twitter:image extraction
    → Nemotron re-score
        │
  download_image(image, section)
  → requests.get(url, timeout=20)
  → saved to /generated_images/{safe_section}-{url_hash_12chars}.ext
  → image["url"] = local file path
  → image["remote_url"] = original URL preserved
        │
state.image_output = list of selected images (up to image_count)
```

### Image Dict Schema
Each entry in `image_output`:
```python
{
    "url":             str,   # local path after download
    "remote_url":      str,   # original remote URL
    "local_path":      str,   # same as url after download
    "alt":             str,   # descriptive alt text
    "caption":         str,   # short caption for display
    "source_url":      str,   # page the image came from
    "context":         str,   # article section text used for query
    "placement":       str,   # which section to place it after
    "relevance_score": int,   # Nemotron score 0–100
    "relevance_reason":str,   # Nemotron justification
}
```

---

## 11. RAG Layer

The RAG layer handles ingestion at the start of every run. Retrieval is available but not currently wired into agent prompts — agents use `blog_snippet()` and `research_snippet()` from state directly.

| File | What it does |
|---|---|
| `rag/scraper.py` | Crawl4AI async scrape with `requests` fallback. Returns `{"html": str, ...}`. |
| `rag/html_cleaner.py` | BeautifulSoup — removes `script, style, svg, iframe, noscript, footer, header, nav, aside`. |
| `rag/markdown_converter.py` | `markdownify` — converts cleaned HTML to ATX markdown stored as `cleaned_blog`. |
| `rag/chunker.py` | `RecursiveCharacterTextSplitter` using `CHUNK_SIZE` / `CHUNK_OVERLAP` from `.env`. Splits at `\n\n`, `\n`, `. `, ` `. |
| `rag/ingestion.py` | Embeds chunks via `config/embeddings.py`, upserts to Pinecone with `{url, chunk_index}` metadata. |
| `rag/retriever.py` | Pinecone similarity retriever returning `TOP_K` chunks. Available, not active. |

### Embedding Provider Routing (`config/embeddings.py`)

`EMBEDDING_PROVIDER` in `.env` controls which backend loads at import time:

| Provider | Model | Dimensions | API Key |
|---|---|---|---|
| `openai` | `text-embedding-3-large` | 3072 | `OPENAI_API_KEY` |
| `gemini` | `models/gemini-embedding-001` | 3072 | `GEMINI_API_KEY` (free) |
| `huggingface` | `sentence-transformers/all-MiniLM-L6-v2` | 384 | None (local) |
| `auto` | Detected from model name | varies | — |

**Auto-detection rules:** model name starts with `text-embedding-3` → openai; contains `gemini-embedding` or starts with `models/gemini` → gemini; anything else → huggingface.

**Pinecone dimension mismatch:** `pinecone_manager.py` checks stored index dimension against `EMBEDDING_DIMENSION`. If they differ, the index is deleted and recreated automatically. All previously ingested vectors are lost — the next `ingest_blog()` repopulates it.

---

## 12. Persistence Layer — SQLite Schema

**File:** `blog/data/blog_enhancer.db`  
**Mode:** WAL (Write-Ahead Logging) — concurrent reads while writing  
**Thread safety:** Each method opens its own connection — safe for background threads  
**Auto-created:** On first import of `db.database`

### Table: `runs`
One row per completed or failed enhancement run.

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK AUTOINCREMENT | |
| `url` | TEXT NOT NULL | Source blog URL |
| `created_at` | TEXT NOT NULL | ISO-8601 UTC |
| `duration_seconds` | REAL DEFAULT 0 | Wall-clock time |
| `llm_provider` | TEXT DEFAULT '' | groq / gemini / openai |
| `research_level` | TEXT DEFAULT '' | easy / medium / advanced |
| `language_quality` | TEXT DEFAULT '' | easy / medium / advanced |
| `max_pages` | INTEGER DEFAULT 5 | |
| `image_count` | INTEGER DEFAULT 3 | |
| `optimizer_iterations` | INTEGER DEFAULT 0 | |
| `evaluation_iterations` | INTEGER DEFAULT 0 | |
| `baseline_overall` | INTEGER DEFAULT 0 | Original blog score |
| `baseline_language` | INTEGER DEFAULT 0 | |
| `baseline_facts` | INTEGER DEFAULT 0 | |
| `baseline_structure` | INTEGER DEFAULT 0 | |
| `baseline_seo` | INTEGER DEFAULT 0 | |
| `baseline_geo` | INTEGER DEFAULT 0 | |
| `baseline_freshness` | INTEGER DEFAULT 0 | |
| `enhanced_overall` | INTEGER DEFAULT 0 | Final enhanced score |
| `enhanced_language` | INTEGER DEFAULT 0 | |
| `enhanced_facts` | INTEGER DEFAULT 0 | |
| `enhanced_structure` | INTEGER DEFAULT 0 | |
| `enhanced_seo` | INTEGER DEFAULT 0 | |
| `enhanced_geo` | INTEGER DEFAULT 0 | |
| `enhanced_freshness` | INTEGER DEFAULT 0 | |
| `status` | TEXT DEFAULT 'completed' | completed / failed |
| `error_message` | TEXT DEFAULT '' | Truncated to 2,000 chars |

Indexed on `created_at DESC` and `url`.

### Table: `articles`
One row per run — full text content (separated so list queries stay fast).

| Column | Type | Notes |
|---|---|---|
| `run_id` | INTEGER PK | FK → runs(id) ON DELETE CASCADE |
| `original_blog` | TEXT DEFAULT '' | `cleaned_blog` — original scraped markdown |
| `enhanced_blog` | TEXT DEFAULT '' | `optimized_blog` or `aggregated_blog` |

### Table: `jobs`
One row per active or recently completed generation job. Survives page refreshes.

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK AUTOINCREMENT | |
| `run_id` | INTEGER | Populated on completion, NULL while running |
| `url` | TEXT NOT NULL | Source URL |
| `status` | TEXT DEFAULT 'running' | running / completed / failed |
| `active_agent` | TEXT DEFAULT '' | Current agent name |
| `progress_pct` | INTEGER DEFAULT 0 | 0–100 |
| `streamed_text` | TEXT DEFAULT '' | Last 12,000 chars of streamed article |
| `error_message` | TEXT DEFAULT '' | Truncated to 2,000 chars |
| `started_at` | TEXT NOT NULL | ISO-8601 UTC |
| `updated_at` | TEXT NOT NULL | ISO-8601 UTC, updated every 0.8s |
| `cancel_requested` | INTEGER DEFAULT 0 | 1 = cancel requested by user |

Indexed on `status`.

### BlogDatabase Methods

| Method | Signature | Notes |
|---|---|---|
| `save_run(...)` | All score fields as kwargs | Returns `run_id` |
| `save_failed_run(url, error_message)` | | Records crash before completion |
| `list_runs(limit=100)` | | Returns `list[Run]`, newest first |
| `get_run(run_id)` | | Returns `Run` or `None` |
| `get_article(run_id)` | | Returns `Article` or `None` |
| `delete_run(run_id)` | | CASCADE deletes article |
| `create_job(url)` | | Returns `job_id`, status='running' |
| `update_job(job_id, *, active_agent, progress_pct, streamed_text)` | | Keeps last 12,000 chars of stream |
| `finish_job(job_id, run_id, error)` | | Sets status=completed/failed, progress_pct=100 |
| `cancel_job(job_id)` | | Sets cancel_requested=1 |
| `is_cancel_requested(job_id)` | | Returns bool |
| `get_job(job_id)` | | Returns raw sqlite3.Row |
| `list_active_jobs()` | | Returns all status='running' rows |

---

## 13. Fault Tolerance — Jobs System

### Problem
Streamlit reruns the entire script on every page refresh or button click. A `threading.Thread` started in the script body would be tied to that script execution. A page refresh would abandon the running thread, and the user would have no way to observe or resume a long generation run.

### Solution
All generation progress is written to SQLite via `JobWriter`. The UI reads from the DB on every render cycle — it does not need any in-memory reference to the thread.

### How It Works

```
User clicks Generate
        │
        ▼
_db.create_job(url)                 ← jobs row, status='running'
JobWriter(db, job_id).start()       ← starts _flush_loop daemon thread
        │
        ▼
threading.Thread(_run_generation, ...).start()
        │
        ├─ Inner graph thread:  graph.invoke(state)
        │
        └─ Outer monitor loop (0.3s tick):
             ├─ Reads state.active_agent
             ├─ jw.set_agent(agent, pct)
             ├─ Drains state.stream_queue → jw.append_stream(chunk)
             └─ Checks jw.is_cancel_requested() every tick

JobWriter._flush_loop (0.8s tick):
        └─ db.update_job(job_id, active_agent, progress_pct, streamed_text)

On completion:
        └─ jw.stop(run_id=...) → db.finish_job(...) → status='completed'

On cancel:
        └─ db.cancel_job(job_id) → cancel_requested=1
           monitor loop detects → sets result_holder["error"] = "Cancelled"
           jw.stop(error="Cancelled") → status='failed'
```

### Page Refresh Behaviour

When the user refreshes the page:
1. Streamlit re-runs `streamlit_app.py` from scratch
2. The background generation thread continues — it is a daemon thread in the same Python process
3. The **Active Jobs** tab calls `_db.list_active_jobs()` → finds the running job row
4. Displays `active_agent`, `progress_pct`, `streamed_text` from DB
5. Calls `st.rerun()` every 2 seconds (via `time.sleep(0.1); st.rerun()`) to keep the display live

**Limitation:** If the Streamlit process itself is killed (e.g. server restart), the daemon thread also dies. The job row remains as `status='running'` in the DB. To clean up stale jobs, the system relies on the `updated_at` timestamp — a job that hasn't updated in > 10 minutes is effectively dead. (Automatic stale job detection is a future improvement.)

### JobWriter Class

```python
class JobWriter:
    _FLUSH_INTERVAL = 0.8  # seconds

    def start()                          # starts background flush thread
    def set_agent(agent: str, pct: int)  # thread-safe via _lock
    def append_stream(token: str)        # thread-safe accumulation
    def is_cancel_requested() → bool     # polls DB directly
    def stop(run_id=None, error="")      # stops flush, writes final state, calls finish_job
```

---

## 14. Research Cache

**File:** `utilis/research_cache.py`

In-memory, process-scoped cache. Prevents redundant Tavily + LLM calls for near-duplicate topic queries within the same Streamlit session.

### Design

- **Key:** SHA-256 of normalised (lowercased, stopwords removed) query — first 16 hex chars
- **Similarity check:** TF-IDF term frequency cosine similarity (pure stdlib + math, no external deps)
- **Hit threshold:** cosine ≥ 0.82
- **TTL:** 6 hours (lazy eviction on next `get()` call)
- **Two singletons:** `_researcher_cache` (supervisor) and `_targeted_researcher_cache` (targeted_researcher) — separate to prevent cross-contamination between main research and targeted research

### Cache Entry Schema

```python
{
    "vec":     dict[str, float],   # TF-IDF term vector of the normalised query
    "result":  str,                # cached LLM output (full JSON string)
    "expires": float,              # time.time() + TTL
}
```

### Example: What Gets Cached

When the supervisor runs for "AI employee training corporate" it caches the full JSON response (all 5 briefs). If the same session generates a blog for "corporate AI upskilling for employees", the cosine similarity is high enough (> 0.82) to return the cached briefs without re-running Tavily or the LLM.

---

## 15. Streaming Architecture

Aggregator and optimizer push tokens to the UI as they are generated.

### Token Flow

```
LLM (chain.stream(inputs))
        │
        ▼  chunk.content per token
stream_invoke(prompt, llm, inputs, chunk_callback)
        │
        ▼  chunk_callback = state.stream_chunk
state.stream_chunk(token)
        │
        ▼  state.stream_queue.put_nowait(token)
queue.Queue (in-memory, within the process)
        │
        ├─► JobWriter._monitor_loop drains queue → jw.append_stream(token) → DB
        │
        └─► Streamlit polling loop drains queue → appends to streamed_text → renders preview
```

### Sentinel Protocol

When an agent finishes generating, it calls `state.stream_done()` which puts `None` into the queue. The consumers (monitor loop, UI loop) detect `None` and know streaming for that agent pass is complete.

### stream_invoke Safety

`stream_invoke` only falls back to `invoke_with_retry` for errors that match "stream/not support/chunk" patterns. It **re-raises** `RateLimitError` and `APIStatusError` immediately so `RotatingChatGroq._invoke_chain()` or `GeminiChatLLM._invoke_chain()` can handle them correctly with rotation/backoff. This prevents the 10-minute hang that occurred when 413 errors were silently swallowed by the fallback.

---

## 16. Token Budget Engineering

Every Groq free-tier model has an **8,000 TPM window** (tokens per minute). With `max_tokens=3,500` for prose output, input must stay under **4,500 tokens** to avoid 413 errors. With `max_tokens=700–1,200` for analysis output, input can be up to **7,300 tokens**.

### Per-Agent Budget (estimated tokens = chars ÷ 3.5)

| Agent | System (chars) | Input (chars) | Input tokens | Output tokens | Total | Status |
|---|---|---|---|---|---|---|
| evaluator | 4,057 | 9,000 | 3,730 | 1,200 | 4,930 | OK |
| baseline_evaluator | 4,057 | 5,000 | 2,587 | 800 | 3,387 | OK |
| aggregator | 6,152 | 7,700 | 3,957 | 3,500 | 7,457 | OK (prose provider) |
| optimizer | 5,319 | 11,700 | 4,862 | 3,500 | 8,362 | OK (prose provider) |
| supervisor | 2,221 | 4,000 | 1,777 | 2,000 | 3,777 | OK |
| structure | 2,425 | 2,500 | 1,407 | 700 | 2,107 | OK |
| seo | 2,079 | 2,500 | 1,308 | 700 | 2,008 | OK |
| language | 2,780 | 2,500 | 1,508 | 700 | 2,208 | OK |
| facts | 1,350 | 2,500 | 1,100 | 700 | 1,800 | OK |
| geo | 1,365 | 1,800 | 904 | 700 | 1,604 | OK |
| planner | 1,724 | 2,500 | 1,206 | 600 | 1,806 | OK |
| learner | 932 | 2,500 | 980 | 600 | 1,580 | OK |
| targeted_researcher | 1,652 | 3,500 | 1,472 | 2,000 | 3,472 | OK |

Aggregator and optimizer totals exceed 6,000 tokens because they use the prose provider (Gemini/OpenAI), not Groq's free-tier models. All `force_groq=True` agents stay well under 5,000 tokens.

### Hard Caps Enforced in Code

| Agent | Field | Cap | Where |
|---|---|---|---|
| aggregator | blog snippet | 1,800 chars | `aggregator.py:_BLOG_CHARS` |
| aggregator | language brief | 500 chars | `aggregator.py:_BRIEF_LANG` |
| aggregator | facts brief | 500 chars | `aggregator.py:_BRIEF_FACTS` |
| aggregator | structure brief | 900 chars | `aggregator.py:_BRIEF_STRUCT` |
| aggregator | SEO brief | 900 chars | `aggregator.py:_BRIEF_SEO` |
| aggregator | GEO brief | 400 chars | `aggregator.py:_BRIEF_GEO` |
| aggregator | output | 3,500 tokens | `aggregator.py:_OUTPUT_TOKENS` |
| optimizer | draft | 7,000 chars | `optimizer.py:_DRAFT_MAX_CHARS` |
| optimizer | each feedback | 300 chars | `optimizer.py:_FEEDBACK_MAX_CHARS` |
| optimizer | targeted research | 1,200 chars | `optimizer.py` inline |
| optimizer | output | 3,500 tokens | `optimizer.py:_OUTPUT_TOKENS` |
| evaluator | article | 9,000 chars | `evaluator.py:_EVAL_MAX_CHARS` |
| baseline_evaluator | article | 5,000 chars | `baseline_evaluator.py:_BASELINE_MAX_CHARS` |
| supervisor | search results | 4,000 chars total, 400 per result | `supervisor.py:_MAX_TOTAL_CHARS` |
| JobWriter | streamed_text in DB | 12,000 chars (last N chars) | `database.py:update_job` |

---

## 17. Evaluation Rubric — Deduction Tables

The evaluator uses `evaluator.txt` which implements a **start at 100, deduct per flaw** rubric. All 6 dimensions are scored simultaneously in one LLM call. The model is instructed to count exact occurrences and deduct precisely.

### Language (0–100)
| Flaw | Deduction |
|---|---|
| Each emoji anywhere | −4 |
| Each em dash (—) | −3 |
| Each banned opener ("In today's world", "In today's fast-paced", "In the modern era", "In an era of") | −5 each |
| Each banned closer ("In conclusion,", "To summarize,", "As we have seen,") | −5 each |
| Each instance of "It's worth noting / It is important to note / It goes without saying" | −5 each |
| Each "Furthermore,/Moreover,/Additionally," as empty opener | −4 each |
| Each "leverage" as filler verb | −3 |
| Each "utilize" instead of "use" | −2 |
| Each passive voice block (3+ consecutive passive sentences in one paragraph) | −5 per block |
| Each paragraph with no rhythm variation (all same-length sentences) | −4 per paragraph |
| Each filler paragraph (no entity, no number, no specific info) | −6 per paragraph |
| Weak or generic opening hook | −8 |
| Hook uses a banned phrase (additional) | −5 |

Word count penalties applied before scoring: under 2,500 words: −20 structure, −10 language. Under 2,000 words: −35 structure, −20 language.

### Facts (0–100)
| Flaw | Deduction |
|---|---|
| Each statistic with no named source | −6 |
| Each vague claim ("studies show", "experts agree", "research suggests") | −5 |
| Each named example missing measurable outcome | −4 |
| Each stale year presented as current | −6 |
| Named company examples below 3 total | −7 per missing (e.g., 1 exists = −14) |
| Sourced statistics below 5 total | −5 per missing |

### Structure (0–100)
| Flaw | Deduction |
|---|---|
| Each body section under 3 paragraphs | −6 |
| Each body section under 2 paragraphs (additional) | −4 (total −10) |
| No H3 subheadings anywhere | −10 |
| H3s in fewer than 3 body sections | −5 |
| FAQ section entirely missing | −18 |
| Each FAQ question missing below 5 | −5 |
| Each FAQ question not phrased as real search query | −4 |
| Each vague FAQ answer (no fact/number/source) | −3 |
| FAQ not using H3 format | −8 |
| Missing CTA in conclusion | −10 |
| Generic CTA ("contact us", "learn more") | −6 |
| Introduction under 2 paragraphs | −5 |
| Introduction has no sourced statistic | −4 |
| Specialist review heading appearing as article heading | −12 each |
| Missing opening hook | −6 |

### SEO (0–100)
The evaluator first states: `DETECTED PRIMARY KEYWORD: [phrase]`

| Flaw | Deduction |
|---|---|
| Primary keyword missing from H1 | −18 |
| Primary keyword missing from first 50 words | −12 |
| Primary keyword not in any H2 | −10 |
| Primary keyword in only 1 H2 (needs 2+) | −6 |
| Each missing secondary keyword below 8 total | −3 |
| Each generic H2 heading (Overview/Details/Benefits/Key Points/Background/Summary) | −6 |
| FAQ section missing | −10 |
| FAQ not in H3 format | −8 |
| Each FAQ question not phrased as real search query | −4 |
| No snippet-ready format (no numbered list, no definition block, no table) | −7 |
| Primary keyword not in opening 100 words (if not already deducted) | −6 |

### GEO (0–100)
| Flaw | Deduction |
|---|---|
| Each missing definition block below 3 total | −9 |
| Each missing direct answer passage below 4 total | −7 |
| Each named entity without 1-sentence description (max −20) | −5 |
| No quotable sourced statements (0 total) | −10 |
| Fewer than 2 quotable sourced statements | −5 |

### Freshness (0–100)
| Flaw | Deduction |
|---|---|
| Each stat/trend using stale year presented as current | −9 |
| Each section framed around stale year (e.g., "As of 2023,") | −6 |
| Core article premise is outdated | −20 |

---

## 18. Prompt Files Reference

All prompts are in `/prompts/` and loaded via `utilis/prompt_loader.py` (reads `.txt` as plain string). Sizes after optimisation:

| File | Size (chars) | Purpose |
|---|---|---|
| `learner.txt` | 932 | JSON schema for blog analysis |
| `planner.txt` | 1,724 | Flat JSON SEO + content plan |
| `supervisor.txt` | 2,221 | 5-brief format — no literal `{}` |
| `language.txt` | 2,780 | 7-section prose audit |
| `facts.txt` | 1,350 | E-E-A-T audit |
| `structure.txt` | 2,425 | Skeleton + thin section audit |
| `seo.txt` | 2,079 | 8-step keyword/heading/FAQ audit |
| `geo.txt` | 1,365 | Entity/definition audit |
| `images.txt` | 837 | Image placement rules (appended to aggregator + optimizer) |
| `aggregator.txt` | 5,315 | First draft — prose benchmark, structure, SEO, GEO, language rules |
| `optimizer.txt` | 4,482 | Polish pass — structure repair first, then all dimensions |
| `evaluator.txt` | 4,057 | Start-at-100 deduction tables for 6 dimensions |
| `targeted_researcher.txt` | 1,652 | Dimension-specific research guide |

**Critical: `supervisor.txt` must not contain literal `{` or `}` characters.** LangChain's `ChatPromptTemplate` treats them as template variables. The prompt describes the 5-brief formats using prose descriptions and pipe-separated notation instead.

---

## 19. UI — Streamlit App

### Layout

```
┌─ Sidebar ──────────────────────────────────────┐
│  Settings                                        │
│  ├─ LLM Provider (Groq / Gemini / OpenAI)        │
│  ├─ Blog URL                                     │
│  ├─ Language quality (easy/medium/advanced)      │
│  ├─ Research level + Max pages (linked slider)   │
│  ├─ Number of images (0–20)                      │
│  ├─ Search results to fetch (3–15)               │
│  ├─ Max optimisation passes (1–8, default 3)     │
│  └─ Generate Blog button                         │
│                                                  │
│  Run History (sidebar list, click to view)       │
└─────────────────────────────────────────────────┘

┌─ Main area ─────────────────────────────────────┐
│  Tab: New Blog                                   │
│  ├─ On Generate: creates job, starts thread      │
│  └─ Shows "Job #N started, switch to Active Jobs"│
│                                                  │
│  Tab: Active Jobs                                │
│  ├─ Lists all status='running' jobs from DB      │
│  ├─ Per job: Job ID, URL, agent card, progress   │
│  │   bar, live article preview (last 4,000 chars)│
│  ├─ Cancel button per job                        │
│  └─ Auto-reruns every 2s (time.sleep(0.1) +      │
│     st.rerun())                                  │
│                                                  │
│  Tab: History                                    │
│  ├─ Lists completed runs with View buttons       │
│  ├─ View: score comparison table + article tabs  │
│  └─ Lists failed runs with error messages        │
└─────────────────────────────────────────────────┘
```

### Score Comparison Table

Displays Original | Enhanced | Delta for all 6 dimensions plus Overall.

- Green (#4ade80) = score ≥ 90
- Yellow (#fbbf24) = score 75–89
- Red (#f87171) = score < 75
- Delta: ▲+N (green) / ▼N (red) / — (grey)

### Agent Progress Map (`AGENT_PCT`)

Approximate completion percentages when each agent starts:

| Agent | % |
|---|---|
| prepare | 2 |
| baseline_evaluator | 5 |
| learner | 8 |
| planner | 12 |
| supervisor | 17 |
| language | 22 |
| facts | 27 |
| structure | 32 |
| image | 37 |
| seo | 41 |
| geo | 45 |
| aggregator | 50 |
| evaluator | 62 |
| optimizer | 68 |
| targeted_researcher | 78 |
| evaluator_post | 88 |

Progress bar never reaches 100% during generation (`min(pct/100, 0.97)`). On completion, all live widgets are cleared.

### `render_blog(markdown)`

Custom markdown renderer that handles:
- Inline images: extracted via regex, rendered with `st.image()`
- Markdown tables: parsed into `pd.DataFrame`, rendered with `st.table()`
- All other content: accumulated in buffer and flushed with `st.markdown()`

---

## 20. Configuration and .env Reference

`config/config.py` — `Settings` class loads from `.env` using `python-dotenv` with `override=True`. Falls back to direct file parse for keys that `os.getenv` might miss (handles spaces around `=`).

| Variable | Used by | Notes |
|---|---|---|
| `GROQ_API_KEY` | `RotatingChatGroq` | Required for all Groq calls |
| `GROQ_MODEL` | `llm.py` | Default: `llama-3.3-70b-versatile` |
| `OPENAI_API_KEY` | `OpenAIChatLLM` | Cleared from env when provider ≠ openai |
| `OPENAI_MODEL` | `OpenAIChatLLM` | Default: `gpt-4o` |
| `GEMINI_API_KEY` | `GeminiChatLLM`, embeddings | Free key from aistudio.google.com |
| `GEMINI_MODEL` | `GeminiChatLLM` | Default: `gemini-2.0-flash` |
| `TAVILY_API_KEY` | Supervisor + image search | |
| `FIRECRAWL_API_KEY` | Firecrawl fallback image search | |
| `NEMOTRON_API_KEY` | Image relevance scoring | OpenRouter key |
| `NEMOTRON_MODEL` | `nemotron_image_selector.py` | OpenRouter model ID |
| `NEMOTRON_BASE_URL` | `nemotron_image_selector.py` | Default: `https://openrouter.ai/api/v1` |
| `PINECONE_API_KEY` | Pinecone ingestion | |
| `PINECONE_INDEX_NAME` | Pinecone | Default index: `blog-enhancer` |
| `EMBEDDING_PROVIDER` | `embeddings.py` | `openai` / `gemini` / `huggingface` / `auto` |
| `EMBEDDING_MODEL` | `embeddings.py` | e.g., `models/gemini-embedding-001` |
| `EMBEDDING_DIMENSION` | `embeddings.py`, `pinecone_manager.py` | Must match the model (gemini-embedding-001 = 3072) |
| `CHUNK_SIZE` | `rag/chunker.py` | Default: 1200 |
| `CHUNK_OVERLAP` | `rag/chunker.py` | Default: 250 |
| `TOP_K` | `rag/retriever.py` | Default: 8 |
| `HUGGINGFACEHUB_API_TOKEN` | HuggingFace embeddings | Optional |

### Recommended .env for Free-Tier Operation

```env
# LLM
GROQ_API_KEY=...
GROQ_MODEL=llama-3.3-70b-versatile
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.0-flash

# Embeddings (free via Gemini)
EMBEDDING_PROVIDER=gemini
EMBEDDING_MODEL=models/gemini-embedding-001
EMBEDDING_DIMENSION=3072

# Search + Images
TAVILY_API_KEY=...
FIRECRAWL_API_KEY=...
NEMOTRON_API_KEY=...
NEMOTRON_MODEL=nvidia/nemotron-3-super-120b-a12b:free
NEMOTRON_BASE_URL=https://openrouter.ai/api/v1

# Vector store
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=blog-enhancer

# RAG chunking
CHUNK_SIZE=1200
CHUNK_OVERLAP=250
TOP_K=8
```

---

*End of Architecture Document*
