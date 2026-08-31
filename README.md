# 🚀 Blog Enhancer — Production-Grade Multi-Agent AI Pipeline

[![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![SQLite](https://img.shields.io/badge/sqlite-%2307405e.svg?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![Pinecone](https://img.shields.io/badge/Pinecone-000000?style=for-the-badge)](https://www.pinecone.io/)
[![Groq](https://img.shields.io/badge/Groq-orange?style=for-the-badge)](https://groq.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com/)
[![Gemini](https://img.shields.io/badge/Google%20Gemini-8E75C2?style=for-the-badge&logo=google-gemini&logoColor=white)](https://deepmind.google/technologies/gemini/)

**Blog Enhancer** is a robust, production-grade multi-agent AI pipeline designed to ingest, score, research, and optimize blog posts for maximum quality, SEO, and GEO (Generative Engine Optimization).

Given a source blog URL, the pipeline scrapes the content, runs a baseline evaluation across six quality dimensions, performs deep background research via a supervisor-led agent team, and generates a heavily optimized, fact-checked, long-form post. It then enters an iterative evaluation-optimization loop until the article satisfies the strict quality criteria.

---

## 📖 Table of Contents
1. [Key Features](#-key-features)
2. [Workflow Architecture](#-workflow-architecture)
3. [Project Structure](#-project-structure)
4. [Prerequisites & Requirements](#-prerequisites--requirements)
5. [Step-by-Step Run Guide (from Zip file)](#-step-by-step-run-guide-from-zip-file)
6. [Environment Configuration](#-environment-configuration)
7. [Running the Application](#-running-the-application)
8. [Authors & Contact](#-authors--contact)

---

*   💻 **Offline / Local GPU Mode (Ollama & LM Studio):** Run the entire blog generation pipeline 100% locally on your GPU/CPU with **Qwen 2.5**, **Llama 3.2**, **Mistral**, or **DeepSeek-R1** with zero API costs and full data privacy.
*   🔀 **Hybrid Execution Mode:** Execute fast multi-agent specialist analysis locally on your GPU at zero cost, then hand off to Cloud models (Gemini 2.5 Flash / GPT-4o) for high-end long-form prose writing.
*   ⚡ **Ultra-Fast Sub-60s Generation (Turbo Mode):** Parallelizes 6 specialist analysis agents across concurrent threads, reducing turnaround time from ~3 minutes to **under 45 seconds**.
*   🖼️ **Custom Images & Contextual Placement:** Upload multiple custom images with individual captions, context descriptions, and placement hints. The agents analyze these details to embed each image at its most relevant section.
*   📥 **1-Click Clean Export (DOCX & PDF):** Export the generated article directly to styled Microsoft Word (`.docx`) and Adobe PDF (`.pdf`) documents, completely omitting internal evaluation scores and debug logs.
*   🤖 **Multi-Agent Collaboration (LangGraph):** Routes tasks between baseline evaluators, planners, research supervisors, parallel specialist reviewers, and prose writers.
*   📊 **6-Dimension Evaluation Rubric:** Scores articles on *Language*, *Facts*, *Structure*, *SEO*, *GEO*, and *Freshness*, comparing baseline vs. enhanced scores.
*   ⚡ **Mixed-Provider Cloud Routing:** Runs token-heavy analysis tasks on ultra-fast Groq APIs (free/fast tier) while reserving user-selected models (Gemini/OpenAI) for premium prose generation.
*   🔍 **Supervisor Research Pattern:** Coordinates Tavily searches and constructs 5 dedicated research briefs for specialist agents, eliminating token waste.
*   🛡️ **Fault-Tolerant State & UI Recovery:** SQLite database tracks job progress every `0.8s`. If disconnected or refreshed, jobs resume seamlessly.

---

## 🔄 Workflow Architecture

```mermaid
flowchart TD
    %% Styling
    classDef prep fill:#1e293b,stroke:#3b82f6,stroke-width:2px,color:#fff;
    classDef supervisor fill:#312e81,stroke:#6366f1,stroke-width:2px,color:#fff;
    classDef parallel fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#fff;
    classDef writer fill:#701a75,stroke:#d946ef,stroke-width:2px,color:#fff;
    classDef eval fill:#7c2d12,stroke:#f97316,stroke-width:2px,color:#fff;
    classDef export fill:#0f172a,stroke:#e2e8f0,stroke-width:2px,color:#fff;

    Start([Input: Blog URL or Topic + Custom Images]) --> Scrape[⚡ Fast HTTP Scrape / Fallback Crawl4AI]:::prep
    
    subgraph Phase1 [Phase 1: Ingestion & Baseline]
        Scrape --> Ingest[Background Vector Ingestion - Pinecone]:::prep
        Scrape --> Baseline[Baseline Evaluator - 6 Dimensions]:::eval
        Baseline --> Learner[Learner: Intent & Gap Analysis]:::prep
        Learner --> Planner[Planner: Content & SEO Strategy]:::prep
    end

    Planner --> Supervisor[🔍 Supervisor: Focused Research Briefs]:::supervisor

    subgraph Phase2 [Phase 2: Parallel Specialist Fan-Out (< 4s)]
        Supervisor --> S1[Language Specialist]:::parallel
        Supervisor --> S2[Facts & E-E-A-T Specialist]:::parallel
        Supervisor --> S3[Structure Specialist]:::parallel
        Supervisor --> S4[SEO Keyword Analyst]:::parallel
        Supervisor --> S5[GEO / AI-Search Specialist]:::parallel
        Supervisor --> S6[🖼️ Image & Custom Contextual Matcher]:::parallel
    end

    S1 & S2 & S3 & S4 & S5 & S6 --> Aggregator[✍️ Aggregator: Long-Form Article Synthesis]:::writer

    subgraph Phase3 [Phase 3: Speed Profile & Routing]
        Aggregator --> Eval[Evaluator: Score 6 Quality Dimensions]:::eval
        Eval --> Router{Speed Mode?}:::eval
        Router -- "⚡ Turbo (<45s)" --> Output
        Router -- "⚖️ Balanced / 🔬 Deep" --> OptCheck{Scores >= 90 / Max Passes?}:::eval
        OptCheck -- No --> Opt[Optimizer: Targeted Polish Rewrite]:::writer
        Opt --> EvalPost[Evaluator Post-Optimization]:::eval
        EvalPost --> Output
        OptCheck -- Yes --> Output
    end

    subgraph Phase4 [Phase 4: Output & Export]
        Output[Final Enhanced Blog Article]:::export --> UI[🖥️ Streamlit Interactive UI]:::export
        Output --> DOCX[📥 Download Word .docx without scores]:::export
        Output --> PDF[📄 Download Adobe PDF .pdf without scores]:::export
    end
```

Detailed architectural references, prompt definitions, and token-saving strategies can be found in the [ARCHITECTURE.md](file:///c:/DS_and_AI/Projects_and_Tutorials/Projects/blog/ARCHITECTURE.md) file.

---

## 📂 Project Structure

```
blog/
├── agents/                 # Multi-agent node implementations
│   ├── geo_agent.py
│   ├── language_agent.py
│   ├── planner.py
│   ├── supervisor.py
│   └── ...
├── config/                 # Prompt definitions and JSON schemas
├── data/                   # Persistence directory (SQLite database)
│   └── blog_enhancer.db
├── db/                     # DB model definitions and connection logic
│   ├── database.py
│   └── models.py
├── generated_images/       # Temp directory for image assets
├── graph/                  # LangGraph state machine & DAG definition
│   ├── graph.py
│   └── state.py
├── prompts/                # System templates for various agent layers
├── rag/                    # Vector store and embedding logic
├── utilis/                 # Formatting, timing, and database writing utilities
├── app.py                  # CLI main entrypoint
├── streamlit_app.py        # Streamlit web dashboard entrypoint
├── requirements.txt        # Frozen package dependencies
├── check_deps.py           # Dependency verification script
└── ARCHITECTURE.md         # Comprehensive system documentation
```

---

## 📋 Prerequisites & Requirements

*   **Python:** Version `3.10` or higher is required.
*   **API Keys Needed:**
    *   **Groq API Key:** Essential for agent orchestration and baseline scoring.
    *   **Pinecone API Key & Index:** Used for embedding storage and document chunk retrieval.
    *   **Tavily API Key:** Essential for the search/research agent system.
    *   **Google Gemini API Key / OpenAI API Key:** Used for embeddings and prose generation.

---

## 🛠️ Step-by-Step Run Guide (From Zip File)

Follow these instructions to extract, install, and execute the application on your machine:

### Step 1: Extract the Zip File
Unzip/extract the containing folder to your preferred workspace directory (e.g. `C:\Users\ADMIN\Downloads\blog`).

### Step 2: Open Terminal / Command Prompt
Open your terminal window (PowerShell, Command Prompt, or bash) and navigate to the project root directory:
```powershell
cd "c:\Users\ADMIN\Downloads\blog"
```

### Step 3: Create & Activate a Virtual Environment
It is highly recommended to isolate project dependencies using a virtual environment:

*   **Windows (PowerShell):**
    ```powershell
    python -m venv env
    .\env\Scripts\activate
    ```
*   **macOS / Linux:**
    ```bash
    python3 -m venv env
    source env/bin/activate
    ```

### Step 4: Install Dependencies
Install all required libraries using the pinned `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### Step 5: Configure Environment Variables
1. Check if a `.env` file exists at the root of the project. If not, copy or create one named `.env`.
2. Populate the required API keys (see [Environment Configuration](#-environment-configuration) below).

### Step 6: Verify Dependencies
Before running the main app, verify that all dependencies and imports are correctly resolved:
```bash
python check_deps.py
```
If any packages are missing, it will output the exact `pip install` commands needed.

---

## ⚙️ Environment Configuration

Ensure your `.env` file at the root contains the following variables:

```ini
# LLM Providers API Keys
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key

# Research and Scraper API Keys
TAVILY_API_KEY=your_tavily_api_key
FIRECRAWL_API_KEY=your_firecrawl_api_key

# Vector Database configuration
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=blog-enhancer

# Embedding and Model Settings
EMBEDDING_PROVIDER=gemini
EMBEDDING_MODEL=models/gemini-embedding-001
EMBEDDING_DIMENSION=3072
GROQ_MODEL=llama-3.1-8b-instant
GEMINI_MODEL=gemini-2.5-flash
OPENAI_MODEL=gpt-4o

# RAG configurations
CHUNK_SIZE=1200
CHUNK_OVERLAP=250
TOP_K=8
```

---

## 🚀 Running the Application

You can execute the pipeline in two modes:

### Option A: Streamlit Web UI (Recommended)
The web application provides the most comprehensive experience, visualizing agent updates and storing output history.
```bash
streamlit run streamlit_app.py
```
Once started, open your web browser to `http://localhost:8501`.

### Option B: Command Line Interface (CLI)
For quick, single-run execution without a browser UI:
```bash
python app.py
```
This runs the blog optimization graph using the default URL configured in [app.py](file:///c:/Users/ADMIN/Downloads/blog/app.py).

---

## 👥 Authors & Contact

*   **Author:** Abhinav Gupta 👨‍💻
*   **Email:** [abhinavgupta15.ag@gmail.com](mailto:abhinavgupta15.ag@gmail.com)
*   **Project Context:** Blog Enhancer Multi-Agent SEO Pipeline. Feel free to reach out for questions, feedback, or integration requests!
