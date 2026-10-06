# Job Scout

> Upload your CV. Get real job openings ranked 0–100 for fit, honest gap explanations, and a tailored application pack. Then ask for all of it out loud.

---

## What This Project Is (and Isn't)

Job Scout is a **multi-agent AI system** — not a single LLM call, but a pipeline of specialized agents that collaborate to produce a useful output. Each agent has one job, uses tools, and can be observed, evaluated, and improved independently.

This project is also a **learning vehicle**. The goal is to understand, hands-on, how modern AI engineering works: agents, observability, and evals.

---

## The User Journey

```
1. User uploads a PDF CV
         ↓
2. CV Parser Agent extracts structured profile (skills, experience, education)
         ↓
3. Job Search Agent fetches real job openings from the web
         ↓
4. Ranking Agent scores each job 0–100 for fit against the profile
         ↓
5. Gap Analysis Agent writes honest explanations: where you're strong, where you fall short
         ↓
6. Application Pack Agent generates a tailored cover letter + talking points
         ↓
7. Voice Agent reads the full report out loud (TTS)
```

---

## Architecture: Multi-Agent Pipeline

```
┌─────────────────────────────────────────────────────┐
│                    Orchestrator                      │
│         (coordinates the agent pipeline)            │
└────────┬────────────────────────────────────────────┘
         │
         ├──▶ [1] CV Parser Agent
         │         Tool: PDF extractor
         │         Output: structured JSON profile
         │
         ├──▶ [2] Job Search Agent
         │         Tool: web search / job board API
         │         Output: list of raw job postings
         │
         ├──▶ [3] Ranking Agent
         │         Tool: LLM scoring prompt
         │         Output: jobs with score 0–100 + rationale
         │
         ├──▶ [4] Gap Analysis Agent
         │         Tool: LLM comparison prompt
         │         Output: per-job gap report (honest, not flattering)
         │
         ├──▶ [5] Application Pack Agent
         │         Tool: LLM generation prompt
         │         Output: cover letter + key talking points
         │
         └──▶ [6] Voice Agent
                   Tool: TTS API (e.g. OpenAI TTS, ElevenLabs)
                   Output: audio file of the full report
```

**Why multiple agents instead of one big prompt?**
Each agent is smaller, testable, and observable. You can improve the ranking agent without touching the cover letter generator. You can swap the job search tool without rewriting the pipeline.

---

## Tech Stack & Why Each Tool

| Tool | Role | Why You're Learning It |
|---|---|---|
| **Python** | Language | Standard for AI engineering |
| **LangChain** | Agent framework | Gives you abstractions for chains, tools, memory, agents |
| **Langfuse** | Observability | See every LLM call, input/output, latency — traces + evals UI |
| **FastAPI** | API layer | Expose the pipeline as an HTTP endpoint |
| **Streamlit** | UI | Simple web UI for CV upload and results display |
| **SQLite** | Long-term memory (structured) | Persist sessions, preferences, job history |
| **ChromaDB** | Long-term memory (vector) | Semantic search on past jobs — RAG |

---

## Key Concepts You Will Learn

### 1. Agents vs. Chains
- A **chain** is a fixed sequence of LLM calls: A → B → C
- An **agent** can decide what to do next based on the LLM output (it can call tools, loop, branch)
- Job Scout uses both: a fixed pipeline (chain) of agents that are themselves agentic

### 2. Tools
A tool is a function an agent can call. Examples here:
- `extract_pdf_text(file)` → text
- `search_jobs(query)` → list of postings
- `text_to_speech(text)` → audio

LangChain has a standard way to define and register tools so agents can use them.

### 3. Traces
Every time an agent runs, you want to see:
- What prompt was sent
- What the LLM returned
- How long it took
- What tool was called and with what args

This is a **trace**. Langfuse captures traces automatically when you instrument your code.

### 4. Evals
After you have traces, you want to measure quality:
- Is the ranking accurate? (does a 90-score job actually fit the CV?)
- Is the gap analysis honest and specific?
- Is the cover letter generic or tailored?

**Evals** are automated (or human) tests on LLM outputs. You define what "good" looks like, then run evals on a dataset of examples. Langfuse has a UI for this.

### 5. Skills (Agent Skills / Capabilities)
In some frameworks, "skills" are named, reusable capabilities that an agent exposes — think of them as the public API of an agent. In Claude Code (which powers this dev environment), skills are also a first-class concept. The idea is the same: decompose behavior into named, composable units.

### 7. Memory & Context Management
One of the most important and misunderstood concepts in multi-agent systems. There are three distinct levels:

#### Level 1 — Context Window (working memory)
The LLM's in-call memory. Everything in the prompt is "in its head" for that one call. When the call ends, it's gone.

- **Limit:** models have a max token limit (e.g. 128k for GPT-4o). You can't dump 50 job postings into one prompt.
- **Solution:** process jobs in batches, or use retrieval to select only the most relevant ones.

#### Level 2 — Short-term Memory (session state)
The state the Orchestrator holds while the pipeline runs. The parsed CV is passed from agent to agent as a Python dict — this is short-term memory. It lives for the duration of one session.

```
# What the orchestrator holds in memory during a run:
{
  "profile":        {...},   ← CV Parser output
  "jobs":           [...],   ← Job Search output
  "ranked_jobs":    [...],   ← Ranker output
  "gaps":           [...],   ← Gap Analysis output
  "cover_letters":  [...]    ← Application Pack output
}
```

This is the first type you'll implement. It's just a Python object passed through the pipeline.

#### Level 3 — Long-term Memory (persistent storage)
Memory that survives across sessions. Stored in a database, not in the LLM.

| What to remember | Type | Technology |
|---|---|---|
| User preferences (sector, salary, location) | Structured | SQLite / PostgreSQL |
| Past sessions (CV versions, searches done) | Structured | SQLite / PostgreSQL |
| Jobs seen and rated by the user | Structured | SQLite / PostgreSQL |
| "Find jobs similar to ones I liked" | Semantic / Vector | ChromaDB, pgvector |

The vector case is what you hear called **RAG (Retrieval-Augmented Generation)**: instead of putting everything in the prompt, you retrieve only the relevant pieces and inject them into context. In Job Scout, RAG could power "rank this new job against similar jobs you liked in the past."

#### How the three levels work together in Job Scout

```
Session starts
      │
      ├─ Load from Long-term Memory ──▶ user preferences, past job scores
      │
      ├─ Run pipeline, passing state through Short-term Memory
      │        CV Parser → Job Search → Ranker → Gap → AppPack → Voice
      │
      ├─ Context Window ──▶ each LLM call gets only what it needs (not the whole state)
      │
      └─ Save to Long-term Memory ──▶ results, user feedback, new preferences
```

---

## Project Structure (to be built incrementally)

```
job-scout/
│
├── README.md                  ← you are here
│
├── agents/
│   ├── cv_parser.py           ← Agent 1
│   ├── job_search.py          ← Agent 2
│   ├── ranker.py              ← Agent 3
│   ├── gap_analysis.py        ← Agent 4
│   ├── application_pack.py    ← Agent 5
│   └── voice.py               ← Agent 6
│
├── tools/
│   ├── pdf_extractor.py       ← Tool used by Agent 1
│   ├── job_board_api.py       ← Tool used by Agent 2
│   └── tts.py                 ← Tool used by Agent 6
│
├── memory/
│   ├── session_store.py       ← Save/load full sessions (short → long term)
│   ├── user_prefs.py          ← Persist user preferences across sessions
│   └── vector_store.py        ← RAG: semantic search on past jobs
│
├── orchestrator.py            ← Coordinates the full pipeline + memory lifecycle
│
├── api/
│   └── main.py                ← FastAPI app (HTTP endpoints)
│
├── ui/
│   └── app.py                 ← Streamlit UI
│
├── evals/
│   ├── datasets/              ← Sample CVs + expected outputs
│   └── run_evals.py           ← Evaluation scripts
│
├── .env                       ← API keys (never commit this)
├── pyproject.toml             ← Project dependencies
└── LEARNINGS.md               ← Your personal notes as you build
```

---

## Build Order (Phases)

Build one phase at a time. Each phase produces something runnable.

### Phase 1 — Foundation
- Set up the Python project (pyproject.toml, virtual env)
- Install LangChain, openai (via Groq endpoint)
- Write `cv_parser.py`: take a PDF, return a dict with name, skills, experience
- Test it with your own CV

**Concept learned:** tools, basic LLM calls, structured output

### Phase 2 — Search & Rank
- Write `job_search.py` with a web search tool
- Write `ranker.py` that scores each posting against the CV
- Run the two agents in sequence manually

**Concept learned:** tool use, chaining agents, prompt engineering for scoring

### Phase 3 — Observability
- Add Langfuse tracing to every agent
- Run the pipeline, open Langfuse, inspect the traces

**Concept learned:** traces, what a span is, how to read a trace tree

### Phase 3.5 — Memory Layer
- Implement short-term memory: the session state dict passed through the pipeline
- Implement long-term memory: save each session to SQLite (profile, jobs, scores)
- Add user preferences: persist "I don't want startups under 10 people"
- (Advanced) Add vector store: semantic search on past jobs with ChromaDB

**Concept learned:** the three levels of memory, RAG basics, state management in pipelines

### Phase 4 — Gap Analysis + Application Pack
- Write `gap_analysis.py`
- Write `application_pack.py`
- Write `orchestrator.py` to run the full pipeline end to end

**Concept learned:** multi-step agent pipelines, prompt chaining

### Phase 5 — Voice
- Write `voice.py` using OpenAI TTS or ElevenLabs
- Add it as the last step of the pipeline

**Concept learned:** multimodal outputs, tool wrapping

### Phase 6 — UI & API
- Wrap the orchestrator in a FastAPI endpoint
- Build a Streamlit UI for upload + results

**Concept learned:** serving AI pipelines as web apps

### Phase 7 — Evals
- Define what "good" looks like for ranking and gap analysis
- Build a small dataset of test CVs + expected outputs
- Write eval scripts using Langfuse

**Concept learned:** LLM evals, dataset design, quality measurement

---

## Prerequisites

- Python 3.11+
- **LLM API key:** [Groq](https://console.groq.com) — free/on-demand tier, fast inference via OpenAI-compatible endpoint
- **Speech model** (for Voice Agent):
  - Groq supports `whisper-large-v3` for speech-to-text (STT)
  - OpenAI TTS or ElevenLabs for text-to-speech (TTS)
- **Job board API key:** for Phase 2 (search agent)
  - [SerpAPI](https://serpapi.com) — 100 free searches/month
  - [Adzuna](https://developer.adzuna.com) — free tier available
- Langfuse account or local Docker instance (Phase 3+)

> Add your keys to `.env`. The `GROQ_BASE_URL` points the openai Python client at Groq's endpoint — swap the model name anytime without touching agent code.

---

## LEARNINGS.md

As you build each phase, write down in `LEARNINGS.md`:
- What confused you and how you resolved it
- What the concept actually means in plain words
- What you'd do differently

This file is for you. It becomes your personal AI engineering reference.
