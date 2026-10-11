# Job Scout

> Upload your CV. Get real job openings ranked 0–100 for fit, honest gap explanations, and a tailored application pack.

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
         │         Tool: TavilySearch (LangChain tool)
         │         Output: list of raw job postings
         │
         ├──▶ [3] Ranking Agent
         │         Tool: LLM scoring prompt
         │         Output: jobs with score 0–100 + rationale
         │
         ├──▶ [3b] Query Reformulator (conditional)
         │         Triggered if best score < threshold
         │         Tool: LLM — analyses poor results, generates better query
         │         Output: improved query → loops back to [2] and [3]
         │
         ├──▶ [4] Gap Analysis Agent
         │         Tool: LLM comparison prompt
         │         Output: per-job gap report (honest, not flattering)
         │
         └──▶ [5] Application Pack Agent
                   Tool: LLM generation prompt
                   Output: cover letter + key talking points
```

**Why multiple agents instead of one big prompt?**
Each agent is smaller, testable, and observable. You can improve the ranking agent without touching the cover letter generator. You can swap the job search tool without rewriting the pipeline.

---

## Orchestrator — How It Works

The orchestrator is built with **LangGraph** — a framework for stateful agent pipelines. Each step is a **node** (plain Python function), connected by **edges**. The state flows as a `TypedDict` through the graph; each node returns only the keys it updates.

```
                    ┌─────────────────────────────────────────┐
                    │  PipelineState (TypedDict)              │
                    │  profile, location, jobs, ranked,       │
                    │  custom_query, seen_urls, retried        │
                    └─────────────────────────────────────────┘

START
  │
  ▼
┌──────────────┐
│  search      │  calls Job Search Agent (TavilySearch tool)
│  → jobs      │  on retry: filters out already-seen URLs
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  rank        │  calls Ranking Agent — scores each job 0–100 via LLM
│  → ranked    │
└──────┬───────┘
       │
       ▼  route_after_rank()  ← conditional edge
       │
       ├── score < threshold AND not retried ──▶ ┌──────────────────┐
       │                                          │  reformulate     │
       │                                          │  LLM rewrites    │
       │                                          │  the query       │
       │                                          │  → custom_query  │
       │                                          │  → retried=True  │
       │                                          └────────┬─────────┘
       │                                                   │
       │                                    loops back ────┘ (to search)
       │
       ├── retried=True ──▶ ┌──────────────┐
       │                    │  keep_best   │  compares both result sets
       │                    │              │  keeps the higher-scoring one
       │                    └──────┬───────┘
       │                           │
       └── score ≥ threshold ──▶  END
```

**Memory lifecycle in one run:**
```
Start
 ├─ Load user_prefs (SQLite) → target_location
 ├─ Parse CV → CVProfile
 ├─ Run LangGraph pipeline → PipelineState (short-term memory)
 └─ Save session + ranked jobs (SQLite, long-term memory)
```

---

## Tech Stack & Why Each Tool

| Tool | Role | Why You're Learning It |
|---|---|---|
| **Python** | Language | Standard for AI engineering |
| **LangChain** | Chain & prompt abstractions | ChatPromptTemplate, output parsers, tool wrappers |
| **LangGraph** | Agent pipeline orchestration | Stateful graphs with nodes, edges, conditional routing |
| **Langfuse** | Observability + Evals | See every LLM call, input/output, latency — traces + eval scores UI |
| **Streamlit** | UI | Simple web UI for CV upload and results display |
| **SQLite** | Long-term memory | Persist sessions, preferences, job history |

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

LangChain has a standard way to define and register tools so agents can use them.

### 3. Traces
Every time an agent runs, you want to see:
- What prompt was sent
- What the LLM returned
- How long it took
- What tool was called and with what args

This is a **trace**. Langfuse captures traces automatically when you instrument your code.

### 4. Evals
After you have traces, you want to measure quality automatically and repeatedly. The question is: if you change a prompt, did things get better or worse?

There are two types of evals used in Job Scout:

- **Deterministic**: plain Python checks on output structure — score is always 0–100, cover letter is under 300 words, talking points are 3–5. No LLM needed.
- **Reference-based**: checks that agent behaviour is coherent — a data engineer CV should score higher on a data engineering job than on a marketing job. You don't need an exact expected output, just a logical constraint.

Each eval function returns `(bool, str)` — pass/fail and a comment. The `run_eval()` wrapper opens a Langfuse trace, runs the function, and attaches a numeric score (1.0 = pass, 0.0 = fail) to that trace. Results are visible in Langfuse UI → Scores.

```
python evals/run_evals.py
```

The improvement loop:
```
Change a prompt or model
        ↓
python evals/run_evals.py
        ↓
Compare scores in Langfuse → did the change help?
```

### 5. Memory & Context Management
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
Memory that survives across sessions. Stored in SQLite.

| What is stored | Technology |
|---|---|
| User preferences (location) | SQLite (`user_prefs`) |
| Past sessions and ranked jobs | SQLite (`session_store`) |
| Uploaded CV path | SQLite (`user_prefs`) |

#### How the three levels work together in Job Scout

```
Session starts
      │
      ├─ Load from Long-term Memory ──▶ user preferences, CV path
      │
      ├─ Run pipeline, passing state through Short-term Memory
      │        CV Parser → Job Search → Ranker → Gap → AppPack
      │
      ├─ Context Window ──▶ each LLM call gets only what it needs (not the whole state)
      │
      └─ Save to Long-term Memory ──▶ session results, ranked jobs
```

---

## Project Structure

```
job-scout/
│
├── README.md                  ← you are here
│
├── agents/
│   ├── cv_parser.py           ← Agent 1 — PDF → CVProfile (LangChain chain)
│   ├── job_search.py          ← Agent 2 — TavilySearch tool → JobPosting list
│   ├── ranker.py              ← Agent 3 — scores jobs 0–100 (LangChain chain)
│   ├── query_reformulator.py  ← Agent 3b — rewrites query if results are poor
│   ├── gap_analysis.py        ← Agent 4 — strengths, gaps, recommendation
│   └── application_pack.py   ← Agent 5 — cover letter + talking points
│
├── tools/
│   └── pdf_extractor.py       ← Tool used by Agent 1
│
├── memory/
│   ├── session_store.py       ← Long-term memory: saves sessions to SQLite
│   └── user_prefs.py          ← Long-term memory: user preferences (SQLite)
│
├── data/
│   └── job_scout.db           ← SQLite database (gitignored)
│
├── orchestrator.py            ← LangGraph pipeline: StateGraph, nodes, conditional edges
├── tracing.py                 ← Langfuse CallbackHandler for LangChain tracing
│
├── ui/
│   └── app.py                 ← Streamlit UI
│
├── evals/
│   ├── datasets/
│   │   ├── cv_data_engineer.json    ← synthetic CV: senior data engineer
│   │   ├── cv_junior_developer.json ← synthetic CV: junior frontend developer
│   │   └── job_postings.json        ← job postings labelled relevant/irrelevant
│   └── run_evals.py           ← eval runner: calls agents, scores results, logs to Langfuse
│
├── .env                       ← API keys (never commit this)
├── pyproject.toml             ← Project dependencies
└── LEARNINGS.md               ← Your personal notes as you build
```

---

## Build Order (Phases)

Build one phase at a time. Each phase produces something runnable.

### Phase 1 — Foundation ✅
- Set up the Python project (pyproject.toml, virtual env)
- Install LangChain, openai (via Groq endpoint)
- Write `cv_parser.py`: PDF → CVProfile with `ChatPromptTemplate | ChatOpenAI | PydanticOutputParser`

**Concepts learned:** LangChain chains, PydanticOutputParser, structured LLM output

### Phase 2 — Search & Rank ✅
- Write `job_search.py` using `TavilySearch` (LangChain tool)
- Write `ranker.py` scoring each job 0–100 with a LangChain chain
- Location resolved from CV (`target_location`) or asked to user

**Concepts learned:** LangChain tools, StrOutputParser, prompt engineering for scoring

### Phase 3 — Observability ✅
- Langfuse `CallbackHandler` injected via `config={"callbacks": [...]}` in every chain
- All LLM calls visible in Langfuse with prompts, responses, latency

**Concepts learned:** traces, LangChain callbacks, observability in AI pipelines

### Phase 3.5 — Memory Layer + Smart Retry ✅
- Short-term memory: `PipelineState` TypedDict flows through the LangGraph pipeline
- Long-term memory: sessions and ranked jobs saved to SQLite
- User preferences: `target_location` persisted across runs
- Orchestrator built with LangGraph: `StateGraph`, nodes, `add_conditional_edges`
- If best score < threshold → LLM reformulates query → retry with URL deduplication

**Concepts learned:** LangGraph, TypedDict state, nodes, conditional edges, SQLite memory

### Phase 4 — Gap Analysis + Application Pack ✅
- `gap_analysis.py`: honest per-job analysis — strengths, critical gaps, minor gaps, verdict
- `application_pack.py`: tailored cover letter + 3–5 interview talking points
- Both called on-demand from the orchestrator; gap analysis is reused for the cover letter if same job

**Concepts learned:** multi-step prompt chaining, output reuse across agents

### Phase 5 — UI ✅
- Streamlit app in `ui/app.py` — dark/light mode, sidebar CV management, job cards with score badges
- Gap analysis and cover letter generated on-demand per card (not all at once)
- Results preserved across reruns with `st.session_state`
- CV persisted to disk, preferences saved to SQLite

**Concept learned:** serving AI pipelines as web apps, Streamlit state management

### Phase 6 — Evals ✅
- Two synthetic CV datasets + job postings labelled relevant/irrelevant
- `evals/run_evals.py`: 9 evals across three agents
  - **Deterministic** (no LLM): score range, rationale non-empty, cover letter length, name in letter, talking points count
  - **Reference-based**: relevant jobs score higher than irrelevant jobs; gap analysis cites real CV skills
- Each eval opens a Langfuse trace and attaches a numeric score — visible in Langfuse UI → Scores

**Concept learned:** eval types, dataset design, Langfuse scoring API, the improvement loop

---

## Prerequisites

- Python 3.11+
- **LLM API key:** [Groq](https://console.groq.com) — free/on-demand tier, fast inference via OpenAI-compatible endpoint
- **Search API key:** [Tavily](https://app.tavily.com) — 1000 free searches/month, designed for AI agents
- Langfuse account or local Docker instance (Phase 3+)

> Add your keys to `.env`. The `GROQ_BASE_URL` points the openai Python client at Groq's endpoint — swap the model name anytime without touching agent code.
