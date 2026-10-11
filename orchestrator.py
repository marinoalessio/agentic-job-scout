"""
Orchestrator — LangGraph

Uses LangGraph to build a stateful pipeline with conditional retry logic.

Key LangGraph concepts:
  - StateGraph: the graph that holds nodes and edges
  - TypedDict: defines the schema of the state flowing through the graph
  - Nodes: plain functions that receive the full state and return only the keys they update
  - Edges: fixed connections between nodes
  - Conditional edges: routing based on state values (replaces RunnableBranch)
"""

import os
import sys
from typing import TypedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataclasses import dataclass

from langgraph.graph import END, START, StateGraph

from agents.application_pack import ApplicationPack, generate_application_pack
from agents.cv_parser import CVProfile, parse_cv
from agents.gap_analysis import GapAnalysis, analyse_gap
from agents.job_search import JobPosting, search_jobs
from agents.query_reformulator import reformulate_query
from agents.ranker import RankedJob, rank_jobs
from memory import session_store, user_prefs
from tracing import langfuse_handler

SCORE_THRESHOLD = 90


# --- Pipeline state (TypedDict) ---
# TypedDict defines the schema of the state that flows through the graph.
# Every node receives the full state and returns a dict with only the keys it changes.
# LangGraph automatically merges the returned dict into the existing state.
# This means nodes are independent — each one only knows about what it needs.

class PipelineState(TypedDict):
    profile: CVProfile
    location: str
    jobs: list[JobPosting]
    ranked: list[RankedJob]
    custom_query: str
    seen_urls: set
    original_ranked: list[RankedJob]
    retried: bool           # flag to prevent infinite retry loop
    max_results: int        # how many jobs to fetch (set by caller)


# --- Short-term memory ---

@dataclass
class SessionState:
    profile: CVProfile
    location: str
    jobs: list[JobPosting]
    ranked_jobs: list[RankedJob]
    gap_analysis: GapAnalysis | None = None
    application_pack: ApplicationPack | None = None


# --- Location resolution (outside the graph — runs before pipeline) ---

def resolve_location(profile: CVProfile) -> str:
    saved = user_prefs.get("target_location")
    if saved:
        print(f"[orchestrator] Using saved location preference: {saved}")
        return saved

    if profile.target_location:
        location = profile.target_location
        print(f"[orchestrator] Using target location from CV: {location}")
    elif profile.location:
        location = profile.location
        print(f"[orchestrator] Using current location from CV: {location}")
    else:
        location = input("No location found. Which city/country are you looking for jobs in? ").strip()

    user_prefs.set("target_location", location)
    print(f"[orchestrator] Location '{location}' saved for future runs.")
    return location


# --- Graph nodes ---
# Each node is a plain Python function.
# Input: full PipelineState
# Output: dict with only the keys to update — LangGraph merges it into the state

def search_node(state: PipelineState) -> dict:
    jobs = search_jobs(
        state["profile"],
        location=state["location"],
        custom_query=state.get("custom_query", ""),
        max_results=state.get("max_results", 5),
    )
    # On retry, filter out URLs already seen in the first attempt
    seen = state.get("seen_urls", set())
    if seen:
        jobs = [j for j in jobs if j.url not in seen]
        print(f"[orchestrator] After deduplication: {len(jobs)} new jobs.")
    return {"jobs": jobs}


def rank_node(state: PipelineState) -> dict:
    ranked = rank_jobs(state["profile"], state["jobs"])
    return {"ranked": ranked}


def reformulate_node(state: PipelineState) -> dict:
    print(f"\n[orchestrator] Best score {state['ranked'][0].score}/100 < {SCORE_THRESHOLD}. Reformulating...\n")
    original_query = f"{state['profile'].experience[0].role} job openings in {state['location']}"
    new_query = reformulate_query(original_query, state["profile"], state["ranked"])
    print(f"[orchestrator] New query: '{new_query}'")
    return {
        "custom_query": new_query,
        "seen_urls": {job.url for job in state["ranked"]},
        "original_ranked": state["ranked"],
        "retried": True,
    }


def keep_best_node(state: PipelineState) -> dict:
    """Compares original and retry results, keeps the better set."""
    original = state.get("original_ranked", [])
    if original and original[0].score >= state["ranked"][0].score:
        print("[orchestrator] Reformulation did not improve results. Keeping original.")
        return {"ranked": original}
    print(f"[orchestrator] Improved results (best: {state['ranked'][0].score}/100). Using new results.")
    return {}


# --- Conditional edge ---
# Returns a string that maps to the next node (or END).
# Called after every rank_node execution.

def route_after_rank(state: PipelineState) -> str:
    if state.get("retried", False):
        # Second pass: compare results and finish
        return "keep_best"
    if state.get("ranked") and state["ranked"][0].score < SCORE_THRESHOLD:
        # First pass, score is low: reformulate and retry
        return "retry"
    # First pass, score is good: done
    return "end"


# --- Graph definition ---

builder = StateGraph(PipelineState)

# Nodes
builder.add_node("search", search_node)
builder.add_node("rank", rank_node)
builder.add_node("reformulate", reformulate_node)
builder.add_node("keep_best", keep_best_node)

# Fixed edges
builder.add_edge(START, "search")
builder.add_edge("search", "rank")
builder.add_edge("reformulate", "search")   # retry: go back to search with new query
builder.add_edge("keep_best", END)

# Conditional edge: after ranking, decide what to do
builder.add_conditional_edges(
    "rank",
    route_after_rank,
    {
        "retry": "reformulate",
        "keep_best": "keep_best",
        "end": END,
    },
)

pipeline = builder.compile()


# --- Job selection ---

def select_job(ranked_jobs: list[RankedJob], prompt: str) -> RankedJob:
    """
    Shows the ranked list and asks the user to pick one by number.
    Returns the selected RankedJob.
    """
    print(f"\n{prompt}")
    for i, job in enumerate(ranked_jobs, 1):
        print(f"  {i}. [{job.score}/100] {job.title} ({job.source})")

    while True:
        try:
            choice = int(input(f"\nEnter number (1-{len(ranked_jobs)}): ").strip())
            if 1 <= choice <= len(ranked_jobs):
                return ranked_jobs[choice - 1]
            print(f"Please enter a number between 1 and {len(ranked_jobs)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")


# --- Main function ---

def run(pdf_path: str) -> SessionState:
    print("\n=== JOB SCOUT — Starting pipeline ===\n")

    profile = parse_cv(pdf_path)
    location = resolve_location(profile)

    final_state = pipeline.invoke(
        {
            "profile": profile,
            "location": location,
            "retried": False,
            "custom_query": "",
            "seen_urls": set(),
            "max_results": 5,
        },
        config={"callbacks": [langfuse_handler]},
    )

    ranked = final_state["ranked"]

    session_store.save_session(
        candidate_name=profile.name,
        location=location,
        ranked_jobs=ranked,
    )

    # --- Phase 4: Gap Analysis ---
    gap_job = select_job(ranked, "Which job do you want a gap analysis for?")
    gap = analyse_gap(profile, gap_job)

    # --- Phase 4: Application Pack ---
    pack_job = select_job(ranked, "Which job do you want a cover letter for?")
    if pack_job.url == gap_job.url:
        # Reuse the gap analysis already computed
        pack_gap = gap
    else:
        pack_gap = analyse_gap(profile, pack_job)
    pack = generate_application_pack(profile, pack_job, pack_gap)

    return SessionState(
        profile=profile,
        location=location,
        jobs=final_state["jobs"],
        ranked_jobs=ranked,
        gap_analysis=gap,
        application_pack=pack,
    )


# --- Entry point ---

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python orchestrator.py <path_cv.pdf>")
        sys.exit(1)

    state = run(sys.argv[1])

    print(f"\n=== RESULTS for {state.profile.name} — {state.location} ===\n")
    for job in state.ranked_jobs:
        print(f"[{job.score}/100] {job.title}")
        print(f"  {job.rationale}")
        print(f"  {job.url}\n")

    if state.gap_analysis:
        g = state.gap_analysis
        print(f"\n=== GAP ANALYSIS — {g.job_title} ===\n")
        print("Strengths:")
        for s in g.strengths:
            print(f"  + {s}")
        print("\nCritical gaps:")
        for c in g.critical_gaps:
            print(f"  ✗ {c}")
        print("\nMinor gaps:")
        for m in g.minor_gaps:
            print(f"  ~ {m}")
        print(f"\nRecommendation: {g.recommendation}")

    if state.application_pack:
        p = state.application_pack
        print(f"\n=== APPLICATION PACK — {p.job_title} ===\n")
        print("--- COVER LETTER ---")
        print(p.cover_letter)
        print("\n--- TALKING POINTS ---")
        for i, tp in enumerate(p.talking_points, 1):
            print(f"  {i}. {tp}")
